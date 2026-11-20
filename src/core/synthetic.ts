import { qRotate } from '../math/quat';
import { Random } from '../math/rng';
import type { Vec3 } from '../math/vec3';
import { sub } from '../math/vec3';
import { forwardKinematics, restPose } from './fk';
import { defaultSkeleton } from './skeleton';
import { generateMotion } from './motion';
import type {
  CorruptionSpec,
  MarkerBinding,
  MarkerClip,
  MotionKind,
  SkeletonDefinition,
} from './types';

export interface SyntheticData {
  clip: MarkerClip;
  /** Unexposed ground truth, used by tests and the error overlay. */
  truth: MarkerClip;
  truthMotion: ReturnType<typeof generateMotion>;
}

/** Local-space marker offsets relative to their anchor joint (rest frame). */
export function markerLocalOffsets(skel: SkeletonDefinition): Map<string, Vec3> {
  const rest = restPose(skel);
  const out = new Map<string, Vec3>();
  for (const b of skel.bindings) {
    const anchorIdx = skel.jointIndex.get(
      skel.clusters.find((c) => c.name === b.cluster)!.anchorJoint,
    )!;
    out.set(b.name, sub(b.restPosition, rest.worldPos[anchorIdx]));
  }
  return out;
}

/** Generate clean ground-truth marker trajectories from a skeleton motion. */
export function projectMarkers(
  skel: SkeletonDefinition,
  motion: ReturnType<typeof generateMotion>,
): Float32Array {
  const localOffsets = markerLocalOffsets(skel);
  const markerCount = skel.bindings.length;
  const out = new Float32Array(motion.frameCount * markerCount * 3);
  const anchorOf = new Map<string, number>();
  for (const c of skel.clusters) {
    const ai = skel.jointIndex.get(c.anchorJoint)!;
    for (const m of c.markers) anchorOf.set(m, ai);
  }

  for (let f = 0; f < motion.frameCount; f++) {
    const fk = forwardKinematics(skel, motion.rootPosition[f], motion.localRotations[f]);
    for (let m = 0; m < markerCount; m++) {
      const binding: MarkerBinding = skel.bindings[m];
      const ai = anchorOf.get(binding.name)!;
      const local = localOffsets.get(binding.name)!;
      const wp = qRotate(fk.worldRot[ai], local);
      const ap = fk.worldPos[ai];
      const o = (f * markerCount + m) * 3;
      out[o] = ap[0] + wp[0];
      out[o + 1] = ap[1] + wp[1];
      out[o + 2] = ap[2] + wp[2];
    }
  }
  return out;
}

/**
 * Build the full synthetic test clip:
 * walk/run/jump skeleton → inverse marker solve → noise + drop-outs + swaps.
 */
export function buildSyntheticClip(
  kind: MotionKind,
  frameCount: number,
  fps: number,
  corruption: CorruptionSpec,
): SyntheticData {
  const skel = defaultSkeleton();
  const motion = generateMotion(kind, frameCount, fps);
  const truthPositions = projectMarkers(skel, motion);
  const markerCount = skel.bindings.length;
  const markerNames = skel.bindings.map((b) => b.name);

  const truth: MarkerClip = {
    fps,
    markerNames,
    frameCount,
    positions: truthPositions.slice(),
    valid: new Uint8Array(frameCount * markerCount).fill(1),
    clusters: skel.clusters.map((c) => ({ ...c, markers: [...c.markers] })),
    bindings: skel.bindings.map((b) => ({ ...b, restPosition: [...b.restPosition] as Vec3 })),
  };

  const rng = new Random(corruption.seed);
  const positions = truthPositions.slice();
  const valid = new Uint8Array(frameCount * markerCount).fill(1);

  // 1) Gaussian measurement noise on every observed sample.
  for (let i = 0; i < positions.length; i++) {
    positions[i] += rng.normal(0, corruption.noiseStd);
  }

  // 2) Drop-out gaps: per marker, start gaps at gapRate, length 1..maxGapLength.
  for (let m = 0; m < markerCount; m++) {
    let f = 2;
    while (f < frameCount - 2) {
      if (rng.uniform() < corruption.gapRate) {
        const len = 1 + rng.int(Math.max(1, corruption.maxGapLength));
        const end = Math.min(frameCount - 1, f + len - 1);
        for (let g = f; g <= end; g++) valid[g * markerCount + m] = 0;
        f = end + 2;
      } else {
        f++;
      }
    }
  }

  // 3) Label swaps. Real optical swaps happen between markers that are close
  // in the camera image: ~70% within the same rigid cluster, ~30% between
  // neighbouring clusters of the same body region. Pure global swaps of
  // unrelated limb markers do not occur in practice and are not modelled.
  const clusterMarkerLists = skel.clusters.map((c) =>
    c.markers.map((name) => markerNames.indexOf(name)),
  );
  // Neighbour table: clusters sharing an anchor body region.
  const regionOf = (clusterName: string): string => {
    if (clusterName.includes('Arm') || clusterName.includes('Forearm')) return 'arm';
    if (clusterName.includes('Thigh') || clusterName.includes('Shank') || clusterName.includes('Foot')) return 'leg';
    if (clusterName === 'head' || clusterName === 'torso' || clusterName === 'pelvis') return 'core';
    return 'other';
  };

  const pickSwapPair = (): [number, number] | null => {
    if (rng.uniform() < 0.7) {
      // Intra-cluster pair.
      const list = clusterMarkerLists[rng.int(clusterMarkerLists.length)];
      if (list.length < 2) return null;
      const a = list[rng.int(list.length)];
      let b = list[rng.int(list.length - 1)];
      if (b >= a) b++;
      return [a, b];
    }
    // Inter-cluster pair within the same body region.
    const region = rng.pick(['arm', 'leg', 'core']);
    const candidates = skel.clusters
      .map((c, ci) => ({ c, ci }))
      .filter(({ c }) => regionOf(c.name) === region);
    if (candidates.length < 2) return null;
    const c1 = candidates[rng.int(candidates.length)].ci;
    let c2 = candidates[rng.int(candidates.length - 1)].ci;
    if (c2 >= c1) c2++;
    const a = rng.pick(clusterMarkerLists[c1]);
    const b = rng.pick(clusterMarkerLists[c2]);
    return a === b ? null : [a, b];
  };

  for (let s = 0; s < corruption.swapCount; s++) {
    const pair = pickSwapPair();
    if (!pair) continue;
    const [a, b] = pair;
    const len = 4 + rng.int(Math.max(1, corruption.maxSwapLength));
    const start = rng.int(Math.max(1, frameCount - len - 2));
    const end = Math.min(frameCount - 2, start + len);
    for (let f = start; f <= end; f++) {
      const ia = f * markerCount + a;
      const ib = f * markerCount + b;
      if (!valid[ia] || !valid[ib]) continue;
      const oa = ia * 3;
      const ob = ib * 3;
      for (let k = 0; k < 3; k++) {
        const tmp = positions[oa + k];
        positions[oa + k] = positions[ob + k];
        positions[ob + k] = tmp;
      }
    }
  }

  const clip: MarkerClip = {
    fps,
    markerNames,
    frameCount,
    positions,
    valid,
    clusters: truth.clusters,
    bindings: truth.bindings,
  };

  return { clip, truth, truthMotion: motion };
}
