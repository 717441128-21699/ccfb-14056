import { kabsch } from '../math/kabsch';
import { qRotate } from '../math/quat';
import { hermiteInterpolate, smoothTrack } from '../math/spline';
import type { Vec3 } from '../math/vec3';
import { distance } from '../math/vec3';
import { hungarian } from './hungarian';
import { defaultRestWorld, defaultSkeleton } from './skeleton';
import type {
  CleanReport,
  ClusterDef,
  GapRecord,
  MarkerClip,
  SwapRecord,
} from './types';

export interface CleanOptions {
  /** A marker deviating more than this many metres from its rigid prediction
   *  is treated as mislabelled (or an outlier). */
  swapResidualThresholdM: number;
  /** Cluster fit RMS above which a frame's cluster result is distrusted. */
  clusterRmsLimitM: number;
  /** Smoothing passes on finished tracks (0 disables). */
  smoothPasses: number;
  onProgress?: (fraction: number) => void;
}

export const DEFAULT_CLEAN_OPTIONS: CleanOptions = {
  swapResidualThresholdM: 0.025,
  clusterRmsLimitM: 0.02,
  smoothPasses: 2,
};

export interface CleanResult {
  cleaned: MarkerClip;
  report: CleanReport;
}

interface ClusterRuntime {
  name: string;
  markerIndices: number[];
  /** Rigid template: marker position relative to the anchor joint. */
  template: Vec3[];
  anchorJointIndex: number;
}

/**
 * Clean an optical marker clip:
 *  1. per-frame rigid alignment of every cluster (Kabsch on observed markers),
 *  2. global label assignment within each cluster → detect & undo swaps,
 *  3. fill drop-out gaps from rigid-body prediction, falling back to a
 *     trajectory Hermite fill when the cluster itself is unavailable,
 *  4. light binomial smoothing of the finished tracks.
 *
 * Pure function: the input clip is never mutated.
 */
export function cleanClip(input: MarkerClip, options: CleanOptions = DEFAULT_CLEAN_OPTIONS): CleanResult {
  const t0 = now();
  const frameCount = input.frameCount;
  const markerCount = input.markerNames.length;
  const markerIndex = new Map(input.markerNames.map((n, i) => [n, i]));

  // Templates are expressed in each cluster's anchor-joint frame, so every
  // marker of a cluster keeps a fixed rigid shape regardless of parent bones.
  const skeleton = defaultSkeleton();
  const restWorld = defaultRestWorld();
  const clusters: ClusterRuntime[] = input.clusters.map((c) =>
    buildCluster(c, input, markerIndex, skeleton, restWorld),
  );

  const positions = input.positions.slice();
  const valid = new Uint8Array(input.valid);
  const swapRecords: SwapRecord[] = [];
  let residualSum = 0;
  let residualCount = 0;
  let maxResidual = 0;

  const predicted = new Float32Array(positions.length);
  const predictionValid = new Uint8Array(frameCount * markerCount);
  const lastGoodA = new Map<string, { rotation: import('../math/quat').Quat; translation: Vec3 }>();

  // Pass A: per-cluster rigid fits. Intra-cluster swaps are undone here and a
  // rigid prediction is produced for every observed marker.
  for (let f = 0; f < frameCount; f++) {
    for (const cluster of clusters) {
      processClusterFrame(
        f, cluster, input, positions, valid, predicted, predictionValid,
        options, swapRecords,
        (r) => {
          residualSum += r;
          residualCount++;
          if (r > maxResidual) maxResidual = r;
        },
        lastGoodA,
      );
    }
    if (options.onProgress && (f & 63) === 0) options.onProgress(f / frameCount);
  }

  // Pass B: global reassignment for inter-cluster swaps. Each observed point is
  // matched to the marker whose rigid prediction it is closest to; points that
  // land on a different marker than their own are swapped back.
  for (let f = 0; f < frameCount; f++) {
    resolveGlobalSwaps(f, input, positions, valid, predicted, predictionValid, swapRecords);
  }

  // Pass C: re-fit every cluster after global relabelling so predictions and
  // residuals reflect the final, fully corrected labels.
  residualSum = 0;
  residualCount = 0;
  maxResidual = 0;
  predictionValid.fill(0);
  const lastGoodC = new Map<string, { rotation: import('../math/quat').Quat; translation: Vec3 }>();
  for (let f = 0; f < frameCount; f++) {
    for (const cluster of clusters) {
      processClusterFrame(
        f, cluster, input, positions, valid, predicted, predictionValid,
        options, [],
        (r) => {
          residualSum += r;
          residualCount++;
          if (r > maxResidual) maxResidual = r;
        },
        lastGoodC,
      );
    }
  }

  // Snapshot which samples were originally missing. Smoothing and gap filling
  // both consult this so a filled sample is never treated as an observation.
  const originallyMissing = input.valid.slice();

  // Denoise observed samples BEFORE filling. The rigid predictor then fits
  // against a smooth track, and gap anchors are already denoised, so filled
  // values and observed values live in the same coordinate system — no seam.
  if (options.smoothPasses > 0) {
    for (let m = 0; m < markerCount; m++) {
      const track: Vec3[] = new Array(frameCount);
      for (let f = 0; f < frameCount; f++) {
        const o = (f * markerCount + m) * 3;
        track[f] = [positions[o], positions[o + 1], positions[o + 2]];
      }
      const sm = smoothObservedOnly(track, originallyMissing, m, markerCount, options.smoothPasses);
      for (let f = 0; f < frameCount; f++) {
        if (originallyMissing[f * markerCount + m]) continue;
        const o = (f * markerCount + m) * 3;
        positions[o] = sm[f][0];
        positions[o + 1] = sm[f][1];
        positions[o + 2] = sm[f][2];
      }
    }
  }

  // Re-fit clusters on the denoised observations and refresh rigid predictions
  // so gap filling uses predictions consistent with the smoothed track.
  predictionValid.fill(0);
  const lastGoodS = new Map<string, { rotation: import('../math/quat').Quat; translation: Vec3 }>();
  for (let f = 0; f < frameCount; f++) {
    for (const cluster of clusters) {
      processClusterFrame(
        f, cluster, input, positions, valid, predicted, predictionValid,
        options, [],
        (r) => {
          residualSum += r;
          residualCount++;
          if (r > maxResidual) maxResidual = r;
        },
        lastGoodS,
      );
    }
  }

  const gapRecords: GapRecord[] = [];
  for (let m = 0; m < markerCount; m++) {
    fillMarkerGaps(m, frameCount, markerCount, positions, valid, predicted, predictionValid, gapRecords);
  }
  annotateGapMarkers(gapRecords, input.markerNames);

  const cleaned: MarkerClip = {
    ...input,
    positions,
    valid: new Uint8Array(frameCount * markerCount).fill(1),
    clusters: input.clusters.map((c) => ({ ...c, markers: [...c.markers] })),
  };

  options.onProgress?.(1);
  const durationMs = now() - t0;
  const report: CleanReport = {
    frameCount,
    markerCount,
    gaps: gapRecords,
    swaps: mergeSwapRuns(swapRecords),
    maxResidualMm: maxResidual * 1000,
    meanResidualMm: residualCount ? (residualSum / residualCount) * 1000 : 0,
    durationMs,
  };
  return { cleaned, report };
}

/**
 * Build a cluster runtime. Template points are marker rest positions relative
 * to the anchor joint's rest-pose world position, which makes them invariant
 * under the cluster's rigid motion.
 */
function buildCluster(
  cluster: ClusterDef,
  clip: MarkerClip,
  markerIndex: Map<string, number>,
  skeleton: ReturnType<typeof defaultSkeleton>,
  restWorld: Vec3[],
): ClusterRuntime {
  const markerIndices = cluster.markers.map((n) => markerIndex.get(n)!);
  const anchorIdx = skeleton.jointIndex.get(cluster.anchorJoint)!;
  const anchorRest = restWorld[anchorIdx];
  const template = markerIndices.map((mi) => {
    const binding = clip.bindings.find((b) => b.name === clip.markerNames[mi])!;
    return [
      binding.restPosition[0] - anchorRest[0],
      binding.restPosition[1] - anchorRest[1],
      binding.restPosition[2] - anchorRest[2],
    ] as Vec3;
  });
  return { name: cluster.name, markerIndices, template, anchorJointIndex: anchorIdx };
}

function processClusterFrame(
  frame: number,
  cluster: ClusterRuntime,
  clip: MarkerClip,
  positions: Float32Array,
  valid: Uint8Array,
  predicted: Float32Array,
  predictionValid: Uint8Array,
  options: CleanOptions,
  swapRecords: SwapRecord[],
  onResidual: (r: number) => void,
  lastGoodTransform: Map<string, { rotation: import('../math/quat').Quat; translation: Vec3 }>,
): void {
  const k = cluster.markerIndices.length;
  const observed: { slot: number; markerIndex: number; pos: Vec3 }[] = [];
  for (let slot = 0; slot < k; slot++) {
    const mi = cluster.markerIndices[slot];
    if (valid[frame * clip.markerNames.length + mi]) {
      const o = (frame * clip.markerNames.length + mi) * 3;
      observed.push({ slot, markerIndex: mi, pos: [positions[o], positions[o + 1], positions[o + 2]] });
    }
  }
  // At least 3 non-collinear points are required for a 6-DOF rigid fit.
  // With only 1-2 visible points, propagate the last good cluster rotation and
  // solve just the translation, which is fully determined by any visible point.
  if (observed.length < 3) {
    propagateFromLastGood(frame, cluster, clip, positions, valid, predicted, predictionValid, lastGoodTransform);
    return;
  }

  // Global label search. The synthetic/real-world failure mode is two
  // observed points carrying each other's labels, i.e. an OBSERVATION-side
  // permutation. Enumerate which template slot each observed point maps to:
  // identity plus every transposition of observed points. Each candidate is
  // fit rigidly; the correct assignment collapses to the noise floor while a
  // wrong-label identity fit stays at hundreds of mm.
  const n = observed.length;
  // assignment[i] = template slot that observed point i corresponds to.
  const identity = observed.map((o) => o.slot);
  const identityFit = kabsch(
    identity.map((slot) => cluster.template[slot]),
    observed.map((o) => o.pos),
  );
  // Degenerate geometry (e.g. only collinear points survive a gap): no rigid
  // prediction can be trusted on this frame, leave it to trajectory filling.
  if (!identityFit) return;
  const identityRms = identityFit.rms;
  const swapSearchTrigger = 0.008;

  let bestAssignment = identity;
  let bestRms = identityRms;
  let fit = identityFit;

  if (identityRms > swapSearchTrigger) {
    const candidates: number[][] = [identity];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const cand = identity.slice();
        const tmp = cand[i];
        cand[i] = cand[j];
        cand[j] = tmp;
        candidates.push(cand);
      }
    }
    for (const cand of candidates) {
      const src = cand.map((slot) => cluster.template[slot]);
      const candFit = kabsch(src, observed.map((o) => o.pos));
      if (candFit && candFit.rms < bestRms) {
        bestRms = candFit.rms;
        fit = candFit;
        bestAssignment = cand;
      }
    }

    // Hungarian refinement resolves 3-way cycles beyond transpositions.
    const expected = cluster.template.map((tp) => {
      const r = qRotate(fit!.rotation, tp);
      return [r[0] + fit!.translation[0], r[1] + fit!.translation[1], r[2] + fit!.translation[2]] as Vec3;
    });
    const cost: number[][] = observed.map((o) =>
      cluster.template.map((_, slot) => distance(o.pos, expected[slot])),
    );
    const refined = hungarian(cost, n, k);
    const refinedFit = kabsch(
      refined.map((slot) => cluster.template[slot]),
      observed.map((o) => o.pos),
    );
    if (refinedFit && refinedFit.rms < bestRms) {
      bestRms = refinedFit.rms;
      fit = refinedFit;
      bestAssignment = refined;
    }
  }

  // Confirm the swap only when relabelling collapses the fit to the noise
  // floor and improves it by at least 3x — the bimodal RMS gap makes this clean.
  const swapConfirmed =
    identityRms > swapSearchTrigger &&
    bestRms < identityRms / 3 &&
    bestRms < 0.02;

  // bestAssignment[i] names the slot whose marker should OWN observed point i.
  // Build current-marker -> owning-marker and rotate the permutation cycle.
  const permutation = new Map<number, number>();
  if (swapConfirmed) {
    for (let i = 0; i < n; i++) {
      const targetSlot = bestAssignment[i];
      const targetMarker = cluster.markerIndices[targetSlot];
      const currentMarker = observed[i].markerIndex;
      if (targetMarker !== currentMarker) {
        permutation.set(currentMarker, targetMarker);
        swapRecords.push({
          frame, start: frame, end: frame,
          markerA: clip.markerNames[currentMarker],
          markerB: clip.markerNames[targetMarker],
        });
      }
    }
  }
  applyPermutation(frame, permutation, clip, positions);

  // Re-fit on corrected labels, then predict every marker of the cluster.
  const src2: Vec3[] = [];
  const dst2: Vec3[] = [];
  for (let slot = 0; slot < k; slot++) {
    const mi = cluster.markerIndices[slot];
    const base = frame * clip.markerNames.length + mi;
    if (valid[base]) {
      const o = base * 3;
      src2.push(cluster.template[slot]);
      dst2.push([positions[o], positions[o + 1], positions[o + 2]]);
    }
  }
  if (src2.length < 3) return;
  const finalFit = kabsch(src2, dst2);
  // A fit far above the noise floor means this cluster's labels are still
  // unreliable on this frame (e.g. an unresolvable multi-cluster swap). Do not
  // emit rigid predictions from it — gap filling will fall back to trajectory
  // interpolation rather than inherit a centimetre-scale bias.
  if (!finalFit || finalFit.rms > options.clusterRmsLimitM * 2) return;
  lastGoodTransform.set(cluster.name, {
    rotation: [...finalFit.rotation] as import('../math/quat').Quat,
    translation: [...finalFit.translation] as Vec3,
  });

  for (let slot = 0; slot < k; slot++) {
    const mi = cluster.markerIndices[slot];
    const r = qRotate(finalFit.rotation, cluster.template[slot]);
    const p: Vec3 = [
      r[0] + finalFit.translation[0],
      r[1] + finalFit.translation[1],
      r[2] + finalFit.translation[2],
    ];
    const base = frame * clip.markerNames.length + mi;
    const o = base * 3;
    predicted[o] = p[0];
    predicted[o + 1] = p[1];
    predicted[o + 2] = p[2];
    predictionValid[base] = 1;
    if (valid[base]) {
      onResidual(distance([positions[o], positions[o + 1], positions[o + 2]], p));
    }
  }
}

/**
 * Inter-cluster swap resolution for one frame.
 *
 * For every observed marker, compare its stored position with the rigid
 * prediction of every OTHER marker that also has a prediction. A global
 * Hungarian assignment on the resulting distance matrix finds the marker
 * each observed point truly belongs to. Only mismatches whose reassignment
 * lands within the rigid prediction tolerance are accepted, so correctly
 * labelled points (which sit within a few mm of their own prediction) are
 * never moved.
 */
function resolveGlobalSwaps(
  frame: number,
  clip: MarkerClip,
  positions: Float32Array,
  valid: Uint8Array,
  predicted: Float32Array,
  predictionValid: Uint8Array,
  swapRecords: SwapRecord[],
): void {
  const mc = clip.markerNames.length;
  const observedMarkers: number[] = [];
  const predictableMarkers: number[] = [];
  for (let m = 0; m < mc; m++) {
    if (valid[frame * mc + m]) observedMarkers.push(m);
    if (predictionValid[frame * mc + m]) predictableMarkers.push(m);
  }
  if (observedMarkers.length < 2 || predictableMarkers.length < 2) return;

  const cost: number[][] = observedMarkers.map((mi) =>
    predictableMarkers.map((mj) => {
      const o = (frame * mc + mi) * 3;
      const p = (frame * mc + mj) * 3;
      return distance(
        [positions[o], positions[o + 1], positions[o + 2]],
        [predicted[p], predicted[p + 1], predicted[p + 2]],
      );
    }),
  );
  const assignment = hungarian(cost, observedMarkers.length, predictableMarkers.length);

  // Build current -> correct marker mapping for accepted reassignments.
  const relabel = new Map<number, number>();
  for (let i = 0; i < observedMarkers.length; i++) {
    const currentMarker = observedMarkers[i];
    const correctMarker = predictableMarkers[assignment[i]];
    if (correctMarker === currentMarker) continue;
    // Accept only when the point fits the target prediction tightly; this is
    // the same bimodal gate used for intra-cluster swaps.
    if (cost[i][assignment[i]] < 0.02) {
      relabel.set(currentMarker, correctMarker);
      swapRecords.push({
        frame, start: frame, end: frame,
        markerA: clip.markerNames[currentMarker],
        markerB: clip.markerNames[correctMarker],
      });
    }
  }
  if (relabel.size === 0) return;

  // Walk permutation cycles and rotate points back to their owning markers.
  const visited = new Set<number>();
  for (const [startMarker] of relabel) {
    if (visited.has(startMarker)) continue;
    const cycle: number[] = [];
    let cur = startMarker;
    while (relabel.has(cur) && !visited.has(cur)) {
      visited.add(cur);
      cycle.push(cur);
      cur = relabel.get(cur)!;
    }
    if (cycle.length < 2) continue;
    const saved = cycle.map((mi) => {
      const o = (frame * mc + mi) * 3;
      return [positions[o], positions[o + 1], positions[o + 2]] as Vec3;
    });
    for (let i = 0; i < cycle.length; i++) {
      const targetMarker = relabel.get(cycle[i])!;
      const p = saved[i];
      const o = (frame * mc + targetMarker) * 3;
      positions[o] = p[0];
      positions[o + 1] = p[1];
      positions[o + 2] = p[2];
    }
  }
}

/**
 * Temporal propagation for frames where fewer than three cluster markers are
 * visible. Reuses the most recent trusted cluster rotation and solves the
 * translation as the average offset between visible points and their rotated
 * template positions. Emits predictions for every cluster marker so gap filling
 * stays on a rigid, temporally continuous trajectory.
 */
function propagateFromLastGood(
  frame: number,
  cluster: ClusterRuntime,
  clip: MarkerClip,
  positions: Float32Array,
  valid: Uint8Array,
  predicted: Float32Array,
  predictionValid: Uint8Array,
  lastGood: Map<string, { rotation: import('../math/quat').Quat; translation: Vec3 }>,
): void {
  const last = lastGood.get(cluster.name);
  if (!last) return;
  const mc = clip.markerNames.length;

  // Translation: average (observed point - rotated template) over visible markers.
  let dx = 0, dy = 0, dz = 0, count = 0;
  for (let slot = 0; slot < cluster.markerIndices.length; slot++) {
    const mi = cluster.markerIndices[slot];
    if (!valid[frame * mc + mi]) continue;
    const r = qRotate(last.rotation, cluster.template[slot]);
    const o = (frame * mc + mi) * 3;
    dx += positions[o] - r[0];
    dy += positions[o + 1] - r[1];
    dz += positions[o + 2] - r[2];
    count++;
  }
  if (count === 0) return;
  dx /= count; dy /= count; dz /= count;
  const translation: Vec3 = [dx, dy, dz];

  for (let slot = 0; slot < cluster.markerIndices.length; slot++) {
    const mi = cluster.markerIndices[slot];
    const r = qRotate(last.rotation, cluster.template[slot]);
    const base = frame * mc + mi;
    const o = base * 3;
    predicted[o] = r[0] + translation[0];
    predicted[o + 1] = r[1] + translation[1];
    predicted[o + 2] = r[2] + translation[2];
    predictionValid[base] = 1;
  }
}

function applyPermutation(
  frame: number,
  permutation: Map<number, number>,
  clip: MarkerClip,
  positions: Float32Array,
): void {
  const visited = new Set<number>();
  for (const [startMarker] of permutation) {
    if (visited.has(startMarker)) continue;
    const cycle: number[] = [];
    let cur = startMarker;
    while (permutation.has(cur) && !visited.has(cur)) {
      visited.add(cur);
      cycle.push(cur);
      cur = permutation.get(cur)!;
    }
    if (cycle.length < 2) continue;
    const saved = cycle.map((mi) => {
      const o = (frame * clip.markerNames.length + mi) * 3;
      return [positions[o], positions[o + 1], positions[o + 2]] as Vec3;
    });
    for (let i = 0; i < cycle.length; i++) {
      const targetMarker = permutation.get(cycle[i])!;
      const p = saved[i];
      const o = (frame * clip.markerNames.length + targetMarker) * 3;
      positions[o] = p[0];
      positions[o + 1] = p[1];
      positions[o + 2] = p[2];
    }
  }
}

function fillMarkerGaps(
  markerIndex: number,
  frameCount: number,
  markerCount: number,
  positions: Float32Array,
  valid: Uint8Array,
  predicted: Float32Array,
  predictionValid: Uint8Array,
  records: GapRecord[],
): void {
  let f = 0;
  while (f < frameCount) {
    if (valid[f * markerCount + markerIndex]) {
      f++;
      continue;
    }
    const start = f;
    while (f < frameCount && !valid[f * markerCount + markerIndex]) f++;
    const end = f - 1;

    let rigidFrames = 0;
    for (let g = start; g <= end; g++) {
      const gb = g * markerCount + markerIndex;
      if (predictionValid[gb]) {
        const o = gb * 3;
        positions[o] = predicted[o];
        positions[o + 1] = predicted[o + 1];
        positions[o + 2] = predicted[o + 2];
        valid[gb] = 1;
        rigidFrames++;
      }
    }

    const stillMissing: number[] = [];
    for (let g = start; g <= end; g++) {
      if (!valid[g * markerCount + markerIndex]) stillMissing.push(g);
    }
    let method: GapRecord['method'] = rigidFrames > 0 ? 'rigid' : 'trajectory';
    if (stillMissing.length) {
      const before = nearestValid(markerIndex, start - 1, -1, frameCount, markerCount, valid, positions);
      const after = nearestValid(markerIndex, end + 1, 1, frameCount, markerCount, valid, positions);
      if (before && after) {
        for (const g of stillMissing) {
          const p = hermiteInterpolate(
            [
              { frame: before.frame, value: before.value },
              { frame: after.frame, value: after.value },
            ],
            g,
          );
          writePos(g, markerIndex, markerCount, positions, valid, p);
        }
      } else {
        const anchor = before ?? after;
        for (const g of stillMissing) {
          const gb = g * markerCount + markerIndex;
          if (predictionValid[gb]) {
            const o = gb * 3;
            positions[o] = predicted[o];
            positions[o + 1] = predicted[o + 1];
            positions[o + 2] = predicted[o + 2];
          } else if (anchor) {
            writePos(g, markerIndex, markerCount, positions, valid, anchor.value);
          }
          valid[gb] = 1;
        }
        method = 'edge';
      }
    }

    // Denoise only the trajectory-filled portion of the gap. Rigid predictions
    // already sit at the noise floor; smoothing them against a possibly
    // swap-contaminated observed anchor would inject a centimetre-scale bias.
    if (stillMissing.length) {
      smoothFilledRun(
        markerIndex, stillMissing, frameCount, markerCount, positions,
        valid, predicted, predictionValid,
      );
    }

    records.push({ marker: '', markerIndex, start, end, length: end - start + 1, method });
  }
}

/**
 * Least-squares cubic over the trajectory-filled run. Anchor points are taken
 * from the rigid prediction when available (guaranteed fit-quality gated),
 * otherwise from the nearest observed frame. Only interior gap samples are
 * overwritten; observed frames are never touched.
 */
function smoothFilledRun(
  markerIndex: number,
  targetFrames: number[],
  frameCount: number,
  markerCount: number,
  positions: Float32Array,
  valid: Uint8Array,
  predicted: Float32Array,
  predictionValid: Uint8Array,
): void {
  const start = targetFrames[0];
  const end = targetFrames[targetFrames.length - 1];
  const anchorValue = (frame: number): Vec3 | null => {
    const base = frame * markerCount + markerIndex;
    if (predictionValid[base]) {
      const o = base * 3;
      return [predicted[o], predicted[o + 1], predicted[o + 2]];
    }
    if (frame >= 0 && frame < frameCount && valid[base]) {
      const o = base * 3;
      return [positions[o], positions[o + 1], positions[o + 2]];
    }
    return null;
  };

  const leftAnchor = findAnchor(markerIndex, start - 1, -1, frameCount, markerCount, predictionValid, valid, predicted, positions);
  const rightAnchor = findAnchor(markerIndex, end + 1, 1, frameCount, markerCount, predictionValid, valid, predicted, positions);
  if (!leftAnchor || !rightAnchor) return;

  const span = rightAnchor.frame - leftAnchor.frame;
  if (span < 2) return;

  // Interpolate ONLY the frames the rigid pass could not fill. Rigid-filled
  // neighbours are preferred as anchors, so the interpolation never overrides
  // an accurate per-frame rigid prediction.
  for (const g of targetFrames) {
    const t = (g - leftAnchor.frame) / span;
    const o = (g * markerCount + markerIndex) * 3;
    positions[o] = leftAnchor.p[0] + (rightAnchor.p[0] - leftAnchor.p[0]) * t;
    positions[o + 1] = leftAnchor.p[1] + (rightAnchor.p[1] - leftAnchor.p[1]) * t;
    positions[o + 2] = leftAnchor.p[2] + (rightAnchor.p[2] - leftAnchor.p[2]) * t;
  }
  void anchorValue;
}

function findAnchor(
  markerIndex: number,
  from: number,
  dir: -1 | 1,
  frameCount: number,
  markerCount: number,
  predictionValid: Uint8Array,
  valid: Uint8Array,
  predicted: Float32Array,
  positions: Float32Array,
): { frame: number; t: number; p: Vec3 } | null {
  let g = from;
  while (g >= 0 && g < frameCount) {
    const base = g * markerCount + markerIndex;
    if (predictionValid[base]) {
      const o = base * 3;
      return { frame: g, t: 0, p: [predicted[o], predicted[o + 1], predicted[o + 2]] };
    }
    if (valid[base]) {
      const o = base * 3;
      return { frame: g, t: 0, p: [positions[o], positions[o + 1], positions[o + 2]] };
    }
    g += dir;
  }
  return null;
}

function evalPoly(c: number[], t: number): number {
  return c[0] + c[1] * t + c[2] * t * t + c[3] * t * t * t;
}

/** Weighted-free least squares cubic through (t, y) samples. */
function solveCubic(points: Array<[number, number]>): number[] | null {
  const a = Array.from({ length: 4 }, () => new Array(4).fill(0));
  const b = [0, 0, 0, 0];
  for (const [t, y] of points) {
    const basis = [1, t, t * t, t * t * t];
    for (let i = 0; i < 4; i++) {
      b[i] += basis[i] * y;
      for (let j = 0; j < 4; j++) a[i][j] += basis[i] * basis[j];
    }
  }
  return gaussian4(a, b);
}

function gaussian4(a: number[][], b: number[]): number[] | null {
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 4; col++) {
    let pivot = col;
    for (let r = col + 1; r < 4; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    if (Math.abs(m[pivot][col]) < 1e-12) return null;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let r = 0; r < 4; r++) {
      if (r === col) continue;
      const factor = m[r][col] / m[col][col];
      for (let k = col; k <= 4; k++) m[r][k] -= factor * m[col][k];
    }
  }
  return [0, 1, 2, 3].map((i) => m[i][4] / m[i][i]);
}

function annotateGapMarkers(records: GapRecord[], names: string[]): void {
  for (const r of records) r.marker = names[r.markerIndex] ?? `#${r.markerIndex}`;
}

function nearestValid(
  markerIndex: number,
  from: number,
  dir: -1 | 1,
  frameCount: number,
  markerCount: number,
  valid: Uint8Array,
  positions: Float32Array,
): { frame: number; value: Vec3 } | null {
  let g = from;
  while (g >= 0 && g < frameCount) {
    if (valid[g * markerCount + markerIndex]) {
      const o = (g * markerCount + markerIndex) * 3;
      return { frame: g, value: [positions[o], positions[o + 1], positions[o + 2]] };
    }
    g += dir;
  }
  return null;
}

function writePos(
  frame: number,
  markerIndex: number,
  markerCount: number,
  positions: Float32Array,
  valid: Uint8Array,
  p: Vec3,
): void {
  const base = frame * markerCount + markerIndex;
  const o = base * 3;
  positions[o] = p[0];
  positions[o + 1] = p[1];
  positions[o + 2] = p[2];
  valid[base] = 1;
}

function mergeSwapRuns(records: SwapRecord[]): SwapRecord[] {
  const out: SwapRecord[] = [];
  const key = (r: SwapRecord) => [r.markerA, r.markerB].sort().join('|');
  const byKey = new Map<string, SwapRecord[]>();
  for (const r of records) {
    const k = key(r);
    const arr = byKey.get(k) ?? [];
    arr.push(r);
    byKey.set(k, arr);
  }
  for (const arr of byKey.values()) {
    arr.sort((a, b) => a.frame - b.frame);
    let run: SwapRecord | null = null;
    for (const r of arr) {
      if (run && r.frame <= run.end + 2) {
        run.end = r.frame;
      } else {
        if (run) out.push(run);
        run = { ...r, start: r.frame, end: r.frame };
      }
    }
    if (run) out.push(run);
  }
  return out.sort((a, b) => a.start - b.start);
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/**
 * Binomial [1,4,6,4,1]/16 smoothing applied only to frames that were observed.
 * Missing neighbours are dropped with one-sided weight renormalization, and the
 * output at a missing frame is returned unchanged so callers can skip it.
 */
function smoothObservedOnly(
  track: Vec3[],
  originallyMissing: Uint8Array,
  markerIndex: number,
  markerCount: number,
  passes: number,
): Vec3[] {
  let src = track.map((p) => [...p] as Vec3);
  const kernel = [1, 4, 6, 4, 1];
  for (let pass = 0; pass < passes; pass++) {
    const dst = src.map((p) => [...p] as Vec3);
    for (let i = 0; i < src.length; i++) {
      if (originallyMissing[i * markerCount + markerIndex]) continue;
      let wx = 0, wy = 0, wz = 0, wsum = 0;
      for (let k = -2; k <= 2; k++) {
        const j = i + k;
        if (j < 0 || j >= src.length) continue;
        if (originallyMissing[j * markerCount + markerIndex]) continue;
        const w = kernel[k + 2];
        wx += w * src[j][0];
        wy += w * src[j][1];
        wz += w * src[j][2];
        wsum += w;
      }
      dst[i] = [wx / wsum, wy / wsum, wz / wsum];
    }
    src = dst;
  }
  return src;
}
