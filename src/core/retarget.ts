import { qMultiply, qNormalize, qRotate, type Quat } from '../math/quat';
import type { Vec3 } from '../math/vec3';
import { lerp } from '../math/vec3';
import type { SkeletonDefinition, SolvedSkeleton } from './types';

/**
 * Body proportions used by the retargeter. Multipliers are applied to the
 * source skeleton's rest-pose bone lengths; `rootHeight` shifts the pelvis.
 */
export interface BodyProfile {
  id: string;
  label: string;
  /** Per-bone length multiplier keyed by child joint name; fallback 1. */
  boneScale: Record<string, number>;
  /** Static pelvis height offset in metres (positive = taller). */
  rootHeightOffset: number;
  /** Overall uniform marker/skeleton scale applied at display time. */
  uniformScale: number;
}

export const BODY_PROFILES: BodyProfile[] = [
  {
    id: 'standard',
    label: '标准身材 (1.75m)',
    boneScale: {},
    rootHeightOffset: 0,
    uniformScale: 1,
  },
  {
    id: 'tall',
    label: '高个长腿 (1.90m)',
    boneScale: {
      lHip: 1.0, rHip: 1.0,
      lKnee: 1.18, rKnee: 1.18,
      lAnkle: 1.15, rAnkle: 1.15,
      lElbow: 1.08, rElbow: 1.08,
      lWrist: 1.08, rWrist: 1.08,
      spine1: 1.06, spine2: 1.06, neck: 1.04,
    },
    rootHeightOffset: 0.08,
    uniformScale: 1.05,
  },
  {
    id: 'short',
    label: '矮个壮硕 (1.60m)',
    boneScale: {
      lKnee: 0.86, rKnee: 0.86,
      lAnkle: 0.88, rAnkle: 0.88,
      lElbow: 0.9, rElbow: 0.9,
      lWrist: 0.9, rWrist: 0.9,
      spine1: 0.94, spine2: 0.94,
    },
    rootHeightOffset: -0.06,
    uniformScale: 0.95,
  },
  {
    id: 'longarms',
    label: '长臂窄肩 (臂展 +15%)',
    boneScale: {
      lElbow: 1.15, rElbow: 1.15,
      lWrist: 1.15, rWrist: 1.15,
      lCollar: 0.92, rCollar: 0.92,
      lShoulder: 0.92, rShoulder: 0.92,
    },
    rootHeightOffset: 0,
    uniformScale: 1,
  },
];

export interface RetargetedSkeleton {
  profile: BodyProfile;
  frameCount: number;
  fps: number;
  definition: SkeletonDefinition;
  /** World joint positions after retargeting. */
  positions: Float32Array;
  /** Local rotations copied from the source (pose-preserving). */
  rotations: Float32Array;
}

/** Build a skeleton definition whose rest offsets match the body profile. */
export function scaledSkeleton(base: SkeletonDefinition, profile: BodyProfile): SkeletonDefinition {
  const joints = base.joints.map((j) => {
    const s = profile.boneScale[j.name] ?? 1;
    return {
      ...j,
      restOffset: [j.restOffset[0] * s, j.restOffset[1] * s, j.restOffset[2] * s] as Vec3,
      radius: j.radius * (0.8 + 0.2 * profile.uniformScale),
    };
  });
  return {
    joints,
    jointIndex: base.jointIndex,
    clusters: base.clusters,
    bindings: base.bindings,
  };
}

/**
 * Retarget a solved skeleton onto another body.
 *
 * Rotations are preserved joint-by-joint so the pose reads identically;
 * positions are rebuilt with the target bone lengths via forward kinematics.
 * Planted foot contacts stay world-locked: the target ankle is projected back
 * onto the source ankle's contact position during contact frames.
 */
export function retarget(source: SolvedSkeleton, profile: BodyProfile): RetargetedSkeleton {
  const targetDef = scaledSkeleton(source.definition, profile);
  const jointCount = targetDef.joints.length;
  const frameCount = source.frameCount;
  const out = new Float32Array(frameCount * jointCount * 3);
  const srcPos = source.poses.positions;
  const srcRot = source.poses.rotations;

  const contactMask = new Map<'L' | 'R', Uint8Array>([
    ['L', new Uint8Array(frameCount)],
    ['R', new Uint8Array(frameCount)],
  ]);
  for (const c of source.contacts) {
    for (let f = c.start; f <= c.end; f++) contactMask.get(c.side)![f] = 1;
  }

  for (let f = 0; f < frameCount; f++) {
    // Rebuild positions top-down using target bone lengths and source rotations.
    for (let j = 0; j < jointCount; j++) {
      const joint = targetDef.joints[j];
      const o = (f * jointCount + j) * 3;
      if (joint.parent === null) {
        const sp = (f * jointCount) * 3;
        out[o] = srcPos[sp] * profile.uniformScale;
        out[o + 1] = srcPos[sp + 1] * profile.uniformScale + profile.rootHeightOffset;
        out[o + 2] = srcPos[sp + 2] * profile.uniformScale;
      } else {
        const qo = (f * jointCount + j) * 4;
        const localQ = [srcRot[qo], srcRot[qo + 1], srcRot[qo + 2], srcRot[qo + 3]] as const;
        const parentO = (f * jointCount + joint.parent) * 3;
        const parentWorldQ = worldRotation(source, f, joint.parent);
        const off = qRotate(parentWorldQ, joint.restOffset);
        out[o] = out[parentO] + off[0];
        out[o + 1] = out[parentO] + off[1];
        out[o + 2] = out[parentO] + off[2];
        void localQ;
      }
    }

    // Foot contact lock: keep retargeted ankles exactly on source contact points.
    for (const side of ['L', 'R'] as const) {
      if (!contactMask.get(side)![f]) continue;
      const ankleName = side === 'L' ? 'lAnkle' : 'rAnkle';
      const ai = targetDef.jointIndex.get(ankleName)!;
      const to = (f * jointCount + ai) * 3;
      const so = (f * jointCount + ai) * 3;
      const dx = srcPos[so] - out[to];
      const dy = srcPos[so + 1] - out[to + 1];
      const dz = srcPos[so + 2] - out[to + 2];
      // Translate the whole leg subtree so the planted foot does not move.
      const legIds = side === 'L'
        ? ['lHip', 'lKnee', 'lAnkle', 'lFoot', 'lToe']
        : ['rHip', 'rKnee', 'rAnkle', 'rFoot', 'rToe'];
      for (const name of legIds) {
        const idx = targetDef.jointIndex.get(name)!;
        const jo = (f * jointCount + idx) * 3;
        out[jo] += dx;
        out[jo + 1] += dy;
        out[jo + 2] += dz;
      }
    }
  }

  return {
    profile,
    frameCount,
    fps: source.fps,
    definition: targetDef,
    positions: out,
    rotations: source.poses.rotations.slice(),
  };
}

/** World rotation of a joint on a frame, composed from local rotations. */
function worldRotation(source: SolvedSkeleton, frame: number, joint: number): Quat {
  const def = source.definition;
  const chain: number[] = [];
  let cur: number | null = joint;
  while (cur !== null) {
    chain.push(cur);
    cur = def.joints[cur].parent;
  }
  chain.reverse();
  let q: Quat = [0, 0, 0, 1];
  for (const idx of chain) {
    const qo = (frame * def.joints.length + idx) * 4;
    const lq: Quat = [
      source.poses.rotations[qo],
      source.poses.rotations[qo + 1],
      source.poses.rotations[qo + 2],
      source.poses.rotations[qo + 3],
    ];
    q = multiply(q, lq);
  }
  return q;
}

function multiply(a: Quat, b: Quat): Quat {
  return qNormalize(qMultiply(a, b));
}

/** Interpolate two retargeted poses (used by the preview scrubber). */
export function blendPositions(a: Float32Array, b: Float32Array, t: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 3) {
    const va: Vec3 = [a[i], a[i + 1], a[i + 2]];
    const vb: Vec3 = [b[i], b[i + 1], b[i + 2]];
    const v = lerp(va, vb, t);
    out[i] = v[0];
    out[i + 1] = v[1];
    out[i + 2] = v[2];
  }
  return out;
}
