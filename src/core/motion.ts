import { qFromAxisAngle, qMultiply, qNormalize, quatIdentity, type Quat } from '../math/quat';
import type { Vec3 } from '../math/vec3';
import { defaultSkeleton } from './skeleton';
import type { MotionKind } from './types';

export interface RawMotion {
  fps: number;
  frameCount: number;
  kind: MotionKind;
  /** World position of the pelvis joint per frame. */
  rootPosition: Vec3[];
  /** Local (relative to parent) rotations per joint per frame. */
  localRotations: Quat[][]; // [frame][joint]
}

interface MotionParams {
  cycleHz: number;
  strideM: number;
  rootBobM: number;
  hipSwingRad: number;
  kneeBendRad: number;
  armSwingRad: number;
  spineTwistRad: number;
  footLiftM: number;
  leanRad: number;
}

const PARAMS: Record<MotionKind, MotionParams> = {
  walk: {
    cycleHz: 0.9, strideM: 0.75, rootBobM: 0.022,
    hipSwingRad: 0.32, kneeBendRad: 0.55, armSwingRad: 0.45,
    spineTwistRad: 0.12, footLiftM: 0.05, leanRad: 0.05,
  },
  run: {
    cycleHz: 1.55, strideM: 1.35, rootBobM: 0.05,
    hipSwingRad: 0.55, kneeBendRad: 1.05, armSwingRad: 0.8,
    spineTwistRad: 0.2, footLiftM: 0.16, leanRad: 0.16,
  },
  jump: {
    cycleHz: 0.55, strideM: 0.5, rootBobM: 0.08,
    hipSwingRad: 0.4, kneeBendRad: 0.9, armSwingRad: 0.9,
    spineTwistRad: 0.08, footLiftM: 0.3, leanRad: 0.1,
  },
};

const SKELETON = defaultSkeleton();
const J = (name: string): number => SKELETON.jointIndex.get(name)!;

/**
 * Procedurally generate a walk/run/jump clip.
 * Walk & run are periodic locomotion; jump contains a crouch → takeoff →
 * flight → landing sequence with an airborne window.
 */
export function generateMotion(kind: MotionKind, frameCount: number, fps = 100): RawMotion {
  const p = PARAMS[kind];
  const rootPosition: Vec3[] = new Array(frameCount);
  const localRotations: Quat[][] = new Array(frameCount);
  const skel = SKELETON;
  const jointCount = skel.joints.length;

  // Jump timing (seconds): crouch, takeoff, land, recover.
  const jumpT0 = 1.2;
  const jumpT1 = 1.9;
  const jumpT2 = 2.7;
  const jumpT3 = 3.4;

  for (let f = 0; f < frameCount; f++) {
    const t = f / fps;
    const rots: Quat[] = new Array(jointCount);
    for (let j = 0; j < jointCount; j++) rots[j] = quatIdentity();

    const phase = 2 * Math.PI * p.cycleHz * t;
    const c = Math.cos(phase);
    const s = Math.sin(phase);

    let rootY = 0.95;
    let rootZ = 0;
    let rootX = 0;
    let travelSpeed: number;

    if (kind === 'jump') {
      travelSpeed = 0.55;
      rootX = 0;
      const jumpLift = jumpProfile(t, jumpT0, jumpT1, jumpT2, jumpT3);
      rootY = 0.95 + jumpLift;
      rootZ = -travelSpeed * t;
      buildJumpPose(rots, t, jumpT0, jumpT1, jumpT2, jumpT3, p);
    } else {
      travelSpeed = 2 * p.cycleHz * p.strideM;
      rootZ = -travelSpeed * t;
      rootY = 0.95 + p.rootBobM * Math.cos(2 * phase);
      rootX = 0.015 * Math.sin(phase);
      buildLocomotionPose(rots, phase, c, s, p);
    }

    rootPosition[f] = [rootX, rootY, rootZ];
    localRotations[f] = rots;
  }

  return { fps, frameCount, kind, rootPosition, localRotations };
}

function buildLocomotionPose(
  rots: Quat[],
  phase: number,
  c: number,
  s: number,
  p: MotionParams,
): void {
  // --- Spine: slight forward lean + counter twist ---
  rots[J('spine1')] = qFromAxisAngle([1, 0, 0], p.leanRad * 0.5);
  rots[J('spine2')] = qFromAxisAngle([1, 0, 0], p.leanRad);
  rots[J('neck')] = qFromAxisAngle([0, 1, 0], -p.spineTwistRad * 0.5 * s);
  rots[J('head')] = qFromAxisAngle([0, 1, 0], p.spineTwistRad * 0.5 * s);

  // --- Legs: opposite-phase hip flexion/extension around X, knee bend ---
  const lHip = p.hipSwingRad * s;
  const rHip = -p.hipSwingRad * s;
  rots[J('lHip')] = qFromAxisAngle([1, 0, 0], lHip);
  rots[J('rHip')] = qFromAxisAngle([1, 0, 0], rHip);
  // Knee flexes mostly when the leg swings forward (sin peaks).
  rots[J('lKnee')] = qFromAxisAngle([1, 0, 0], p.kneeBendRad * Math.max(0, s) + 0.08);
  rots[J('rKnee')] = qFromAxisAngle([1, 0, 0], p.kneeBendRad * Math.max(0, -s) + 0.08);
  // Ankles dorsiflex on swing, plantarflex at push-off.
  rots[J('lAnkle')] = qFromAxisAngle([1, 0, 0], -0.35 * Math.max(0, s) + 0.15 * Math.max(0, -s));
  rots[J('rAnkle')] = qFromAxisAngle([1, 0, 0], -0.35 * Math.max(0, -s) + 0.15 * Math.max(0, s));
  rots[J('lFoot')] = quatIdentity();
  rots[J('rFoot')] = quatIdentity();
  rots[J('lToe')] = qFromAxisAngle([1, 0, 0], 0.2 * Math.max(0, -s));
  rots[J('rToe')] = qFromAxisAngle([1, 0, 0], 0.2 * Math.max(0, s));

  // --- Arms: counter-swing, elbows bent ~90° when running ---
  const armBend = p.kneeBendRad * 0.55;
  rots[J('lCollar')] = qFromAxisAngle([0, 0, 1], 0.1);
  rots[J('rCollar')] = qFromAxisAngle([0, 0, 1], -0.1);
  rots[J('lShoulder')] = qMultiply(
    qFromAxisAngle([1, 0, 0], -p.armSwingRad * s),
    qFromAxisAngle([0, 0, 1], 0.12),
  );
  rots[J('rShoulder')] = qMultiply(
    qFromAxisAngle([1, 0, 0], p.armSwingRad * s),
    qFromAxisAngle([0, 0, 1], -0.12),
  );
  rots[J('lElbow')] = qFromAxisAngle([1, 0, 0], armBend + 0.15);
  rots[J('rElbow')] = qFromAxisAngle([1, 0, 0], armBend + 0.15);
  rots[J('lWrist')] = qFromAxisAngle([1, 0, 0], -0.15 * s);
  rots[J('rWrist')] = qFromAxisAngle([1, 0, 0], 0.15 * s);

  void phase;
  void c;
}

function buildJumpPose(
  rots: Quat[],
  t: number,
  t0: number,
  t1: number,
  t2: number,
  t3: number,
  p: MotionParams,
): void {
  let crouch = 0;
  let tuck = 0;
  let land = 0;
  let armUp = 0;

  if (t < t0) {
    // idle breathing
    const b = 0.02 * Math.sin(2 * Math.PI * 0.8 * t);
    rots[J('spine2')] = qFromAxisAngle([1, 0, 0], b);
  } else if (t < t1) {
    const u = smoothstep((t - t0) / (t1 - t0));
    crouch = u;
    armUp = u * 0.6;
  } else if (t < t2) {
    // airborne: tuck then extend
    const u = (t - t1) / (t2 - t1);
    tuck = Math.sin(Math.PI * u);
    armUp = 1;
  } else if (t < t3) {
    const u = smoothstep((t - t2) / (t3 - t2));
    land = 1 - u;
    armUp = 1 - u;
  }

  const kneeFlex = 0.15 + p.kneeBendRad * (0.85 * crouch + 0.7 * tuck + 0.8 * land);
  const hipFlex = p.hipSwingRad * (crouch + tuck * 0.6 + land);
  rots[J('spine2')] = qFromAxisAngle([1, 0, 0], p.leanRad * (crouch - tuck * 0.4));
  rots[J('lHip')] = qFromAxisAngle([1, 0, 0], hipFlex);
  rots[J('rHip')] = qFromAxisAngle([1, 0, 0], hipFlex);
  rots[J('lKnee')] = qFromAxisAngle([1, 0, 0], kneeFlex);
  rots[J('rKnee')] = qFromAxisAngle([1, 0, 0], kneeFlex);
  rots[J('lAnkle')] = qFromAxisAngle([1, 0, 0], -0.4 * tuck + 0.3 * crouch);
  rots[J('rAnkle')] = qFromAxisAngle([1, 0, 0], -0.4 * tuck + 0.3 * crouch);
  rots[J('lShoulder')] = qNormalize(qMultiply(
    qFromAxisAngle([1, 0, 0], -2.6 * armUp),
    qFromAxisAngle([0, 0, 1], 0.2),
  ));
  rots[J('rShoulder')] = qNormalize(qMultiply(
    qFromAxisAngle([1, 0, 0], -2.6 * armUp),
    qFromAxisAngle([0, 0, 1], -0.2),
  ));
  rots[J('lElbow')] = qFromAxisAngle([1, 0, 0], 0.4 + 0.6 * (1 - armUp));
  rots[J('rElbow')] = qFromAxisAngle([1, 0, 0], 0.4 + 0.6 * (1 - armUp));
}

/** Vertical lift of the root across the jump, metres. */
function jumpProfile(t: number, t0: number, t1: number, t2: number, t3: number): number {
  if (t < t0) return 0;
  if (t < t1) return -0.28 * smoothstep((t - t0) / (t1 - t0));
  if (t < t2) {
    const u = (t - t1) / (t2 - t1);
    return -0.28 + 0.62 * Math.sin(Math.PI * u);
  }
  if (t < t3) return -0.28 * (1 - smoothstep((t - t2) / (t3 - t2)));
  return 0;
}

function smoothstep(u: number): number {
  const x = Math.min(1, Math.max(0, u));
  return x * x * (3 - 2 * x);
}
