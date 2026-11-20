import type { Quat } from '../math/quat';
import { qFromUnitVectors, qMultiply, qNormalize } from '../math/quat';
import { add, cross, distance, normalize, scale, sub, vec3 } from '../math/vec3';
import type { Vec3 } from '../math/vec3';

export interface TwoBoneResult {
  mid: Vec3;
  end: Vec3;
  qStart: Quat;
  qMid: Quat;
}

/**
 * Analytic two-bone IK in world space.
 *
 * @param start      root joint world position
 * @param len1       first bone length
 * @param len2       second bone length
 * @param target     desired end-effector position
 * @param poleHint   preferred bend direction (e.g. knee forward)
 * @param qRestStart local-frame rest rotation of the start joint (maps the
 *                   rest bone direction onto its local axis)
 * @param qRestMid   rest rotation of the mid joint
 * @param restDir    bone direction in the start joint's rest frame
 */
export function solveTwoBone(
  start: Vec3,
  len1: number,
  len2: number,
  target: Vec3,
  poleHint: Vec3,
  qRestStart: Quat,
  qRestMid: Quat,
  restDir: Vec3 = [0, -1, 0],
): TwoBoneResult {
  const toTarget = sub(target, start);
  let d = distance(start, target);
  d = Math.min(d, len1 + len2 - 1e-6);
  const reach = d < 1e-6 ? 1e-6 : d;
  // Law of cosines: interior angle at the root joint
  const cosA = Math.min(1, Math.max(-1, (len1 * len1 + reach * reach - len2 * len2) / (2 * len1 * reach)));
  const a = Math.acos(cosA);

  const dir = d < 1e-6 ? [0, -1, 0] as Vec3 : normalize(toTarget);
  // Build a bend plane from the chain direction and the pole hint.
  const poleDot = poleHint[0] * dir[0] + poleHint[1] * dir[1] + poleHint[2] * dir[2];
  const bend = normalize(sub(poleHint, scale(dir, poleDot)));
  const mid = add(start, add(scale(dir, len1 * Math.cos(a)), scale(bend, len1 * Math.sin(a))));
  const end = target;

  // Rotations: world-space bone direction, then convert back into local frames.
  const bone1Dir = normalize(sub(mid, start));
  const bone2Dir = normalize(sub(end, mid));
  const qWorldStart = qFromUnitVectors(restDir, bone1Dir);
  const qStart = qNormalize(qMultiply(qWorldStart, qConjLocal(qRestStart)));

  // Mid joint rotation is relative to the start joint's frame.
  const qWorldMid = qFromUnitVectors(restDir, bone2Dir);
  const qMid = qNormalize(qMultiply(qConjLocal(qWorldStart), qMultiply(qWorldMid, qConjLocal(qRestMid))));

  return { mid, end, qStart, qMid };
}

function qConjLocal(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

/** Clamp a target inside the reachable two-bone envelope. */
export function clampReach(start: Vec3, target: Vec3, len1: number, len2: number): Vec3 {
  const maxD = len1 + len2 - 1e-5;
  const minD = Math.abs(len1 - len2) + 1e-5;
  const v = sub(target, start);
  const d = Math.hypot(v[0], v[1], v[2]);
  if (d < 1e-9) return add(start, [0, -minD, 0]);
  const nd = Math.min(maxD, Math.max(minD, d));
  return add(start, scale(scale(v, 1 / d), nd));
}

export function vec(x = 0, y = 0, z = 0): Vec3 {
  return vec3(x, y, z);
}
