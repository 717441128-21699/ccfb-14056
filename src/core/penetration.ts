/**
 * 肢体穿模检测:把骨骼视为胶囊体,检测非相邻胶囊对的相交。
 * 输出穿透深度最大的逐帧问题(同帧同骨对只保留一条),并在连续帧区间上做聚合,
 * 避免时间轴被每帧重复事件淹没。
 */
import type { PenetrationIssue, Rig, SkeletonClip } from './types.js';
import { JOINT, JOINT_COUNT } from './rig.js';

/** 各关节处的肢体半径(米,基准体型;整体随身高缩放)。 */
const JOINT_RADIUS: Record<number, number> = {
  [JOINT.Hips]: 0.09,
  [JOINT.Chest]: 0.11,
  [JOINT.Head]: 0.09,
  [JOINT.LArm]: 0.05, [JOINT.LForeArm]: 0.045, [JOINT.LHand]: 0.035,
  [JOINT.RArm]: 0.05, [JOINT.RForeArm]: 0.045, [JOINT.RHand]: 0.035,
  [JOINT.LUpLeg]: 0.07, [JOINT.LLeg]: 0.055, [JOINT.LFoot]: 0.045,
  [JOINT.RUpLeg]: 0.07, [JOINT.RLeg]: 0.055, [JOINT.RFoot]: 0.045,
};

/** 需要检测的胶囊段 [起点关节, 终点关节]。 */
const SEGMENTS: Array<[number, number]> = [
  [JOINT.LUpLeg, JOINT.LLeg], [JOINT.LLeg, JOINT.LFoot],
  [JOINT.RUpLeg, JOINT.RLeg], [JOINT.RLeg, JOINT.RFoot],
  [JOINT.LArm, JOINT.LForeArm], [JOINT.LForeArm, JOINT.LHand],
  [JOINT.RArm, JOINT.RForeArm], [JOINT.RForeArm, JOINT.RHand],
  [JOINT.Chest, JOINT.Neck], [JOINT.Neck, JOINT.Head],
  [JOINT.Hips, JOINT.Chest],
];

function segRadius(a: number, b: number): number {
  return ((JOINT_RADIUS[a] ?? 0.05) + (JOINT_RADIUS[b] ?? 0.05)) * 0.5;
}

/** 两线段最近距离。 */
function segmentDistance(
  p: Float32Array, po: number, q: Float32Array, qo: number,
  a: Float32Array, ao: number, b: Float32Array, bo: number,
): number {
  const d1x = q[qo] - p[po], d1y = q[qo + 1] - p[po + 1], d1z = q[qo + 2] - p[po + 2];
  const d2x = b[bo] - a[ao], d2y = b[bo + 1] - a[ao + 1], d2z = b[bo + 2] - a[ao + 2];
  const rx = p[po] - a[ao], ry = p[po + 1] - a[ao + 1], rz = p[po + 2] - a[ao + 2];
  const a11 = d1x * d1x + d1y * d1y + d1z * d1z || 1e-9;
  const a22 = d2x * d2x + d2y * d2y + d2z * d2z || 1e-9;
  const a12 = d1x * d2x + d1y * d2y + d1z * d2z;
  const b1 = d1x * rx + d1y * ry + d1z * rz;
  const b2 = d2x * rx + d2y * ry + d2z * rz;
  const denom = a11 * a22 - a12 * a12 || 1e-9;
  let s = (a12 * b2 - a22 * b1) / denom;
  let t = (a11 * b2 - a12 * b1) / denom;
  s = Math.min(1, Math.max(0, s));
  t = Math.min(1, Math.max(0, t));
  // 再钳一次 s(标准做法:固定 t 后重投影)
  const s2 = (t * a12 - b1) / a11;
  if (s2 >= 0 && s2 <= 1) s = s2;
  const cx = p[po] + s * d1x - (a[ao] + t * d2x);
  const cy = p[po + 1] + s * d1y - (a[ao + 1] + t * d2y);
  const cz = p[po + 2] + s * d1z - (a[ao + 2] + t * d2z);
  return Math.hypot(cx, cy, cz);
}

function areAdjacent(rig: Rig, a: number, b: number): boolean {
  return rig.joints[a].parent === b || rig.joints[b].parent === a;
}

export interface PenetrationOptions {
  /** 深度小于该值(米)的接触忽略。 */
  minDepth?: number;
  heightScale?: number;
}

export function detectPenetration(
  skel: SkeletonClip,
  rig: Rig,
  opts: PenetrationOptions = {},
): PenetrationIssue[] {
  const minDepth = opts.minDepth ?? 0.005;
  const scale = opts.heightScale ?? 1;
  const F = skel.frameCount;
  const issues: PenetrationIssue[] = [];

  // 预计算互不相邻的胶囊对
  const pairs: Array<{ sa: number; ea: number; sb: number; eb: number; r: number }> = [];
  for (let i = 0; i < SEGMENTS.length; i++) {
    for (let j = i + 1; j < SEGMENTS.length; j++) {
      const [sa, ea] = SEGMENTS[i];
      const [sb, eb] = SEGMENTS[j];
      if (new Set([sa, ea, sb, eb]).size < 4) continue;
      if (areAdjacent(rig, sa, sb) || areAdjacent(rig, sa, eb) || areAdjacent(rig, ea, sb) || areAdjacent(rig, ea, eb)) continue;
      pairs.push({ sa, ea, sb, eb, r: (segRadius(sa, ea) + segRadius(sb, eb)) * scale });
    }
  }

  for (let f = 0; f < F; f++) {
    const jp = skel.jointPositions;
    for (const pair of pairs) {
      const d = segmentDistance(
        jp, (f * JOINT_COUNT + pair.sa) * 3, jp, (f * JOINT_COUNT + pair.ea) * 3,
        jp, (f * JOINT_COUNT + pair.sb) * 3, jp, (f * JOINT_COUNT + pair.eb) * 3,
      );
      const depth = pair.r - d;
      if (depth >= minDepth) {
        issues.push({ frame: f, jointA: pair.sa, jointB: pair.sb, depth });
      }
    }
  }
  return issues;
}

/** 把逐帧问题按(骨对)聚合成区间事件,每条记录深度取区间最大值。 */
export function aggregateIssues(issues: PenetrationIssue[]): PenetrationIssue[] {
  const map = new Map<string, PenetrationIssue & { endFrame: number }>();
  const sorted = [...issues].sort((a, b) => a.frame - b.frame);
  for (const iss of sorted) {
    const key = `${iss.jointA}-${iss.jointB}`;
    const prev = map.get(key);
    if (prev && iss.frame <= prev.endFrame + 3) {
      prev.endFrame = iss.frame;
      prev.depth = Math.max(prev.depth, iss.depth);
    } else {
      map.set(key, { ...iss, endFrame: iss.frame });
    }
  }
  return [...map.values()].map(({ endFrame: _end, ...rest }) => rest);
}
