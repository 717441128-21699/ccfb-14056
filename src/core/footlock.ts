/**
 * 足部接触检测与防滑锁定。
 *
 * 检测:脚踝高度贴近地面且水平速度低于阈值的连续帧段;
 * 锁定:把段内脚踝 x/z 钉到段内中位数位置、y 钉到接触高度,再用两骨 IK 反推膝盖,
 * 保证骨长不被破坏。输出 FootLock 供时间轴标注。
 */
import type { FootLock, Rig, SkeletonClip } from './types.js';
import { JOINT, JOINT_COUNT } from './rig.js';
import { twoBoneIK, buildJointFrame } from './frame.js';

export interface FootLockOptions {
  /** 脚踝低于该离地高度(米)才可能判为接触。 */
  contactHeight?: number;
  /** 水平速度阈值(米/秒)。 */
  speedThreshold?: number;
  /** 最短接触帧数,短于该长度的段视为噪声。 */
  minDuration?: number;
}

interface SideChain {
  side: 'left' | 'right';
  hip: number; knee: number; ankle: number; toe: number;
}

const CHAINS: SideChain[] = [
  { side: 'left', hip: JOINT.LUpLeg, knee: JOINT.LLeg, ankle: JOINT.LFoot, toe: JOINT.LToe },
  { side: 'right', hip: JOINT.RUpLeg, knee: JOINT.RLeg, ankle: JOINT.RFoot, toe: JOINT.RToe },
];

export function lockFeet(
  skel: SkeletonClip,
  rig: Rig,
  rest: Float32Array,
  opts: FootLockOptions = {},
): { locks: FootLock[] } {
  const h = opts.contactHeight ?? 0.12;
  const vTh = opts.speedThreshold ?? 0.25;
  const minDur = opts.minDuration ?? Math.max(3, Math.round(skel.fps * 0.05));
  const F = skel.frameCount;
  const dt = 1 / skel.fps;
  const locks: FootLock[] = [];

  const firstChild = new Array<number | null>(JOINT_COUNT).fill(null);
  for (let j = 0; j < JOINT_COUNT; j++) {
    const p = rig.joints[j].parent;
    if (p !== null && firstChild[p] === null) firstChild[p] = j;
  }

  for (const chain of CHAINS) {
    const contact: number[] = [];
    for (let f = 0; f < F; f++) {
      const ao = (f * JOINT_COUNT + chain.ankle) * 3;
      const y = skel.jointPositions[ao + 1];
      let speed = Infinity;
      if (f > 0 && f < F - 1) {
        const pa = ((f - 1) * JOINT_COUNT + chain.ankle) * 3;
        const pb = ((f + 1) * JOINT_COUNT + chain.ankle) * 3;
        speed = Math.hypot(
          skel.jointPositions[pb] - skel.jointPositions[pa],
          skel.jointPositions[pb + 2] - skel.jointPositions[pa + 2],
        ) / (2 * dt);
      }
      contact.push(y <= h && speed <= vTh ? 1 : 0);
    }

    let start = -1;
    for (let f = 0; f <= F; f++) {
      const on = f < F && contact[f] === 1;
      if (on && start < 0) start = f;
      if (!on && start >= 0) {
        const end = f - 1;
        if (end - start + 1 >= minDur) {
          applyLock(skel, rig, rest, chain, start, end, firstChild, locks);
        }
        start = -1;
      }
    }
  }
  return { locks };
}

function applyLock(
  skel: SkeletonClip, rig: Rig, rest: Float32Array,
  chain: SideChain, start: number, end: number,
  firstChild: Array<number | null>, locks: FootLock[],
): void {
  // 锁定目标:段内 x/z 的中位数,y 的最小值(避免整体抬离地面)
  const xs: number[] = [], zs: number[] = [], ys: number[] = [];
  for (let f = start; f <= end; f++) {
    const ao = (f * JOINT_COUNT + chain.ankle) * 3;
    xs.push(skel.jointPositions[ao]);
    ys.push(skel.jointPositions[ao + 1]);
    zs.push(skel.jointPositions[ao + 2]);
  }
  const median = (a: number[]): number => {
    const s = [...a].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)];
  };
  const lx = median(xs), ly = Math.min(...ys), lz = median(zs);

  for (let f = start; f <= end; f++) {
    const jp = skel.jointPositions.subarray(f * JOINT_COUNT * 3, (f + 1) * JOINT_COUNT * 3);
    const jr = skel.jointRotations.subarray(f * JOINT_COUNT * 9, (f + 1) * JOINT_COUNT * 9);
    const ao = chain.ankle * 3;
    // 脚趾先记录相对脚踝的偏移,踝移动后平移跟随
    const toeRel = [
      jp[chain.toe * 3] - jp[ao],
      jp[chain.toe * 3 + 1] - jp[ao + 1],
      jp[chain.toe * 3 + 2] - jp[ao + 2],
    ];
    jp[ao] = lx; jp[ao + 1] = ly; jp[ao + 2] = lz;
    twoBoneIK(jp, rest, chain.hip, chain.knee, chain.ankle);
    jp[chain.toe * 3] = lx + toeRel[0];
    jp[chain.toe * 3 + 1] = ly + toeRel[1];
    jp[chain.toe * 3 + 2] = lz + toeRel[2];
    // 膝/踝/趾世界旋转随新位置重建
    for (const j of [chain.knee, chain.ankle, chain.toe]) {
      buildJointFrame(rig, rest, jp, j, firstChild, jr.subarray(j * 9, j * 9 + 9));
    }
  }
  locks.push({
    side: chain.side,
    startFrame: start,
    endFrame: end,
    position: [lx, ly, lz],
  });
}
