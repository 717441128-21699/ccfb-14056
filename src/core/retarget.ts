/**
 * 骨骼重定向:把源骨架的世界空间动画套到不同身材的目标骨架上。
 *
 * 策略(方向保持 + 骨长替换):
 *  - 根关节(Hips)按身高比例平移缩放;
 *  - 自顶向下遍历,每个关节沿"源骨架父子方向"放置到目标骨架的静帧骨长处;
 *  - 接触帧的脚踝强制保持源接触点(平移跟随根节点),由两骨 IK 吸收比例差异;
 *  - 最后用目标骨架的静帧框架重建世界旋转。
 */
import type { Rig, SkeletonClip } from './types.js';
import { JOINT, JOINT_COUNT, buildRig, type BuiltRig, type CharacterDimensions } from './rig.js';
import { buildJointFrame, twoBoneIK } from './frame.js';

export interface RetargetOptions {
  /** 源骨架静帧数据。 */
  sourceBuilt: BuiltRig;
  targetDims: CharacterDimensions;
  /** 源数据中已锁定的足部接触(可选)。 */
  contactFrames?: { left: Uint8Array; right: Uint8Array };
}

export function retarget(skel: SkeletonClip, opts: RetargetOptions): { skel: SkeletonClip; targetBuilt: BuiltRig } {
  const targetBuilt = buildRig(opts.targetDims);
  const srcRest = opts.sourceBuilt.restJointPositions;
  const dstRig: Rig = targetBuilt.rig;
  const dstRest = targetBuilt.restJointPositions;
  const F = skel.frameCount;

  const srcHipY = srcRest[JOINT.Hips * 3 + 1];
  const dstHipY = dstRest[JOINT.Hips * 3 + 1];
  const rootScale = dstHipY / srcHipY;

  const outPos = new Float32Array(F * JOINT_COUNT * 3);
  const outRot = new Float32Array(F * JOINT_COUNT * 9);

  const firstChild = new Array<number | null>(JOINT_COUNT).fill(null);
  for (let j = 0; j < JOINT_COUNT; j++) {
    const p = dstRig.joints[j].parent;
    if (p !== null && firstChild[p] === null) firstChild[p] = j;
  }

  // 目标静帧骨长
  const dstLen = new Float32Array(JOINT_COUNT);
  for (let j = 0; j < JOINT_COUNT; j++) {
    const p = dstRig.joints[j].parent;
    if (p === null) { dstLen[j] = 0; continue; }
    dstLen[j] = Math.hypot(
      dstRest[j * 3] - dstRest[p * 3],
      dstRest[j * 3 + 1] - dstRest[p * 3 + 1],
      dstRest[j * 3 + 2] - dstRest[p * 3 + 2],
    );
  }

  for (let f = 0; f < F; f++) {
    const sp = skel.jointPositions.subarray(f * JOINT_COUNT * 3, (f + 1) * JOINT_COUNT * 3);
    const op = outPos.subarray(f * JOINT_COUNT * 3, (f + 1) * JOINT_COUNT * 3);

    // 根节点:水平平移保持、垂直按比例
    op[0] = sp[0] * rootScale;
    op[1] = dstHipY + (sp[1] - srcHipY) * rootScale;
    op[2] = sp[2] * rootScale;

    // 自顶向下沿源方向放置(关节顺序即拓扑顺序)
    for (let j = 1; j < JOINT_COUNT; j++) {
      const p = dstRig.joints[j].parent!;
      let dx = sp[j * 3] - sp[p * 3];
      let dy = sp[j * 3 + 1] - sp[p * 3 + 1];
      let dz = sp[j * 3 + 2] - sp[p * 3 + 2];
      const n = Math.hypot(dx, dy, dz) || 1;
      dx /= n; dy /= n; dz /= n;
      op[j * 3] = op[p * 3] + dx * dstLen[j];
      op[j * 3 + 1] = op[p * 3 + 1] + dy * dstLen[j];
      op[j * 3 + 2] = op[p * 3 + 2] + dz * dstLen[j];
    }

    // 接触脚钉住源接触点(随根缩放),IK 修正膝盖
    const contact = opts.contactFrames;
    const chains = [
      { mask: contact?.left, hip: JOINT.LUpLeg, knee: JOINT.LLeg, ankle: JOINT.LFoot, toe: JOINT.LToe },
      { mask: contact?.right, hip: JOINT.RUpLeg, knee: JOINT.RLeg, ankle: JOINT.RFoot, toe: JOINT.RToe },
    ];
    for (const ch of chains) {
      if (!ch.mask || !ch.mask[f]) continue;
      const ao = ch.ankle * 3;
      const ho = ch.hip * 3;
      // 期望接触点(随根缩放),保留水平位置、垂直贴合目标静帧高度
      const want = [
        sp[ao] * rootScale,
        dstRest[ao + 1] + (sp[ao + 1] - srcRest[ao + 1]) * rootScale,
        sp[ao + 2] * rootScale,
      ];
      // 钳制到目标腿长可达域,避免 IK 外的极端骨长破坏
      const l1 = Math.hypot(
        dstRest[ch.knee * 3] - dstRest[ho], dstRest[ch.knee * 3 + 1] - dstRest[ho + 1], dstRest[ch.knee * 3 + 2] - dstRest[ho + 2],
      );
      const l2 = Math.hypot(
        dstRest[ao] - dstRest[ch.knee * 3], dstRest[ao + 1] - dstRest[ch.knee * 3 + 1], dstRest[ao + 2] - dstRest[ch.knee * 3 + 2],
      );
      let vx = want[0] - op[ho], vy = want[1] - op[ho + 1], vz = want[2] - op[ho + 2];
      const dist = Math.hypot(vx, vy, vz) || 1;
      const maxReach = (l1 + l2) * 0.999;
      if (dist > maxReach) {
        vx = vx / dist * maxReach; vy = vy / dist * maxReach; vz = vz / dist * maxReach;
      }
      op[ao] = op[ho] + vx; op[ao + 1] = op[ho + 1] + vy; op[ao + 2] = op[ho + 2] + vz;
      twoBoneIK(op, dstRest, ch.hip, ch.knee, ch.ankle);
      // 脚趾按目标静帧的相对偏移重建,保证趾骨长度与目标骨架一致
      const restToe = [
        dstRest[ch.toe * 3] - dstRest[ao],
        dstRest[ch.toe * 3 + 1] - dstRest[ao + 1],
        dstRest[ch.toe * 3 + 2] - dstRest[ao + 2],
      ];
      op[ch.toe * 3] = op[ao] + restToe[0];
      op[ch.toe * 3 + 1] = op[ao + 1] + restToe[1];
      op[ch.toe * 3 + 2] = op[ao + 2] + restToe[2];
    }

    // 重建世界旋转
    const orr = outRot.subarray(f * JOINT_COUNT * 9, (f + 1) * JOINT_COUNT * 9);
    for (let j = 0; j < JOINT_COUNT; j++) {
      buildJointFrame(dstRig, dstRest, op, j, firstChild, orr.subarray(j * 9, j * 9 + 9));
    }
  }

  return {
    skel: { frameCount: F, fps: skel.fps, jointPositions: outPos, jointRotations: outRot },
    targetBuilt,
  };
}

/** 由 FootLock 列表构造逐帧接触掩码。 */
export function contactMasksFromLocks(
  locks: { side: 'left' | 'right'; startFrame: number; endFrame: number }[],
  frameCount: number,
): { left: Uint8Array; right: Uint8Array } {
  const left = new Uint8Array(frameCount);
  const right = new Uint8Array(frameCount);
  for (const lk of locks) {
    const mask = lk.side === 'left' ? left : right;
    for (let f = lk.startFrame; f <= lk.endFrame; f++) mask[f] = 1;
  }
  return { left, right };
}
