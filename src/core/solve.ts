/**
 * 标记点 -> 骨骼动画解算。
 *
 * 每个刚体在每帧做 Kabsch 配准得到驱动关节的世界旋转 R 与世界位置 P;
 * 未被刚体直接驱动的关节由父关节位置 + 父世界旋转 * 静帧偏移推导;
 * 最后由层级关系把世界旋转换算成局部旋转一并存储(世界数据仍为主输出)。
 */
import type { MarkerClip, Rig, SkeletonClip } from './types.js';
import { JOINT, JOINT_COUNT } from './rig.js';
import { rigidBodyFit } from './math.js';
import { buildJointFrame } from './frame.js';

export interface SolveOptions {
  /** 刚体配准 RMS 超过该值(米)视为该帧刚体不可用,沿用上一有效帧姿态。 */
  maxBodyRms?: number;
}

export function solveSkeleton(
  clip: MarkerClip,
  rig: Rig,
  restJointPositions: Float32Array,
  opts: SolveOptions = {},
): SkeletonClip {
  const maxRms = opts.maxBodyRms ?? 0.02;
  const F = clip.frameCount, M = clip.markerNames.length;
  const positions = new Float32Array(F * JOINT_COUNT * 3);
  const rotations = new Float32Array(F * JOINT_COUNT * 9);
  const localRotations = new Float32Array(F * JOINT_COUNT * 9);

  const firstChild = new Array<number | null>(JOINT_COUNT).fill(null);
  for (let j = 0; j < JOINT_COUNT; j++) {
    const p = rig.joints[j].parent;
    if (p !== null && firstChild[p] === null) firstChild[p] = j;
  }

  // 刚体 -> 关节的驱动表;未直接驱动的关节走层级推导
  const bodyJoint = rig.bodyToJoint;
  const drivenByBody = new Int16Array(JOINT_COUNT).fill(-1);
  rig.bodies.forEach((_, b) => { drivenByBody[bodyJoint[b]] = b; });

  // 上一有效帧的刚体位姿,供丢帧兜底
  const lastR: Float32Array[] = rig.bodies.map(() => new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
  const lastT: Float32Array[] = rig.bodies.map(() => new Float32Array(3));
  const lastValid = new Uint8Array(rig.bodies.length);

  const world = new Float32Array(16 * 3);
  const mask = new Uint8Array(16);
  const R = new Float32Array(9), t = new Float32Array(3);

  for (let f = 0; f < F; f++) {
    const jp = positions.subarray(f * JOINT_COUNT * 3, (f + 1) * JOINT_COUNT * 3);
    const jr = rotations.subarray(f * JOINT_COUNT * 9, (f + 1) * JOINT_COUNT * 9);

    // 1) 刚体配准
    for (let b = 0; b < rig.bodies.length; b++) {
      const body = rig.bodies[b];
      const n = body.markerIndices.length;
      let active = 0;
      for (let i = 0; i < n; i++) {
        const m = body.markerIndices[i];
        const ok = clip.observed[f * M + m] === 1;
        mask[i] = ok ? 1 : 0;
        if (ok) {
          world.set(clip.positions.subarray((f * M + m) * 3, (f * M + m) * 3 + 3), i * 3);
          active++;
        }
      }
      const j = bodyJoint[b];
      if (active >= 3) {
        const rms = rigidBodyFit(body.localPositions, world, mask, n, R, 0, t, 0);
        if (rms !== null && rms <= maxRms) {
          lastR[b].set(R); lastT[b].set(t); lastValid[b] = 1;
        }
      }
      if (!lastValid[b]) {
        // 首帧即不可用:退回静帧
        jr.set([1, 0, 0, 0, 1, 0, 0, 0, 1], j * 9);
        jp.set(restJointPositions.subarray(j * 3, j * 3 + 3), j * 3);
        continue;
      }
      const useR = lastR[b], useT = lastT[b];
      jr.set(useR, j * 9);
      // 刚体局部坐标以驱动关节为原点(local=0),故配准平移即关节世界位置
      jp[j * 3] = useT[0];
      jp[j * 3 + 1] = useT[1];
      jp[j * 3 + 2] = useT[2];
    }

    // 2) 未驱动关节:父位置 + 父世界旋转 * 静帧偏移(按拓扑顺序)
    for (let j = 0; j < JOINT_COUNT; j++) {
      if (drivenByBody[j] >= 0) continue;
      const p = rig.joints[j].parent;
      if (p === null) {
        jp.set(restJointPositions.subarray(j * 3, j * 3 + 3), j * 3);
        continue;
      }
      const off = rig.joints[j].restOffset;
      const pr = jr.subarray(p * 9, p * 9 + 9);
      jp[j * 3] = jp[p * 3] + pr[0] * off[0] + pr[1] * off[1] + pr[2] * off[2];
      jp[j * 3 + 1] = jp[p * 3 + 1] + pr[3] * off[0] + pr[4] * off[1] + pr[5] * off[2];
      jp[j * 3 + 2] = jp[p * 3 + 2] + pr[6] * off[0] + pr[7] * off[1] + pr[8] * off[2];
    }

    // 3) 未驱动关节的世界旋转由关节坐标框架重建
    for (let j = 0; j < JOINT_COUNT; j++) {
      if (drivenByBody[j] >= 0) continue;
      buildJointFrame(rig, restJointPositions, jp, j, firstChild, jr.subarray(j * 9, j * 9 + 9));
    }

    // 4) 局部旋转 R_local = R_parent^T * R_world
    const lr = localRotations.subarray(f * JOINT_COUNT * 9, (f + 1) * JOINT_COUNT * 9);
    for (let j = 0; j < JOINT_COUNT; j++) {
      const p = rig.joints[j].parent;
      const rw = jr.subarray(j * 9, j * 9 + 9);
      if (p === null) {
        lr.set(rw, j * 9);
      } else {
        const rp = jr.subarray(p * 9, p * 9 + 9);
        const out = lr.subarray(j * 9, j * 9 + 9);
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            out[r * 3 + c] =
              rp[c] * rw[r * 3] + rp[3 + c] * rw[r * 3 + 1] + rp[6 + c] * rw[r * 3 + 2];
          }
        }
      }
    }
  }

  return { frameCount: F, fps: clip.fps, jointPositions: positions, jointRotations: rotations };
}

export { JOINT };
