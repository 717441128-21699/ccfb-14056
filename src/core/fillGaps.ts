/**
 * 丢点补全:
 *  1) 刚体补点(Kabsch):同刚体至少 3 个非共线观测点时,拟合刚体位姿并预测丢点;
 *     逐帧多趟迭代,使"补出来的点"能反过来参与下一趟拟合;
 *  2) 轨迹补点:刚体信息不足(观测 ≤2)的剩余空档,用前后有效控制点做 Catmull-Rom 插值。
 * 同时输出每个补点的来源(FillMethod),供 UI 与测试区分。
 */
import type { Gap, MarkerClip, Rig } from './types.js';
import { FillMethod } from './types.js';
import { rigidBodyFit } from './math.js';

export interface FillResult {
  clip: MarkerClip;
  fillMethod: Uint8Array;
  rigidFilled: number;
  /** 刚体时间先验补点数(观测不足 3 个,精度低于直接刚体配准)。 */
  priorFilled: number;
  trajectoryFilled: number;
  /** 未能补全的帧位数(首尾无锚点等极端情况)。 */
  unresolved: number;
}

export function fillGaps(source: MarkerClip, rig: Rig, _gaps: Gap[]): FillResult {
  const F = source.frameCount, M = source.markerNames.length;
  const positions = new Float32Array(source.positions);
  const observed = new Uint8Array(source.observed);
  // 原始观测快照:第一趟刚体拟合只信任真实观测,补点不回灌,避免误差累积
  const originalObserved = new Uint8Array(source.observed);
  const fillMethod = new Uint8Array(F * M);
  for (let i = 0; i < F * M; i++) fillMethod[i] = observed[i] ? FillMethod.Observed : FillMethod.Trajectory;

  let rigidFilled = 0;
  let priorFilled = 0;
  const R = new Float32Array(9);
  const t = new Float32Array(3);
  const world = new Float32Array(16 * 3);
  const mask = new Uint8Array(16);
  // 每个刚体最近一次可靠位姿(时间先验),用于观测点不足时的低置信补点
  const priorR = rig.bodies.map(() => new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
  const priorT = rig.bodies.map(() => new Float32Array(3));
  const priorValid = new Uint8Array(rig.bodies.length);
  const priorFrame = new Int32Array(rig.bodies.length).fill(-1);

  // 单趟扫描:刚体配准只基于原始观测,补点绝不回灌,从根本上杜绝迭代漂移。
  for (let f = 0; f < F; f++) {
    for (let b = 0; b < rig.bodies.length; b++) {
      const body = rig.bodies[b];
      const n = body.markerIndices.length;
      let active = 0;
      for (let i = 0; i < n; i++) {
        const m = body.markerIndices[i];
        const ok = originalObserved[f * M + m] === 1;
        mask[i] = ok ? 1 : 0;
        if (ok) {
          world.set(positions.subarray((f * M + m) * 3, (f * M + m) * 3 + 3), i * 3);
          active++;
        }
      }
      let poseOk = false;
      let highConfidence = false;
      if (active >= 3) {
        const rms = rigidBodyFit(body.localPositions, world, mask, n, R, 0, t, 0);
        if (rms !== null && rms <= 0.02 && Array.from(R).every(Number.isFinite) && Array.from(t).every(Number.isFinite)) {
          poseOk = true;
          highConfidence = true; // 全部基于原始观测 => 高置信刚体补点
          priorR[b].set(R); priorT[b].set(t); priorValid[b] = 1; priorFrame[b] = f;
        }
      } else if (active === 2 && priorValid[b] && f - priorFrame[b] <= 2) {
        // 仅 2 点观测且紧邻可靠帧:固定为先验旋转,平移用观测点最小二乘(低置信)
        R.set(priorR[b]);
        let sx = 0, sy = 0, sz = 0;
        for (let i = 0; i < n; i++) {
          if (!mask[i]) continue;
          const lo = i * 3;
          sx += world[i * 3] - (R[0] * body.localPositions[lo] + R[1] * body.localPositions[lo + 1] + R[2] * body.localPositions[lo + 2]);
          sy += world[i * 3 + 1] - (R[3] * body.localPositions[lo] + R[4] * body.localPositions[lo + 1] + R[5] * body.localPositions[lo + 2]);
          sz += world[i * 3 + 2] - (R[6] * body.localPositions[lo] + R[7] * body.localPositions[lo + 1] + R[8] * body.localPositions[lo + 2]);
        }
        t[0] = sx / active; t[1] = sy / active; t[2] = sz / active;
        let residual = 0;
        for (let i = 0; i < n; i++) {
          if (!mask[i]) continue;
          const lo = i * 3;
          const dx = R[0] * body.localPositions[lo] + R[1] * body.localPositions[lo + 1] + R[2] * body.localPositions[lo + 2] + t[0] - world[i * 3];
          const dy = R[3] * body.localPositions[lo] + R[4] * body.localPositions[lo + 1] + R[5] * body.localPositions[lo + 2] + t[1] - world[i * 3 + 1];
          const dz = R[6] * body.localPositions[lo] + R[7] * body.localPositions[lo + 1] + R[8] * body.localPositions[lo + 2] + t[2] - world[i * 3 + 2];
          residual += dx * dx + dy * dy + dz * dz;
        }
        if (Math.sqrt(residual / active) <= 0.015 && Array.from(t).every(Number.isFinite)) poseOk = true;
      } else if (active === 0 && priorValid[b] && f - priorFrame[b] <= 12) {
        // 整个刚体全丢:沿用最近完整位姿(低置信)
        R.set(priorR[b]);
        t.set(priorT[b]);
        if (Array.from(t).every(Number.isFinite)) poseOk = true;
      }
      if (!poseOk) continue;

      for (let i = 0; i < n; i++) {
        if (mask[i]) continue;
        const m = body.markerIndices[i];
        if (observed[f * M + m]) continue;
        const lo = i * 3;
        const px = R[0] * body.localPositions[lo] + R[1] * body.localPositions[lo + 1] + R[2] * body.localPositions[lo + 2] + t[0];
        const py = R[3] * body.localPositions[lo] + R[4] * body.localPositions[lo + 1] + R[5] * body.localPositions[lo + 2] + t[1];
        const pz = R[6] * body.localPositions[lo] + R[7] * body.localPositions[lo + 1] + R[8] * body.localPositions[lo + 2] + t[2];
        if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
        // 运动学合理性:预测点与该标记点最近时间邻域观测的距离不得超过 30cm,挡住离群补点
        if (!temporallyPlausible(positions, observed, f, m, M, F, px, py, pz, 0.3)) continue;
        const io = (f * M + m) * 3;
        positions[io] = px; positions[io + 1] = py; positions[io + 2] = pz;
        observed[f * M + m] = 1;
        fillMethod[f * M + m] = highConfidence ? FillMethod.RigidBody : FillMethod.Prior;
        if (highConfidence) rigidFilled++; else priorFilled++;
      }
    }
  }

  // 轨迹兜底:逐标记点处理。刚体迭代后仍可能有零散缺失帧(长档中段、刚体长期不可用),
  // 这里对每个标记点收集全序列有效锚点,再对所有仍缺失且落在锚点区间内的帧做三次插值。
  let trajectoryFilled = 0;
  let unresolved = 0;
  const isFiniteMarker = (f: number, m: number): boolean => {
    if (!observed[f * M + m]) return false;
    const o = (f * M + m) * 3;
    return Number.isFinite(positions[o]) && Number.isFinite(positions[o + 1]) && Number.isFinite(positions[o + 2]);
  };
  for (let m = 0; m < M; m++) {
    // 该标记点是否还有缺失
    let hasMissing = false;
    for (let f = 0; f < F; f++) if (!observed[f * M + m]) { hasMissing = true; break; }
    if (!hasMissing) continue;

    const anchorFrames: number[] = [];
    for (let f = 0; f < F; f++) if (isFiniteMarker(f, m)) anchorFrames.push(f);
    if (anchorFrames.length < 2) {
      for (let f = 0; f < F; f++) if (!observed[f * M + m]) unresolved++;
      continue;
    }
    const pts = new Float32Array(anchorFrames.length * 3);
    anchorFrames.forEach((af, i) => pts.set(positions.subarray((af * M + m) * 3, (af * M + m) * 3 + 3), i * 3));

    for (let f = 0; f < F; f++) {
      if (observed[f * M + m]) continue;
      if (f < anchorFrames[0] || f > anchorFrames[anchorFrames.length - 1]) {
        unresolved++;
        continue;
      }
      // 定位 f 落在哪两个相邻锚点 i、j 之间
      let j = 0;
      while (j < anchorFrames.length && anchorFrames[j] < f) j++;
      const i = Math.max(0, j - 1);
      j = Math.min(anchorFrames.length - 1, j);
      const out = new Float32Array(3);
      catmullRomFrames(anchorFrames, pts, i, j, f, out);
      if (!Number.isFinite(out[0]) || !Number.isFinite(out[1]) || !Number.isFinite(out[2])) {
        unresolved++;
        continue;
      }
      const io = (f * M + m) * 3;
      positions[io] = out[0]; positions[io + 1] = out[1]; positions[io + 2] = out[2];
      observed[f * M + m] = 1;
      fillMethod[f * M + m] = FillMethod.Trajectory;
      trajectoryFilled++;
    }
  }

  return {
    clip: { ...source, positions, observed },
    fillMethod,
    rigidFilled,
    priorFilled,
    trajectoryFilled,
    unresolved,
  };
}

/**
 * 运动学合理性校验:补点预测 (px,py,pz) 与该标记点最近的有效时间邻域观测
 * (前后各搜至多 12 帧)的距离须 ≤ maxDist。无任何邻域观测时放行(交给后续阶段)。
 */
function temporallyPlausible(
  positions: Float32Array, observed: Uint8Array,
  f: number, m: number, M: number, F: number,
  px: number, py: number, pz: number, maxDist: number,
): boolean {
  let nearest: number | null = null;
  for (let d = 1; d <= 12 && nearest === null; d++) {
    for (const ff of [f - d, f + d]) {
      if (ff < 0 || ff >= F || !observed[ff * M + m]) continue;
      const o = (ff * M + m) * 3;
      if (!Number.isFinite(positions[o])) continue;
      nearest = Math.hypot(px - positions[o], py - positions[o + 1], pz - positions[o + 2]);
      break;
    }
  }
  return nearest === null || nearest <= maxDist;
}

/**
 * 非均匀(向心)Catmull-Rom 插值,节点用真实帧号。
 * 给定锚点序列 frames/pts 与目标帧 f(位于锚点 i、j 之间),输出插值坐标到 out。
 * 向心参数化用相邻点距离的平方根作时间增量,能抑制长档上的过冲与自相交。
 * 边界处缺省锚点用端点钳制。
 */
function catmullRomFrames(
  frames: number[], pts: Float32Array,
  i: number, j: number, f: number, out: Float32Array,
): void {
  const n = frames.length;
  const i0 = Math.max(0, i - 1);
  const i3 = Math.min(n - 1, j + 1);
  const P = (idx: number): Float32Array => pts.subarray(idx * 3, idx * 3 + 3);
  const p0 = P(i0), p1 = P(i), p2 = P(j), p3 = P(i3);

  // 时间节点(秒等价于帧号,只差常数因子)
  const d = (a: Float32Array, b: Float32Array): number => Math.max(1e-6, Math.sqrt(
    (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2,
  ));
  const t0 = 0;
  const t1 = t0 + Math.sqrt(d(p0, p1));
  const t2 = t1 + Math.sqrt(d(p1, p2));
  const t3 = t2 + Math.sqrt(d(p2, p3));
  // 目标时刻在 [t1, t2] 内,按帧位置线性映射
  const spanFrame = Math.max(1, frames[j] - frames[i]);
  const t = t1 + (t2 - t1) * ((f - frames[i]) / spanFrame);

  for (let k = 0; k < 3; k++) {
    const a1 = ((t1 - t) / (t1 - t0)) * p0[k] + ((t - t0) / (t1 - t0)) * p1[k];
    const a2 = ((t2 - t) / (t2 - t1)) * p1[k] + ((t - t1) / (t2 - t1)) * p2[k];
    const a3 = ((t3 - t) / (t3 - t2)) * p2[k] + ((t - t2) / (t3 - t2)) * p3[k];
    const b1 = ((t2 - t) / (t2 - t0)) * a1 + ((t - t0) / (t2 - t0)) * a2;
    const b2 = ((t3 - t) / (t3 - t1)) * a2 + ((t - t1) / (t3 - t1)) * a3;
    out[k] = ((t2 - t) / (t2 - t1)) * b1 + ((t - t1) / (t2 - t1)) * b2;
  }
}

/**
 * 可选的轻量时域平滑(只作用于补点帧,不动原始观测),
 * 用补点帧前后观测点做 3 点加权,抑制刚体拟合在短档边界的抖动。
 */
export function smoothFilled(clip: MarkerClip, fillMethod: Uint8Array): void {
  const { positions, frameCount: F, markerNames } = clip;
  const M = markerNames.length;
  for (let m = 0; m < M; m++) {
    for (let f = 1; f < F - 1; f++) {
      if (fillMethod[f * M + m] === FillMethod.Observed) continue;
      if (fillMethod[(f - 1) * M + m] !== FillMethod.Observed || fillMethod[(f + 1) * M + m] !== FillMethod.Observed) continue;
      for (let k = 0; k < 3; k++) {
        const io = (f * M + m) * 3 + k;
        positions[io] = 0.25 * positions[io - M * 3] + 0.5 * positions[io] + 0.25 * positions[io + M * 3];
      }
    }
  }
}
