/**
 * 标签互换检测与修复(逐帧指派 + 时序平滑)。
 *
 * 每帧独立求解:把当前观测点与各标记点的轨迹预测做最优二分匹配(贪心最近邻),
 * 若某观测被指派给了非自身标签,即发生标签错位。连续帧上稳定的错位构成互换区间。
 * 该方案不维护跨帧开关状态,对连续/嵌套互换天然稳健。
 */
import type { MarkerClip, SwapEvent } from './types.js';

export interface RelabelResult {
  clip: MarkerClip;
  swaps: SwapEvent[];
}

interface MarkerState {
  pos: Float32Array;
  vel: Float32Array;
  initialized: boolean;
  lastFrame: number;
}

interface Interval { a: number; b: number; start: number; end: number }

export function detectAndFixSwaps(source: MarkerClip): RelabelResult {
  const F = source.frameCount;
  const M = source.markerNames.length;
  const positions = source.positions;
  const observed = source.observed;
  const states: MarkerState[] = Array.from({ length: M }, () => ({
    pos: new Float32Array(3),
    vel: new Float32Array(3),
    initialized: false,
    lastFrame: -1,
  }));
  const sigma = estimateNoise(source);
  // 指派错位判定门限:观测与某标签预测的距离低于此值才认为可能属于该标签
  const assignThreshold = Math.max(0.03, 10 * sigma);

  // 每帧每个标签实际对应的观测标记点(默认指向自身)
  const labelOfObs = new Int32Array(F * M);
  for (let i = 0; i < F * M; i++) labelOfObs[i] = i % M;

  const swaps: SwapEvent[] = [];

  for (let f = 1; f < F; f++) {
    // 本帧可见标记点只看原始观测;尚未初始化的标签先用当前原始观测引导,
    // 避免早期标签错位导致某标签的轨迹状态永远无法建立。
    const visible: number[] = [];
    for (let m = 0; m < M; m++) if (observed[f * M + m]) visible.push(m);
    for (const m of visible) {
      const st = states[m];
      if (!st.initialized) {
        st.pos.set(positions.subarray((f * M + m) * 3, (f * M + m) * 3 + 3));
        st.lastFrame = f;
        st.initialized = true;
      }
    }
    if (visible.length < 2) {
      updateStates(states, source, f, visible, visible);
      continue;
    }
    const predOf = new Map<number, number[]>();
    for (const m of visible) {
      const st = states[m];
      const dt = Math.max(1, f - st.lastFrame);
      predOf.set(m, [st.pos[0] + st.vel[0] * dt, st.pos[1] + st.vel[1] * dt, st.pos[2] + st.vel[2] * dt]);
    }

    // 代价矩阵:观测 i 取自标签 j 预测的距离
    const cost: number[][] = visible.map((obsM) =>
      visible.map((labelM) => {
        const o = positions.subarray((f * M + obsM) * 3, (f * M + obsM) * 3 + 3);
        const p = predOf.get(labelM)!;
        return Math.hypot(o[0] - p[0], o[1] - p[1], o[2] - p[2]);
      }),
    );

    // 成对互换检测:对任意两个可见点,若交叉距离显著小于自身距离且交叉距离够小,
    // 即判定这两个标签发生互换。不依赖全局指派,避免正常运动点的耦合干扰。
    const swappedLabelOfObs = new Map<number, number>();
    const seenPair = new Set<string>();
    for (let i = 0; i < visible.length; i++) {
      for (let j = i + 1; j < visible.length; j++) {
        const selfI = cost[i][i], selfJ = cost[j][j];
        const cross = Math.max(cost[i][j], cost[j][i]);
        if (cross < assignThreshold && cross < 0.4 * Math.min(selfI, selfJ)) {
          const obsI = visible[i], obsJ = visible[j];
          swappedLabelOfObs.set(obsI, obsJ);
          swappedLabelOfObs.set(obsJ, obsI);
          const key = pairKey(obsI, obsJ);
          if (!seenPair.has(key)) {
            seenPair.add(key);
            swaps.push({ frame: f, markerIndexA: obsI, markerIndexB: obsJ, residualBefore: Math.min(selfI, selfJ), residualAfter: cross });
          }
        }
      }
    }

    // 构造标签->观测映射:互换点交叉读取,其余指向自身
    const obsForLabel = new Int32Array(M).fill(-1);
    for (const labelM of visible) {
      obsForLabel[labelM] = swappedLabelOfObs.has(labelM) ? swappedLabelOfObs.get(labelM)! : labelM;
    }
    // 用"指派回正确标签后的位置"更新轨迹状态(把观测放回它本该在的标签)
    const correctedObs: number[] = [];
    for (let labelIdx = 0; labelIdx < visible.length; labelIdx++) {
      const labelM = visible[labelIdx];
      const obsM = obsForLabel[labelM];
      if (obsM >= 0) correctedObs[labelM] = obsM;
    }
    updateStates(states, source, f, visible, visible.map((labelM) => correctedObs[labelM] ?? labelM));

    // 记录本帧标签->观测映射供 apply 阶段使用
    for (let labelIdx = 0; labelIdx < visible.length; labelIdx++) {
      labelOfObs[f * M + visible[labelIdx]] = obsForLabel[visible[labelIdx]] >= 0
        ? obsForLabel[visible[labelIdx]]
        : visible[labelIdx];
    }
  }

  const intervals = buildIntervals(labelOfObs, observed, F, M);
  const clip = intervals.length ? applyRelabels(source, intervals) : source;
  return { clip, swaps: dedupeSwapEvents(swaps) };
}

/** 用(可能被重映射的)观测位置更新各标签的轨迹状态。obsSource[label]=实际读取的观测标记点。 */
function updateStates(
  states: MarkerState[], source: MarkerClip, f: number,
  labels: number[], obsSource: number[],
): void {
  const M = source.markerNames.length;
  labels.forEach((labelM, k) => {
    const obsM = obsSource[k];
    if (!source.observed[f * M + obsM]) return;
    const st = states[labelM];
    const p = source.positions.subarray((f * M + obsM) * 3, (f * M + obsM) * 3 + 3);
    if (st.initialized && f > st.lastFrame) {
      const dt = f - st.lastFrame;
      st.vel[0] = (p[0] - st.pos[0]) / dt;
      st.vel[1] = (p[1] - st.pos[1]) / dt;
      st.vel[2] = (p[2] - st.pos[2]) / dt;
    }
    st.pos.set(p);
    st.lastFrame = f;
    st.initialized = true;
  });
}

/** 由逐帧标签映射提取稳定的两两互换区间(连续帧上相同点对合并)。 */
function buildIntervals(labelOfObs: Int32Array, observed: Uint8Array, F: number, M: number): Interval[] {
  const intervals: Interval[] = [];
  const open = new Map<string, Interval>();
  for (let f = 0; f < F; f++) {
    const seen = new Set<string>();
    for (let label = 0; label < M; label++) {
      if (!observed[f * M + label]) continue;
      const obsM = labelOfObs[f * M + label];
      if (obsM === label || obsM < 0) continue;
      const key = pairKey(label, obsM);
      if (seen.has(key)) continue;
      seen.add(key);
      const cur = open.get(key);
      if (cur && cur.end === f - 1) cur.end = f;
      else {
        const iv = { a: Math.min(label, obsM), b: Math.max(label, obsM), start: f, end: f };
        open.set(key, iv);
        intervals.push(iv);
      }
    }
  }
  return intervals;
}

/** 同一互换只保留首个事件帧,避免逐帧重复。 */
function dedupeSwapEvents(swaps: SwapEvent[]): SwapEvent[] {
  const seen = new Set<string>();
  const out: SwapEvent[] = [];
  for (const s of swaps) {
    const key = pairKey(s.markerIndexA, s.markerIndexB);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out;
}

function pairKey(a: number, b: number): string {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

/** 把互换区间应用到数据:交换区间内两标记点的坐标与观测状态。 */
function applyRelabels(source: MarkerClip, intervals: Interval[]): MarkerClip {
  const M = source.markerNames.length;
  const positions = new Float32Array(source.positions);
  const observed = new Uint8Array(source.observed);
  for (const r of intervals) {
    for (let f = r.start; f <= r.end; f++) {
      const ia = f * M + r.a, ib = f * M + r.b;
      for (let k = 0; k < 3; k++) {
        const tmp = positions[ia * 3 + k];
        positions[ia * 3 + k] = positions[ib * 3 + k];
        positions[ib * 3 + k] = tmp;
      }
      const tm = observed[ia]; observed[ia] = observed[ib]; observed[ib] = tm;
    }
  }
  return { ...source, positions, observed };
}

/** 由相邻帧二阶差分的中位绝对偏差估计测量噪声尺度(米)。 */
function estimateNoise(clip: MarkerClip): number {
  const { positions, frameCount: F, markerNames } = clip;
  const M = markerNames.length;
  const diffs: number[] = [];
  for (let m = 0; m < M; m++) {
    for (let f = 2; f < F; f++) {
      if (!clip.observed[f * M + m] || !clip.observed[(f - 1) * M + m] || !clip.observed[(f - 2) * M + m]) continue;
      for (let k = 0; k < 3; k++) {
        const o = (f * M + m) * 3 + k;
        diffs.push(Math.abs(positions[o] - 2 * positions[o - M * 3] + positions[o - 2 * M * 3]));
      }
    }
  }
  if (!diffs.length) return 0.0012;
  diffs.sort((a, b) => a - b);
  const mad = diffs[Math.floor(diffs.length / 2)];
  return Math.max(0.0005, (mad * 1.4826) / Math.sqrt(6));
}
