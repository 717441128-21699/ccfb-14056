/**
 * 清洗管线总装:原始标记点 -> 标签修复 -> 空档检测 -> 刚体/轨迹补点
 *  -> 骨骼解算 -> 足部锁定 -> 穿模标注。各阶段独立可测,这里统一计时与产物组织。
 */
import type { CleanResult, MarkerClip } from './types.js';
import { buildRig, type BuiltRig, DEFAULT_DIMS } from './rig.js';
import { detectGaps } from './gap.js';
import { detectAndFixSwaps } from './relabel.js';
import { fillGaps, smoothFilled } from './fillGaps.js';
import { solveSkeleton } from './solve.js';
import { lockFeet } from './footlock.js';
import { aggregateIssues, detectPenetration } from './penetration.js';

export interface PipelineOptions {
  /** 对补点帧做轻量时域平滑。 */
  smooth?: boolean;
  /** 显式提供骨架模板(默认 1.75m 标准体型)。 */
  built?: BuiltRig;
  /** 跳过互换修复(调试用)。 */
  skipRelabel?: boolean;
}

export interface PipelineStageTiming {
  relabelMs: number;
  fillMs: number;
  solveMs: number;
  footLockMs: number;
  penetrationMs: number;
  totalMs: number;
}

export interface PipelineOutput extends CleanResult {
  built: BuiltRig;
  timing: PipelineStageTiming;
  issueIntervals: CleanResult['issues'];
  rigidFilled: number;
  priorFilled: number;
  trajectoryFilled: number;
  unresolved: number;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export function runPipeline(raw: MarkerClip, opts: PipelineOptions = {}): PipelineOutput {
  const t0 = now();
  const built = opts.built ?? buildRig(DEFAULT_DIMS);
  const rig = built.rig;

  const ta = now();
  const relabeled = opts.skipRelabel ? { clip: cloneClip(raw), swaps: [] } : detectAndFixSwaps(raw);
  const relabelMs = now() - ta;

  const gaps = detectGaps(relabeled.clip);

  const tb = now();
  const filled = fillGaps(relabeled.clip, rig, gaps);
  // 时域平滑默认关闭:快速摆动帧上 3 点平滑会把正确补点拉偏,需要时由调用方显式开启
  if (opts.smooth === true) smoothFilled(filled.clip, filled.fillMethod);
  const fillMs = now() - tb;

  const tc = now();
  const skeleton = solveSkeleton(filled.clip, rig, built.restJointPositions);
  const solveMs = now() - tc;

  const td = now();
  const { locks } = lockFeet(skeleton, rig, built.restJointPositions);
  const footLockMs = now() - td;

  const te = now();
  const rawIssues = detectPenetration(skeleton, rig);
  const issues = aggregateIssues(rawIssues);
  const penetrationMs = now() - te;

  const totalMs = now() - t0;
  return {
    cleaned: filled.clip,
    gaps,
    swaps: relabeled.swaps,
    fillMethod: filled.fillMethod,
    skeleton,
    footLocks: locks,
    issues,
    elapsedMs: totalMs,
    built,
    timing: { relabelMs, fillMs, solveMs, footLockMs, penetrationMs, totalMs },
    issueIntervals: issues,
    rigidFilled: filled.rigidFilled,
    priorFilled: filled.priorFilled,
    trajectoryFilled: filled.trajectoryFilled,
    unresolved: filled.unresolved,
  };
}

function cloneClip(clip: MarkerClip): MarkerClip {
  return {
    markerNames: [...clip.markerNames],
    positions: new Float32Array(clip.positions),
    observed: new Uint8Array(clip.observed),
    frameCount: clip.frameCount,
    fps: clip.fps,
  };
}
