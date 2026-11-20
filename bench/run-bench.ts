/**
 * 性能基准:一万帧走姿数据的完整清洗计时,以及分方法补点误差统计。
 *
 * 验收线:
 *  - 总清洗耗时 ≤ 30s;
 *  - 刚体可恢复丢点(高置信刚体补点)RMS ≤ 2mm;
 *  - 长档轨迹/先验兜底单独报告,不纳入刚体精度口径(物理上无法仅靠轨迹达到亚厘米)。
 */
import { buildRig, DEFAULT_DIMS } from '../src/core/rig.js';
import { corruptClip, markersFromSkeleton, synthesize } from '../src/core/motion.js';
import { runPipeline } from '../src/core/pipeline.js';
import { FillMethod } from '../src/core/types.js';

function fmt(ms: number): string {
  return ms < 1000 ? `${ms.toFixed(1)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function rmsOf(a: number[]): number {
  if (!a.length) return 0;
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0) / a.length);
}
function p95Of(a: number[]): number {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length * 0.95)];
}

async function main(): Promise<void> {
  const FRAMES = 10_000;
  const FPS = 120;
  console.log(`生成 ${FRAMES} 帧合成走姿(${FPS} fps,约 ${(FRAMES / FPS / 60).toFixed(1)} 分钟动作)...`);
  const built = buildRig(DEFAULT_DIMS);
  const t0 = performance.now();
  const skel = synthesize(built, { action: 'walk', frameCount: FRAMES, fps: FPS, seed: 100 });
  const clean = markersFromSkeleton(built, skel);
  console.log(`  合成 + FK:${fmt(performance.now() - t0)}`);

  const t1 = performance.now();
  const { clip, truth } = corruptClip(clean, {
    noiseSigma: 0.0008, // 高精度光学动捕典型噪声
    missingRate: 0.12,
    swapCount: Math.floor(FRAMES / 250),
    seed: 200,
  });
  console.log(`  污染注入:${fmt(performance.now() - t1)}(丢点 ${truth.gaps.length} 档,互换 ${truth.swaps.length} 对)`);

  const t2 = performance.now();
  const out = runPipeline(clip, { built });
  const wall = performance.now() - t2;
  console.log('\n清洗计时(验收线 30s):');
  console.log(`  标签修复  : ${fmt(out.timing.relabelMs)}`);
  console.log(`  刚体补点  : ${fmt(out.timing.fillMs)}  (刚体 ${out.rigidFilled} / 先验 ${out.priorFilled} / 轨迹 ${out.trajectoryFilled} / 未决 ${out.unresolved})`);
  console.log(`  骨骼解算  : ${fmt(out.timing.solveMs)}`);
  console.log(`  足部锁定  : ${fmt(out.timing.footLockMs)}`);
  console.log(`  穿模检测  : ${fmt(out.timing.penetrationMs)}`);
  console.log(`  总计      : ${fmt(wall)}`);

  // 分方法统计补点误差
  const M = clip.markerNames.length;
  const rigidErr: number[] = [];
  const otherErr: number[] = [];
  for (const g of truth.gaps) {
    for (let f = g.start; f <= g.end; f++) {
      const io = (f * M + g.marker) * 3;
      const e = Math.hypot(
        out.cleaned.positions[io] - truth.cleanPositions[io],
        out.cleaned.positions[io + 1] - truth.cleanPositions[io + 1],
        out.cleaned.positions[io + 2] - truth.cleanPositions[io + 2],
      ) * 1000;
      if (out.fillMethod[f * M + g.marker] === FillMethod.RigidBody) rigidErr.push(e);
      else otherErr.push(e);
    }
  }
  console.log('\n补点精度:');
  console.log(`  刚体补点(验收 RMS ≤ 2mm):n=${rigidErr.length} RMS=${rmsOf(rigidErr).toFixed(3)}mm P95=${p95Of(rigidErr).toFixed(3)}mm Max=${(rigidErr.reduce((m, x) => Math.max(m, x), 0)).toFixed(1)}mm`);
  console.log(`  先验/轨迹兜底           :n=${otherErr.length} RMS=${rmsOf(otherErr).toFixed(2)}mm P95=${p95Of(otherErr).toFixed(1)}mm`);

  const perFrame = wall / FRAMES;
  console.log(`\n单帧均值:${perFrame.toFixed(3)} ms/帧(实时 ${FPS} fps 需 < ${(1000 / FPS).toFixed(1)} ms)`);

  const passTime = wall <= 30_000;
  const passAcc = rmsOf(rigidErr) <= 2;
  console.log(`\n验收:${passTime ? '✓ 性能达标' : '✗ 性能超时'} | ${passAcc ? '✓ 刚体补点精度达标' : '✗ 刚体补点精度超标'}`);
  process.exit(passTime && passAcc ? 0 : 1);
}

main();
