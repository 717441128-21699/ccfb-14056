/**
 * 核心管线自动化测试(零依赖断言,Node 下 tsx 直接运行)。
 * 覆盖:Kabsch 精度、合成->污染->清洗全链路、互换识别、补点 2mm 门槛、
 * 足部锁定防滑、穿模检测、重定向骨长、CSV/JSON 往返。
 */
import { kabschRotation, rigidBodyFit } from '../src/core/math.js';
import { buildRig, DEFAULT_DIMS, DIM_PRESETS, JOINT } from '../src/core/rig.js';
import { corruptClip, fillErrorStats, markersFromSkeleton, synthesize } from '../src/core/motion.js';
import { runPipeline } from '../src/core/pipeline.js';
import { FillMethod } from '../src/core/types.js';
import { detectGaps } from '../src/core/gap.js';
import { lockFeet } from '../src/core/footlock.js';
import { detectPenetration } from '../src/core/penetration.js';
import { retarget, contactMasksFromLocks } from '../src/core/retarget.js';
import { parseCsv, serializeMarkerClip, deserializeMarkerClip, toCsv } from '../src/core/io.js';

let passed = 0;
let failed = 0;
const failures: string[] = [];

function assert(cond: boolean, msg: string): void {
  if (cond) { passed++; }
  else { failed++; failures.push(msg); console.error(`  ✗ ${msg}`); }
}

function approx(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}

// ---------- 1. Kabsch:已知旋转平移的配准 ----------
function testKabsch(): void {
  const rng = mulberry(7);
  const local = new Float32Array(4 * 3);
  for (let i = 0; i < local.length; i++) local[i] = rng() * 0.2 - 0.1;
  const angle = 0.7;
  const c = Math.cos(angle), s = Math.sin(angle);
  const Rtrue = new Float32Array([
    c, -s, 0,
    s, c, 0,
    0, 0, 1,
  ]);
  const ttrue = new Float32Array([1.2, -0.4, 0.8]);
  const world = new Float32Array(4 * 3);
  for (let i = 0; i < 4; i++) {
    for (let r = 0; r < 3; r++) {
      // world = Rtrue * local + t(R 行主序:第 r 行在 R[3r],R[3r+1],R[3r+2])
      world[i * 3 + r] =
        Rtrue[r * 3] * local[i * 3] + Rtrue[r * 3 + 1] * local[i * 3 + 1] + Rtrue[r * 3 + 2] * local[i * 3 + 2] + ttrue[r];
    }
  }
  const mask = new Uint8Array(4).fill(1);
  const R = new Float32Array(9), t = new Float32Array(3);
  const rms = rigidBodyFit(local, world, mask, 4, R, 0, t, 0);
  assert(rms !== null && rms < 1e-6, 'Kabsch 无噪声配准 RMS 应接近 0');
  let rErr = 0;
  for (let i = 0; i < 9; i++) rErr = Math.max(rErr, Math.abs(R[i] - Rtrue[i]));
  assert(rErr < 1e-5, 'Kabsch 旋转矩阵与真值一致');
  assert(approx(t[0], ttrue[0], 1e-5) && approx(t[1], ttrue[1], 1e-5) && approx(t[2], ttrue[2], 1e-5), 'Kabsch 平移与真值一致');

  // 共线点应判定退化
  const line = new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]);
  const lineQ = new Float32Array([0, 0, 0, 0, 1, 0, 0, 2, 0]);
  const ok = kabschRotation(line, lineQ, 3, new Float32Array(9), 0);
  assert(ok === false, '共线点集应判定为退化');
}

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- 2. 合成数据:FK 往返一致 ----------
function testFkRoundTrip(): void {
  const built = buildRig(DEFAULT_DIMS);
  const skel = synthesize(built, { action: 'walk', frameCount: 60, fps: 120, seed: 1 });
  const markers = markersFromSkeleton(built, skel);
  assert(markers.frameCount === 60 && markers.markerNames.length === built.markerNames.length, 'FK 标记点尺寸正确');
  assert(markers.observed.every((v) => v === 1), '干净数据全部观测到');
  // 所有标记点落在合理人体空间内
  let finite = true;
  for (const v of markers.positions) if (!Number.isFinite(v)) finite = false;
  assert(finite, 'FK 坐标全部有限');
}

// ---------- 3. 空档检测与补点 2mm 门槛 ----------
function testGapFill(): void {
  const built = buildRig(DEFAULT_DIMS);
  const skel = synthesize(built, { action: 'walk', frameCount: 360, fps: 120, seed: 2 });
  const clean = markersFromSkeleton(built, skel);
  // 0.8mm 为高精度光学动捕典型噪声;刚体补点在该噪声下应进入 2mm 门槛
  const { clip, truth } = corruptClip(clean, { noiseSigma: 0.0008, missingRate: 0.12, swapCount: 0, seed: 11 });
  const gaps = detectGaps(clip);
  const missingCells = clip.observed.length - clip.observed.reduce((a, b) => a + b, 0);
  const coveredCells = gaps.reduce((a, g) => a + g.length, 0);
  assert(coveredCells === missingCells, `空档覆盖全部丢点单元(检测 ${coveredCells} / 实际 ${missingCells})`);

  const out = runPipeline(clip, { built, skipRelabel: true });
  assert(out.unresolved === 0, '所有丢点均被补全');
  assert(out.rigidFilled > 0, '存在刚体补点');

  // 刚体补点精度(核心验收):仅统计由刚体关系恢复的丢点
  const M = clip.markerNames.length;
  const rigidErrs: number[] = [];
  for (const g of truth.gaps) {
    for (let f = g.start; f <= g.end; f++) {
      if (out.fillMethod[f * M + g.marker] !== FillMethod.RigidBody) continue;
      const io = (f * M + g.marker) * 3;
      rigidErrs.push(Math.hypot(
        out.cleaned.positions[io] - truth.cleanPositions[io],
        out.cleaned.positions[io + 1] - truth.cleanPositions[io + 1],
        out.cleaned.positions[io + 2] - truth.cleanPositions[io + 2],
      ) * 1000);
    }
  }
  const rigidRms = Math.sqrt(rigidErrs.reduce((s, e) => s + e * e, 0) / (rigidErrs.length || 1));
  const rigidSorted = [...rigidErrs].sort((a, b) => a - b);
  const rigidP95 = rigidSorted[Math.floor(rigidSorted.length * 0.95)] ?? 0;
  assert(rigidRms < 2, `刚体补点 RMS < 2mm(实测 ${rigidRms.toFixed(3)}mm, n=${rigidErrs.length})`);
  assert(rigidP95 < 3.5, `刚体补点 P95 < 3.5mm(实测 ${rigidP95.toFixed(3)}mm)`);

  // 整体(含长档轨迹/先验兜底)误差:长档受物理信息限制,放宽到 50mm(厘米级)
  const stats = fillErrorStats(out.cleaned, truth);
  assert(stats.rmsMm < 50, `整体补点 RMS < 50mm(实测 ${stats.rmsMm.toFixed(2)}mm)`);
}

// ---------- 4. 标签互换识别 ----------
function testSwapDetection(): void {
  const built = buildRig(DEFAULT_DIMS);
  const skel = synthesize(built, { action: 'walk', frameCount: 480, fps: 120, seed: 3 });
  const clean = markersFromSkeleton(built, skel);
  const { clip, truth } = corruptClip(clean, { noiseSigma: 0.001, missingRate: 0.03, swapCount: 6, seed: 23 });
  assert(truth.swaps.length === 6, '污染器注入 6 个互换');
  const out = runPipeline(clip, { built });
  // 至少识别出 5/6,且每次识别涉及的标记点对与真值重合
  const matched = truth.swaps.filter((ts) =>
    out.swaps.some((ds) =>
      Math.abs(ds.frame - ts.start) <= 12 &&
      ((ds.markerIndexA === ts.a && ds.markerIndexB === ts.b) ||
        (ds.markerIndexA === ts.b && ds.markerIndexB === ts.a)),
    ),
  );
  assert(matched.length >= 5, `互换识别召回 ≥ 5/6(实测 ${matched.length}/6)`);
  // 在真实互换区间内,修复后两标记点位置应贴合干净真值(厘米级,证明标签确实换回)
  const M = clip.markerNames.length;
  const swapErrs: number[] = [];
  for (const ts of truth.swaps) {
    for (let f = ts.start; f <= Math.min(ts.end, ts.start + 3); f++) {
      for (const m of [ts.a, ts.b]) {
        if (!out.cleaned.observed[f * M + m]) continue;
        const io = (f * M + m) * 3;
        swapErrs.push(Math.hypot(
          out.cleaned.positions[io] - truth.cleanPositions[io],
          out.cleaned.positions[io + 1] - truth.cleanPositions[io + 1],
          out.cleaned.positions[io + 2] - truth.cleanPositions[io + 2],
        ) * 1000);
      }
    }
  }
  const swapRmsMm = Math.sqrt(swapErrs.reduce((s, e) => s + e * e, 0) / Math.max(1, swapErrs.length));
  assert(swapRmsMm < 15, `互换区间修复后贴合真值(实测 ${swapRmsMm.toFixed(1)}mm, n=${swapErrs.length})`);
}

// ---------- 5. 足部锁定:接触帧不打滑 ----------
function testFootLock(): void {
  const built = buildRig(DEFAULT_DIMS);
  const skel = synthesize(built, { action: 'walk', frameCount: 480, fps: 120, seed: 4 });
  const { locks } = lockFeet(skel, built.rig, built.restJointPositions);
  assert(locks.length >= 2, `走姿检测到多次足部接触(实测 ${locks.length} 段)`);
  let slipMax = 0;
  for (const lk of locks) {
    const j = lk.side === 'left' ? JOINT.LFoot : JOINT.RFoot;
    for (let f = lk.startFrame + 1; f <= lk.endFrame; f++) {
      const a = ((f - 1) * 21 + j) * 3;
      const b = (f * 21 + j) * 3;
      const slip = Math.hypot(skel.jointPositions[b] - skel.jointPositions[a], skel.jointPositions[b + 2] - skel.jointPositions[a + 2]);
      slipMax = Math.max(slipMax, slip);
    }
  }
  assert(slipMax < 1e-6, `锁定段水平位移为 0(实测最大 ${slipMax.toExponential(2)}m)`);
}

// ---------- 6. 穿模检测:构造手穿大腿场景 ----------
function testPenetration(): void {
  const built = buildRig(DEFAULT_DIMS);
  const skel = synthesize(built, { action: 'walk', frameCount: 60, fps: 120, seed: 5 });
  const normal = detectPenetration(skel, built.rig);
  // 把右手强行放到左大腿位置若干帧
  for (let f = 10; f < 20; f++) {
    const ho = (f * 21 + JOINT.RHand) * 3;
    const to = (f * 21 + JOINT.LUpLeg) * 3;
    skel.jointPositions[ho] = skel.jointPositions[to];
    skel.jointPositions[ho + 1] = skel.jointPositions[to + 1];
    skel.jointPositions[ho + 2] = skel.jointPositions[to + 2];
  }
  const hit = detectPenetration(skel, built.rig);
  const inWindow = hit.filter((i) => i.frame >= 10 && i.frame < 20);
  assert(inWindow.length > 0, '手穿大腿帧被标出');
  assert(inWindow.every((i) => i.depth > 0.005), '标出的穿透深度超过 5mm 阈值');
  console.log(`    (正常走姿穿模帧 ${normal.length},注入后总问题帧 ${hit.length})`);
}

// ---------- 7. 重定向:目标身材骨长正确,接触点保持 ----------
function testRetarget(): void {
  const built = buildRig(DEFAULT_DIMS);
  const skel = synthesize(built, { action: 'walk', frameCount: 240, fps: 120, seed: 6 });
  const { locks } = lockFeet(skel, built.rig, built.restJointPositions);
  const masks = contactMasksFromLocks(locks, skel.frameCount);
  const { skel: out, targetBuilt } = retarget(skel, { sourceBuilt: built, targetDims: DIM_PRESETS.tall, contactFrames: masks });
  assert(out.jointPositions.length === skel.jointPositions.length, '重定向输出帧数/关节数不变');
  // 骨长匹配目标静帧
  let maxBoneErr = 0;
  for (let j = 1; j < 21; j++) {
    const p = targetBuilt.rig.joints[j].parent!;
    const targetLen = Math.hypot(
      targetBuilt.restJointPositions[j * 3] - targetBuilt.restJointPositions[p * 3],
      targetBuilt.restJointPositions[j * 3 + 1] - targetBuilt.restJointPositions[p * 3 + 1],
      targetBuilt.restJointPositions[j * 3 + 2] - targetBuilt.restJointPositions[p * 3 + 2],
    );
    const f = 100;
    const got = Math.hypot(
      out.jointPositions[(f * 21 + j) * 3] - out.jointPositions[(f * 21 + p) * 3],
      out.jointPositions[(f * 21 + j) * 3 + 1] - out.jointPositions[(f * 21 + p) * 3 + 1],
      out.jointPositions[(f * 21 + j) * 3 + 2] - out.jointPositions[(f * 21 + p) * 3 + 2],
    );
    maxBoneErr = Math.max(maxBoneErr, Math.abs(got - targetLen));
  }
  assert(maxBoneErr < 0.002, `重定向后骨长与目标身材一致(最大误差 ${(maxBoneErr * 1000).toFixed(2)}mm)`);
  // 接触帧脚踝水平坐标近似保持(按根缩放;可达域钳制允许厘米级偏差)
  const lock = locks[0];
  if (lock) {
    const j = lock.side === 'left' ? JOINT.LFoot : JOINT.RFoot;
    const f = lock.startFrame + 1;
    const scale = targetBuilt.restJointPositions[1] / built.restJointPositions[1];
    const dx = out.jointPositions[(f * 21 + j) * 3] - skel.jointPositions[(f * 21 + j) * 3] * scale;
    const dz = out.jointPositions[(f * 21 + j) * 3 + 2] - skel.jointPositions[(f * 21 + j) * 3 + 2] * scale;
    assert(Math.hypot(dx, dz) < 0.02, `重定向接触脚水平位置保持(偏差 ${(Math.hypot(dx, dz) * 1000).toFixed(1)}mm)`);
  }
}

// ---------- 8. I/O 往返 ----------
function testIo(): void {
  const built = buildRig(DEFAULT_DIMS);
  const skel = synthesize(built, { action: 'run', frameCount: 30, fps: 120, seed: 8 });
  const clean = markersFromSkeleton(built, skel);
  const { clip } = corruptClip(clean, { noiseSigma: 0.0008, missingRate: 0.1, swapCount: 0, seed: 9 });

  const json = serializeMarkerClip(clip);
  const back = deserializeMarkerClip(json);
  assert(back.frameCount === clip.frameCount && back.markerNames.length === clip.markerNames.length, 'JSON 往返尺寸一致');
  let err = 0;
  for (let i = 0; i < clip.observed.length; i++) {
    if (clip.observed[i]) err = Math.max(err, Math.abs(back.positions[i * 3] - clip.positions[i * 3]));
  }
  assert(err < 1e-6, 'JSON 往返坐标一致');

  const csv = toCsv(clip);
  const parsed = parseCsv(csv, clip.fps);
  assert(parsed.frameCount === clip.frameCount, 'CSV 往返帧数一致');
  let missingMatch = 0;
  for (let i = 0; i < clip.observed.length; i++) if (!clip.observed[i] && !parsed.observed[i]) missingMatch++;
  const totalMissing = clip.observed.length - clip.observed.reduce((a, b) => a + b, 0);
  assert(missingMatch === totalMissing, 'CSV 丢点空单元格往返一致');
}

// ---------- 9. 跑/跳动作全链路不崩且刚体补点误差受控 ----------
function testAllActions(): void {
  for (const action of ['run', 'jump'] as const) {
    const built = buildRig(DEFAULT_DIMS);
    const skel = synthesize(built, { action, frameCount: 300, fps: 120, seed: 13 });
    const clean = markersFromSkeleton(built, skel);
    const { clip, truth } = corruptClip(clean, { noiseSigma: 0.0008, missingRate: 0.1, swapCount: 3, seed: 14 });
    const out = runPipeline(clip, { built });
    assert(out.unresolved === 0, `${action}:无未补全丢点`);

    const M = clip.markerNames.length;
    const rigidErrs: number[] = [];
    for (const g of truth.gaps) for (let f = g.start; f <= g.end; f++) {
      if (out.fillMethod[f * M + g.marker] !== FillMethod.RigidBody) continue;
      const io = (f * M + g.marker) * 3;
      rigidErrs.push(Math.hypot(
        out.cleaned.positions[io] - truth.cleanPositions[io],
        out.cleaned.positions[io + 1] - truth.cleanPositions[io + 1],
        out.cleaned.positions[io + 2] - truth.cleanPositions[io + 2],
      ) * 1000);
    }
    const rigidRms = Math.sqrt(rigidErrs.reduce((s, e) => s + e * e, 0) / (rigidErrs.length || 1));
    assert(rigidRms < 2.5, `${action}:刚体补点 RMS < 2.5mm(实测 ${rigidRms.toFixed(2)}mm, n=${rigidErrs.length})`);

    const stats = fillErrorStats(out.cleaned, truth);
    assert(stats.rmsMm < 120, `${action}:整体补点(含长档兜底)RMS < 120mm(实测 ${stats.rmsMm.toFixed(1)}mm)`);
  }
}

async function main(): Promise<void> {
  const tests: Array<[string, () => void]> = [
    ['Kabsch 配准', testKabsch],
    ['FK 往返', testFkRoundTrip],
    ['空档补点 2mm', testGapFill],
    ['互换识别', testSwapDetection],
    ['足部锁定', testFootLock],
    ['穿模检测', testPenetration],
    ['身材重定向', testRetarget],
    ['I/O 往返', testIo],
    ['跑/跳全链路', testAllActions],
  ];
  for (const [name, fn] of tests) {
    console.log(`• ${name}`);
    fn();
  }
  console.log(`\n结果:${passed} 通过 / ${failed} 失败`);
  if (failed) {
    console.error('失败项:', failures);
    process.exit(1);
  }
}

main();
