/**
 * Web UI 主控制器:数据生成/导入、清洗管线调度、时间轴播放与渲染联动。
 */
import { buildRig, DEFAULT_DIMS, DIM_PRESETS, JOINT, type CharacterDimensions, type BuiltRig } from '../core/rig.js';
import { synthesize, markersFromSkeleton, corruptClip } from '../core/motion.js';
import type { MarkerClip } from '../core/types.js';
import { runPipeline, type PipelineOutput } from '../core/pipeline.js';
import { parseCsv, toCsv, serializeMarkerClip } from '../core/io.js';
import { Renderer } from './renderer.js';

interface State {
  raw: MarkerClip | null;
  result: PipelineOutput | null;
  built: BuiltRig;
  frame: number;
  playing: boolean;
}

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const state: State = {
  raw: null,
  result: null,
  built: buildRig(DEFAULT_DIMS),
  frame: 0,
  playing: false,
};

const canvas = $<HTMLCanvasElement>('glcanvas');
const renderer = new Renderer(canvas);

function dimsFromUi(): CharacterDimensions {
  const key = $<HTMLSelectElement>('bodySel').value as keyof typeof DIM_PRESETS;
  return DIM_PRESETS[key] ?? DEFAULT_DIMS;
}

function generateDemo(): void {
  const action = $<HTMLSelectElement>('actionSel').value as 'walk' | 'run' | 'jump';
  const frameCount = Math.max(60, Number($<HTMLInputElement>('frameCount').value));
  const fps = Number($<HTMLInputElement>('fps').value);
  const noise = Number($<HTMLInputElement>('noise').value) / 1000;
  const missing = Number($<HTMLInputElement>('missing').value);
  const swapCount = Number($<HTMLInputElement>('swaps').value);

  const built = buildRig(dimsFromUi());
  state.built = built;
  renderer.setParents(built.rig.joints.map((j) => j.parent ?? -1));

  const skel = synthesize(built, { action, frameCount, fps, seed: 42 });
  const clean = markersFromSkeleton(built, skel);
  state.raw = corruptClip(clean, { noiseSigma: noise, missingRate: missing, swapCount, seed: 7 }).clip;
  state.result = null;
  state.frame = 0;
  $<HTMLInputElement>('scrub').max = String(frameCount - 1);
  setButtons();
  renderFrame();
}

function runCleaning(): void {
  if (!state.raw) return;
  state.result = runPipeline(state.raw, { built: state.built });
  state.frame = 0;
  setButtons();
  renderReport();
  renderFrame();
}

function setButtons(): void {
  const hasData = state.raw !== null;
  const hasResult = state.result !== null;
  $<HTMLButtonElement>('runBtn').disabled = !hasData;
  $<HTMLButtonElement>('exportBtn').disabled = !hasResult;
  $<HTMLButtonElement>('exportJsonBtn').disabled = !hasResult;
}

function renderReport(): void {
  const r = state.result;
  if (!r) { $<HTMLElement>('stats').textContent = '尚未运行。'; return; }
  const el = $<HTMLElement>('stats');
  const lockCount = r.footLocks.length;
  el.innerHTML = [
    `<span class="k">帧数</span>: ${r.cleaned.frameCount}`,
    `<span class="k">标记点</span>: ${r.cleaned.markerNames.length}`,
    `<span class="k">检测空档</span>: ${r.gaps.length} 段`,
    `<span class="k">标签互换</span>: ${r.swaps.length} 对`,
    `<span class="k">刚体补点</span>: <span class="ok">${r.rigidFilled}</span>`,
    `<span class="k">先验补点</span>: <span class="warn">${r.priorFilled}</span>`,
    `<span class="k">轨迹补点</span>: <span class="warn">${r.trajectoryFilled}</span>`,
    `<span class="k">足部锁定</span>: ${lockCount} 段`,
    `<span class="k">穿模问题</span>: ${r.issues.length} 处`,
    `<span class="k">总耗时</span>: ${r.elapsedMs.toFixed(0)} ms`,
    `<span class="k">阶段</span>: 标签 ${r.timing.relabelMs.toFixed(0)} / 补点 ${r.timing.fillMs.toFixed(0)} / 解算 ${r.timing.solveMs.toFixed(0)} ms`,
  ].join('<br>');
}

function currentIssueJoints(frame: number): Set<number> {
  const out = new Set<number>();
  if (!state.result) return out;
  for (const iss of state.result.issues) if (iss.frame === frame) out.add(iss.jointA);
  return out;
}
function currentLockJoints(frame: number): Set<number> {
  const out = new Set<number>();
  if (!state.result) return out;
  for (const lk of state.result.footLocks) {
    if (frame >= lk.startFrame && frame <= lk.endFrame) {
      out.add(lk.side === 'left' ? JOINT.LFoot : JOINT.RFoot);
    }
  }
  return out;
}

function renderFrame(): void {
  const f = state.frame;
  const r = state.result;
  const markers = r ? r.cleaned : state.raw;
  const skel = r?.skeleton ?? null;
  const fill = r?.fillMethod ?? null;
  renderer.render(
    f,
    markers,
    skel,
    fill,
    currentIssueJoints(f),
    currentLockJoints(f),
    {
      showMarkers: $<HTMLInputElement>('showMarkers').checked,
      showSkeleton: $<HTMLInputElement>('showSkeleton').checked,
      showIssues: $<HTMLInputElement>('showIssues').checked,
      showLocks: $<HTMLInputElement>('showLocks').checked,
      colorFill: $<HTMLInputElement>('colorFill').checked,
    },
  );
  $<HTMLInputElement>('scrub').value = String(f);
  const total = markers?.frameCount ?? 0;
  $<HTMLElement>('frameLabel').textContent = `${f} / ${total - 1}`;
}

// 播放循环
let lastTick = 0;
function tick(now: number): void {
  if (state.playing && state.raw) {
    if (now - lastTick > 1000 / (Number($<HTMLInputElement>('fps').value) || 120)) {
      const max = state.raw.frameCount - 1;
      state.frame = state.frame >= max ? 0 : state.frame + 1;
      renderFrame();
      lastTick = now;
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// 事件绑定
$<HTMLButtonElement>('demoBtn').addEventListener('click', generateDemo);
$<HTMLButtonElement>('runBtn').addEventListener('click', runCleaning);
$<HTMLButtonElement>('playBtn').addEventListener('click', () => {
  state.playing = !state.playing;
  $<HTMLButtonElement>('playBtn').textContent = state.playing ? '暂停' : '播放';
});
$<HTMLInputElement>('scrub').addEventListener('input', (e) => {
  state.frame = Number((e.target as HTMLInputElement).value);
  renderFrame();
});
for (const id of ['showMarkers', 'showSkeleton', 'showIssues', 'showLocks', 'colorFill', 'bodySel']) {
  $(id).addEventListener('change', () => {
    if (id === 'bodySel' && state.raw) generateDemo();
    renderFrame();
  });
}

$<HTMLInputElement>('fileInput').addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const text = await file.text();
  const fps = Number($<HTMLInputElement>('fps').value);
  const clip = parseCsv(text, fps);
  const built = buildRig(dimsFromUi());
  state.built = built;
  renderer.setParents(built.rig.joints.map((j) => j.parent ?? -1));
  state.raw = clip;
  state.result = null;
  state.frame = 0;
  $<HTMLInputElement>('scrub').max = String(clip.frameCount - 1);
  setButtons();
  renderFrame();
});

$<HTMLButtonElement>('exportBtn').addEventListener('click', () => {
  if (!state.result) return;
  download('cleaned.csv', toCsv(state.result.cleaned));
});
$<HTMLButtonElement>('exportJsonBtn').addEventListener('click', () => {
  if (!state.result) return;
  download('cleaned.json', JSON.stringify(serializeMarkerClip(state.result.cleaned)));
});

function download(name: string, content: string): void {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

// 图例
$<HTMLElement>('legend').innerHTML = [
  ['#d9e0eb', '原始观测'],
  ['#4c9aff', '刚体补点'],
  ['#b07dff', '先验补点'],
  ['#f0b429', '轨迹补点'],
  ['#ff5d5d', '穿模标注'],
  ['#38c275', '足部锁定'],
].map(([c, t]) => `<div><span class="dot" style="background:${c}"></span>${t}</div>`).join('');

// 启动即生成一份演示数据
generateDemo();
