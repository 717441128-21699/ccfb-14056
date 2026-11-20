import { cleanClip, DEFAULT_CLEAN_OPTIONS } from './core/cleaner';
import type { CleanReport } from './core/types';
import { defaultSkeleton } from './core/skeleton';
import { buildSyntheticClip, type SyntheticData } from './core/synthetic';
import { solveSkeleton, DEFAULT_SOLVE_OPTIONS, type SolveOptions } from './core/solver';
import { retarget, BODY_PROFILES, type BodyProfile, type RetargetedSkeleton } from './core/retarget';
import { parseMarkerCsv, parseMarkerJson, toMarkerCsv, toMarkerJson, toBvh, downloadText } from './io/formats';
import type { CorruptionSpec, MarkerClip, MotionKind, SolvedSkeleton } from './core/types';
import { OrbitControls } from './render/controls';
import { SceneRenderer } from './render/renderer';

interface UiState {
  raw: MarkerClip | null;
  cleaned: MarkerClip | null;
  report: CleanReport | null;
  solved: SolvedSkeleton | null;
  retargeted: RetargetedSkeleton | null;
  synthetic: SyntheticData | null;
  frame: number;
  playing: boolean;
  view: 'raw' | 'cleaned';
  bodyProfile: BodyProfile;
}

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

class App {
  private state: UiState = {
    raw: null,
    cleaned: null,
    report: null,
    solved: null,
    retargeted: null,
    synthetic: null,
    frame: 0,
    playing: false,
    view: 'cleaned',
    bodyProfile: BODY_PROFILES[0],
  };
  private renderer!: SceneRenderer;
  private controls!: OrbitControls;
  private raf = 0;
  private lastTime = 0;
  private worker: Worker | null = null;

  init(): void {
    const canvas = $<HTMLCanvasElement>('glCanvas');
    this.renderer = new SceneRenderer(canvas);
    this.controls = new OrbitControls(canvas, this.renderer.camera, () => this.render());
    this.bindUi();
    this.loop(0);
    this.generate();
  }

  private bindUi(): void {
    $('btnGenerate').addEventListener('click', () => this.generate());
    $('btnClean').addEventListener('click', () => this.clean());
    $('btnSolve').addEventListener('click', () => this.solve());
    $('btnImport').addEventListener('click', () => $<HTMLInputElement>('fileInput').click());
    $('fileInput').addEventListener('change', (e) => this.importFile(e));
    $('btnExportCsv').addEventListener('click', () => this.exportCsv());
    $('btnExportJson').addEventListener('click', () => this.exportJson());
    $('btnExportBvh').addEventListener('click', () => this.exportBvh());

    $('viewRaw').addEventListener('change', () => { this.state.view = 'raw'; this.render(); });
    $('viewCleaned').addEventListener('change', () => { this.state.view = 'cleaned'; this.render(); });
    for (const p of BODY_PROFILES) {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.label;
      $<HTMLSelectElement>('bodySelect').appendChild(opt);
    }
    $('bodySelect').addEventListener('change', (e) => {
      const id = (e.target as HTMLSelectElement).value;
      this.state.bodyProfile = BODY_PROFILES.find((p) => p.id === id)!;
      this.retarget();
    });

    const scrub = $<HTMLInputElement>('scrubber');
    scrub.addEventListener('input', () => {
      this.state.frame = Number(scrub.value);
      this.updateHud();
      this.render();
    });
    $('btnPlay').addEventListener('click', () => {
      this.state.playing = !this.state.playing;
      $('btnPlay').textContent = this.state.playing ? '暂停' : '播放';
    });
  }

  private corruptionSpec(): CorruptionSpec {
    return {
      noiseStd: Number($<HTMLInputElement>('noiseStd').value) / 1000,
      gapRate: Number($<HTMLInputElement>('gapRate').value),
      maxGapLength: Number($<HTMLInputElement>('maxGap').value),
      swapCount: Number($<HTMLInputElement>('swapCount').value),
      maxSwapLength: Number($<HTMLInputElement>('maxSwap').value),
      seed: Number($<HTMLInputElement>('seed').value),
    };
  }

  private generate(): void {
    const kind = $<HTMLSelectElement>('motionKind').value as MotionKind;
    const frames = Number($<HTMLInputElement>('frameCount').value);
    const fps = Number($<HTMLInputElement>('fps').value);
    const synthetic = buildSyntheticClip(kind, frames, fps, this.corruptionSpec());
    this.state.synthetic = synthetic;
    this.state.raw = synthetic.clip;
    this.state.cleaned = null;
    this.state.report = null;
    this.state.solved = null;
    this.state.retargeted = null;
    this.state.frame = 0;
    this.setScrubber(frames);
    this.setStatus(`已生成 ${frames} 帧 · ${synthetic.clip.markerNames.length} 标记 · ${kind}`);
    this.updateReport(null);
    this.render();
  }

  private async clean(): Promise<void> {
    if (!this.state.raw) return;
    this.setStatus('清洗中…');
    const useWorker = typeof Worker !== 'undefined' && this.state.raw.frameCount > 2000;
    const options = { ...DEFAULT_CLEAN_OPTIONS };
    if (useWorker) {
      const result = await this.runWorker({ kind: 'clean', clip: this.state.raw, options });
      this.state.cleaned = result.cleaned as MarkerClip;
      this.state.report = result.report as CleanReport;
    } else {
      const r = cleanClip(this.state.raw, options);
      this.state.cleaned = r.cleaned;
      this.state.report = r.report;
    }
    this.state.solved = null;
    this.state.retargeted = null;
    this.updateReport(this.state.report);
    this.setStatus(`清洗完成 · ${this.state.report!.gaps.length} 个间隙 · ${this.state.report!.swaps.length} 段互换`);
    this.render();
  }

  private async solve(): Promise<void> {
    if (!this.state.cleaned) {
      this.setStatus('请先清洗数据');
      return;
    }
    this.setStatus('骨骼解算中…');
    const options: SolveOptions = { ...DEFAULT_SOLVE_OPTIONS };
    if (typeof Worker !== 'undefined' && this.state.cleaned.frameCount > 2000) {
      const result = await this.runWorker({ kind: 'solve', clip: this.state.cleaned, options });
      this.state.solved = result.solved as SolvedSkeleton;
    } else {
      this.state.solved = solveSkeleton(this.state.cleaned, defaultSkeleton(), options);
    }
    this.retarget();
    this.populateEvents();
    this.setStatus(
      `解算完成 · ${this.state.solved.contacts.length} 个着地区间 · ${this.state.solved.penetrations.length} 个穿模事件`,
    );
    this.render();
  }

  private populateEvents(): void {
    const panel = $('eventPanel');
    if (!this.state.solved) return;
    const { contacts, penetrations } = this.state.solved;
    const rows: string[] = [];
    for (const c of contacts) {
      rows.push(
        `<div class="event-row ok"><span>着地 ${c.side === 'L' ? '左脚' : '右脚'}</span><span>帧 ${c.start}–${c.end}</span></div>`,
      );
    }
    const byPair = new Map<string, number>();
    for (const p of penetrations) {
      const key = `${p.boneA} ↔ ${p.boneB}`;
      byPair.set(key, (byPair.get(key) ?? 0) + 1);
    }
    for (const [pair, count] of byPair) {
      rows.push(
        `<div class="event-row danger"><span>穿模 ${pair}</span><span>${count} 帧</span></div>`,
      );
    }
    panel.innerHTML = rows.length ? rows.join('') : '<p class="muted">无异常事件</p>';
  }

  private retarget(): void {
    if (!this.state.solved) return;
    this.state.retargeted = retarget(this.state.solved, this.state.bodyProfile);
    this.render();
  }

  private runWorker(message: unknown): Promise<Record<string, unknown>> {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./render/pipeline.worker.ts', import.meta.url), { type: 'module' });
      this.worker = worker;
      worker.onmessage = (ev: MessageEvent<Record<string, unknown>>) => {
        if (ev.data.kind === 'progress') {
          this.setStatus(`处理中 ${Math.round((ev.data.fraction as number) * 100)}%`);
        } else if (ev.data.kind === 'error') {
          worker.terminate();
          reject(new Error(ev.data.message as string));
        } else {
          worker.terminate();
          resolve(ev.data);
        }
      };
      worker.onerror = (e) => {
        worker.terminate();
        reject(e);
      };
      worker.postMessage(message);
    });
  }

  private async importFile(e: Event): Promise<void> {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const text = await file.text();
    const fps = Number($<HTMLInputElement>('fps').value);
    try {
      const clip = file.name.endsWith('.json') ? parseMarkerJson(text) : parseMarkerCsv(text, fps);
      this.state.raw = clip;
      this.state.cleaned = null;
      this.state.solved = null;
      this.state.retargeted = null;
      this.state.frame = 0;
      this.setScrubber(clip.frameCount);
      this.setStatus(`已导入 ${file.name} · ${clip.frameCount} 帧 · ${clip.markerNames.length} 标记`);
      this.render();
    } catch (err) {
      this.setStatus(`导入失败: ${(err as Error).message}`);
    }
    input.value = '';
  }

  private exportCsv(): void {
    const clip = this.state.cleaned ?? this.state.raw;
    if (!clip) return;
    downloadText('cleaned-markers.csv', toMarkerCsv(clip), 'text/csv');
  }

  private exportJson(): void {
    const clip = this.state.cleaned ?? this.state.raw;
    if (!clip) return;
    downloadText('cleaned-markers.json', toMarkerJson(clip), 'application/json');
  }

  private exportBvh(): void {
    if (!this.state.solved) {
      this.setStatus('请先解算骨骼再导出 BVH');
      return;
    }
    downloadText('skeleton.bvh', toBvh(this.state.solved), 'text/plain');
  }

  private updateReport(report: CleanReport | null): void {
    const el = $('reportPanel');
    if (!report) {
      el.innerHTML = '<p class="muted">运行清洗后显示报告</p>';
      return;
    }
    const gapByMethod = report.gaps.reduce<Record<string, number>>((acc: Record<string, number>, g) => {
      acc[g.method] = (acc[g.method] ?? 0) + 1;
      return acc;
    }, {});
    el.innerHTML = `
      <div class="stat"><span>帧 / 标记</span><b>${report.frameCount} / ${report.markerCount}</b></div>
      <div class="stat"><span>丢点间隙</span><b>${report.gaps.length}</b></div>
      <div class="stat"><span>互换段</span><b>${report.swaps.length}</b></div>
      <div class="stat"><span>平均残差</span><b>${report.meanResidualMm.toFixed(2)} mm</b></div>
      <div class="stat"><span>最大残差</span><b>${report.maxResidualMm.toFixed(2)} mm</b></div>
      <div class="stat"><span>填补方式</span><b>${Object.entries(gapByMethod).map(([k, v]) => `${k}:${v}`).join(' · ')}</b></div>
      <div class="stat"><span>耗时</span><b>${report.durationMs.toFixed(0)} ms</b></div>`;
  }

  private setScrubber(frames: number): void {
    const scrub = $<HTMLInputElement>('scrubber');
    scrub.max = String(frames - 1);
    scrub.value = '0';
  }

  private setStatus(text: string): void {
    $('status').textContent = text;
  }

  private updateHud(): void {
    const clip = this.state.cleaned ?? this.state.raw;
    if (!clip) return;
    const t = this.state.frame / clip.fps;
    $('frameLabel').textContent = `帧 ${this.state.frame} / ${clip.frameCount - 1} · ${t.toFixed(2)}s`;
    $('timePill').textContent = `${t.toFixed(2)} s`;
  }

  private loop = (time: number): void => {
    const clip = this.state.cleaned ?? this.state.raw;
    if (this.state.playing && clip) {
      if (time - this.lastTime > 1000 / clip.fps) {
        this.state.frame = (this.state.frame + 1) % clip.frameCount;
        $<HTMLInputElement>('scrubber').value = String(this.state.frame);
        this.updateHud();
        this.lastTime = time;
      }
    }
    this.render();
    this.raf = requestAnimationFrame(this.loop);
  };

  private render(): void {
    const clip = this.state.view === 'raw' ? this.state.raw : (this.state.cleaned ?? this.state.raw);
    if (!clip) return;
    const frame = Math.min(this.state.frame, clip.frameCount - 1);
    const highlight = new Set<string>();
    if (this.state.report) {
      for (const s of this.state.report.swaps) {
        if (frame >= s.start && frame <= s.end) {
          highlight.add(s.markerA);
          highlight.add(s.markerB);
        }
      }
    }
    const skeletonState = this.state.retargeted
      ? {
          positions: this.state.retargeted.positions,
          definition: this.state.retargeted.definition,
          frame,
          contacts: this.state.solved?.contacts,
        }
      : this.state.solved
        ? {
            positions: this.state.solved.poses.positions,
            definition: this.state.solved.definition,
            frame,
            contacts: this.state.solved.contacts,
          }
        : undefined;
    this.renderer.render({
      markers: { clip, frame },
      skeleton: skeletonState,
      highlight,
      showFilled: true,
    });
    this.updateHud();
  }
}

new App().init();
