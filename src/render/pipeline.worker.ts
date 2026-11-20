/// <reference lib="webworker" />
import { cleanClip, type CleanOptions } from '../core/cleaner';
import { solveSkeleton, type SolveOptions } from '../core/solver';
import type { MarkerClip, SolvedSkeleton } from '../core/types';

export type WorkerRequest =
  | { kind: 'clean'; clip: MarkerClip; options: CleanOptions }
  | { kind: 'solve'; clip: MarkerClip; options: SolveOptions };

export type WorkerResponse =
  | { kind: 'progress'; fraction: number }
  | { kind: 'cleaned'; cleaned: MarkerClip; report: Awaited<ReturnType<typeof cleanClip>>['report'] }
  | { kind: 'solved'; solved: SolvedSkeleton }
  | { kind: 'error'; message: string };

const ctx = self as DedicatedWorkerGlobalScope;

ctx.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  try {
    if (msg.kind === 'clean') {
      const result = cleanClip(msg.clip, {
        ...msg.options,
        onProgress: (fraction) => ctx.postMessage({ kind: 'progress', fraction } satisfies WorkerResponse),
      });
      ctx.postMessage({
        kind: 'cleaned',
        cleaned: result.cleaned,
        report: result.report,
      } satisfies WorkerResponse);
    } else if (msg.kind === 'solve') {
      // Skeleton definition is reconstructed from the clip's bindings inside
      // the solver entry, which imports the default humanoid layout.
      const { defaultSkeleton } = await import('../core/skeleton');
      const solved = solveSkeleton(msg.clip, defaultSkeleton(), {
        ...msg.options,
        onProgress: (fraction) => ctx.postMessage({ kind: 'progress', fraction } satisfies WorkerResponse),
      });
      ctx.postMessage({ kind: 'solved', solved } satisfies WorkerResponse);
    }
  } catch (err) {
    ctx.postMessage({ kind: 'error', message: (err as Error).message } satisfies WorkerResponse);
  }
};
