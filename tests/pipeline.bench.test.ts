import { describe, expect, it } from 'vitest';
import { cleanClip, DEFAULT_CLEAN_OPTIONS } from '../src/core/cleaner';
import { buildSyntheticClip } from '../src/core/synthetic';
import { DEFAULT_CORRUPTION } from '../src/core/types';

describe('pipeline performance — 10k frames under 30 s', () => {
  it('cleans 10,000 frames of 42 markers in under 30 seconds', () => {
    const { clip } = buildSyntheticClip('walk', 10000, 100, DEFAULT_CORRUPTION);
    const t0 = performance.now();
    const { cleaned, report } = cleanClip(clip, { ...DEFAULT_CLEAN_OPTIONS, smoothPasses: 1 });
    const elapsedMs = performance.now() - t0;
    // The report also times the internal work; assert both stay under budget.
    expect(report.durationMs).toBeLessThan(30000);
    expect(elapsedMs).toBeLessThan(30000);
    expect(cleaned.frameCount).toBe(10000);
    expect(cleaned.valid.every((v) => v === 1)).toBe(true);
  });

  it('throughput stays above 500 frames per second', () => {
    const { clip } = buildSyntheticClip('run', 4000, 100, DEFAULT_CORRUPTION);
    const t0 = performance.now();
    const { report } = cleanClip(clip);
    const elapsedSec = (performance.now() - t0) / 1000;
    const fps = report.frameCount / elapsedSec;
    expect(fps).toBeGreaterThan(500);
  });
});
