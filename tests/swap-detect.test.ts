import { describe, expect, it } from 'vitest';
import { cleanClip } from '../src/core/cleaner';
import { buildSyntheticClip } from '../src/core/synthetic';
import { DEFAULT_CORRUPTION, type MotionKind } from '../src/core/types';

function labelAgreement(
  cleaned: { positions: Float32Array; markerNames: string[]; frameCount: number },
  truth: { positions: Float32Array },
): number {
  const mc = cleaned.markerNames.length;
  let good = 0;
  let total = 0;
  for (let f = 0; f < cleaned.frameCount; f++) {
    for (let m = 0; m < mc; m++) {
      const o = (f * mc + m) * 3;
      const d = Math.hypot(
        cleaned.positions[o] - truth.positions[o],
        cleaned.positions[o + 1] - truth.positions[o + 1],
        cleaned.positions[o + 2] - truth.positions[o + 2],
      );
      total++;
      if (d < 0.01) good++;
    }
  }
  return good / total;
}

describe('label swap detection', () => {
  for (const kind of ['walk', 'run', 'jump'] as MotionKind[]) {
    it(`detects and restores swapped labels on ${kind}`, () => {
      const spec = { ...DEFAULT_CORRUPTION, gapRate: 0.0004, swapCount: 20, maxSwapLength: 25 };
      const { clip, truth } = buildSyntheticClip(kind, 600, 100, spec);
      const { cleaned, report } = cleanClip(clip);
      const agreement = labelAgreement(cleaned, truth);
      expect(report.swaps.length).toBeGreaterThan(10);
      // At least 95% of all marker frames must carry their correct label.
      expect(agreement).toBeGreaterThan(0.95);
    });
  }

  it('restores an injected exact two-marker swap frame by frame', () => {
    const { clip, truth } = buildSyntheticClip('walk', 200, 100, {
      ...DEFAULT_CORRUPTION,
      gapRate: 0,
      swapCount: 0,
      seed: 7,
    });
    const a = clip.markerNames.indexOf('LUA_A');
    const b = clip.markerNames.indexOf('LUA_B');
    const f = 80;
    const oa = (f * clip.markerNames.length + a) * 3;
    const ob = (f * clip.markerNames.length + b) * 3;
    for (let k = 0; k < 3; k++) {
      const t = clip.positions[oa + k];
      clip.positions[oa + k] = clip.positions[ob + k];
      clip.positions[ob + k] = t;
    }
    const { cleaned, report } = cleanClip(clip, {
      swapResidualThresholdM: 0.025, clusterRmsLimitM: 0.02, smoothPasses: 0,
    });
    for (let k = 0; k < 3; k++) {
      // Restored to within 2 mm after the label swap is undone (no smoothing).
      expect(Math.abs(cleaned.positions[oa + k] - truth.positions[oa + k])).toBeLessThan(0.002);
      expect(Math.abs(cleaned.positions[ob + k] - truth.positions[ob + k])).toBeLessThan(0.002);
    }
    expect(report.swaps.some((s) => s.start <= f && s.end >= f)).toBe(true);
  });

  it('does not invent swaps on clean, slightly noisy data', () => {
    const { clip } = buildSyntheticClip('walk', 300, 100, {
      ...DEFAULT_CORRUPTION,
      gapRate: 0,
      swapCount: 0,
      noiseStd: 0.001,
    });
    const { report } = cleanClip(clip);
    expect(report.swaps.length).toBe(0);
    expect(report.meanResidualMm).toBeLessThan(3);
  });
});
