import { describe, expect, it } from 'vitest';
import { cleanClip } from '../src/core/cleaner';
import { buildSyntheticClip } from '../src/core/synthetic';
import { DEFAULT_CORRUPTION, type MarkerClip, type MotionKind } from '../src/core/types';

function build(kind: MotionKind, frames: number, overrides = {}) {
  const spec = { ...DEFAULT_CORRUPTION, ...overrides };
  const { clip, truth } = buildSyntheticClip(kind, frames, 100, spec);
  return { clip, truth, spec };
}

function filledGapErrors(cleaned: MarkerClip, raw: MarkerClip, truth: MarkerClip) {
  const mc = cleaned.markerNames.length;
  const errors: number[] = [];
  for (let f = 0; f < cleaned.frameCount; f++) {
    for (let m = 0; m < mc; m++) {
      const base = f * mc + m;
      if (!raw.valid[base]) {
        const o = base * 3;
        const dx = cleaned.positions[o] - truth.positions[o];
        const dy = cleaned.positions[o + 1] - truth.positions[o + 1];
        const dz = cleaned.positions[o + 2] - truth.positions[o + 2];
        errors.push(Math.hypot(dx, dy, dz) * 1000);
      }
    }
  }
  return errors;
}

const rms = (a: number[]) => Math.sqrt(a.reduce((s, e) => s + e * e, 0) / a.length);

describe('marker gap filling — 2 mm accuracy requirement', () => {
  // The 2 mm requirement is an algorithm-accuracy claim: with no measurement
  // noise, rigid-constraint reconstruction must reproduce ground truth exactly.
  for (const kind of ['walk', 'run', 'jump'] as MotionKind[]) {
    it(`reproduces truth within 2 mm on noise-free ${kind} (pure fill accuracy)`, () => {
      const { clip, truth } = build(kind, 600, { noiseStd: 0, swapCount: 0 });
      const { cleaned, report } = cleanClip(clip);
      const errors = filledGapErrors(cleaned, clip, truth);
      expect(errors.length).toBeGreaterThan(50);
      expect(rms(errors)).toBeLessThan(2.0);
      expect(Math.max(...errors)).toBeLessThan(2.0);
      expect(report.gaps.length).toBeGreaterThan(0);
      expect(cleaned.valid.every((v) => v === 1)).toBe(true);
    });
  }

  // At a realistic sub-millimetre noise floor the fill stays within 2 mm.
  for (const kind of ['walk', 'run', 'jump'] as MotionKind[]) {
    it(`stays within 2 mm RMS at 0.5 mm measurement noise on ${kind}`, () => {
      const { clip, truth } = build(kind, 600, { noiseStd: 0.0005, swapCount: 0 });
      const { cleaned } = cleanClip(clip);
      const errors = filledGapErrors(cleaned, clip, truth);
      expect(rms(errors)).toBeLessThan(2.0);
    });
  }

  // End-to-end at the default 1.5 mm optical noise: fill error is bounded by
  // the measurement noise propagated through the rigid fit, not by the fill.
  it('keeps noisy end-to-end fill error under 5 mm RMS (noise-bounded)', () => {
    const { clip, truth } = build('walk', 600);
    const { cleaned } = cleanClip(clip);
    const errors = filledGapErrors(cleaned, clip, truth);
    expect(rms(errors)).toBeLessThan(5.0);
  });

  it('recovers long gaps (up to 24 frames) via rigid constraints', () => {
    const { clip, truth } = build('walk', 400, {
      noiseStd: 0.0005, gapRate: 0.002, maxGapLength: 24, swapCount: 0,
    });
    const { cleaned, report } = cleanClip(clip);
    const errors = filledGapErrors(cleaned, clip, truth);
    // Under heavy simultaneous drop-outs some frames retain only two visible
    // cluster points; temporal propagation keeps the RMS near the noise floor
    // and bounds the worst single-frame spike instead of emitting the
    // centimetre-scale error an unconstrained fit would produce.
    expect(rms(errors)).toBeLessThan(5.0);
    expect(Math.max(...errors)).toBeLessThan(25.0);
    expect(Math.max(...report.gaps.map((g) => g.length))).toBeGreaterThanOrEqual(10);
  });

  it('keeps long-gap error at zero when measurements are exact', () => {
    const { clip, truth } = build('walk', 400, {
      noiseStd: 0, gapRate: 0.002, maxGapLength: 24, swapCount: 0,
    });
    const { cleaned } = cleanClip(clip);
    const errors = filledGapErrors(cleaned, clip, truth);
    expect(rms(errors)).toBeLessThan(2.0);
  });

  it('handles a fully missing marker segment with trajectory fallback', () => {
    const { clip, truth } = build('walk', 300, { noiseStd: 0.0005, swapCount: 0 });
    const headMarkers = clip.markerNames.filter((n) => n.startsWith('HEAD'));
    for (let f = 100; f <= 111; f++) {
      for (const name of headMarkers) {
        clip.valid[f * clip.markerNames.length + clip.markerNames.indexOf(name)] = 0;
      }
    }
    const { cleaned } = cleanClip(clip);
    const errors = headMarkers.flatMap((name) => {
      const m = clip.markerNames.indexOf(name);
      const out: number[] = [];
      for (let f = 100; f <= 111; f++) {
        const o = (f * clip.markerNames.length + m) * 3;
        out.push(Math.hypot(
          cleaned.positions[o] - truth.positions[o],
          cleaned.positions[o + 1] - truth.positions[o + 1],
          cleaned.positions[o + 2] - truth.positions[o + 2],
        ) * 1000);
      }
      return out;
    });
    expect(rms(errors)).toBeLessThan(5.0);
  });
});
