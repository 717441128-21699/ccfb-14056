import { describe, expect, it } from 'vitest';
import { parseMarkerCsv, toMarkerCsv, parseMarkerJson, toMarkerJson } from '../src/io/formats';
import { buildSyntheticClip } from '../src/core/synthetic';
import { DEFAULT_CORRUPTION } from '../src/core/types';

describe('CSV / JSON round trips', () => {
  it('preserves positions and missing-sample mask through CSV', () => {
    const { clip } = buildSyntheticClip('walk', 120, 100, DEFAULT_CORRUPTION);
    const csv = toMarkerCsv(clip);
    const restored = parseMarkerCsv(csv, clip.fps);
    expect(restored.frameCount).toBe(clip.frameCount);
    expect(restored.markerNames).toEqual(clip.markerNames);
    const mc = clip.markerNames.length;
    let missing = 0;
    for (let i = 0; i < clip.valid.length; i++) {
      expect(restored.valid[i]).toBe(clip.valid[i]);
      if (!clip.valid[i]) missing++;
    }
    expect(missing).toBeGreaterThan(0);
    for (let f = 0; f < clip.frameCount; f++) {
      for (let m = 0; m < mc; m++) {
        if (!clip.valid[f * mc + m]) continue;
        const o = (f * mc + m) * 3;
        // CSV serializes to 6 decimals (~micrometre quantization).
        expect(restored.positions[o]).toBeCloseTo(clip.positions[o], 4);
        expect(restored.positions[o + 1]).toBeCloseTo(clip.positions[o + 1], 4);
        expect(restored.positions[o + 2]).toBeCloseTo(clip.positions[o + 2], 4);
      }
    }
  });

  it('preserves data through JSON', () => {
    const { clip } = buildSyntheticClip('run', 100, 100, { ...DEFAULT_CORRUPTION, swapCount: 0 });
    const json = toMarkerJson(clip);
    const restored = parseMarkerJson(json);
    expect(restored.frameCount).toBe(clip.frameCount);
    expect(restored.markerNames).toEqual(clip.markerNames);
    const mc = clip.markerNames.length;
    for (let f = 0; f < clip.frameCount; f++) {
      for (let m = 0; m < mc; m++) {
        expect(restored.valid[f * mc + m]).toBe(clip.valid[f * mc + m]);
        if (!clip.valid[f * mc + m]) continue;
        const o = (f * mc + m) * 3;
        expect(restored.positions[o]).toBeCloseTo(clip.positions[o], 6);
      }
    }
  });

  it('rejects a malformed CSV header', () => {
    const bad = 'frame,a.x,a.y,b.x\n0,1,2,3';
    expect(() => parseMarkerCsv(bad, 100)).toThrow();
  });
});
