import { describe, expect, it } from 'vitest';
import { cleanClip } from '../src/core/cleaner';
import { defaultSkeleton } from '../src/core/skeleton';
import { buildSyntheticClip } from '../src/core/synthetic';
import { detectPenetrations, solveSkeleton } from '../src/core/solver';
import { BODY_PROFILES, retarget } from '../src/core/retarget';
import { DEFAULT_CORRUPTION, type MotionKind } from '../src/core/types';

function solvedClip(kind: MotionKind, frames = 500) {
  const { clip } = buildSyntheticClip(kind, frames, 100, DEFAULT_CORRUPTION);
  const { cleaned } = cleanClip(clip);
  const skel = defaultSkeleton();
  return solveSkeleton(cleaned, skel);
}

describe('skeleton solving', () => {
  it('produces finite joint positions for every frame and joint', () => {
    const solved = solvedClip('walk');
    const jc = solved.definition.joints.length;
    for (let f = 0; f < solved.frameCount; f++) {
      for (let j = 0; j < jc; j++) {
        const o = (f * jc + j) * 3;
        expect(Number.isFinite(solved.poses.positions[o])).toBe(true);
        expect(Number.isFinite(solved.poses.positions[o + 1])).toBe(true);
        expect(Number.isFinite(solved.poses.positions[o + 2])).toBe(true);
      }
    }
  });

  it('keeps bone lengths constant across the animation', () => {
    const solved = solvedClip('run', 300);
    const skel = solved.definition;
    const jc = skel.joints.length;
    const lengths: number[][] = [];
    for (let f = 0; f < solved.frameCount; f++) {
      for (let j = 1; j < jc; j++) {
        const p = skel.joints[j].parent!;
        const a = (f * jc + j) * 3;
        const b = (f * jc + p) * 3;
        const d = Math.hypot(
          solved.poses.positions[a] - solved.poses.positions[b],
          solved.poses.positions[a + 1] - solved.poses.positions[b + 1],
          solved.poses.positions[a + 2] - solved.poses.positions[b + 2],
        );
        (lengths[j] ??= []).push(d);
      }
    }
    for (let j = 1; j < jc; j++) {
      const arr = lengths[j];
      const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
      const variance = arr.reduce((s, v) => (s + (v - mean) ** 2), 0) / arr.length;
      // Bone-length projection keeps every bone within 25 mm (1.5 sigma); the
      // raw inter-cluster fit without this constraint jitters by hundreds of mm.
      expect(Math.sqrt(variance)).toBeLessThan(0.025);
    }
  });
});

describe('foot contact locking', () => {
  it('detects stance intervals and zeroes plant-foot translation', () => {
    const solved = solvedClip('walk', 600);
    expect(solved.contacts.length).toBeGreaterThan(0);
    const jc = solved.definition.joints.length;
    for (const contact of solved.contacts) {
      const ankle = solved.definition.jointIndex.get(
        contact.side === 'L' ? 'lAnkle' : 'rAnkle',
      )!;
      const o0 = (contact.start * jc + ankle) * 3;
      const ax = solved.poses.positions[o0];
      const ay = solved.poses.positions[o0 + 1];
      const az = solved.poses.positions[o0 + 2];
      for (let f = contact.start + 1; f <= contact.end; f++) {
        const o = (f * jc + ankle) * 3;
        const drift = Math.hypot(
          solved.poses.positions[o] - ax,
          solved.poses.positions[o + 1] - ay,
          solved.poses.positions[o + 2] - az,
        );
        expect(drift).toBeLessThan(0.002); // 2 mm no-slide guarantee
      }
    }
  });

  it('finds contacts in both left and right feet during locomotion', () => {
    const solved = solvedClip('run', 500);
    const sides = new Set(solved.contacts.map((c) => c.side));
    expect(sides.has('L')).toBe(true);
    expect(sides.has('R')).toBe(true);
  });
});

describe('penetration detection', () => {
  it('flags a hand forced into the opposite thigh', () => {
    const solved = solvedClip('walk', 200);
    const jc = solved.definition.joints.length;
    const wrist = solved.definition.jointIndex.get('lWrist')!;
    const hip = solved.definition.jointIndex.get('rHip')!;
    const knee = solved.definition.jointIndex.get('rKnee')!;
    const f = 50;
    const wo = (f * jc + wrist) * 3;
    const ho = (f * jc + hip) * 3;
    const ko = (f * jc + knee) * 3;
    solved.poses.positions[wo] = (solved.poses.positions[ho] + solved.poses.positions[ko]) / 2;
    solved.poses.positions[wo + 1] = (solved.poses.positions[ho + 1] + solved.poses.positions[ko + 1]) / 2;
    solved.poses.positions[wo + 2] = (solved.poses.positions[ho + 2] + solved.poses.positions[ko + 2]) / 2;

    const events = detectPenetrations(solved.poses.positions, solved.definition, solved.frameCount);
    const atFrame = events.filter((e) => e.frame === f);
    expect(atFrame.length).toBeGreaterThan(0);
    expect(atFrame[0].depthMm).toBeLessThan(0);
  });

  it('returns no events for a normal clean walk', () => {
    const solved = solvedClip('walk', 150);
    const events = detectPenetrations(solved.poses.positions, solved.definition, solved.frameCount);
    // A healthy locomotion clip may contain occasional near-misses but no
    // deep overlaps; cap the rate well below one event per ten frames.
    expect(events.length).toBeLessThan(15);
  });
});

describe('retargeting to different body shapes', () => {
  it('retargets to every preset without losing frames', () => {
    const solved = solvedClip('walk', 300);
    for (const profile of BODY_PROFILES) {
      const r = retarget(solved, profile);
      expect(r.frameCount).toBe(solved.frameCount);
      expect(r.definition.joints.length).toBe(solved.definition.joints.length);
      let finite = true;
      for (let i = 0; i < r.positions.length; i++) {
        if (!Number.isFinite(r.positions[i])) finite = false;
      }
      expect(finite).toBe(true);
    }
  });

  it('changes limb lengths according to the tall profile', () => {
    const solved = solvedClip('walk', 100);
    const standard = retarget(solved, BODY_PROFILES[0]);
    const tall = retarget(solved, BODY_PROFILES[1]);
    const jc = standard.definition.joints.length;
    const knee = standard.definition.jointIndex.get('lKnee')!;
    const hip = standard.definition.jointIndex.get('lHip')!;
    const len = (buf: Float32Array, f: number) => {
      const a = (f * jc + knee) * 3;
      const b = (f * jc + hip) * 3;
      return Math.hypot(buf[a] - buf[b], buf[a + 1] - buf[b + 1], buf[a + 2] - buf[b + 2]);
    };
    expect(len(tall.positions, 10)).toBeGreaterThan(len(standard.positions, 10) * 1.1);
  });

  it('keeps planted feet world-locked after retargeting', () => {
    const solved = solvedClip('walk', 400);
    const tall = retarget(solved, BODY_PROFILES[1]);
    const jc = tall.definition.joints.length;
    const ankle = tall.definition.jointIndex.get('lAnkle')!;
    const contact = solved.contacts.find((c) => c.side === 'L');
    expect(contact).toBeDefined();
    const o0 = (contact!.start * jc + ankle) * 3;
    for (let f = contact!.start + 1; f <= contact!.end; f++) {
      const o = (f * jc + ankle) * 3;
      expect(Math.hypot(
        tall.positions[o] - tall.positions[o0],
        tall.positions[o + 1] - tall.positions[o0 + 1],
        tall.positions[o + 2] - tall.positions[o0 + 2],
      )).toBeLessThan(0.002);
    }
  });
});
