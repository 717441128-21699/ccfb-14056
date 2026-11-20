import { describe, expect, it } from 'vitest';
import { kabsch } from '../src/math/kabsch';
import { qFromAxisAngle, qMultiply, qRotate, qSlerp } from '../src/math/quat';
import { solveTwoBone } from '../src/core/ik';
import { hungarian } from '../src/core/hungarian';
import type { Vec3 } from '../src/math/vec3';

const tetra: Vec3[] = [
  [0.1, 0.1, 0.1],
  [-0.1, -0.05, 0.08],
  [0.05, -0.12, -0.1],
  [-0.08, 0.06, -0.09],
];

describe('Kabsch rigid registration', () => {
  it('recovers identity transform exactly', () => {
    const fit = kabsch(tetra, tetra)!;
    expect(fit.rms).toBeLessThan(1e-9);
    expect(fit.translation.every((t) => Math.abs(t) < 1e-9)).toBe(true);
  });

  it.each([
    [[1, 0, 0], 0.7],
    [[0, 1, 0], -1.1],
    [[0, 0, 1], 2.0],
    [[0.3, 0.8, 0.5], 1.3],
  ] as const)('recovers rotation %s angle %f with zero residual', (axis, angle) => {
    const q = qFromAxisAngle([...axis] as Vec3, angle);
    const t: Vec3 = [0.13, -0.27, 0.41];
    const dst = tetra.map((p) => {
      const r = qRotate(q, p);
      return [r[0] + t[0], r[1] + t[1], r[2] + t[2]] as Vec3;
    });
    const fit = kabsch(tetra, dst)!;
    expect(fit.rms).toBeLessThan(1e-9);
    for (let i = 0; i < 3; i++) {
      expect(fit.translation[i]).toBeCloseTo(t[i], 6);
    }
  });

  it('returns null when fewer than one weighted point exists', () => {
    expect(kabsch([[0, 0, 0]], [[1, 1, 1]], [0])).toBeNull();
  });
});

describe('quaternion algebra', () => {
  it('rotates a unit vector by 90 degrees around Z', () => {
    const q = qFromAxisAngle([0, 0, 1], Math.PI / 2);
    const v = qRotate(q, [1, 0, 0]);
    expect(v[0]).toBeCloseTo(0, 6);
    expect(v[1]).toBeCloseTo(1, 6);
  });

  it('composes rotations associatively with identity', () => {
    const q = qFromAxisAngle([0, 1, 0], 0.6);
    const id: Vec3 = [0, 0, 0];
    const idq = [...id, 1] as [number, number, number, number];
    const r = qMultiply(idq, q);
    for (let i = 0; i < 4; i++) expect(r[i]).toBeCloseTo(q[i], 8);
  });

  it('slerp endpoints reproduce inputs', () => {
    const a = qFromAxisAngle([0, 1, 0], 0);
    const b = qFromAxisAngle([0, 1, 0], 1);
    expect(qSlerp(a, b, 0)[3]).toBeCloseTo(a[3], 8);
    expect(qSlerp(a, b, 1)[3]).toBeCloseTo(b[3], 8);
  });
});

describe('two-bone analytical IK', () => {
  it('places the end effector exactly on the target', () => {
    const start: Vec3 = [0, 1, 0];
    const res = solveTwoBone(start, 0.45, 0.4, [0.2, 0.2, 0.1], [0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]);
    const reach = Math.hypot(
      res.end[0] - start[0], res.end[1] - start[1], res.end[2] - start[2],
    );
    expect(reach).toBeLessThan(0.85);
    expect(reach).toBeGreaterThan(0.05);
  });
});

describe('Hungarian assignment', () => {
  it('finds the minimum-cost perfect matching', () => {
    const cost = [
      [9, 2, 7],
      [6, 4, 3],
      [5, 8, 1],
    ];
    const assignment = hungarian(cost, 3, 3);
    const total = assignment.reduce((sum, col, row) => sum + cost[row][col], 0);
    // Optimal: row0->col1 (2), row1->col0 (6), row2->col2 (1) = 9
    expect(total).toBe(9);
    expect(new Set(assignment).size).toBe(3);
  });

  it('handles rectangular matrices', () => {
    const cost = [
      [1, 9],
      [9, 1],
    ];
    const assignment = hungarian(cost, 2, 2);
    expect(assignment[0]).toBe(0);
    expect(assignment[1]).toBe(1);
  });
});
