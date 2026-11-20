import type { Quat } from './quat';
import { qNormalize } from './quat';
import type { Vec3 } from './vec3';

export interface RigidTransform {
  rotation: Quat;
  translation: Vec3;
  rms: number;
}

/**
 * Weighted Kabsch.
 *
 * 1. Weighted centroids and cross-covariance S (3x3).
 * 2. Optimal rotation via the Bar-Itzhack / Farrell symmetric 4x4 matrix F,
 *    whose largest-eigenvalue eigenvector is the rotation quaternion.
 *    Jacobi iteration gives all four eigenvalues/eigenvectors exactly,
 *    avoiding the convergence quirks of power iteration on near-degenerate
 *    marker triads.
 */
export function kabsch(source: Vec3[], target: Vec3[], weights?: number[]): RigidTransform | null {
  const n = source.length;
  let wSum = 0;
  let cx = 0, cy = 0, cz = 0;
  let dx = 0, dy = 0, dz = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? Math.max(0, weights[i]) : 1;
    if (w <= 0) continue;
    wSum += w;
    cx += source[i][0] * w; cy += source[i][1] * w; cz += source[i][2] * w;
    dx += target[i][0] * w; dy += target[i][1] * w; dz += target[i][2] * w;
  }
  if (wSum < 1e-12) return null;
  cx /= wSum; cy /= wSum; cz /= wSum;
  dx /= wSum; dy /= wSum; dz /= wSum;

  let sxx = 0, sxy = 0, sxz = 0;
  let syx = 0, syy = 0, syz = 0;
  let szx = 0, szy = 0, szz = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? Math.max(0, weights[i]) : 1;
    if (w <= 0) continue;
    const px = source[i][0] - cx;
    const py = source[i][1] - cy;
    const pz = source[i][2] - cz;
    const qx = target[i][0] - dx;
    const qy = target[i][1] - dy;
    const qz = target[i][2] - dz;
    sxx += w * px * qx; sxy += w * px * qy; sxz += w * px * qz;
    syx += w * py * qx; syy += w * py * qy; syz += w * py * qz;
    szx += w * pz * qx; szy += w * pz * qy; szz += w * pz * qz;
  }

  // Farrell (2008) eq. 14 / Bar-Itzhack symmetric matrix F.
  // Row order corresponds to quaternion [qw, qx, qy, qz].
  const s11 = sxx, s12 = sxy, s13 = sxz;
  const s21 = syx, s22 = syy, s23 = syz;
  const s31 = szx, s32 = szy, s33 = szz;
  const F = [
    s11 + s22 + s33, s23 - s32,         s31 - s13,         s12 - s21,
    s23 - s32,        s11 - s22 - s33, s12 + s21,         s31 + s13,
    s31 - s13,        s12 + s21,       -s11 + s22 - s33, s23 + s32,
    s12 - s21,        s31 + s13,        s23 + s32,        -s11 - s22 + s33,
  ];

  const { eigenvectors, eigenvalues } = jacobi4(F);
  let best = 0;
  for (let i = 1; i < 4; i++) if (eigenvalues[i] > eigenvalues[best]) best = i;
  // F's eigenvector order is [w, x, y, z]; convert to the project's [x,y,z,w].
  const e = eigenvectors[best];
  let q = qNormalize([e[1], e[2], e[3], e[0]] as Quat);
  if (rotationDeterminant(q) < 0) q = [-q[0], -q[1], -q[2], -q[3]] as Quat;

  const rc = qRotateStatic(q, [cx, cy, cz]);
  const translation: Vec3 = [dx - rc[0], dy - rc[1], dz - rc[2]];

  let residSq = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? Math.max(0, weights[i]) : 1;
    if (w <= 0) continue;
    const rp = qRotateStatic(q, source[i]);
    const ex = rp[0] + translation[0] - target[i][0];
    const ey = rp[1] + translation[1] - target[i][1];
    const ez = rp[2] + translation[2] - target[i][2];
    residSq += w * (ex * ex + ey * ey + ez * ez);
  }
  return { rotation: q, translation, rms: Math.sqrt(residSq / wSum) };
}

/**
 * Jacobi eigen decomposition for a symmetric 4x4 matrix.
 * Returns unit eigenvectors as rows, aligned with the returned eigenvalues.
 */
function jacobi4(matrix: number[]): { eigenvalues: number[]; eigenvectors: number[][] } {
  const a = matrix.slice();
  const v = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) {
      for (let q = p + 1; q < 4; q++) off += a[p * 4 + q] * a[p * 4 + q];
    }
    if (off < 1e-24) break;

    for (let p = 0; p < 3; p++) {
      for (let q = p + 1; q < 4; q++) {
        const apq = a[p * 4 + q];
        if (Math.abs(apq) < 1e-18) continue;
        const app = a[p * 4 + p];
        const aqq = a[q * 4 + q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (let k = 0; k < 4; k++) {
          const akp = a[k * 4 + p];
          const akq = a[k * 4 + q];
          a[k * 4 + p] = c * akp - s * akq;
          a[k * 4 + q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const apk = a[p * 4 + k];
          const aqk = a[q * 4 + k];
          a[p * 4 + k] = c * apk - s * aqk;
          a[q * 4 + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {
          const vkp = v[k * 4 + p];
          const vkq = v[k * 4 + q];
          v[k * 4 + p] = c * vkp - s * vkq;
          v[k * 4 + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const eigenvalues = [a[0], a[5], a[10], a[15]];
  // Columns of v are the eigenvectors; return them as rows aligned by index.
  const eigenvectors = [
    [v[0], v[4], v[8], v[12]],
    [v[1], v[5], v[9], v[13]],
    [v[2], v[6], v[10], v[14]],
    [v[3], v[7], v[11], v[15]],
  ];
  return { eigenvalues, eigenvectors };
}

function rotationDeterminant(q: Quat): number {
  const [x, y, z, w] = q;
  return (
    (1 - 2 * (y * y + z * z)) *
    (1 - 2 * (x * x + z * z)) *
    (1 - 2 * (x * x + y * y))
  );
}

function qRotateStatic(q: Quat, v: Vec3): Vec3 {
  const [x, y, z, w] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}
