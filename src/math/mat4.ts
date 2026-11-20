import type { Quat } from './quat';
import { qToMat3 } from './quat';
import type { Vec3 } from './vec3';

/**
 * Column-major 4x4 matrix, same layout as WebGL / Float32Array.
 * element(col, row) = m[col * 4 + row].
 */
export type Mat4 = Float32Array;

export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row] * b[col * 4 + k];
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

export function mat4FromQuatPos(q: Quat, p: Vec3): Mat4 {
  const r = qToMat3(q);
  const m = new Float32Array(16);
  // r is row-major; transpose into column-major
  m[0] = r[0]; m[1] = r[3]; m[2] = r[6];
  m[4] = r[1]; m[5] = r[4]; m[6] = r[7];
  m[8] = r[2]; m[9] = r[5]; m[10] = r[8];
  m[12] = p[0]; m[13] = p[1]; m[14] = p[2];
  m[15] = 1;
  return m;
}

export function perspective(fovyRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

export function lookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const zx = eye[0] - center[0];
  const zy = eye[1] - center[1];
  const zz = eye[2] - center[2];
  let len = Math.hypot(zx, zy, zz) || 1;
  const fz0 = zx / len;
  const fz1 = zy / len;
  const fz2 = zz / len;
  // x = normalize(cross(up, z))
  let xx = up[1] * fz2 - up[2] * fz1;
  let xy = up[2] * fz0 - up[0] * fz2;
  let xz = up[0] * fz1 - up[1] * fz0;
  len = Math.hypot(xx, xy, xz) || 1;
  xx /= len; xy /= len; xz /= len;
  // y = cross(z, x)
  const yx = fz1 * xz - fz2 * xy;
  const yy = fz2 * xx - fz0 * xz;
  const yz = fz0 * xy - fz1 * xx;
  const m = new Float32Array(16);
  m[0] = xx; m[1] = yx; m[2] = fz0;
  m[4] = xy; m[5] = yy; m[6] = fz1;
  m[8] = xz; m[9] = yz; m[10] = fz2;
  m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  m[14] = -(fz0 * eye[0] + fz1 * eye[1] + fz2 * eye[2]);
  m[15] = 1;
  return m;
}

export function mat4TransformPoint(m: Mat4, p: Vec3): Vec3 {
  const x = p[0];
  const y = p[1];
  const z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  const invW = w === 1 ? 1 : 1 / w;
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) * invW,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) * invW,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) * invW,
  ];
}

/** Inverse of a rigid (rotation + translation, uniform scale allowed) matrix. */
export function mat4InvertRigid(m: Mat4): Mat4 {
  const out = new Float32Array(16);
  // transpose 3x3
  out[0] = m[0]; out[1] = m[4]; out[2] = m[8];
  out[4] = m[1]; out[5] = m[5]; out[6] = m[9];
  out[8] = m[2]; out[9] = m[6]; out[10] = m[10];
  out[3] = 0; out[7] = 0; out[11] = 0; out[15] = 1;
  out[12] = -(out[0] * m[12] + out[4] * m[13] + out[8] * m[14]);
  out[13] = -(out[1] * m[12] + out[5] * m[13] + out[9] * m[14]);
  out[14] = -(out[2] * m[12] + out[6] * m[13] + out[10] * m[14]);
  return out;
}
