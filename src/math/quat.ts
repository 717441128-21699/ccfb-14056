import type { Vec3 } from './vec3';
import { cross, dot, normalize } from './vec3';

/** Quaternion as [x, y, z, w]. */
export type Quat = [number, number, number, number];

export function quatIdentity(): Quat {
  return [0, 0, 0, 1];
}

export function qClone(q: Quat): Quat {
  return [q[0], q[1], q[2], q[3]];
}

/** Hamilton product a * b. */
export function qMultiply(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function qNormalize(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  if (l < 1e-12) return [0, 0, 0, 1];
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

export function qConjugate(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

export function qRotate(q: Quat, v: Vec3): Vec3 {
  // q * v * q^-1, expanded
  const [x, y, z, w] = q;
  const vx = v[0];
  const vy = v[1];
  const vz = v[2];
  // t = 2 * cross(q.xyz, v)
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

/** Quaternion rotating unit vector `from` onto unit vector `to`. */
export function qFromUnitVectors(from: Vec3, to: Vec3): Quat {
  const d = dot(from, to);
  if (d > 1 - 1e-10) return [0, 0, 0, 1];
  if (d < -1 + 1e-10) {
    // 180°: pick any axis perpendicular to `from`
    const axis = Math.abs(from[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const n = normalize(cross(from, axis as Vec3));
    return [n[0], n[1], n[2], 0];
  }
  const c = cross(from, to);
  return qNormalize([c[0], c[1], c[2], 1 + d]);
}

/** Quaternion from axis (need not be unit) and angle in radians. */
export function qFromAxisAngle(axis: Vec3, angle: number): Quat {
  const n = normalize(axis);
  const h = angle * 0.5;
  const s = Math.sin(h);
  return [n[0] * s, n[1] * s, n[2] * s, Math.cos(h)];
}

/** Spherical linear interpolation. */
export function qSlerp(a: Quat, b: Quat, t: number): Quat {
  let cosOmega = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (cosOmega < 0) {
    cosOmega = -cosOmega;
    bb = [-b[0], -b[1], -b[2], -b[3]];
  }
  if (cosOmega > 1 - 1e-6) {
    // close enough: normalized lerp
    return qNormalize([
      a[0] + (bb[0] - a[0]) * t,
      a[1] + (bb[1] - a[1]) * t,
      a[2] + (bb[2] - a[2]) * t,
      a[3] + (bb[3] - a[3]) * t,
    ]);
  }
  const omega = Math.acos(Math.min(1, cosOmega));
  const sinOmega = Math.sin(omega);
  const wa = Math.sin((1 - t) * omega) / sinOmega;
  const wb = Math.sin(t * omega) / sinOmega;
  return [
    a[0] * wa + bb[0] * wb,
    a[1] * wa + bb[1] * wb,
    a[2] * wa + bb[2] * wb,
    a[3] * wa + bb[3] * wb,
  ];
}

/** Rotation matrix (3x3, row-major) from a unit quaternion. */
export function qToMat3(q: Quat): number[] {
  const [x, y, z, w] = q;
  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const yx = y * x2;
  const yy = y * y2;
  const zx = z * x2;
  const zy = z * y2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;
  return [
    1 - (yy + zz), yx - wz, zx + wy,
    yx + wz, 1 - (xx + zz), zy - wx,
    zx - wy, zy + wx, 1 - (xx + yy),
  ];
}
