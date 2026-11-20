/**
 * 轻量三维数学库:Vec3 / Mat3 / Quat,以及 Horn 四元数法 Kabsch 配准。
 * 全部面向扁平 TypedArray 的批量数据优化,避免热路径上产生对象垃圾。
 */

export type Vec3Like = ArrayLike<number>;

// ---------- Vec3(以 [x,y,z] 三元组或扁平数组 + 偏移传递) ----------

export function v3Set(out: Float32Array | number[], o: number, x: number, y: number, z: number): void {
  out[o] = x; out[o + 1] = y; out[o + 2] = z;
}

export function v3Copy(out: Float32Array | number[], o: number, a: Vec3Like, ao = 0): void {
  out[o] = a[ao]; out[o + 1] = a[ao + 1]; out[o + 2] = a[ao + 2];
}

export function v3Sub(a: Vec3Like, ao: number, b: Vec3Like, bo: number, out: Float32Array, o: number): void {
  out[o] = a[ao] - b[bo];
  out[o + 1] = a[ao + 1] - b[bo + 1];
  out[o + 2] = a[ao + 2] - b[bo + 2];
}

export function v3Add(a: Vec3Like, ao: number, b: Vec3Like, bo: number, out: Float32Array, o: number): void {
  out[o] = a[ao] + b[bo];
  out[o + 1] = a[ao + 1] + b[bo + 1];
  out[o + 2] = a[ao + 2] + b[bo + 2];
}

export function v3Scale(a: Vec3Like, ao: number, s: number, out: Float32Array, o: number): void {
  out[o] = a[ao] * s;
  out[o + 1] = a[ao + 1] * s;
  out[o + 2] = a[ao + 2] * s;
}

export function v3Dot(a: Vec3Like, ao: number, b: Vec3Like, bo: number): number {
  return a[ao] * b[bo] + a[ao + 1] * b[bo + 1] + a[ao + 2] * b[bo + 2];
}

export function v3Cross(a: Vec3Like, ao: number, b: Vec3Like, bo: number, out: Float32Array, o: number): void {
  const ax = a[ao], ay = a[ao + 1], az = a[ao + 2];
  const bx = b[bo], by = b[bo + 1], bz = b[bo + 2];
  out[o] = ay * bz - az * by;
  out[o + 1] = az * bx - ax * bz;
  out[o + 2] = ax * by - ay * bx;
}

export function v3Len(a: Vec3Like, ao = 0): number {
  return Math.hypot(a[ao], a[ao + 1], a[ao + 2]);
}

export function v3Dist(a: Vec3Like, ao: number, b: Vec3Like, bo: number): number {
  return Math.hypot(a[ao] - b[bo], a[ao + 1] - b[bo + 1], a[ao + 2] - b[bo + 2]);
}

// ---------- Mat3(行主序: m[row*3+col]) ----------

export function m3Identity(out: Float32Array, o: number): void {
  out[o] = 1; out[o + 1] = 0; out[o + 2] = 0;
  out[o + 3] = 0; out[o + 4] = 1; out[o + 5] = 0;
  out[o + 6] = 0; out[o + 7] = 0; out[o + 8] = 1;
}

/** 矩阵乘法 out = a * b(均为行主序)。 */
export function m3Mul(a: Vec3Like, ao: number, b: Vec3Like, bo: number, out: Float32Array, o: number): void {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[o + r * 3 + c] =
        a[ao + r * 3] * b[bo + c] +
        a[ao + r * 3 + 1] * b[bo + 3 + c] +
        a[ao + r * 3 + 2] * b[bo + 6 + c];
    }
  }
}

/** out = a^T(行主序)。 */
export function m3Transpose(a: Vec3Like, ao: number, out: Float32Array, o: number): void {
  out[o] = a[ao]; out[o + 1] = a[ao + 3]; out[o + 2] = a[ao + 6];
  out[o + 3] = a[ao + 1]; out[o + 4] = a[ao + 4]; out[o + 5] = a[ao + 7];
  out[o + 6] = a[ao + 2]; out[o + 7] = a[ao + 5]; out[o + 8] = a[ao + 8];
}

/** 旋转矩阵作用于向量: out = R * a(R 为行主序) */
export function m3Rotate(R: Vec3Like, ro: number, a: Vec3Like, ao: number, out: Float32Array, o: number): void {
  const x = a[ao], y = a[ao + 1], z = a[ao + 2];
  out[o] = R[ro] * x + R[ro + 1] * y + R[ro + 2] * z;
  out[o + 1] = R[ro + 3] * x + R[ro + 4] * y + R[ro + 5] * z;
  out[o + 2] = R[ro + 6] * x + R[ro + 7] * y + R[ro + 8] * z;
}

// ---------- Quat(x,y,z,w),用于姿态平滑与重定向 ----------

export function qFromMat3(R: Vec3Like, ro: number, out: Float32Array, o: number): void {
  const m00 = R[ro], m01 = R[ro + 1], m02 = R[ro + 2];
  const m10 = R[ro + 3], m11 = R[ro + 4], m12 = R[ro + 5];
  const m20 = R[ro + 6], m21 = R[ro + 7], m22 = R[ro + 8];
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    out[o + 3] = 0.25 * s;
    out[o] = (m21 - m12) / s;
    out[o + 1] = (m02 - m20) / s;
    out[o + 2] = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    out[o + 3] = (m21 - m12) / s;
    out[o] = 0.25 * s;
    out[o + 1] = (m01 + m10) / s;
    out[o + 2] = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    out[o + 3] = (m02 - m20) / s;
    out[o] = (m01 + m10) / s;
    out[o + 1] = 0.25 * s;
    out[o + 2] = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    out[o + 3] = (m10 - m01) / s;
    out[o] = (m02 + m20) / s;
    out[o + 1] = (m12 + m21) / s;
    out[o + 2] = 0.25 * s;
  }
}

export function qToMat3(q: Vec3Like, qo: number, out: Float32Array, o: number): void {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  out[o] = 1 - 2 * (yy + zz);
  out[o + 1] = 2 * (xy - wz);
  out[o + 2] = 2 * (xz + wy);
  out[o + 3] = 2 * (xy + wz);
  out[o + 4] = 1 - 2 * (xx + zz);
  out[o + 5] = 2 * (yz - wx);
  out[o + 6] = 2 * (xz - wy);
  out[o + 7] = 2 * (yz + wx);
  out[o + 8] = 1 - 2 * (xx + yy);
}

export function qNormalize(q: Vec3Like, qo: number, out: Float32Array, o: number): void {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  const n = Math.hypot(x, y, z, w) || 1;
  out[o] = x / n; out[o + 1] = y / n; out[o + 2] = z / n; out[o + 3] = w / n;
}

/** 球面线性插值,qoA/qoB 为两个单位四元数,t ∈ [0,1]。 */
export function qSlerp(qA: Vec3Like, ao: number, qB: Vec3Like, bo: number, t: number, out: Float32Array, o: number): void {
  let ax = qA[ao], ay = qA[ao + 1], az = qA[ao + 2], aw = qA[ao + 3];
  let bx = qB[bo], by = qB[bo + 1], bz = qB[bo + 2], bw = qB[bo + 3];
  let cos = ax * bx + ay * by + az * bz + aw * bw;
  if (cos < 0) {
    bx = -bx; by = -by; bz = -bz; bw = -bw; cos = -cos;
  }
  let s0: number, s1: number;
  if (cos > 0.9995) {
    s0 = 1 - t; s1 = t;
  } else {
    const ang = Math.acos(Math.min(1, cos));
    const sinAng = Math.sin(ang);
    s0 = Math.sin((1 - t) * ang) / sinAng;
    s1 = Math.sin(t * ang) / sinAng;
  }
  out[o] = s0 * ax + s1 * bx;
  out[o + 1] = s0 * ay + s1 * by;
  out[o + 2] = s0 * az + s1 * bz;
  out[o + 3] = s0 * aw + s1 * bw;
  qNormalize(out, o, out, o);
}

/**
 * Kabsch 算法:求最优旋转 R,使 sum ||R p_i - q_i||² 最小。
 * 调用前 p、q 应为已减去各自质心的对应点(同序、等长 n)。
 * 返回 false 表示点集退化(共线,连两轴都无法确定)。
 *
 * 走双侧对称特征分解的 3x3 SVD:V 取自 H^T H、U 取自 H H^T(H=Σ q p^T),
 * 两组基各自天然正交,用 H 对齐奇异轴符号、叉乘补齐第三轴并做 det 校正,
 * 因此对共面标记点与绕主轴的旋转(出现小/零奇异值)同样稳健。R = U V^T。
 */
export function kabschRotation(
  p: Float32Array, qPoints: Float32Array, n: number,
  out: Float32Array, outOffset: number,
): boolean {
  const H = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < n; i++) {
    const px = p[i * 3], py = p[i * 3 + 1], pz = p[i * 3 + 2];
    const qx = qPoints[i * 3], qy = qPoints[i * 3 + 1], qz = qPoints[i * 3 + 2];
    H[0][0] += qx * px; H[0][1] += qx * py; H[0][2] += qx * pz;
    H[1][0] += qy * px; H[1][1] += qy * py; H[1][2] += qy * pz;
    H[2][0] += qz * px; H[2][1] += qz * py; H[2][2] += qz * pz;
  }

  // 双侧对称特征分解:V 取自 H^T H,U 取自 H H^T,各自天然正交,
  // 再用 H 对齐每个奇异轴的符号。对小/零奇异值(绕主轴旋转)也稳健。
  const Ht = transpose3(H);
  const eV = jacobiEigen3(mul3(Ht, H));
  const eU = jacobiEigen3(mul3(H, Ht));
  const oV = [0, 1, 2].sort((a, b) => eV.values[b] - eV.values[a]);
  const oU = [0, 1, 2].sort((a, b) => eU.values[b] - eU.values[a]);
  if (eV.values[oV[0]] <= 0 || eV.values[oV[1]] <= 1e-12 * eV.values[oV[0]]) return false;

  const Vcols: number[][] = oV.map((idx) => [eV.vectors[0][idx], eV.vectors[1][idx], eV.vectors[2][idx]]);
  const Ucols: number[][] = oU.map((idx) => [eU.vectors[0][idx], eU.vectors[1][idx], eU.vectors[2][idx]]);
  // 符号对齐:令 u_i 与 H v_i 同向(零奇异轴则与前两轴叉乘补齐)
  for (let k = 0; k < 2; k++) {
    if (dot3(Ucols[k], matVec(H, Vcols[k])) < 0) Ucols[k] = scale3(Ucols[k], -1);
  }
  Vcols[2] = cross3d(Vcols[0], Vcols[1]);
  Ucols[2] = cross3d(Ucols[0], Ucols[1]);
  // det 校正
  if (det3(mul3(columnsOf(Ucols), transpose3(columnsOf(Vcols)))) < 0) {
    Ucols[2] = scale3(Ucols[2], -1);
    Vcols[2] = scale3(Vcols[2], -1);
  }

  const R = mul3(columnsOf(Ucols), transpose3(columnsOf(Vcols)));
  const o = outOffset;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[o + r * 3 + c] = R[r][c];
  return true;
}

function columnsOf(cols: number[][]): number[][] {
  return [
    [cols[0][0], cols[1][0], cols[2][0]],
    [cols[0][1], cols[1][1], cols[2][1]],
    [cols[0][2], cols[1][2], cols[2][2]],
  ];
}

function cross3d(a: number[], b: number[]): number[] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function transpose3(m: number[][]): number[][] {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
}
function mul3(a: number[][], b: number[][]): number[][] {
  const r = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return r;
}
function det3(m: number[][]): number {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
function matVec(m: number[][], v: number[]): number[] {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}
function dot3(a: number[], b: number[]): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function scale3(a: number[], s: number): number[] { return [a[0] * s, a[1] * s, a[2] * s]; }

/** 3x3 对称矩阵 Jacobi 特征分解:values 为特征值,vectors 的列为对应特征向量。 */
function jacobiEigen3(m: number[][]): { values: number[]; vectors: number[][] } {
  const a = m.map((r) => [...r]);
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let iter = 0; iter < 100; iter++) {
    let p = 0, q = 1;
    for (let r = 0; r < 3; r++) {
      for (let c = r + 1; c < 3; c++) {
        if (Math.abs(a[r][c]) > Math.abs(a[p][q])) { p = r; q = c; }
      }
    }
    if (Math.abs(a[p][q]) < 1e-15) break;
    const phi = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(phi), sn = Math.sin(phi);
    for (let k = 0; k < 3; k++) {
      const akp = a[k][p], akq = a[k][q];
      a[k][p] = c * akp - sn * akq;
      a[k][q] = sn * akp + c * akq;
    }
    for (let k = 0; k < 3; k++) {
      const apk = a[p][k], aqk = a[q][k];
      a[p][k] = c * apk - sn * aqk;
      a[q][k] = sn * apk + c * aqk;
    }
    for (let k = 0; k < 3; k++) {
      const vkp = v[k][p], vkq = v[k][q];
      v[k][p] = c * vkp - sn * vkq;
      v[k][q] = sn * vkp + c * vkq;
    }
  }
  return { values: [a[0][0], a[1][1], a[2][2]], vectors: v };
}

/**
 * 带对应点的完整刚体配准:给定模板局部点集 local(已相对质心)与当前帧观测点 world,
 * 求世界旋转 R 与质心位置 t,使 world ≈ R*local + t。
 * observed 掩码指示哪些点参与(长度与点数相同)。至少 3 个非共线点。
 * 返回 RMS 残差(米);退化返回 null。
 */
export function rigidBodyFit(
  local: Float32Array,
  world: Float32Array,
  observed: Uint8Array,
  count: number,
  outR: Float32Array, rOffset: number,
  outT: Float32Array, tOffset: number,
): number | null {
  const active: number[] = [];
  for (let i = 0; i < count; i++) if (observed[i]) active.push(i);
  const n = active.length;
  if (n < 3) return null;

  const pc = new Float32Array(3);
  const wc = new Float32Array(3);
  for (const i of active) {
    for (let k = 0; k < 3; k++) {
      pc[k] += local[i * 3 + k];
      wc[k] += world[i * 3 + k];
    }
  }
  for (let k = 0; k < 3; k++) { pc[k] /= n; wc[k] /= n; }

  const p = new Float32Array(n * 3);
  const q = new Float32Array(n * 3);
  active.forEach((i, j) => {
    for (let k = 0; k < 3; k++) {
      p[j * 3 + k] = local[i * 3 + k] - pc[k];
      q[j * 3 + k] = world[i * 3 + k] - wc[k];
    }
  });
  if (!kabschRotation(p, q, n, outR, rOffset)) return null;
  // 局部点非零质心:t = worldCentroid - R * localCentroid(R 行主序)
  outT[tOffset] = wc[0] - (outR[rOffset] * pc[0] + outR[rOffset + 1] * pc[1] + outR[rOffset + 2] * pc[2]);
  outT[tOffset + 1] = wc[1] - (outR[rOffset + 3] * pc[0] + outR[rOffset + 4] * pc[1] + outR[rOffset + 5] * pc[2]);
  outT[tOffset + 2] = wc[2] - (outR[rOffset + 6] * pc[0] + outR[rOffset + 7] * pc[1] + outR[rOffset + 8] * pc[2]);

  let se = 0;
  const tmp = new Float32Array(3);
  for (const i of active) {
    m3Rotate(outR, rOffset, local, i * 3, tmp, 0);
    se += (tmp[0] + outT[tOffset] - world[i * 3]) ** 2
      + (tmp[1] + outT[tOffset + 1] - world[i * 3 + 1]) ** 2
      + (tmp[2] + outT[tOffset + 2] - world[i * 3 + 2]) ** 2;
  }
  return Math.sqrt(se / n);
}

/**
 * 三次 Catmull-Rom(均匀)插值,过全部控制点。
 * t 为全局帧位置(可为小数),输出到 out。边界外做钳制。
 */
export function catmullRom(points: Float32Array, stride: number, offset: number, count: number, t: number, out: Float32Array, o: number): void {
  if (t <= 0) { v3Copy(out, o, points, offset); return; }
  if (t >= count - 1) { v3Copy(out, o, points, offset + (count - 1) * stride); return; }
  const i = Math.floor(t);
  const u = t - i;
  const i0 = Math.max(0, i - 1), i1 = i, i2 = Math.min(count - 1, i + 1), i3 = Math.min(count - 1, i + 2);
  const p0 = offset + i0 * stride, p1 = offset + i1 * stride;
  const p2 = offset + i2 * stride, p3 = offset + i3 * stride;
  const u2 = u * u, u3 = u2 * u;
  for (let k = 0; k < 3; k++) {
    out[o + k] = 0.5 * (
      2 * points[p1 + k]
      + (-points[p0 + k] + points[p2 + k]) * u
      + (2 * points[p0 + k] - 5 * points[p1 + k] + 4 * points[p2 + k] - points[p3 + k]) * u2
      + (-points[p0 + k] + 3 * points[p1 + k] - 3 * points[p2 + k] + points[p3 + k]) * u3
    );
  }
}
