/**
 * 关节世界旋转框架构造与两骨 IK,供动作合成与骨骼解算共享。
 */
import type { Rig } from './types.js';

const AXES: Float32Array[] = [
  new Float32Array([1, 0, 0]), new Float32Array([0, 1, 0]), new Float32Array([0, 0, 1]),
];

function setCol(m: Float32Array, col: number, x: number, y: number, z: number): void {
  m[col] = x; m[3 + col] = y; m[6 + col] = z;
}

/**
 * 由父子关节方向构造关节世界旋转(相对 T-pose)。
 * 主坐标轴取 T-pose 下指向首个子关节(叶节点取父->自身)的方向,
 * 再选与主轴最不平行的世界轴作参考构造正交框架,当前框架与静帧框架配准。
 */
export function buildJointFrame(
  rig: Rig,
  rest: Float32Array, world: Float32Array,
  joint: number, firstChild: ArrayLike<number | null>,
  out: Float32Array,
): void {
  const parent = rig.joints[joint].parent;
  const child = firstChild[joint];
  const from = child ?? joint;
  const to = child !== null ? child : parent!;
  const r0 = rest[joint * 3], r1 = rest[joint * 3 + 1], r2 = rest[joint * 3 + 2];
  let a0x = rest[from * 3] - r0, a0y = rest[from * 3 + 1] - r1, a0z = rest[from * 3 + 2] - r2;
  const n0 = Math.hypot(a0x, a0y, a0z) || 1;
  a0x /= n0; a0y /= n0; a0z /= n0;
  const w0 = world[joint * 3], w1 = world[joint * 3 + 1], w2 = world[joint * 3 + 2];
  let ax: number, ay: number, az: number;
  if (child !== null) {
    ax = world[child * 3] - w0; ay = world[child * 3 + 1] - w1; az = world[child * 3 + 2] - w2;
  } else {
    ax = world[joint * 3] - world[to * 3];
    ay = world[joint * 3 + 1] - world[to * 3 + 1];
    az = world[joint * 3 + 2] - world[to * 3 + 2];
  }
  const n = Math.hypot(ax, ay, az) || 1;
  ax /= n; ay /= n; az /= n;

  let ref = AXES[0];
  let bestDot = 1;
  for (const cand of AXES) {
    const d = Math.abs(a0x * cand[0] + a0y * cand[1] + a0z * cand[2]);
    if (d < bestDot) { bestDot = d; ref = cand; }
  }
  const frame = (axx: number, ayy: number, azz: number, rx: number, ry: number, rz: number): Float32Array => {
    const dot = rx * axx + ry * ayy + rz * azz;
    let bx = rx - axx * dot, by = ry - ayy * dot, bz = rz - azz * dot;
    const bn = Math.hypot(bx, by, bz) || 1;
    bx /= bn; by /= bn; bz /= bn;
    const cx = ayy * bz - azz * by;
    const cy = azz * bx - axx * bz;
    const cz = axx * by - ayy * bx;
    const m = new Float32Array(9);
    setCol(m, 0, axx, ayy, azz);
    setCol(m, 1, bx, by, bz);
    setCol(m, 2, cx, cy, cz);
    return m;
  };
  const mRest = frame(a0x, a0y, a0z, ref[0], ref[1], ref[2]);
  const mCur = frame(ax, ay, az, ref[0], ref[1], ref[2]);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        mCur[r] * mRest[c] + mCur[3 + r] * mRest[3 + c] + mCur[6 + r] * mRest[6 + c];
    }
  }
}

/**
 * 两骨 IK:调整 knee,使 hip->ankle 距离约束在 [|l1-l2|, l1+l2] 内,
 * 膝弯曲方向取静帧膝盖相对髋踝连线的侧向。原地修改 world 中 hip/knee/ankle 三节点之外不动,
 * 仅写 knee 坐标(踝目标由调用方锁定)。
 */
export function twoBoneIK(
  world: Float32Array, rest: Float32Array,
  hip: number, knee: number, ankle: number,
): void {
  const hx = world[hip * 3], hy = world[hip * 3 + 1], hz = world[hip * 3 + 2];
  const ax = world[ankle * 3], ay = world[ankle * 3 + 1], az = world[ankle * 3 + 2];
  const l1 = Math.hypot(
    rest[knee * 3] - rest[hip * 3],
    rest[knee * 3 + 1] - rest[hip * 3 + 1],
    rest[knee * 3 + 2] - rest[hip * 3 + 2],
  );
  const l2 = Math.hypot(
    rest[ankle * 3] - rest[knee * 3],
    rest[ankle * 3 + 1] - rest[knee * 3 + 1],
    rest[ankle * 3 + 2] - rest[knee * 3 + 2],
  );
  let dx = ax - hx, dy = ay - hy, dz = az - hz;
  const d = Math.min(Math.max(Math.hypot(dx, dy, dz), Math.abs(l1 - l2) + 1e-6), l1 + l2 - 1e-6);
  const ux = dx / (Math.hypot(dx, dy, dz) || 1);
  const uy = dy / (Math.hypot(dx, dy, dz) || 1);
  const uz = dz / (Math.hypot(dx, dy, dz) || 1);
  // 余弦定理求髋处夹角
  const cosA = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d || 1);
  const along = l1 * Math.min(1, Math.max(-1, cosA));
  const perp = Math.sqrt(Math.max(0, l1 * l1 - along * along));

  // 静帧膝盖方向(相对髋->踝轴),保留为弯曲极向量
  const rkx = rest[knee * 3] - rest[hip * 3];
  const rky = rest[knee * 3 + 1] - rest[hip * 3 + 1];
  const rkz = rest[knee * 3 + 2] - rest[hip * 3 + 2];
  const rdx = rest[ankle * 3] - rest[hip * 3];
  const rdy = rest[ankle * 3 + 1] - rest[hip * 3 + 1];
  const rdz = rest[ankle * 3 + 2] - rest[hip * 3 + 2];
  const rdn = Math.hypot(rdx, rdy, rdz) || 1;
  const proj = (rkx * rdx + rky * rdy + rkz * rdz) / (rdn * rdn);
  let px = rkx - rdx * proj, py = rky - rdy * proj, pz = rkz - rdz * proj;
  const pn = Math.hypot(px, py, pz) || 1;
  px /= pn; py /= pn; pz /= pn;

  world[knee * 3] = hx + ux * along + px * perp;
  world[knee * 3 + 1] = hy + uy * along + py * perp;
  world[knee * 3 + 2] = hz + uz * along + pz * perp;
}
