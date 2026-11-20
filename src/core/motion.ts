/**
 * 程序化动作合成与测试数据生成。
 * 1) 用步态相位解析地生成走/跑/跳的关节世界坐标;
 * 2) 由骨骼朝向框架推世界旋转;
 * 3) FK 反算标记点世界坐标;
 * 4) 可复现地注入高斯噪声、丢点空档与标签互换,并记录真值供测试核验。
 */
import type { MarkerClip, SkeletonClip } from './types.js';
import { FillMethod } from './types.js';
import type { BuiltRig } from './rig.js';
import { JOINT, JOINT_COUNT } from './rig.js';

// ---------- 可复现随机数 ----------

export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller 标准正态采样。 */
function gaussian(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

function rotX(p: Float32Array, po: number, ang: number, out: Float32Array, o: number): void {
  const c = Math.cos(ang), s = Math.sin(ang);
  const x = p[po], y = p[po + 1], z = p[po + 2];
  out[o] = x;
  out[o + 1] = y * c - z * s;
  out[o + 2] = y * s + z * c;
}

// ---------- 关节世界朝向框架 ----------

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
function buildFrame(
  rest: Float32Array, world: Float32Array,
  joint: number, parent: number | null, firstChild: number | null,
  out: Float32Array,
): void {
  const from = firstChild ?? joint;
  const to = firstChild !== null ? firstChild : parent!;
  const r0 = rest[joint * 3], r1 = rest[joint * 3 + 1], r2 = rest[joint * 3 + 2];
  let a0x = rest[from * 3] - r0, a0y = rest[from * 3 + 1] - r1, a0z = rest[from * 3 + 2] - r2;
  let n0 = Math.hypot(a0x, a0y, a0z) || 1;
  a0x /= n0; a0y /= n0; a0z /= n0;
  const w0 = world[joint * 3], w1 = world[joint * 3 + 1], w2 = world[joint * 3 + 2];
  let ax = world[from * 3] - (firstChild !== null ? w0 : world[to * 3]);
  let ay = world[from * 3 + 1] - (firstChild !== null ? w1 : world[to * 3 + 1]);
  let az = world[from * 3 + 2] - (firstChild !== null ? w2 : world[to * 3 + 2]);
  if (firstChild === null) { ax = world[joint * 3] - world[to * 3]; ay = world[joint * 3 + 1] - world[to * 3 + 1]; az = world[joint * 3 + 2] - world[to * 3 + 2]; }
  const n = Math.hypot(ax, ay, az) || 1;
  ax /= n; ay /= n; az /= n;

  // 选与主轴最不平行的参考轴
  let ref = AXES[0];
  let bestDot = 1;
  for (const cand of AXES) {
    const d = Math.abs(a0x * cand[0] + a0y * cand[1] + a0z * cand[2]);
    if (d < bestDot) { bestDot = d; ref = cand; }
  }
  const frame = (axx: number, ayy: number, azz: number, rx: number, ry: number, rz: number): Float32Array => {
    let bx = rx - axx * (rx * axx + ry * ayy + rz * azz);
    let by = ry - ayy * (rx * axx + ry * ayy + rz * azz);
    let bz = rz - azz * (rx * axx + ry * ayy + rz * azz);
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
  // R = M_cur * M_rest^T
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        mCur[r] * mRest[c] + mCur[3 + r] * mRest[3 + c] + mCur[6 + r] * mRest[6 + c];
    }
  }
}

// ---------- 步态参数 ----------

export type Action = 'walk' | 'run' | 'jump';

interface GaitParams {
  cycle: number; // 单步周期(秒)
  speed: number; // 前进速度 m/s
  stanceFrac: number; // 支撑相占比
  bob: number; sway: number; lean: number;
  lift: number; // 摆动脚抬起高度
  armSwing: number; // 手臂前后摆幅(米)
}

const GAITS: Record<'walk' | 'run', GaitParams> = {
  walk: { cycle: 1.1, speed: 1.35, stanceFrac: 0.62, bob: 0.018, sway: 0.025, lean: 0.05, lift: 0.035, armSwing: 0.16 },
  run: { cycle: 0.72, speed: 3.4, stanceFrac: 0.38, bob: 0.045, sway: 0.02, lean: 0.16, lift: 0.14, armSwing: 0.28 },
};

export interface SynthOptions {
  action: Action;
  frameCount: number;
  fps?: number;
  seed?: number;
}

/** 生成一段走/跑/跳骨骼动画(世界坐标 + 世界旋转)。 */
export function synthesize(built: BuiltRig, opts: SynthOptions): SkeletonClip {
  const fps = opts.fps ?? 120;
  const F = opts.frameCount;
  const rest = built.restJointPositions;
  const positions = new Float32Array(F * JOINT_COUNT * 3);
  const rotations = new Float32Array(F * JOINT_COUNT * 9);

  const boneLen = (a: number, b: number): number => {
    return Math.hypot(
      rest[b * 3] - rest[a * 3],
      rest[b * 3 + 1] - rest[a * 3 + 1],
      rest[b * 3 + 2] - rest[a * 3 + 2],
    );
  };
  const thighL = boneLen(JOINT.LUpLeg, JOINT.LLeg);
  const shankL = boneLen(JOINT.LLeg, JOINT.LFoot);
  const maxLeg = thighL + shankL;
  const armReach = boneLen(JOINT.LArm, JOINT.LForeArm) + boneLen(JOINT.LForeArm, JOINT.LHand);

  const firstChild = new Array<number | null>(JOINT_COUNT).fill(null);
  for (let j = 0; j < JOINT_COUNT; j++) {
    const p = built.rig.joints[j].parent;
    if (p !== null && firstChild[p] === null) firstChild[p] = j;
  }

  for (let f = 0; f < F; f++) {
    const t = f / fps;
    const wp = positions.subarray(f * JOINT_COUNT * 3, (f + 1) * JOINT_COUNT * 3);
    if (opts.action === 'jump') {
      placeJumpFrame(built, t, wp, armReach);
    } else {
      placeLocomotionFrame(built, GAITS[opts.action], t, wp, maxLeg, armReach);
    }
    const wr = rotations.subarray(f * JOINT_COUNT * 9, (f + 1) * JOINT_COUNT * 9);
    for (let j = 0; j < JOINT_COUNT; j++) {
      buildFrame(rest, wp, j, built.rig.joints[j].parent, firstChild[j], wr.subarray(j * 9, j * 9 + 9));
    }
  }
  return { frameCount: F, fps, jointPositions: positions, jointRotations: rotations };
}

function setJoint(wp: Float32Array, j: number, x: number, y: number, z: number): void {
  wp[j * 3] = x; wp[j * 3 + 1] = y; wp[j * 3 + 2] = z;
}

function placeLocomotionFrame(
  built: BuiltRig, g: GaitParams, t: number, wp: Float32Array,
  maxLeg: number, armReach: number,
): void {
  const rest = built.restJointPositions;
  const hipBaseY = rest[JOINT.Hips * 3 + 1];
  const phi = (t / g.cycle) % 1;
  const hipZ = g.speed * t;
  const stepLen = g.speed * g.cycle / 2;
  const bob = g.bob * Math.abs(Math.sin(2 * Math.PI * phi));
  const hipY = hipBaseY + bob;
  const hipX = g.sway * Math.sin(2 * Math.PI * phi);
  setJoint(wp, JOINT.Hips, hipX, hipY, hipZ);

  // 上身:前倾 lean,沿静帧偏移做 rotX
  const lean = g.lean;
  const chain = [JOINT.Spine, JOINT.Chest, JOINT.Neck, JOINT.Head];
  for (const j of chain) {
    const p = built.rig.joints[j].parent!;
    const off = new Float32Array(3);
    off[0] = rest[j * 3] - rest[p * 3];
    off[1] = rest[j * 3 + 1] - rest[p * 3 + 1];
    off[2] = rest[j * 3 + 2] - rest[p * 3 + 2];
    const r = new Float32Array(3);
    rotX(off, 0, lean, r, 0);
    setJoint(wp, j, wp[p * 3] + r[0], wp[p * 3 + 1] + r[1], wp[p * 3 + 2] + r[2]);
  }
  // 肩膀相对 Chest 保持静帧偏移(随前倾)
  for (const sh of [JOINT.LShoulder, JOINT.RShoulder]) {
    const p = JOINT.Chest;
    const off = new Float32Array(3);
    off[0] = rest[sh * 3] - rest[p * 3];
    off[1] = rest[sh * 3 + 1] - rest[p * 3 + 1];
    off[2] = rest[sh * 3 + 2] - rest[p * 3 + 2];
    const r = new Float32Array(3);
    rotX(off, 0, lean, r, 0);
    setJoint(wp, sh, wp[p * 3] + r[0], wp[p * 3 + 1] + r[1], wp[p * 3 + 2] + r[2]);
  }

  // 双腿(相位差半周期)
  placeLeg(built, wp, JOINT.LUpLeg, JOINT.LLeg, JOINT.LFoot, JOINT.LToe,
    phi, 1, g, hipZ, stepLen, maxLeg);
  placeLeg(built, wp, JOINT.RUpLeg, JOINT.RLeg, JOINT.RFoot, JOINT.RToe,
    (phi + 0.5) % 1, -1, g, hipZ, stepLen, maxLeg);

  // 双臂:摆动相位与对侧腿一致
  placeArm(built, wp, JOINT.LShoulder, JOINT.LArm, JOINT.LForeArm, JOINT.LHand,
    (phi + 0.5) % 1, 1, g, armReach, lean);
  placeArm(built, wp, JOINT.RShoulder, JOINT.RArm, JOINT.RForeArm, JOINT.RHand,
    phi, -1, g, armReach, lean);
}

function placeLeg(
  built: BuiltRig, wp: Float32Array,
  hipJ: number, kneeJ: number, ankleJ: number, toeJ: number,
  phase: number, side: number, g: GaitParams,
  hipZ: number, stepLen: number, maxLeg: number,
): void {
  const rest = built.restJointPositions;
  const ankleH = rest[JOINT.LFoot * 3 + 1]; // 脚踝离地高度(基准体型下即其 y)
  const hipX = wp[hipJ * 3];
  const hipY = wp[hipJ * 3 + 1];
  // 当前周期该脚的接地点绝对 z
  const cycleIndex = Math.floor((hipZ / g.speed) / g.cycle - phase + 1e-9);
  const hipAtCycleStart = g.speed * (cycleIndex * g.cycle);
  const contactZ = hipAtCycleStart + stepLen / 2;

  let footX = hipX + side * 0.02;
  let footY: number;
  let footZ: number;
  if (phase < g.stanceFrac) {
    // 支撑相:脚固定在接地点
    footZ = contactZ;
    footY = ankleH;
  } else {
    const sw = (phase - g.stanceFrac) / (1 - g.stanceFrac);
    const e = smoothstep(sw);
    footZ = contactZ + stepLen * e;
    footY = ankleH + g.lift * Math.sin(Math.PI * sw);
  }

  // 髋
  setJoint(wp, hipJ, hipX, hipY, hipZ);
  // 膝:髋踝中点 + 前向弯曲,弯曲量由骨长约束
  const dx = footX - hipX, dy = footY - hipY, dz = footZ - hipZ;
  const d = Math.min(Math.hypot(dx, dy, dz), maxLeg * 0.995);
  const bend = 0.5 * Math.sqrt(Math.max(maxLeg * maxLeg - d * d, 0));
  const kneeX = hipX + dx * 0.5 + side * 0.015;
  const kneeY = hipY + dy * 0.5 + bend * 0.25;
  const kneeZ = hipZ + dz * 0.5 + bend;
  setJoint(wp, kneeJ, kneeX, kneeY, kneeZ);
  setJoint(wp, ankleJ, footX, footY, footZ);
  // 脚趾:沿脚方向,摆动相略微上扬
  const toeOff = rest[toeJ * 3 + 2] - rest[ankleJ * 3 + 2];
  const pitch = phase >= g.stanceFrac ? 0.25 * Math.sin(Math.PI * ((phase - g.stanceFrac) / (1 - g.stanceFrac))) : 0;
  setJoint(wp, toeJ, footX, footY + toeOff * Math.sin(pitch), footZ + toeOff * Math.cos(pitch));
}

function placeArm(
  built: BuiltRig, wp: Float32Array,
  shoulderJ: number, armJ: number, elbowJ: number, handJ: number,
  phase: number, side: number, g: GaitParams, armReach: number, lean: number,
): void {
  const rest = built.restJointPositions;
  const sh = wp.slice(shoulderJ * 3, shoulderJ * 3 + 3);
  // 上臂根节点沿肩膀略偏
  setJoint(wp, armJ,
    sh[0] + side * 0.02,
    sh[1] - 0.01,
    sh[2]);
  const swing = g.armSwing * Math.sin(2 * Math.PI * phase);
  // 手臂自然下垂并前后摆动(旋转到下垂姿态:静帧 T-pose 方向 +x/-x)
  const hang = new Float32Array(3);
  rotX(new Float32Array([side * armReach, 0, 0]), 0, Math.PI / 2 - lean, hang, 0);
  const handX = sh[0] + side * 0.03;
  const handY = sh[1] + hang[1] - 0.02;
  const handZ = sh[2] + hang[2] + swing;
  setJoint(wp, handJ, handX, handY, handZ);
  // 肘:肩手连线 55% 处略向下
  const ax = wp[armJ * 3], ay = wp[armJ * 3 + 1], az = wp[armJ * 3 + 2];
  setJoint(wp, elbowJ,
    ax + (handX - ax) * 0.55,
    ay + (handY - ay) * 0.55 - 0.04,
    az + (handZ - az) * 0.55);
  void rest;
}

function placeJumpFrame(built: BuiltRig, t: number, wp: Float32Array, armReach: number): void {
  const rest = built.restJointPositions;
  const duration = 1.6;
  const u = (t % duration) / duration;
  const hipBaseY = rest[JOINT.Hips * 3 + 1];
  const ankleH = rest[JOINT.LFoot * 3 + 1];

  // 髋部高度关键帧:下蹲 -> 起跳 -> 腾空顶点 -> 落地缓冲 -> 回正
  const keys: Array<[number, number]> = [
    [0.0, 0], [0.28, -0.26], [0.42, 0.02], [0.58, 0.42], [0.72, 0.05], [0.82, -0.22], [1.0, 0],
  ];
  let dy = keys[keys.length - 1][1];
  for (let i = 0; i < keys.length - 1; i++) {
    if (u >= keys[i][0] && u <= keys[i + 1][0]) {
      const k = (u - keys[i][0]) / (keys[i + 1][0] - keys[i][0]);
      dy = keys[i][1] + (keys[i + 1][1] - keys[i][1]) * smoothstep(k);
      break;
    }
  }
  const hipY = hipBaseY + dy;
  const hipZ = 0.35 * (u < 0.5 ? smoothstep(u * 2) : 1);
  setJoint(wp, JOINT.Hips, 0, hipY, hipZ);

  // 上身基本竖直,顶点时略后仰
  const lean = u > 0.45 && u < 0.7 ? -0.08 * Math.sin(Math.PI * (u - 0.45) / 0.25) : 0.04;
  for (const j of [JOINT.Spine, JOINT.Chest, JOINT.Neck, JOINT.Head, JOINT.LShoulder, JOINT.RShoulder]) {
    const p = built.rig.joints[j].parent!;
    const off = new Float32Array([
      rest[j * 3] - rest[p * 3],
      rest[j * 3 + 1] - rest[p * 3 + 1],
      rest[j * 3 + 2] - rest[p * 3 + 2],
    ]);
    const r = new Float32Array(3);
    rotX(off, 0, lean, r, 0);
    setJoint(wp, j, wp[p * 3] + r[0], wp[p * 3 + 1] + r[1], wp[p * 3 + 2] + r[2]);
  }

  // 腿:地面时脚踩地,腾空时随身体收起
  const inFlight = u > 0.42 && u < 0.76;
  const tuck = inFlight ? Math.sin(Math.PI * (u - 0.42) / 0.34) : 0;
  for (const [hipJ, kneeJ, ankleJ, toeJ, side] of [
    [JOINT.LUpLeg, JOINT.LLeg, JOINT.LFoot, JOINT.LToe, 1],
    [JOINT.RUpLeg, JOINT.RLeg, JOINT.RFoot, JOINT.RToe, -1],
  ] as const) {
    const footY = inFlight ? ankleH + dy + 0.18 * tuck : ankleH;
    const footZ = hipZ + (inFlight ? 0.12 * tuck : 0);
    const footX = side * (rest[ankleJ * 3] - rest[JOINT.Hips * 3]);
    setJoint(wp, hipJ, side * Math.abs(rest[hipJ * 3]), hipY, hipZ);
    const ddx = footX - wp[hipJ * 3], ddy = footY - hipY, ddz = footZ - hipZ;
    const d = Math.hypot(ddx, ddy, ddz);
    const thighL = Math.hypot(rest[kneeJ * 3] - rest[hipJ * 3], rest[kneeJ * 3 + 1] - rest[hipJ * 3 + 1], rest[kneeJ * 3 + 2] - rest[hipJ * 3 + 2]);
    const shankL = Math.hypot(rest[ankleJ * 3] - rest[kneeJ * 3], rest[ankleJ * 3 + 1] - rest[kneeJ * 3 + 1], rest[ankleJ * 3 + 2] - rest[kneeJ * 3 + 2]);
    const maxLeg = thighL + shankL;
    const bend = 0.5 * Math.sqrt(Math.max(maxLeg * maxLeg - d * d, 0));
    setJoint(wp, kneeJ,
      wp[hipJ * 3] + ddx * 0.5 + side * 0.015,
      hipY + ddy * 0.5 + bend * 0.25,
      hipZ + ddz * 0.5 + bend);
    setJoint(wp, ankleJ, footX, footY, footZ);
    const toeOff = rest[toeJ * 3 + 2] - rest[ankleJ * 3 + 2];
    setJoint(wp, toeJ, footX, footY, footZ + toeOff);
  }

  // 手臂:下蹲后摆、起跳上举、落地前挥
  const armPhase = u < 0.42 ? -0.4 * smoothstep(u / 0.42)
    : u < 0.6 ? -0.4 + 0.9 * smoothstep((u - 0.42) / 0.18)
      : 0.5 - 0.7 * smoothstep(Math.min(1, (u - 0.6) / 0.4));
  for (const [shoulderJ, armJ, elbowJ, handJ, side] of [
    [JOINT.LShoulder, JOINT.LArm, JOINT.LForeArm, JOINT.LHand, 1],
    [JOINT.RShoulder, JOINT.RArm, JOINT.RForeArm, JOINT.RHand, -1],
  ] as const) {
    const sh = wp.slice(shoulderJ * 3, shoulderJ * 3 + 3);
    setJoint(wp, armJ, sh[0] + side * 0.02, sh[1] - 0.01, sh[2]);
    const ang = Math.PI / 2 - armPhase; // 越大越上举
    const dir = new Float32Array(3);
    rotX(new Float32Array([side * armReach, 0, 0]), 0, ang, dir, 0);
    const hx = sh[0] + side * 0.03, hy = sh[1] + dir[1], hz = sh[2] + dir[2];
    setJoint(wp, handJ, hx, hy, hz);
    const ax = wp[armJ * 3], ay = wp[armJ * 3 + 1], az = wp[armJ * 3 + 2];
    setJoint(wp, elbowJ, ax + (hx - ax) * 0.55, ay + (hy - ay) * 0.55 - 0.03, az + (hz - az) * 0.55);
  }
}

// ---------- FK 反算标记点 ----------

/**
 * 由骨骼动画反算标记点:marker = jointPos + R_joint * localOffset。
 * 返回全观测 MarkerClip(observed 全 1)。
 */
export function markersFromSkeleton(built: BuiltRig, skel: SkeletonClip): MarkerClip {
  const M = built.markerNames.length;
  const F = skel.frameCount;
  const positions = new Float32Array(F * M * 3);
  const observed = new Uint8Array(F * M).fill(1);
  const { rig, markerBody } = built;

  for (let f = 0; f < F; f++) {
    const jp = skel.jointPositions.subarray(f * JOINT_COUNT * 3);
    const jr = skel.jointRotations.subarray(f * JOINT_COUNT * 9);
    for (let m = 0; m < M; m++) {
      const b = markerBody[m];
      const j = rig.bodyToJoint[b];
      const body = rig.bodies[b];
      const localIdx = body.markerIndices.indexOf(m);
      const lo = localIdx * 3;
      const lx = body.localPositions[lo], ly = body.localPositions[lo + 1], lz = body.localPositions[lo + 2];
      const ro = j * 9, po = j * 3, mo = (f * M + m) * 3;
      positions[mo] = jp[po] + jr[ro] * lx + jr[ro + 1] * ly + jr[ro + 2] * lz;
      positions[mo + 1] = jp[po + 1] + jr[ro + 3] * lx + jr[ro + 4] * ly + jr[ro + 5] * lz;
      positions[mo + 2] = jp[po + 2] + jr[ro + 6] * lx + jr[ro + 7] * ly + jr[ro + 8] * lz;
    }
  }
  return { markerNames: [...built.markerNames], positions, observed, frameCount: F, fps: skel.fps };
}

// ---------- 数据污染 ----------

export interface CorruptOptions {
  /** 高斯噪声标准差(米)。 */
  noiseSigma?: number;
  /** 目标丢点比例(0~1)。 */
  missingRate?: number;
  /** 互换事件数量。 */
  swapCount?: number;
  seed?: number;
}

export interface CorruptionTruth {
  /** 真值空档 [marker, start, end][]。 */
  gaps: Array<{ marker: number; start: number; end: number }>;
  /** 真值互换 [startFrame, endFrame, markerA, markerB][]。 */
  swaps: Array<{ start: number; end: number; a: number; b: number }>;
  /** 注入噪声前的干净标记点数据(用于误差统计)。 */
  cleanPositions: Float32Array;
}

export interface CorruptResult {
  clip: MarkerClip;
  truth: CorruptionTruth;
}

/** 向干净标记点数据注入噪声、丢点与标签互换(原地复制后修改,不污染输入)。 */
export function corruptClip(source: MarkerClip, opts: CorruptOptions = {}): CorruptResult {
  const sigma = opts.noiseSigma ?? 0.0012;
  const missingRate = opts.missingRate ?? 0.12;
  const swapCount = opts.swapCount ?? Math.max(2, Math.floor(source.frameCount / 300));
  const rng = makeRng(opts.seed ?? 42);

  const F = source.frameCount, M = source.markerNames.length;
  const positions = new Float32Array(source.positions);
  const observed = new Uint8Array(source.observed);
  const cleanPositions = new Float32Array(source.positions);

  // 1) 高斯噪声
  if (sigma > 0) {
    for (let i = 0; i < positions.length; i++) positions[i] += gaussian(rng) * sigma;
  }

  // 2) 丢点空档:按标记点独立采样。真实光学动捕以 1~6 帧短档为主,
  //    长档(遮挡/出画)罕见,这里以 3% 概率生成、上限 20 帧。
  const gaps: CorruptionTruth['gaps'] = [];
  const budget = Math.floor(F * M * missingRate);
  let dropped = 0;
  let guard = 0;
  while (dropped < budget && guard++ < budget * 10 + 100) {
    const m = Math.floor(rng() * M);
    const longGap = rng() < 0.03;
    const len = longGap ? 8 + Math.floor(rng() * 13) : 1 + Math.floor(rng() * 6);
    const start = 3 + Math.floor(rng() * Math.max(1, F - len - 6));
    const end = Math.min(F - 4, start + len - 1);
    let added = 0;
    for (let f = start; f <= end; f++) {
      const idx = f * M + m;
      if (observed[idx]) { observed[idx] = 0; positions[idx * 3] = NaN; positions[idx * 3 + 1] = NaN; positions[idx * 3 + 2] = NaN; added++; }
    }
    if (added > 0) gaps.push({ marker: m, start, end });
    dropped += added;
  }

  // 3) 标签互换:选两个不同刚体上的标记点,在连续帧区间交换坐标
  const swaps: CorruptionTruth['swaps'] = [];
  for (let s = 0; s < swapCount; s++) {
    const len = 8 + Math.floor(rng() * 38);
    const start = 5 + Math.floor(rng() * Math.max(1, F - len - 10));
    const end = start + len - 1;
    let a = -1, b = -1;
    for (let tries = 0; tries < 30 && a < 0; tries++) {
      const ca = Math.floor(rng() * M), cb = Math.floor(rng() * M);
      if (ca === cb) continue;
      const conflict = swaps.some(
        (ev) => start <= ev.end && end >= ev.start && (ev.a === ca || ev.b === ca || ev.a === cb || ev.b === cb),
      );
      if (!conflict) { a = ca; b = cb; }
    }
    if (a < 0) continue;
    swaps.push({ start, end, a, b });
    for (let f = start; f <= end; f++) {
      const ia = f * M + a, ib = f * M + b;
      if (!observed[ia] || !observed[ib]) continue;
      for (let k = 0; k < 3; k++) {
        const tmp = positions[ia * 3 + k];
        positions[ia * 3 + k] = positions[ib * 3 + k];
        positions[ib * 3 + k] = tmp;
      }
    }
  }

  return {
    clip: { markerNames: [...source.markerNames], positions, observed, frameCount: F, fps: source.fps },
    truth: { gaps, swaps, cleanPositions },
  };
}

/** 统计补点结果相对干净真值的误差(仅在丢点帧上统计),返回毫米级统计。 */
export function fillErrorStats(
  filled: MarkerClip, truth: CorruptionTruth,
): { rmsMm: number; maxMm: number; p95Mm: number; count: number } {
  const errs: number[] = [];
  const M = filled.markerNames.length;
  for (const g of truth.gaps) {
    for (let f = g.start; f <= g.end; f++) {
      const io = (f * M + g.marker) * 3;
      const dx = filled.positions[io] - truth.cleanPositions[io];
      const dy = filled.positions[io + 1] - truth.cleanPositions[io + 1];
      const dz = filled.positions[io + 2] - truth.cleanPositions[io + 2];
      errs.push(Math.hypot(dx, dy, dz) * 1000);
    }
  }
  errs.sort((a, b) => a - b);
  const rms = Math.sqrt(errs.reduce((s, e) => s + e * e, 0) / (errs.length || 1));
  return {
    rmsMm: rms,
    maxMm: errs[errs.length - 1] ?? 0,
    p95Mm: errs[Math.floor(errs.length * 0.95)] ?? 0,
    count: errs.length,
  };
}

export { FillMethod };
