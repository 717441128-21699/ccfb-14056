/**
 * 人体骨架与标记点布局模板。
 * 21 关节层级 + 13 个标记点刚体(共 54 个光学标记点),T-pose 静帧坐标(米,y 轴向上,面朝 +z)。
 * 刚体局部坐标相对其驱动关节原点;同一身材下标记点与骨架严格刚性绑定。
 */
import type { Rig, RigidBodyDef, Vec3 } from './types.js';

export interface CharacterDimensions {
  /** 身高(米)。 */
  height: number;
  /** 各部位相对基准(1.75m 体型)的长度比例,1 为基准。 */
  thighMul: number;
  shankMul: number;
  torsoMul: number;
  armMul: number;
  shoulderWidthMul: number;
  hipWidthMul: number;
}

export const DEFAULT_DIMS: CharacterDimensions = {
  height: 1.75,
  thighMul: 1, shankMul: 1, torsoMul: 1, armMul: 1,
  shoulderWidthMul: 1, hipWidthMul: 1,
};

/** 几个预设身材,用于重定向预览。 */
export const DIM_PRESETS: Record<string, CharacterDimensions> = {
  standard: DEFAULT_DIMS,
  tall: { height: 1.90, thighMul: 1.08, shankMul: 1.1, torsoMul: 1.02, armMul: 1.06, shoulderWidthMul: 1.05, hipWidthMul: 1.02 },
  short: { height: 1.55, thighMul: 0.92, shankMul: 0.9, torsoMul: 0.95, armMul: 0.94, shoulderWidthMul: 0.95, hipWidthMul: 0.98 },
  broad: { height: 1.78, thighMul: 1.02, shankMul: 1, torsoMul: 1.05, armMul: 1.02, shoulderWidthMul: 1.2, hipWidthMul: 1.15 },
};

// 关节名(索引即关节 id)
export const JOINT = {
  Hips: 0, Spine: 1, Chest: 2, Neck: 3, Head: 4,
  LShoulder: 5, LArm: 6, LForeArm: 7, LHand: 8,
  RShoulder: 9, RArm: 10, RForeArm: 11, RHand: 12,
  LUpLeg: 13, LLeg: 14, LFoot: 15, LToe: 16,
  RUpLeg: 17, RLeg: 18, RFoot: 19, RToe: 20,
} as const;

export const JOINT_COUNT = 21;
export const JOINT_NAMES = [
  'Hips', 'Spine', 'Chest', 'Neck', 'Head',
  'LShoulder', 'LArm', 'LForeArm', 'LHand',
  'RShoulder', 'RArm', 'RForeArm', 'RHand',
  'LUpLeg', 'LLeg', 'LFoot', 'LToe',
  'RUpLeg', 'RLeg', 'RFoot', 'RToe',
];

interface BodySpec {
  name: string;
  drivenJoint: number;
  /** 标记点名与相对驱动关节的局部偏移(基准体型)。 */
  markers: Array<{ name: string; offset: Vec3 }>;
}

const s = (x: number, y: number, z: number): Vec3 => [x, y, z];

function limbMarkers(prefix: string, dir: Vec3, len: number, rad: number): BodySpec['markers'] {
  // 沿肢体方向均匀布 3~4 个点,周向交替偏移保证非共线
  const [dx, dy, dz] = dir;
  const u: Vec3 = [dx, dy, dz]; // 轴向单位向量(调用处给单位向量)
  // 构造两个垂直方向
  const ref: Vec3 = Math.abs(u[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const v1 = cross3(u, ref);
  const v2 = cross3(u, v1);
  const pts: Array<[number, number, number]> = [
    [0.25, 1, 0], [0.5, -0.5, 0.866], [0.78, -0.5, -0.866], [0.95, 1, 0],
  ];
  return pts.map(([t, a, b], i) => ({
    name: `${prefix}_M${i + 1}`,
    offset: s(
      u[0] * len * t + (v1[0] * a + v2[0] * b) * rad,
      u[1] * len * t + (v1[1] * a + v2[1] * b) * rad,
      u[2] * len * t + (v1[2] * a + v2[2] * b) * rad,
    ),
  }));
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm3(a: Vec3): Vec3 {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}

/** 基准体型下的刚体/标记点规格。 */
function bodySpecs(): BodySpec[] {
  return [
    {
      name: 'Pelvis', drivenJoint: JOINT.Hips,
      markers: [
        { name: 'PEL_LASIS', offset: s(0.10, 0.02, 0.11) },
        { name: 'PEL_RASIS', offset: s(-0.10, 0.02, 0.11) },
        { name: 'PEL_LPSIS', offset: s(0.09, -0.03, -0.10) },
        { name: 'PEL_RPSIS', offset: s(-0.09, -0.03, -0.10) },
      ],
    },
    {
      name: 'Chest', drivenJoint: JOINT.Chest,
      markers: [
        { name: 'CHEST_LFR', offset: s(0.09, 0.06, 0.09) },
        { name: 'CHEST_RFR', offset: s(-0.09, 0.06, 0.09) },
        { name: 'CHEST_LBK', offset: s(0.09, 0.05, -0.09) },
        { name: 'CHEST_RBK', offset: s(-0.09, 0.05, -0.09) },
      ],
    },
    {
      name: 'Head', drivenJoint: JOINT.Head,
      markers: [
        { name: 'HEAD_L', offset: s(0.08, 0.02, 0) },
        { name: 'HEAD_R', offset: s(-0.08, 0.02, 0) },
        { name: 'HEAD_FR', offset: s(0, 0.05, 0.09) },
        { name: 'HEAD_BK', offset: s(0, 0.05, -0.09) },
      ],
    },
    {
      name: 'LUpperArm', drivenJoint: JOINT.LArm,
      markers: limbMarkers('LUA', norm3([1, 0, 0]), 0.27, 0.045),
    },
    {
      name: 'RUpperArm', drivenJoint: JOINT.RArm,
      markers: limbMarkers('RUA', norm3([-1, 0, 0]), 0.27, 0.045),
    },
    {
      name: 'LForeArm', drivenJoint: JOINT.LForeArm,
      markers: limbMarkers('LFA', norm3([1, 0, 0]), 0.25, 0.04),
    },
    {
      name: 'RForeArm', drivenJoint: JOINT.RForeArm,
      markers: limbMarkers('RFA', norm3([-1, 0, 0]), 0.25, 0.04),
    },
    {
      name: 'LHand', drivenJoint: JOINT.LHand,
      markers: [
        { name: 'LHND_DOR', offset: s(0.05, 0.01, 0) },
        { name: 'LHND_LAT', offset: s(0.02, 0, 0.04) },
        { name: 'LHND_MED', offset: s(0.02, 0, -0.04) },
        { name: 'LHND_WRI', offset: s(0.0, 0.03, 0) },
      ],
    },
    {
      name: 'RHand', drivenJoint: JOINT.RHand,
      markers: [
        { name: 'RHND_DOR', offset: s(-0.05, 0.01, 0) },
        { name: 'RHND_LAT', offset: s(-0.02, 0, -0.04) },
        { name: 'RHND_MED', offset: s(-0.02, 0, 0.04) },
        { name: 'RHND_WRI', offset: s(0.0, 0.03, 0) },
      ],
    },
    {
      name: 'LThigh', drivenJoint: JOINT.LUpLeg,
      markers: limbMarkers('LTH', norm3([0, -1, 0]), 0.42, 0.06),
    },
    {
      name: 'RThigh', drivenJoint: JOINT.RUpLeg,
      markers: limbMarkers('RTH', norm3([0, -1, 0]), 0.42, 0.06),
    },
    {
      name: 'LShank', drivenJoint: JOINT.LLeg,
      markers: limbMarkers('LSH', norm3([0, -1, 0]), 0.43, 0.05),
    },
    {
      name: 'RShank', drivenJoint: JOINT.RLeg,
      markers: limbMarkers('RSH', norm3([0, -1, 0]), 0.43, 0.05),
    },
    {
      name: 'LFoot', drivenJoint: JOINT.LFoot,
      markers: [
        { name: 'LFT_HEEL', offset: s(0, 0.02, -0.08) },
        { name: 'LFT_TOE', offset: s(0, 0.02, 0.20) },
        { name: 'LFT_OUT', offset: s(0.06, 0.07, 0.03) },
        { name: 'LFT_IN', offset: s(-0.06, 0.07, 0.03) },
      ],
    },
    {
      name: 'RFoot', drivenJoint: JOINT.RFoot,
      markers: [
        { name: 'RFT_HEEL', offset: s(0, 0.02, -0.08) },
        { name: 'RFT_TOE', offset: s(0, 0.02, 0.20) },
        { name: 'RFT_OUT', offset: s(-0.06, 0.07, 0.03) },
        { name: 'RFT_IN', offset: s(0.06, 0.07, 0.03) },
      ],
    },
  ];
}

export interface BuiltRig {
  rig: Rig;
  /** T-pose 各关节世界坐标 J*3。 */
  restJointPositions: Float32Array;
  /** T-pose 各标记点世界坐标 M*3。 */
  restMarkerPositions: Float32Array;
  /** 每个标记点属于哪个刚体。 */
  markerBody: Uint16Array;
  markerNames: string[];
}

/**
 * 按身材参数生成骨架与标记点布局。
 * 整体随身高缩放,四肢/躯干长度与宽度再乘各自比例。
 */
export function buildRig(dims: CharacterDimensions = DEFAULT_DIMS): BuiltRig {
  const baseH = 1.75;
  const overall = dims.height / baseH;
  const specs = bodySpecs();

  // 1) 关节层级与偏移(基准偏移手工指定,再乘比例)
  const hipW = 0.10 * overall * dims.hipWidthMul;
  const shW = 0.18 * overall * dims.shoulderWidthMul;
  const torso = overall * dims.torsoMul;
  const arm = overall * dims.armMul;
  const thigh = overall * dims.thighMul;
  const shank = overall * dims.shankMul;

  const hipY = 0.95 * overall;
  const defs: Array<{ name: string; parent: number | null; offset: Vec3 }> = [
    { name: 'Hips', parent: null, offset: s(0, hipY, 0) },
    { name: 'Spine', parent: JOINT.Hips, offset: s(0, 0.20 * torso, 0) },
    { name: 'Chest', parent: JOINT.Spine, offset: s(0, 0.20 * torso, 0) },
    { name: 'Neck', parent: JOINT.Chest, offset: s(0, 0.15 * torso, 0) },
    { name: 'Head', parent: JOINT.Neck, offset: s(0, 0.10 * torso, 0) },
    { name: 'LShoulder', parent: JOINT.Chest, offset: s(shW * 0.4, 0.05 * torso, 0) },
    { name: 'LArm', parent: JOINT.LShoulder, offset: s(shW * 0.6, 0, 0) },
    { name: 'LForeArm', parent: JOINT.LArm, offset: s(0.27 * arm, 0, 0) },
    { name: 'LHand', parent: JOINT.LForeArm, offset: s(0.25 * arm, 0, 0) },
    { name: 'RShoulder', parent: JOINT.Chest, offset: s(-shW * 0.4, 0.05 * torso, 0) },
    { name: 'RArm', parent: JOINT.RShoulder, offset: s(-shW * 0.6, 0, 0) },
    { name: 'RForeArm', parent: JOINT.RArm, offset: s(-0.27 * arm, 0, 0) },
    { name: 'RHand', parent: JOINT.RForeArm, offset: s(-0.25 * arm, 0, 0) },
    { name: 'LUpLeg', parent: JOINT.Hips, offset: s(hipW, -0.05 * overall, 0) },
    { name: 'LLeg', parent: JOINT.LUpLeg, offset: s(0, -0.42 * thigh, 0) },
    { name: 'LFoot', parent: JOINT.LLeg, offset: s(0, -0.43 * shank, 0) },
    { name: 'LToe', parent: JOINT.LFoot, offset: s(0, 0, 0.18 * overall) },
    { name: 'RUpLeg', parent: JOINT.Hips, offset: s(-hipW, -0.05 * overall, 0) },
    { name: 'RLeg', parent: JOINT.RUpLeg, offset: s(0, -0.42 * thigh, 0) },
    { name: 'RFoot', parent: JOINT.RLeg, offset: s(0, -0.43 * shank, 0) },
    { name: 'RToe', parent: JOINT.RFoot, offset: s(0, 0, 0.18 * overall) },
  ];

  const restJointPositions = new Float32Array(JOINT_COUNT * 3);
  for (let j = 0; j < JOINT_COUNT; j++) {
    const d = defs[j];
    if (d.parent === null) {
      restJointPositions.set(d.offset, j * 3);
    } else {
      const po = d.parent * 3;
      restJointPositions[j * 3] = restJointPositions[po] + d.offset[0];
      restJointPositions[j * 3 + 1] = restJointPositions[po + 1] + d.offset[1];
      restJointPositions[j * 3 + 2] = restJointPositions[po + 2] + d.offset[2];
    }
  }

  // 2) 刚体与标记点:按所在部位做长度/宽度缩放
  const bodies: RigidBodyDef[] = [];
  const markerNames: string[] = [];
  const markerBody: Uint16Array = new Uint16Array(specs.reduce((n, b) => n + b.markers.length, 0));
  const restMarkerPositions = new Float32Array(markerBody.length * 3);
  const jointIndex = new Map(JOINT_NAMES.map((n, i) => [n, i]));

  let cursor = 0;
  specs.forEach((spec, b) => {
    const indices: number[] = [];
    const local = new Float32Array(spec.markers.length * 3);
    spec.markers.forEach((mk, k) => {
      const idx = cursor++;
      indices.push(idx);
      markerNames.push(mk.name);
      markerBody[idx] = b;
      const sc = markerScale(spec.drivenJoint, dims, overall);
      const lo = k * 3;
      local[lo] = mk.offset[0] * sc.width;
      local[lo + 1] = mk.offset[1] * sc.length;
      local[lo + 2] = mk.offset[2] * sc.width;
      // T-pose 下标记点世界坐标 = 驱动关节世界坐标 + 局部偏移
      const jo = spec.drivenJoint * 3;
      restMarkerPositions[idx * 3] = restJointPositions[jo] + local[lo];
      restMarkerPositions[idx * 3 + 1] = restJointPositions[jo + 1] + local[lo + 1];
      restMarkerPositions[idx * 3 + 2] = restJointPositions[jo + 2] + local[lo + 2];
    });
    bodies.push({ name: spec.name, markerIndices: indices, localPositions: local });
  });

  const bodyToJoint = specs.map((b) => b.drivenJoint);
  const rig: Rig = {
    joints: defs.map((d) => ({ name: d.name, parent: d.parent, restOffset: d.offset })),
    jointIndex,
    bodyToJoint,
    bodies,
  };
  return { rig, restJointPositions, restMarkerPositions, markerBody, markerNames };
}

function markerScale(joint: number, dims: CharacterDimensions, overall: number): { length: number; width: number } {
  const leg = joint >= JOINT.LUpLeg;
  const arm = joint === JOINT.LArm || joint === JOINT.RArm || joint === JOINT.LForeArm || joint === JOINT.RForeArm;
  if (leg) {
    const thigh = joint === JOINT.LUpLeg || joint === JOINT.RUpLeg;
    return { length: overall * (thigh ? dims.thighMul : dims.shankMul), width: overall * dims.hipWidthMul };
  }
  if (arm) return { length: overall * dims.armMul, width: overall * dims.shoulderWidthMul };
  if (joint === JOINT.Hips) return { length: overall * dims.torsoMul, width: overall * dims.hipWidthMul };
  if (joint === JOINT.Chest) return { length: overall * dims.torsoMul, width: overall * dims.shoulderWidthMul };
  return { length: overall, width: overall };
}
