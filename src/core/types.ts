/** 核心数据类型定义 —— 浏览器与 Node 双端共享 */

/** 三维向量以长度 3 的普通数组/元组传递,批量数据用扁平 Float32Array 存储。 */
export type Vec3 = [number, number, number];

/**
 * 标记点片段(光学动捕原始数据)。
 *
 * positions: 长度 frameCount * markerCount * 3 的扁平数组。
 *   丢点位置写入 NaN;observed[f*M+m] 为 1 表示该帧该点有效。
 * markerNames: 与标记点索引一一对应。
 * fps: 采样率(每秒帧数)。
 * frameInterval: 由 fps 推导,单位秒。
 */
export interface MarkerClip {
  markerNames: string[];
  positions: Float32Array;
  observed: Uint8Array;
  frameCount: number;
  fps: number;
}

/** 一段连续丢点空档。 */
export interface Gap {
  markerIndex: number;
  startFrame: number;
  endFrame: number; // 含端点,即 [start, end] 全部为丢点
  length: number;
}

/** 标记点标签互换事件(两标记点轨迹在某帧起交换了身份)。 */
export interface SwapEvent {
  frame: number;
  markerIndexA: number;
  markerIndexB: number;
  /** 互换前两标签预测位置与实测位置的归一化残差,用于置信度评估。 */
  residualBefore: number;
  residualAfter: number;
}

/**
 * 刚体定义:一组附着在同一刚性肢体段上的标记点。
 * localPositions: 模板帧中各标记点相对刚体质心的局部坐标(含质心平移约定见 rigidBody.ts)。
 */
export interface RigidBodyDef {
  name: string;
  markerIndices: number[];
  localPositions: Float32Array; // markerIndices.length * 3
}

/** 单根骨骼的层级定义。 */
export interface JointDef {
  name: string;
  parent: number | null;
  /** 相对父关节的静止偏移(米)。 */
  restOffset: Vec3;
}

/** 人体骨架模板。 */
export interface Rig {
  joints: JointDef[];
  /** 关节名 -> 索引。 */
  jointIndex: ReadonlyMap<string, number>;
  /** 每个刚体负责驱动的关节索引(刚体旋转直接作为该关节世界朝向的约束)。 */
  bodyToJoint: number[];
  bodies: RigidBodyDef[];
}

/** 解算后的骨骼动画:每帧每个关节的世界位置与世界旋转(3x3,行主序,9 个数)。 */
export interface SkeletonClip {
  frameCount: number;
  fps: number;
  jointPositions: Float32Array; // F*J*3
  jointRotations: Float32Array; // F*J*9
}

/** 足部接触锁定记录。 */
export interface FootLock {
  side: 'left' | 'right';
  startFrame: number;
  endFrame: number;
  position: Vec3; // 锁定的脚踝世界坐标(xz 锁定,y 取接触高度)
}

/** 穿模/穿透问题标注。 */
export interface PenetrationIssue {
  frame: number;
  jointA: number;
  jointB: number;
  depth: number; // 穿透深度(米),正数表示穿透
}

/** 清洗管线产出的完整结果。 */
export interface CleanResult {
  cleaned: MarkerClip;
  gaps: Gap[];
  swaps: SwapEvent[];
  /** 每个补点帧逐点的补点来源标记。 */
  fillMethod: Uint8Array; // F*M,取值见 FillMethod
  skeleton: SkeletonClip;
  footLocks: FootLock[];
  issues: PenetrationIssue[];
  /** 清洗耗时(毫秒)。 */
  elapsedMs: number;
}

export const enum FillMethod {
  Observed = 0,
  RigidBody = 1,
  Trajectory = 2,
  /** 刚体时间先验:观测点不足 3 个时沿用邻近可靠位姿,精度低于直接刚体配准。 */
  Prior = 3,
}
