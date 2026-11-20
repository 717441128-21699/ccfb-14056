import type { Quat } from '../math/quat';
import type { Vec3 } from '../math/vec3';

/** A contiguous interval of frames [start, end] (inclusive). */
export interface FrameRange {
  start: number;
  end: number;
}

export interface ClusterDef {
  name: string;
  /** Marker names belonging to this rigid body (3+ recommended). */
  markers: string[];
  /** Joint whose world transform this cluster is rigidly attached to. */
  anchorJoint: string;
}

export interface MarkerBinding {
  name: string;
  cluster: string;
  /** Position of the marker in the skeleton rest pose, world coordinates (m). */
  restPosition: Vec3;
}

/** Frame-major marker data: positions[f * markerCount * 3 + m * 3 + c]. */
export interface MarkerClip {
  fps: number;
  markerNames: string[];
  frameCount: number;
  /** xyz per marker per frame; NaN when the marker is missing. */
  positions: Float32Array;
  /** 1 = observed, 0 = missing (drop-out). */
  valid: Uint8Array;
  clusters: ClusterDef[];
  /** Rest-pose marker bindings, used by the solver / synthetic generator. */
  bindings: MarkerBinding[];
}

export type GapFillMethod = 'rigid' | 'trajectory' | 'edge';

export interface GapRecord {
  marker: string;
  markerIndex: number;
  start: number;
  end: number;
  length: number;
  method: GapFillMethod;
  /** Mean |filled - truth| in mm when ground truth is available. */
  meanErrorMm?: number;
  maxErrorMm?: number;
}

export interface SwapRecord {
  frame: number;
  start: number;
  end: number;
  markerA: string;
  markerB: string;
}

export interface CleanReport {
  frameCount: number;
  markerCount: number;
  gaps: GapRecord[];
  swaps: SwapRecord[];
  /** Per-frame max cluster fitting residual in mm (after cleaning). */
  maxResidualMm: number;
  meanResidualMm: number;
  durationMs: number;
}

export interface ContactInterval {
  side: 'L' | 'R';
  start: number;
  end: number;
}

export interface PenetrationEvent {
  frame: number;
  boneA: string;
  boneB: string;
  /** Deepest penetration distance in mm (negative = overlapping). */
  depthMm: number;
  point: Vec3;
}

export interface JointDef {
  name: string;
  parent: number | null;
  /** Rest-pose offset relative to the parent joint, metres. */
  restOffset: Vec3;
  /** Capsule radius for penetration tests, metres. */
  radius: number;
}

export interface SkeletonDefinition {
  joints: JointDef[];
  /** Index lookup built from joints. */
  jointIndex: Map<string, number>;
  clusters: ClusterDef[];
  bindings: MarkerBinding[];
}

/** Per-frame solved skeleton: joint world positions and rotations. */
export interface SkeletonPose {
  positions: Float32Array; // frameCount * jointCount * 3
  rotations: Float32Array; // frameCount * jointCount * 4 (quat xyzw)
}

export interface SolvedSkeleton {
  definition: SkeletonDefinition;
  fps: number;
  frameCount: number;
  poses: SkeletonPose;
  contacts: ContactInterval[];
  penetrations: PenetrationEvent[];
}

export type MotionKind = 'walk' | 'run' | 'jump';

export interface CorruptionSpec {
  /** Gaussian measurement noise standard deviation, metres. */
  noiseStd: number;
  /** Probability that any given marker starts a gap on a frame. */
  gapRate: number;
  /** Maximum gap length in frames. */
  maxGapLength: number;
  /** Number of label-swap events to inject over the clip. */
  swapCount: number;
  /** Maximum duration of a swap in frames. */
  maxSwapLength: number;
  seed: number;
}

export const DEFAULT_CORRUPTION: CorruptionSpec = {
  noiseStd: 0.0015,
  gapRate: 0.0009,
  maxGapLength: 24,
  swapCount: 14,
  maxSwapLength: 30,
  seed: 20261120,
};

export function quatToArray(q: Quat): [number, number, number, number] {
  return [q[0], q[1], q[2], q[3]];
}
