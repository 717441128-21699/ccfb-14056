import { qMultiply, qRotate, type Quat } from '../math/quat';
import type { Vec3 } from '../math/vec3';
import type { SkeletonDefinition } from './types';

export interface FKFrame {
  worldPos: Vec3[];
  worldRot: Quat[];
}

/** Compute one frame of forward kinematics from local joint rotations. */
export function forwardKinematics(
  skel: SkeletonDefinition,
  rootPosition: Vec3,
  localRotations: Quat[],
): FKFrame {
  const n = skel.joints.length;
  const worldPos: Vec3[] = new Array(n);
  const worldRot: Quat[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const j = skel.joints[i];
    if (j.parent === null) {
      worldPos[i] = [rootPosition[0], rootPosition[1], rootPosition[2]];
      worldRot[i] = qNormalizeLocal(localRotations[i]);
    } else {
      const parentRot = worldRot[j.parent];
      const localOff = qRotate(parentRot, j.restOffset);
      const pp = worldPos[j.parent];
      worldPos[i] = [pp[0] + localOff[0], pp[1] + localOff[1], pp[2] + localOff[2]];
      worldRot[i] = qMultiply(parentRot, localRotations[i]);
    }
  }
  return { worldPos, worldRot };
}

function qNormalizeLocal(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return l < 1e-12 ? [0, 0, 0, 1] : [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

/**
 * Rest-pose world position of every joint (identity local rotations).
 */
export function restPose(skel: SkeletonDefinition): FKFrame {
  const n = skel.joints.length;
  const worldPos: Vec3[] = new Array(n);
  const worldRot: Quat[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const j = skel.joints[i];
    worldRot[i] = [0, 0, 0, 1];
    if (j.parent === null) {
      worldPos[i] = [j.restOffset[0], j.restOffset[1], j.restOffset[2]];
    } else {
      const pp = worldPos[j.parent];
      worldPos[i] = [
        pp[0] + j.restOffset[0],
        pp[1] + j.restOffset[1],
        pp[2] + j.restOffset[2],
      ];
    }
  }
  return { worldPos, worldRot };
}
