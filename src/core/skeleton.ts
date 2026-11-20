import type { Vec3 } from '../math/vec3';
import type { ClusterDef, JointDef, MarkerBinding, SkeletonDefinition } from './types';

/**
 * Standard 23-joint humanoid skeleton in a T-pose, Y up, facing -Z.
 * Total standing height ≈ 1.75 m. Offsets are relative to the parent joint.
 */
function buildJoints(): { joints: JointDef[]; index: Map<string, number> } {
  const raw: Array<[name: string, parent: string | null, offset: Vec3, radius: number]> = [
    ['pelvis', null, [0, 0.95, 0], 0.09],
    ['spine1', 'pelvis', [0, 0.12, 0], 0.075],
    ['spine2', 'spine1', [0, 0.13, 0], 0.07],
    ['neck', 'spine2', [0, 0.14, 0], 0.055],
    ['head', 'neck', [0, 0.10, 0], 0.09],
    ['lCollar', 'spine2', [0.04, 0.06, 0], 0.05],
    ['lShoulder', 'lCollar', [0.10, 0, 0], 0.05],
    ['lElbow', 'lShoulder', [0.28, 0, 0], 0.038],
    ['lWrist', 'lElbow', [0.25, 0, 0], 0.03],
    ['lHand', 'lWrist', [0.09, 0, 0], 0.035],
    ['rCollar', 'spine2', [-0.04, 0.06, 0], 0.05],
    ['rShoulder', 'rCollar', [-0.10, 0, 0], 0.05],
    ['rElbow', 'rShoulder', [-0.28, 0, 0], 0.038],
    ['rWrist', 'rElbow', [-0.25, 0, 0], 0.03],
    ['rHand', 'rWrist', [-0.09, 0, 0], 0.035],
    ['lHip', 'pelvis', [0.10, -0.04, 0], 0.06],
    ['lKnee', 'lHip', [0, -0.43, 0], 0.05],
    ['lAnkle', 'lKnee', [0, -0.42, 0], 0.042],
    ['lFoot', 'lAnkle', [0, -0.03, 0.07], 0.04],
    ['lToe', 'lFoot', [0, 0, 0.12], 0.03],
    ['rHip', 'pelvis', [-0.10, -0.04, 0], 0.06],
    ['rKnee', 'rHip', [0, -0.43, 0], 0.05],
    ['rAnkle', 'rKnee', [0, -0.42, 0], 0.042],
    ['rFoot', 'rAnkle', [0, -0.03, 0.07], 0.04],
    ['rToe', 'rFoot', [0, 0, 0.12], 0.03],
  ];
  // Two-phase build: the name index must exist before parent names resolve.
  const index = new Map<string, number>();
  raw.forEach(([name], i) => index.set(name, i));
  const joints: JointDef[] = raw.map(([name, parent, offset, radius]) => ({
    name,
    parent: parent === null ? null : index.get(parent)!,
    restOffset: offset,
    radius,
  }));
  return { joints, index };
}

/**
 * 42 optical markers grouped into 13 rigid clusters.
 * Rest positions are defined in the anchor joint's local frame; the binding
 * table stores the resolved rest-pose world position.
 * Local marker: [side, alongBone, front/back] metres.
 */
function buildClustersAndBindings(jointIndex: Map<string, number>, restWorld: Vec3[]): {
  clusters: ClusterDef[];
  bindings: MarkerBinding[];
} {
  // [markerName, clusterName, anchorJoint, localOffset]
  const table: Array<[string, string, string, Vec3]> = [
    // Pelvis cluster — 6 markers around the hip belt
    ['PELV_L', 'pelvis', 'pelvis', [0.11, 0.03, 0.0]],
    ['PELV_R', 'pelvis', 'pelvis', [-0.11, 0.03, 0.0]],
    ['SACR', 'pelvis', 'pelvis', [0.0, 0.02, -0.10]],
    ['PELV_F', 'pelvis', 'pelvis', [0.0, 0.0, 0.12]],
    ['WAIST_L', 'pelvis', 'pelvis', [0.12, 0.09, -0.02]],
    ['WAIST_R', 'pelvis', 'pelvis', [-0.12, 0.09, -0.02]],
    // Torso cluster — 4 markers spanning left/right, up/down and front/back
    ['STER', 'torso', 'spine2', [0.06, 0.05, 0.10]],
    ['XIPH', 'torso', 'spine2', [-0.06, -0.03, 0.09]],
    ['T7', 'torso', 'spine2', [0.05, 0.06, -0.10]],
    ['T10', 'torso', 'spine2', [-0.05, -0.05, -0.09]],
    // Head cluster — tetrahedron around the crown
    ['HEAD_L', 'head', 'head', [0.075, 0.06, 0.0]],
    ['HEAD_R', 'head', 'head', [-0.075, 0.06, 0.0]],
    ['HEAD_F', 'head', 'head', [0.0, 0.05, 0.09]],
    ['HEAD_B', 'head', 'head', [0.0, 0.08, -0.08]],
    // Left upper arm — 4 non-coplanar triad markers
    ['LUA_A', 'lUpperArm', 'lShoulder', [0.035, -0.07, 0.035]],
    ['LUA_B', 'lUpperArm', 'lShoulder', [-0.03, -0.16, -0.035]],
    ['LUA_C', 'lUpperArm', 'lShoulder', [0.035, -0.24, 0.02]],
    ['LUA_D', 'lUpperArm', 'lShoulder', [-0.028, -0.12, 0.04]],
    // Left forearm (+ hand marker rides with forearm cluster)
    ['LFA_A', 'lForearm', 'lElbow', [0.03, -0.08, 0.03]],
    ['LFA_B', 'lForearm', 'lElbow', [-0.025, -0.16, -0.03]],
    ['LFA_C', 'lForearm', 'lElbow', [0.025, -0.22, 0.02]],
    ['LHND', 'lForearm', 'lWrist', [0.03, -0.02, 0.01]],
    // Right upper arm
    ['RUA_A', 'rUpperArm', 'rShoulder', [-0.035, -0.07, 0.035]],
    ['RUA_B', 'rUpperArm', 'rShoulder', [0.03, -0.16, -0.035]],
    ['RUA_C', 'rUpperArm', 'rShoulder', [-0.035, -0.24, 0.02]],
    ['RUA_D', 'rUpperArm', 'rShoulder', [0.028, -0.12, 0.04]],
    // Right forearm
    ['RFA_A', 'rForearm', 'rElbow', [-0.03, -0.08, 0.03]],
    ['RFA_B', 'rForearm', 'rElbow', [0.025, -0.16, -0.03]],
    ['RFA_C', 'rForearm', 'rElbow', [-0.025, -0.22, 0.02]],
    ['RHND', 'rForearm', 'rWrist', [-0.03, -0.02, 0.01]],
    // Left thigh
    ['LTH_A', 'lThigh', 'lHip', [0.045, -0.10, 0.045]],
    ['LTH_B', 'lThigh', 'lHip', [-0.035, -0.22, -0.045]],
    ['LTH_C', 'lThigh', 'lHip', [0.045, -0.33, 0.02]],
    ['LTH_D', 'lThigh', 'lHip', [-0.04, -0.16, 0.05]],
    // Right thigh
    ['RTH_A', 'rThigh', 'rHip', [-0.045, -0.10, 0.045]],
    ['RTH_B', 'rThigh', 'rHip', [0.035, -0.22, -0.045]],
    ['RTH_C', 'rThigh', 'rHip', [-0.045, -0.33, 0.02]],
    ['RTH_D', 'rThigh', 'rHip', [0.04, -0.16, 0.05]],
    // Left shank
    ['LSK_A', 'lShank', 'lKnee', [0.04, -0.10, 0.04]],
    ['LSK_B', 'lShank', 'lKnee', [-0.03, -0.22, -0.04]],
    ['LSK_C', 'lShank', 'lKnee', [0.035, -0.32, 0.02]],
    ['LSK_D', 'lShank', 'lKnee', [-0.035, -0.18, 0.045]],
    // Right shank
    ['RSK_A', 'rShank', 'rKnee', [-0.04, -0.10, 0.04]],
    ['RSK_B', 'rShank', 'rKnee', [0.03, -0.22, -0.04]],
    ['RSK_C', 'rShank', 'rKnee', [-0.035, -0.32, 0.02]],
    ['RSK_D', 'rShank', 'rKnee', [0.035, -0.18, 0.045]],
    // Left foot — heel + toe spread across all three axes, 4 markers
    ['LHEE', 'lFoot', 'lAnkle', [0.0, 0.01, -0.06]],
    ['LTOE_M', 'lFoot', 'lFoot', [0.0, 0.0, 0.14]],
    ['LTOE_L', 'lFoot', 'lFoot', [0.05, 0.0, 0.12]],
    ['LFOOT_D', 'lFoot', 'lAnkle', [-0.045, 0.02, -0.02]],
    // Right foot
    ['RHEE', 'rFoot', 'rAnkle', [0.0, 0.01, -0.06]],
    ['RTOE_M', 'rFoot', 'rFoot', [0.0, 0.0, 0.14]],
    ['RTOE_L', 'rFoot', 'rFoot', [-0.05, 0.0, 0.12]],
    ['RFOOT_D', 'rFoot', 'rAnkle', [0.045, 0.02, -0.02]],
  ];

  const clusterMap = new Map<string, ClusterDef>();
  const bindings: MarkerBinding[] = table.map(([name, clusterName, anchor, local]) => {
    let c = clusterMap.get(clusterName);
    if (!c) {
      c = { name: clusterName, markers: [], anchorJoint: anchor };
      clusterMap.set(clusterName, c);
    }
    c.markers.push(name);
    const anchorIdx = jointIndex.get(anchor)!;
    const anchorPos = restWorld[anchorIdx];
    const rest: Vec3 = [
      anchorPos[0] + local[0],
      anchorPos[1] + local[1],
      anchorPos[2] + local[2],
    ];
    return { name, cluster: clusterName, restPosition: rest };
  });

  return { clusters: [...clusterMap.values()], bindings };
}

let cached: SkeletonDefinition | null = null;
let cachedRestWorld: Vec3[] = [];

export function defaultSkeleton(): SkeletonDefinition {
  if (cached) return cached;
  const { joints, index } = buildJoints();
  // Rest-pose world positions by walking the hierarchy (T-pose, identity rotations).
  const restWorld: Vec3[] = new Array(joints.length);
  for (let i = 0; i < joints.length; i++) {
    const j = joints[i];
    if (j.parent === null) {
      restWorld[i] = [j.restOffset[0], j.restOffset[1], j.restOffset[2]];
    } else {
      const p = restWorld[j.parent];
      restWorld[i] = [
        p[0] + j.restOffset[0],
        p[1] + j.restOffset[1],
        p[2] + j.restOffset[2],
      ];
    }
  }
  const { clusters, bindings } = buildClustersAndBindings(index, restWorld);
  cachedRestWorld = restWorld;
  cached = { joints, jointIndex: index, clusters, bindings };
  return cached;
}

/** Rest-pose world position of every joint in the default skeleton. */
export function defaultRestWorld(): Vec3[] {
  defaultSkeleton();
  return cachedRestWorld;
}
