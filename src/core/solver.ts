import { kabsch } from '../math/kabsch';
import { qFromUnitVectors, qMultiply, qNormalize, qRotate, quatIdentity, type Quat } from '../math/quat';
import type { Vec3 } from '../math/vec3';
import { cross, distance, dot, normalize, sub, vec3 } from '../math/vec3';
import { clampReach, solveTwoBone } from './ik';
import type {
  ContactInterval,
  MarkerClip,
  PenetrationEvent,
  SkeletonDefinition,
  SolvedSkeleton,
} from './types';

export interface SolveOptions {
  /** Foot markers below this speed (m/s) and height (m) count as planted. */
  contactSpeed: number;
  contactHeight: number;
  /** Capsule pairs closer than radiusA+radiusB minus this are flagged (m). */
  penetrationTolerance: number;
  onProgress?: (fraction: number) => void;
}

export const DEFAULT_SOLVE_OPTIONS: SolveOptions = {
  contactSpeed: 0.12,
  contactHeight: 0.08,
  penetrationTolerance: 0.005,
};

interface ClusterRT {
  rotation: Quat;
  translation: Vec3;
  rms: number;
}

export function solveSkeleton(
  clip: MarkerClip,
  skel: SkeletonDefinition,
  options: SolveOptions = DEFAULT_SOLVE_OPTIONS,
): SolvedSkeleton {
  const frameCount = clip.frameCount;
  jointCountCheck(skel);
  const jointCount = skel.joints.length;
  const markerCount = clip.markerNames.length;
  const positions = clip.positions;
  const valid = clip.valid;
  const dt = 1 / clip.fps;

  const positionsOut = new Float32Array(frameCount * jointCount * 3);
  const rotationsOut = new Float32Array(frameCount * jointCount * 4);

  // Per-cluster rigid transforms for every frame. Templates are marker rest
  // positions relative to the cluster anchor joint, so the fitted translation
  // is the anchor frame's centroid location in world space.
  const clusterRT = new Map<string, ClusterRT[]>();
  const clusterTemplate = new Map<string, Vec3[]>();
  for (const c of clip.clusters) {
    clusterRT.set(c.name, new Array(frameCount));
    const anchorIdx = skel.jointIndex.get(c.anchorJoint)!;
    const anchorRest = restWorldOf(skel, anchorIdx);
    clusterTemplate.set(
      c.name,
      c.markers.map((name) => {
        const b = skel.bindings.find((bb) => bb.name === name)!;
        return [
          b.restPosition[0] - anchorRest[0],
          b.restPosition[1] - anchorRest[1],
          b.restPosition[2] - anchorRest[2],
        ] as Vec3;
      }),
    );
  }

  for (let f = 0; f < frameCount; f++) {
    for (const cluster of clip.clusters) {
      const src: Vec3[] = [];
      const dst: Vec3[] = [];
      const template = clusterTemplate.get(cluster.name)!;
      cluster.markers.forEach((name, slot) => {
        const mi = clip.markerNames.indexOf(name);
        const base = f * markerCount + mi;
        if (!valid[base]) return;
        const o = base * 3;
        src.push(template[slot]);
        dst.push([positions[o], positions[o + 1], positions[o + 2]]);
      });
      if (src.length >= 3) {
        const fit = kabsch(src, dst);
        if (fit) clusterRT.get(cluster.name)![f] = fit;
      }
    }
  }

  // Fill cluster transforms across occasional failed frames by nearest neighbour.
  for (const c of clip.clusters) {
    const arr = clusterRT.get(c.name)!;
    for (let f = 0; f < frameCount; f++) {
      if (arr[f]) continue;
      const other = arr.find((x) => x) ?? null;
      if (other) arr[f] = other;
    }
  }

  const rtAt = (frame: number, clusterName: string): ClusterRT => clusterRT.get(clusterName)![frame];

  // ---------- Contact detection from foot-marker trajectories ----------
  const contacts = detectContacts(clip, skel, options);
  const contactMask = buildContactMask(contacts, frameCount);

  // Ankle IK targets cached per frame for the foot-lock stage.
  const ankleTargets = new Map<'L' | 'R', Vec3[]>([
    ['L', new Array(frameCount)],
    ['R', new Array(frameCount)],
  ]);

  // ---------- Per-frame joint reconstruction ----------
  for (let f = 0; f < frameCount; f++) {
    const jointPos: Vec3[] = new Array(jointCount);
    const jointRot: Quat[] = new Array(jointCount);

    const pelvisRT = rtAt(f, 'pelvis');
    const torsoRT = rtAt(f, 'torso');
    const headRT = rtAt(f, 'head');

    jointPos[skel.jointIndex.get('pelvis')!] = anchorJointPosition(pelvisRT, skel, 'pelvis');
    jointRot[skel.jointIndex.get('pelvis')!] = pelvisRT.rotation;

    // Spine chain: anchor joints placed by their cluster transforms.
    placeFromCluster(jointPos, jointRot, skel, 'spine2', 'torso', torsoRT);
    placeFromCluster(jointPos, jointRot, skel, 'head', 'head', headRT);
    interpolateJoint(jointPos, jointRot, skel, 'spine1', 'pelvis', 'spine2', 0.5);
    interpolateJoint(jointPos, jointRot, skel, 'neck', 'spine2', 'head', 0.6);

    // Collar/shoulder/elbow/wrist/hand from limb clusters.
    placeLimb(jointPos, jointRot, skel, f, rtAt, 'lShoulder', 'lUpperArm');
    placeLimb(jointPos, jointRot, skel, f, rtAt, 'rShoulder', 'rUpperArm');
    placeLimb(jointPos, jointRot, skel, f, rtAt, 'lElbow', 'lForearm');
    placeLimb(jointPos, jointRot, skel, f, rtAt, 'rElbow', 'rForearm');
    placeLimb(jointPos, jointRot, skel, f, rtAt, 'lWrist', 'lForearm');
    placeLimb(jointPos, jointRot, skel, f, rtAt, 'rWrist', 'rForearm');
    placeCollar(jointPos, jointRot, skel, 'lCollar', 'spine2', 'lShoulder');
    placeCollar(jointPos, jointRot, skel, 'rCollar', 'spine2', 'rShoulder');
    placeHand(jointPos, jointRot, skel, 'lHand', 'lWrist', 'lForearm', rtAt(f, 'lForearm'));
    placeHand(jointPos, jointRot, skel, 'rHand', 'rWrist', 'rForearm', rtAt(f, 'rForearm'));

    // Hips from pelvis transform + rest offsets; knees/ankles via leg IK to
    // ankle targets derived from the shank/foot clusters.
    placeHip(jointPos, jointRot, skel, 'lHip', pelvisRT);
    placeHip(jointPos, jointRot, skel, 'rHip', pelvisRT);

    const lAnkleTarget = ankleTarget(clip, skel, f, ['LHEE', 'LTOE_M', 'LTOE_L'], 'lAnkle');
    const rAnkleTarget = ankleTarget(clip, skel, f, ['RHEE', 'RTOE_M', 'RTOE_L'], 'rAnkle');
    ankleTargets.get('L')![f] = lAnkleTarget;
    ankleTargets.get('R')![f] = rAnkleTarget;

    solveLeg(jointPos, jointRot, skel, 'lHip', 'lKnee', 'lAnkle', lAnkleTarget, [0, 0, 1]);
    solveLeg(jointPos, jointRot, skel, 'rHip', 'rKnee', 'rAnkle', rAnkleTarget, [0, 0, 1]);
    placeFoot(jointPos, jointRot, skel, f, rtAt, 'lFoot', 'lFoot', 'lAnkle');
    placeFoot(jointPos, jointRot, skel, f, rtAt, 'rFoot', 'rFoot', 'rAnkle');
    placeToe(jointPos, jointRot, skel, 'lToe', 'lFoot', rtAt(f, 'lFoot'));
    placeToe(jointPos, jointRot, skel, 'rToe', 'rFoot', rtAt(f, 'rFoot'));

    // Enforce constant bone lengths: project every child joint onto the sphere
    // of its rest bone length around the already-placed parent. This removes
    // the inter-cluster fitting jitter that would otherwise stretch bones.
    enforceBoneLengths(jointPos, skel);

    // Store world-space results.
    for (let j = 0; j < jointCount; j++) {
      const o = (f * jointCount + j) * 3;
      positionsOut[o] = jointPos[j][0];
      positionsOut[o + 1] = jointPos[j][1];
      positionsOut[o + 2] = jointPos[j][2];
      const q = jointRot[j] ?? quatIdentity();
      const qo = (f * jointCount + j) * 4;
      rotationsOut[qo] = q[0];
      rotationsOut[qo + 1] = q[1];
      rotationsOut[qo + 2] = q[2];
      rotationsOut[qo + 3] = q[3];
    }
    options.onProgress?.((f + 1) / frameCount);
  }

  // ---------- Foot lock: freeze planted feet so they cannot slide ----------
  applyFootLock(positionsOut, rotationsOut, skel, contactMask, ankleTargets, clip.fps);

  // Recompute local rotations from the locked world transforms.
  const finalRotations = computeLocalRotations(positionsOut, rotationsOut, skel, frameCount);

  const penetrations = detectPenetrations(positionsOut, skel, frameCount, options);

  return {
    definition: skel,
    fps: clip.fps,
    frameCount,
    poses: { positions: positionsOut, rotations: finalRotations },
    contacts,
    penetrations,
  };
}

function jointCountCheck(skel: SkeletonDefinition): void {
  if (skel.joints.length === 0) throw new Error('empty skeleton');
}

/**
 * Project each non-root joint onto its rest bone-length sphere around the
 * parent. Direction is taken from the current (cluster-derived) position, so
 * only the radial distance changes — pose orientation is preserved.
 */
function enforceBoneLengths(pos: Vec3[], skel: SkeletonDefinition): void {
  for (let j = 1; j < skel.joints.length; j++) {
    const joint = skel.joints[j];
    const parent = joint.parent!;
    if (!pos[j] || !pos[parent]) continue;
    const restLen = Math.hypot(
      joint.restOffset[0], joint.restOffset[1], joint.restOffset[2],
    );
    if (restLen < 1e-9) continue;
    const delta = sub(pos[j], pos[parent]);
    const d = Math.hypot(delta[0], delta[1], delta[2]);
    if (d < 1e-9) continue;
    const k = restLen / d;
    pos[j] = [
      pos[parent][0] + delta[0] * k,
      pos[parent][1] + delta[1] * k,
      pos[parent][2] + delta[2] * k,
    ];
  }
}

/**
 * Recover an anchor joint's world position from a cluster rigid transform.
 * The Kabsch translation locates the template CENTROID, not the anchor, so the
 * anchor's centroid-relative rest offset must be rotated back out.
 *   marker_world = R * template + t, with template = markerRest - anchorRest
 * Centroid template c = mean(template); anchor maps to R*(-c) + t.
 */
function anchorJointPosition(
  rt: ClusterRT,
  skel: SkeletonDefinition,
  clusterName: string,
): Vec3 {
  const cluster = skel.clusters.find((c) => c.name === clusterName)!;
  const anchorIdx = skel.jointIndex.get(cluster.anchorJoint)!;
  const anchorRest = restWorldOf(skel, anchorIdx);
  let cx = 0, cy = 0, cz = 0;
  for (const name of cluster.markers) {
    const b = skel.bindings.find((bb) => bb.name === name)!;
    cx += b.restPosition[0] - anchorRest[0];
    cy += b.restPosition[1] - anchorRest[1];
    cz += b.restPosition[2] - anchorRest[2];
  }
  const n = cluster.markers.length;
  const centroidRel: Vec3 = [cx / n, cy / n, cz / n];
  // The anchor sits at template position -centroidRel (the centroid is the
  // template origin). Rotate that anchor template point and translate it.
  const anchorTemplate: Vec3 = [-centroidRel[0], -centroidRel[1], -centroidRel[2]];
  const rotated = qRotate(rt.rotation, anchorTemplate);
  return [
    rt.translation[0] + rotated[0],
    rt.translation[1] + rotated[1],
    rt.translation[2] + rotated[2],
  ];
}

function placeFromCluster(
  pos: Vec3[], rot: Quat[], skel: SkeletonDefinition,
  jointName: string, clusterName: string, rt: ClusterRT,
): void {
  const idx = skel.jointIndex.get(jointName)!;
  pos[idx] = anchorJointPosition(rt, skel, clusterName);
  rot[idx] = [...rt.rotation] as Quat;
}

function interpolateJoint(
  pos: Vec3[], rot: Quat[], skel: SkeletonDefinition,
  name: string, parent: string, child: string, alpha: number,
): void {
  const i = skel.jointIndex.get(name)!;
  const pi = skel.jointIndex.get(parent)!;
  const ci = skel.jointIndex.get(child)!;
  if (!pos[pi] || !pos[ci]) return;
  pos[i] = [
    pos[pi][0] + (pos[ci][0] - pos[pi][0]) * alpha,
    pos[pi][1] + (pos[ci][1] - pos[pi][1]) * alpha,
    pos[pi][2] + (pos[ci][2] - pos[pi][2]) * alpha,
  ];
  rot[i] = qNormalize([
    rot[pi][0] + (rot[ci][0] - rot[pi][0]) * alpha,
    rot[pi][1] + (rot[ci][1] - rot[pi][1]) * alpha,
    rot[pi][2] + (rot[ci][2] - rot[pi][2]) * alpha,
    rot[pi][3] + (rot[ci][3] - rot[pi][3]) * alpha,
  ]);
}

function placeLimb(
  pos: Vec3[], rot: Quat[], skel: SkeletonDefinition, frame: number,
  rtAt: (f: number, name: string) => ClusterRT,
  jointName: string, clusterName: string,
): void {
  const rt = rtAt(frame, clusterName);
  const idx = skel.jointIndex.get(jointName)!;
  pos[idx] = anchorJointPosition(rt, skel, clusterName);
  rot[idx] = [...rt.rotation] as Quat;
}

function placeCollar(
  pos: Vec3[], rot: Quat[], skel: SkeletonDefinition,
  name: string, spineName: string, shoulderName: string,
): void {
  const i = skel.jointIndex.get(name)!;
  const si = skel.jointIndex.get(spineName)!;
  const shi = skel.jointIndex.get(shoulderName)!;
  if (!pos[shi]) return;
  // Collar sits along the line from spine to shoulder.
  pos[i] = [
    pos[si][0] + (pos[shi][0] - pos[si][0]) * 0.35,
    pos[si][1] + (pos[shi][1] - pos[si][1]) * 0.35,
    pos[si][2] + (pos[shi][2] - pos[si][2]) * 0.35,
  ];
  rot[i] = rot[si] ? [...rot[si]] as Quat : quatIdentity();
}

function placeHand(
  pos: Vec3[], rot: Quat[], skel: SkeletonDefinition,
  name: string, wristName: string, _cluster: string, rt: ClusterRT,
): void {
  const i = skel.jointIndex.get(name)!;
  const wi = skel.jointIndex.get(wristName)!;
  const rest = skel.joints[i];
  const off = qRotateLocal(rt.rotation, rest.restOffset);
  pos[i] = [pos[wi][0] + off[0], pos[wi][1] + off[1], pos[wi][2] + off[2]];
  rot[i] = [...rt.rotation] as Quat;
}

function placeHip(pos: Vec3[], rot: Quat[], skel: SkeletonDefinition, name: string, pelvis: ClusterRT): void {
  const i = skel.jointIndex.get(name)!;
  const off = qRotateLocal(pelvis.rotation, skel.joints[i].restOffset);
  const pi = skel.joints[i].parent!;
  const pp = pos[pi];
  pos[i] = [pp[0] + off[0], pp[1] + off[1], pp[2] + off[2]];
  rot[i] = [...pelvis.rotation] as Quat;
}

function solveLeg(
  pos: Vec3[], rot: Quat[], skel: SkeletonDefinition,
  hipName: string, kneeName: string, ankleName: string,
  ankleTargetPos: Vec3, poleHint: Vec3,
): void {
  const hi = skel.jointIndex.get(hipName)!;
  const ki = skel.jointIndex.get(kneeName)!;
  const ai = skel.jointIndex.get(ankleName)!;
  const len1 = Math.hypot(...skel.joints[ki].restOffset);
  const len2 = Math.hypot(...skel.joints[ai].restOffset);
  const target = clampReach(pos[hi], ankleTargetPos, len1, len2);
  const res = solveTwoBone(pos[hi], len1, len2, target, poleHint, quatIdentity(), quatIdentity());
  pos[ki] = res.mid;
  pos[ai] = res.end;
  rot[hi] = res.qStart;
  rot[ki] = res.qMid;
  rot[ai] = quatIdentity();
}

function placeFoot(
  pos: Vec3[], rot: Quat[], skel: SkeletonDefinition, _frame: number,
  rtAt: (f: number, name: string) => ClusterRT,
  name: string, clusterName: string, ankleName: string,
): void {
  const i = skel.jointIndex.get(name)!;
  const ai = skel.jointIndex.get(ankleName)!;
  const rt = rtAt(_frame, clusterName);
  const off = qRotateLocal(rt.rotation, skel.joints[i].restOffset);
  pos[i] = [pos[ai][0] + off[0], pos[ai][1] + off[1], pos[ai][2] + off[2]];
  rot[i] = [...rt.rotation] as Quat;
}

function placeToe(
  pos: Vec3[], rot: Quat[], skel: SkeletonDefinition,
  name: string, footName: string, rt: ClusterRT,
): void {
  const i = skel.jointIndex.get(name)!;
  const fi = skel.jointIndex.get(footName)!;
  const off = qRotateLocal(rt.rotation, skel.joints[i].restOffset);
  pos[i] = [pos[fi][0] + off[0], pos[fi][1] + off[1], pos[fi][2] + off[2]];
  rot[i] = [...rt.rotation] as Quat;
}

function qRotateLocal(q: Quat, v: Vec3): Vec3 {
  return qRotate(q, v);
}
function ankleTarget(
  clip: MarkerClip,
  skel: SkeletonDefinition,
  frame: number,
  markerNames: string[],
  ankleName: string,
): Vec3 {
  let x = 0, y = 0, z = 0, n = 0;
  for (const name of markerNames) {
    const mi = clip.markerNames.indexOf(name);
    const base = frame * clip.markerNames.length + mi;
    if (!clip.valid[base]) continue;
    const o = base * 3;
    x += clip.positions[o]; y += clip.positions[o + 1]; z += clip.positions[o + 2];
    n++;
  }
  if (n === 0) return vec3(0, 0.1, 0);
  // Shift the foot-marker centroid back to the ankle using the rest-pose vector.
  const centroid: Vec3 = [x / n, y / n, z / n];
  const restCentroid = averageRest(skel, markerNames);
  const ankleIdx = skel.jointIndex.get(ankleName)!;
  const ankleRestPos = restWorldOf(skel, ankleIdx);
  const correction: Vec3 = [
    ankleRestPos[0] - restCentroid[0],
    ankleRestPos[1] - restCentroid[1],
    ankleRestPos[2] - restCentroid[2],
  ];
  return [centroid[0] + correction[0], centroid[1] + correction[1], centroid[2] + correction[2]];
}

function averageRest(skel: SkeletonDefinition, names: string[]): Vec3 {
  let x = 0, y = 0, z = 0;
  for (const name of names) {
    const b = skel.bindings.find((bb) => bb.name === name)!;
    x += b.restPosition[0]; y += b.restPosition[1]; z += b.restPosition[2];
  }
  const n = names.length;
  return [x / n, y / n, z / n];
}

function restWorldOf(skel: SkeletonDefinition, joint: number): Vec3 {
  const j = skel.joints[joint];
  if (j.parent === null) return [...j.restOffset] as Vec3;
  const p = restWorldOf(skel, j.parent);
  return [p[0] + j.restOffset[0], p[1] + j.restOffset[1], p[2] + j.restOffset[2]];
}

// ---------------------------------------------------------------------------
// Contact detection & foot locking
// ---------------------------------------------------------------------------

function detectContacts(clip: MarkerClip, skel: SkeletonDefinition, options: SolveOptions): ContactInterval[] {
  const out: ContactInterval[] = [];
  const dt = 1 / clip.fps;
  const feet: Array<{ side: 'L' | 'R'; markers: string[] }> = [
    { side: 'L', markers: ['LHEE', 'LTOE_M', 'LTOE_L'] },
    { side: 'R', markers: ['RHEE', 'RTOE_M', 'RTOE_L'] },
  ];
  const floorY = estimateFloorY(clip);

  for (const foot of feet) {
    // Per-frame mean foot speed and height above the estimated floor.
    const speed = new Array<number>(clip.frameCount).fill(Infinity);
    const height = new Array<number>(clip.frameCount).fill(Infinity);
    for (let f = 0; f < clip.frameCount; f++) {
      let speedSum = 0;
      let heightSum = 0;
      let n = 0;
      for (const name of foot.markers) {
        const mi = clip.markerNames.indexOf(name);
        const base = f * clip.markerNames.length + mi;
        if (!clip.valid[base]) continue;
        const o = base * 3;
        heightSum += clip.positions[o + 1] - floorY;
        if (f > 0) {
          const po = ((f - 1) * clip.markerNames.length + mi) * 3;
          const dx = clip.positions[o] - clip.positions[po];
          const dy = clip.positions[o + 1] - clip.positions[po + 1];
          const dz = clip.positions[o + 2] - clip.positions[po + 2];
          speedSum += Math.hypot(dx, dy, dz) / dt;
        }
        n++;
      }
      if (n > 0) {
        speed[f] = speedSum / n;
        height[f] = heightSum / n;
      }
    }

    // Adaptive speed gate: a stance is the low-speed region of each foot's own
    // speed cycle. Use whichever is looser of the configured absolute limit or
    // the 20th percentile, so planted phases are found even on noisy/lively data.
    const finiteSpeeds = speed.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
    const p20 = finiteSpeeds[Math.floor(finiteSpeeds.length * 0.2)] ?? options.contactSpeed;
    const speedGate = Math.max(options.contactSpeed, p20 * 1.25);
    const heightGate = options.contactHeight;

    const planted = new Uint8Array(clip.frameCount);
    for (let f = 0; f < clip.frameCount; f++) {
      if (speed[f] <= speedGate && height[f] <= heightGate) planted[f] = 1;
    }
    // Close single-frame holes and require a minimum 3-frame stance.
    for (let f = 1; f < clip.frameCount - 1; f++) {
      if (!planted[f] && planted[f - 1] && planted[f + 1]) planted[f] = 1;
    }
    let f = 0;
    while (f < clip.frameCount) {
      if (!planted[f]) { f++; continue; }
      const start = f;
      while (f < clip.frameCount && planted[f]) f++;
      const end = f - 1;
      if (end - start + 1 >= 3) out.push({ side: foot.side, start, end });
    }
  }
  return out.sort((a, b) => a.start - b.start);
}

function estimateFloorY(clip: MarkerClip): number {
  let minY = Infinity;
  for (let f = 0; f < clip.frameCount; f += Math.max(1, Math.floor(clip.frameCount / 200))) {
    for (const name of ['LHEE', 'RHEE', 'LTOE_M', 'RTOE_M']) {
      const mi = clip.markerNames.indexOf(name);
      if (mi < 0) continue;
      const base = f * clip.markerNames.length + mi;
      if (clip.valid[base]) minY = Math.min(minY, clip.positions[base * 3 + 1]);
    }
  }
  return Number.isFinite(minY) ? minY : 0;
}

function buildContactMask(contacts: ContactInterval[], frameCount: number): Map<'L' | 'R', Uint8Array> {
  const mask = new Map<'L' | 'R', Uint8Array>([
    ['L', new Uint8Array(frameCount)],
    ['R', new Uint8Array(frameCount)],
  ]);
  for (const c of contacts) {
    for (let f = c.start; f <= c.end; f++) mask.get(c.side)![f] = 1;
  }
  return mask;
}

/**
 * Lock every planted stance to the world position of its first frame.
 * The whole leg chain (ankle, foot, toe) is translated by the same delta so
 * the foot cannot rotate around a sliding pivot either.
 */
function applyFootLock(
  positions: Float32Array,
  rotations: Float32Array,
  skel: SkeletonDefinition,
  mask: Map<'L' | 'R', Uint8Array>,
  ankleTargets: Map<string, Vec3[]>,
  fps: number,
): void {
  void fps;
  const jointCount = skel.joints.length;
  const chain = (side: 'L' | 'R'): number[] => {
    const prefix = side === 'L' ? 'l' : 'r';
    return ['Ankle', 'Foot', 'Toe'].map((s) => skel.jointIndex.get(prefix + s)!);
  };
  for (const side of ['L', 'R'] as const) {
    const m = mask.get(side)!;
    const ids = chain(side);
    let runStart = -1;
    for (let f = 0; f <= m.length; f++) {
      if (f < m.length && m[f]) {
        if (runStart < 0) runStart = f;
        continue;
      }
      if (runStart < 0) continue;
      const anchor = ids[0];
      const ao = (runStart * jointCount + anchor) * 3;
      const ax = positions[ao];
      const ay = positions[ao + 1];
      const az = positions[ao + 2];
      for (let g = runStart; g < f; g++) {
        const go = (g * jointCount + anchor) * 3;
        const dx = ax - positions[go];
        const dy = ay - positions[go + 1];
        const dz = az - positions[go + 2];
        for (const j of ids) {
          const jo = (g * jointCount + j) * 3;
          positions[jo] += dx;
          positions[jo + 1] += dy;
          positions[jo + 2] += dz;
        }
        // Keep the ankle/foot orientation of the stance's first frame.
        for (const j of ids) {
          const qo = (g * jointCount + j) * 4;
          const q0j = (runStart * jointCount + j) * 4;
          rotations[qo] = rotations[q0j];
          rotations[qo + 1] = rotations[q0j + 1];
          rotations[qo + 2] = rotations[q0j + 2];
          rotations[qo + 3] = rotations[q0j + 3];
        }
      }
      // Snap the IK target cache so downstream consumers agree.
      const targets = ankleTargets.get(side);
      if (targets) {
        for (let g = runStart; g < f; g++) {
          const go = (g * jointCount + anchor) * 3;
          targets[g] = [positions[go], positions[go + 1], positions[go + 2]];
        }
      }
      runStart = -1;
    }
  }
}

/** Convert world rotations into local (parent-relative) rotations. */
function computeLocalRotations(
  positions: Float32Array,
  worldRotations: Float32Array,
  skel: SkeletonDefinition,
  frameCount: number,
): Float32Array {
  const jointCount = skel.joints.length;
  const out = new Float32Array(frameCount * jointCount * 4);
  for (let f = 0; f < frameCount; f++) {
    for (let j = 0; j < jointCount; j++) {
      const qo = (f * jointCount + j) * 4;
      const joint = skel.joints[j];
      if (joint.parent === null) {
        out[qo] = worldRotations[qo];
        out[qo + 1] = worldRotations[qo + 1];
        out[qo + 2] = worldRotations[qo + 2];
        out[qo + 3] = worldRotations[qo + 3];
      } else {
        const pq: Quat = [
          worldRotations[(f * jointCount + joint.parent) * 4],
          worldRotations[(f * jointCount + joint.parent) * 4 + 1],
          worldRotations[(f * jointCount + joint.parent) * 4 + 2],
          worldRotations[(f * jointCount + joint.parent) * 4 + 3],
        ];
        const wq: Quat = [worldRotations[qo], worldRotations[qo + 1], worldRotations[qo + 2], worldRotations[qo + 3]];
        const local = qNormalize(qMultiply(conj(pq), wq));
        out[qo] = local[0];
        out[qo + 1] = local[1];
        out[qo + 2] = local[2];
        out[qo + 3] = local[3];
      }
    }
  }
  return out;
}

function conj(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

// ---------------------------------------------------------------------------
// Penetration detection: capsule-capsule nearest distance
// ---------------------------------------------------------------------------

export function detectPenetrations(
  positions: Float32Array,
  skel: SkeletonDefinition,
  frameCount: number,
  options: SolveOptions = DEFAULT_SOLVE_OPTIONS,
): PenetrationEvent[] {
  const pairs = buildPenetrationPairs(skel);
  const events: PenetrationEvent[] = [];
  const jointCount = skel.joints.length;

  for (let f = 0; f < frameCount; f++) {
    for (const pair of pairs) {
      const a0 = (f * jointCount + pair.a) * 3;
      const a1 = (f * jointCount + pair.aChild) * 3;
      const b0 = (f * jointCount + pair.b) * 3;
      const b1 = (f * jointCount + pair.bChild) * 3;
      const p1: Vec3 = [positions[a0], positions[a0 + 1], positions[a0 + 2]];
      const q1: Vec3 = [positions[a1], positions[a1 + 1], positions[a1 + 2]];
      const p2: Vec3 = [positions[b0], positions[b0 + 1], positions[b0 + 2]];
      const q2: Vec3 = [positions[b1], positions[b1 + 1], positions[b1 + 2]];
      const d = segmentSegmentDistance(p1, q1, p2, q2);
      const radiusSum = skel.joints[pair.a].radius + skel.joints[pair.b].radius;
      const gap = d - radiusSum;
      if (gap < -options.penetrationTolerance) {
        events.push({
          frame: f,
          boneA: skel.joints[pair.a].name,
          boneB: skel.joints[pair.b].name,
          depthMm: gap * 1000,
          point: midpoint(p1, q1),
        });
      }
    }
  }
  return events;
}

interface PenPair {
  a: number;
  aChild: number;
  b: number;
  bChild: number;
}

function buildPenetrationPairs(skel: SkeletonDefinition): PenPair[] {
  const limbs: Array<[string, string]> = [
    ['lUpperArm', 'lElbow'], ['lForearm', 'lWrist'],
    ['rUpperArm', 'rElbow'], ['rForearm', 'rWrist'],
    ['lThigh', 'lKnee'], ['lShank', 'lAnkle'],
    ['rThigh', 'rKnee'], ['rShank', 'rAnkle'],
    ['torso', 'pelvis'],
  ];
  const seg = limbs.map(([a, b]) => ({
    a: skel.jointIndex.get(a === 'lUpperArm' ? 'lShoulder' : a === 'lForearm' ? 'lElbow'
      : a === 'rUpperArm' ? 'rShoulder' : a === 'rForearm' ? 'rElbow'
      : a === 'lThigh' ? 'lHip' : a === 'lShank' ? 'lKnee'
      : a === 'rThigh' ? 'rHip' : a === 'rShank' ? 'rKnee'
      : a === 'torso' ? 'spine2' : 'pelvis')!,
    aChild: skel.jointIndex.get(b)!,
    name: a,
  }));
  const pairs: PenPair[] = [];
  for (let i = 0; i < seg.length; i++) {
    for (let k = i + 1; k < seg.length; k++) {
      // Skip adjacent / same-limb chains that legitimately touch.
      const ni = limbs[i][0];
      const nk = limbs[k][0];
      if (sameChain(ni, nk)) continue;
      // Skip segments sharing a joint or separated by one joint in the
      // hierarchy — those always overlap by construction (e.g. hip vs spine,
      // shoulder girdle vs torso) and are not self-contact anomalies.
      const endpointsA = new Set([seg[i].a, seg[i].aChild]);
      const endpointsB = new Set([seg[k].a, seg[k].aChild]);
      if ([...endpointsA].some((x) => endpointsB.has(x))) continue;
      if (topologyDistance(skel, seg[i].a, seg[k].a) <= 3) continue;
      pairs.push({ a: seg[i].a, aChild: seg[i].aChild, b: seg[k].a, bChild: seg[k].aChild });
    }
  }
  return pairs;
}

/** Number of hierarchy edges between two joints. */
function topologyDistance(skel: SkeletonDefinition, a: number, b: number): number {
  const ancestors = new Map<number, number>();
  let cur: number | null = a;
  let depth = 0;
  while (cur !== null) {
    ancestors.set(cur, depth++);
    cur = skel.joints[cur].parent;
  }
  cur = b;
  depth = 0;
  while (cur !== null && !ancestors.has(cur)) {
    cur = skel.joints[cur].parent;
    depth++;
  }
  if (cur === null) return Infinity;
  return depth + ancestors.get(cur)!;
}

function sameChain(a: string, b: string): boolean {
  const side = (n: string) => (n.startsWith('l') ? 'L' : n.startsWith('r') ? 'R' : 'C');
  const limb = (n: string) => (n.includes('Arm') || n.includes('Forearm') ? 'arm' : n.includes('Thigh') || n.includes('Shank') ? 'leg' : 'core');
  return side(a) === side(b) && limb(a) === limb(b);
}

function midpoint(a: Vec3, b: Vec3): Vec3 {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
}

/**
 * Closest distance between two finite segments (Ericson, Real-Time Collision
 * Detection, section 5.1.9).
 */
function segmentSegmentDistance(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3): number {
  const d1 = sub(q1, p1);
  const d2 = sub(q2, p2);
  const r = sub(p1, p2);
  const a = dot(d1, d1);
  const e = dot(d2, d2);
  const f = dot(d2, r);
  const eps = 1e-10;
  let s: number;
  let t: number;
  if (a <= eps && e <= eps) return distance(p1, p2);
  if (a <= eps) {
    s = 0;
    t = Math.min(1, Math.max(0, f / e));
  } else {
    const c = dot(d1, r);
    if (e <= eps) {
      t = 0;
      s = Math.min(1, Math.max(0, -c / a));
    } else {
      const b = dot(d1, d2);
      const denom = a * e - b * b;
      s = denom > eps ? Math.min(1, Math.max(0, (b * f - c * e) / denom)) : 0;
      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = Math.min(1, Math.max(0, -c / a));
      } else if (t > 1) {
        t = 1;
        s = Math.min(1, Math.max(0, (b - c) / a));
      }
    }
  }
  const c1 = [p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s] as Vec3;
  const c2 = [p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t] as Vec3;
  return distance(c1, c2);
}
