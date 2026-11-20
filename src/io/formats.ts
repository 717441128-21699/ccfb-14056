import { qToMat3 } from '../math/quat';
import type { Vec3 } from '../math/vec3';
import { defaultSkeleton } from '../core/skeleton';
import type { MarkerClip, SkeletonDefinition, SolvedSkeleton } from '../core/types';

/** Minimal RFC-4180-ish CSV splitter: handles quoted fields and doubled quotes. */
export function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      out.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

/**
 * CSV layout:
 *   frame, time, <marker>.x, <marker>.y, <marker>.z, ...
 * Missing samples are blank cells. Header order defines marker identity.
 */
export function parseMarkerCsv(text: string, fps: number): MarkerClip {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error('CSV 至少需要表头和一行数据');
  const header = splitCsvLine(lines[0]);
  // Leading columns are frame and time; everything after comes in x/y/z triples.
  const leadingCols = header[0].toLowerCase() === 'frame' && /^time$/i.test(header[1] ?? '') ? 2 : 1;
  if ((header.length - leadingCols) % 3 !== 0) {
    throw new Error('CSV 坐标列必须为连续的 x/y/z 三元组');
  }
  const markerNames: string[] = [];
  const colOfMarker = new Map<string, number>();
  for (let c = leadingCols; c + 2 <= header.length - 1; c += 3) {
    const name = header[c].replace(/\.x$/i, '');
    if (header[c + 1].replace(/\.y$/i, '') !== name || header[c + 2].replace(/\.z$/i, '') !== name) {
      throw new Error(`CSV 表头第 ${c + 1} 列起不是连续的 x/y/z 三元组`);
    }
    colOfMarker.set(name, c);
    markerNames.push(name);
  }
  const frameCount = lines.length - 1;
  const markerCount = markerNames.length;
  const positions = new Float32Array(frameCount * markerCount * 3);
  const valid = new Uint8Array(frameCount * markerCount);

  for (let f = 0; f < frameCount; f++) {
    const cells = splitCsvLine(lines[f + 1]);
    for (let m = 0; m < markerCount; m++) {
      const c = colOfMarker.get(markerNames[m])!;
      const x = parseFloat(cells[c]);
      const y = parseFloat(cells[c + 1]);
      const z = parseFloat(cells[c + 2]);
      const base = f * markerCount + m;
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        const o = base * 3;
        positions[o] = x;
        positions[o + 1] = y;
        positions[o + 2] = z;
        valid[base] = 1;
      }
    }
  }

  const skel = defaultSkeleton();
  const clusters = skel.clusters
    .map((c) => ({ ...c, markers: c.markers.filter((n) => markerNames.includes(n)) }))
    .filter((c) => c.markers.length >= 3);
  const bindings = skel.bindings.filter((b) => markerNames.includes(b.name));
  return { fps, markerNames, frameCount, positions, valid, clusters, bindings };
}

export function toMarkerCsv(clip: MarkerClip): string {
  const header = ['frame', 'time'];
  for (const n of clip.markerNames) header.push(`${n}.x`, `${n}.y`, `${n}.z`);
  const rows = [header.join(',')];
  const mc = clip.markerNames.length;
  for (let f = 0; f < clip.frameCount; f++) {
    const cells = [String(f), (f / clip.fps).toFixed(4)];
    for (let m = 0; m < mc; m++) {
      const base = f * mc + m;
      if (!clip.valid[base]) {
        cells.push('', '', '');
      } else {
        const o = base * 3;
        cells.push(
          clip.positions[o].toFixed(6),
          clip.positions[o + 1].toFixed(6),
          clip.positions[o + 2].toFixed(6),
        );
      }
    }
    rows.push(cells.join(','));
  }
  return rows.join('\n');
}

interface ClipJson {
  format: 'mocap-clean-markers';
  version: 1;
  fps: number;
  markerNames: string[];
  frames: Array<Record<string, [number, number, number] | null>>;
}

export function toMarkerJson(clip: MarkerClip): string {
  const mc = clip.markerNames.length;
  const frames: ClipJson['frames'] = [];
  for (let f = 0; f < clip.frameCount; f++) {
    const row: ClipJson['frames'][number] = {};
    for (let m = 0; m < mc; m++) {
      const base = f * mc + m;
      if (!clip.valid[base]) {
        row[clip.markerNames[m]] = null;
      } else {
        const o = base * 3;
        row[clip.markerNames[m]] = [clip.positions[o], clip.positions[o + 1], clip.positions[o + 2]];
      }
    }
    frames.push(row);
  }
  const payload: ClipJson = {
    format: 'mocap-clean-markers',
    version: 1,
    fps: clip.fps,
    markerNames: clip.markerNames,
    frames,
  };
  return JSON.stringify(payload, null, 1);
}

export function parseMarkerJson(text: string): MarkerClip {
  const data = JSON.parse(text) as ClipJson;
  if (data.format !== 'mocap-clean-markers') throw new Error('无法识别的 JSON 格式');
  const markerNames = data.markerNames;
  const mc = markerNames.length;
  const frameCount = data.frames.length;
  const positions = new Float32Array(frameCount * mc * 3);
  const valid = new Uint8Array(frameCount * mc);
  data.frames.forEach((row, f) => {
    markerNames.forEach((name, m) => {
      const v = row[name];
      if (!v) return;
      const base = f * mc + m;
      const o = base * 3;
      positions[o] = v[0];
      positions[o + 1] = v[1];
      positions[o + 2] = v[2];
      valid[base] = 1;
    });
  });
  const skel = defaultSkeleton();
  const clusters = skel.clusters
    .map((c) => ({ ...c, markers: c.markers.filter((n) => markerNames.includes(n)) }))
    .filter((c) => c.markers.length >= 3);
  const bindings = skel.bindings.filter((b) => markerNames.includes(b.name));
  return { fps: data.fps, markerNames, frameCount, positions, valid, clusters, bindings };
}

export function downloadText(filename: string, text: string, mime = 'text/plain'): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * BVH export of a solved skeleton. Joint offsets come from the rest pose;
 * per-frame channels are the joint's world rotation as Euler ZXY plus the
 * root translation. This keeps the file readable in Blender / MotionBuilder.
 */
export function toBvh(solved: SolvedSkeleton): string {
  const skel: SkeletonDefinition = solved.definition;
  const lines: string[] = [];
  lines.push('HIERARCHY');
  writeBvhJoint(skel, 0, lines, true);
  lines.push('MOTION');
  lines.push(`Frames: ${solved.frameCount}`);
  lines.push(`Frame Time: ${1 / solved.fps}`);

  const jc = skel.joints.length;
  for (let f = 0; f < solved.frameCount; f++) {
    const cells: string[] = [];
    for (let j = 0; j < jc; j++) {
      const po = (f * jc + j) * 3;
      const qo = (f * jc + j) * 4;
      if (skel.joints[j].parent === null) {
        cells.push(
          solved.poses.positions[po].toFixed(6),
          solved.poses.positions[po + 1].toFixed(6),
          solved.poses.positions[po + 2].toFixed(6),
        );
      }
      const e = eulerZxy([
        solved.poses.rotations[qo],
        solved.poses.rotations[qo + 1],
        solved.poses.rotations[qo + 2],
        solved.poses.rotations[qo + 3],
      ]);
      cells.push(
        (e[0] * 180 / Math.PI).toFixed(4),
        (e[1] * 180 / Math.PI).toFixed(4),
        (e[2] * 180 / Math.PI).toFixed(4),
      );
    }
    lines.push(cells.join(' '));
  }
  return lines.join('\n');
}

function writeBvhJoint(skel: SkeletonDefinition, joint: number, lines: string[], isRoot: boolean): void {
  const def = skel.joints[joint];
  const indent = '  '.repeat(depth(skel, joint));
  lines.push(`${indent}${isRoot ? 'ROOT' : 'JOINT'} ${def.name}`);
  lines.push(`${indent}{`);
  const off = def.restOffset;
  lines.push(`${indent}  OFFSET ${off[0].toFixed(6)} ${off[1].toFixed(6)} ${off[2].toFixed(6)}`);
  const children = skel.joints.map((j, i) => (j.parent === joint ? i : -1)).filter((i) => i >= 0);
  if (children.length === 0) {
    lines.push(`${indent}  CHANNELS 3 Zrotation Xrotation Yrotation`);
    lines.push(`${indent}  End Site`);
    lines.push(`${indent}  {`);
    lines.push(`${indent}    OFFSET 0.000000 0.000000 0.000000`);
    lines.push(`${indent}  }`);
  } else {
    if (isRoot) {
      lines.push(`${indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation`);
    } else {
      lines.push(`${indent}  CHANNELS 3 Zrotation Xrotation Yrotation`);
    }
    for (const c of children) writeBvhJoint(skel, c, lines, false);
  }
  lines.push(`${indent}}`);
}

function depth(skel: SkeletonDefinition, joint: number): number {
  let d = 0;
  let cur: number | null = joint;
  while (cur !== null) {
    cur = skel.joints[cur].parent;
    d++;
  }
  return d - 1;
}

/** Extract Euler angles (ZXY order) from a unit quaternion. */
export function eulerZxy(q: [number, number, number, number]): [number, number, number] {
  const [x, y, z, w] = q;
  const m = qToMat3(q);
  // R = Rz(z) * Rx(x) * Ry(y); derive via matrix entries.
  const sy = Math.min(1, Math.max(-1, m[2])); // m[0][2]
  const yAngle = Math.asin(sy);
  let xAngle: number;
  let zAngle: number;
  if (Math.abs(m[2]) < 0.999999) {
    xAngle = Math.atan2(-m[5], m[8]); // -m[1][2], m[2][2]
    zAngle = Math.atan2(-m[1], m[0]); // -m[0][1], m[0][0]
  } else {
    xAngle = Math.atan2(m[7], m[4]);
    zAngle = 0;
  }
  void x; void y; void z; void w;
  return [zAngle, xAngle, yAngle];
}

export type { Vec3 };
