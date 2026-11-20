/**
 * 标记点数据导入导出。
 * 支持:
 *  - 本工具 JSON 格式(自描述,含 fps 与 markerNames);
 *  - 通用 CSV:首行表头 frame,x_<name>,y_<name>,z_<name>... 丢点留空。
 * 另含骨骼动画 JSON 导出,供下游 DCC 工具消费。
 */
import type { MarkerClip, SkeletonClip } from './types.js';
import { JOINT_NAMES } from './rig.js';

export interface SerializedMarkerClip {
  format: 'mocap-marker-clip';
  version: 1;
  fps: number;
  markerNames: string[];
  /** 每帧一个 [x,y,z,...] 数组,丢点为 null。 */
  frames: Array<number[] | null>;
}

export function serializeMarkerClip(clip: MarkerClip): SerializedMarkerClip {
  const M = clip.markerNames.length;
  const frames: Array<number[] | null> = [];
  for (let f = 0; f < clip.frameCount; f++) {
    const row: number[] = [];
    for (let m = 0; m < M; m++) {
      if (clip.observed[f * M + m]) {
        const o = (f * M + m) * 3;
        row.push(clip.positions[o], clip.positions[o + 1], clip.positions[o + 2]);
      } else {
        row.push(NaN, NaN, NaN);
      }
    }
    frames.push(row);
  }
  return { format: 'mocap-marker-clip', version: 1, fps: clip.fps, markerNames: [...clip.markerNames], frames };
}

export function deserializeMarkerClip(data: SerializedMarkerClip): MarkerClip {
  if (data.format !== 'mocap-marker-clip' || data.version !== 1) {
    throw new Error('无法识别的标记点文件格式');
  }
  const F = data.frames.length;
  const M = data.markerNames.length;
  const positions = new Float32Array(F * M * 3);
  const observed = new Uint8Array(F * M);
  positions.fill(NaN);
  data.frames.forEach((row, f) => {
    if (!row) return;
    for (let m = 0; m < M; m++) {
      const x = row[m * 3];
      if (Number.isFinite(x)) {
        observed[f * M + m] = 1;
        positions[(f * M + m) * 3] = x;
        positions[(f * M + m) * 3 + 1] = row[m * 3 + 1];
        positions[(f * M + m) * 3 + 2] = row[m * 3 + 2];
      }
    }
  });
  return { markerNames: [...data.markerNames], positions, observed, frameCount: F, fps: data.fps };
}

/**
 * CSV 解析。表头示例:
 * frame,x_HEAD_L,y_HEAD_L,z_HEAD_L,x_HEAD_R,...
 * 空单元格或 NaN 视为丢点。帧号必须连续(从 0 或 1 开始均可)。
 */
export function parseCsv(text: string, fps = 120): MarkerClip {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error('CSV 内容为空或只有表头');
  const header = splitCsvLine(lines[0]);
  const frameCol = header.findIndex((h) => h.trim().toLowerCase() === 'frame');
  const cols: Array<{ marker: string; axis: 0 | 1 | 2; col: number }> = [];
  for (let c = 0; c < header.length; c++) {
    if (c === frameCol) continue;
    const m = /^(x|y|z)[_\s-]+(.+)$/i.exec(header[c].trim());
    if (!m) continue;
    cols.push({ marker: m[2], axis: (m[1].toLowerCase() === 'x' ? 0 : m[1].toLowerCase() === 'y' ? 1 : 2) as 0 | 1 | 2, col: c });
  }
  const markerNames = [...new Set(cols.map((c) => c.marker))];
  const M = markerNames.length;
  if (M === 0) throw new Error('CSV 表头未找到 x_/y_/z_ 形式的标记点列');
  const F = lines.length - 1;
  const positions = new Float32Array(F * M * 3);
  const observed = new Uint8Array(F * M);
  positions.fill(NaN);
  for (let f = 0; f < F; f++) {
    const cells = splitCsvLine(lines[f + 1]);
    for (const cc of cols) {
      const raw = (cells[cc.col] ?? '').trim();
      if (!raw || raw.toLowerCase() === 'nan') continue;
      const v = Number(raw);
      if (!Number.isFinite(v)) continue;
      const m = markerNames.indexOf(cc.marker);
      observed[f * M + m] = 1;
      positions[(f * M + m) * 3 + cc.axis] = v;
    }
  }
  return { markerNames, positions, observed, frameCount: F, fps };
}

export function toCsv(clip: MarkerClip): string {
  const M = clip.markerNames.length;
  const header = ['frame', ...clip.markerNames.flatMap((n) => [`x_${n}`, `y_${n}`, `z_${n}`])];
  const rows: string[] = [header.join(',')];
  for (let f = 0; f < clip.frameCount; f++) {
    const cells: string[] = [String(f)];
    for (let m = 0; m < M; m++) {
      if (!clip.observed[f * M + m]) {
        cells.push('', '', '');
      } else {
        const o = (f * M + m) * 3;
        cells.push(trimNum(clip.positions[o]), trimNum(clip.positions[o + 1]), trimNum(clip.positions[o + 2]));
      }
    }
    rows.push(cells.join(','));
  }
  return rows.join('\n');
}

/** 骨骼动画导出:每帧关节世界坐标/旋转 + 关节名,JSON。 */
export function serializeSkeleton(skel: SkeletonClip): string {
  return JSON.stringify({
    format: 'mocap-skeleton-clip',
    version: 1,
    fps: skel.fps,
    joints: JOINT_NAMES,
    positions: Array.from(skel.jointPositions),
    rotations: Array.from(skel.jointRotations),
  });
}

function trimNum(v: number): string {
  return Math.abs(v) < 1e-9 ? '0' : v.toFixed(6);
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ',') {
      out.push(cur); cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}
