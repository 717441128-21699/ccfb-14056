import type { Vec3 } from './vec3';
import { lerp } from './vec3';

/**
 * Cubic Hermite interpolation through anchor points at integer frame indices.
 * Endpoint tangents use one-sided differences; interior tangents are the
 * average of neighboring secants (Catmull-Rom style, tension 0.5).
 */
export function hermiteInterpolate(
  anchors: { frame: number; value: Vec3 }[],
  frame: number,
): Vec3 {
  if (anchors.length === 1) return anchors[0].value;
  if (frame <= anchors[0].frame) return anchors[0].value;
  const last = anchors[anchors.length - 1];
  if (frame >= last.frame) return last.value;

  let seg = 0;
  while (seg + 1 < anchors.length && anchors[seg + 1].frame < frame) seg++;
  const a = anchors[seg];
  const b = anchors[seg + 1];
  const span = b.frame - a.frame;
  const t = (frame - a.frame) / span;

  const prev = anchors[seg - 1];
  const next = anchors[seg + 2];
  const ta = tangent(prev ?? a, a, b);
  const tb = tangent(a, b, next ?? b);
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  return [
    h00 * a.value[0] + h10 * span * ta[0] + h01 * b.value[0] + h11 * span * tb[0],
    h00 * a.value[1] + h10 * span * ta[1] + h01 * b.value[1] + h11 * span * tb[1],
    h00 * a.value[2] + h10 * span * ta[2] + h01 * b.value[2] + h11 * span * tb[2],
  ];
}

function tangent(p0: { frame: number; value: Vec3 }, p1: { frame: number; value: Vec3 }, p2: { frame: number; value: Vec3 }): Vec3 {
  const d01 = p1.frame - p0.frame;
  const d12 = p2.frame - p1.frame;
  if (d01 === 0 || d12 === 0) return [0, 0, 0];
  return [
    ((p1.value[0] - p0.value[0]) / d01 + (p2.value[0] - p1.value[0]) / d12) * 0.5,
    ((p1.value[1] - p0.value[1]) / d01 + (p2.value[1] - p1.value[1]) / d12) * 0.5,
    ((p1.value[2] - p0.value[2]) / d01 + (p2.value[2] - p1.value[2]) / d12) * 0.5,
  ];
}

/**
 * 5-tap binomial smoothing [1,4,6,4,1]/16 applied `passes` times.
 * Edge samples are preserved (one-sided normalization). Works in place on
 * a flat xyz frame-major buffer for one marker's channel.
 */
export function smoothTrack(track: Vec3[], passes = 2): Vec3[] {
  if (track.length < 3 || passes <= 0) return track;
  let src = track;
  let dst: Vec3[] = new Array(track.length);
  const kernel = [1, 4, 6, 4, 1];
  for (let p = 0; p < passes; p++) {
    for (let i = 0; i < src.length; i++) {
      let wx = 0, wy = 0, wz = 0, wsum = 0;
      for (let k = -2; k <= 2; k++) {
        const j = i + k;
        if (j < 0 || j >= src.length) continue;
        const w = kernel[k + 2];
        wx += w * src[j][0];
        wy += w * src[j][1];
        wz += w * src[j][2];
        wsum += w;
      }
      dst[i] = [wx / wsum, wy / wsum, wz / wsum];
    }
    [src, dst] = [dst, src];
  }
  return src;
}

/** Frame-rate independent linear blend helper used by gap filling. */
export function gapFillLinear(before: Vec3, after: Vec3, t: number): Vec3 {
  return lerp(before, after, t);
}
