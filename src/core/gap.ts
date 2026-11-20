import type { Gap, MarkerClip } from './types.js';

/** 扫描 observed 掩码,列出每个标记点的连续丢点空档。 */
export function detectGaps(clip: MarkerClip): Gap[] {
  const { frameCount: F, markerNames, observed } = clip;
  const M = markerNames.length;
  const gaps: Gap[] = [];
  for (let m = 0; m < M; m++) {
    let start = -1;
    for (let f = 0; f <= F; f++) {
      const missing = f < F && observed[f * M + m] === 0;
      if (missing && start < 0) start = f;
      if (!missing && start >= 0) {
        gaps.push({ markerIndex: m, startFrame: start, endFrame: f - 1, length: f - start });
        start = -1;
      }
    }
  }
  gaps.sort((a, b) => a.startFrame - b.startFrame || a.markerIndex - b.markerIndex);
  return gaps;
}
