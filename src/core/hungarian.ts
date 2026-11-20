/**
 * Hungarian algorithm for the minimum-weight rectangular assignment problem.
 * O(n^3) Jonker–Volgenant style via the classic Kuhn–Munkres augmentation;
 * problem sizes here are at most ~6x6 (markers within a rigid cluster).
 *
 * Returns assignment row -> col, or -1 for unassigned rows when n > m.
 */
export function hungarian(cost: number[][], n: number, m: number): number[] {
  const size = Math.max(n, m);
  const u = new Array(size + 1).fill(0);
  const v = new Array(size + 1).fill(0);
  const p = new Array(size + 1).fill(0);
  const way = new Array(size + 1).fill(0);
  const INF = Number.POSITIVE_INFINITY;

  for (let i = 1; i <= size; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(size + 1).fill(INF);
    const used = new Array(size + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;
      for (let j = 1; j <= size; j++) {
        if (used[j]) continue;
        const cur = costAt(cost, i0 - 1, j - 1, n, m) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= size; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const assignment = new Array(n).fill(-1);
  for (let j = 1; j <= size; j++) {
    const row = p[j] - 1;
    const col = j - 1;
    if (row >= 0 && row < n && col < m) assignment[row] = col;
  }
  return assignment;
}

function costAt(cost: number[][], i: number, j: number, n: number, m: number): number {
  if (i < n && j < m) return cost[i][j];
  // Padding for rectangular problems: neutral finite cost.
  return 1e6;
}
