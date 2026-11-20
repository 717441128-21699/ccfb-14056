/**
 * WebGL2 渲染器:实例化小球渲染标记点、GL_LINES 渲染骨段与接触标注,
 * 内置轨道相机(左键旋转、右键平移、滚轮缩放)。
 */
import type { MarkerClip, SkeletonClip } from '../core/types.js';
import { JOINT_COUNT } from '../core/rig.js';

const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aInstance;
layout(location=2) in vec3 aColor;
layout(location=3) in float aScale;
uniform mat4 uViewProj;
out vec3 vColor;
void main() {
  vColor = aColor;
  vec3 p = aPos * aScale + aInstance;
  gl_Position = uViewProj * vec4(p, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in vec3 vColor;
out vec4 outColor;
void main() { outColor = vec4(vColor, 1.0); }`;

const LINE_VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
uniform mat4 uViewProj;
out vec3 vColor;
void main() { vColor = aColor; gl_Position = uViewProj * vec4(aPos, 1.0); }`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? 'shader error');
  return sh;
}
function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link error');
  return p;
}

/** 低精度单位球(经纬细分),用于实例化标记点。 */
function unitSphere(lat = 6, lon = 8): { verts: Float32Array; indices: Uint16Array } {
  const verts: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= lat; i++) {
    const theta = (i / lat) * Math.PI;
    for (let j = 0; j <= lon; j++) {
      const phi = (j / lon) * Math.PI * 2;
      verts.push(Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi));
    }
  }
  for (let i = 0; i < lat; i++) {
    for (let j = 0; j < lon; j++) {
      const a = i * (lon + 1) + j;
      const b = a + lon + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return { verts: new Float32Array(verts), indices: new Uint16Array(indices) };
}

function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}
function mat4LookAt(eye: number[], center: number[], up: number[]): Float32Array {
  const z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
  let zl = Math.hypot(z0, z1, z2); const z = [z0 / zl, z1 / zl, z2 / zl];
  const x0 = up[1] * z[2] - up[2] * z[1], x1 = up[2] * z[0] - up[0] * z[2], x2 = up[0] * z[1] - up[1] * z[0];
  const xl = Math.hypot(x0, x1, x2); const x = [x0 / xl, x1 / xl, x2 / xl];
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
    -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1,
  ]);
}
function mul4(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++)
    o[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
  return o;
}

export interface RenderOptions {
  showMarkers: boolean;
  showSkeleton: boolean;
  showIssues: boolean;
  showLocks: boolean;
  colorFill: boolean;
}

export class Renderer {
  private gl: WebGL2RenderingContext;
  private sphereProg: WebGLProgram;
  private lineProg: WebGLProgram;
  private sphere: { verts: Float32Array; indices: Uint16Array };
  private vaoSphere: WebGLVertexArrayObject;
  private instBuf: WebGLBuffer;
  private colorBuf: WebGLBuffer;
  private scaleBuf: WebGLBuffer;
  private lineVao: WebGLVertexArrayObject;
  private linePosBuf: WebGLBuffer;
  private lineColorBuf: WebGLBuffer;
  private camera = { yaw: 0.6, pitch: 0.35, dist: 4.5, target: [0, 0.9, 0] as number[] };
  private maxMarkers = 0;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('当前浏览器不支持 WebGL2');
    this.gl = gl;
    this.sphereProg = program(gl, VERT, FRAG);
    this.lineProg = program(gl, LINE_VERT, FRAG);
    this.sphere = unitSphere();

    // 球体基础几何 VAO
    this.vaoSphere = gl.createVertexArray()!;
    gl.bindVertexArray(this.vaoSphere);
    const baseBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, baseBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.sphere.verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const idxBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.sphere.indices, gl.STATIC_DRAW);
    this.instBuf = gl.createBuffer()!;
    this.colorBuf = gl.createBuffer()!;
    this.scaleBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuf);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(2, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.scaleBuf);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(3, 1);

    // 线段 VAO
    this.lineVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.lineVao);
    this.linePosBuf = gl.createBuffer()!;
    this.lineColorBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.linePosBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineColorBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    gl.enable(gl.DEPTH_TEST);
    this.attachControls(canvas);
  }

  private attachControls(canvas: HTMLCanvasElement): void {
    let dragging: 'rotate' | 'pan' | null = null;
    let lx = 0, ly = 0;
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('mousedown', (e) => {
      dragging = e.button === 2 ? 'pan' : 'rotate';
      lx = e.clientX; ly = e.clientY;
    });
    window.addEventListener('mouseup', () => { dragging = null; });
    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      if (dragging === 'rotate') {
        this.camera.yaw -= dx * 0.008;
        this.camera.pitch = Math.max(-1.4, Math.min(1.4, this.camera.pitch + dy * 0.008));
      } else {
        const s = this.camera.dist * 0.0015;
        this.camera.target[0] -= dx * s;
        this.camera.target[2] -= dy * s;
      }
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.dist = Math.max(0.5, Math.min(20, this.camera.dist * (1 + Math.sign(e.deltaY) * 0.08)));
    }, { passive: false });
  }

  private viewProj(): Float32Array {
    const gl = this.gl;
    const { yaw, pitch, dist, target } = this.camera;
    const eye = [
      target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
      target[1] + dist * Math.sin(pitch),
      target[2] + dist * Math.cos(pitch) * Math.cos(yaw),
    ];
    const aspect = gl.canvas.width / Math.max(1, gl.canvas.height);
    const proj = mat4Perspective(Math.PI / 4, aspect, 0.05, 100);
    const view = mat4LookAt(eye, target, [0, 1, 0]);
    return mul4(proj, view);
  }

  resize(): void {
    const gl = this.gl;
    const c = gl.canvas as HTMLCanvasElement;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.floor(c.clientWidth * dpr), h = Math.floor(c.clientHeight * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; gl.viewport(0, 0, w, h); }
  }

  render(
    frame: number,
    markers: MarkerClip | null,
    skel: SkeletonClip | null,
    fillMethod: Uint8Array | null,
    issueJoints: Set<number>,
    lockJoints: Set<number>,
    opts: RenderOptions,
  ): void {
    const gl = this.gl;
    this.resize();
    gl.clearColor(0.078, 0.09, 0.114, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const vp = this.viewProj();

    // 收集线段(骨骼 + 穿模 + 锁定)
    const linePos: number[] = [];
    const lineColor: number[] = [];
    const pushLine = (a: ArrayLike<number>, b: ArrayLike<number>, col: number[]): void => {
      linePos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      lineColor.push(...col, ...col);
    };

    if (skel && opts.showSkeleton) {
      const jp = skel.jointPositions;
      const parent = (this as unknown as { _parents?: number[] })._parents;
      if (parent) for (let j = 1; j < JOINT_COUNT; j++) {
        const pa = parent[j];
        if (pa < 0) continue;
        pushLine(jp.subarray(frame * JOINT_COUNT * 3 + pa * 3, frame * JOINT_COUNT * 3 + pa * 3 + 3),
          jp.subarray(frame * JOINT_COUNT * 3 + j * 3, frame * JOINT_COUNT * 3 + j * 3 + 3),
          [0.55, 0.62, 0.72]);
      }
    }

    // 标记点实例数据
    if (markers && opts.showMarkers) {
      const M = markers.markerNames.length;
      const inst: number[] = [];
      const colors: number[] = [];
      const scales: number[] = [];
      const fillColor: Record<number, number[]> = {
        0: [0.85, 0.88, 0.92], // 原始观测
        1: [0.3, 0.6, 1.0],    // 刚体补点
        2: [0.94, 0.71, 0.16], // 轨迹补点
        3: [0.69, 0.49, 1.0],  // 先验补点
      };
      const defaultCol = [0.85, 0.88, 0.92];
      for (let m = 0; m < M; m++) {
        if (!markers.observed[frame * M + m]) continue;
        const o = (frame * M + m) * 3;
        inst.push(markers.positions[o], markers.positions[o + 1], markers.positions[o + 2]);
        const method = fillMethod ? fillMethod[frame * M + m] : 0;
        colors.push(...(opts.colorFill ? (fillColor[method] ?? defaultCol) : defaultCol));
        scales.push(0.018);
      }
      this.drawSpheres(inst, colors, scales, vp);
    }

    // 穿模关节高亮(红色小球由线段十字表示)
    if (skel && opts.showIssues) {
      const jp = skel.jointPositions;
      for (const j of issueJoints) {
        const c = jp.subarray(frame * JOINT_COUNT * 3 + j * 3, frame * JOINT_COUNT * 3 + j * 3 + 3);
        const r = 0.05;
        pushLine([c[0] - r, c[1], c[2]], [c[0] + r, c[1], c[2]], [1, 0.36, 0.36]);
        pushLine([c[0], c[1] - r, c[2]], [c[0], c[1] + r, c[2]], [1, 0.36, 0.36]);
      }
    }
    // 足部锁定标注(绿色)
    if (skel && opts.showLocks) {
      const jp = skel.jointPositions;
      for (const j of lockJoints) {
        const c = jp.subarray(frame * JOINT_COUNT * 3 + j * 3, frame * JOINT_COUNT * 3 + j * 3 + 3);
        const r = 0.04;
        pushLine([c[0], c[1], c[2] - r], [c[0], c[1], c[2] + r], [0.22, 0.76, 0.46]);
      }
    }

    if (linePos.length) this.drawLines(linePos, lineColor, vp);
  }

  setParents(parents: number[]): void {
    (this as unknown as { _parents: number[] })._parents = parents;
  }

  private drawSpheres(inst: number[], colors: number[], scales: number[], vp: Float32Array): void {
    const gl = this.gl;
    gl.useProgram(this.sphereProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.sphereProg, 'uViewProj'), false, vp);
    gl.bindVertexArray(this.vaoSphere);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(inst), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.scaleBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(scales), gl.DYNAMIC_DRAW);
    const count = inst.length / 3;
    gl.drawElementsInstanced(gl.TRIANGLES, this.sphere.indices.length, gl.UNSIGNED_SHORT, 0, count);
    this.maxMarkers = Math.max(this.maxMarkers, count);
  }

  private drawLines(pos: number[], colors: number[], vp: Float32Array): void {
    const gl = this.gl;
    gl.useProgram(this.lineProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, 'uViewProj'), false, vp);
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.linePosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineColorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.LINES, 0, pos.length / 3);
  }
}
