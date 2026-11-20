import { lookAt, mat4FromQuatPos, perspective, type Mat4 } from '../math/mat4';
import type { Vec3 } from '../math/vec3';
import type { SkeletonDefinition } from '../core/types';

const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
uniform mat4 uViewProj;
uniform mat4 uModel;
uniform vec3 uColor;
uniform float uScale;
out vec3 vNormal;
out vec3 vColor;
void main() {
  vec3 p = aPosition * uScale;
  gl_Position = uViewProj * uModel * vec4(p, 1.0);
  vNormal = mat3(uModel) * aNormal;
  vColor = uColor;
}`;

const FRAG = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vColor;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  float light = 0.45 + 0.55 * max(dot(n, normalize(vec3(0.4, 0.8, 0.3))), 0.0);
  outColor = vec4(vColor * light, 1.0);
}`;

const LINE_VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uViewProj;
void main() { gl_Position = uViewProj * vec4(aPos, 1.0); }`;

const LINE_FRAG = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 outColor;
void main() { outColor = vec4(uColor, 1.0); }`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) ?? 'shader compile failed');
  }
  return sh;
}

function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p) ?? 'link failed');
  }
  return p;
}

function sphereGeometry(lat = 10, lon = 12): { pos: Float32Array; norm: Float32Array; idx: Uint16Array } {
  const pos: number[] = [];
  const norm: number[] = [];
  const idx: number[] = [];
  for (let y = 0; y <= lat; y++) {
    const theta = (y / lat) * Math.PI;
    const sy = Math.cos(theta);
    const r = Math.sin(theta);
    for (let x = 0; x <= lon; x++) {
      const phi = (x / lon) * Math.PI * 2;
      const px = r * Math.cos(phi);
      const pz = r * Math.sin(phi);
      pos.push(px, sy, pz);
      norm.push(px, sy, pz);
    }
  }
  for (let y = 0; y < lat; y++) {
    for (let x = 0; x < lon; x++) {
      const a = y * (lon + 1) + x;
      const b = a + lon + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  return { pos: new Float32Array(pos), norm: new Float32Array(norm), idx: new Uint16Array(idx) };
}

export interface CameraState {
  target: Vec3;
  distance: number;
  yaw: number;
  pitch: number;
}

export class SceneRenderer {
  private gl: WebGL2RenderingContext;
  private meshProg: WebGLProgram;
  private lineProg: WebGLProgram;
  private sphere: { pos: Float32Array; norm: Float32Array; idx: Uint16Array };
  private vao: WebGLVertexArrayObject;
  private lineVao: WebGLVertexArrayObject;
  private lineBuffer: WebGLBuffer;
  camera: CameraState = { target: [0, 0.9, 0], distance: 3.2, yaw: 0.6, pitch: 0.35 };

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.meshProg = program(gl, VERT, FRAG);
    this.lineProg = program(gl, LINE_VERT, LINE_FRAG);
    this.sphere = sphereGeometry();

    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const pbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, pbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.sphere.pos, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const nbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, nbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.sphere.norm, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    const ibo = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.sphere.idx, gl.STATIC_DRAW);

    this.lineVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.lineVao);
    this.lineBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  private viewProj(): Mat4 {
    const { target, distance, yaw, pitch } = this.camera;
    const eye: Vec3 = [
      target[0] + distance * Math.cos(pitch) * Math.sin(yaw),
      target[1] + distance * Math.sin(pitch),
      target[2] + distance * Math.cos(pitch) * Math.cos(yaw),
    ];
    const view = lookAt(eye, target, [0, 1, 0]);
    const proj = perspective(Math.PI / 4, this.canvas.width / this.canvas.height, 0.05, 100);
    return mat4MulLocal(proj, view);
  }

  private drawSphere(center: Vec3, radius: number, color: Vec3, viewProj: Mat4): void {
    const gl = this.gl;
    gl.useProgram(this.meshProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.meshProg, 'uViewProj'), false, viewProj);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.meshProg, 'uModel'), false, mat4FromQuatPos([0, 0, 0, 1], center));
    gl.uniform3fv(gl.getUniformLocation(this.meshProg, 'uColor'), color);
    gl.uniform1f(gl.getUniformLocation(this.meshProg, 'uScale'), radius);
    gl.bindVertexArray(this.vao);
    gl.drawElements(gl.TRIANGLES, this.sphere.idx.length, gl.UNSIGNED_SHORT, 0);
  }

  private drawLines(points: Float32Array, color: Vec3, viewProj: Mat4): void {
    const gl = this.gl;
    gl.useProgram(this.lineProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, 'uViewProj'), false, viewProj);
    gl.uniform3fv(gl.getUniformLocation(this.lineProg, 'uColor'), color);
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, points, gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.LINES, 0, points.length / 3);
  }

  render(state: RenderState): void {
    const gl = this.gl;
    this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.07, 0.08, 0.1, 1);
    gl.enable(gl.DEPTH_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const vp = this.viewProj();

    // Ground grid.
    this.drawLines(buildGrid(), [0.2, 0.22, 0.26], vp);

    if (state.markers) this.renderMarkers(state, vp);
    if (state.skeleton) this.renderSkeleton(state, vp);
    if (state.extraLines) {
      for (const line of state.extraLines) this.drawLines(line.points, line.color, vp);
    }
  }

  private renderMarkers(state: RenderState, vp: Mat4): void {
    const { clip, frame } = state.markers!;
    const mc = clip.markerNames.length;
    const highlight = state.highlight;
    const colorFor = (mi: number): Vec3 => {
      if (highlight?.has(clip.markerNames[mi])) return [1, 0.35, 0.2];
      return clip.valid[frame * mc + mi] ? [0.25, 0.85, 0.95] : [0.9, 0.7, 0.2];
    };
    for (let m = 0; m < mc; m++) {
      const base = frame * mc + m;
      if (!clip.valid[base] && !state.showFilled) continue;
      const o = base * 3;
      this.drawSphere([clip.positions[o], clip.positions[o + 1], clip.positions[o + 2]], 0.012, colorFor(m), vp);
    }
  }

  private renderSkeleton(state: RenderState, vp: Mat4): void {
    const { positions, definition, frame, contacts } = state.skeleton!;
    const jc = definition.joints.length;
    const lines: number[] = [];
    const contactJoints = new Set<number>();
    if (contacts) {
      for (const c of contacts) {
        if (frame < c.start || frame > c.end) continue;
        const prefix = c.side === 'L' ? 'l' : 'r';
        for (const s of ['Ankle', 'Foot', 'Toe']) contactJoints.add(definition.jointIndex.get(prefix + s)!);
      }
    }
    for (let j = 1; j < jc; j++) {
      const p = definition.joints[j].parent!;
      const a = (frame * jc + p) * 3;
      const b = (frame * jc + j) * 3;
      lines.push(positions[a], positions[a + 1], positions[a + 2]);
      lines.push(positions[b], positions[b + 1], positions[b + 2]);
    }
    this.drawLines(new Float32Array(lines), [0.7, 0.85, 0.6], vp);
    for (let j = 0; j < jc; j++) {
      const o = (frame * jc + j) * 3;
      const color: Vec3 = contactJoints.has(j) ? [0.2, 0.95, 0.45] : [0.85, 0.75, 0.55];
      this.drawSphere([positions[o], positions[o + 1], positions[o + 2]], definition.joints[j].radius * 0.6, color, vp);
    }
  }
}

function mat4MulLocal(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

function buildGrid(): Float32Array {
  const pts: number[] = [];
  const size = 2;
  const step = 0.25;
  for (let i = -size; i <= size; i += step) {
    pts.push(i, 0, -size, i, 0, size);
    pts.push(-size, 0, i, size, 0, i);
  }
  return new Float32Array(pts);
}

export interface SkeletonRenderState {
  positions: Float32Array;
  definition: SkeletonDefinition;
  frame: number;
  contacts?: { side: 'L' | 'R'; start: number; end: number }[];
}

export interface RenderState {
  markers?: {
    clip: import('../core/types').MarkerClip;
    frame: number;
  };
  skeleton?: SkeletonRenderState;
  extraLines?: { points: Float32Array; color: Vec3 }[];
  highlight?: Set<string>;
  showFilled?: boolean;
}
