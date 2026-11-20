import type { CameraState } from './renderer';

/** Drag-to-orbit and wheel-to-zoom controller attached to a canvas. */
export class OrbitControls {
  private dragging = false;
  private lastX = 0;
  private lastY = 0;

  constructor(
    private canvas: HTMLCanvasElement,
    private camera: CameraState,
    private onChange: () => void,
  ) {
    canvas.addEventListener('pointerdown', this.onDown);
    canvas.addEventListener('pointermove', this.onMove);
    window.addEventListener('pointerup', this.onUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
  }

  dispose(): void {
    this.canvas.removeEventListener('pointerdown', this.onDown);
    this.canvas.removeEventListener('pointermove', this.onMove);
    window.removeEventListener('pointerup', this.onUp);
    this.canvas.removeEventListener('wheel', this.onWheel);
  }

  private onDown = (e: PointerEvent): void => {
    this.dragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.canvas.setPointerCapture(e.pointerId);
  };

  private onMove = (e: PointerEvent): void => {
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.camera.yaw -= dx * 0.01;
    this.camera.pitch = Math.min(1.4, Math.max(-1.4, this.camera.pitch + dy * 0.01));
    this.onChange();
  };

  private onUp = (): void => {
    this.dragging = false;
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.camera.distance = Math.min(8, Math.max(0.6, this.camera.distance * (1 + Math.sign(e.deltaY) * 0.08)));
    this.onChange();
  };
}
