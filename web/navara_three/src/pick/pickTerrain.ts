import type { Nullable } from "@navaramap/core";
import {
  WebGLRenderer,
  WebGLRenderTarget,
  Vector2,
  Vector3,
  Vector4,
  PerspectiveCamera,
  Scene,
  RGBAFormat,
  Texture,
  ShaderMaterial,
  PlaneGeometry,
  Mesh,
  OrthographicCamera,
} from "three";

class DepthPickPass {
  private quad: Mesh;
  private scene: Scene;
  private camera: OrthographicCamera;
  private sampleTarget: WebGLRenderTarget;
  private material: ShaderMaterial;

  constructor() {
    // Create reusable resources
    this.material = new ShaderMaterial({
      uniforms: {
        tDepth: { value: null },
        samplePos: { value: new Vector2() },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDepth;
        uniform vec2 samplePos;
        varying vec2 vUv;
        
        void main() {
          vec4 depthColor = texture2D(tDepth, samplePos);
          gl_FragColor = depthColor;
        }
      `,
    });

    const geometry = new PlaneGeometry(2, 2);
    this.quad = new Mesh(geometry, this.material);

    this.scene = new Scene();
    this.scene.add(this.quad);

    this.camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.sampleTarget = new WebGLRenderTarget(1, 1, { format: RGBAFormat });
  }

  update(depthTexture: Texture, samplePos: Vector2) {
    this.material.uniforms.tDepth.value = depthTexture;
    this.material.uniforms.samplePos.value.copy(samplePos);
  }

  render(renderer: WebGLRenderer): Uint8Array {
    renderer.setRenderTarget(this.sampleTarget);
    renderer.render(this.scene, this.camera);
    renderer.setRenderTarget(null);

    // Read the pixel
    const pixels = new Uint8Array(4);
    renderer.readRenderTargetPixels(this.sampleTarget, 0, 0, 1, 1, pixels);

    return pixels;
  }

  dispose() {
    this.quad.geometry.dispose();
    this.material.dispose();
    this.sampleTarget.dispose();
  }
}

/**
 * Reconstructs the world-space position for a WebGPU depth sample.
 * `depth` is the NDC depth read from the scene target's depth texture
 * ([0, 1], row 0 = screen top); `x`/`y` are CSS pixels. Mirrors
 * `TerrainPicker._reconstructWorldPosition` but with the WebGPU conventions:
 * clip z = depth directly (no ×2−1) and the logarithmic depth formula is
 * Ulrich's variant used by TSL (`log2(-viewZ/near)/log2(far/near)`).
 */
export function reconstructWebgpuWorldPosition(
  x: number,
  y: number,
  depth: number,
  renderer: WebGLRenderer,
  camera: PerspectiveCamera,
): Vector3 {
  const size = renderer.getDrawingBufferSize(new Vector2());
  const pixelRatio = renderer.getPixelRatio();
  const texelCenter = (cssCoord: number, sizePx: number) =>
    Math.max(0, Math.min(sizePx - 1, Math.floor(cssCoord * pixelRatio))) + 0.5;
  const ndcX = (texelCenter(x, size.x) / size.x) * 2 - 1;
  const ndcY = -((texelCenter(y, size.y) / size.y) * 2 - 1);

  const near = camera.near;
  const far = camera.far;
  const logarithmic = !!(
    renderer as { logarithmicDepthBuffer?: boolean }
  ).logarithmicDepthBuffer;
  // Inverse of TSL's viewZToLogarithmicDepth / perspectiveDepthToViewZ.
  const viewZ = logarithmic
    ? -near * Math.pow(far / near, depth)
    : (near * far) / ((far - near) * depth - far);

  // Any clip z lands on the same view-space ray; scale it to the known
  // viewZ (same trick as the WebGL picker).
  const eye = new Vector4(ndcX, ndcY, depth, 1).applyMatrix4(
    camera.projectionMatrixInverse,
  );
  if (eye.w !== 0) eye.divideScalar(eye.w);
  const scale = viewZ / eye.z;
  return new Vector3(eye.x * scale, eye.y * scale, viewZ).applyMatrix4(
    camera.matrixWorld,
  );
}

/**
 * Analytic ray/WGS84-ellipsoid intersection, used as the WebGPU
 * pickDepthPosition fallback: the scene depth texture only contains
 * depth-writing objects (terrain tiles don't write depth), so over bare
 * ground the GPU sample misses — but the zoom-to-cursor handler still
 * needs a ground distance there or the zoom overshoots below the
 * ellipsoid. `x`/`y` are CSS pixels; returns the ECEF hit point or null
 * when the ray misses the ellipsoid (e.g. pointing at the sky).
 */
export function intersectRayEllipsoid(
  x: number,
  y: number,
  renderer: WebGLRenderer,
  camera: PerspectiveCamera,
): Nullable<Vector3> {
  const size = renderer.getDrawingBufferSize(new Vector2());
  const pixelRatio = renderer.getPixelRatio();
  const rect = (
    renderer as { domElement?: HTMLCanvasElement }
  ).domElement?.getBoundingClientRect();
  const cssW = rect?.width || size.x / pixelRatio;
  const cssH = rect?.height || size.y / pixelRatio;
  const relX = x - (rect?.left ?? 0);
  const relY = y - (rect?.top ?? 0);
  const ndcX = (relX / cssW) * 2 - 1;
  const ndcY = -((relY / cssH) * 2 - 1);

  const v = new Vector4(ndcX, ndcY, 1, 1).applyMatrix4(
    camera.projectionMatrixInverse,
  );
  if (v.w !== 0) v.divideScalar(v.w);
  const dir = new Vector3(v.x, v.y, v.z).transformDirection(camera.matrixWorld);
  const o = camera.position; // ECEF world space

  // WGS84: x²/a² + y²/a² + z²/b² = 1
  const a = 6378137.0;
  const b = 6356752.314245;
  const aa = a * a;
  const bb = b * b;
  const A = (dir.x * dir.x + dir.y * dir.y) / aa + (dir.z * dir.z) / bb;
  const B = 2 * ((o.x * dir.x + o.y * dir.y) / aa + (o.z * dir.z) / bb);
  const C = (o.x * o.x + o.y * o.y) / aa + (o.z * o.z) / bb - 1;
  const disc = B * B - 4 * A * C;
  if (disc < 0) return null;
  const sqrt = Math.sqrt(disc);
  let t = (-B - sqrt) / (2 * A);
  if (t < 0) t = (-B + sqrt) / (2 * A);
  if (t < 0) return null;
  return new Vector3(o.x + dir.x * t, o.y + dir.y * t, o.z + dir.z * t);
}

export class TerrainPicker {  private depthPickPass: DepthPickPass;

  constructor() {
    this.depthPickPass = new DepthPickPass();
  }
  pick(
    x: number,
    y: number,
    renderer: WebGLRenderer,
    depthTexture: Texture,
    camera: PerspectiveCamera,
  ): Nullable<Vector3> {
    const logDepthOrDepth = this._sampleDepthAt(x, y, renderer, depthTexture);
    if (logDepthOrDepth === null || logDepthOrDepth > 0.99) {
      return null;
    }

    return this._reconstructWorldPosition(
      x,
      y,
      logDepthOrDepth,
      renderer,
      camera,
    );
  }

  // Ref: https://github.com/mrdoob/three.js/blob/f38421e7bf5bc37aac7d4ebbe66ad0cc15550c39/src/renderers/shaders/ShaderChunk/packing.glsl.js#L56-L58
  private _unpackRGBAToDepth(rgba: Uint8Array): number {
    // Constants
    const UnpackDownscale = 255 / 256; // 0..1 -> fraction (excluding 1)
    const PackFactors = [1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0];

    // Calculate unpack factors
    const UnpackFactors4 = [
      UnpackDownscale / PackFactors[0],
      UnpackDownscale / PackFactors[1],
      UnpackDownscale / PackFactors[2],
      1.0 / PackFactors[3],
    ];

    // readRenderTargetPixels returns bytes 0..255; a GLSL texel channel is byte / 255.
    return (
      (rgba[0] / 255.0) * UnpackFactors4[0] +
      (rgba[1] / 255.0) * UnpackFactors4[1] +
      (rgba[2] / 255.0) * UnpackFactors4[2] +
      (rgba[3] / 255.0) * UnpackFactors4[3]
    );
  }

  // Clamped device-pixel center for a CSS-pixel screen coordinate. Sampling
  // and ray reconstruction must both use this exact point: the depth copy
  // target is linearly filtered, and any off-center coordinate would blend
  // neighboring packed-RGBA texels into meaningless depth values.
  private _texelCenter(cssCoord: number, size: number, pixelRatio: number) {
    return (
      Math.max(0, Math.min(size - 1, Math.floor(cssCoord * pixelRatio))) + 0.5
    );
  }

  // Helper function to sample depth from depth texture at screen position
  private _sampleDepthAt(
    x: number,
    y: number,
    renderer: WebGLRenderer,
    depthTexture: Texture,
  ): number | null {
    if (!depthTexture) {
      return null;
    }

    const width = renderer.getContext().drawingBufferWidth;
    const height = renderer.getContext().drawingBufferHeight;
    const pixelRatio = renderer.getPixelRatio();

    const centerX = this._texelCenter(x, width, pixelRatio);
    const centerY = this._texelCenter(y, height, pixelRatio);

    // Update the depth pick pass with current parameters
    const samplePos = new Vector2(centerX / width, 1.0 - centerY / height); // Flip Y
    this.depthPickPass.update(depthTexture, samplePos);

    // Render and get pixels
    const pixels = this.depthPickPass.render(renderer);

    // Unpack RGBA to depth using the same formula as the shader
    return this._unpackRGBAToDepth(pixels);
  }

  dispose() {
    this.depthPickPass.dispose();
  }

  private _reconstructWorldPosition(
    x: number,
    y: number,
    depth: number,
    renderer: WebGLRenderer,
    camera: PerspectiveCamera,
  ): Vector3 {
    const near = camera.near;
    const far = camera.far;

    const width = renderer.getContext().drawingBufferWidth;
    const height = renderer.getContext().drawingBufferHeight;
    const pixelRatio = renderer.getPixelRatio();

    // Convert the sampled texel's center to NDC [-1,1], so the reconstructed
    // ray goes through the exact pixel whose depth was read.
    const screenCoords = new Vector2(
      (this._texelCenter(x, width, pixelRatio) / width) * 2.0 - 1.0,
      -((this._texelCenter(y, height, pixelRatio) / height) * 2.0 - 1.0),
    );

    let clipCoords: Vector4;

    if (renderer.capabilities.logarithmicDepthBuffer) {
      const logDepthBufFC = 2.0 / Math.log2(far + 1.0);

      const linearDepth = Math.pow(2, depth / (logDepthBufFC * 0.5)) - 1.0;
      const depthFromCamera = linearDepth + near;

      clipCoords = new Vector4(
        screenCoords.x,
        screenCoords.y,
        depth * 2.0 - 1.0, // Convert depth [0,1] to NDC [-1,1]
        1.0,
      );

      const eyeCoordinate = clipCoords
        .clone()
        .applyMatrix4(camera.projectionMatrixInverse);

      if (eyeCoordinate.w !== 0) {
        eyeCoordinate.divideScalar(eyeCoordinate.w);
      }

      // For logarithmic depth, we need to scale the eye coordinate by the actual depth
      const eyeZ = -depthFromCamera; // Negative because camera looks down -Z
      const scaleFactor = eyeZ / eyeCoordinate.z;
      eyeCoordinate.x *= scaleFactor;
      eyeCoordinate.y *= scaleFactor;
      eyeCoordinate.z = eyeZ;

      // Transform to world space
      return new Vector3(
        eyeCoordinate.x,
        eyeCoordinate.y,
        eyeCoordinate.z,
      ).applyMatrix4(camera.matrixWorld);
    } else {
      // Linear depth buffer case
      clipCoords = new Vector4(
        screenCoords.x,
        screenCoords.y,
        depth * 2.0 - 1.0,
        1.0,
      );

      // Transform to eye coordinates
      const eyeCoordinate = clipCoords
        .clone()
        .applyMatrix4(camera.projectionMatrixInverse);
      eyeCoordinate.divideScalar(eyeCoordinate.w);

      // Transform to world space
      return new Vector3(
        eyeCoordinate.x,
        eyeCoordinate.y,
        eyeCoordinate.z,
      ).applyMatrix4(camera.matrixWorld);
    }
  }
}
