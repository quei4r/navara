import HillshadeParsFragment from "@shaders/glsl/chunks/hillshade_pars_fragment.glsl";
import {
  DataTexture,
  Mesh,
  OrthographicCamera,
  PlaneGeometry,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  UnsignedByteType,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
} from "three";
import type { MeshBasicNodeMaterial } from "three/webgpu";

import type { HillshadeConfig } from "../event/HillshadeContext";

import { getWebGPU } from "./webgpuLoader";

type RendererLike = WebGLRenderer & { isWebGPURenderer?: boolean };

// Sentinel value for invalid/no-data heights (must match dem_util.glsl)
const INVALID_HEIGHT = -999999.0;
const INVALID_THRESHOLD = -999998.0;
// Artifact detection threshold (must match dem_util.glsl)
const MAX_REASONABLE_DIFF = 1000.0;

/**
 * TSL node group for the WebGPU path, created lazily (WebGPU path only) so
 * the WebGL path never touches the dynamically imported node system.
 * The factory has no declared return type on purpose: letting TS infer the
 * exact generic instantiation of each node keeps Fn() return inference
 * intact downstream (an `any`-typed node degrades it to a union the TSL
 * math overloads reject).
 */
function createHillshadeNodes() {
  const { texture, uniform } = getWebGPU().tsl;
  return {
    dem: texture(
      new DataTexture(
        new Uint8Array([0, 0, 0, 255]),
        1,
        1,
        RGBAFormat,
        UnsignedByteType,
      ),
    ),
    texelSize: uniform(new Vector2(0, 0)),
    metersPerTexel: uniform(1.0),
    outputSize: uniform(new Vector2(1, 1)),
    contentSizeMinus1: uniform(new Vector2(1, 1)),
    padOffset: uniform(new Vector2(0, 0)),
    maxCoord: uniform(new Vector2(0, 0)),
    rgbScaler: uniform(new Vector3(256, 1, 1 / 256)),
    boundary: uniform(0),
    minOffset: uniform(0),
    maxOffset: uniform(0),
    epsilon: uniform(1.0),
    offset: uniform(-32768),
  };
}

type HillshadeNodes = ReturnType<typeof createHillshadeNodes>;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/**
 * Generator for hillshade normal maps using offscreen rendering.
 * Converts DEM textures into pre-computed normal maps to improve runtime performance.
 *
 * WebGL path: GLSL ShaderMaterial (original implementation).
 * WebGPU path: TSL NodeMaterial port of the same math — ShaderMaterial is not
 * supported by WebGPURenderer's NodeBuilder. The TSL version keeps the exact
 * same texture-layout conventions (texel row 0 holds DEM v=1 data) so normal
 * maps are interchangeable between backends.
 */
export class HillshadeNormalMapGenerator {
  private renderer: RendererLike;
  private isWebGPU: boolean;
  private scene: Scene;
  private camera: OrthographicCamera;
  private material: ShaderMaterial | null = null;
  private quad: Mesh;

  // WebGPU (TSL) state — nodes created only on the WebGPU path
  private nodeMaterial: MeshBasicNodeMaterial | null = null;
  private nodes: HillshadeNodes | null = null;

  constructor(renderer: RendererLike) {
    this.renderer = renderer;
    this.isWebGPU = renderer.isWebGPURenderer === true;

    // Setup scene and camera for offscreen rendering
    this.scene = new Scene();
    this.camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);

    if (this.isWebGPU) {
      this.nodes = createHillshadeNodes();
      this.nodeMaterial = this.buildNodeMaterial();
    } else {
      // Create shader material for normal computation
      // Uses hillshade_pars_fragment.glsl for DEM decoding
      this.material = new ShaderMaterial({
        uniforms: {
          uDemTexture: { value: null },
          uTexelSize: { value: [0, 0] },
          uMetersPerTexel: { value: 1.0 },
          uOutputSize: { value: [0, 0] }, // Output render target size (content size without padding)
          // Hillshade decoder uniforms (will be set per-generation)
          uHillshadeRGBScaler: { value: [256, 1, 1 / 256] },
          uHillshadeBoundary: { value: 0 },
          uHillshadeMinOffset: { value: 0 },
          uHillshadeMaxOffset: { value: 0 },
          uHillshadeEpsilon: { value: 1.0 },
          uHillshadeOffset: { value: -32768 },
        },
        vertexShader: `
          void main() {
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          #define USE_HILLSHADE 1

          uniform sampler2D uDemTexture;
          uniform vec2 uTexelSize;
          uniform float uMetersPerTexel;
          uniform vec2 uOutputSize;

          // Import DEM decoding and normal computation from hillshade shader
          ${HillshadeParsFragment}

          void main() {
            ivec2 texSize = textureSize(uDemTexture, 0);

            // Compute UV from gl_FragCoord using correct OUTPUT size (not DEM padded size)
            // gl_FragCoord ranges from (0.5, 0.5) to (width-0.5, height-0.5)
            // Map to UV [0,1] spanning pixel centers: (fragCoord - 0.5) / (size - 1)
            vec2 pixelCoord = gl_FragCoord.xy - 0.5;
            vec2 uv = pixelCoord / (uOutputSize - 1.0);

            // Flip Y to match DEM texture coordinate system (top-down)
            // gl_FragCoord is OpenGL coords (y=0 at bottom), but DEM texture is top-down (y=0 at top)
            uv.y = 1.0 - uv.y;

            // Check if this is valid terrain data
            float testHeight = sampleHeightBilinear(uDemTexture, uv, texSize);

            if (!isValidHeight(testHeight)) {
              // Invalid data (ocean/no-data), output default upward normal (0, 0, 1)
              // Store directly in RGB channels, mapped from [-1,1] to [0,1]
              gl_FragColor = vec4(0.5, 0.5, 1.0, 1.0);
              return;
            }

            // Compute normal from DEM
            vec3 normal = computeNormalFromDEM(uDemTexture, uv, uTexelSize, uMetersPerTexel);

            // Store normal directly in RGB channels (linear, can use hardware bilinear filtering)
            // Map from [-1,1] to [0,1] for 8-bit storage
            gl_FragColor = vec4(normal * 0.5 + 0.5, 1.0);
          }
        `,
      });
    }

    // Create fullscreen quad
    const geometry = new PlaneGeometry(2, 2);
    const quadMaterial = this.nodeMaterial ?? this.material;
    if (!quadMaterial) {
      throw new Error("HillshadeNormalMapGenerator: no material created");
    }
    this.quad = new Mesh(geometry, quadMaterial);
    this.scene.add(this.quad);
  }

  /**
   * TSL port of the GLSL normal-map shader (hillshade_pars_fragment.glsl +
   * dem_util.glsl). Padding detection and content extents are precomputed on
   * the CPU and passed as uniforms instead of textureSize()/bit tricks.
   */
  private buildNodeMaterial(): MeshBasicNodeMaterial {
    const { webgpu, tsl } = getWebGPU();
    const {
      abs,
      clamp,
      dot,
      float,
      floor,
      Fn,
      fract,
      If,
      ivec2,
      mix,
      normalize,
      screenCoordinate,
      select,
      vec2,
      vec3,
      vec4,
    } = tsl;
    const nodes = this.nodes;
    if (!nodes) {
      throw new Error(
        "buildNodeMaterial requires WebGPU-path nodes (createHillshadeNodes)",
      );
    }
    const nDem = nodes.dem;
    const nContentSizeMinus1 = nodes.contentSizeMinus1;
    const nPadOffset = nodes.padOffset;
    const nMaxCoord = nodes.maxCoord;
    const nTexelSize = nodes.texelSize;
    const nMetersPerTexel = nodes.metersPerTexel;
    const nOutputSize = nodes.outputSize;
    const nRGBScaler = nodes.rgbScaler;
    const nBoundary = nodes.boundary;
    const nMinOffset = nodes.minOffset;
    const nMaxOffset = nodes.maxOffset;
    const nEpsilon = nodes.epsilon;
    const nOffset = nodes.offset;

    // decodeDEMHeight + decodeHeightForHillshade
    const decodeHeight = Fn(([color]: [any]) => {
      const rgb = color.rgb.mul(255.0);
      const x = dot(rgb, nRGBScaler);
      const h = select(
        x.greaterThan(nBoundary),
        x.add(nMaxOffset),
        x.add(nMinOffset),
      )
        .mul(nEpsilon)
        .add(nOffset);
      // |x - boundary| <= 1.0 marks no-data (GLSL: epsilon_cmp = 1.0)
      const atBoundary = abs(x.sub(nBoundary)).lessThanEqual(float(1.0));
      return select(atBoundary, float(INVALID_HEIGHT), h);
    });

    const isValidHeight = (h: any) => h.greaterThan(float(INVALID_THRESHOLD));

    // sampleHeightBilinear: prepareDEMBilinear + interpolateDEMHeights
    const sampleHeight = Fn(([uv]: [any]) => {
      const pixelCoord = uv.mul(nContentSizeMinus1).add(nPadOffset);
      const pixelFloor = floor(pixelCoord);
      const pixelFrac: any = fract(pixelCoord);

      // Clamp in float space, then truncate to int texel coords
      // (equivalent to GLSL ivec2-clamp since pixelFloor holds integral values)
      const fZero = vec2(0.0, 0.0);
      const p00 = ivec2(clamp(pixelFloor, fZero, nMaxCoord));
      const p10 = ivec2(
        clamp(pixelFloor.add(vec2(1.0, 0.0)), fZero, nMaxCoord),
      );
      const p01 = ivec2(
        clamp(pixelFloor.add(vec2(0.0, 1.0)), fZero, nMaxCoord),
      );
      const p11 = ivec2(
        clamp(pixelFloor.add(vec2(1.0, 1.0)), fZero, nMaxCoord),
      );

      const h00 = decodeHeight(nDem.load(p00)).toVar();
      // Artifact detection then invalid-neighbor fallback, per dem_util.glsl
      const fixNeighbor = (h: any) => {
        const jumped = isValidHeight(h).and(
          abs(h.sub(h00)).greaterThan(float(MAX_REASONABLE_DIFF)),
        );
        const fixed = select(jumped, h00, h);
        return select(isValidHeight(fixed), fixed, h00);
      };
      const h10 = fixNeighbor(decodeHeight(nDem.load(p10))).toVar();
      const h01 = fixNeighbor(decodeHeight(nDem.load(p01))).toVar();
      const h11 = fixNeighbor(decodeHeight(nDem.load(p11))).toVar();

      const h0 = mix(h00, h10, pixelFrac.x);
      const h1 = mix(h01, h11, pixelFrac.x);
      const result = mix(h0, h1, pixelFrac.y);

      // Base (top-left) sample invalid -> whole sample invalid
      return select(isValidHeight(h00), result, float(INVALID_HEIGHT));
    });

    // computeNormalFromDEM (3x3 Sobel)
    const computeNormal = Fn(([uv]: [any]) => {
      const ts = nTexelSize;
      const a = sampleHeight(uv.add(vec2(ts.x.negate(), ts.y))).toVar();
      const b = sampleHeight(uv.add(vec2(0.0, ts.y))).toVar();
      const c = sampleHeight(uv.add(vec2(ts.x, ts.y))).toVar();
      const d = sampleHeight(uv.add(vec2(ts.x.negate(), 0.0))).toVar();
      const e = sampleHeight(uv).toVar();
      const f = sampleHeight(uv.add(vec2(ts.x, 0.0))).toVar();
      const g = sampleHeight(
        uv.add(vec2(ts.x.negate(), ts.y.negate())),
      ).toVar();
      const h = sampleHeight(uv.add(vec2(0.0, ts.y.negate()))).toVar();
      const i = sampleHeight(uv.add(vec2(ts.x, ts.y.negate()))).toVar();

      // Replace invalid neighbors with the center sample
      const aF = select(isValidHeight(a), a, e);
      const bF = select(isValidHeight(b), b, e);
      const cF = select(isValidHeight(c), c, e);
      const dF = select(isValidHeight(d), d, e);
      const fF = select(isValidHeight(f), f, e);
      const gF = select(isValidHeight(g), g, e);
      const hF = select(isValidHeight(h), h, e);
      const iF = select(isValidHeight(i), i, e);

      const dX = cF.add(fF).add(fF).add(iF).sub(aF).sub(dF).sub(dF).sub(gF);
      const dY = gF.add(hF).add(hF).add(iF).sub(aF).sub(bF).sub(bF).sub(cF);

      const slopeX = dX.mul(0.25).div(nMetersPerTexel);
      const slopeY = dY.mul(0.25).div(nMetersPerTexel);

      const result = vec3(0.0, 0.0, 1.0).toVar();
      If(isValidHeight(e), () => {
        result.assign(normalize(vec3(slopeX.negate(), slopeY, float(1.0))));
      });
      return result;
    });

    const material = new webgpu.MeshBasicNodeMaterial({ toneMapped: false });
    material.colorNode = Fn(() => {
      // Same mapping as the GLSL path: pixel centers mapped to [0,1] over the
      // output size. screenCoordinate follows WebGPU convention (y=0 at top),
      // and the flip is kept so texel row 0 of the output holds DEM v=1 data —
      // identical layout to the WebGL-produced normal maps.
      const uv0 = screenCoordinate.sub(0.5).div(nOutputSize.sub(1.0));
      const uv = vec2(uv0.x, float(1.0).sub(uv0.y));
      const normal = computeNormal(uv);
      return vec4(normal.mul(0.5).add(0.5), 1.0);
    })();

    return material;
  }

  /**
   * Render normal map to an existing RenderTarget (GPU-only, no readback)
   * Used by HillshadeContext's RenderTarget pool to avoid GPU→CPU→GPU round-trip
   * @param renderTarget - Target to render to (owned by HillshadeContext)
   * @param demTexture - Source DEM texture (padded)
   * @param metersPerTexel - Meters per texel for normal calculation
   * @param hillshadeConfig - Hillshade decoder configuration
   */
  renderToTarget(
    renderTarget: WebGLRenderTarget,
    demTexture: DataTexture,
    metersPerTexel: number,
    hillshadeConfig: HillshadeConfig,
  ): void {
    const contentWidth = renderTarget.width;
    const contentHeight = renderTarget.height;

    // Calculate texel size for DEM sampling
    const texelSize = 1.0 / (contentWidth - 1);

    if (this.nodeMaterial && this.nodes) {
      // WebGPU/TSL path: update uniform nodes
      const n = this.nodes;
      n.dem.value = demTexture;
      n.texelSize.value.set(texelSize, texelSize);
      n.metersPerTexel.value = metersPerTexel;
      n.outputSize.value.set(contentWidth, contentHeight);

      // Precompute padding layout (GLSL derives this from textureSize())
      const paddedSize = demTexture.image.width;
      const hasPadding = !isPowerOfTwo(paddedSize);
      const contentSize = hasPadding ? paddedSize - 2 : paddedSize;
      n.contentSizeMinus1.value.set(contentSize - 1, contentSize - 1);
      n.padOffset.value.set(hasPadding ? 1 : 0, hasPadding ? 1 : 0);
      n.maxCoord.value.set(paddedSize - 1, paddedSize - 1);

      n.rgbScaler.value.set(
        hillshadeConfig.rgbScaler[0],
        hillshadeConfig.rgbScaler[1],
        hillshadeConfig.rgbScaler[2],
      );
      n.boundary.value = hillshadeConfig.boundary;
      n.minOffset.value = hillshadeConfig.minOffset;
      n.maxOffset.value = hillshadeConfig.maxOffset;
      n.epsilon.value = hillshadeConfig.epsilon;
      n.offset.value = hillshadeConfig.offset;
    } else if (this.material) {
      const material = this.material;

      // Update shader uniforms
      material.uniforms.uDemTexture.value = demTexture;
      material.uniforms.uTexelSize.value = [texelSize, texelSize];
      material.uniforms.uMetersPerTexel.value = metersPerTexel;
      material.uniforms.uOutputSize.value = [contentWidth, contentHeight];

      // Update hillshade decoder uniforms from Rust config
      material.uniforms.uHillshadeRGBScaler.value = hillshadeConfig.rgbScaler;
      material.uniforms.uHillshadeBoundary.value = hillshadeConfig.boundary;
      material.uniforms.uHillshadeMinOffset.value = hillshadeConfig.minOffset;
      material.uniforms.uHillshadeMaxOffset.value = hillshadeConfig.maxOffset;
      material.uniforms.uHillshadeEpsilon.value = hillshadeConfig.epsilon;
      material.uniforms.uHillshadeOffset.value = hillshadeConfig.offset;
    }

    const currentRenderTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(renderTarget);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(currentRenderTarget);
  }

  /**
   * Debug: Save normal map to file
   * @param renderTarget - RenderTarget to read from
   * @param filename - Output filename
   */
  dbgSaveNormalMap(renderTarget: WebGLRenderTarget, filename: string): void {
    const width = renderTarget.width;
    const height = renderTarget.height;
    const pixels = new Uint8Array(width * height * 4);

    this.renderer.readRenderTargetPixels(
      renderTarget,
      0,
      0,
      width,
      height,
      pixels,
    );

    // Create canvas and write pixels
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(width, height);
    imageData.data.set(pixels);
    ctx.putImageData(imageData, 0, 0);

    // Download
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  dispose(): void {
    this.material?.dispose();
    this.nodeMaterial?.dispose();
    this.quad.geometry.dispose();
  }
}
