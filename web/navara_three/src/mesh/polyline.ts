import {
  PolylineMesh as NavaraPolylineMesh,
  PolylineMaterial,
} from "@navaramap/engine";
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  InterleavedBuffer,
  InterleavedBufferAttribute,
  Matrix4,
  ShaderMaterial,
  Vector2,
  Vector3,
  Vector4,
} from "three";
import invariant from "tiny-invariant";

import {
  attachBatchedMaterial,
  registerBatchedMaterial,
  POLYLINE_BATCH_SUPPORT,
  type BatchedAttributeName,
  type BatchTextureSupport,
} from "../batchTexture";
import type { EventContext } from "../event/context";
import { applyLitOption } from "../material";
import { createPolylineMaterialEnhancer } from "../material/enhancer";
import { getWebGPU } from "../utils";

import {
  BatchedFeatureMesh,
  type BatchedFeatureAttributes,
} from "./batchedFeature";
import { GEOMETRY_TYPES } from "./constants";
import { releaseGeometryArraysAfterUpload } from "./releaseGeometryArrays";
import { setupRTECallback } from "./rtcRteHelper";
import {
  createWebgpuBatchSampler,
  eagerAllocateBatchTexture,
  syncWebgpuBatchTexture,
} from "./webgpuBatchTexture";

// Sentinel value for picking coordinate when not picking (reused to avoid allocations)
const PICKING_COORD_SENTINEL = new Vector2(-1, -1);

/**
 * Post-upload CPU-array release for InterleavedBuffers, mirroring the
 * BufferAttribute `disposeArray` idiom in releaseGeometryArrays.ts (which
 * skips interleaved attributes).
 */
function disposeInterleavedArray(this: InterleavedBuffer) {
  (this as unknown as { array: unknown }).array = null;
}

/**
 * Minimum stroke width used while rendering the pick pass. A hairline stroke
 * covers too few pick-buffer texels to click reliably — on the draped path the
 * 512px tile atlas resolves a thin line to scattered partial texels, so the
 * decoded batch id is background almost everywhere the line is visible.
 * Fattening the stroke only for the pick render gives lines the same "click
 * tolerance" pointing devices get elsewhere; the visible width is untouched.
 */
const MIN_PICK_WIDTH = 10;

/**
 * TSL uniform handles + prop source for the WebGPU node material built by
 * initWebGPUMaterial. Values are synced from the enhancer state once per
 * render (see syncWebgpuHandles) because the enhancer remains mounted on the
 * classic material it was created with.
 */
type PolylineWebgpuHandles = {
  uMinMaxHeightAndWidth: { value: Vector3 };
  uMaxWidth: { value: number };
  uAddHeight: { value: number };
  uViewportAndPixelRatio: { value: Vector3 };
  uFrustumNearFar: { value: Vector2 };
  uFrustumRatio: { value: Vector4 };
  gateColor: { value: number };
  gateHeight: { value: number };
  uPickable: { value: number };
  /** Batch texture node; re-pointed if the shared uniform swaps textures. */
  batchTex?: { value: unknown };
  /** Enhancer-mounted classic material; per-frame prop/uniform source. */
  src: ShaderMaterial;
  rte?: {
    matrix: { value: Matrix4 };
    camHigh: { value: Vector3 };
    camLow: { value: Vector3 };
  };
};

type Attributes = BatchedFeatureAttributes<{
  position: BufferAttribute;
  // RTE mode attributes (only present when useRTE=true)
  position_3d_high?: BufferAttribute;
  position_3d_low?: BufferAttribute;
  start_3d_high?: BufferAttribute;
  start_3d_low?: BufferAttribute;
  end_3d_high?: BufferAttribute;
  end_3d_low?: BufferAttribute;
  // Non-RTE mode attributes (only present when useRTE=false)
  start?: BufferAttribute;
  forward_offset?: BufferAttribute;
  // Common attributes (present for non-flat polylines)
  start_normal?: BufferAttribute;
  end_normal_and_texture_coordinate_normalization_x?: BufferAttribute;
  // Always present
  right_normal_and_texture_coordinate_normalization_y: BufferAttribute;
  attrBatchId: BufferAttribute;
}>;

export class PolylineMesh extends BatchedFeatureMesh<
  BufferGeometry<Attributes>,
  ShaderMaterial
> {
  readonly ctx: EventContext;
  /** Material enhancer for managing shader state */
  private _enhancedMaterial?: ReturnType<typeof createPolylineMaterialEnhancer>;
  /** Flag indicating geometry initialization failed - mesh should never be visible */
  private _geometryInitFailed = false;
  /** First-write flag gating the replacing batch height in the WebGPU node
   *  graph (the classic path uses a write-time USE_BATCH_HEIGHT define). */
  private _batchHeightUsed = false;

  constructor(ctx: EventContext) {
    super(new BufferGeometry<Attributes>(), new ShaderMaterial());
    this.ctx = ctx;
  }

  /**
   * Geometry type of this mesh.
   */
  readonly geometryType = GEOMETRY_TYPES.Polyline;

  init(mesh: NavaraPolylineMesh) {
    this.batchLength = mesh.batch_length;
    const geometryResult = this.initGeometry(mesh);

    // If geometry init failed (missing required buffers), mark as permanently invisible
    if (!geometryResult.success) {
      console.warn(
        "PolylineMesh: Failed to initialize geometry due to missing required buffers. Mesh will be permanently invisible.",
      );
      this._geometryInitFailed = true;
      this.visible = false;
    }

    this.initMaterial(mesh, geometryResult.useRTE);

    this.addEventListener("removedFromWorld", () => {
      this.dispose();
    });

    return this;
  }

  private initGeometry(
    mesh: NavaraPolylineMesh,
  ): { success: true; useRTE: boolean } | { success: false; useRTE: false } {
    const { buf } = this.ctx;
    const g = mesh.geometry;
    const position = buf.removeF32(g.position.data);
    const position_high = g.position_high
      ? buf.removeF32(g.position_high.data)
      : null;
    const position_low = g.position_low
      ? buf.removeF32(g.position_low.data)
      : null;
    const start_high = g.start_high ? buf.removeF32(g.start_high.data) : null;
    const start_low = g.start_low ? buf.removeF32(g.start_low.data) : null;
    const end_high = g.end_high ? buf.removeF32(g.end_high.data) : null;
    const end_low = g.end_low ? buf.removeF32(g.end_low.data) : null;
    const start_normals = g.start_normals
      ? buf.removeF32(g.start_normals.data)
      : null;
    const end_normal_and_texture_coordinate_normalization_x =
      g.end_normal_and_texture_coordinate_normalization_x
        ? buf.removeF32(
            g.end_normal_and_texture_coordinate_normalization_x.data,
          )
        : null;
    const right_normal_and_texture_coordinate_normalization_y = buf.removeF32(
      g.right_normal_and_texture_coordinate_normalization_y.data,
    );
    const indices = buf.removeU32(g.indices);
    const batchIds = g.batch_ids ? buf.removeF32(g.batch_ids.data) : undefined;
    const batchIdSize = g.batch_ids ? g.batch_ids.size : 0;
    const batchIndex = g.batch_index
      ? buf.removeU32(g.batch_index.data)
      : undefined;
    const batchIndexSize = g.batch_index ? g.batch_index.size : 0;

    if (
      !position ||
      !right_normal_and_texture_coordinate_normalization_y ||
      !indices
    ) {
      return { success: false, useRTE: false };
    }

    const geometry = this.geometry;
    const useRTE = !!(
      position_high &&
      position_low &&
      start_high &&
      start_low &&
      end_high &&
      end_low
    );

    geometry.setAttribute(
      "position",
      new BufferAttribute(position, g.position.size),
    );

    if (useRTE) {
      // RTE attributes
      if (position_high && position_low) {
        geometry.setAttribute(
          "position_3d_high",
          new BufferAttribute(position_high, 3),
        );
        geometry.setAttribute(
          "position_3d_low",
          new BufferAttribute(position_low, 3),
        );
      }
      if (start_high && start_low) {
        geometry.setAttribute(
          "start_3d_high",
          new BufferAttribute(start_high, 3),
        );
        geometry.setAttribute(
          "start_3d_low",
          new BufferAttribute(start_low, 3),
        );
      }
      if (end_high && end_low) {
        geometry.setAttribute("end_3d_high", new BufferAttribute(end_high, 3));
        geometry.setAttribute("end_3d_low", new BufferAttribute(end_low, 3));
      }
    } else if (g.start && g.forward_offset) {
      const start = buf.removeF32(g.start.data);
      const forward_offset = buf.removeF32(g.forward_offset.data);

      if (!start || !forward_offset) {
        return { success: false, useRTE: false };
      }

      // Non-RTE mode: use regular start attribute
      geometry.setAttribute("start", new BufferAttribute(start, g.start.size));

      geometry.setAttribute(
        "forward_offset",
        new BufferAttribute(forward_offset, g.forward_offset.size),
      );
    }

    if (start_normals && g.start_normals) {
      geometry.setAttribute(
        "start_normal",
        new BufferAttribute(start_normals, g.start_normals.size),
      );
    }
    if (
      end_normal_and_texture_coordinate_normalization_x &&
      g.end_normal_and_texture_coordinate_normalization_x
    ) {
      geometry.setAttribute(
        "end_normal_and_texture_coordinate_normalization_x",
        new BufferAttribute(
          end_normal_and_texture_coordinate_normalization_x,
          g.end_normal_and_texture_coordinate_normalization_x.size,
        ),
      );
    }
    geometry.setAttribute(
      "right_normal_and_texture_coordinate_normalization_y",
      new BufferAttribute(
        right_normal_and_texture_coordinate_normalization_y,
        g.right_normal_and_texture_coordinate_normalization_y.size,
      ),
    );

    if (batchIds) {
      geometry.setAttribute(
        "attrBatchId",
        new BufferAttribute(batchIds, batchIdSize),
      );
    }

    if (batchIndex) {
      this._setBatchIndex(Float32Array.from(batchIndex), batchIndexSize);
    }

    geometry.setIndex(new BufferAttribute(indices, 1));
    // geometry.computeVertexNormals();

    // The renderer lazily calls computeBoundingSphere(), which reads
    // position.array, for every polyline: non-RTE ones through frustum culling
    // (RTE disables it below), and ALL of them through depth sorting —
    // renderer.sortObjects reads geometry.boundingSphere even when
    // frustumCulled is false. Compute it eagerly so that read happens before
    // we drop the CPU arrays. In RTE mode `position` still holds the f32
    // world-space coordinates, so the resulting sphere is valid.
    geometry.computeBoundingSphere();

    // WebGPU guarantees only 8 vertex buffers per pipeline; the polyline
    // attribute set (11-12 separate buffers) exceeds that. Repack into a few
    // InterleavedBuffers (one vertex-buffer slot each) on that backend only —
    // the classic WebGL path keeps the original layout untouched.
    if (this.isWebGPUBackend()) {
      this._interleaveAttributesForWebGPU(useRTE);
    }

    // With the bounding sphere resolved and batch-id data consumed on the GPU,
    // no CPU read survives the first upload. Drop the JS-heap copies to keep a
    // single resident (GPU) copy — see releaseGeometryArraysAfterUpload for the
    // context-loss trade-off.
    releaseGeometryArraysAfterUpload(geometry);

    return { success: true, useRTE };
  }

  /**
   * Repack the polyline attributes into InterleavedBuffers so the WebGPU
   * pipeline stays within the 8 vertex-buffer limit. RTE additionally drops
   * the `position` attribute: the shader never reads it (RTE decodes
   * position_3d_high/low instead) and the bounding sphere is already
   * computed. Interleaved attributes are skipped by
   * releaseGeometryArraysAfterUpload, so the underlying InterleavedBuffers
   * get the same post-upload array release attached here.
   */
  private _interleaveAttributesForWebGPU(useRTE: boolean): void {
    const geometry = this.geometry;
    const groups: string[][] = useRTE
      ? [
          ["position_3d_high", "position_3d_low"],
          ["start_3d_high", "start_3d_low", "end_3d_high", "end_3d_low"],
          [
            "start_normal",
            "end_normal_and_texture_coordinate_normalization_x",
            "right_normal_and_texture_coordinate_normalization_y",
          ],
          ["attrBatchId", "_batchid"],
        ]
      : [
          ["start", "forward_offset"],
          [
            "start_normal",
            "end_normal_and_texture_coordinate_normalization_x",
            "right_normal_and_texture_coordinate_normalization_y",
          ],
          ["attrBatchId", "_batchid"],
        ];

    for (const names of groups) {
      const present = names
        .map((name) => ({
          name: name as keyof Attributes,
          attr: geometry.getAttribute(name as keyof Attributes),
        }))
        .filter(
          (e): e is { name: keyof Attributes; attr: BufferAttribute } =>
            e.attr != null,
        );
      if (present.length < 2) continue;

      const count = present[0].attr.count;
      const stride = present.reduce((s, e) => s + e.attr.itemSize, 0);
      const data = new Float32Array(count * stride);
      let offset = 0;
      for (const { attr } of present) {
        const src = attr.array as Float32Array;
        const size = attr.itemSize;
        for (let i = 0; i < count; i++) {
          for (let c = 0; c < size; c++) {
            data[i * stride + offset + c] = src[i * size + c];
          }
        }
        offset += size;
      }

      const interleaved = new InterleavedBuffer(data, stride);
      interleaved.onUpload(disposeInterleavedArray);
      offset = 0;
      for (const { name, attr } of present) {
        geometry.setAttribute(
          name,
          new InterleavedBufferAttribute(
            interleaved,
            attr.itemSize,
            offset,
          ) as unknown as BufferAttribute,
        );
        offset += attr.itemSize;
      }
    }

    if (useRTE) {
      geometry.deleteAttribute("position");
    }
  }

  private initMaterial(mesh: NavaraPolylineMesh, useRTE: boolean) {
    const { uniforms } = this.ctx;
    const meshMaterial = mesh.material;

    const [minHeight, maxHeight] = meshMaterial.__internal__?.minMaxHeights ?? [
      0, 0,
    ];

    this.castShadow = !!meshMaterial.castShadow;
    this.receiveShadow = !!meshMaterial.receiveShadow;
    applyLitOption(this.material, meshMaterial.lit);

    const isTexturized = mesh.should_be_texturized;

    // Shader selection is handled by enhancer's transformShader
    this.material.depthTest = false;

    // Disable lighting for texturized rendering - the texture will be applied to the lit tile
    this.material.lights = !isTexturized;
    this.material.vertexColors = false;

    // Ignored if it is cloned.
    if (!this._enhancedMaterial) {
      // Create enhanced material with encapsulated state
      const enhancer = createPolylineMaterialEnhancer(this.material);
      this._enhancedMaterial = enhancer;
    }
    const enhancer = this._enhancedMaterial;

    enhancer.mount({
      base: {
        color: meshMaterial.color,
        minMaxHeight: [minHeight, maxHeight],
        addHeight: meshMaterial.height ?? 0,
        width: meshMaterial.width,
        maxWidth: meshMaterial.maxWidth,
        isTexturized,
        drapeRtSize: this.ctx.tileTextureCompositor.size,
        pickable: false,
        useRTE,
        transparent: meshMaterial.transparent,
        opacity: meshMaterial.opacity,
        depthWrite: meshMaterial.depthWrite,
        // External shared uniforms from CommonUniforms
        viewportAndPixelRatio: uniforms.viewportAndPixelRatio,
        frustumNearFar: uniforms.frustumNearFar,
        frustumRatio: uniforms.frustumRatio,
      },
    });

    // Initialize enhancer uniforms early so they're available before onBeforeCompile
    const mutates = enhancer.mutates();
    mutates.updateUniforms(this.material.uniforms, enhancer.states());

    // Set up RTE callback if needed
    const state = enhancer.states();
    let rteCallback: ReturnType<typeof setupRTECallback> | undefined;
    if (state.useRTE) {
      rteCallback = setupRTECallback(
        this,
        (modelViewMatrixRTE, cameraPositionHigh, cameraPositionLow) => {
          mutates.updateRteUniforms(
            modelViewMatrixRTE,
            cameraPositionHigh,
            cameraPositionLow,
            state,
          );
          const w = this.material.userData.nvrWebgpu as
            PolylineWebgpuHandles | undefined;
          if (w?.rte) {
            w.rte.matrix.value.copy(modelViewMatrixRTE);
            w.rte.camHigh.value.copy(cameraPositionHigh);
            w.rte.camLow.value.copy(cameraPositionLow);
          }
        },
        new Matrix4(),
        new Matrix4(),
      );
      this.onBeforeShadow = rteCallback;

      // Disable frustum culling for RTE mode
      this.frustumCulled = false;
    }

    if (this.isWebGPUBackend()) {
      // The GLSL enhancer pipeline (onBeforeCompile) never runs on the WebGPU
      // backend; swap in an equivalent TSL node material instead. The classic
      // material stays alive as the enhancer's state store.
      this.onBeforeRender = (
        renderer,
        scene,
        camera,
        geometry,
        material,
        group,
      ) => {
        this.syncWebgpuHandles();
        rteCallback?.(renderer, scene, camera, geometry, material, group);
      };
      this._initBatchedMaterial();
      this.initWebGPUMaterial(useRTE, meshMaterial.lit, isTexturized);
      this._update(meshMaterial, mesh.active);
      return;
    }

    if (rteCallback) {
      this.onBeforeRender = rteCallback;
    }

    // Set up custom program cache key based on config flags that affect shader defines
    this.material.customProgramCacheKey = enhancer.programCacheKey;

    // Set onBeforeCompile to use enhancer
    this.material.onBeforeCompile = enhancer.transformShader;

    this.ctx.viewContext.applyShadowMaterial(this.material);

    this._initBatchedMaterial();

    this._update(meshMaterial, mesh.active);
  }

  private isWebGPUBackend(): boolean {
    const renderer = this.ctx.viewContext?.getRenderer() as
      { isWebGPURenderer?: boolean } | undefined;
    return !!renderer?.isWebGPURenderer;
  }

  /**
   * WebGPU polyline material: a TSL node material reproducing the classic
   * GLSL enhancer pipeline (polyline.vert/frag.glsl), which never runs on
   * this backend:
   *  - vertex: Cesium shadow-volume screen-space line-width expansion
   *    (start/end/right plane intersection, metersPerPixel width, maxWidth
   *    clamp, end forward-push), height extrusion along the miter height
   *    normal, per-feature batch-texture color/show/opacity/height/lineWidth
   *    (float RGBA bit-decode, see batch_texture_*.glsl), RTE high/low decode
   *    + ellipsoidal horizon culling (both segment endpoints behind the
   *    horizon collapses the clip position to vec4(0));
   *  - fragment: batch color/show/opacity, pick coloring
   *    (nvr_batchIdToColor) folded through uPickable, Lambert lighting with
   *    vNormal = miter height normal.
   * Uniform state is fed per frame from the enhancer by syncWebgpuHandles;
   * RTE matrices ride the same onBeforeRender callback as the classic path.
   * Not ported (logged once): selective-effect emissive.
   */
  private initWebGPUMaterial(
    useRTE: boolean,
    lit: boolean | undefined,
    isTexturized: boolean,
  ): void {
    const { webgpu, tsl: T } = getWebGPU();
    const src = this.material;

    // The node graph fetches the batch texture in the vertex stage with
    // layout rows baked in as constants, so every supported row/slot must be
    // allocated before the material is built (the classic path allocates
    // lazily on first write). Eager identity writes also whiten the enhancer's
    // color uniform (batch-color multiplier convention) — restore it; a real
    // color write re-whitens through the enhancer's batchColorEnabled path.
    this._initBatchDataTexture();
    const colorUniform = src.uniforms.color?.value as Color | undefined;
    const keepColor = colorUniform?.clone();
    eagerAllocateBatchTexture(src, POLYLINE_BATCH_SUPPORT, {
      showOpacity: true,
    });
    if (colorUniform && keepColor) colorUniform.copy(keepColor);
    const batch =
      this.batchLength != null &&
      this.geometry.getAttribute("_batchid") != null
        ? createWebgpuBatchSampler(T, src, this.batchLength)
        : null;

    const handles: PolylineWebgpuHandles = {
      uMinMaxHeightAndWidth: T.uniform(new Vector3(0, 0, 1)) as unknown as {
        value: Vector3;
      },
      uMaxWidth: T.uniform(1000) as unknown as { value: number },
      uAddHeight: T.uniform(0) as unknown as { value: number },
      uViewportAndPixelRatio: T.uniform(new Vector3(1, 1, 1)) as unknown as {
        value: Vector3;
      },
      uFrustumNearFar: T.uniform(new Vector2(1, 1000)) as unknown as {
        value: Vector2;
      },
      uFrustumRatio: T.uniform(new Vector4(1, 1, 1, 1)) as unknown as {
        value: Vector4;
      },
      gateColor: T.uniform(0) as unknown as { value: number },
      gateHeight: T.uniform(0) as unknown as { value: number },
      uPickable: T.uniform(0) as unknown as { value: number },
      src,
    };
    const {
      uMinMaxHeightAndWidth,
      uMaxWidth,
      uAddHeight,
      uViewportAndPixelRatio,
      uFrustumNearFar,
      uFrustumRatio,
      gateColor,
      gateHeight,
      uPickable,
      // TSL's chained node methods don't survive the library's generic
      // typings; the graph is runtime-checked by the node builder instead
      // (same `as any` idiom as the polygon WebGPU material).
      /* eslint-disable @typescript-eslint/no-explicit-any */
    } = handles as unknown as Record<string, any>;

    // Vertex: batch-texture lookups (per-feature color/show/opacity, height,
    // line width). texture().load() (texel fetch) works in the vertex stage
    // and needs no sampler — float32 linear filtering is an optional WebGPU
    // feature. color/show/opacity sample identity defaults (white / visible /
    // 1) until written, so only the replacing attributes (height) and the
    // material-color-replacing batch color need gates.
    let addHeight: any = uAddHeight;
    // Negative batchLineWidth means "use the default width" (see
    // line_width_vertex.glsl + polyline.vert.glsl).
    let batchLineWidth: any = T.float(-1);
    let vBatchColor: any = T.varying(T.vec3(1, 1, 1), "nvr_batchColor");
    let vShow: any = T.varying(T.float(1), "nvr_show");
    let vOpacity: any = T.varying(T.float(1), "nvr_opacity");
    if (batch) {
      handles.batchTex = batch.texNode;
      const colorNode = batch.vec3("color");
      if (colorNode) vBatchColor = T.varying(colorNode, "nvr_batchColor");
      const showOpacity = batch.showOpacity();
      if (showOpacity) {
        vShow = T.varying(showOpacity.show, "nvr_show");
        vOpacity = T.varying(showOpacity.opacity, "nvr_opacity");
      }
      const batchHeight = batch.scalar("height");
      if (batchHeight) addHeight = T.mix(uAddHeight, batchHeight, gateHeight);
      batchLineWidth = batch.scalar("lineWidth") ?? batchLineWidth;
    }

    // line_width_vertex.glsl + polyline.vert.glsl: a non-negative batch line
    // width wins over the material default (negative = sentinel).
    const baseLineWidth = T.select(
      batchLineWidth.greaterThanEqual(0),
      batchLineWidth,
      uMinMaxHeightAndWidth.z,
    );

    const Base = (lit === false || isTexturized
      ? webgpu.MeshBasicNodeMaterial
      : webgpu.MeshLambertNodeMaterial) as unknown as new () => ShaderMaterial;
    let m: any;
    // Seam-clip planes for the fragment stage (non-texturized path only).
    let seamPlanes: {
      startN: unknown;
      startW: unknown;
      endN: unknown;
      endW: unknown;
    } | null = null;

    if (isTexturized) {
      // flatPolyline.vert.glsl: positions in normalized [-1, 1] tile
      // coordinates, miter offset in xy; lighting is applied to the tile the
      // drape bakes into, not to the line.
      m = new Base();
      const miter: any = T.attribute(
        "right_normal_and_texture_coordinate_normalization_y",
      );
      // Positions span [-1, 1] (2.0 units) across the 512-texel render
      // target, so one texel of width is 2.0 / 512.0 in normalized
      // coordinates. Dividing by projectionMatrix[0][0] cancels the parent
      // tile zoom-in magnification (see flatPolyline.vert.glsl).
      const lineWidth = baseLineWidth
        .mul(2.0 / 512.0)
        .div((T.cameraProjectionMatrix as any).element(0).x);
      m.positionNode = T.positionGeometry.add(
        T.vec3(miter.xy.mul(lineWidth.mul(0.5).mul(miter.w)), 0),
      );
    } else {
      // polyline.vert.glsl: Cesium PolylineShadowVolume screen-space width
      // expansion in eye coordinates.
      const startNormal: any = T.attribute("start_normal");
      const endNormalAndX: any = T.attribute(
        "end_normal_and_texture_coordinate_normalization_x",
      );
      const rightNormalAndY: any = T.attribute(
        "right_normal_and_texture_coordinate_normalization_y",
      );

      let ecStart: any;
      let ecEnd: any;
      let offset: any;
      let positionEC: any;
      let startPlaneN: any;
      let endPlaneN: any;
      let rightPlaneN: any;
      // Reference points for the top/bottom vertex classification: RTE works
      // in eye coordinates (positionEC vs ecStart/ecEnd), non-RTE in local
      // coordinates (position vs start/start+forward_offset).
      let refNear: any;
      let refFar: any;
      let positionRaw: any;
      let notCulled: any = T.float(1);

      if (useRTE) {
        const rteMatrix = T.uniform(new Matrix4());
        const camHigh = T.uniform(new Vector3());
        const camLow = T.uniform(new Vector3());
        handles.rte = {
          matrix: rteMatrix as unknown as { value: Matrix4 },
          camHigh: camHigh as unknown as { value: Vector3 },
          camLow: camLow as unknown as { value: Vector3 },
        };
        // u_rteOne (== 1.0) blocks fast-math reassociation of the RTE
        // recombination — see rte_pars_vertex.glsl.
        const uRteOne = T.uniform(1.0);
        const posHigh: any = T.attribute("position_3d_high");
        const posLow: any = T.attribute("position_3d_low");
        const startHigh: any = T.attribute("start_3d_high");
        const startLow: any = T.attribute("start_3d_low");
        const endHigh: any = T.attribute("end_3d_high");
        const endLow: any = T.attribute("end_3d_low");
        const cameraRelative = (high: any, low: any): any =>
          high.sub(camHigh).mul(uRteOne).add(low.sub(camLow));

        // Horizon culling per segment (horizon_culling_pars_vertex.glsl):
        // collapse only when BOTH endpoints are beyond the ellipsoidal
        // horizon. The clip position is zeroed below (gl_Position = vec4(0)).
        const ONE_OVER_WGS84_RADII = T.vec3(
          1 / 6378137.0,
          1 / 6378137.0,
          1 / 6356752.3142451793,
        );
        const camScaled = camHigh.add(camLow).mul(ONE_OVER_WGS84_RADII);
        const a = camScaled.dot(camScaled).sub(1);
        const horizonCulled = (target: any): any =>
          camScaled
            .sub(target.mul(ONE_OVER_WGS84_RADII))
            .dot(camScaled)
            .greaterThan(a);
        notCulled = horizonCulled(startHigh.add(startLow))
          .and(horizonCulled(endHigh.add(endLow)))
          .not()
          .toFloat();

        ecStart = rteMatrix.mul(
          T.vec4(cameraRelative(startHigh, startLow), 1),
        ).xyz;
        ecEnd = rteMatrix.mul(T.vec4(cameraRelative(endHigh, endLow), 1)).xyz;
        offset = ecEnd.sub(ecStart);
        positionEC = rteMatrix.mul(
          T.vec4(cameraRelative(posHigh, posLow), 1),
        ).xyz;
        startPlaneN = rteMatrix.mul(T.vec4(startNormal, 0)).xyz;
        endPlaneN = rteMatrix.mul(T.vec4(endNormalAndX.xyz, 0)).xyz;
        rightPlaneN = rteMatrix.mul(T.vec4(rightNormalAndY.xyz, 0)).xyz;
        refNear = ecStart;
        refFar = ecEnd;
        positionRaw = positionEC;
      } else {
        const start: any = T.attribute("start");
        const forwardOffset: any = T.attribute("forward_offset");
        ecStart = T.modelViewMatrix.mul(T.vec4(start, 1)).xyz;
        // normalMatrix * forward_offset
        offset = T.transformNormalToView(forwardOffset);
        ecEnd = ecStart.add(offset);
        positionEC = T.modelViewMatrix.mul(T.vec4(T.positionGeometry, 1)).xyz;
        startPlaneN = T.transformNormalToView(startNormal);
        endPlaneN = T.transformNormalToView(endNormalAndX.xyz);
        rightPlaneN = T.transformNormalToView(rightNormalAndY.xyz);
        refNear = start;
        refFar = start.add(forwardOffset);
        positionRaw = T.positionGeometry;
      }

      // Start/end/right planes (Hessian form) in eye coordinates.
      const startPlaneW = startPlaneN.dot(ecStart).negate();
      const endPlaneW = endPlaneN.dot(ecEnd).negate();
      // Seam clipping (polyline.frag.glsl, #829): the segment is pushed past
      // both ends below to cover joint gaps; flat varyings let the fragment
      // clip it back to the start/end planes so adjacent segments meet on
      // their shared miter plane.
      const flatVarying = (node: any, name: string): any => {
        const v = T.varying(node, name);
        T.nodeObject(v).setInterpolation("flat");
        return v;
      };
      seamPlanes = {
        startN: flatVarying(startPlaneN, "nvr_startPlaneN"),
        startW: flatVarying(startPlaneW, "nvr_startPlaneW"),
        endN: flatVarying(endPlaneN, "nvr_endPlaneN"),
        endW: flatVarying(endPlaneW, "nvr_endPlaneW"),
      };
      const absStartPlaneDistance = startPlaneN
        .dot(positionEC)
        .add(startPlaneW)
        .abs();
      const absEndPlaneDistance = endPlaneN
        .dot(positionEC)
        .add(endPlaneW)
        .abs();
      const nearerStart = absStartPlaneDistance.lessThan(absEndPlaneDistance);

      const planeDirection: any = T.select(nearerStart, startPlaneN, endPlaneN);
      const upOrDown = rightPlaneN.cross(planeDirection).normalize();
      const normalECBase = planeDirection.cross(upOrDown).normalize();
      const heightNormal: any = (
        T.select(
          nearerStart,
          rightPlaneN.cross(startPlaneN),
          endPlaneN.cross(rightPlaneN),
        ) as any
      ).normalize();

      // Height extrusion: pick min or max height by which side of the miter
      // the vertex sits on (dot of the offset direction with heightNormal).
      const distToRef = positionRaw
        .sub(T.select(nearerStart, refNear, refFar))
        .normalize();
      const height = heightNormal.mul(
        T.select(
          distToRef.dot(heightNormal).greaterThan(0),
          uMinMaxHeightAndWidth.y,
          uMinMaxHeightAndWidth.x,
        ),
      );
      positionEC = positionEC.add(height);
      // Per-feature height offset from the batch texture or the uniform.
      positionEC = positionEC.add(heightNormal.mul(addHeight));

      // vNormal = normalize(heightNormal) (normal_vertex.glsl — the classic
      // shader assigns transformedNormal = heightNormal directly).
      const vNormal = T.varying(heightNormal.normalize(), "nvr_normal");

      // metersPerPixel: pixel size in meters at this eye-space depth.
      const vp = uViewportAndPixelRatio;
      const distToPixel = positionEC.z.negate();
      const inverseNear = T.float(1).div(uFrustumNearFar.x);
      const pixelHeight = distToPixel
        .mul(2)
        .mul(uFrustumRatio.x.mul(inverseNear))
        .div(vp.y.mul(vp.z));
      const pixelWidth = distToPixel
        .mul(2)
        .mul(uFrustumRatio.z.mul(inverseNear))
        .div(vp.x.mul(vp.z));
      const metersPerPixel = T.max(pixelWidth, pixelHeight).mul(vp.z);

      // Distance to push along R, clamped by maxWidth to bound overdraw.
      let lineWidth = baseLineWidth.mul(T.max(T.float(0), metersPerPixel));
      lineWidth = T.min(lineWidth, uMaxWidth);

      // Extend the shadow volume past each end to cover the concave-side gap
      // at joints.
      const forwardDirectionEC = offset.normalize();
      const pushDirection = T.select(nearerStart, T.float(-1), T.float(1));
      positionEC = positionEC.add(
        forwardDirectionEC.mul(pushDirection).mul(lineWidth.mul(0.5)),
      );

      // Distance to push along N.
      lineWidth = lineWidth.div(normalECBase.dot(rightPlaneN));
      const normalEC = normalECBase.mul(T.sign(endNormalAndX.w));
      positionEC = positionEC.add(normalEC.mul(lineWidth));

      const viewPositionNode = positionEC;
      const cullFactorNode = notCulled;
      class RtePolylineNodeMaterial extends Base {
        // positionView is fully computed above (modelView already applied);
        // the standard modelViewMatrix must not touch it.
        setupPositionView(): unknown {
          return viewPositionNode;
        }
        // Horizon-culled segments collapse to gl_Position = vec4(0).
        setupModelViewProjection(): unknown {
          return T.cameraProjectionMatrix
            .mul(T.positionView)
            .mul(cullFactorNode);
        }
      }
      m = new RtePolylineNodeMaterial();
      m.positionNode = viewPositionNode;
      m.normalNode = vNormal;
    }

    // Fragment: pick color (matches pick.glsl nvr_batchIdToColor). The id
    // varying must be FLAT like the classic `flat out float nvr_vBatchId`:
    // batchIds reach 2^24 where f32 has ulp=1, so smooth interpolation can
    // arrive a step off at some pixels and corrupt the low byte (the id then
    // misses the property store — same idiom as pickableMeshWrapper.ts).
    let pickId: any = T.float(0);
    if (this.geometry.getAttribute("attrBatchId") != null) {
      pickId = T.varying(T.attribute("attrBatchId"), "nvr_pickId");
      T.nodeObject(pickId).setInterpolation("flat");
    }
    const pickColor = T.vec3(
      pickId.div(65536).floor().div(255),
      pickId.div(256).mod(256).floor().div(255),
      pickId.mod(256).floor().div(255),
    );
    m.colorNode = T.Fn(() => {
      // Seam clipping (polyline.frag.glsl, #829): discard past the start/end
      // planes so pushed-out joint covers meet on the shared miter plane.
      // positionView is the eye-space position (== -vViewPosition).
      if (seamPlanes) {
        const posEc: any = T.positionView;
        const sp = seamPlanes as Record<string, any>;
        sp.startN
          .dot(posEc)
          .add(sp.startW)
          .lessThan(0)
          .discard();
        sp.endN
          .dot(posEc)
          .add(sp.endW)
          .lessThan(0)
          .discard();
      }
      // show_fragment.glsl: discard hidden / fully transparent features.
      vShow.lessThan(0.5).discard();
      vOpacity.lessThanEqual(0).discard();
      return T.vec4(
        T.materialColor.rgb
          .mul(T.mix(T.vec3(1, 1, 1), vBatchColor, gateColor))
          .mul(uPickable.oneMinus()),
        1,
      );
    })();
    m.emissiveNode =
      // MeshBasicNodeMaterial (flat/texturized) has no `emissive` property —
      // materialEmissive would build a color uniform with an undefined value
      // and crash the WebGPU uniform update. Basic: pick color only, gated by
      // uPickable (colorNode already zeroes diffuse when picking).
      "emissive" in m
        ? T.mix(T.materialEmissive, pickColor, uPickable)
        : pickColor.mul(uPickable);
    m.opacityNode = T.mix(
      T.materialOpacity.mul(vOpacity),
      T.float(1),
      uPickable,
    );
    /* eslint-enable @typescript-eslint/no-explicit-any */

    // Polyline always renders without depth testing (see initMaterial).
    m.depthTest = false;

    // Share the classic material's userData (uPickable etc.) with the node
    // material; batch writes address this.material, so attach it to the
    // classic material's batch texture state (a module-private WeakMap keyed
    // by material, not userData) for updateBatchAttribute to keep landing.
    Object.assign(m.userData, src.userData);
    attachBatchedMaterial(src, m);
    m.userData.nvrWebgpu = handles;
    this.material = m as ShaderMaterial;
  }

  /**
   * Per-render sync of the TSL uniform handles from the enhancer state (which
   * keeps mutating the classic material it was mounted on). No-op unless the
   * WebGPU node material is installed.
   */
  private syncWebgpuHandles(): void {
    const w = this.material.userData.nvrWebgpu as
      PolylineWebgpuHandles | undefined;
    if (!w) return;
    const s = this.getEnhancer().states();
    w.uMinMaxHeightAndWidth.value.set(
      s.minMaxHeight[0],
      s.minMaxHeight[1],
      s.width,
    );
    w.uMaxWidth.value = s.maxWidth;
    w.uAddHeight.value = s.addHeight;
    w.gateColor.value = s.batchColorEnabled ? 1 : 0;
    w.gateHeight.value = this._batchHeightUsed ? 1 : 0;
    w.uPickable.value = s.pickable ? 1 : 0;
    if (w.batchTex) syncWebgpuBatchTexture({ texNode: w.batchTex }, w.src);

    // External shared uniforms (CommonUniforms): the classic material's
    // uniform entries hold live tuple references updated externally per frame.
    const uniforms = w.src.uniforms;
    const vp = uniforms.viewportAndPixelRatio?.value;
    if (vp) w.uViewportAndPixelRatio.value.set(vp[0], vp[1], vp[2]);
    const nf = uniforms.frustumNearFar?.value;
    if (nf) w.uFrustumNearFar.value.set(nf[0], nf[1]);
    const fr = uniforms.frustumRatio?.value;
    if (fr) w.uFrustumRatio.value.set(fr[0], fr[1], fr[2], fr[3]);

    // Props the enhancer writes on its own (classic) material.
    const m = this.material;
    m.transparent = w.src.transparent;
    m.opacity = w.src.opacity;
    m.depthWrite = w.src.depthWrite;
    (m as unknown as { color: Color }).color.copy(
      w.src.uniforms.color.value as Color,
    );
  }

  _getBatchTextureSupport(): BatchTextureSupport {
    return POLYLINE_BATCH_SUPPORT;
  }

  _initBatchDataTexture(): void {
    invariant(this.batchLength != null);
    // Register batchLength; the texture itself is created lazily on the
    // first attribute write.
    const uniform = registerBatchedMaterial(
      this.material,
      { ...this._getBatchTextureSupport(), batchLength: this.batchLength },
      this.ctx.viewContext.getRenderer(),
    );
    this.getEnhancer().update({ base: { batchDataTexture: uniform } });
  }

  _updateBatchAttribute(
    batchId: number,
    attribute: BatchedAttributeName,
    value: number | number[] | boolean,
  ): boolean {
    // Write the texture first: a rejected write must not stamp any define —
    // the shaders have no safety net for an unwritten receiver.
    if (!super._updateBatchAttribute(batchId, attribute, value)) return false;

    if (attribute === "color") {
      // When batch color is first used, set material.color to white
      // (multiplier identity: white * batch color = batch color).
      if (!this.getEnhancer().states().batchColorEnabled) {
        this.getEnhancer().update({
          base: { batchColorEnabled: true, color: 0xffffff },
        });
      }
    }
    if (attribute === "height") {
      // First-write flag gating the replacing batch height in the WebGPU node
      // graph (the classic path uses a write-time USE_BATCH_HEIGHT define).
      this._batchHeightUsed = true;
    }
    return true;
  }

  _update(material: PolylineMaterial, active: boolean) {
    // If geometry initialization failed, keep mesh permanently invisible
    // to prevent WebGL errors from missing attributes/buffers
    if (this._geometryInitFailed) {
      this.visible = false;
      return;
    }

    const enhancer = this.getEnhancer();

    // Update mesh properties (not handled by enhancer).
    // Draped polylines render only through the offscreen bake, never the main
    // scene, and the Rust drape resolve — not this LOD `active` flag — decides
    // which baked tile is composited per terrain region. Tying bake visibility
    // to `active` makes the drape flicker on every LOD swap (the tile goes
    // invisible → bakes empty before its replacement is ready). Keep a draped
    // mesh bakeable whenever its material is shown, so every built tile stays a
    // stable drape source — the vector twin of a loaded raster texture.
    const shown = material.show ?? true;
    this.visible = enhancer.states().isTexturized ? shown : shown && active;
    this.castShadow = !!material.castShadow;
    this.receiveShadow = !!material.receiveShadow;
    applyLitOption(this.material, material.lit);

    const base = enhancer.states();

    // Build update props from material
    const minMaxHeights = material.__internal__?.minMaxHeights;
    enhancer.update({
      base: {
        // `material.color` is used only when `batchColorEnabled` is `false`.
        // Otherwise the color comes from the batch data texture.
        color: base.batchColorEnabled ? undefined : material.color,
        minMaxHeight:
          minMaxHeights !== undefined
            ? [minMaxHeights[0], minMaxHeights[1]]
            : undefined,
        addHeight: material.height,
        width: material.width,
        maxWidth: material.maxWidth,
        transparent: material.transparent,
        opacity: material.opacity,
        depthWrite: material.depthWrite,
        effectIdsMask:
          this.ctx.viewContext.selectiveEffectRegistry?.computeMask(
            material.effectIds ?? [],
          ) ?? 0,
        emissiveColor: material.emissiveColor ?? 0,
        emissiveIntensity: material.emissiveIntensity ?? 0,
      },
    });

    // Update material.lights flag based on isTexturized state
    // (lighting should be disabled for texturized/draped polylines)
    this.material.lights = !base.isTexturized;
  }

  /**
   * Get the enhancer, throwing if not initialized.
   * @throws Error if enhancer is not initialized
   */
  private getEnhancer(): NonNullable<typeof this._enhancedMaterial> {
    if (!this._enhancedMaterial) {
      throw new Error(
        "PolylineMesh material enhancer is not initialized. This usually indicates a failure during construction or geometry/material setup.",
      );
    }
    return this._enhancedMaterial;
  }

  get color() {
    // After the WebGPU material swap the color uniform lives on the classic
    // (enhancer-mounted) material kept in userData.nvrWebgpu.
    const w = this.material.userData?.nvrWebgpu as
      PolylineWebgpuHandles | undefined;
    if (w) return w.src.uniforms.color.value;
    return this.material.uniforms.color.value;
  }

  get draped(): boolean {
    return this.getEnhancer().states().isTexturized;
  }

  get emissiveColor(): number {
    return this.getEnhancer().states().emissiveColor;
  }
  get emissiveIntensity(): number {
    return this.getEnhancer().states().emissiveIntensity;
  }
  get effectIdsMask(): number {
    return this.getEnhancer().states().effectIdsMask;
  }

  /** Visible stroke width to restore after a pick-pass width bump. */
  private _pickSavedWidth?: number;

  onBeforePicking(pickingCoord?: Vector2) {
    const enhancer = this.getEnhancer();
    const width = enhancer.states().width;
    if (width < MIN_PICK_WIDTH) {
      this._pickSavedWidth = width;
      enhancer.update({ base: { pickable: true, width: MIN_PICK_WIDTH } });
    } else {
      enhancer.update({ base: { pickable: true } });
    }
    this.needsUpdate();

    const mutates = enhancer.mutates();
    if (pickingCoord) {
      mutates.setPickingCoord(pickingCoord);
    } else {
      mutates.setPickingCoord(PICKING_COORD_SENTINEL);
    }
  }

  onAfterPicking() {
    const enhancer = this.getEnhancer();
    if (this._pickSavedWidth !== undefined) {
      enhancer.update({
        base: { pickable: false, width: this._pickSavedWidth },
      });
      this._pickSavedWidth = undefined;
    } else {
      enhancer.update({ base: { pickable: false } });
    }
    this.needsUpdate();
    enhancer.mutates().setPickingCoord(PICKING_COORD_SENTINEL);
  }

  clone() {
    const cloned = new PolylineMesh(this.ctx) as this;
    cloned.geometry = this.geometry;
    cloned.material = this.material;
    cloned._enhancedMaterial = this._enhancedMaterial;
    return cloned;
  }

  dispose() {
    this.ctx.viewContext.removeShadowMaterial(this.material);
  }
}
