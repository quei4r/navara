import {
  PointMesh as NavaraPointMesh,
  BillboardMesh as NavaraBillboardMesh,
} from "@navaramap/engine";
import { degreeToRadian } from "@navaramap/three-api";
import {
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  Object3D,
  ShaderMaterial,
  BufferAttribute,
  type BufferGeometry,
  Color,
  DataTexture,
  type Material,
  PerspectiveCamera,
  Vector2,
  Vector3,
} from "three";
import invariant from "tiny-invariant";

import {
  readBatchScalar,
  readBatchShowOpacity,
  registerBatchedMaterial,
  SPRITE_BATCH_SUPPORT,
  updateBatchAttribute,
  type BatchedAttributeName,
  type BatchTextureSupport,
} from "../../batchTexture";
import {
  DECLUTTER_FADE_MS,
  type DeclutterCandidate,
  type DeclutterParticipant,
} from "../../declutter/types";
import type { EventContext } from "../../event/context";
import { createInstancedSpriteMaterialEnhancer } from "../../material/enhancer";
import type { CustomObject3DEventMap } from "../../object3DEvent";
import { getWebGPU } from "../../utils";
import { buildBatchIndexMap } from "../batchIndexMap";
import { GEOMETRY_TYPES, type GeometryType } from "../constants";
import { PickableMesh } from "../pickableMesh";

import { BillboardAtlas, type AtlasRect } from "./billboardAtlas";
import { loadAtlasImageFromUrl } from "./billboardAtlasImageLoader";

type SpriteGeometryType = Extract<
  GeometryType,
  typeof GEOMETRY_TYPES.Point | typeof GEOMETRY_TYPES.Billboard
>;

export type InstancedSpriteOptions = {
  renderOrder?: number;
  ctx: EventContext;
  geometryType: SpriteGeometryType;
};

type PositionsInfo = {
  position:
    | Float32Array<ArrayBufferLike>
    | {
        high: Float32Array<ArrayBufferLike>;
        low: Float32Array<ArrayBufferLike>;
      };
  batchIDs: Float32Array<ArrayBufferLike> | null;
  positionSize: number;
  batchIDSize: number;
  nPositions: number;
  RTE: boolean;
};

/** Reusable Vector2 to avoid per-frame allocations in onBeforeRender. */
const _tmpSize = new Vector2();

/**
 * TSL uniform handles + prop source for the WebGPU node material built by
 * _initWebGPUMaterial. Values are synced from the enhancer's uniform refs
 * (which live on the classic material it stays mounted on) once per render —
 * see _syncWebgpuHandles.
 */
type InstancedSpriteWebgpuHandles = {
  uRTCCenter: { value: Vector3 };
  uRTCCenterView: { value: Vector3 };
  uEyeRTEHigh: { value: Vector3 };
  uEyeRTELow: { value: Vector3 };
  uScale: { value: number };
  uCenter: { value: Vector2 };
  uSizeInMeters: { value: number };
  uOffsetDepth: { value: number };
  uAlphaTest: { value: number };
  uFarPlane: { value: number };
  uAtlasSize: { value: Vector2 };
  uFovRad: { value: number };
  uScreenHeightPx: { value: number };
  uPickable: { value: number };
  /** Texture node whose `.value` is swapped as the billboard atlas grows. */
  spriteTex?: { value: unknown };
  src: ShaderMaterial;
};

/** Stand-in for a not-yet-packed billboard atlas. Empty atlas rects cull
 *  their instances in the vertex stage, so it is never visibly sampled. */
let _placeholderTexture: DataTexture | null = null;
const getPlaceholderTexture = (): DataTexture => {
  if (!_placeholderTexture) {
    _placeholderTexture = new DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1);
    _placeholderTexture.needsUpdate = true;
  }
  return _placeholderTexture;
};

/** Reusable scratch for per-feature style writes. */
const _tmpColorArray: [number, number, number] = [0, 0, 0];

// Coupled with crates/navara_feature/src/geometry/point.rs::pixel_to_world
export class InstancedSpriteMesh
  extends Mesh<BufferGeometry, Material | Material[], CustomObject3DEventMap>
  implements PickableMesh, DeclutterParticipant
{
  private _geometryType: SpriteGeometryType = GEOMETRY_TYPES.Point;
  /**
   * Feature (batch) index → this feature's instance ids. A feature owns
   * multiple instances for MultiPoint geometry and for points derived from
   * line/polygon vertices via `geometryTypes`, so per-feature styling must
   * fan out to all of them. `null` means instances and features are 1:1.
   */
  private _batchIndexToInstances: Map<number, number[]> | null = null;
  /**
   * Per-instance feature (batch) index, mirroring the `_batchid` attribute for
   * CPU-side reads (declutter). `null` means instances and features are 1:1.
   */
  private _instanceBatchIndex: Float32Array | null = null;
  /** Feature count — the batch data texture's column count. */
  private _batchLength = 0;
  /** Instance count of the current geometry; bounds the identity fallback. */
  private _instanceCount = 0;
  private _atlas?: BillboardAtlas;
  private _defaultUrl?: string;
  /** Atlas rect of the current default image; re-applied to an instance when
   * its per-feature override is cleared. */
  private _defaultRect?: AtlasRect;
  /** Instance ids whose image was overridden per-feature; the default image
   * from the material no longer applies to them. */
  private _imageOverrides = new Set<number>();
  /** Latest override URL requested per instance. An async pack only applies
   * if it still matches, so a newer override or a clear wins over slow loads. */
  private _requestedImageUrls = new Map<number, string>();
  /** Forwards the atlas byte footprint to the engine's memory ledger; wired
   * by the feature-added handler once the owning entity bits are known. */
  private _atlasBytesReporter?: (bytes: number) => void;
  private _reportedAtlasBytes = 0;
  private _active = true;
  readonly ctx: EventContext;
  /** Material enhancer for encapsulated state management */
  private _enhancedMaterial?: ReturnType<
    typeof createInstancedSpriteMaterialEnhancer
  >;
  /** The classic ShaderMaterial the enhancer stays mounted on when the WebGPU
   *  node material replaced it for rendering (state store + visibility). */
  private _classicMaterial?: ShaderMaterial;
  /** Per-instance world anchors in ECEF meters (f64, 3 per instance), kept in
   *  sync with the position attributes for the declutter pass. */
  private _anchors: Float64Array | null = null;
  /** Whether this mesh's instances participate in screen-space decluttering. */
  private _declutter = false;
  /** Layer-level placement priority from the material. */
  private _declutterPriority = 0;
  /** Per-instance priorities set through the evaluator (NaN = no override,
   *  fall back to the layer value). Lazily allocated on first use. */
  private _declutterPriorityOverrides: Float32Array | null = null;
  /** Per-instance fade targets for `instanceDeclutterHide` (0 = shown,
   *  1 = hidden); the attribute animates toward these in stepDeclutterFade. */
  private _declutterTargets: Float32Array | null = null;
  /** True while any instance's hide factor may differ from its target. */
  private _declutterAnimating = false;

  constructor(options: InstancedSpriteOptions) {
    super();
    this.renderOrder = options.renderOrder ?? this.renderOrder;
    this.ctx = options.ctx;
    this._geometryType = options.geometryType;
    this.ctx.declutter?.register(this);
    // `processObjectRemoved` dispatches this for every removed mesh; it is the
    // reliable teardown signal (this class's dispose() is not called there).
    this.addEventListener("removedFromWorld", () => {
      this.ctx.declutter?.unregister(this);
    });
  }

  setActive(active: boolean) {
    this._active = active;
    this.updateVisibility();
    this.ctx.declutter?.markDirty();
  }

  /**
   * Set the geometry type for this mesh.
   *
   * This affects how FeatureEvaluator provides meshGeomType to evaluators,
   * which is used by MapLibreStylePlugin to determine which properties to apply
   * (e.g., billboard vs text rendering for symbol layers).
   *
   * Changing this after initialization is supported and will affect subsequent
   * feature evaluations, but does NOT automatically trigger re-evaluation.
   * Call layer.forceUpdate() if you need to re-evaluate existing features.
   */
  setGeometryType(type: SpriteGeometryType): void {
    this._geometryType = type;
  }

  /**
   * Get the geometry type of this mesh.
   * Defaults to GEOMETRY_TYPES.Point if not explicitly set.
   */
  get geometryType(): SpriteGeometryType {
    return this._geometryType;
  }

  // --- DeclutterParticipant ---

  collectDeclutterCandidates(out: DeclutterCandidate[]): void {
    if (!this.visible || !this._declutter || !this._anchors) return;
    const enhancer = this._enhancedMaterial;
    if (!enhancer) return;
    const material = this.material as ShaderMaterial;

    const state = enhancer.states();
    const cx = Math.min(Math.max(state.center[0], -0.5), 0.5);
    const cy = Math.min(Math.max(state.center[1], -0.5), 0.5);
    // Mirror of instancedSprite.vert.glsl — aspect is per-instance
    // (from the atlas rect), not a material-level uniform; there is no
    // material-wide "aspect" state to read.
    const uvRect = state.billboard
      ? (this.geometry?.getAttribute("instanceUvRect") as
          InstancedBufferAttribute | undefined)
      : undefined;
    const anchors = this._anchors;
    const batchIndices = this._instanceBatchIndex;
    const overrides = this._declutterPriorityOverrides;
    const targets = this._declutterTargets;
    const count = Math.min(this._instanceCount, anchors.length / 3);

    for (let i = 0; i < count; i++) {
      // Style values mirror the shader: read the batch data texture, falling
      // back to the material-level state where a slot was never allocated.
      const batchIndex = batchIndices ? batchIndices[i] : i;
      const showOpacity = readBatchShowOpacity(material, batchIndex);
      if (showOpacity && showOpacity.show < 0.5) continue; // hidden by user `show`
      const batchSize = readBatchScalar(material, batchIndex, "size");
      const size =
        batchSize !== undefined && batchSize >= 0.0 ? batchSize : state.scale;
      if (size <= 0.0) continue;
      const addHeight =
        readBatchScalar(material, batchIndex, "height") ?? state.addHeight;

      const override = overrides ? overrides[i] : Number.NaN;
      const rectH = uvRect ? uvRect.getW(i) : 0;
      const rectW = uvRect ? uvRect.getZ(i) : 0;
      // Mirror of the shader's empty-rect cull: an instance with no image
      // packed yet is not drawn, so it must not reserve declutter space and
      // hide the labels around it.
      if (uvRect && (rectW <= 0.0 || rectH <= 0.0)) continue;
      const aspect = uvRect ? rectW / rectH : 1.0;

      // Mirror of instancedSprite.vert.glsl:122-125 — the quad spans
      // (position.xy - center) * vec2(aspect, 1) * size around the anchor.
      out.push({
        anchorX: anchors[i * 3],
        anchorY: anchors[i * 3 + 1],
        anchorZ: anchors[i * 3 + 2],
        addHeight,
        minX: (-0.5 - cx) * aspect * size,
        maxX: (0.5 - cx) * aspect * size,
        minY: (-0.5 - cy) * size,
        maxY: (0.5 - cy) * size,
        sizeInMeters: state.sizeInMeters,
        // NaN-safe: an unset override falls back to the layer priority.
        priority: Number.isNaN(override) ? this._declutterPriority : override,
        isShown: targets ? targets[i] === 0 : true,
        owner: this,
        handle: i,
      });
    }
  }

  applyDeclutter(handle: number, hidden: boolean): void {
    this.setDeclutterHiddenByInstance(handle, hidden);
  }

  stepDeclutterFade(deltaMs: number): boolean {
    if (!this._declutterAnimating) return false;
    const attr = this.geometry?.getAttribute("instanceDeclutterHide") as
      InstancedBufferAttribute | undefined;
    const targets = this._declutterTargets;
    if (!attr || !targets) {
      this._declutterAnimating = false;
      return false;
    }

    const step = deltaMs / DECLUTTER_FADE_MS;
    const count = Math.min(attr.count, targets.length);
    let stillAnimating = false;
    let changed = false;
    for (let i = 0; i < count; i++) {
      let value = attr.getX(i);
      const target = targets[i];
      if (value === target) continue;
      value =
        value < target
          ? Math.min(value + step, target)
          : Math.max(value - step, target);
      attr.setX(i, value);
      changed = true;
      if (value !== target) stillAnimating = true;
    }
    if (changed) attr.needsUpdate = true;
    this._declutterAnimating = stillAnimating;
    return stillAnimating;
  }

  private _cacheDeclutterState(m: NavaraPointMesh | NavaraBillboardMesh) {
    const nextDeclutter = m.material.declutter ?? true;
    if (this._declutter && !nextDeclutter) {
      // Leaving declutter mode: clear hides the pass applied — the mesh stops
      // producing candidates, so nothing else would ever re-show them.
      this._clearDeclutterHidden();
    }
    this._declutter = nextDeclutter;
    this._declutterPriority = m.material.declutterPriority ?? 0;
  }

  private _clearDeclutterHidden(): void {
    const targets = this._declutterTargets;
    if (!targets) return;
    targets.fill(0.0);
    // Everything fades back in from wherever its hide factor currently is.
    this._declutterAnimating = true;
  }

  /** Reconstruct absolute ECEF anchors from the same arrays the position
   *  attributes receive (RTE high+low split, or RTC-relative + center). */
  private _cacheAnchors(
    positionsInfo: PositionsInfo,
    transform: { tx: number; ty: number; tz: number },
  ): void {
    const { nPositions, positionSize, RTE } = positionsInfo;
    const anchors =
      this._anchors && this._anchors.length === nPositions * 3
        ? this._anchors
        : new Float64Array(nPositions * 3);

    if (RTE) {
      const pos = positionsInfo.position as {
        high: Float32Array<ArrayBufferLike>;
        low: Float32Array<ArrayBufferLike>;
      };
      for (let i = 0; i < nPositions; i++) {
        const s = i * positionSize;
        anchors[i * 3] = pos.high[s] + pos.low[s];
        anchors[i * 3 + 1] = pos.high[s + 1] + pos.low[s + 1];
        anchors[i * 3 + 2] = (pos.high[s + 2] ?? 0.0) + (pos.low[s + 2] ?? 0.0);
      }
    } else {
      const pos = positionsInfo.position as Float32Array<ArrayBufferLike>;
      for (let i = 0; i < nPositions; i++) {
        const s = i * positionSize;
        anchors[i * 3] = pos[s] + transform.tx;
        anchors[i * 3 + 1] = pos[s + 1] + transform.ty;
        anchors[i * 3 + 2] = (pos[s + 2] ?? 0.0) + transform.tz;
      }
    }
    this._anchors = anchors;
  }

  async _init(m: NavaraPointMesh | NavaraBillboardMesh) {
    const positionsInfo = this.extractPositions(m);
    if (positionsInfo === null) {
      console.warn("No position data found for InstancedSpriteMesh");
      return;
    }

    this._cacheDeclutterState(m);
    this._cacheAnchors(positionsInfo, m.transform);

    // Create Geometry
    this.geometry = this._initGeometry(positionsInfo, m);

    // Create Material
    this.material = await this._initMaterial(positionsInfo, m);

    this.frustumCulled = false; // Disable since bounding box doesn't account for instance positions
    this.ctx.declutter?.markDirty();
  }

  async _update(m: NavaraPointMesh | NavaraBillboardMesh) {
    const enhancer = this.getEnhancer();
    // On the WebGPU backend the enhancer's classic material is the state
    // store (this.material is the node material); identical object otherwise.
    const material = (this._classicMaterial ?? this.material) as ShaderMaterial;

    this._cacheDeclutterState(m);

    if (material.visible !== m.material.show) {
      material.visible = m.material.show ?? true;
      this.updateVisibility();
    }

    // Update enhancer state for uniform-backed properties. Style defaults
    // (color/opacity/addHeight) are uniforms; per-feature overrides live in
    // the batch data texture and win once written.
    // `color` only until batch color is enabled: from then on every feature
    // reads the texture (written value or the fixed default), same semantics
    // as PolygonMesh._update.
    const batchColorEnabled = !!(
      material.userData.defines as Record<string, unknown> | undefined
    )?.USE_BATCH_COLOR;
    enhancer.update({
      base: {
        scale: m.material.size ?? 100.0,
        center: [m.material.center?.x ?? 0.0, m.material.center?.y ?? 0.0],
        sizeInMeters: m.material.sizeInMeters ?? true,
        offsetDepth: m.material.offsetDepth ?? true,
        transparent: m.material.transparent ?? true,
        depthTest: m.material.depthTest ?? true,
        color: batchColorEnabled ? undefined : (m.material.color ?? 0xffffff),
        opacity: m.material.opacity ?? 1.0,
        addHeight: m.material.height ?? 0.0,
        effectIdsMask:
          this.ctx.viewContext.selectiveEffectRegistry?.computeMask(
            m.material.effectIds ?? [],
          ) ?? 0,
        emissiveColor: m.material.emissiveColor ?? 0,
        emissiveIntensity: m.material.emissiveIntensity ?? 0,
      },
    });

    // Position updates (per-instance attributes)
    {
      const positionsInfo = this.extractPositions(m);

      if (positionsInfo) {
        this._cacheAnchors(positionsInfo, m.transform);
        if (positionsInfo.RTE) {
          const pos = positionsInfo.position as {
            high: Float32Array<ArrayBufferLike>;
            low: Float32Array<ArrayBufferLike>;
          };
          const pLow = this.geometry.getAttribute(
            "instancePositionLOW",
          ) as InstancedBufferAttribute;
          const pHigh = this.geometry.getAttribute(
            "instancePositionHIGH",
          ) as InstancedBufferAttribute;
          pLow.copyArray(pos.low);
          pHigh.copyArray(pos.high);
          pLow.needsUpdate = true;
          pHigh.needsUpdate = true;
        } else {
          const pos = positionsInfo.position as Float32Array<ArrayBufferLike>;
          const p = this.geometry.getAttribute(
            "instancePosition",
          ) as InstancedBufferAttribute;
          p.copyArray(pos);
          p.needsUpdate = true;
        }
      }
    }

    // Billboard-specific updates
    if (m instanceof NavaraBillboardMesh) {
      enhancer.update({
        base: { alphaTest: m.material.alphaTest ?? 0.0 },
      });

      if (m.material.url) {
        await this._setDefaultImage(m.material.url);
      }
    }

    this.ctx.declutter?.markDirty();
  }

  private _initGeometry(
    positionsInfo: PositionsInfo,
    m: NavaraPointMesh | NavaraBillboardMesh,
  ) {
    invariant(positionsInfo.batchIDs, "Batch IDs not found!");

    // prettier-ignore
    const vertices = new Float32Array([
      -0.5, -0.5, 0.0, // v0
       0.5, -0.5, 0.0, // v1
       0.5,  0.5, 0.0, // v2
      -0.5, -0.5, 0.0, // v3
       0.5,  0.5, 0.0, // v4
      -0.5,  0.5, 0.0, // v5
    ]);

    // prettier-ignore
    const uvs = new Float32Array([
      0.0, 0.0, // v0
      1.0, 0.0, // v1
      1.0, 1.0, // v2
      0.0, 0.0, // v3
      1.0, 1.0, // v4
      0.0, 1.0, // v5
    ]);

    const instanceCount = positionsInfo.nPositions;

    // Create the Instanced Mesh
    // We use InstancedBufferGeometry to inject our custom attributes
    const instancedGeometry = new InstancedBufferGeometry();
    instancedGeometry.setAttribute(
      "position",
      new BufferAttribute(vertices, 3),
    );
    instancedGeometry.setAttribute("uv", new BufferAttribute(uvs, 2));
    instancedGeometry.instanceCount = instanceCount;

    this._instanceCount = instanceCount;
    this._rebuildBatchIndexMap(m);
    this._batchLength = m.batch_length;

    // Per-instance feature index for the batch data texture lookup. An
    // identity mapping (the common case) gets a plain ramp.
    let batchIdArray = this._instanceBatchIndex;
    if (!batchIdArray) {
      batchIdArray = new Float32Array(instanceCount);
      for (let i = 0; i < instanceCount; i++) batchIdArray[i] = i;
    }
    instancedGeometry.setAttribute(
      "_batchid",
      new InstancedBufferAttribute(batchIdArray, 1),
    );

    if (m instanceof NavaraBillboardMesh) {
      // instanceUvRect: vec4(x, y, w, h) — this instance's atlas sub-rect in
      // pixels. Zeroed until an image is packed; the vertex shader culls empty
      // rects, so instances stay invisible rather than garbled.
      instancedGeometry.setAttribute(
        "instanceUvRect",
        new InstancedBufferAttribute(new Float32Array(instanceCount * 4), 4),
      );
    }

    if (positionsInfo.RTE) {
      const pos = positionsInfo.position as {
        high: Float32Array<ArrayBufferLike>;
        low: Float32Array<ArrayBufferLike>;
      };
      instancedGeometry.setAttribute(
        "instancePositionLOW",
        new InstancedBufferAttribute(pos.low, positionsInfo.positionSize),
      );
      instancedGeometry.setAttribute(
        "instancePositionHIGH",
        new InstancedBufferAttribute(pos.high, positionsInfo.positionSize),
      );
    } else {
      const pos = positionsInfo.position as Float32Array<ArrayBufferLike>;
      instancedGeometry.setAttribute(
        "instancePosition",
        new InstancedBufferAttribute(pos, positionsInfo.positionSize),
      );
    }
    // Declutter hide factors (0 = shown … 1 = hidden). Decluttered instances
    // start hidden and fade in once the placement pass grants them space, so
    // dense tiles don't flash their full clutter before the first pass runs.
    const initialHide = this._declutter ? 1.0 : 0.0;
    const declutterBuffer = new Float32Array(instanceCount).fill(initialHide);
    this._declutterTargets = new Float32Array(instanceCount).fill(initialHide);
    instancedGeometry.setAttribute(
      "instanceDeclutterHide",
      new InstancedBufferAttribute(declutterBuffer, 1),
    );
    instancedGeometry.setAttribute(
      "instanceBatchID",
      new InstancedBufferAttribute(
        positionsInfo.batchIDs,
        positionsInfo.batchIDSize,
      ),
    );

    return instancedGeometry;
  }

  private async _initMaterial(
    positionsInfo: PositionsInfo,
    m: NavaraPointMesh | NavaraBillboardMesh,
  ) {
    const isBillboard = m instanceof NavaraBillboardMesh;
    const material = new ShaderMaterial();

    // Create enhancer
    const enhancer = createInstancedSpriteMaterialEnhancer(material);
    this._enhancedMaterial = enhancer;

    // Mount with initial props
    enhancer.mount({
      base: {
        useRTE: positionsInfo.RTE,
        billboard: isBillboard,
        scale: m.material.size ?? 100.0,
        center: [m.material.center?.x ?? 0.0, m.material.center?.y ?? 0.0],
        sizeInMeters: m.material.sizeInMeters ?? true,
        offsetDepth: m.material.offsetDepth ?? true,
        alphaTest: isBillboard ? (m.material.alphaTest ?? 0.0) : 0.0,
        pickable: false,
        transparent: m.material.transparent ?? true,
        depthTest: m.material.depthTest ?? true,
        color: m.material.color ?? 0xffffff,
        opacity: m.material.opacity ?? 1.0,
        addHeight: m.material.height ?? 0.0,
        effectIdsMask:
          this.ctx.viewContext.selectiveEffectRegistry?.computeMask(
            m.material.effectIds ?? [],
          ) ?? 0,
        emissiveColor: m.material.emissiveColor ?? 0,
        emissiveIntensity: m.material.emissiveIntensity ?? 0,
        rtcCenter: [m.transform.tx, m.transform.ty, m.transform.tz],
      },
    });

    // Initialize uniforms early so they're available before onBeforeCompile
    const mutates = enhancer.mutates();
    mutates.updateUniforms(material.uniforms, enhancer.states());

    let nodeMaterial: Material | null = null;
    if (this._isWebGPUBackend()) {
      // The GLSL enhancer pipeline (onBeforeCompile) never runs on the WebGPU
      // backend, and the shared Renderer only invokes object.onBeforeRender —
      // not material.onBeforeRender — so the per-frame uniform hook moves to
      // the mesh. The classic material stays alive as the enhancer's state
      // store; a TSL node material replaces it for rendering.
      this._classicMaterial = material;
      this.onBeforeRender = (renderer, _scene, camera) => {
        const pCam = camera as PerspectiveCamera;
        mutates.updateFarPlane(pCam.far);
        mutates.updateFovRad(degreeToRadian(pCam.fov));
        mutates.updateScreenHeightPx(
          renderer.getDrawingBufferSize(_tmpSize).y / renderer.getPixelRatio(),
        );

        if (positionsInfo.RTE) {
          mutates.updateRteUniforms(
            camera.position.x,
            camera.position.y,
            camera.position.z,
            enhancer.states(),
          );
        } else {
          mutates.updateRtcUniforms(
            camera.matrixWorldInverse,
            enhancer.states(),
          );
        }
        this._syncWebgpuHandles();
      };
      nodeMaterial = this._initWebGPUMaterial(material);
    } else {
      // Set up onBeforeRender for per-frame updates (farPlane + RTE eye position)
      material.onBeforeRender = (
        _renderer,
        _scene,
        camera,
        _geometry,
        _mat,
        _group,
      ) => {
        const pCam = camera as PerspectiveCamera;
        mutates.updateFarPlane(pCam.far);
        mutates.updateFovRad(degreeToRadian(pCam.fov));
        mutates.updateScreenHeightPx(
          _renderer.getDrawingBufferSize(_tmpSize).y /
            _renderer.getPixelRatio(),
        );

        if (positionsInfo.RTE) {
          mutates.updateRteUniforms(
            camera.position.x,
            camera.position.y,
            camera.position.z,
            enhancer.states(),
          );
        } else {
          mutates.updateRtcUniforms(
            camera.matrixWorldInverse,
            enhancer.states(),
          );
        }
      };

      // Set custom program cache key and onBeforeCompile
      material.customProgramCacheKey = enhancer.programCacheKey;
      material.onBeforeCompile = enhancer.transformShader;
    }

    // Register for per-feature styling. Slots and the texture itself are
    // allocated lazily on the first attribute write.
    const batchUniform = registerBatchedMaterial(
      material,
      { ...this._getBatchTextureSupport(), batchLength: this._batchLength },
      this.ctx.viewContext.getRenderer(),
    );
    enhancer.update({ base: { batchDataTexture: batchUniform } });

    // Handle billboard texture
    if (isBillboard && m.material.url) {
      await this._setDefaultImage(m.material.url);
    }

    material.visible = m.material.show ?? true;
    this.updateVisibility();
    return nodeMaterial ?? material;
  }

  private _isWebGPUBackend(): boolean {
    const renderer = this.ctx.viewContext?.getRenderer() as
      { isWebGPURenderer?: boolean } | undefined;
    return !!renderer?.isWebGPURenderer;
  }

  /**
   * WebGPU instancedSprite material: a TSL node material reproducing the
   * classic GLSL enhancer pipeline (instancedSprite.vert/frag.glsl), which
   * never runs on this backend:
   *  - vertex: per-instance params/color/batchId/declutter attributes, RTE
   *    high/low or RTC view-space anchor resolve, ellipsoidal horizon culling
   *    + hidden/decluttered/empty-rect culling (clip position collapsed to
   *    vec4(0)), view-space billboard expansion with pxToWorld scaling when
   *    sizeInMeters is off, atlas sub-rect UVs normalized by uAtlasSize;
   *  - fragment: atlas tint (billboard) or anti-aliased circle (point),
   *    alphaTest discard, manual logarithmic depth with the offsetDepth
   *    shift, pick coloring (nvr_batchIdToColor) via uPickable.
   * Uniform state is fed per render from the enhancer's classic-material
   * uniform refs by _syncWebgpuHandles; the atlas texture rides a TSL texture
   * node whose value is swapped when the atlas is repacked.
   * Not ported (same as the polyline WebGPU material): selective-effect
   * emissive, gbuffer MRT writes.
   */
  private _initWebGPUMaterial(src: ShaderMaterial): Material {
    const { webgpu, tsl } = getWebGPU();
    // TSL's chained node methods don't survive the library's generic
    // typings; the graph is runtime-checked by the node builder instead.
    // (Same `as any` idiom as the polygon/polyline WebGPU materials, applied
    // to the whole TSL namespace.)
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const T = tsl as any;
    const state = this.getEnhancer().states();
    const useRTE = state.useRTE;
    const billboard = state.billboard;

    const handles: InstancedSpriteWebgpuHandles = {
      uRTCCenter: T.uniform(new Vector3()) as unknown as { value: Vector3 },
      uRTCCenterView: T.uniform(new Vector3()) as unknown as { value: Vector3 },
      uEyeRTEHigh: T.uniform(new Vector3()) as unknown as { value: Vector3 },
      uEyeRTELow: T.uniform(new Vector3()) as unknown as { value: Vector3 },
      uScale: T.uniform(100) as unknown as { value: number },
      uCenter: T.uniform(new Vector2()) as unknown as { value: Vector2 },
      uSizeInMeters: T.uniform(1) as unknown as { value: number },
      uOffsetDepth: T.uniform(1) as unknown as { value: number },
      uAlphaTest: T.uniform(0) as unknown as { value: number },
      uFarPlane: T.uniform(1000) as unknown as { value: number },
      uAtlasSize: T.uniform(new Vector2(1, 1)) as unknown as { value: Vector2 },
      uFovRad: T.uniform(1) as unknown as { value: number },
      uScreenHeightPx: T.uniform(1080) as unknown as { value: number },
      uPickable: T.uniform(0) as unknown as { value: number },
      src,
    };
    if (billboard) {
      handles.spriteTex = T.texture(
        (src.uniforms.uTexture?.value as DataTexture | null) ??
          getPlaceholderTexture(),
      ) as unknown as { value: unknown };
    }
    const {
      uRTCCenter,
      uRTCCenterView,
      uEyeRTEHigh,
      uEyeRTELow,
      uScale,
      uCenter,
      uSizeInMeters,
      uOffsetDepth,
      uAlphaTest,
      uFarPlane,
      uAtlasSize,
      uFovRad,
      uScreenHeightPx,
      uPickable,
      spriteTex,
    } = handles as unknown as Record<string, any>;

    // ---- Vertex ----

    // Per-instance attributes (instancedSprite.vert.glsl). The quad uv is
    // exactly position.xy + 0.5; deriving it keeps the `uv` attribute out of
    // the node graph (WebGPU caps vertex buffers at 8, and the RTE billboard
    // path needs every remaining slot).
    const params: any = T.attribute("instanceParams");
    const instanceColor: any = T.attribute("instanceColor");
    const instanceBatchID: any = T.attribute("instanceBatchID");
    const declutterHide: any = T.attribute("instanceDeclutterHide");
    const uvRect: any = billboard ? T.attribute("instanceUvRect") : null;
    const quad: any = T.positionGeometry;
    const uvQuad = quad.xy.add(0.5);

    const instanceHeight = params.x;
    const instanceSize = params.y;
    const instanceShow = params.z;
    const vOpacity = T.varying(
      params.w.mul(declutterHide.oneMinus()),
      "nvr_opacity",
    );

    // Anchor resolve + view-space base position (RTE high/low split, or RTC
    // with the center pre-transformed to view space on the CPU in float64).
    let absTransformed: any;
    let baseMv: any;
    if (useRTE) {
      // u_rteOne (== 1.0) blocks fast-math reassociation of the high/low
      // recombination — see chunks/rte_pars_vertex.glsl.
      const uRteOne = T.uniform(1.0);
      const high: any = T.attribute("instancePositionHIGH");
      const low: any = T.attribute("instancePositionLOW");
      absTransformed = high.add(low);
      const resolved = high
        .sub(uEyeRTEHigh)
        .mul(uRteOne)
        .add(low.sub(uEyeRTELow));
      // viewMatrixRTE with the translation column removed == rotation applied
      // to a w=0 vector.
      baseMv = T.cameraViewMatrix.mul(T.vec4(resolved, 0)).xyz;
    } else {
      const pos: any = T.attribute("instancePosition");
      absTransformed = pos.add(uRTCCenter);
      baseMv = T.cameraViewMatrix.mul(T.vec4(pos, 0)).xyz.add(uRTCCenterView);
    }

    // Horizon culling (horizon_culling_*.glsl): the classic vertex shader
    // collapses the clip position to vec4(0); same via the cull factor below.
    const ONE_OVER_WGS84_RADII = T.vec3(
      1 / 6378137.0,
      1 / 6378137.0,
      1 / 6356752.3142451793,
    );
    const camScaled = T.cameraPosition.mul(ONE_OVER_WGS84_RADII);
    const horizonA = camScaled.dot(camScaled).sub(1);
    const horizonCulled = camScaled
      .sub(absTransformed.mul(ONE_OVER_WGS84_RADII))
      .dot(camScaled)
      .greaterThan(horizonA);

    // An empty atlas rect means no image is packed for this instance yet —
    // cull rather than stretch texel (0, 0) over the whole quad.
    const hasImage = billboard
      ? uvRect.z.greaterThan(0).and(uvRect.w.greaterThan(0))
      : T.bool(true);
    const culled = instanceShow
      .lessThanEqual(0.5)
      .or(declutterHide.greaterThanEqual(0.999))
      .or(hasImage.not())
      .or(horizonCulled);
    const notCulled = culled.not().toFloat();

    const vUv = T.varying(
      billboard ? uvRect.xy.add(uvQuad.mul(uvRect.zw)).div(uAtlasSize) : uvQuad,
      "nvr_uv",
    );
    const vBatchIDNode = T.varying(instanceBatchID, "nvr_batchId");
    T.nodeObject(vBatchIDNode).setInterpolation("flat");
    const vBatchID = vBatchIDNode;
    const vColor = T.varying(instanceColor, "nvr_instanceColor");

    // mvr_getMvHeightOffset: height along the ellipsoid normal, rotated into
    // view space (a w=0 transform, so the translation column never contributes).
    const mvHeightOffset = T.cameraViewMatrix.mul(
      T.vec4(absTransformed.normalize().mul(instanceHeight), 0),
    ).xyz;
    const mv = baseMv.add(mvHeightOffset);

    const center = uCenter.clamp(-0.5, 0.5);

    // Per-instance size wins when set (>= 0); a negative value means "use
    // uScale". Per-instance image aspect comes from the atlas rect.
    const scale = T.select(
      instanceSize.greaterThanEqual(0),
      instanceSize,
      uScale,
    );
    let clampedScale = T.max(T.float(0), scale);
    const aspect = billboard
      ? T.select(uvRect.w.greaterThan(0), uvRect.z.div(uvRect.w), T.float(1))
      : T.float(1);
    // pxToWorld: constant screen-pixel size when sizeInMeters is off.
    const pxScale = clampedScale.mul(
      T.tan(uFovRad.mul(0.5)).mul(mv.z.abs()).mul(2).div(uScreenHeightPx),
    );
    clampedScale = T.select(
      uSizeInMeters.greaterThan(0.5),
      clampedScale,
      pxScale,
    );

    // Screen-aligned expansion in view space.
    const finalMv = mv.add(
      T.vec3(quad.xy.sub(center).mul(T.vec2(aspect, 1)).mul(clampedScale), 0),
    );
    // gl_Position.w + 1.0; for a perspective camera w == -viewZ.
    const vFragDepth = T.varying(finalMv.z.negate().add(1), "nvr_fragDepth");

    const Base = webgpu.MeshBasicNodeMaterial as unknown as new () => Material;
    const viewPositionNode = finalMv;
    const cullFactorNode = notCulled;
    class InstancedSpriteNodeMaterial extends Base {
      // positionView is fully computed above (view matrix already applied);
      // the standard modelViewMatrix must not touch it.
      setupPositionView(): unknown {
        return viewPositionNode;
      }
      setupModelViewProjection(): unknown {
        return T.cameraProjectionMatrix.mul(T.positionView).mul(cullFactorNode);
      }
    }
    const m: any = new InstancedSpriteNodeMaterial();
    m.positionNode = viewPositionNode;

    // ---- Fragment ----

    // Billboard: atlas sample tinted by the instance color (RGB only, texture
    // alpha preserved). Point: anti-aliased circle (point.frag.glsl —
    // clamp((radius - len) / border, 0, 1) unrolled).
    const texC = billboard ? spriteTex.sample(vUv) : T.vec4(1, 1, 1, 1);
    const circleAlpha = T.float(0.5)
      .sub(vUv.sub(0.5).length())
      .div(0.01)
      .clamp(0, 1);
    const alphaForTest = billboard ? texC.a : circleAlpha;
    const alphaForColor = billboard ? texC.a.mul(vOpacity) : vOpacity;
    const baseRgb = billboard ? texC.rgb.mul(vColor) : vColor;

    // Pick color (matches pick.glsl nvr_batchIdToColor).
    const pickColor = T.vec3(
      vBatchID.div(65536).floor().div(255),
      vBatchID.div(256).mod(256).floor().div(255),
      vBatchID.mod(256).floor().div(255),
    );

    m.colorNode = T.Fn(() => {
      alphaForTest.lessThanEqual(uAlphaTest).discard();
      return T.select(
        uPickable.greaterThan(0).and(alphaForTest.greaterThan(0)),
        T.vec4(pickColor, 1),
        T.vec4(baseRgb, alphaForColor),
      );
    })();

    // Manual logarithmic depth (the classic shader writes gl_FragDepth);
    // offsetDepth shifts sprites slightly nearer the ellipsoid surface.
    const fragDepth = vFragDepth.log().div(uFarPlane.add(1).log());
    m.depthNode = T.select(
      uOffsetDepth.greaterThan(0.5),
      fragDepth.sub(0.01),
      fragDepth,
    );
    /* eslint-enable @typescript-eslint/no-explicit-any */

    m.userData.nvrWebgpu = handles;
    this._syncWebgpuHandles(handles);
    return m as Material;
  }

  /**
   * Per-render sync of the TSL uniform handles from the enhancer's uniform
   * refs (which live on the classic material, still mutated by the enhancer's
   * mutates and the per-frame hook). No-op unless the WebGPU node material is
   * installed.
   */
  private _syncWebgpuHandles(prebuilt?: InstancedSpriteWebgpuHandles): void {
    const w =
      prebuilt ??
      ((this.material as Material | undefined)?.userData.nvrWebgpu as
        InstancedSpriteWebgpuHandles | undefined);
    if (!w) return;
    const u = w.src.uniforms;
    const setV2 = (
      h: { value: Vector2 },
      r: { value: Vector2 } | undefined,
    ): void => {
      if (r) h.value.set(r.value.x, r.value.y);
    };
    const setV3 = (
      h: { value: Vector3 },
      r: { value: Vector3 } | undefined,
    ): void => {
      if (r) h.value.set(r.value.x, r.value.y, r.value.z);
    };
    const setN = (
      h: { value: number },
      r: { value: number } | undefined,
    ): void => {
      if (r !== undefined && typeof r.value === "number") h.value = r.value;
    };
    const setB = (
      h: { value: number },
      r: { value: boolean } | undefined,
    ): void => {
      if (r !== undefined) h.value = r.value ? 1 : 0;
    };

    setV3(w.uRTCCenter, u.uRTCCenter as { value: Vector3 } | undefined);
    setV3(w.uRTCCenterView, u.uRTCCenterView as { value: Vector3 } | undefined);
    setV3(w.uEyeRTEHigh, u.uEyeRTEHigh as { value: Vector3 } | undefined);
    setV3(w.uEyeRTELow, u.uEyeRTELow as { value: Vector3 } | undefined);
    setN(w.uScale, u.uScale as { value: number } | undefined);
    setV2(w.uCenter, u.uCenter as { value: Vector2 } | undefined);
    setB(w.uSizeInMeters, u.uSizeInMeters as { value: boolean } | undefined);
    setB(w.uOffsetDepth, u.uOffsetDepth as { value: boolean } | undefined);
    setN(w.uAlphaTest, u.uAlphaTest as { value: number } | undefined);
    setN(w.uFarPlane, u.uFarPlane as { value: number } | undefined);
    setV2(w.uAtlasSize, u.uAtlasSize as { value: Vector2 } | undefined);
    setN(w.uFovRad, u.uFovRad as { value: number } | undefined);
    setN(w.uScreenHeightPx, u.uScreenHeightPx as { value: number } | undefined);
    setN(w.uPickable, u.nvr_uPickable as { value: number } | undefined);

    // The atlas texture object is replaced whenever the atlas grows;
    // re-point the TSL texture node (NodeSampledTexture rebinds on change).
    if (w.spriteTex) {
      w.spriteTex.value =
        (u.uTexture?.value as DataTexture | null) ?? getPlaceholderTexture();
    }

    // Props the enhancer writes on its own (classic) material.
    const m = this.material as Material | undefined;
    if (m) {
      m.transparent = w.src.transparent;
      m.depthTest = w.src.depthTest;
      m.depthWrite = w.src.depthWrite;
    }
  }

  private updateVisibility() {
    // On the WebGPU backend the classic material holds the show/hide flag
    // (this.material is the node material); identical object otherwise.
    const material = this._classicMaterial ?? this.material;
    const materialVisible =
      material instanceof ShaderMaterial ? material.visible : true;
    this.visible = this._active && materialVisible;
  }

  private extractPositions(
    m: NavaraPointMesh | NavaraBillboardMesh,
  ): PositionsInfo | null {
    const { buf } = this.ctx;
    const g = m.geometry;

    const batchIdsData = g.batch_ids;
    const batchIDs = buf.removeF32(batchIdsData.data);
    const batchIDSize = batchIdsData.size;

    const positionData = g.position;
    const position = positionData
      ? buf.removeF32(positionData.data)
      : undefined;

    if (position && positionData) {
      const positionSize = positionData.size;
      const nPositions = position.length / positionSize;

      return {
        position,
        batchIDs,
        batchIDSize,
        positionSize,
        nPositions,
        RTE: false,
      };
    }

    const positionHighData = g.position_3d_high;
    const positionLowData = g.position_3d_low;
    const positionHigh = positionHighData
      ? buf.removeF32(positionHighData.data)
      : undefined;
    const positionLow = positionLowData
      ? buf.removeF32(positionLowData.data)
      : undefined;

    if (positionHigh && positionLow && positionHighData && positionLowData) {
      const positionLowSize = positionLowData.size;
      const positionHighSize = positionHighData.size;
      invariant(
        positionLowSize === positionHighSize,
        "Position high and low size mismatch",
      );

      const nPositions = positionHigh.length / positionHighSize;

      return {
        position: { high: positionHigh, low: positionLow },
        batchIDs,
        batchIDSize,
        positionSize: positionHighSize,
        nPositions,
        RTE: true,
      };
    }

    return null;
  }

  private _ensureAtlas(): BillboardAtlas {
    this._atlas ??= new BillboardAtlas({ loadImage: loadAtlasImageFromUrl });
    return this._atlas;
  }

  /**
   * Wire the callback that reports this mesh's atlas footprint to the
   * engine's memory ledger, and immediately report the current footprint —
   * the default image may have been packed during `_init`, before the
   * feature-added handler could wire the reporter.
   */
  setAtlasBytesReporter(reporter: (bytes: number) => void): void {
    this._atlasBytesReporter = reporter;
    this._reportAtlasBytes();
  }

  /** Report the atlas footprint if it changed since the last report. Must run
   * after every `pack()` — the atlas may have grown even when the pack failed
   * (growth up to `maxSize` happens before "no space" is decided). */
  private _reportAtlasBytes(): void {
    if (!this._atlasBytesReporter) return;
    const bytes = this._atlas?.byteLength ?? 0;
    if (bytes === this._reportedAtlasBytes) return;
    this._reportedAtlasBytes = bytes;
    this._atlasBytesReporter(bytes);
  }

  /**
   * Push the atlas texture and size to the material. The texture object is
   * replaced whenever the atlas grows, so this must run after every pack().
   */
  private _syncAtlasUniforms(): void {
    const atlas = this._atlas;
    if (!atlas) return;
    this.getEnhancer().update({
      base: {
        texture: { value: atlas.texture },
        atlasSize: [atlas.size, atlas.size],
      },
    });
  }

  /**
   * Load the material-level image and apply its rect to every instance that
   * hasn't been overridden per-feature via setFeatureImageByBatchId.
   */
  private async _setDefaultImage(url: string): Promise<void> {
    if (this._defaultUrl === url) return;
    this._defaultUrl = url;

    const rect = await this._ensureAtlas().pack(url);
    this._reportAtlasBytes();
    if (!rect) return;
    // A newer default image won the race while this one was loading.
    if (this._defaultUrl !== url) return;

    this._defaultRect = rect;
    this._syncAtlasUniforms();
    const rectAttr = this.geometry.getAttribute(
      "instanceUvRect",
    ) as InstancedBufferAttribute;
    for (let i = 0; i < rectAttr.count; i++) {
      if (this._imageOverrides.has(i)) continue;
      rectAttr.setXYZW(i, rect.x, rect.y, rect.w, rect.h);
    }
    rectAttr.needsUpdate = true;
    this._onAtlasRectsChanged();
  }

  /**
   * An instance's atlas rect changed. The rect drives the quad's aspect and
   * whether the instance draws at all, so the declutter bounds computed before
   * the image landed are stale — and because packs resolve long after the
   * frame that requested them, the view has usually gone idle by now. Without
   * the render request the page keeps showing the pre-image state until the
   * next camera move.
   */
  private _onAtlasRectsChanged(): void {
    this.ctx.declutter?.markDirty();
    this.ctx.renderFlag.forceUpdate = true;
  }

  onBeforePicking(): void {
    this.getEnhancer().update({ base: { pickable: true } });
  }

  onAfterPicking(): void {
    this.getEnhancer().update({ base: { pickable: false } });
  }

  getRenderable(): Object3D {
    return this;
  }

  /**
   * Get the enhancer, throwing if not initialized.
   */
  private getEnhancer(): NonNullable<typeof this._enhancedMaterial> {
    if (!this._enhancedMaterial) {
      throw new Error(
        "InstancedSpriteMesh material enhancer is not initialized. This usually indicates a failure during construction or geometry/material setup.",
      );
    }
    return this._enhancedMaterial;
  }

  /**
   * See {@link buildBatchIndexMap}. The batch_index handle stays owned by the
   * ECS geometry, whose destroy path frees it (`remove_from_buf`).
   */
  private _rebuildBatchIndexMap(
    m: NavaraPointMesh | NavaraBillboardMesh,
  ): void {
    const batchIndexData = m.geometry.batch_index?.data;
    const built = buildBatchIndexMap(
      batchIndexData !== undefined ? this.ctx.buf.u32(batchIndexData) : null,
    );
    this._instanceBatchIndex = built?.perInstance ?? null;
    this._batchIndexToInstances = built?.byBatchIndex ?? null;
  }

  /** All instances owned by the feature at `batchIndex`; empty when unknown. */
  private instancesOfBatchIndex(batchIndex: number): number[] {
    if (this._batchIndexToInstances) {
      return this._batchIndexToInstances.get(batchIndex) ?? [];
    }
    return batchIndex >= 0 && batchIndex < this._instanceCount
      ? [batchIndex]
      : [];
  }

  _getBatchTextureSupport(): BatchTextureSupport {
    return SPRITE_BATCH_SUPPORT;
  }

  /**
   * Write one per-feature style into the batch data texture (bounds and
   * value validation, slot allocation, and define stamping live in
   * {@link updateBatchAttribute}). One write covers every instance of the
   * feature — no per-instance fan-out.
   */
  private _updateBatchAttribute(
    batchIndex: number,
    attribute: BatchedAttributeName,
    value: number | number[] | boolean,
  ): boolean {
    return updateBatchAttribute(
      this.material as ShaderMaterial,
      batchIndex,
      attribute,
      value,
    );
  }

  setFeatureColorByBatchIndex(batchIndex: number, color: Color) {
    this._updateBatchAttribute(
      batchIndex,
      "color",
      color.toArray(_tmpColorArray),
    );
  }

  setFeatureShowByBatchIndex(batchIndex: number, rawVisible: boolean) {
    if (this._updateBatchAttribute(batchIndex, "show", rawVisible)) {
      this.ctx.declutter?.markDirty();
    }
  }

  setFeatureOpacityByBatchIndex(batchIndex: number, opacity: number) {
    this._updateBatchAttribute(batchIndex, "opacity", opacity);
  }

  setFeatureHeightByBatchIndex(batchIndex: number, height: number) {
    if (this._updateBatchAttribute(batchIndex, "height", height)) {
      this.ctx.declutter?.markDirty();
    }
  }

  setFeatureEmissiveByBatchIndex(batchIndex: number, emissive: Color) {
    this._updateBatchAttribute(
      batchIndex,
      "emissive",
      emissive.toArray(_tmpColorArray),
    );
  }

  setFeatureEmissiveIntensityByBatchIndex(
    batchIndex: number,
    intensity: number,
  ) {
    this._updateBatchAttribute(batchIndex, "emissiveIntensity", intensity);
  }

  /**
   * Set one instance's declutter fade target; the attribute animates toward
   * it in {@link stepDeclutterFade}. Deliberately separate from the batch
   * texture's `show` so user-driven visibility and declutter results compose
   * instead of clobbering each other.
   */
  setDeclutterHiddenByInstance(instanceIndex: number, hidden: boolean) {
    const targets = this._declutterTargets;
    if (!targets || instanceIndex < 0 || instanceIndex >= targets.length) {
      return;
    }
    targets[instanceIndex] = hidden ? 1.0 : 0.0;
    // Cheap over-approximation; stepDeclutterFade clears it when everything
    // has reached its target.
    this._declutterAnimating = true;
  }

  /**
   * Set a per-feature placement priority (higher wins), overriding the
   * layer-level `declutterPriority` for this instance. Driven by the feature
   * evaluator.
   */
  setFeatureDeclutterPriorityByBatchIndex(
    batchIndex: number,
    priority: number,
  ) {
    const instances = this.instancesOfBatchIndex(batchIndex);
    if (instances.length === 0) return;

    if (!this._declutterPriorityOverrides) {
      const count = this._anchors ? this._anchors.length / 3 : 0;
      if (count === 0) return;
      this._declutterPriorityOverrides = new Float32Array(count).fill(
        Number.NaN,
      );
    }
    let changed = false;
    for (const instanceId of instances) {
      if (instanceId >= this._declutterPriorityOverrides.length) continue;
      if (this._declutterPriorityOverrides[instanceId] === priority) continue;
      this._declutterPriorityOverrides[instanceId] = priority;
      changed = true;
    }
    if (changed) {
      this.ctx.declutter?.markDirty();
    }
  }

  setFeatureSizeByBatchIndex(batchIndex: number, size: number) {
    if (this._updateBatchAttribute(batchIndex, "size", size)) {
      this.ctx.declutter?.markDirty();
    }
  }

  /**
   * Give one feature its own image, packed into this mesh's texture atlas.
   * Loads are deduplicated by URL, so styling many features with few distinct
   * images fetches each image once. On load failure the feature keeps its
   * current (default) image. Passing a nullish `url` clears the override and
   * reverts the feature to the material's default image. No-op for
   * non-billboard (point) meshes.
   */
  async setFeatureImageByBatchIndex(
    batchIndex: number,
    url: string | null | undefined,
  ): Promise<void> {
    const instances = this.instancesOfBatchIndex(batchIndex);
    if (instances.length === 0) return;

    const rectAttr = this.geometry.getAttribute("instanceUvRect") as
      InstancedBufferAttribute | undefined;
    if (!rectAttr) return;

    if (url == null) {
      let cleared = false;
      for (const instanceId of instances) {
        this._requestedImageUrls.delete(instanceId);
        if (!this._imageOverrides.delete(instanceId)) continue;
        // Zero rect (invisible) until the default image finishes loading,
        // same as instances that never had an override.
        const rect = this._defaultRect;
        rectAttr.setXYZW(
          instanceId,
          rect?.x ?? 0,
          rect?.y ?? 0,
          rect?.w ?? 0,
          rect?.h ?? 0,
        );
        cleared = true;
      }
      if (cleared) {
        rectAttr.needsUpdate = true;
        this._onAtlasRectsChanged();
      }
      return;
    }

    for (const instanceId of instances) {
      this._requestedImageUrls.set(instanceId, url);
    }
    const rect = await this._ensureAtlas().pack(url);
    this._reportAtlasBytes();
    if (!rect) return;
    let applied = false;
    for (const instanceId of instances) {
      // A newer override or a clear superseded this load while in flight.
      if (this._requestedImageUrls.get(instanceId) !== url) continue;
      this._imageOverrides.add(instanceId);
      rectAttr.setXYZW(instanceId, rect.x, rect.y, rect.w, rect.h);
      applied = true;
    }
    if (applied) {
      this._syncAtlasUniforms();
      rectAttr.needsUpdate = true;
      this._onAtlasRectsChanged();
    }
  }

  dispose(): void {
    this.ctx.declutter?.unregister(this);
    this.geometry?.dispose();

    // The material's uTexture points at the atlas texture; the atlas owns it.
    this._atlas?.dispose();
    this._atlas = undefined;

    // Clear this mesh's atlas term from the memory ledger. When disposal was
    // caused by the owning tile's eviction the entity is already gone and the
    // report is a no-op; when only this feature was removed the tile survives
    // and the term must not linger.
    if (this._reportedAtlasBytes !== 0) {
      this._reportedAtlasBytes = 0;
      this._atlasBytesReporter?.(0);
    }

    (this.material as ShaderMaterial).dispose();
    // The classic ShaderMaterial survives as the enhancer's state store when
    // the WebGPU node material replaced it for rendering.
    if (this._classicMaterial) {
      this._classicMaterial.dispose();
      this._classicMaterial = undefined;
    }

    // Clear internal collections to release references
    this._batchIndexToInstances = null;
    this._instanceBatchIndex = null;
    this._anchors = null;
    this._declutterTargets = null;
    this._declutterPriorityOverrides = null;
    this._imageOverrides.clear();
    this._requestedImageUrls.clear();
  }
}
