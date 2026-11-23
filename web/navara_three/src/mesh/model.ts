import { Unimplemented } from "@navaramap/core";
import {
  ModelMaterial as NavaraModelMaterial,
  ModelMesh as NavaraModelMesh,
  PointColorsFormat,
  Vec3,
} from "@navaramap/engine";
import {
  BufferAttribute,
  BufferGeometry,
  Box3Helper,
  Color,
  DataTexture,
  DirectionalLight,
  Group,
  Mesh,
  Points,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  Object3D,
  SkinnedMesh,
  Texture,
  Vector3,
  type Camera,
  type NormalBufferAttributes,
  type Object3D as ThreeObject3D,
  PointsMaterial,
} from "three";
import invariant from "tiny-invariant";

import {
  attachBatchedMaterial,
  initBatchedMaterial,
  MODEL_BATCH_SUPPORT,
  registerBatchedMaterial,
  updateBatchAttribute,
  type BatchTextureConfig,
  type BatchTextureSupport,
} from "../batchTexture";
import type { EventContext } from "../event/context";
import { applyLitOption, createShadowDepthMaterial } from "../material";
import {
  createModelMaterialEnhancer,
  createPntsEnhancer,
  tintBaseColor,
} from "../material/enhancer/model";
import type { ModelMaterialProps, PntsProps } from "../material/enhancer/model";
import type { CustomObject3DEventMap } from "../object3DEvent";
import { getWebGPU } from "../utils";

import { GEOMETRY_TYPES } from "./constants";
import type { FeatureMesh } from "./featureMesh";
import { vertexGlobalBatchIds } from "./modelBatchIds";
import type { PickableMesh } from "./pickableMesh";
import { releaseGeometryArraysAfterUpload } from "./releaseGeometryArrays";
import {
  createWebgpuBatchSampler,
  eagerAllocateBatchTexture,
  syncWebgpuBatchTexture,
} from "./webgpuBatchTexture";

export type ModelMaterial = MeshStandardMaterial | MeshPhysicalMaterial;

/** `CONSTANT_RGBA` (`0xRRGGBBAA`, sRGB) as the PNTS base color. */
function constantRgba(
  rgba: number,
): Pick<PntsProps, "baseColor" | "baseOpacity"> {
  return {
    baseColor: new Color().setHex(rgba >>> 8),
    baseOpacity: (rgba & 0xff) / 0xff,
  };
}

// TODO: Height to adjust the height based on its property.
export type ModelBatchedAttributeName =
  "color" | "show" | "opacity" | "emissive" | "emissiveIntensity";

/** @deprecated Use {@link MODEL_BATCH_SUPPORT}; `batchLength` is per-mesh
 *  state, not part of the mesh type's capability. */
export const MODEL_BATCH_TEXTURE_CONFIG: BatchTextureConfig = {
  ...MODEL_BATCH_SUPPORT,
  batchLength: 0,
};

type ModelMaterialEnhancer = ReturnType<typeof createModelMaterialEnhancer>;
type PntsMaterialEnhancer = ReturnType<typeof createPntsEnhancer>;

/**
 * TSL uniform handles + per-frame prop sources for the WebGPU node material
 * built by _convertMeshToNodeMaterial (same idiom as polygon.ts's
 * PolygonWebgpuHandles). Values are synced from the enhancer state once per
 * render (see _syncMeshWebgpuHandles) because only plain material props flow
 * through the enhancer-mounted node material on their own.
 */
type ModelWebgpuHandles = {
  gateColor: { value: number };
  gateEmissive: { value: number };
  uPickable: { value: number };
  /** Batch texture node; re-pointed if the shared uniform swaps textures. */
  batchTex?: { value: unknown };
  /** Selective effect: 1 when effectIdsMask > 0 (see effectIdsMask). */
  uEffectGate: { value: number };
  uEmissiveIntensity: { value: number };
  uSpecular: { value: number };
  uShininess: { value: number };
  uSpecularStrength: { value: number };
  uIor: { value: number };
  uWater: { value: number };
  uWaterScaleNormal: { value: number };
  uWaterSpeed: { value: number };
  uReflectivity: { value: number };
  uTime: { value: number };
  /** Sun (first DirectionalLight) direction toward the light, view space. */
  uSunDirView: { value: Vector3 };
  /** Sun color premultiplied by intensity (matches GLSL lights uniforms). */
  uSunColor: { value: Color };
  /** Water normal map TextureNode; value swapped from CommonUniforms. */
  waterNormalNode: { value: Texture } | null;
};

/** TSL uniform handles for the WebGPU point-cloud material. */
type PntsWebgpuHandles = {
  uAddHeight: { value: number };
  uGeodeticNormal: { value: Vector3 };
  uDivideColor: { value: number };
};

/** Sampled-but-gated-off water normal map needs a valid texture binding. */
const WEBGPU_WATER_NORMAL_PLACEHOLDER = new DataTexture(
  new Uint8Array([128, 128, 255, 255]),
  1,
  1,
);
WEBGPU_WATER_NORMAL_PLACEHOLDER.needsUpdate = true;

/** Water IOR baked into the classic water_pars_fragment.glsl. */
const WATER_IOR = 1.333;
/** PNTS vertex-color normalization divisor (pntsEnhancer/shader.ts). */
const PNTS_COLOR_DIVISOR = 65535;

// Scratch vectors for _syncSunHandles (no per-frame allocation).
const _sunWorldPos = new Vector3();
const _sunTargetPos = new Vector3();

export class ModelMesh
  extends Object3D<CustomObject3DEventMap>
  implements FeatureMesh, PickableMesh
{
  readonly ctx: EventContext;
  /** Enhanced materials with encapsulated state, one per child mesh */
  private _enhancers = new Map<
    Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
    ModelMaterialEnhancer
  >();

  /** Enhanced materials for point cloud objects, one per Points child */
  private _pntsEnhancers = new Map<
    Points<BufferGeometry<NormalBufferAttributes>, PointsMaterial>,
    PntsMaterialEnhancer
  >();

  /** First-write flags gating the replacing batch emissive in the WebGPU node
   *  graph, per child mesh (the classic path uses a write-time
   *  USE_BATCH_EMISSIVE define). */
  private _batchEmissiveUsed = new WeakMap<
    Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
    boolean
  >();

  // model credit for attribution
  credit: string | undefined;
  batchLength?: number;

  constructor(
    ctx: EventContext,
    gltfInfo: {
      scene: Group;
      credit?: string;
    },
    m: NavaraModelMesh,
  ) {
    super();
    this.ctx = ctx;
    this.credit = gltfInfo.credit;
    this.batchLength = m.batch_length;
    this.add(gltfInfo.scene);
    this.init(m);
    this.addEventListener("removedFromWorld", () => {
      this.dispose();
    });
  }

  /**
   * Geometry type of this mesh.
   */
  readonly geometryType = GEOMETRY_TYPES.Model;

  get water(): boolean {
    for (const enhancer of this._enhancers.values()) {
      // Assume the first enhancer has the common value.
      return enhancer.states().water.useWater;
    }
    return false;
  }

  set water(v: boolean) {
    for (const enhancer of this._enhancers.values()) {
      enhancer.update({ water: { water: v } });
    }
  }

  private init(m: NavaraModelMesh) {
    const { buf } = this.ctx;
    const batchIdsData = m.geometry.batch_ids;
    const dataSize = batchIdsData?.size ?? 0;
    // buf.u32 returns a short-lived view into WASM memory; copy to retain it
    // across the traversal below.
    const batchIds = batchIdsData
      ? (buf.u32(batchIdsData.data)?.slice() ?? null)
      : new Uint32Array(dataSize);

    const meshMaterial = m.material;
    const uniforms = this.ctx.uniforms;

    const updateProps = this.buildUpdateProps(meshMaterial);
    const modelInitialProps: ModelMaterialProps = {
      ...updateProps,
      water: {
        ...updateProps.water,
        getWaterState: () => uniforms.water,
      },
    };

    const internal = meshMaterial.__internal__;
    const geodeticNormal: Vec3 =
      internal?.pointCloudGeodeticNormal ?? new Vec3(0, 0, 0);
    const layout = internal?.pointCloudLayout;
    const pntsInitialProps: PntsProps = {
      color: meshMaterial.color ?? 0,
      pointSize: meshMaterial.pointSize ?? 1,
      height: meshMaterial.height ?? 0,
      opacity: meshMaterial.opacity,
      transparent: meshMaterial.transparent,
      depthWrite: meshMaterial.depthWrite,
      geodeticNormal,
      srgbVertexColor: internal?.pointCloud,
      ...(layout?.colorsFormat === PointColorsFormat.Constant
        ? constantRgba(layout.constantColor)
        : {}),
    };

    this.traverse((object: Object3D) => {
      if (object instanceof Mesh) {
        this._setupMeshNode(
          object,
          meshMaterial,
          batchIds,
          dataSize,
          modelInitialProps,
        );
        this._releaseGeometryArrays(object);
      } else if (object instanceof Points) {
        this._setupPointsNode(object, pntsInitialProps);
        this._releaseGeometryArrays(object);
      }
    });

    this.visible = meshMaterial.show ?? true;
    this._syncBoundingBoxes(!!meshMaterial.showBoundingBox);
  }

  /** One per mesh / point node while `showBoundingBox` is on. */
  private _boxHelpers: Box3Helper[] = [];

  /**
   * Adds or removes a bounding-box helper on every mesh (blue) and point
   * (red) node so they match `show`. Runs at init and on every material
   * change, so toggling `showBoundingBox` reaches models already loaded.
   */
  private _syncBoundingBoxes(show: boolean) {
    if (!show) {
      for (const helper of this._boxHelpers) {
        helper.removeFromParent();
        helper.dispose();
      }
      this._boxHelpers = [];
      return;
    }
    if (this._boxHelpers.length > 0) return;
    const nodes: (Mesh | Points)[] = [];
    this.traverse((object) => {
      if (object instanceof Mesh || object instanceof Points) {
        nodes.push(object);
      }
    });
    for (const node of nodes) {
      const box = (node.geometry as BufferGeometry).boundingBox;
      if (!box) continue;
      const helper = new Box3Helper(
        box,
        node instanceof Points ? 0xff0000 : 0x0000ff,
      );
      node.add(helper);
      this._boxHelpers.push(helper);
    }
  }

  /**
   * Drop the CPU-side typed arrays of a loaded glTF/point-cloud node after its
   * first GPU upload. The GLTFLoader has already computed bounding volumes at
   * parse time (and the point-cloud path assigns them from the WASM AABB), and
   * `_setupMeshNode` has already read `_batchid` before this runs, so no
   * CPU read survives the first upload.
   *
   * Skipped for skinned meshes and geometries with morph targets: those keep
   * their CPU arrays for per-frame skinning/morphing on the CPU.
   */
  private _releaseGeometryArrays(node: Mesh | Points) {
    if (node instanceof SkinnedMesh) return;
    const geometry = node.geometry;
    if (!(geometry instanceof BufferGeometry)) return;
    if (
      geometry.morphAttributes &&
      Object.keys(geometry.morphAttributes).length
    )
      return;
    releaseGeometryArraysAfterUpload(geometry);
  }

  _getBatchTextureSupport(): BatchTextureSupport {
    return MODEL_BATCH_SUPPORT;
  }

  _initBatchedMaterial(
    mesh: Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
  ) {
    initBatchedMaterial(mesh.material, {
      ...this._getBatchTextureSupport(),
      batchLength: 0,
    });
  }

  _initBatchDataTexture(
    mesh: Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
  ): void {
    invariant(this.batchLength != null);

    const uniform = registerBatchedMaterial(
      mesh.material,
      { ...this._getBatchTextureSupport(), batchLength: this.batchLength },
      this.ctx.viewContext.getRenderer(),
    );
    this._enhancers.get(mesh)?.update({ base: { batchDataTexture: uniform } });
  }

  _updateBatchAttribute(
    mesh: Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
    batchId: number,
    attribute: ModelBatchedAttributeName,
    value: number | number[] | boolean,
  ): void {
    // Write the texture first: a rejected write must not stamp any define —
    // the shaders have no safety net for an unwritten receiver.
    const wrote = updateBatchAttribute(
      mesh.material,
      batchId,
      attribute,
      value,
    );
    if (!wrote) return;

    const enhancer = this._enhancers.get(mesh);
    if (enhancer && attribute === "color") {
      // When batch color is first used, untint material.color back to the
      // base color so the batch color multiplies it like a white layer color.
      if (!enhancer.states().base.batchColorEnabled) {
        enhancer.update({
          base: { batchColorEnabled: true, color: 0xffffff },
        });
      }
    }
    if (attribute === "emissive" || attribute === "emissiveIntensity") {
      // First-write flag gating the replacing batch emissive in the WebGPU
      // node graph (the classic path uses a write-time USE_BATCH_EMISSIVE
      // define).
      this._batchEmissiveUsed.set(mesh, true);
    }
  }

  private _setupMeshNode(
    mesh: Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
    meshMaterial: NavaraModelMaterial,
    batchIds: Uint32Array<ArrayBufferLike> | null,
    dataSize: number,
    initialProps: ModelMaterialProps,
  ) {
    // Content without features (e.g. GLB in a 1.0 tileset) has no batch IDs.
    // The shader then reads `batchId` as 0, i.e. nothing to pick.
    if (batchIds) {
      this._setBatchIdAttribute(mesh, batchIds, dataSize);
    }

    // The enhancer supports standard/physical only. Unlit materials
    // (`KHR_materials_unlit`, e.g. Google Photorealistic 3D Tiles) render as is.
    if (!(mesh.material instanceof MeshStandardMaterial)) return;

    mesh.castShadow = !!meshMaterial.castShadow;
    mesh.receiveShadow = !!meshMaterial.receiveShadow;
    applyLitOption(mesh.material, meshMaterial.lit);

    mesh.material.depthTest = true;
    mesh.material.depthWrite = true;

    const webgpu = this.isWebGPUBackend();
    if (webgpu) {
      // The GLSL enhancer pipeline never runs on the WebGPU backend; swap in
      // an equivalent TSL node material first so the enhancer mounts on it
      // (its prop writes then land on the rendered material directly).
      this._convertMeshToNodeMaterial(mesh, meshMaterial.lit);
    }

    const enhancer = createModelMaterialEnhancer(mesh.material);
    this._enhancers.set(mesh, enhancer);

    enhancer.mount(initialProps);

    if (!webgpu) {
      mesh.material.customProgramCacheKey = () => enhancer.programCacheKey();
      mesh.material.onBeforeCompile = enhancer.transformShader;
    }

    this._initBatchedMaterial(mesh);

    this.initDepthMaterial(mesh, enhancer);

    this.ctx.viewContext.applyShadowMaterial(mesh.material);
  }

  private _setBatchIdAttribute(
    mesh: Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
    batchIds: Uint32Array<ArrayBufferLike>,
    dataSize: number,
  ) {
    // B3DM (1.0) uses _batchid; glTF with EXT_mesh_features (1.1) uses _FEATURE_ID_N.
    // Assign _FEATURE_ID_0 to _batchid so the batch texture shader works unchanged.
    // Also accept lowercase _feature_id_0 as a compatibility fallback.
    const attrs = mesh.geometry.attributes;
    const featureIdAttribute =
      attrs?.["_FEATURE_ID_0"] ?? attrs?.["_feature_id_0"];
    if (!attrs?._batchid && featureIdAttribute) {
      // TODO: Support other feature ID semantics such as `_FEATURE_ID_n`.
      // Need to clone, since it might be switch to different feature ID attributes.
      mesh.geometry.setAttribute("_batchid", featureIdAttribute.clone());
    }
    const featureIndices = attrs?._batchid;
    const attrBatchIds = featureIndices
      ? vertexGlobalBatchIds(featureIndices, batchIds)
      : new Float32Array(attrs?.position?.count).fill(batchIds[0]);

    mesh.geometry.setAttribute(
      "batchId",
      new BufferAttribute(attrBatchIds, dataSize),
    );
  }

  private _setupPointsNode(
    points: Points<BufferGeometry<NormalBufferAttributes>, PointsMaterial>,
    initialProps: PntsProps,
  ) {
    if (this.isWebGPUBackend()) {
      this._convertPointsToNodeMaterial(points);
    }

    const enhancer = createPntsEnhancer(points.material);
    this._pntsEnhancers.set(points, enhancer);

    // glTF points: the layer tints their own base color and alpha instead of
    // replacing them, as `tintBaseColor` does for meshes.
    const { color, opacity } = points.material;
    enhancer.mount(
      initialProps.srgbVertexColor
        ? initialProps
        : { ...initialProps, baseColor: color.clone(), baseOpacity: opacity },
    );

    if (!this.isWebGPUBackend()) {
      points.material.customProgramCacheKey = () => enhancer.programCacheKey();
      points.material.onBeforeCompile = enhancer.transformShader;
    }
  }

  private isWebGPUBackend(): boolean {
    const renderer = this.ctx.viewContext?.getRenderer() as
      { isWebGPURenderer?: boolean } | undefined;
    return !!renderer?.isWebGPURenderer;
  }

  /**
   * WebGPU model material: swap the glTF material for an equivalent TSL node
   * material. The automatic classic→node upgrade covers basic shading, but
   * not Navara's per-feature extras, so the node graph additionally wires:
   *  - the batch texture (COLOR_SHOW row: per-feature color/show/opacity,
   *    indexed by the `_batchid` attribute, float RGBA bit-decode);
   *  - pick coloring (nvr_batchIdToColor over the global `batchId` attribute);
   *  - `lit: false` → MeshBasicNodeMaterial (the classic NVR_UNLIT define is
   *    a no-op here), and glTFs loaded with KHR_materials_unlit stay unlit.
   *  - selective-effect emissive: with effectIdsMask > 0 the feature's
   *    diffuse×emissiveIntensity is folded into the emissive term so the
   *    (global, HDR-threshold) WebGPU bloom pass highlights it. The classic
   *    MRT effect-id buffer + selective bloom/outline passes do not exist on
   *    this backend, so per-effect selection and outline are approximated,
   *    not reproduced.
   *  - specular enhancer: Blinn-Phong sun glint (computeSpecular in
   *    spucular_pars_fragment.glsl) added to outgoing light via emissive;
   *  - water enhancer: animated normal-map perturbation + Fresnel-weighted
   *    sun glint (computeWaterSpecularSimple in water_pars_fragment.glsl),
   *    driven by the shared time uniform and waterTexture.
   * Gate uniforms are synced per frame from the enhancer state.
   * Not ported: the sky-env reflection term (tSkyEnvMap is always null on
   * the WebGPU forward path — see ThreeView) and the applyWaterNormal
   * G-buffer normal mix (no MRT G-buffer on this backend). Specular/water
   * only apply to lit materials, matching the classic standard/physical-only
   * enhancer.
   */
  private _convertMeshToNodeMaterial(
    mesh: Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
    lit: boolean | undefined,
  ): void {
    const { webgpu, tsl: T } = getWebGPU();
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const src = mesh.material as any;
    const unlit = lit === false || src.type === "MeshBasicMaterial";
    const m: any = unlit
      ? new webgpu.MeshBasicNodeMaterial()
      : src.type === "MeshPhysicalMaterial"
        ? new webgpu.MeshPhysicalNodeMaterial()
        : new webgpu.MeshStandardNodeMaterial();

    m.color.copy(src.color);
    m.opacity = src.opacity;
    m.transparent = src.transparent;
    m.alphaTest = src.alphaTest;
    m.side = src.side;
    m.depthTest = src.depthTest;
    m.depthWrite = src.depthWrite;
    m.map = src.map ?? null;
    // MeshBasicNodeMaterial has no emissive slot, but the model enhancer
    // writes material.emissive unconditionally — provide one (materialEmissive
    // in the node graph binds to this object).
    m.emissive ??= new Color(0x000000);
    if (src.emissive) {
      m.emissive.copy(src.emissive);
      m.emissiveIntensity = src.emissiveIntensity ?? 1;
      m.emissiveMap = src.emissiveMap ?? null;
    }
    if (!unlit) {
      m.metalness = src.metalness;
      m.roughness = src.roughness;
      m.normalMap = src.normalMap ?? null;
      if (src.normalScale && m.normalScale) {
        m.normalScale.copy(src.normalScale);
      }
      m.aoMap = src.aoMap ?? null;
      m.envMap = src.envMap ?? null;
    }
    Object.assign(m.userData, src.userData);

    // The batch texture's layout rows are baked into the node graph as
    // constants, so every supported row/slot must be allocated before the
    // graph is built (the classic path allocates lazily on first write).
    // Eager identity writes whiten src.color (batch-color multiplier
    // convention), but m.color was already copied above and the enhancer
    // re-whitens through batchColorEnabled on a real color write.
    this._initBatchDataTexture(mesh);
    eagerAllocateBatchTexture(src, MODEL_BATCH_SUPPORT, { showOpacity: true });
    const batch =
      this.batchLength != null &&
      mesh.geometry.getAttribute("_batchid") != null
        ? createWebgpuBatchSampler(T, src, this.batchLength)
        : null;

    const handles: ModelWebgpuHandles = {
      gateColor: T.uniform(0),
      gateEmissive: T.uniform(0),
      uPickable: T.uniform(0),
      uEffectGate: T.uniform(0),
      uEmissiveIntensity: T.uniform(0),
      uSpecular: T.uniform(0),
      uShininess: T.uniform(0),
      uSpecularStrength: T.uniform(0),
      uIor: T.uniform(WATER_IOR),
      uWater: T.uniform(0),
      uWaterScaleNormal: T.uniform(0),
      uWaterSpeed: T.uniform(0),
      uReflectivity: T.uniform(0),
      uTime: T.uniform(0),
      uSunDirView: T.uniform(new Vector3(0, 0, 1)),
      uSunColor: T.uniform(new Color(0, 0, 0)),
      waterNormalNode: null,
    };
    const {
      gateColor,
      gateEmissive,
      uPickable,
      uEffectGate,
      uEmissiveIntensity,
      uSpecular,
      uShininess,
      uSpecularStrength,
      uIor,
      uWater,
      uWaterScaleNormal,
      uWaterSpeed,
      uTime,
      uSunDirView,
      uSunColor,
    } = handles as unknown as Record<string, any>;
    let vBatchColor: any = T.varying(T.vec3(1, 1, 1), "nvr_mBatchColor");
    let vShow: any = T.varying(T.float(1), "nvr_mShow");
    let vOpacity: any = T.varying(T.float(1), "nvr_mOpacity");
    let vBatchEmissive: any = T.varying(T.vec3(0, 0, 0), "nvr_mBatchEmissive");
    if (batch) {
      handles.batchTex = batch.texNode;
      const colorNode = batch.vec3("color");
      if (colorNode) vBatchColor = T.varying(colorNode, "nvr_mBatchColor");
      const showOpacity = batch.showOpacity();
      if (showOpacity) {
        vShow = T.varying(showOpacity.show, "nvr_mShow");
        vOpacity = T.varying(showOpacity.opacity, "nvr_mOpacity");
      }
      const emissiveNode = batch.vec3("emissive");
      const emissiveIntensity = batch.scalar("emissiveIntensity");
      if (emissiveNode && emissiveIntensity) {
        vBatchEmissive = T.varying(
          emissiveNode.mul(emissiveIntensity),
          "nvr_mBatchEmissive",
        );
      }
    }

    // Pick color from the GLOBAL batch id (matches pick.glsl). The id
    // varying must be FLAT like the classic `flat out float nvr_vBatchId`:
    // batchIds reach 2^24 where f32 has ulp=1, so smooth interpolation can
    // arrive a step off at some pixels and corrupt the low byte (same idiom
    // as pickableMeshWrapper.ts).
    let pickId: any = T.float(0);
    if (mesh.geometry.getAttribute("batchId") != null) {
      pickId = T.varying(T.attribute("batchId"), "nvr_mPickId");
      T.nodeObject(pickId).setInterpolation("flat");
    }
    const pickColor = T.vec3(
      pickId.div(65536).floor().div(255),
      pickId.div(256).mod(256).floor().div(255),
      pickId.mod(256).floor().div(255),
    );

    // Vertex color is folded into the graph manually: when batch color is
    // active it REPLACES the vertex color (see batch_texture_vertex.glsl), so
    // the automatic NodeMaterial vertexColors multiply would double-apply.
    // Presence of the COLOR_0 attribute decides, not material.vertexColors:
    // GLTFLoader sets that flag, but Navara b3dm materials may render vertex
    // colors in classic mode with the flag unset. T.vertexColor() falls back
    // to white when the attribute is absent.
    const hasColor = mesh.geometry.hasAttribute("color");
    const vColor: any = hasColor ? T.vertexColor().rgb : null;
    m.vertexColors = false;
    m.colorNode = T.Fn(() => {
      // show_fragment.glsl: discard hidden / fully transparent features.
      vShow.lessThan(0.5).discard();
      vOpacity.lessThanEqual(0).discard();
      const featureColor = T.mix(
        vColor ?? T.vec3(1, 1, 1),
        vBatchColor,
        gateColor,
      );
      return T.vec4(
        T.materialColor.rgb.mul(featureColor).mul(uPickable.oneMinus()),
        1,
      );
    })();
    // Enhancer effects folded into the emissive term (added to outgoing
    // light like the classic outgoingLight += specular):
    //  - selective effect: diffuse×emissiveIntensity when an effect selects
    //    this model (GBUFFER_EFFECT_WRITE_BUILTIN's emissive payload);
    //  - specular/water sun glints (lit materials only, see the docstring).
    let fxEmissive: any = T.materialEmissive.add(
      T.materialColor.rgb.mul(uEmissiveIntensity).mul(uEffectGate),
    );
    if (!unlit) {
      // F_Schlick(F0, 1, ndotV) with F0 derived from the IOR
      // (IorToFresnel0 in water_pars_fragment.glsl).
      const fresnelFromIor = (ior: any, ndotV: any): any => {
        const f0 = T.pow2(ior.sub(1).div(ior.add(1)));
        return f0.add(
          T.float(1).sub(f0).mul(ndotV.clamp(0, 1).oneMinus().pow(5)),
        );
      };
      // specularColor(): Blinn-Phong glint toward the sun. uSunDirView
      // points TOWARD the light (three's directional light uniform
      // convention), so the incident direction is its negation.
      const sunSpecular = (normal: any, toEye: any): any => {
        const ndotL = normal.dot(uSunDirView).max(0);
        const reflection = T.reflect(uSunDirView.negate(), normal).normalize();
        const glint = toEye
          .dot(reflection)
          .max(0)
          .pow(uShininess)
          .mul(uSunColor)
          .mul(uSpecularStrength);
        // Classic early-outs when ndotL <= 0.
        return glint.mul(ndotL.greaterThan(0).toFloat());
      };
      const toEye: any = T.positionViewDirection;
      const origNormal: any = T.normalViewGeometry;

      // Non-water specular (uSpecular gate, suppressed while water is on —
      // the classic WATER #ifdef replaces this branch entirely).
      const specNonWater = sunSpecular(origNormal, toEye)
        .mul(fresnelFromIor(uIor, origNormal.dot(toEye).max(0)))
        .mul(uSpecular)
        .mul(uWater.oneMinus());

      // Water: animated normal from 4 time-shifted normal-map taps
      // (getNoise in water_pars_fragment.glsl), uv = world xy × scale.
      const waterNormalTex = T.texture(WEBGPU_WATER_NORMAL_PLACEHOLDER);
      handles.waterNormalNode = waterNormalTex as unknown as {
        value: Texture;
      };
      const wuv: any = T.positionWorld.xy.mul(uWaterScaleNormal);
      const wt: any = uTime.mul(uWaterSpeed);
      const noise = waterNormalTex
        .sample(wuv.div(103).add(T.vec2(wt.div(17), wt.div(29))))
        .add(
          waterNormalTex.sample(
            wuv.div(107).sub(T.vec2(wt.div(-19), wt.div(31))),
          ),
        )
        .add(
          waterNormalTex.sample(
            wuv.div(T.vec2(8907, 9803)).add(T.vec2(wt.div(101), wt.div(97))),
          ),
        )
        .add(
          waterNormalTex.sample(
            wuv.div(T.vec2(1091, 1027)).sub(T.vec2(wt.div(109), wt.div(-113))),
          ),
        )
        .mul(0.5)
        .sub(1);
      const waterNormal: any = noise.xzy.mul(T.vec3(1.5, 1.0, 1.5)).normalize();
      const specWater = sunSpecular(waterNormal, toEye)
        .mul(fresnelFromIor(T.float(WATER_IOR), waterNormal.dot(toEye).max(0)))
        .mul(uWater);

      fxEmissive = fxEmissive.add(specNonWater).add(specWater);

      // The water normal replaces the shading normal (classic skips
      // normal_fragment_maps in the WATER branch). materialNormal keeps the
      // normal-map path alive when water is off.
      m.normalNode = T.mix(T.materialNormal, waterNormal, uWater);
    }
    m.emissiveNode = T.mix(
      // batch_emissive_fragment.glsl: a written batch emissive REPLACES the
      // material's emissive term (folded rgb × intensity in the vertex stage).
      T.mix(fxEmissive, vBatchEmissive, gateEmissive),
      pickColor,
      uPickable,
    );
    m.opacityNode = T.mix(
      T.materialOpacity.mul(vOpacity),
      T.float(1),
      uPickable,
    );
    /* eslint-enable @typescript-eslint/no-explicit-any */

    // Batch writes address mesh.material, which is about to become the node
    // material; attach it to the classic material's batch texture state so
    // updateBatchAttribute keeps landing (the state lives in a module-private
    // WeakMap keyed by material, not in userData).
    attachBatchedMaterial(src, m);

    m.userData.nvrWebgpu = handles;
    mesh.material = m as ModelMaterial;

    // Per-render sync of the gate uniforms from the enhancer state (the
    // enhancer itself is mounted on this node material, so plain material
    // props flow without mirroring; only the TSL uniforms need feeding).
    mesh.onBeforeRender = (_renderer, _scene, camera) => {
      const enhancer = this._enhancers.get(mesh);
      if (!enhancer) return;
      const { base: b, water: w } = enhancer.states();
      handles.gateColor.value = b.batchColorEnabled ? 1 : 0;
      handles.gateEmissive.value = this._batchEmissiveUsed.get(mesh) ? 1 : 0;
      handles.uPickable.value = b.pickable ? 1 : 0;
      if (handles.batchTex) {
        syncWebgpuBatchTexture({ texNode: handles.batchTex }, mesh.material);
      }
      handles.uEffectGate.value = b.effectIdsMask > 0 ? 1 : 0;
      handles.uEmissiveIntensity.value = b.emissiveIntensity;
      handles.uWater.value = w.useWater ? 1 : 0;
      handles.uSpecular.value = w.specular ? 1 : 0;
      handles.uShininess.value = w.shininess;
      handles.uSpecularStrength.value = w.specularStrength;
      handles.uIor.value = w.ior;
      handles.uWaterScaleNormal.value =
        ({ small: 1, medium: 2, large: 4, ocean: 8 } as Record<string, number>)[
          w.waterScale
        ] ?? 1;
      handles.uWaterSpeed.value = w.flowSpeed;
      handles.uReflectivity.value = w.specularStrength;
      handles.uTime.value = this.ctx.uniforms?.time?.value ?? 0;
      const waterTex = (this.ctx.uniforms as any)?.waterTexture?.value;
      if (
        waterTex &&
        handles.waterNormalNode &&
        handles.waterNormalNode.value !== waterTex
      ) {
        handles.waterNormalNode.value = waterTex;
      }
      if (!unlit) {
        this._syncSunHandles(handles, camera);
      }
      // See the vertexColors note above.
      (mesh.material as { vertexColors: boolean }).vertexColors = false;
    };
  }

  /** Cached sun light lookup (the lights group holds only a few objects). */
  private _sunLight: DirectionalLight | null | undefined;

  /**
   * Feed the sun uniforms from the scene's first DirectionalLight (the same
   * light the classic GLSL reads as directionalLights[0]): direction toward
   * the light in view space, color premultiplied by intensity.
   */
  private _syncSunHandles(handles: ModelWebgpuHandles, camera: Camera): void {
    if (this._sunLight === undefined) {
      let found: DirectionalLight | null = null;
      this.ctx.scenes?.light?.traverse((o: ThreeObject3D) => {
        if (!found && (o as DirectionalLight).isDirectionalLight) {
          found = o as DirectionalLight;
        }
      });
      // Cache only on a hit — lights can be registered after the first
      // mesh render, and a cached null would never recover.
      if (found) this._sunLight = found;
    }
    const sun = this._sunLight;
    if (!sun) return;
    // Matches WebGLLights: direction = light→target offset in view space.
    // Reads LOCAL positions: scenes.light's root is identity and the WebGPU
    // path never renders that scene, so matrixWorld there is not refreshed
    // every frame (the lightMirror sync reads .position the same way).
    _sunWorldPos.copy(sun.position);
    _sunTargetPos.copy(sun.target.position);
    _sunWorldPos.sub(_sunTargetPos);
    if (_sunWorldPos.lengthSq() === 0) return;
    _sunWorldPos.transformDirection(camera.matrixWorldInverse);
    handles.uSunDirView.value.copy(_sunWorldPos);
    handles.uSunColor.value.copy(sun.color).multiplyScalar(sun.intensity);
  }

  /**
   * WebGPU point-cloud material: PointsNodeMaterial carrying over the basic
   * props plus the pnts enhancer's GLSL extras as TSL:
   *  - height offset along the geodetic normal (classic adds
   *    viewMatrix×uGeodeticNormal×uAddHeight to mvPosition — reproduced by
   *    overriding setupPositionView);
   *  - divideColor: vertex colors scaled by 1/65535 (PNTS color range
   *    normalization), folded manually like the mesh path (the automatic
   *    vertexColors multiply would double-apply).
   * Uniform state is fed per frame from the enhancer via onBeforeRender.
   */
  private _convertPointsToNodeMaterial(
    points: Points<BufferGeometry<NormalBufferAttributes>, PointsMaterial>,
  ): void {
    const { webgpu, tsl: T } = getWebGPU();
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const src = points.material;
    const handles: PntsWebgpuHandles = {
      uAddHeight: T.uniform(0) as unknown as { value: number },
      uGeodeticNormal: T.uniform(new Vector3()) as unknown as {
        value: Vector3;
      },
      uDivideColor: T.uniform(0) as unknown as { value: number },
    };
    const { uAddHeight, uGeodeticNormal, uDivideColor } =
      handles as unknown as Record<string, any>;

    // PointsNodeMaterial.setupPositionView is
    // modelViewMatrix × positionLocal; add the classic view-space offset
    // (pntsEnhancer/shader.ts: mvPosition += viewMatrix*geodeticNormal*h).
    class NvrPointsNodeMaterial extends webgpu.PointsNodeMaterial {
      setupPositionView(): any {
        return T.modelViewMatrix
          .mul(T.vec3(T.positionLocal))
          .xyz.add(
            T.cameraViewMatrix
              .mul(T.vec4(uGeodeticNormal, 0))
              .xyz.mul(uAddHeight),
          );
      }
    }
    const m = new NvrPointsNodeMaterial();
    m.color.copy(src.color);
    m.size = src.size;
    m.sizeAttenuation = src.sizeAttenuation;
    m.map = src.map ?? null;
    m.opacity = src.opacity;
    m.transparent = src.transparent;
    m.alphaTest = src.alphaTest;
    m.depthTest = src.depthTest;
    m.depthWrite = src.depthWrite;

    // diffuse × (vertexColor / 65535 when divideColor) — see
    // pntsEnhancer/shader.ts's color_vertex replacement. The divisor only
    // applies to the vertex-color contribution: without a color attribute
    // the classic USE_COLOR path compiles out entirely and the raw material
    // color is used.
    const hasColor = points.geometry.hasAttribute("color");
    const divisor: any = T.mix(
      T.float(1),
      T.float(1 / PNTS_COLOR_DIVISOR),
      uDivideColor,
    );
    const vColor: any = hasColor
      ? T.vertexColor().rgb.mul(divisor)
      : T.vec3(1, 1, 1);
    m.vertexColors = false;
    m.colorNode = T.vec4(T.materialColor.rgb.mul(vColor), 1);
    /* eslint-enable @typescript-eslint/no-explicit-any */

    Object.assign(m.userData, src.userData);
    m.userData.nvrWebgpu = handles;
    points.material = m as unknown as PointsMaterial;

    // Per-render sync of the TSL uniforms from the enhancer state (color and
    // point size are plain material props the enhancer writes directly).
    points.onBeforeRender = () => {
      const enhancer = this._pntsEnhancers.get(points);
      if (!enhancer) return;
      const s = enhancer.states();
      handles.uAddHeight.value = s.height;
      handles.uGeodeticNormal.value.set(
        s.geodeticNormal.x,
        s.geodeticNormal.y,
        s.geodeticNormal.z,
      );
      handles.uDivideColor.value = (s as { divideColor?: boolean }).divideColor ? 1 : 0;
    };
  }

  /**
   * Override a material that is used to generate a shadow map.
   */
  initDepthMaterial(
    mesh: Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
    enhancer: ModelMaterialEnhancer,
  ) {
    mesh.customDepthMaterial = createShadowDepthMaterial(
      mesh.material,
      (shader) => enhancer.transformShader(shader),
    );
  }

  /**
   * Host-app visibility override (LACS 补一一〇 P24).
   *
   * `_update` rewrites `this.visible` from the descriptor's material every
   * frame, so an application that sets `mesh.visible = false` to cull an
   * object (e.g. sub-pixel models at city scale) has the change silently
   * reverted on the next update — the object keeps being drawn and the cull
   * buys nothing. Honouring this flag is inert unless the host sets it.
   */
  private _externallyHidden = false;

  /** Hide/show regardless of the descriptor's material state. */
  setExternallyHidden(hidden: boolean): void {
    this._externallyHidden = hidden;
    if (hidden) this.visible = false;
  }

  _update(material: NavaraModelMaterial, active: boolean) {
    this.visible = (material.show ?? true) && active && !this._externallyHidden;
    this._syncBoundingBoxes(!!material.showBoundingBox);

    // Each enhancer map is populated only for its node type, so empty maps
    // iterate trivially. Running both lets mixed-mode glTFs (POINTS + TRIANGLES)
    // update each node with the right props without a tile-level flag.
    const updateProps = this.buildUpdateProps(material);
    for (const [mesh, enhancer] of this._enhancers) {
      // Once per-feature batch colors own the material, `material.color` must
      // not re-tint the untinted base color (mirrors polygon/polyline's
      // `batchColorEnabled ? undefined : material.color` guard).
      enhancer.update(
        enhancer.states().base.batchColorEnabled
          ? { ...updateProps, base: { ...updateProps.base, color: undefined } }
          : updateProps,
      );
      mesh.castShadow = !!material.castShadow;
      mesh.receiveShadow = !!material.receiveShadow;
      applyLitOption(mesh.material, material.lit);
    }

    const pntsProps: PntsProps = {
      color: material.color,
      pointSize: material.pointSize,
      height: material.height,
      opacity: material.opacity,
      transparent: material.transparent,
      depthWrite: material.depthWrite,
    };
    for (const enhancer of this._pntsEnhancers.values()) {
      enhancer.update(pntsProps);
    }
  }

  /**
   * Build update props from NavaraModelMaterial for enhancer.update().
   */
  private buildUpdateProps(material: NavaraModelMaterial): ModelMaterialProps {
    return {
      base: {
        color: material.color,
        metalness: material.metalness,
        roughness: material.roughness,
        emissiveColor: material.emissiveColor,
        emissiveIntensity: material.emissiveIntensity,
        transparent: material.transparent,
        opacity: material.opacity,
        depthWrite: material.depthWrite,
        effectIdsMask:
          this.ctx.viewContext.selectiveEffectRegistry?.computeMask(
            material.effectIds ?? [],
          ) ?? 0,
      },
      water: {
        water: material.water,
        waveExaggeration: material.waveExaggeration,
        waterScale: material.waterScale,
        waveStrength: material.waveStrength,
        waveDirection: material.waveDirection,
        flowSpeed: material.flowSpeed,
        shininess: material.shininess,
        specularStrength: material.specularStrength,
        specular: material.specular,
        ior: material.ior,
      },
    };
  }

  traverseMesh(
    callback: (
      m: Mesh<BufferGeometry<NormalBufferAttributes>, ModelMaterial>,
    ) => void,
  ) {
    this.traverse((object: Object3D) => {
      if (!(object instanceof Mesh)) {
        return;
      }
      callback(object);
    });
  }

  _setFeatureColor(color: Color, m?: ModelMaterial) {
    if (m) {
      tintBaseColor(m, color);
    }
  }

  _getFeatureColor(): Color {
    throw new Unimplemented();
  }

  _setFeatureShow(visible: boolean): void {
    this.visible = visible;
  }

  _setFrustumCulled(culled: boolean): void {
    this.frustumCulled = culled;
  }

  onBeforePicking(): void {
    this.setPickable(true);
  }

  onAfterPicking(): void {
    this.setPickable(false);
  }

  private setPickable(pickable: boolean): void {
    for (const enhancer of this._enhancers.values()) {
      enhancer.update({ base: { pickable } });
    }
  }

  getRenderable(): Object3D {
    return this;
  }

  _setFeatureHeight(_height: number) {
    // Height adjustment via batch textures is currently not implemented.
  }

  _setFeatureOpacity(opacity: number): void {
    // Only reached on the non-batched evaluator path (no _batchid attribute),
    // where updateBatchAttribute would reject the write (batchLength 0) —
    // apply directly to the materials instead, like _setFeatureColor/Show.
    this.traverseMesh((m) => {
      this._enhancers.get(m)?.update({ base: { opacity } });
    });
  }

  dispose() {
    this.traverseMesh((m) => {
      this.ctx.viewContext.removeShadowMaterial(m.material);
    });
    for (const enhancer of this._enhancers.values()) {
      enhancer.mutates().water.dispose();
    }
  }
}
