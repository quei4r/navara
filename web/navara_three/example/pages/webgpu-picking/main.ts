/**
 * WebGPU picking verification page.
 *
 * Runs the view on the experimental WebGPU forward path (`?webgl` forces the
 * WebGPURenderer's WebGL2 backend — same node pipeline, no WebGPU device) and
 * exercises the ported GPU picking: boxes use the turnkey
 * `PickableMeshWrapper`, spheres use `PickableInstancedMeshWrapper`. Clicking
 * a mesh highlights it and records the pick on `window.__pickDebug`, which
 * the automated click test (`scripts/`-style playwright probes) reads to
 * assert that the picked batchId belongs to the clicked mesh.
 */
import ThreeView, {
  Color,
  MeshDesc,
  PickableInstancedMeshWrapper,
  PickableMeshWrapper,
  geodeticToVector3,
  northUpEastToFixedFrame,
  type MeshConfig,
  type MeshUpdate,
  type ViewContext,
} from "@navaramap/three";
import {
  BoxGeometry,
  Color as ThreeColor,
  InstancedMesh,
  Matrix4,
  Mesh,
  SphereGeometry,
  Vector3,
  type WebGLRenderer,
} from "three";
import { MeshBasicNodeMaterial, WebGPURenderer } from "three/webgpu";

import { atZoneTime } from "../../helpers/control";

// ============================================================================
// Custom descs over node materials — the WebGPU forward path renders
// NodeMaterial only, so the stock WebGL-material descs can't be used here.
// ============================================================================

type GpuBoxConfig = MeshConfig & {
  pickable?: boolean;
  gpuBox?: { size?: number; color?: Color };
};
type GpuBoxUpdate = MeshUpdate & {
  gpuBox?: { color?: Color };
};

class GpuBoxMeshDesc extends MeshDesc<
  GpuBoxConfig,
  GpuBoxUpdate,
  Mesh<BoxGeometry, MeshBasicNodeMaterial>
> {
  private cfg: GpuBoxConfig;
  private pickable?: PickableMeshWrapper;

  constructor(view: ThreeView, ctx: ViewContext, config: GpuBoxConfig) {
    super(view, ctx, config);
    this.cfg = config;
  }

  get batchId(): number | undefined {
    return this.pickable?.batchId;
  }

  createMesh() {
    const cfg = this.cfg.gpuBox ?? {};
    const size = cfg.size ?? 60;
    const material = new MeshBasicNodeMaterial({
      color: cfg.color?.raw ?? 0xffffff,
    });
    const mesh = new Mesh(new BoxGeometry(size, size, size), material);
    if (this.cfg.pickable) {
      this.pickable = new PickableMeshWrapper(mesh, this.ctx);
      this.ctx.registerPickableMesh(this.id, this.pickable);
    }
    return mesh;
  }

  onUpdateConfig(updates: GpuBoxUpdate): void {
    if (updates.gpuBox?.color !== undefined && this._instance) {
      this._instance.material.color = updates.gpuBox.color.raw;
      this.emit("needsUpdate");
    }
    super.onUpdateConfig(updates);
  }

  override onDestroy(): void {
    if (this.pickable) {
      this.ctx.unregisterPickableMesh(this.id);
      this.pickable = undefined;
    }
    this._instance?.geometry.dispose();
    this._instance?.material.dispose();
    super.onDestroy();
  }
}

type GpuSpheresConfig = MeshConfig & {
  pickable?: boolean;
  gpuSpheres?: { radius?: number; count?: number; spacing?: number };
};
type GpuSpheresUpdate = MeshUpdate & {
  gpuSpheres?: { colors?: number[] };
};

class GpuInstancedSpheresMeshDesc extends MeshDesc<
  GpuSpheresConfig,
  GpuSpheresUpdate,
  InstancedMesh
> {
  private cfg: GpuSpheresConfig;
  private pickable?: PickableInstancedMeshWrapper;

  constructor(view: ThreeView, ctx: ViewContext, config: GpuSpheresConfig) {
    super(view, ctx, config);
    this.cfg = config;
  }

  get batchIds(): number[] | undefined {
    return this.pickable?.batchIds;
  }

  createMesh() {
    const cfg = this.cfg.gpuSpheres ?? {};
    const count = cfg.count ?? 4;
    const radius = cfg.radius ?? 35;
    const spacing = cfg.spacing ?? 120;
    const material = new MeshBasicNodeMaterial({ color: 0xffffff });
    const mesh = new InstancedMesh(
      new SphereGeometry(radius, 24, 16),
      material,
      count,
    );
    const m = new Matrix4();
    for (let i = 0; i < count; i++) {
      m.makeTranslation((i - (count - 1) / 2) * spacing, radius, 0);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, new ThreeColor().setHSL(i / count, 0.8, 0.55));
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (this.cfg.pickable) {
      this.pickable = new PickableInstancedMeshWrapper(mesh, count, this.ctx);
      this.ctx.registerPickableMesh(this.id, this.pickable);
    }
    return mesh;
  }

  onUpdateConfig(updates: GpuSpheresUpdate): void {
    const instance = this._instance;
    if (updates.gpuSpheres?.colors && instance) {
      updates.gpuSpheres.colors.forEach((hex, i) => {
        instance.setColorAt(i, new Color().setHex(hex).raw);
      });
      if (instance.instanceColor) {
        instance.instanceColor.needsUpdate = true;
      }
      this.emit("needsUpdate");
    }
    super.onUpdateConfig(updates);
  }

  override onDestroy(): void {
    if (this.pickable) {
      this.ctx.unregisterPickableMesh(this.id);
      this.pickable = undefined;
    }
    this._instance?.geometry.dispose();
    super.onDestroy();
  }
}

type CustomDescriptions = {
  mesh: GpuBoxConfig | GpuSpheresConfig;
};

// ============================================================================
// Scene setup
// ============================================================================

type PickRecord = { name: string; batchId: number } | null;

type PickTarget = {
  name: string;
  batchId: number;
  worldPos: () => Vector3;
  setHighlight: (on: boolean) => void;
};

const bootstrap = async () => {
  const canvas = document.createElement("canvas");
  canvas.id = "navara-canvas";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  const root = document.createElement("div");
  root.id = "navara-root";
  root.style.width = "100vw";
  root.style.height = "100vh";
  root.appendChild(canvas);
  document.body.appendChild(root);

  const params = new URLSearchParams(window.location.search);
  const renderer = new WebGPURenderer({
    canvas,
    antialias: true,
    stencil: true,
    logarithmicDepthBuffer: true,
    forceWebGL: params.has("webgl"),
  });
  await renderer.init();

  const view = new ThreeView<CustomDescriptions>({
    canvas,
    renderer: renderer as unknown as WebGLRenderer,
    animation: true,
  });
  await view.init();

  // Daylight so the unlit test colors read clearly against the lit globe.
  view.atmosphere.date = atZoneTime(view.atmosphere.date, 10);

  view.registerMesh("gpuBox", GpuBoxMeshDesc);
  view.registerMesh("gpuSpheres", GpuInstancedSpheresMeshDesc);

  // Camera looks down at the test meshes from above.
  view.setCamera({
    lng: 139.7671,
    lat: 35.6802,
    height: 1100,
    heading: 0,
    pitch: -70,
    roll: 0,
  });

  const origin = geodeticToVector3({
    lat: 35.681236,
    lng: 139.767125,
    height: 0,
  });
  const nueFrame = northUpEastToFixedFrame(origin);

  const targets: PickTarget[] = [];
  const byBatchId = new Map<number, PickTarget>();

  // Three individual boxes in a row.
  const BOX_COLORS = [0xff4444, 0x44dd44, 0x4488ff];
  for (let i = 0; i < BOX_COLORS.length; i++) {
    const local = new Matrix4().makeTranslation((i - 1) * 150, 60, -120);
    const layer = view.addMesh<GpuBoxMeshDesc>({
      pickable: true,
      gpuBox: { size: 70, color: new Color().setHex(BOX_COLORS[i]) },
      matrixWorld: nueFrame.clone().multiply(local),
    });
    const mesh = layer.ref.raw;
    if (!mesh) continue;
    const target: PickTarget = {
      name: `Box-${i}`,
      batchId: layer.ref.batchId ?? 0,
      worldPos: () => mesh.getWorldPosition(new Vector3()),
      setHighlight: (on) =>
        layer.update({
          gpuBox: { color: new Color().setHex(on ? 0xffffff : BOX_COLORS[i]) },
        }),
    };
    targets.push(target);
    byBatchId.set(target.batchId, target);
  }

  // One instanced sphere row; each instance gets its own batchId.
  const spheresLayer = view.addMesh<GpuInstancedSpheresMeshDesc>({
    pickable: true,
    gpuSpheres: { radius: 35, count: 4, spacing: 120 },
    matrixWorld: nueFrame
      .clone()
      .multiply(new Matrix4().makeTranslation(0, 0, 150)),
  });
  const spheresMesh = spheresLayer.ref.raw;
  if (!spheresMesh) throw new Error("spheres layer did not produce a mesh");
  const sphereOrigColors =
    spheresLayer.ref.batchIds?.map((_, i) =>
      new ThreeColor().setHSL(i / 4, 0.8, 0.55).getHex(),
    ) ?? [];
  spheresLayer.ref.batchIds?.forEach((batchId, i) => {
    const m = new Matrix4();
    spheresMesh.getMatrixAt(i, m);
    const target: PickTarget = {
      name: `Sphere-${i}`,
      batchId,
      worldPos: () =>
        new Vector3().setFromMatrixPosition(
          m.clone().premultiply(spheresMesh.matrixWorld),
        ),
      setHighlight: (on) => {
        const colors = (sphereOrigColors ?? []).slice();
        if (on) colors[i] = 0xffffff;
        spheresLayer.update({ gpuSpheres: { colors } });
      },
    };
    targets.push(target);
    byBatchId.set(batchId, target);
  });

  // On-screen readout (also what the visual check reads).
  const readout = document.createElement("div");
  readout.style.cssText =
    "position:fixed;left:12px;bottom:12px;padding:8px 12px;background:rgba(0,0,0,.75);color:#7f7;font:16px monospace;z-index:10;";
  readout.textContent = "picked: (none)";
  document.body.appendChild(readout);

  let lastPicked: PickRecord | undefined = undefined;
  let selected: PickTarget | null = null;

  view.on("featureClick", (pickInfo) => {
    if (selected) {
      selected.setHighlight(false);
      selected = null;
    }
    const target = pickInfo ? byBatchId.get(pickInfo.batchId) : undefined;
    if (target) {
      selected = target;
      target.setHighlight(true);
      lastPicked = { name: target.name, batchId: target.batchId };
      readout.textContent = `picked: ${target.name} (#${target.batchId})`;
    } else {
      lastPicked = null;
      readout.textContent = "picked: (none)";
    }
  });

  // Project a target's world position to CSS pixel coords for the click test.
  const screenPosOf = (name: string) => {
    const target = targets.find((t) => t.name === name);
    if (!target) return null;
    const ndc = target.worldPos().project(view.camera.raw);
    const rect = canvas.getBoundingClientRect();
    return {
      x: rect.left + (ndc.x * 0.5 + 0.5) * rect.width,
      y: rect.top + (1 - (ndc.y * 0.5 + 0.5)) * rect.height,
    };
  };

  (window as unknown as Record<string, unknown>).__pickDebug = {
    ready: true,
    view,
    renderer,
    targets: targets.map((t) => ({ name: t.name, batchId: t.batchId })),
    screenPosOf,
    getLastPicked: () => lastPicked,
    clearLastPicked: () => {
      lastPicked = undefined;
    },
  };

  // Await one rendered frame so positions/projection are live for probes.
  view.forceUpdate();
};

bootstrap().catch((e) => {
  console.error(e);
  const div = document.createElement("div");
  div.style.cssText =
    "position:fixed;inset:0;display:flex;align-items:center;justify-content:center;color:#f66;font:14px monospace;white-space:pre-wrap;padding:2em;";
  div.textContent = String(e?.stack ?? e);
  document.body.appendChild(div);
});
