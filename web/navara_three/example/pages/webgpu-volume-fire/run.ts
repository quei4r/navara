import ThreeView, {
  JAPAN_GSI_ELEVATION_DECODER,
  MeshDesc,
  type MeshConfig,
  type PassKey,
  type ViewContext,
} from "@navaramap/three";
import {
  ACESFilmicToneMapping,
  AdditiveBlending,
  BoxGeometry,
  ClampToEdgeWrapping,
  Color,
  HalfFloatType,
  LinearFilter,
  Mesh,
  PointLight,
  RepeatWrapping,
  RGBAFormat,
  Vector3,
} from "three";
import { snoiseVec3 } from "three/addons/tsl/math/curlNoise.js";
import {
  Fn,
  If,
  cameraPosition,
  float,
  floor,
  fract,
  frameId,
  instanceIndex,
  interleavedGradientNoise,
  lights,
  max,
  min,
  mix,
  mx_noise_float,
  screenCoordinate,
  smoothstep,
  storage,
  storageTexture,
  texture3D,
  textureStore,
  uint,
  uniform,
  vec2,
  vec3,
  vec4,
  uvec3,
} from "three/tsl";
import {
  MeshBasicNodeMaterial,
  MeshLambertNodeMaterial,
  Storage3DTexture,
  StorageInstancedBufferAttribute,
  VolumeNodeMaterial,
  type WebGPURenderer,
} from "three/webgpu";

import { TERRAIN_DATASETS, TILE_DATASETS } from "../../helpers/constants";

// ---------------------------------------------------------------------------
// Faithful port of the three.js `webgpu_volume_fire` example (MIT), adapted
// for Navara: the simulation runs in the example's units (12×12×24 m grid)
// while the rendered box is RENDER_SCALE times bigger, so float32
// world-position quantisation stays harmless at globe scale. The march maps
// world → sim space through the mesh's orthonormal local frame.
// ---------------------------------------------------------------------------

const GRID_SIZE_X = 100;
const GRID_SIZE_Y = 100;
const GRID_SIZE_Z = 200;
const CELL_COUNT = GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z;
const PRESSURE_ITERATIONS = 2;
const TEXEL_X = 1 / GRID_SIZE_X;
const TEXEL_Y = 1 / GRID_SIZE_Y;
const TEXEL_Z = 1 / GRID_SIZE_Z;
// Emitter disc height (grid Y cell), above the velocity boundary fade
const EMIT_CELL_Y = 12;

// Simulation space (the example's constants).
const SIZE_X = 12;
const SIZE_Y = 12;
const SIZE_Z = 24;

// Rendered volume size (world metres).
const RENDER_SCALE = 10;
const RENDER_X = SIZE_X * RENDER_SCALE;
const RENDER_Y = SIZE_Y * RENDER_SCALE;
const RENDER_Z = SIZE_Z * RENDER_SCALE;

const uSimSize = uniform(new Vector3(SIZE_X, SIZE_Y, SIZE_Z));
const uSimScale = uniform(RENDER_SCALE);

// Simulation uniforms (example defaults).
const uDt = uniform(1 / 120);
const uTime = uniform(0);
const uBuoyancy = uniform(3.0);
const uWeight = uniform(0.15);
const uTurbulence = uniform(3.2);
const uTurbulenceDecay = uniform(0.1);
const uTurbFrequency = uniform(10.0);
const uVelDamping = uniform(0.25);
const uCooling = uniform(1.0);
const uDissipation = uniform(0.4);
// Emission defaults are hotter than the example's (7 / 5.5): its emitter
// is the teapot's whole vertex volume, while this page's is a flat disc —
// a plane source needs higher rates to reach the same field strength.
const uEmitDensity = uniform(10.0);
const uEmitTemperature = uniform(15.0);
const uEmitRadius = uniform(0.2);

// Render uniforms (example defaults; intensity doubled for the same
// plane-vs-volume emitter reason).
const uFireIntensity = uniform(50.0);
const uFireGlowSpread = uniform(5.0);
// Optional ramp input normalization (t/uFireTempNorm) for fields that run
// hotter than the example's 0..1-cooled plume; 1.0 = the example mapping.
const uFireTempNorm = uniform(1.0);
const uShadowAbsorption = uniform(2.0);
const uShadowAmbient = uniform(0.5);
const uFireStartColor = uniform(new Color(0xffe68c));
const uFireMidColor = uniform(new Color(0xff7305));
const uFireEndColor = uniform(new Color(0xff0000));
const uAsymmetry = uniform(0.0);
const uPowderStrength = uniform(0.59);
const uMultiScattering = uniform(1.0);

// Local-frame placement, refreshed per frame from the mesh's world matrix.
const uWorldOrigin = uniform(new Vector3());
const uAxisRight = uniform(new Vector3(1, 0, 0));
const uAxisUp = uniform(new Vector3(0, 1, 0));
const uAxisFwd = uniform(new Vector3(0, 0, 1));
const uLightDirLocal = uniform(new Vector3(0, 1, 0));

// Smoke key light in SIM units (the example's SpotLight position: box is
// 12×12×24 sim-metres, light hovers above-left of the flame).
const uKeyLightPosSim = uniform(new Vector3(-4.5, 15.4, 9.0));

// Fixed world-space sun direction (the smoke's key light).
const SUN_WORLD_DIR = new Vector3(0.55, 0.5, 0.62).normalize();

type ComputePass = Parameters<WebGPURenderer["compute"]>[0];

class FireSim {
  readonly velTexA: Storage3DTexture;
  readonly dyeTexA: Storage3DTexture;
  readonly dyeTexNode: ReturnType<typeof texture3D>;

  private readonly dyeTexWriteNode: ReturnType<typeof storageTexture>;
  private readonly curlNoiseTexNode: ReturnType<typeof texture3D>;
  private readonly textures: Storage3DTexture[] = [];

  private readonly curlNoisePass: ComputePass;
  private readonly advectVelocityPass: ComputePass;
  private readonly divergencePass: ComputePass;
  private readonly jacobiPassAB: ComputePass;
  private readonly jacobiPassBA: ComputePass;
  private readonly projectPass: ComputePass;
  private readonly advectDyePass: ComputePass;

  private simulationTime = 0;
  private accumulator = 0;
  private lastT: number | null = null;
  private curlNoiseReady = false;

  private static readonly simCap = (() => {
    if (typeof window === "undefined") return 40;
    const v = Number(new URLSearchParams(window.location.search).get("simcap"));
    return Number.isFinite(v) && v > 0 ? v : 40;
  })();

  constructor(
    private readonly renderer: Pick<WebGPURenderer, "compute">,
    private readonly mesh: Mesh,
  ) {
    const createStorage3D = (wrapRepeat: boolean): Storage3DTexture => {
      const texture = new Storage3DTexture(
        GRID_SIZE_X,
        GRID_SIZE_Y,
        GRID_SIZE_Z,
      );
      texture.format = RGBAFormat;
      texture.type = HalfFloatType;
      texture.minFilter = LinearFilter;
      texture.magFilter = LinearFilter;
      texture.wrapS = wrapRepeat ? RepeatWrapping : ClampToEdgeWrapping;
      texture.wrapT = texture.wrapS;
      texture.wrapR = texture.wrapS;
      this.textures.push(texture);
      return texture;
    };

    const velTexA = createStorage3D(false);
    const velTexB = createStorage3D(false);
    const dyeTexA = createStorage3D(false);
    const dyeTexB = createStorage3D(false);
    const divTex = createStorage3D(false);
    const pressTexA = createStorage3D(false);
    const pressTexB = createStorage3D(false);
    const curlNoiseTex = createStorage3D(true);
    this.velTexA = velTexA;
    this.dyeTexA = dyeTexA;

    this.dyeTexNode = texture3D(dyeTexA);
    this.dyeTexWriteNode = storageTexture(dyeTexB).toWriteOnly();
    this.curlNoiseTexNode = texture3D(curlNoiseTex);

    // Shared helpers (example: getVoxelCoord / coordToUVW).
    const getVoxelCoord = (id: typeof instanceIndex) => {
      const x = id.mod(GRID_SIZE_X as never);
      const y = id.div(GRID_SIZE_X as never).mod(GRID_SIZE_Y as never);
      const z = id.div((GRID_SIZE_X * GRID_SIZE_Y) as never);
      return uvec3(x, y, z);
    };
    const coordToUVW = (coord: ReturnType<typeof uvec3>) =>
      vec3(coord)
        .add(0.5 as never)
        .div(vec3(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z));

    // 0) Precomputed curl-noise force field (example: computeCurlNoisePass).
    this.curlNoisePass = Fn(() => {
      const coord = getVoxelCoord(instanceIndex);
      const uvw = coordToUVW(coord);

      const freq = uTurbFrequency;
      const e = float(0.1).div(freq);
      const dx = vec3(e, 0.0, 0.0);
      const dy = vec3(0.0, e, 0.0);
      const dz = vec3(0.0, 0.0, e);

      const p = uvw.mul(vec3(SIZE_X / SIZE_Y, 1.0, SIZE_Z / SIZE_Y));
      const p_x0 = snoiseVec3(p.sub(dx).mul(freq));
      const p_x1 = snoiseVec3(p.add(dx).mul(freq));
      const p_y0 = snoiseVec3(p.sub(dy).mul(freq));
      const p_y1 = snoiseVec3(p.add(dy).mul(freq));
      const p_z0 = snoiseVec3(p.sub(dz).mul(freq));
      const p_z1 = snoiseVec3(p.add(dz).mul(freq));

      const x = p_y1.z.sub(p_y0.z).sub(p_z1.y).add(p_z0.y);
      const y = p_z1.x.sub(p_z0.x).sub(p_x1.z).add(p_x0.z);
      const z = p_x1.y.sub(p_x0.y).sub(p_y1.x).add(p_y0.x);

      textureStore(
        curlNoiseTex,
        coord,
        vec4(vec3(x, y, z).mul(5.0), 0.0),
      ).toWriteOnly();
    })()
      .compute(CELL_COUNT)
      .setName("fireCurlNoise");

    // 1) Advect velocity + forces (example: advectVelocityPass; the teapot
    //    wind term is omitted — there is no dragged emitter here).
    this.advectVelocityPass = Fn(() => {
      const coord = getVoxelCoord(instanceIndex);
      const uvw = coordToUVW(coord);

      const vel = texture3D(velTexA, uvw, 0).xyz;
      const prevPos = uvw.sub(vel.div(uSimSize).mul(uDt));
      const newVel = texture3D(velTexA, prevPos, 0).xyz.toVar();

      const dye = this.dyeTexNode.sample(uvw).level(0 as never);
      const density = dye.r;
      const temperature = dye.g;
      const age = dye.b;

      const buoyancyForce = temperature
        .mul(uBuoyancy)
        .sub(density.mul(uWeight))
        .mul(SIZE_Y);
      newVel.addAssign(vec3(0, buoyancyForce, 0).mul(uDt));

      const thermalNoisePos = uvw.add(
        vec3(0, age.negate().mul(0.6), age.mul(0.13)).div(uTurbFrequency),
      );
      const decay = age.mul(uTurbulenceDecay.negate()).exp();
      const thermalTurbulence = this.curlNoiseTexNode
        .sample(thermalNoisePos)
        .level(0 as never)
        .xyz.mul(uTurbulence)
        .mul(temperature)
        .mul(decay);

      const ambientNoisePos = uvw
        .mul(0.5)
        .add(vec3(0, uTime.mul(0.25), uTime.mul(0.06)).div(uTurbFrequency));
      const ambientTurbulence = this.curlNoiseTexNode
        .sample(ambientNoisePos)
        .level(0 as never)
        .xyz.mul(uTurbulence.mul(0.2))
        .mul(density);

      newVel.addAssign(
        thermalTurbulence.add(ambientTurbulence).mul(SIZE_Y).mul(uDt),
      );
      newVel.mulAssign(max(float(1).sub(uVelDamping.mul(uDt)), 0));

      const edge = min(uvw, vec3(1).sub(uvw));
      const boundary = smoothstep(0.0, 0.08, min(edge.x, min(edge.y, edge.z)));
      newVel.mulAssign(boundary);

      textureStore(velTexB, coord, vec4(newVel, 0)).toWriteOnly();
    })()
      .compute(CELL_COUNT)
      .setName("fireAdvectVelocity");

    // 2) Divergence (example: divergencePass).
    this.divergencePass = Fn(() => {
      const coord = getVoxelCoord(instanceIndex);
      const uvw = coordToUVW(coord);

      const vR = texture3D(velTexB, uvw.add(vec3(TEXEL_X, 0, 0)), 0).x;
      const vL = texture3D(velTexB, uvw.sub(vec3(TEXEL_X, 0, 0)), 0).x;
      const vU = texture3D(velTexB, uvw.add(vec3(0, TEXEL_Y, 0)), 0).y;
      const vD = texture3D(velTexB, uvw.sub(vec3(0, TEXEL_Y, 0)), 0).y;
      const vF = texture3D(velTexB, uvw.add(vec3(0, 0, TEXEL_Z)), 0).z;
      const vB = texture3D(velTexB, uvw.sub(vec3(0, 0, TEXEL_Z)), 0).z;

      const divergence = vR.sub(vL).add(vU.sub(vD)).add(vF.sub(vB)).mul(0.5);

      textureStore(divTex, coord, vec4(divergence, 0, 0, 0)).toWriteOnly();
    })()
      .compute(CELL_COUNT)
      .setName("fireDivergence");

    // 3) Jacobi pressure (example: jacobi ping-pong).
    const jacobi = (
      pressRead: Storage3DTexture,
      pressWrite: Storage3DTexture,
    ) =>
      Fn(() => {
        const coord = getVoxelCoord(instanceIndex);
        const uvw = coordToUVW(coord);

        const pR = texture3D(pressRead, uvw.add(vec3(TEXEL_X, 0, 0)), 0).x;
        const pL = texture3D(pressRead, uvw.sub(vec3(TEXEL_X, 0, 0)), 0).x;
        const pU = texture3D(pressRead, uvw.add(vec3(0, TEXEL_Y, 0)), 0).x;
        const pD = texture3D(pressRead, uvw.sub(vec3(0, TEXEL_Y, 0)), 0).x;
        const pF = texture3D(pressRead, uvw.add(vec3(0, 0, TEXEL_Z)), 0).x;
        const pB = texture3D(pressRead, uvw.sub(vec3(0, 0, TEXEL_Z)), 0).x;

        const divergence = texture3D(divTex, uvw, 0).x;

        const pressure = pR
          .add(pL)
          .add(pU)
          .add(pD)
          .add(pF)
          .add(pB)
          .sub(divergence)
          .div(6);

        textureStore(pressWrite, coord, vec4(pressure, 0, 0, 0)).toWriteOnly();
      })()
        .compute(CELL_COUNT)
        .setName("fireJacobi");

    this.jacobiPassAB = jacobi(pressTexA, pressTexB);
    this.jacobiPassBA = jacobi(pressTexB, pressTexA);

    // 4) Projection (example: projectPass).
    this.projectPass = Fn(() => {
      const coord = getVoxelCoord(instanceIndex);
      const uvw = coordToUVW(coord);

      const pR = texture3D(pressTexA, uvw.add(vec3(TEXEL_X, 0, 0)), 0).x;
      const pL = texture3D(pressTexA, uvw.sub(vec3(TEXEL_X, 0, 0)), 0).x;
      const pU = texture3D(pressTexA, uvw.add(vec3(0, TEXEL_Y, 0)), 0).x;
      const pD = texture3D(pressTexA, uvw.sub(vec3(0, TEXEL_Y, 0)), 0).x;
      const pF = texture3D(pressTexA, uvw.add(vec3(0, 0, TEXEL_Z)), 0).x;
      const pB = texture3D(pressTexA, uvw.sub(vec3(0, 0, TEXEL_Z)), 0).x;

      const gradient = vec3(pR.sub(pL), pU.sub(pD), pF.sub(pB)).mul(0.5);
      const vel = texture3D(velTexB, uvw, 0).xyz.sub(gradient);

      textureStore(velTexA, coord, vec4(vel, 0)).toWriteOnly();
    })()
      .compute(CELL_COUNT)
      .setName("fireProject");

    // 5) Advect dye (example: advectDyePass).
    this.advectDyePass = Fn(() => {
      const coord = getVoxelCoord(instanceIndex);
      const uvw = coordToUVW(coord);

      const vel = texture3D(velTexA, uvw, 0).xyz;
      const prevPos = uvw.sub(vel.div(uSimSize).mul(uDt));

      const dye = this.dyeTexNode.sample(prevPos).level(0 as never);

      const density = dye.r
        .mul(max(float(1).sub(uDissipation.mul(uDt)), 0))
        .toVar();
      const temperature = dye.g
        .mul(max(float(1).sub(uCooling.mul(uDt)), 0))
        .toVar();

      const gridDims = vec3(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
      const nearestUVW = floor(prevPos.mul(gridDims)).add(0.5).div(gridDims);
      const age = this.dyeTexNode
        .sample(nearestUVW)
        .level(0 as never)
        .b.add(uDt)
        .toVar();

      temperature.assign(temperature.clamp(0 as never, 12 as never));

      If(density.lessThanEqual(0.01), () => {
        age.assign(0.0);
      });

      // Emitter: flickering disc on the bottom slab (the example's
      // teapot-vertex emitter mapped to a fire pit). This runs INSIDE the
      // advect pass on purpose: a separate emit pass can only read the
      // pre-advection texture and would overwrite the advected emitter row
      // with (stale value + emission) every substep, feeding density back
      // into itself until the row saturates (~35x). The emitter spans a
      // few grid rows (Y 9..15, like the teapot's vertex volume): a single
      // row gets cleared by advection faster than it accumulates, leaving
      // the field too thin to read. Y=9+ sits above the velocity boundary
      // fade (smoothstep 0→0.08 zeroes wall-adjacent velocity).
      If(
        coord.y
          .greaterThanEqual(uint(EMIT_CELL_Y - 3))
          .and(coord.y.lessThanEqual(uint(EMIT_CELL_Y + 3))),
        () => {
          const r = vec2(uvw.x.sub(0.5), uvw.z.sub(0.5)).length();

          const flicker = mx_noise_float(
            vec3(uvw.x, uTime.mul(2.5), uvw.z).mul(9.0),
          )
            .mul(0.5)
            .add(0.5);

          // The example writes smoothstep(radius, radius*0.4, x) — inverted
          // edges, undefined in WGSL and driver-dependent. Ascending edges
          // plus oneMinus is the well-defined equivalent.
          const mask = smoothstep(
            uEmitRadius.mul(0.4),
            uEmitRadius,
            r.add(flicker.mul(0.02)),
          ).oneMinus();

          // Emission per substep: the example scales by 1/120 (one substep
          // of the fixed 120 Hz sim clock).
          const densityVal = uEmitDensity
            .mul(float(1 / 120))
            .mul(flicker.mul(0.85).add(0.15))
            .mul(mask);
          const tempVal = uEmitTemperature
            .mul(float(1 / 120))
            .mul(flicker.mul(0.85).add(0.15))
            .mul(mask);

          const newDensity = density.add(densityVal).toVar();
          age.assign(
            mix(age, float(0.0), densityVal.div(max(newDensity, 0.001))),
          );
          density.assign(newDensity);
          temperature.assign(
            temperature.add(tempVal).clamp(0.0 as never, 12.0 as never),
          );
        },
      );

      textureStore(
        this.dyeTexWriteNode,
        coord,
        vec4(density, temperature, age, 1.0),
      ).toWriteOnly();
    })()
      .compute(CELL_COUNT)
      .setName("fireAdvectDye");
  }

  updatePlacement() {
    this.mesh.updateWorldMatrix(true, false);
    const m = this.mesh.matrixWorld.elements;
    const right = new Vector3(m[0], m[1], m[2]);
    const up = new Vector3(m[4], m[5], m[6]);
    const fwd = new Vector3(m[8], m[9], m[10]);

    uWorldOrigin.value.set(m[12], m[13], m[14]);
    uAxisRight.value.copy(right);
    uAxisUp.value.copy(up);
    uAxisFwd.value.copy(fwd);

    uLightDirLocal.value
      .set(
        right.dot(SUN_WORLD_DIR),
        up.dot(SUN_WORLD_DIR),
        fwd.dot(SUN_WORLD_DIR),
      )
      .normalize();
  }

  tick(tMs: number) {
    if (!this.curlNoiseReady) {
      this.curlNoiseReady = true;
      this.renderer.compute(this.curlNoisePass);
    }
    this.updatePlacement();

    // Example animate(): fixed 1/120 steps with an accumulator. The catch-up
    // cap is raised from the example's 8 to 40 by default: in throttled
    // environments (CI/headless running at ~1 fps) the flame still develops
    // in seconds, while at 60 fps the cap never engages (2 steps per frame).
    // Override with ?simcap=N.
    const t = tMs * 0.001;
    if (this.lastT === null) this.lastT = t - 1 / 60;
    const maxDelta = (1 / 120) * FireSim.simCap;
    const delta = Math.min(t - this.lastT, maxDelta);
    this.lastT = t;

    this.accumulator += delta;
    const simStep = 1 / 120;
    if (this.accumulator > maxDelta) {
      this.accumulator = maxDelta;
    }

    while (this.accumulator >= simStep) {
      this.simulationTime += simStep;
      uTime.value = this.simulationTime % 1000;
      this.step();
      this.accumulator -= simStep;
    }
  }

  private step() {
    const r = this.renderer;
    r.compute(this.advectVelocityPass);
    r.compute(this.divergencePass);
    for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
      r.compute(i % 2 === 0 ? this.jacobiPassAB : this.jacobiPassBA);
    }
    r.compute(this.projectPass);
    r.compute(this.advectDyePass);

    const tmp = this.dyeTexNode.value;
    this.dyeTexNode.value = this.dyeTexWriteNode.value;
    this.dyeTexWriteNode.value = tmp;
  }

  /**
   * Debug probe: read back the dye column at the volume center — density and
   * temperature per grid-Y cell. Used to verify the sim without eyeballing
   * screenshots (window.__fireDebug.probe()).
   */
  async probeColumn(): Promise<{
    density: number[];
    temperature: number[];
    velocityY: number[];
  }> {
    const N = GRID_SIZE_Y;
    const attr = new StorageInstancedBufferAttribute(
      new Float32Array(N * 3),
      1,
    );
    const buf = storage(attr, "float", N * 3);
    const dyeTex = this.dyeTexNode.value;
    const velTex = this.velTexA;
    const pass = Fn(() => {
      const uvw = vec3(0.5, instanceIndex.toFloat().add(0.5).div(N), 0.5);
      const dye = texture3D(dyeTex, uvw, 0);
      const vel = texture3D(velTex, uvw, 0);
      buf.element(instanceIndex).assign(dye.r);
      buf.element(instanceIndex.add(N as never)).assign(dye.g);
      buf.element(instanceIndex.add((N * 2) as never)).assign(vel.y);
    })().compute(N);
    const r = this.renderer as WebGPURenderer;
    await r.computeAsync(pass);
    const ab = await r.getArrayBufferAsync(attr);
    const a = new Float32Array(ab);
    return {
      density: Array.from(a.slice(0, N)),
      temperature: Array.from(a.slice(N, N * 2)),
      velocityY: Array.from(a.slice(N * 2)),
    };
  }

  dispose() {
    for (const texture of this.textures) {
      texture.dispose();
    }
  }
}

// ---------------------------------------------------------------------------
// Volume material: the example's scattering/emissive nodes, with the
// world → sim-space mapping adapted to the geodetic local frame.
// ---------------------------------------------------------------------------

const createFireMaterial = (sim: FireSim) => {
  const material = new VolumeNodeMaterial() as VolumeNodeMaterial & {
    scatteringNode?: (params: { positionRay: any }) => unknown;
    scatteringEmissiveNode?: (params: { positionRay: any }) => unknown;
  };
  // The raymarch runs in earth-scale world units (the box is
  // RENDER_SCALE times the 12x12x24 sim frame), so steps must be dense
  // enough to still sample the plume; ?steps=N overrides.
  const stepsParam =
    typeof window !== "undefined"
      ? Number(new URLSearchParams(window.location.search).get("steps"))
      : 0;
  material.steps = stepsParam > 0 ? stepsParam : 64;
  // VolumetricLightingModel adds every scene light (Navara lights the
  // globe with ambient/hemisphere/sun) into the scattering density at
  // every march step, whiting out the box at earth-scale step sizes.
  // Give the volume its own light set containing a single zero-intensity
  // point light: the raymarch needs a non-empty lights node to run, but
  // the zero light contributes nothing — the fire is lit by its
  // scattering/emissive nodes.
  material.lights = true;
  material.lightsNode = lights([new PointLight(0xffffff, 0, 1)]);
  material.transparent = true;
  material.blending = AdditiveBlending;
  material.depthWrite = false;
  material.depthTest = true;

  material.offsetNode = fract(
    interleavedGradientNoise(screenCoordinate).add(
      float(frameId).mul(0.618033988749895),
    ),
  );

  const fireRamp = Fn(([t]: [any]) => {
    const color = vec3(0).toVar();
    color.assign(mix(vec3(0.0), uFireEndColor, smoothstep(0.05, 0.35, t)));
    color.assign(mix(color, uFireMidColor, smoothstep(0.35, 0.65, t)));
    color.assign(mix(color, uFireStartColor, smoothstep(0.65, 1.0, t)));
    return color;
  });

  const henyeyGreenstein = Fn(([cosTheta, g]: [any, any]) => {
    const g2 = g.mul(g);
    const denom = float(1.0).add(g2).sub(float(2.0).mul(g).mul(cosTheta));
    const oneMinusG2 = float(1.0).sub(g2);
    return oneMinusG2.div(denom.pow(1.5)).mul(0.079577);
  });

  // World march position → sim uvw [0..1]: subtract the earth-scale origin,
  // rotate into the local frame, undo the render scale.
  const worldToUVW = (positionRay: any) => {
    const d = positionRay.sub(uWorldOrigin);
    const localSim = vec3(
      uAxisRight.dot(d),
      uAxisUp.dot(d),
      uAxisFwd.dot(d),
    ).div(uSimScale);
    return localSim.div(uSimSize).add(0.5);
  };

  const getVolumeSample = (positionRay: any) => {
    const uvw = worldToUVW(positionRay).toVar();

    // Domain warping via the velocity field (example option A).
    // Domain warping via the velocity field (the example's optional nicety).
    // Disabled by default: the solver's f16 velocity field intermittently
    // carries values that poison the warp (and with it the whole volume);
    // opt back in with ?warp.
    const useWarp =
      typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).has("warp");
    const noiseDistortion = useWarp
      ? texture3D(sim.velTexA, uvw, 0).xyz.div(uSimSize).mul(0.15)
      : vec3(0.0);
    const distortedUVW = uvw
      .add(noiseDistortion)
      .clamp(0.0 as never, 1.0 as never)
      .toVar();

    const sample = sim.dyeTexNode.sample(distortedUVW).level(0 as never);

    const density = sample.r.toVar();
    const age = sample.b;
    const temperature = sample.g;

    // High-frequency detail noise — trig pseudo-noise (the curl-noise addon
    // snoise misbehaves in this setup; NaNs zero the emissive contribution).
    const detailPos = localSimOf(positionRay)
      .mul(5.5)
      .add(vec3(0, age.mul(0.8).negate(), 0));
    const detail = detailPos.x
      .mul(1.3)
      .add(detailPos.y.mul(1.7))
      .add(detailPos.z.mul(0.9))
      .sin();
    density.mulAssign(detail.mul(0.35).add(0.85));

    const edge = min(distortedUVW, vec3(1).sub(distortedUVW));
    density.mulAssign(smoothstep(0.0, 0.06, min(edge.x, min(edge.y, edge.z))));

    return { density, temperature, age, distortedUVW };
  };

  // Local sim-space position (metres, example units) for noise fields.
  const localSimOf = (positionRay: any) =>
    vec3(
      uAxisRight.dot(positionRay.sub(uWorldOrigin)),
      uAxisUp.dot(positionRay.sub(uWorldOrigin)),
      uAxisFwd.dot(positionRay.sub(uWorldOrigin)),
    ).div(uSimScale);

  material.scatteringNode = Fn(({ positionRay }: { positionRay: any }) => {
    const { density } = getVolumeSample(positionRay);

    // Key-light self-shadowing (example: 2 shadow-march steps towards the
    // key light, in sim units).
    const lightDir = uKeyLightPosSim.sub(localSimOf(positionRay)).normalize();
    const shadowStepSize = 0.35;
    const shadowDensitySum = float(0.0).toVar();

    for (let i = 0; i < 2; i++) {
      const stepDist = float(i + 0.5).mul(shadowStepSize);
      const shadowSim = localSimOf(positionRay).add(lightDir.mul(stepDist));
      const shadowUVW = shadowSim.div(uSimSize).add(0.5);

      const shadowEdge = min(shadowUVW, vec3(1).sub(shadowUVW));
      const shadowFade = smoothstep(
        0.0,
        0.06,
        min(shadowEdge.x, min(shadowEdge.y, shadowEdge.z)),
      );

      shadowDensitySum.addAssign(
        texture3D(sim.dyeTexA, shadowUVW, 0).r.mul(shadowFade),
      );
    }

    const tau = shadowDensitySum.mul(shadowStepSize).mul(uShadowAbsorption);
    const beer = tau.negate().exp();

    const multiScatter = tau.mul(0.25).negate().exp().mul(0.5);
    const baseTransmittance = mix(
      beer,
      beer.add(multiScatter),
      uMultiScattering,
    );

    const powder = float(1.0).sub(tau.mul(2.0).negate().exp());
    const finalTransmittance = mix(
      baseTransmittance,
      baseTransmittance.mul(powder),
      uPowderStrength,
    );

    const lightTransmittance = finalTransmittance
      .add(uShadowAmbient)
      .clamp(0.0 as never, 1.0 as never);

    const viewDirLocal = localDirOf(positionRay);
    const cosTheta = viewDirLocal
      .dot(lightDir)
      .clamp(-1.0 as never, 1.0 as never);
    const phase = henyeyGreenstein(cosTheta, uAsymmetry);

    return (
      vec3(density)
        .mul(lightTransmittance)
        .mul(phase.mul(12.56637))
        // Scattering is a per-unit-length coefficient; the march measures
        // distance in earth-scale world units while the field is authored in
        // sim units, so divide by the world-per-sim scale to preserve the
        // plume's optical depth.
        .div(uSimScale)
    );
  });

  // Local-frame view direction.
  const localDirOf = (positionRay: any) => {
    const viewDirWorld = cameraPosition.sub(positionRay).normalize();
    return vec3(
      uAxisRight.dot(viewDirWorld),
      uAxisUp.dot(viewDirWorld),
      uAxisFwd.dot(viewDirWorld),
    ).normalize();
  };

  material.scatteringEmissiveNode = Fn(
    ({ positionRay }: { positionRay: any }) => {
      const { density, temperature } = getVolumeSample(positionRay);

      const firePower = float(6.0).sub(uFireGlowSpread);
      const tNorm = temperature.div(uFireTempNorm);
      const fire = fireRamp(tNorm.clamp(0 as never, 1 as never))
        .mul(tNorm.pow(firePower))
        .mul(uFireIntensity);

      // The example's spotlight distance attenuation, evaluated in sim units
      // (positionRay is earth-scale; localSimOf maps it into the 12×12×24 m
      // simulation frame). Without this the emission is ~100x under-powered
      // against the white smoke scattering.
      const distance = localSimOf(positionRay).sub(uKeyLightPosSim).length();
      const attenuation = float(400.0).div(distance.pow(2.0));

      return fire.mul(density.add(0.15)).mul(attenuation).div(uSimScale);
    },
  );

  // Debug channel probe: ?viz=temp renders the raw temperature field as
  // grayscale emission (no scattering), ?viz=density the raw density field.
  const vizMode =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("viz")
      : null;
  if (vizMode === "temp" || vizMode === "density") {
    material.scatteringNode = Fn(() => vec3(0.0));
    material.scatteringEmissiveNode = Fn(
      ({ positionRay }: { positionRay: any }) => {
        const { density, temperature } = getVolumeSample(positionRay);
        const v = vizMode === "temp" ? temperature.div(12.0) : density;
        return vec3(v, v, v);
      },
    );
  } else if (vizMode === "scatter" || vizMode === "fire") {
    // Isolate terms: ?viz=scatter keeps only smoke scattering, ?viz=fire
    // keeps only fire emission (both as emissive output).
    const keep = vizMode === "scatter" ? "s" : "e";
    const sNode = material.scatteringNode;
    const eNode = material.scatteringEmissiveNode;
    const keptNode = keep === "s" ? sNode : eNode;
    if (keptNode) {
      material.scatteringNode = Fn(() => vec3(0.0));
      material.scatteringEmissiveNode = Fn(
        ({ positionRay }: { positionRay: any }) =>
          keptNode({ positionRay }) as never,
      );
    }
  }

  // Debug: ?fboost=N multiplies fire emission to probe the temperature field.
  const dbgParams =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search)
      : null;
  const fboost = Number(dbgParams?.get("fboost"));
  if (fboost > 0) {
    uFireIntensity.value *= fboost;
  }
  // ?emitD=N / ?diss=N / ?cool=N override smoke emission/dissipation/cooling
  const emitD = Number(dbgParams?.get("emitD"));
  if (emitD > 0) uEmitDensity.value = emitD;
  const emitT = Number(dbgParams?.get("emitT"));
  if (emitT > 0) uEmitTemperature.value = emitT;
  const diss = Number(dbgParams?.get("diss"));
  if (diss > 0) uDissipation.value = diss;
  const cool = Number(dbgParams?.get("cool"));
  if (cool > 0) uCooling.value = cool;
  // ?samb=N / ?sabs=N override shadow ambient / absorption (smoke darkness)
  const samb = Number(dbgParams?.get("samb"));
  if (samb >= 0 && dbgParams?.has("samb")) uShadowAmbient.value = samb;
  const sabs = Number(dbgParams?.get("sabs"));
  if (sabs > 0) uShadowAbsorption.value = sabs;
  // ?tnorm=N overrides the fire-ramp temperature normalization
  const tnorm = Number(dbgParams?.get("tnorm"));
  if (tnorm > 0) uFireTempNorm.value = tnorm;

  return material;
};

// ---------------------------------------------------------------------------
// MeshDesc
// ---------------------------------------------------------------------------

type VolumeFireDescription = {
  volumeFire?: {
    steps?: number;
    emitDensity?: number;
    emitTemperature?: number;
    emitRadius?: number;
    fireIntensity?: number;
  };
};

export type VolumeFireConfig = MeshConfig & VolumeFireDescription;

export type VolumeFireUpdate = Pick<MeshConfig, "position" | "visible"> &
  VolumeFireDescription;

export type FireSimHandles = {
  tick: (tMs: number) => void;
  dispose: () => void;
};

export class VolumeFireMeshDesc extends MeshDesc<
  VolumeFireConfig,
  VolumeFireUpdate,
  Mesh
> {
  private config: VolumeFireConfig;

  constructor(view: ThreeView, ctx: ViewContext, config: VolumeFireConfig) {
    super(view, ctx, config);
    this.config = config;
  }

  protected override getPassKey(): PassKey {
    return "transparent";
  }

  createMesh(): Mesh {
    const cfg = this.config.volumeFire ?? {};

    const mesh = new Mesh(
      new BoxGeometry(RENDER_X, RENDER_Y, RENDER_Z),
      new MeshBasicNodeMaterial(),
    );

    const renderer = this.ctx.getRenderer() as unknown as Pick<
      WebGPURenderer,
      "compute"
    >;
    const sim = new FireSim(renderer, mesh);
    const material = createFireMaterial(sim);

    if (cfg.steps !== undefined) {
      material.steps = cfg.steps;
    }
    if (cfg.emitDensity !== undefined) {
      uEmitDensity.value = cfg.emitDensity;
    }
    if (cfg.emitTemperature !== undefined) {
      uEmitTemperature.value = cfg.emitTemperature;
    }
    if (cfg.emitRadius !== undefined) {
      uEmitRadius.value = cfg.emitRadius;
    }
    if (cfg.fireIntensity !== undefined) {
      uFireIntensity.value = cfg.fireIntensity;
    }

    mesh.material = material as unknown as typeof mesh.material;
    // The volume box never casts (its raymarch material has no depth output
    // that would make sense in a shadow pass).
    mesh.castShadow = false;

    // The smoke's key light: decay-0 point light (the volume lighting
    // model's direct() path skips lights without a finite distance).
    const light = new PointLight(0xffffff, 2.5, 0, 0);
    light.position.set(0, RENDER_Y * 0.3, 0);
    mesh.add(light);

    (mesh.userData as { sim?: FireSimHandles }).sim = {
      tick: (t) => sim.tick(t),
      dispose: () => sim.dispose(),
    };

    if (typeof window !== "undefined") {
      (window as unknown as Record<string, unknown>).__fireDebug = {
        mesh,
        sim,
        scenes: this.ctx.scenes,
        renderer,
        probe: () => sim.probeColumn(),
      };
    }

    return mesh;
  }

  protected disposeMesh(): void {
    (
      this._instance?.userData as { sim?: FireSimHandles } | undefined
    )?.sim?.dispose();
    this._instance?.geometry.dispose();
    this._instance = undefined;
  }
}

// A shadow-casting pillar to demonstrate the WebGPU path's native
// directional-light shadow map on the terrain.
export class PillarMeshDesc extends MeshDesc<MeshConfig, MeshConfig, Mesh> {
  createMesh(): Mesh {
    const mesh = new Mesh(
      new BoxGeometry(24, 60, 24),
      new MeshLambertNodeMaterial({ color: 0x4a4f57 }),
    );
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  protected disposeMesh(): void {
    this._instance?.geometry.dispose();
    this._instance = undefined;
  }
}

export type PillarConfig = MeshConfig & {
  pillar?: {
    color?: number;
  };
};

export type CustomDescriptions = {
  mesh: VolumeFireConfig | PillarConfig;
};

export const run = async (
  view: ThreeView<CustomDescriptions>,
  renderer: WebGPURenderer,
) => {
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  await view.init();

  view.registerMesh("volumeFire", VolumeFireMeshDesc);
  view.registerMesh("pillar", PillarMeshDesc);

  view.setCamera({
    lng: 139.75711454748298 + 0.0015,
    lat: 35.67564356091717,
    height: 120,
    heading: 270,
    pitch: -10,
    roll: 0,
  });

  const enableTiles = !new URLSearchParams(window.location.search).has(
    "notiles",
  );

  if (enableTiles) {
    const seamlessphoto = view.addSource({
      type: "raster-tile",
      url: TILE_DATASETS.gsiSeamlessphoto.url,
      maxZoom: 23,
    });
    view.addLayer({
      type: "raster",
      source: seamlessphoto,
    });

    const gsiDem = view.addSource({
      type: "raster-dem",
      url: TERRAIN_DATASETS.gsi.url,
      maxZoom: 15,
      minZoom: 5,
      elevationDecoder: JAPAN_GSI_ELEVATION_DECODER(),
    });
    view.addLayer({
      type: "terrain",
      source: gsiDem,
      terrain: {
        receiveShadow: true,
      },
    });

    // DEM slot for the WebGPU hillshade normal (and hillshade raster on the
    // WebGL path).
    view.addLayer({
      type: "raster",
      source: gsiDem,
      hillshade: {},
    });
  }

  // Default to Tokyo 17:30 (08:30 UTC) dusk: low warm light keeps terrain
  // readable while the additive volume flame still pops against the darker
  // sky. ?hour=N (UTC) overrides — hour=3 is Tokyo noon for the
  // sky/light/shadow demo, hour=11 is Tokyo 20:00 night.
  const hourStr = new URLSearchParams(window.location.search).get("hour");
  const hourParam = hourStr === null ? NaN : Number(hourStr);
  const hour = Number.isFinite(hourParam) && hourParam >= 0 ? hourParam : 8.5;
  const noon = new Date();
  noon.setUTCHours(Math.floor(hour), Math.round((hour % 1) * 60), 0, 0);
  view.atmosphere.date = noon;

  view.addMesh<PillarMeshDesc>({
    pillar: {},
    geodetic: {
      // ~250 m east of the fire: outside the 240 m-deep volume box so the
      // pillar occludes neither the flame core nor the smoke — it only casts
      // its directional-light shadow onto the terrain.
      lng: 139.75711454748298 + 0.0028,
      lat: 35.67564356091717,
      height: enableTiles ? 2 : 8,
      heightReference: enableTiles ? "terrain" : undefined,
    },
    position: { x: 0, y: 30, z: 0 },
  });

  const noFire = new URLSearchParams(window.location.search).has("nofire");
  const handle = noFire
    ? null
    : view.addMesh<VolumeFireMeshDesc>({
        volumeFire: {},
        geodetic: {
          lng: 139.75711454748298,
          lat: 35.67564356091717,
          height: enableTiles ? 4 : 10,
          heightReference: enableTiles ? "terrain" : undefined,
        },
        position: { x: 0, y: RENDER_Y / 2, z: 0 },
      });

  const raw = handle?.ref.raw;
  const sim = (raw?.userData as { sim?: FireSimHandles } | undefined)?.sim;
  if (sim && typeof window !== "undefined") {
    (
      window as unknown as { __fireDebug: Record<string, unknown> }
    ).__fireDebug.view = view;
    view.on("preUpdate", (t: number) => sim.tick(t));
  }
};
