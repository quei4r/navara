import {
  AmbientLight,
  Color,
  DirectionalLight,
  Object3D,
  Vector3,
  type Camera,
  type Scene,
  type WebGLRenderer,
} from "three";

import type { Atmosphere } from "../atmosphere";

const EARTH_RADIUS = 6378137;

// Sky tint ramp by sun elevation (dot of sun dir with local up).
const NIGHT_SKY = new Color(0x02040a);
const TWILIGHT_SKY = new Color(0x8a4a2a);
const DAY_SKY = new Color(0x87b0e0);

/**
 * WebGPU-path stand-in for the (WebGL/postprocessing-based) atmosphere and
 * CSM subsystems: a sun-elevation-driven sky background colour, an ambient
 * + directional sun light pair driven by {@link Atmosphere}'s ephemeris,
 * and native shadow-map support with a camera-following ortho frustum.
 *
 * SkyMesh is not usable here: its scattering assumes a Y-up world, which
 * ECEF globe coordinates violate.
 *
 * @experimental WebGPU forward path only.
 */
export class WebGPUEnvironment {
  readonly sun: DirectionalLight;

  private readonly ambient: AmbientLight;
  private readonly target = new Object3D();
  private readonly lightDir = new Vector3(0, 1, 0);
  private readonly localUp = new Vector3(0, 1, 0);
  private readonly groundPos = new Vector3();
  private readonly skyColor = new Color();

  constructor(
    private readonly renderer: WebGLRenderer,
    private readonly atmosphere: Atmosphere,
    _scene: Scene,
    lights: Object3D,
    debugFlags?: { shadowOff?: boolean },
  ) {
    const shadowOff = debugFlags?.shadowOff ?? false;
    this.ambient = new AmbientLight(0xdfe8f5, 0.45);
    this.sun = new DirectionalLight(0xffffff, 2.2);
    this.sun.castShadow = !shadowOff;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.00015;
    this.sun.shadow.normalBias = 2;
    const cam = this.sun.shadow.camera;
    cam.near = 1;
    cam.far = 80000;
    this.sun.target = this.target;
    lights.add(this.ambient);
    lights.add(this.sun);
    lights.add(this.target);

    renderer.shadowMap.enabled = !shadowOff;
    // Per-frame manual update, mirroring the WebGL path's CustomRenderPass.
    renderer.shadowMap.autoUpdate = false;
  }

  /** Refresh sun direction, sky colour and the shadow frustum. */
  update(camera: Camera) {
    const p = camera.position;

    this.lightDir.copy(this.atmosphere.sunDirection).normalize();
    this.localUp.copy(p).normalize();

    // Sky colour and light intensities follow the sun's elevation.
    const elevation = this.lightDir.dot(this.localUp);
    if (elevation < -0.08) {
      this.skyColor.copy(NIGHT_SKY);
    } else if (elevation < 0.12) {
      const t = (elevation + 0.08) / 0.2;
      this.skyColor.copy(TWILIGHT_SKY).lerp(DAY_SKY, t);
    } else {
      this.skyColor.copy(DAY_SKY);
    }
    // Applied via the renderer's clear colour: scene.background would be
    // skipped because autoClearColor is disabled on this path.
    this.renderer.setClearColor(this.skyColor);

    const dayFactor = Math.min(Math.max((elevation + 0.05) / 0.3, 0), 1);
    this.sun.intensity = 2.2 * dayFactor;
    this.sun.color.setRGB(
      1,
      0.96 - 0.25 * (1 - dayFactor),
      0.9 - 0.5 * (1 - dayFactor),
    );
    this.ambient.intensity = 0.15 + 0.3 * dayFactor;

    // Ground point under the camera (spherical approximation).
    this.groundPos.copy(this.localUp).multiplyScalar(EARTH_RADIUS);

    this.sun.position
      .copy(this.groundPos)
      .addScaledVector(this.lightDir, 30000);
    this.target.position.copy(this.groundPos);
    this.target.updateMatrixWorld();

    // Fit the ortho frustum to the view: ~1.5× camera altitude, clamped.
    const altitude = p.distanceTo(this.groundPos);
    const extent = Math.min(Math.max(altitude * 1.5, 200), 30000);
    const cam = this.sun.shadow.camera;
    cam.left = -extent;
    cam.right = extent;
    cam.top = extent;
    cam.bottom = -extent;
    cam.updateProjectionMatrix();
  }
}
