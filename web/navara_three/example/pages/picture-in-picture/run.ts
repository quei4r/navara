import ThreeView, {
  PictureInPicturePlugin,
  type PictureInPictureCameraOptions,
} from "@navaramap/three";
import {
  DefaultPlugin,
  type DefaultDescriptions,
} from "@navaramap/three-default-plugin";

export type CustomDescriptions = DefaultDescriptions;

const ESRI_IMAGERY =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

// Xiamen, same area as the other webgpu-* debug pages. Coastal plain, so the
// ground-level PiP views work on the plain ellipsoid (no DEM needed).
const HOME = { lng: 118.0894, lat: 24.4455 };

const PRESETS: { label: string; camera: PictureInPictureCameraOptions }[] = [
  {
    label: "Ground N",
    camera: { ...HOME, height: 120, heading: 0, pitch: -15 },
  },
  {
    label: "Ground E",
    camera: { ...HOME, height: 120, heading: 90, pitch: -15 },
  },
  {
    label: "Rooftop",
    camera: { ...HOME, height: 600, heading: 200, pitch: -55 },
  },
];

export const run = async (view: ThreeView<CustomDescriptions>) => {
  view.addPlugin(new DefaultPlugin());
  const pip = new PictureInPicturePlugin({ corner: "bottom-right" });
  view.addPlugin(pip);
  await view.init();

  view.attribution?.add([
    { attribution: "Esri, Maxar, Earthstar Geographics" },
  ]);

  // Noon sun.
  const noon = new Date();
  noon.setUTCHours(4, 0, 0, 0);
  view.atmosphere.date = noon;

  const imagery = view.addSource({
    type: "raster-tile",
    url: ESRI_IMAGERY,
    maxZoom: 19,
  });
  view.addLayer({ type: "raster", source: imagery });

  view.setCamera({
    ...HOME,
    height: 6000,
    heading: 0,
    pitch: -60,
    roll: 0,
  });

  pip.setCamera(PRESETS[0].camera);

  const bar = document.createElement("div");
  bar.style.cssText =
    "position:fixed;top:12px;left:12px;display:flex;gap:8px;z-index:10;";
  document.body.appendChild(bar);
  const addButton = (label: string, onClick: () => void) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.style.cssText =
      "padding:6px 12px;background:#14202bdd;color:#dfe9f5;" +
      "border:1px solid #3a4a5a;border-radius:4px;cursor:pointer;";
    button.addEventListener("click", onClick);
    bar.appendChild(button);
    return button;
  };
  for (const preset of PRESETS) {
    addButton(preset.label, () => pip.setCamera(preset.camera));
  }
  let visible = true;
  const toggle = addButton("Hide PiP", () => {
    visible = !visible;
    if (visible) pip.show();
    else pip.hide();
    toggle.textContent = visible ? "Hide PiP" : "Show PiP";
  });

  (window as unknown as Record<string, unknown>).__pipDebug = {
    view,
    pip,
    presets: PRESETS,
  };
};
