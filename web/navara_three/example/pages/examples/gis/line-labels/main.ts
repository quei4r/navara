import ThreeView, { Color, fetchFontFamilyFromCss } from "@navaramap/three";
import { TileJsonPlugin } from "@navaramap/three-plugins";

import { initializeExample } from "../../../../helpers/initialize";

const view = new ThreeView();

const tilejson = new TileJsonPlugin();
view.addPlugin(tilejson);

await view.init();

view.addFontFamily(
  await fetchFontFamilyFromCss(
    "Arsenal",
    "https://fonts.googleapis.com/css2?family=Arsenal:wght@700",
  ),
);

view.setCamera({
  lng: -0.1281,
  lat: 51.5045,
  height: 650,
  heading: 0,
  pitch: -50,
  roll: 0,
});

const basemap = await tilejson.addSource({
  type: "raster-tile",
  url: "https://papers.reearth.land/styles/papers-light/tilejson.json",
});
view.addLayer({ type: "raster", source: basemap });

const streets = await tilejson.addSource({
  type: "vector-tile",
  url: "https://tiles.openfreemap.org/planet",
});

const labels = view.addLayer({
  type: "vector",
  source: streets,
  sourceLayers: ["transportation_name"],
  text: {
    font: "Arsenal",
    // Derive labels from line geometry, repeated every `spacing` screen pixels.
    geometryTypes: ["line"],
    placement: "line",
    spacing: 250,
    textFacing: "flat",
    maxAngle: 5,
    size: 18,
    height: 0,
    sizeInMeters: false,
    clampToGround: true,
    color: new Color().setStyle("#0091ff"),
    outlineColor: new Color().setStyle("#ffffff"),
    outlineWidth: 4,
  },
});

labels.on("featureUpdated", ({ evaluator }) => {
  evaluator.evaluate(
    ({ properties }) => {
      const name = properties?.["name"] as string | undefined;
      return { text: name ?? "", show: !!name };
    },
    { filters: ["name"] },
  );
});

initializeExample(view);
