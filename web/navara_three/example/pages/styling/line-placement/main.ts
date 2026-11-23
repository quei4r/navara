import ThreeView from "@navaramap/three";

import { run } from "./run";

const view = new ThreeView({ debug: true, picking: true });

run(view);
