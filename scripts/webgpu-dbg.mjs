import { chromium } from "/mnt/d/code/LACS/vendor/navara-src/web/navara_three/node_modules/playwright/index.mjs";

const url = process.argv[2];
const waitMs = Number(process.argv[3] ?? 8000);

const ctx = await chromium.launchPersistentContext("/tmp/chrome-dbg", {
  channel: "chrome",
  headless: false,
  args: [
    "--enable-features=WebGPU,Vulkan",
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--window-position=0,0",
  ],
  viewport: { width: 1280, height: 800 },
});
const page = ctx.pages()[0] ?? (await ctx.newPage());
page.on("pageerror", (e) => console.log("PAGEERROR:", String(e).split("\n")[0].slice(0, 200)));
page.on("console", (m) => {
  if (m.type() === "error") console.log("CONSOLE-ERR:", m.text().slice(0, 200));
});
await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
await page.waitForTimeout(waitMs);

const report = await page.evaluate(() => {
  const d = window.__fireDebug;
  if (!d) return { error: "no __fireDebug" };
  const mesh = d.mesh;
  const m = (mesh && mesh.matrixWorld && mesh.matrixWorld.elements) || [];
  return {
    meshExists: !!mesh,
    visible: mesh && mesh.visible,
    parent: mesh && mesh.parent && mesh.parent.type,
    parentChildren: mesh && mesh.parent && mesh.parent.children.length,
    frustumCulled: mesh && mesh.frustumCulled,
    geometryAttrs: Object.keys((mesh && mesh.geometry && mesh.geometry.attributes) || {}),
    matrixWorldSample: [m[12], m[13], m[14]].map((x) =>
      typeof x === "number" ? x.toPrecision(6) : x,
    ),
    transparentSceneChildren: d.scenes && d.scenes.transparent && d.scenes.transparent.children.length,
    globeSceneChildren: d.scenes && d.scenes.globe && d.scenes.globe.children.length,
    materialType: (mesh && mesh.material && (mesh.material.type || mesh.material.constructor.name)) || null,
    materialTransparent: mesh && mesh.material && mesh.material.transparent,
    rendererInfo: d.renderer
      ? {
          frame: d.renderer.info && d.renderer.info.render && d.renderer.info.render.frame,
          drawCalls: d.renderer.info && d.renderer.info.render && d.renderer.info.render.calls,
          backend: d.renderer.backend && d.renderer.backend.constructor.name,
        }
      : null,
    camera: d.view
      ? (() => {
          const cam = d.view.camera && d.view.camera.raw;
          const mesh2 = d.mesh;
          if (!cam || !mesh2) return null;
          const e = mesh2.matrixWorld.elements;
          const wx = e[12], wy = e[13], wz = e[14];
          const v = cam.matrixWorldInverse.elements;
          const vx = v[0]*wx + v[4]*wy + v[8]*wz + v[12];
          const vy = v[1]*wx + v[5]*wy + v[9]*wz + v[13];
          const vz = v[2]*wx + v[6]*wy + v[10]*wz + v[14];
          const p = cam.projectionMatrix.elements;
          const cx = p[0]*vx + p[4]*vy + p[8]*vz + p[12];
          const cy = p[1]*vx + p[5]*vy + p[9]*vz + p[13];
          const cw = p[3]*vx + p[7]*vy + p[11]*vz + p[15];
          const cwSafe = cw !== 0 ? cw : 1;
          const ce = cam.matrixWorld.elements;
          return {
            ndc: [cx / cwSafe, cy / cwSafe],
            viewZ: vz,
            camPos: [ce[12], ce[13], ce[14]].map((x) => x.toExponential(3)),
            near: cam.near,
            far: cam.far,
            fov: cam.fov,
          };
        })()
      : null,
    canvasDataURL: d.renderer && d.renderer.domElement ? d.renderer.domElement.toDataURL("image/png").slice(0, 100) : null,
    canvasDataURLLength: d.renderer && d.renderer.domElement ? d.renderer.domElement.toDataURL("image/png").length : 0,
    canvasSize: d.renderer && d.renderer.domElement ? [d.renderer.domElement.width, d.renderer.domElement.height] : null,
    canvasClient: d.renderer && d.renderer.domElement ? [d.renderer.domElement.clientWidth, d.renderer.domElement.clientHeight] : null,
  };
});
console.log(JSON.stringify(report, null, 1));
await ctx.close().catch(() => {});
process.exit(0);
