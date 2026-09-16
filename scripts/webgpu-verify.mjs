// Headed-Chrome WebGPU verification for the Navara webgpu-volume-fire example.
// Usage: DISPLAY=:1 node scripts/webgpu-verify.mjs [url] [out.png] [waitMs]
import { chromium } from "/mnt/d/code/LACS/vendor/navara-src/web/navara_three/node_modules/playwright/index.mjs";

const url = process.argv[2] ?? "http://localhost:5175/examples/webgpu-volume-fire";
const out = process.argv[3] ?? "/tmp/navara-shots/webgpu-fire.png";
const waitMs = Number(process.argv[4] ?? 15000);

const ctx = await chromium.launchPersistentContext("/tmp/chrome-wgpu-navara", {
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
const errors = [];
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text().slice(0, 300));
});
page.on("pageerror", (e) => errors.push(String(e).slice(0, 300)));
page.on("crash", () => errors.push("<<RENDERER CRASH>>"));

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
await page.waitForTimeout(waitMs);

let info = null;
try {
  info = await page.evaluate(() => ({
    webgpu: "gpu" in navigator,
    canvas: !!document.querySelector("#navara-canvas"),
  }));
} catch (e) {
  errors.push("evaluate failed: " + String(e).slice(0, 200));
}
try {
  await page.screenshot({ path: out });
  console.log("SHOT:", out);
} catch (e) {
  console.log("SHOT FAILED:", String(e).slice(0, 200));
}
console.log("INFO:", JSON.stringify(info));
console.log("ERRORS:", errors.length ? errors.slice(0, 20).join("\n") : "none");
await ctx.close().catch(() => {});
process.exit(0);
