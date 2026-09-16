// WebGPU verification via direct canvas readback (bypasses offscreen-window
// compositing skips). Usage: node webgpu-canvas-shot.mjs [url] [out.png] [waitMs]
import { chromium } from "/mnt/d/code/LACS/vendor/navara-src/web/navara_three/node_modules/playwright/index.mjs";
import { writeFileSync } from "node:fs";

const url = process.argv[2] ?? "http://localhost:5175/examples/webgpu-volume-fire";
const out = process.argv[3] ?? "/tmp/navara-shots/fire.png";
const waitMs = Number(process.argv[4] ?? 18000);

const ctx = await chromium.launchPersistentContext("/tmp/chrome-shot", {
  channel: "chrome",
  headless: false,
  args: [
    "--enable-features=WebGPU,Vulkan",
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--window-position=0,0",
    // Keep rAF unthrottled even when the window is occluded on the desktop
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
  ],
  viewport: { width: 1280, height: 800 },
});

const page = ctx.pages()[0] ?? (await ctx.newPage());
const errors = [];
page.on("pageerror", (e) => errors.push(String(e).split("\n")[0].slice(0, 200)));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text().slice(0, 200));
});
await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
await page.waitForTimeout(waitMs);

const b64 = await page.evaluate(() => {
  const d = window.__fireDebug;
  const canvas =
    d?.renderer?.domElement ?? document.querySelector("canvas");
  return canvas?.toDataURL("image/png").split(",")[1] ?? null;
});
// Canvas readback can silently return a stale/blank buffer on pages where
// the presented frame is gone; a CDP window capture is the fallback.
const BLANK_SUSPICION_BYTES = 40000;
if (b64 && Buffer.from(b64, "base64").length >= BLANK_SUSPICION_BYTES) {
  writeFileSync(out, Buffer.from(b64, "base64"));
  console.log("CANVAS-SHOT:", out, Buffer.from(b64, "base64").length, "bytes");
} else {
  await page.screenshot({ path: out });
  console.log("WINDOW-SHOT:", out, b64 ? "(canvas readback looked blank)" : "(no canvas)");
}
console.log("ERRORS:", errors.length ? errors.slice(0, 8).join(" | ") : "none");
await ctx.close().catch(() => {});
process.exit(0);
