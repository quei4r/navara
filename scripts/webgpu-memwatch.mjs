import { chromium } from "/mnt/d/code/LACS/vendor/navara-src/web/navara_three/node_modules/playwright/index.mjs";

const url = process.argv[2] ?? "http://localhost:5175/examples/webgpu-volume-fire?notiles&nosim&nomesh";
const seconds = Number(process.argv[3] ?? 25);

const ctx = await chromium.launchPersistentContext("/tmp/chrome-wgpu-navara", {
  channel: "chrome",
  headless: false,
  args: [
    "--enable-features=WebGPU,Vulkan",
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--window-position=3200,3200",
  ],
  viewport: { width: 1280, height: 800 },
});

const page = ctx.pages()[0] ?? (await ctx.newPage());
const cdp = await ctx.newCDPSession(page);
await cdp.send("Performance.enable");

let ended = "alive-at-end";
page.on("crash", () => (ended = "CRASHED"));
page.on("pageerror", (e) => (ended = "PAGEERROR: " + String(e).split("\n")[0].slice(0, 120)));

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });

for (let i = 0; i < seconds; i++) {
  try {
    await page.waitForTimeout(1000);
    const m = await cdp.send("Performance.getMetrics");
    const get = (name) => m.metrics.find((x) => x.name === name)?.value ?? -1;
    console.log(
      `s${i + 1}`,
      "JSHeap=" + (get("JSHeapUsedSize") / 1048576).toFixed(0) + "M",
      "Nodes=" + get("Nodes"),
      "Listeners=" + get("JSEventListeners"),
      "Docs=" + get("Documents"),
      "GPUmem=" + get("JSHeapTotalSize") ? "" : "",
    );
  } catch (e) {
    ended += " (metrics failed @s" + (i + 1) + ": " + String(e).slice(0, 80) + ")";
    break;
  }
  if (!ended.startsWith("alive")) break;
}
console.log("ended:", ended);
await ctx.close().catch(() => {});
process.exit(0);
