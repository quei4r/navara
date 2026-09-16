import { chromium } from "/mnt/d/code/LACS/vendor/navara-src/web/navara_three/node_modules/playwright/index.mjs";

const url = process.argv[2] ?? "http://localhost:5175/examples/webgpu-volume-fire?notiles&kdebug";
const seconds = Number(process.argv[3] ?? 30);
const out = process.argv[4] ?? "/tmp/navara-shots/webgpu-fire-long.png";

const ctx = await chromium.launchPersistentContext("/tmp/chrome-wgpu-navara", {
  channel: "chrome",
  headless: false,
  dumpio: true,
  args: [
    "--enable-logging=stderr",
    "--enable-features=WebGPU,Vulkan",
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--window-position=3200,3200",
  ],
  viewport: { width: 1280, height: 800 },
});

const page = ctx.pages()[0] ?? (await ctx.newPage());
let submits = 0;
let ended = "alive-at-end";
const pageErrors = [];
page.on("console", (m) => {
  if (m.text().includes("[sim] submit")) submits++;
  if (m.text().includes("[wgpu]")) console.log(m.text().slice(0, 600));
});
page.on("pageerror", (e) => {
  pageErrors.push(String(e).split("\n")[0].slice(0, 150));
  ended = "PAGEERROR (after " + submits + " submits): " + pageErrors[0];
});
page.on("crash", () => {
  ended = "CRASHED after " + submits + " submits";
});

await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
for (let i = 0; i < seconds; i++) {
  try {
    await page.waitForTimeout(1000);
  } catch (e) {
    ended += " (waitForTimeout failed @second " + (i + 1) + ")";
    break;
  }
  if (!ended.startsWith("alive")) break;
}
console.log("submits:", submits, "| ended:", ended);
try {
  await page.screenshot({ path: out });
  console.log("SHOT OK:", out);
} catch (e) {
  console.log("SHOT FAIL");
}
await ctx.close().catch(() => {});
process.exit(0);
