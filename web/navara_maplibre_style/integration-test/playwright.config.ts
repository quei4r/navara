import { defineConfig } from "@playwright/test";

const PORT = 5180;

export default defineConfig({
  testDir: "render",
  testMatch: "render.test.ts",
  outputDir: "test-results",
  fullyParallel: true,
  workers: Number(process.env.RENDER_WORKERS ?? 4),
  timeout: 60_000,
  reporter: [["list"]],
  globalSetup: "./render/setup.ts",
  globalTeardown: "./render/report.ts",
  use: {
    baseURL: `http://localhost:${PORT}`,
    launchOptions: {
      args: ["--ignore-gpu-blocklist", "--use-gl=angle"],
    },
  },
  webServer: {
    command: `vite --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/lib/test-page.html`,
    reuseExistingServer: !process.env.CI,
  },
});
