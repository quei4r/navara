import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { RenderTestStyle, SupportTable } from "./classify";

export const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
/** maplibre-gl-js checkout made by render/setup.ts (gitignored). */
export const UPSTREAM_DIR = path.join(ROOT, "maplibre-gl-js");
export const ASSETS_DIR = path.join(UPSTREAM_DIR, "test/integration/assets");
/** Outputs are written next to the fixtures, as in maplibre-gl-js. */
export const RENDER_TESTS_DIR = path.join(
  UPSTREAM_DIR,
  "test/integration/render/tests",
);
/** Report output. */
export const RENDER_DIR = path.join(ROOT, "render");
export const LOCK_PATH = path.join(ROOT, "maplibre.lock.json");

/** The plugin's support table; cases using anything else are skipped. */
export const SUPPORT_TABLE_PATH = path.join(
  ROOT,
  "../../../docs/src/content/docs/maplibre_style/supported.json",
);

export type Lock = { repository: string; commit: string };

export function readLock(): Lock {
  return JSON.parse(fs.readFileSync(LOCK_PATH, "utf8")) as Lock;
}

export function readSupportTable(): SupportTable {
  return JSON.parse(
    fs.readFileSync(SUPPORT_TABLE_PATH, "utf8"),
  ) as SupportTable;
}

export function readCaseStyle(id: string): RenderTestStyle {
  return JSON.parse(
    fs.readFileSync(path.join(RENDER_TESTS_DIR, id, "style.json"), "utf8"),
  ) as RenderTestStyle;
}

/** Hand-made variants for manual runs in the dev server; never run by the suite. */
const SCRATCH_PREFIX = "_tmp/";

/** Every render test in the checkout, sorted. */
export function listCaseIds(): string[] {
  if (!fs.existsSync(RENDER_TESTS_DIR)) {
    throw new Error(
      "No maplibre-gl-js fixtures; `pnpm test:maplibre` checks them out",
    );
  }
  return fs
    .readdirSync(RENDER_TESTS_DIR, { recursive: true, encoding: "utf8" })
    .filter((f) => path.basename(f) === "style.json")
    .map((f) => path.dirname(f).split(path.sep).join("/"))
    .filter((id) => !id.startsWith(SCRATCH_PREFIX))
    .sort();
}

/** Per-case `result.json` written by render.test.ts and read by the report. */
export type CaseResult = {
  id: string;
  width: number;
  height: number;
  pixelRatio: number;
  /** Mismatched pixels / total pixels against the closest expected image. */
  diffRatio?: number;
  expected?: string;
  sizeMismatch?: boolean;
  console: string[];
  error?: string;
  durationMs: number;
  /** Epoch ms; the report compares it with the run start. */
  finishedAt: number;
};
