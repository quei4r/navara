/**
 * Aggregates the result.json of every render test into render/results.json
 * and render/results.html, so a filtered run (`-g <pattern>`) updates only its
 * rows; those rows are listed first and marked. Cases supported.json does not
 * cover are listed with the reasons, and the rest without a result as not run.
 * Then opens the HTML in the default browser (skipped when `CI` is set). Runs
 * as Playwright's globalTeardown, or standalone via `pnpm report`.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { classifyStyle } from "../lib/classify";
import {
  RENDER_DIR,
  RENDER_TESTS_DIR,
  listCaseIds,
  readCaseStyle,
  readSupportTable,
  type CaseResult,
} from "../lib/config";

type ReportRow = CaseResult & { unsupported?: string[] };
type Unsupported = { id: string; reasons: string[] };

/** Image links from render/results.html into the checkout. */
const TESTS_URL = path
  .relative(RENDER_DIR, RENDER_TESTS_DIR)
  .split(path.sep)
  .join("/");

/** Skipped when setup.ts failed, since no test ran. */
export default function teardown() {
  if (process.env.RENDER_RUN_STARTED) writeReport();
}

function writeReport() {
  const support = readSupportTable();
  const results: ReportRow[] = [];
  const unsupported: Unsupported[] = [];
  const notRun: string[] = [];
  for (const id of listCaseIds()) {
    const { reasons } = classifyStyle(readCaseStyle(id), support);
    const file = path.join(RENDER_TESTS_DIR, id, "result.json");
    if (fs.existsSync(file)) {
      const result = JSON.parse(fs.readFileSync(file, "utf8")) as ReportRow;
      if (reasons.length > 0) result.unsupported = reasons;
      results.push(result);
    } else if (reasons.length > 0) {
      unsupported.push({ id, reasons });
    } else {
      notRun.push(id);
    }
  }
  // Set by setup.ts; absent when run standalone, so nothing is "updated".
  const runStarted = Number(process.env.RENDER_RUN_STARTED ?? Infinity);
  const updated = new Set(
    results.filter((r) => r.finishedAt >= runStarted).map((r) => r.id),
  );
  // Updated rows, then errors, then by id so a category reads together.
  results.sort(
    (a, b) =>
      Number(updated.has(b.id)) - Number(updated.has(a.id)) ||
      Number(!!b.error) - Number(!!a.error) ||
      a.id.localeCompare(b.id),
  );
  fs.writeFileSync(
    path.join(RENDER_DIR, "results.json"),
    `${JSON.stringify(results, null, 2)}\n`,
  );
  const htmlPath = path.join(RENDER_DIR, "results.html");
  fs.writeFileSync(htmlPath, renderHtml(results, updated, unsupported, notRun));
  console.log(
    `Render report: ${htmlPath} (${updated.size} updated, ${results.length} rendered, ${unsupported.length} unsupported, ${notRun.length} not run)`,
  );
  if (!process.env.CI) openInBrowser(htmlPath);
}

function openInBrowser(file: string) {
  const [command, ...args] =
    process.platform === "darwin"
      ? ["open", file]
      : process.platform === "win32"
        ? ["cmd", "/c", "start", "", file]
        : ["xdg-open", file];
  // Detached so the test run exits without waiting on the browser.
  spawn(command, args, { detached: true, stdio: "ignore" })
    .on("error", (err) =>
      console.warn(`Could not open ${file}: ${err.message}`),
    )
    .unref();
}

function encodeCaseId(id: string) {
  return id.split("/").map(encodeURIComponent).join("/");
}

/** Small fixtures are upscaled (pixelated) so individual pixels stay readable. */
const MIN_DISPLAY_SIZE = 256;

/** Navara left, MapLibre's closest expected image right, at the same size. */
function renderPair(r: CaseResult): string {
  const scale = Math.max(
    1,
    Math.floor(MIN_DISPLAY_SIZE / Math.max(r.width, r.height)),
  );
  const size = `width="${r.width * scale}" height="${r.height * scale}"`;
  const dir = `${TESTS_URL}/${encodeCaseId(r.id)}`;
  const figure = (label: string, file: string) =>
    `<figure><figcaption>${label}</figcaption><img src="${dir}/${encodeURIComponent(file)}" ${size} alt="${label}: ${escapeXml(r.id)}" loading="lazy"></figure>`;
  return `<div class="pair">${figure("Navara", "actual.png")}${figure("MapLibre", r.expected ?? "expected.png")}</div>`;
}

/** Collapsed: most cases are unsupported, so reasons are summarized first. */
function renderUnsupported(unsupported: Unsupported[]): string {
  if (unsupported.length === 0) return "";
  const counts = new Map<string, number>();
  for (const { reasons } of unsupported) {
    reasons.forEach((r) => counts.set(r, (counts.get(r) ?? 0) + 1));
  }
  const summary = [...counts]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([r, n]) => `<tr><td>${n}</td><td>${escapeXml(r)}</td></tr>`)
    .join("\n");
  const cases = unsupported
    .map(
      ({ id, reasons }) =>
        `<tr><td><code>${escapeXml(id)}</code></td><td>${reasons.map(escapeXml).join("<br>")}</td></tr>`,
    )
    .join("\n");
  return `<details><summary>${unsupported.length} unsupported cases (skipped)</summary>
<h3>Reasons</h3>
<table class="reasons">${summary}</table>
<h3>Cases</h3>
<table class="unsupported">${cases}</table>
</details>`;
}

function renderHtml(
  results: ReportRow[],
  updated: Set<string>,
  unsupported: Unsupported[],
  notRun: string[],
): string {
  const rows = results
    .map((r) => {
      const diff =
        r.diffRatio === undefined ? "–" : `${(r.diffRatio * 100).toFixed(2)}%`;
      const notes = [
        // The stack trace stays in result.json.
        r.error && `<b>error:</b> ${escapeXml(r.error.split("\n")[0])}`,
        r.sizeMismatch && "size mismatch (resized)",
        r.unsupported &&
          `<b>unsupported:</b> ${r.unsupported.map(escapeXml).join(", ")}`,
      ]
        .filter(Boolean)
        .join("<br>");
      const image = r.error ? "" : renderPair(r);
      const classes = [r.error && "error", updated.has(r.id) && "updated"];
      return `<tr class="${classes.filter(Boolean).join(" ")}">
<td><code>${escapeXml(r.id)}</code><br><span class="meta">diff ${r.error ? diff : `<a href="${TESTS_URL}/${encodeCaseId(r.id)}/diff.png">${diff}</a>`}</span><div class="notes">${notes}</div></td>
<td>${image}</td></tr>`;
    })
    .concat(
      notRun.map(
        (id) =>
          `<tr class="not-run"><td><code>${escapeXml(id)}</code><br><span class="meta">not run</span></td><td></td></tr>`,
      ),
    )
    .join("\n");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MapLibre Style Render Tests</title>
<style>
:root { color-scheme: light dark; --bg: #fff; --fg: #1a1a1a; --muted: #666; --line: #ddd; --err: #c62828; --accent: #1565c0; }
@media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #eee; --muted: #aaa; --line: #333; --err: #ef5350; --accent: #64b5f6; } }
body { font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 16px; }
table { border-collapse: collapse; width: 100%; }
td { border-top: 1px solid var(--line); padding: 8px; vertical-align: top; }
td:first-child { width: 28%; word-break: break-all; }
.pair { display: flex; flex-wrap: wrap; gap: 8px; }
figure { margin: 0; }
figcaption { font-size: 12px; color: var(--muted); }
/* Unpainted pixels of MapLibre's images are transparent; match the white backdrop Navara renders on. */
img { display: block; max-width: 100%; height: auto; background: #fff; image-rendering: pixelated; }
.meta { color: var(--muted); }
.notes { font-size: 12px; color: var(--muted); margin-top: 4px; }
tr.error code { color: var(--err); }
tr.updated td:first-child { border-left: 4px solid var(--accent); }
tr.not-run code { color: var(--muted); }
details { margin-top: 24px; }
summary { cursor: pointer; font-weight: 600; }
table.reasons td:first-child { width: 4em; text-align: right; }
</style></head><body>
<h1>MapLibre Style Render Tests</h1>
<p>${updated.size ? `${updated.size} updated by this run (marked, listed first); ` : ""}${results.length} cases rendered (errors first)${notRun.length ? `, ${notRun.length} not run yet (listed last)` : ""}${unsupported.length ? `, ${unsupported.length} skipped as unsupported by supported.json (below)` : ""}. Left: Navara, right: MapLibre expected. The diff ratio (pixelmatch, as in maplibre-gl-js) is a hint for review, not a pass criterion.</p>
<table>${rows}</table>
${renderUnsupported(unsupported)}
</body></html>
`;
}

const XML_ENTITIES: Record<string, string> = {
  "<": "&lt;",
  ">": "&gt;",
  "&": "&amp;",
  '"': "&quot;",
};

function escapeXml(s: string): string {
  return s.replace(/[<>&"]/g, (c) => XML_ENTITIES[c] ?? c);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  writeReport();
}
