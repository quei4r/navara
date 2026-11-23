import { execFileSync } from "node:child_process";

import { chromium } from "@playwright/test";

import { UPSTREAM_DIR, readLock } from "../lib/config";

const SPARSE_PATTERNS = [
  "/LICENSE.txt",
  "/test/integration/assets/",
  "/test/integration/render/tests/",
];

function git(...args: string[]): string {
  return execFileSync("git", ["-C", UPSTREAM_DIR, ...args], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
}

/** Undefined before the first checkout. */
function headCommit(): string | undefined {
  try {
    return git("rev-parse", "--verify", "--quiet", "HEAD").trim();
  } catch {
    return undefined;
  }
}

/**
 * Checks out maplibre-gl-js `test/integration/{assets,render/tests}` at the
 * commit in maplibre.lock.json into maplibre-gl-js/ (gitignored: fetched, not
 * redistributed). The fetch is shallow, blobless, and sparse, and runs only
 * when the checkout is elsewhere; a failed fetch leaves the previous one.
 */
function syncFixtures() {
  const { repository, commit } = readLock();
  if (!/^[0-9a-f]{40}$/.test(commit)) {
    throw new Error(`maplibre.lock.json: commit must be a full SHA: ${commit}`);
  }
  // Also creates the directory, so `-C` never falls through to this repository.
  execFileSync("git", ["init", "--quiet", UPSTREAM_DIR]);
  git("config", "remote.origin.url", `https://github.com/${repository}.git`);
  // Before the commit check, so pattern edits reach existing checkouts too.
  git("sparse-checkout", "set", "--no-cone", ...SPARSE_PATTERNS);
  if (headCommit() === commit) return;

  console.log(`Fetching ${repository}@${commit} render tests…`);
  git("fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", commit);
  git("checkout", "--quiet", "--detach", commit);
}

/**
 * Fails the run once when Chromium cannot start, instead of in every test.
 * Each Playwright version needs its own browser build, so a version bump
 * leaves it missing until it is installed.
 */
async function checkBrowser() {
  try {
    await (await chromium.launch()).close();
  } catch (e) {
    throw new Error(
      "Cannot launch Chromium. Install the browser for this Playwright version:\n" +
        "  pnpm -C web/navara_maplibre_style/integration-test exec playwright install chromium",
      { cause: e },
    );
  }
}

/**
 * Checks the browser, syncs the fixtures, and marks the run's start for the
 * report, which lists the cases updated by this run first. Workers inherit
 * `process.env` from the runner.
 */
export default async function setup() {
  await checkBrowser();
  syncFixtures();
  process.env.RENDER_RUN_STARTED = String(Date.now());
}
