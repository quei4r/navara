---
name: navara-maplibre-render-review
description: >
  Re-run and review the MapLibre render tests (web/navara_maplibre_style/integration-test)
  that compare MapLibreStylePlugin output with maplibre-gl-js expected images.
  Use after changing the MapLibre Style plugin or rendering code it relies on,
  when asked to check MapLibre style compatibility, or when updating ISSUES.md.
---

# Reviewing MapLibre render tests

How to run, where the fixtures come from (the test setup checks out
maplibre-gl-js into `maplibre-gl-js/`), what the files mean, and how
unsupported cases are skipped:
[web/navara_maplibre_style/integration-test/README.md](../../web/navara_maplibre_style/integration-test/README.md).
Known differences:
[ISSUES.md](../../web/navara_maplibre_style/integration-test/ISSUES.md) — read
it first so you do not re-report an open issue.

## Flow: fix a feature, re-run only what uses it

1. **Rebuild only what the test page loads from dist.** The plugin is served
   from `web/navara_maplibre_style/src`, so plugin edits need no build. A change
   to `web/navara_three` (or other web packages) needs `pnpm build`; a Rust
   change needs `cargo make build`.
2. **Run the affected cases** from the repo root. Each test is tagged with the
   layer types, paint/layout properties, source types, and expression
   operators its style uses:
   - `pnpm test:maplibre -g @circle-radius`
   - `pnpm test:maplibre -g "@fill-opacity|@line-opacity"`
   - `pnpm test:maplibre -g projection/globe` (by case id)
   - Prefix `RENDER_INCLUDE_UNSUPPORTED=1` to also render cases skipped as
     unsupported, e.g. before marking a feature supported in supported.json.

   Preview the selection with
   `pnpm -C web/navara_maplibre_style/integration-test exec playwright test --list -g …`.
   `--list` skips the fixture checkout, so it lists the current checkout and
   fails before the first run; run the suite once after a fresh clone or a
   `maplibre.lock.json` bump.
   Run everything only for cross-cutting changes (camera, background,
   draping, color handling).

3. **Review the updated cases.** The report opens automatically and lists the
   cases this run updated first. When reviewing yourself:
   1. Read `result.json` in the case directory
      (`maplibre-gl-js/test/integration/render/tests/<id>/`). `expected` names the MapLibre
      image, and `console` holds failed requests (`[http 404] …`) and plugin
      warnings or errors.
   2. Read `actual.png` (Navara) and that expected image, and compare them.
   3. Use `diffRatio` only to prioritise. Globe rendering keeps it high even
      when the output is right.
4. **Record the outcome in ISSUES.md**, following its header, and tell the
   user which cases improved, regressed, or stayed the same. Check every open
   entry whose cases this run covered; delete the ones that now match without
   asking.

## Judging a case

| Acceptable (not a bug)                                     | Bug                                                   |
| ---------------------------------------------------------- | ----------------------------------------------------- |
| Globe curvature: bowed lines, the globe limb visible at z0 | Missing layers or features                            |
| Soft edges of draped features in small canvases            | Wrong color, opacity, width, or radius                |
| Text and shaping differences                               | Wrong stacking order                                  |
|                                                            | Position offsets beyond the globe/Mercator difference |
|                                                            | Artifacts                                             |
|                                                            | A style that should render but errors                 |

## Gotchas

- **"Invisible" is not "not drawn".**
  - Before concluding a feature is missing, render a variant: copy the
    style to `maplibre-gl-js/test/integration/render/tests/_tmp/<name>/style.json`
    and change the background,
    zoom, or color.
  - Start `pnpm -C web/navara_maplibre_style/integration-test dev`, open
    `/lib/test-page.html`, and call `harness.run("_tmp/<name>")`.
  - The suite never runs `_tmp/`. Delete the variants afterwards.
- **Blank output usually means nothing loaded.** Check the `console` field
  first. 404s for `*.terrain.png` are expected.
- **Separate harness problems from Navara problems.**
  - The harness renders the initial style only.
  - It paints a white backdrop where MapLibre leaves pixels transparent; the
    two compare as equal.
  - Navara ignores `projection` and always renders a globe.
  - It adds a pinned white `sun` and an `ambient` light, since Navara shades
    fill-extrusion and hillshade only through scene lights. Shading that is
    close but not identical to MapLibre's is not a bug; a black extrusion
    means the lights are missing.
  - If a case looks wrong for a reason unrelated to the plugin, check
    `lib/test-page.ts` before filing an issue.
- **Never edit the maplibre-gl-js checkout by hand** (outside `_tmp/`). It is
  a sparse checkout of the commit in `maplibre.lock.json`. Cases are
  classified against supported.json on every run, so a supported.json change
  takes effect on the next run.
- **Keep classification in supported.json, not in `lib/classify.ts`.** A
  property the plugin ignores without changing the result (e.g. `projection`,
  `raster-fade-duration`) is marked `⚠️ Ignored; …` there. Layer keys and
  source encoding defaults come from the style spec. Only `metadata.test`
  options and test operations, which supported.json does not cover, are
  listed in `lib/classify.ts`. When a case differs only because of a test option (a debug overlay, `fadeDuration`, …), add it to `UNSUPPORTED_TEST_OPTIONS`.
