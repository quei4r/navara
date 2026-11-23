import type { Events } from "@navaramap/engine";
import { describe, expect, it, vi } from "vitest";

import { JobDispatcher, type JobHandler } from "./JobDispatcher";
import { makeEvent } from "./mock";

type Ev = { ind: number; gen: number; free: () => void };

const ev = (ind: number, gen = 1): Ev => ({ ind, gen, free: vi.fn() });

const flush = () => new Promise((r) => setTimeout(r, 0));

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (v: T) => void;
  reject: (e: unknown) => void;
};
const deferred = <T>(): Deferred<T> => {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

type Handler = JobHandler<Ev, Ev, Ev, string>;

const makeHandler = (
  start: Handler["start"],
  overrides: Partial<Handler> = {},
) => {
  const handler = {
    name: "test",
    events: {
      requested: "mesh_added",
      changed: "mesh_updated",
      removed: "mesh_removed",
    },
    start: vi.fn(start),
    change: vi.fn(),
    teardown: vi.fn(),
    failed: vi.fn(),
    ...overrides,
  } satisfies Handler;
  return handler;
};

const events = (stacks: {
  mesh_added?: Ev[];
  mesh_updated?: Ev[];
  mesh_removed?: Ev[];
}) => makeEvent(stacks as unknown as Parameters<typeof makeEvent>[0]);

describe("JobDispatcher", () => {
  it("starts every requested job in the same tick and tears down a live one on removed", async () => {
    const handler = makeHandler(async () => "product");
    const d = new JobDispatcher();
    d.register(handler);

    const a = ev(1);
    const b = ev(2);
    d.tick(events({ mesh_added: [a, b] }));
    expect(handler.start).toHaveBeenCalledTimes(2);
    expect(d.inFlight()).toBe(2);

    await flush();
    expect(d.inFlight()).toBe(0);
    expect(a.free).toHaveBeenCalledTimes(1);
    expect(b.free).toHaveBeenCalledTimes(1);

    const removed = ev(1);
    d.tick(events({ mesh_removed: [removed] }));
    expect(handler.teardown).toHaveBeenCalledWith("product", removed);
    expect(removed.free).toHaveBeenCalledTimes(1);
  });

  it("counts in-flight calls per registered kind", async () => {
    const started = deferred<string>();
    const changed = deferred<undefined>();
    const meshes = makeHandler(() => started.promise, {
      change: vi.fn(() => changed.promise),
    });
    const features = makeHandler(async () => "feature", {
      events: {
        requested: "renderable_feature_added",
        removed: "renderable_feature_removed",
      },
    });
    const d = new JobDispatcher();
    const meshKind = d.register(meshes);
    const featureKind = d.register(features);

    d.tick(
      makeEvent({
        mesh_added: [ev(1), ev(2)],
        renderable_feature_added: [ev(3)],
      } as unknown as Parameters<typeof makeEvent>[0]),
    );
    expect(meshKind.inFlight()).toBe(2);
    expect(featureKind.inFlight()).toBe(1);

    started.resolve("product");
    await flush();
    expect(meshKind.inFlight()).toBe(0);
    expect(featureKind.inFlight()).toBe(0);

    // An async change holds its job in flight again.
    d.tick(events({ mesh_updated: [ev(1)] }));
    expect(meshKind.inFlight()).toBe(1);
    changed.resolve(undefined);
    await flush();
    expect(meshKind.inFlight()).toBe(0);

    d.tick(events({ mesh_updated: [ev(2)] }));
    d.dispose();
    expect(meshKind.inFlight()).toBe(0);
  });

  it("aborts an in-flight job on removed and tears it down once start settles", async () => {
    const started = deferred<string>();
    let signal: AbortSignal | undefined;
    const handler = makeHandler((_, abort) => {
      signal = abort.signal;
      return started.promise;
    });
    const d = new JobDispatcher();
    d.register(handler);

    const added = ev(1);
    d.tick(events({ mesh_added: [added] }));
    const removed = ev(1);
    d.tick(events({ mesh_removed: [removed] }));

    expect(signal?.aborted).toBe(true);
    expect(handler.teardown).not.toHaveBeenCalled();
    expect(removed.free).not.toHaveBeenCalled();

    started.resolve("late product");
    await flush();
    expect(handler.teardown).toHaveBeenCalledTimes(1);
    expect(handler.teardown).toHaveBeenCalledWith("late product", removed);
    expect(added.free).toHaveBeenCalledTimes(1);
    expect(removed.free).toHaveBeenCalledTimes(1);
    expect(d.inFlight()).toBe(0);
  });

  it("drops an add whose remove arrives in the same tick", () => {
    const handler = makeHandler(async () => "product");
    const d = new JobDispatcher();
    d.register(handler);

    const added = ev(1);
    const removed = ev(1);
    d.tick(events({ mesh_added: [added], mesh_removed: [removed] }));

    expect(handler.start).not.toHaveBeenCalled();
    expect(handler.teardown).toHaveBeenCalledWith(undefined, removed);
    expect(added.free).toHaveBeenCalledTimes(1);
    expect(removed.free).toHaveBeenCalledTimes(1);
  });

  it("tears down an unknown id with no product", () => {
    const handler = makeHandler(async () => "product");
    const d = new JobDispatcher();
    d.register(handler);

    const removed = ev(9);
    d.tick(events({ mesh_removed: [removed] }));
    expect(handler.teardown).toHaveBeenCalledWith(undefined, removed);
    expect(removed.free).toHaveBeenCalledTimes(1);
  });

  it("holds the latest change while start is in flight and applies it once", async () => {
    const started = deferred<string>();
    const handler = makeHandler(() => started.promise);
    const d = new JobDispatcher();
    d.register(handler);

    d.tick(events({ mesh_added: [ev(1)] }));
    const c1 = ev(1);
    const c2 = ev(1);
    d.tick(events({ mesh_updated: [c1] }));
    d.tick(events({ mesh_updated: [c2] }));
    expect(handler.change).not.toHaveBeenCalled();
    expect(c1.free).toHaveBeenCalledTimes(1);
    expect(c2.free).not.toHaveBeenCalled();

    started.resolve("product");
    await flush();
    expect(handler.change).toHaveBeenCalledTimes(1);
    expect(handler.change).toHaveBeenCalledWith("product", c2);
    expect(c2.free).toHaveBeenCalledTimes(1);
  });

  it("applies a change directly to a live product and frees it after an async change", async () => {
    const changing = deferred<undefined>();
    const handler = makeHandler(async () => "product", {
      change: vi.fn(() => changing.promise),
    });
    const d = new JobDispatcher();
    d.register(handler);

    d.tick(events({ mesh_added: [ev(1)] }));
    await flush();

    const c = ev(1);
    d.tick(events({ mesh_updated: [c] }));
    expect(handler.change).toHaveBeenCalledWith("product", c);
    expect(c.free).not.toHaveBeenCalled();
    expect(d.inFlight()).toBe(1);

    // A remove during the async change waits for it.
    const removed = ev(1);
    d.tick(events({ mesh_removed: [removed] }));
    expect(handler.teardown).not.toHaveBeenCalled();

    changing.resolve(undefined);
    await flush();
    expect(c.free).toHaveBeenCalledTimes(1);
    expect(handler.teardown).toHaveBeenCalledWith("product", removed);
    expect(d.inFlight()).toBe(0);
  });

  it("settles a synchronous change in the same tick without holding the job in flight", async () => {
    const handler = makeHandler(async () => "product", {
      change: vi.fn(() => undefined),
    });
    const d = new JobDispatcher();
    d.register(handler);

    d.tick(events({ mesh_added: [ev(1)] }));
    await flush();

    const c = ev(1);
    d.tick(events({ mesh_updated: [c] }));
    expect(handler.change).toHaveBeenCalledWith("product", c);
    expect(c.free).toHaveBeenCalledTimes(1);
    expect(d.inFlight()).toBe(0);

    // A remove right after it tears down at once.
    const removed = ev(1);
    d.tick(events({ mesh_removed: [removed] }));
    expect(handler.teardown).toHaveBeenCalledWith("product", removed);
  });

  it("contains a synchronous throw from change", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const handler = makeHandler(async () => "product", {
      change: vi.fn(() => {
        throw new Error("boom");
      }),
    });
    const d = new JobDispatcher();
    d.register(handler);

    d.tick(events({ mesh_added: [ev(1)] }));
    await flush();

    const c = ev(1);
    d.tick(events({ mesh_updated: [c] }));
    expect(error).toHaveBeenCalled();
    expect(c.free).toHaveBeenCalledTimes(1);
    expect(d.inFlight()).toBe(0);
    error.mockRestore();
  });

  it("never overlaps changes on one job and applies only the latest waiting one", async () => {
    const first = deferred<undefined>();
    const handler = makeHandler(async () => "product", {
      change: vi.fn((_: string, c: Ev) =>
        c.ind === 1 && c.gen === 1 ? first.promise : undefined,
      ),
    });
    const d = new JobDispatcher();
    d.register(handler);
    d.tick(events({ mesh_added: [ev(1)] }));
    await flush();

    const c1 = ev(1, 1);
    const c2 = ev(1, 1);
    const c3 = ev(1, 1);
    d.tick(events({ mesh_updated: [c1] }));
    d.tick(events({ mesh_updated: [c2] }));
    d.tick(events({ mesh_updated: [c3] }));
    expect(handler.change).toHaveBeenCalledTimes(1);
    expect(c2.free).toHaveBeenCalledTimes(1);

    first.resolve(undefined);
    await flush();
    expect(handler.change).toHaveBeenCalledTimes(2);
    expect(handler.change).toHaveBeenLastCalledWith("product", c3);
    expect(c1.free).toHaveBeenCalledTimes(1);
    expect(c3.free).toHaveBeenCalledTimes(1);
    expect(d.inFlight()).toBe(0);
  });

  it("fires the abort signal when a live job is removed", async () => {
    let signal: AbortSignal | undefined;
    const handler = makeHandler(async (_, abort) => {
      signal = abort.signal;
      return "product";
    });
    const d = new JobDispatcher();
    d.register(handler);
    d.tick(events({ mesh_added: [ev(1)] }));
    await flush();
    expect(signal?.aborted).toBe(false);

    d.tick(events({ mesh_removed: [ev(1)] }));
    expect(signal?.aborted).toBe(true);
    expect(handler.teardown).toHaveBeenCalledWith("product", expect.anything());
  });

  it("drops a change for a job that resolved without a product", async () => {
    const handler = makeHandler(async () => undefined);
    const d = new JobDispatcher();
    d.register(handler);

    d.tick(events({ mesh_added: [ev(1)] }));
    await flush();
    const c = ev(1);
    d.tick(events({ mesh_updated: [c] }));
    expect(handler.change).not.toHaveBeenCalled();
    expect(c.free).toHaveBeenCalledTimes(1);
  });

  it("forgets a job that settled without a product", async () => {
    const handler = makeHandler(async () => undefined);
    const d = new JobDispatcher();
    d.register(handler);

    d.tick(events({ mesh_added: [ev(1)] }));
    await flush();
    // A repeated request starts again instead of being deduped.
    d.tick(events({ mesh_added: [ev(1)] }));
    expect(handler.start).toHaveBeenCalledTimes(2);

    await flush();
    const removed = ev(1);
    d.tick(events({ mesh_removed: [removed] }));
    expect(handler.teardown).toHaveBeenCalledWith(undefined, removed);
  });

  it("reports a rejected start through failed, but not after an abort", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const rejecting = deferred<string>();
    const handler = makeHandler((_, abort) =>
      abort.signal.aborted ? Promise.reject(new Error("x")) : rejecting.promise,
    );
    const d = new JobDispatcher();
    d.register(handler);

    const a = ev(1);
    d.tick(events({ mesh_added: [a] }));
    const error = new Error("boom");
    rejecting.reject(error);
    await flush();
    expect(handler.failed).toHaveBeenCalledWith(a, error);
    expect(a.free).toHaveBeenCalledTimes(1);

    // Aborted: rejection is expected, no report.
    const pending = deferred<string>();
    (handler.start as ReturnType<typeof vi.fn>).mockImplementation(
      () => pending.promise,
    );
    d.tick(events({ mesh_added: [ev(2)] }));
    d.tick(events({ mesh_removed: [ev(2)] }));
    pending.reject(new Error("aborted"));
    await flush();
    expect(handler.failed).toHaveBeenCalledTimes(1);
    expect(handler.teardown).toHaveBeenCalledWith(undefined, expect.anything());
    consoleError.mockRestore();
  });

  it("reports a start that rejects after aborting its own signal", async () => {
    const handler = makeHandler((_, abort) => {
      abort.abort();
      return Promise.reject(new Error("self-aborted"));
    });
    const d = new JobDispatcher();
    d.register(handler);

    const a = ev(1);
    d.tick(events({ mesh_added: [a] }));
    await flush();
    expect(handler.failed).toHaveBeenCalledWith(a, expect.any(Error));
    expect(handler.teardown).not.toHaveBeenCalled();
  });

  it("contains a synchronous throw from start", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const handler = makeHandler(() => {
      throw new Error("sync");
    });
    const d = new JobDispatcher();
    d.register(handler);

    const a = ev(1);
    expect(() => d.tick(events({ mesh_added: [a] }))).not.toThrow();
    await flush();
    expect(handler.failed).toHaveBeenCalledTimes(1);
    expect(a.free).toHaveBeenCalledTimes(1);
    expect(d.inFlight()).toBe(0);
    consoleError.mockRestore();
  });

  it("frees exactly what it took", async () => {
    const handler = makeHandler(async () => "product");
    const d = new JobDispatcher();
    d.register(handler);
    const sync = vi.fn();
    d.registerSync("mesh_geometry_replaced", sync);

    const taken = [ev(1), ev(2), ev(1), ev(3), ev(2), ev(7)];
    const [a1, a2, c1, r3, r2, g7] = taken;
    d.tick(
      makeEvent({
        mesh_added: [a1, a2],
        mesh_updated: [c1],
        mesh_removed: [r3, r2],
        mesh_geometry_replaced: [g7],
      } as unknown as Parameters<typeof makeEvent>[0]),
    );
    await flush();

    expect(sync).toHaveBeenCalledWith(g7);
    for (const e of taken) {
      expect(e.free).toHaveBeenCalledTimes(1);
    }
    expect(d.inFlight()).toBe(0);
  });

  it("calls onSettled only when a product landed", async () => {
    const onSettled = vi.fn();
    const handler = makeHandler(async (e: Ev) =>
      e.ind === 1 ? "product" : undefined,
    );
    const d = new JobDispatcher({ onSettled });
    d.register(handler);

    d.tick(events({ mesh_added: [ev(1), ev(2)] }));
    await flush();
    expect(onSettled).toHaveBeenCalledTimes(1);
  });

  it("aborts everything on dispose and tears down late products without an event", async () => {
    const started = deferred<string>();
    let signal: AbortSignal | undefined;
    const handler = makeHandler((_, abort) => {
      signal = abort.signal;
      return started.promise;
    });
    const d = new JobDispatcher();
    d.register(handler);

    d.tick(events({ mesh_added: [ev(1)] }));
    d.dispose();
    expect(signal?.aborted).toBe(true);

    started.resolve("late");
    await flush();
    expect(handler.teardown).toHaveBeenCalledWith("late", undefined);

    d.tick(events({ mesh_added: [ev(2)] }));
    expect(handler.start).toHaveBeenCalledTimes(1);
  });

  it("ignores stacks it has no handler for", () => {
    const d = new JobDispatcher();
    const untouched = ev(1);
    expect(() =>
      d.tick({
        take_mesh_added: () => [untouched],
      } as unknown as Events),
    ).not.toThrow();
    expect(untouched.free).not.toHaveBeenCalled();
  });
});
