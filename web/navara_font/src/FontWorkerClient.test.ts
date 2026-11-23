import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { FontWorkerClient } from "./FontWorkerClient";

const mocks = vi.hoisted(() => ({
  reserveWorkerCore: vi.fn(),
  release: vi.fn(),
}));

type Message = { id: number; type: string };

/** A worker that answers only when a test tells it to. */
class MockWorker {
  static last: MockWorker;
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  received: Message[] = [];
  constructor() {
    MockWorker.last = this;
  }
  postMessage(message: Message) {
    this.received.push(message);
  }
  terminate() {}
  /** Answer the oldest unanswered request of `type`. */
  reply(type: string) {
    const index = this.received.findIndex((m) => m.type === type);
    const [message] = this.received.splice(index, 1);
    this.onmessage?.({
      data: { id: message.id, type: "ok", payload: { ok: true, results: [] } },
    } as MessageEvent);
  }
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("Worker", MockWorker);
  mocks.reserveWorkerCore.mockReset().mockReturnValue(mocks.release);
  mocks.release.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("holds one pool core while heavy requests run and releases it after a short idle", async () => {
  const client = new FontWorkerClient("font-worker.js", {
    reserveWorkerCore: mocks.reserveWorkerCore,
  });
  const worker = MockWorker.last;
  expect(mocks.reserveWorkerCore).toHaveBeenCalledOnce();

  const shaping = client.prepareTextBatch("font.ttf", ["a"]);
  worker.reply("init");
  await vi.advanceTimersByTimeAsync(200);
  // Shaping is still in flight.
  expect(mocks.release).not.toHaveBeenCalled();

  worker.reply("prepareTextBatch");
  await shaping;
  await vi.advanceTimersByTimeAsync(50);
  // A request within the delay keeps the same reservation.
  const loading = client.loadFont(
    "b.ttf",
    new ArrayBuffer(1),
    undefined,
    false,
  );
  worker.reply("loadFont");
  await loading;
  await vi.advanceTimersByTimeAsync(99);
  expect(mocks.release).not.toHaveBeenCalled();
  expect(mocks.reserveWorkerCore).toHaveBeenCalledOnce();

  await vi.advanceTimersByTimeAsync(1);
  expect(mocks.release).toHaveBeenCalledOnce();

  client.dispose();
  expect(mocks.release).toHaveBeenCalledOnce();
});

it("does not hold a core for glyph reference updates", async () => {
  const client = new FontWorkerClient("font-worker.js", {
    reserveWorkerCore: mocks.reserveWorkerCore,
  });
  const worker = MockWorker.last;
  worker.reply("init");
  await vi.advanceTimersByTimeAsync(100);
  expect(mocks.release).toHaveBeenCalledOnce();

  client.retainGlyphs("atlas", new BigUint64Array([1n]));
  expect(mocks.reserveWorkerCore).toHaveBeenCalledOnce();
  worker.reply("retainGlyphs");
  client.dispose();
});

it("holds no core without a reservation function", () => {
  const client = new FontWorkerClient("font-worker.js");
  MockWorker.last.reply("init");
  client.dispose();
  expect(mocks.reserveWorkerCore).not.toHaveBeenCalled();
});

it("releases the core at once when disposed", async () => {
  const client = new FontWorkerClient("font-worker.js", {
    reserveWorkerCore: mocks.reserveWorkerCore,
  });
  // The unanswered init is rejected by the dispose.
  const ready = client.ready().catch(() => undefined);
  client.dispose();
  expect(mocks.release).toHaveBeenCalledOnce();
  await ready;
});
