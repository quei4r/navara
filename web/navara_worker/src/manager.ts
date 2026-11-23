/**
 * Counts the worker pool's tasks in flight; `queueTask` increments and
 * decrements it around every task. `total` is the pool size. A thread
 * outside the pool takes a core with `reserveWorkerCore` instead: a count
 * here would free no core while the engine keeps two tasks per worker.
 */
export type ConcurrencyManager = {
  /** Total concurrency */
  total: number;
  /** Increment the count according to the activated process. Counts past
   * `total` too (the engine keeps more pool tasks in flight than there are
   * workers), so every increment is matched by its decrement. */
  increment: () => void;
  /** Decrement the count according to the deactivated process */
  decrement: () => void;
  /** Reset the count */
  reset: () => void;
  /** The number of activated processes */
  actives: () => number;
  /** The number of idle processes */
  idles: () => number;
  /** Whether it can activate the process */
  canIncrement: () => boolean;
  /** Whether there is an activated process */
  canDecrement: () => boolean;
};
/** The pool occupancy as read outside `@navaramap/worker`. */
export type ReadonlyConcurrencyManager = Pick<
  ConcurrencyManager,
  "total" | "actives" | "idles"
>;

export const createConcurrencyManager = (total: number): ConcurrencyManager => {
  let actives = 0;

  const canIncrement = () => actives < total;
  const canDecrement = () => actives > 0;

  return {
    total,
    actives: () => actives,
    idles: () => Math.max(0, total - actives),
    increment: () => {
      actives++;
    },
    decrement: () => {
      if (!canDecrement()) return;
      actives--;
    },
    canIncrement,
    canDecrement,
    reset: () => {
      actives = 0;
    },
  };
};
