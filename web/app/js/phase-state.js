/** Drain native phase consumers before synchronous destruction. Retirement belongs to the
 * caller; a stale generation or device loss must never clear a replacement engine. */
export async function clearPhasesWhenIdle(engine, current) {
  for (;;) {
    const pending = engine._queue;
    await pending;
    if (!current()) return false;
    // Work may have joined the queue while the awaited consumer was suspended.
    if (pending !== engine._queue) continue;
    engine.clearPhases();
    return true;
  }
}

/** Keep the active file list equal to the successfully imported native prefix on failure.
 * Unloaded files stay available, switched off, so a retry never invents active phases. */
export async function replacePhaseEntries(engine, entries, current) {
  if (!await clearPhasesWhenIdle(engine, current)) return false;
  const selected = entries.filter(entry => entry.on);
  let loaded = 0, complete = false;
  try {
    for (const entry of selected) {
      await engine.addPhase(entry.file);
      loaded++;
      if (!current()) return false;
    }
    complete = true;
    return true;
  } finally {
    if (!complete) selected.forEach((entry, i) => { entry.on = i < loaded; });
  }
}
