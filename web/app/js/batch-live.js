// Render only inside the queue's settled-step callback. User input changes the
// requested colour without entering a private engine that may still be suspended.
export function createBatchLiveMap(canvas, controls, reportError) {
  const buttons = [...controls.querySelectorAll("button[data-kind]")];
  let job = null, ready = false, failed = false;
  let kind = "ipf_z", lastDraw = -Infinity, drawMs = 0;

  function updateControls() {
    for (const button of buttons) {
      button.disabled = !ready || failed || job.status !== "indexing";
      button.setAttribute("aria-pressed", String(button.dataset.kind === kind));
    }
  }

  for (const button of buttons) button.addEventListener("click", () => {
    if (kind === button.dataset.kind) return;
    kind = button.dataset.kind;
    canvas.hidden = true;
    lastDraw = -Infinity;
    updateControls();
  });

  return {
    showJob(next) {
      if (next !== job) {
        job = next;
        ready = failed = false;
        canvas.hidden = true;
        canvas.width = canvas.height = 1;
        lastDraw = -Infinity;
        drawMs = 0;
      }
      updateControls();
    },

    opened(active) {
      if (active !== job) return;
      ready = true;
      updateControls();
    },

    draw(engine, active, status, visible) {
      if (active !== job || !ready || failed) return;
      if (!visible) return;
      const start = performance.now();
      // Budget each intermediate raster for at most a twentieth of the time.
      if (status.left !== 0 && start - lastDraw < Math.max(250, 20 * drawMs)) return;
      try {
        const { rows, cols } = active.scanInfo;
        const pixels = engine.map(kind);
        const image = new ImageData(pixels, cols, rows);
        if (canvas.width !== cols) canvas.width = cols;
        if (canvas.height !== rows) canvas.height = rows;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Map canvas unavailable");
        ctx.putImageData(image, 0, 0);
        canvas.hidden = false;
      } catch (error) {
        failed = true;
        canvas.hidden = true;
        updateControls();
        reportError(error);
      }
      lastDraw = performance.now();
      drawMs = lastDraw - start;
    },
  };
}
