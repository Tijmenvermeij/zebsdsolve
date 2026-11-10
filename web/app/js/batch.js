// Batch indexing owns its engine and GPU device. It never borrows the interactive engine.
const HINTS = ["map_rows", "map_cols", "pattern_height", "pattern_width"];
const SCAN_FIELDS = ["rows", "cols", "patH", "patW", "dtype", "format"];
const clone = (value) => structuredClone(value);
const message = (error) => String(error?.message || error);
const descriptor = (file) => ({ name: file.name, size: file.size, lastModified: file.lastModified ?? null });
const RESERVATION_LOCK = "zebsdsolve-batch-output-reservations-v1";
let reservationTail = Promise.resolve();
async function reserveOutput(operation) {
  // A stable Web Lock serializes reservations across queues and browser tabs on this origin.
  if (globalThis.navigator?.locks?.request) return globalThis.navigator.locks.request(RESERVATION_LOCK, { mode: "exclusive" }, operation);
  // Non-browser runtimes share this module-local mutex.
  const previous = reservationTail;
  let release;
  reservationTail = new Promise((resolve) => { release = resolve; });
  await previous;
  try { return await operation(); } finally { release(); }
}
function freeze(value) {
  if (value && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}
export function outputStem(value) {
  let name = String(value || "scan").replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").replace(/[. ]+$/g, "").trim();
  if (!name) name = "scan";
  if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(name)) name = `_${name}`;
  return name.slice(0, 160).replace(/[. ]+$/g, "");
}
function snapshot(spec) {
  const sourceFiles = Object.freeze(Array.from(spec.sourceFiles || []));
  const masterFiles = Object.freeze(Array.from(spec.masterFiles || []));
  if (!sourceFiles.length || !masterFiles.length) throw new Error("A batch job needs scan files and master patterns.");
  if (!spec.settings || !spec.geometry || !spec.scanInfo) throw new Error("A batch job needs settings, geometry and scan metadata.");
  const openHints = Object.fromEntries(HINTS.map((key) => [key, spec.openHints?.[key] ?? spec.settings[key] ?? 0]));
  return {
    sourceFiles, masterFiles,
    settings: freeze(clone(spec.settings)), geometry: freeze(clone(spec.geometry)),
    openGeometry: spec.openGeometry == null ? null : freeze(clone(spec.openGeometry)),
    openHints: freeze(openHints), scanInfo: freeze(clone(spec.scanInfo)),
    phases: freeze(clone(spec.phases || [])), version: clone(spec.version ?? null),
    sources: freeze(sourceFiles.map(descriptor)), masters: freeze(masterFiles.map(descriptor)),
  };
}
const STOPPED = Symbol("stopped");

export class BatchQueue {
  constructor({ createEngine, artifacts, onChange = () => {} }) {
    this.createEngine = createEngine; this.artifacts = artifacts; this.onChange = onChange;
    this.jobs = []; this.running = false; this.directory = null; this.error = null;
    this.manifestName = null; this._nextId = 1; this._stop = false;
    this._engine = null; this._retainedJob = null; this._active = null;
  }
  _change() { this.onChange(); }
  add(spec) {
    const job = { id: `job-${this._nextId++}`, ...snapshot(spec), outputStem: outputStem(spec.outputStem || `${spec.sourceFiles[0].name.replace(/\.[^.]+$/, "")}_zebsdsolve`),
      status: "queued", progress: null, error: null, outputs: [], folderName: null, directoryName: null };
    this.jobs.push(job); this._change(); return job;
  }
  replace(id, spec) {
    const job = this._job(id);
    if (this.running || job.status !== "queued" || job.folderName) throw new Error("Only an unstarted queued job can be updated.");
    Object.assign(job, snapshot(spec), { outputStem: outputStem(spec.outputStem ?? job.outputStem) });
    this._change(); return job;
  }
  _job(id) { const job = this.jobs.find((item) => item.id === id); if (!job) throw new Error("Unknown batch job."); return job; }
  remove(id) {
    if (this.running) throw new Error("Stop the batch before removing jobs.");
    const job = this._job(id);
    if (job === this._retainedJob) throw new Error("Clear the batch to discard pending exports.");
    this.jobs.splice(this.jobs.indexOf(job), 1); this._change();
  }
  rename(id, name) {
    const job = this._job(id);
    if (this.running || job.folderName) throw new Error("Only an unstarted job can be renamed.");
    job.outputStem = outputStem(name); this._change();
  }
  retry(id) {
    if (this.running) throw new Error("Stop the batch before retrying.");
    const job = this._job(id);
    if (["failed", "cancelled"].includes(job.status)) { job.status = "queued"; job.error = null; job.progress = null; }
    else if (job.status !== "export-pending") throw new Error("This job does not need a retry.");
    this._change(); return job;
  }
  setDirectory(handle) {
    if (this.running || this._retainedJob) throw new Error("Finish or clear pending work before changing the output folder.");
    if (this.jobs.some((job) => job.folderName)) throw new Error("Clear the batch before changing its output folder.");
    this.directory = handle; this.manifestName = null; this._manifest = null; this._change();
  }
  stop() {
    this._stop = true;
    if (this._active?.status === "indexing") this._engine?.cancel?.();
    this._change();
  }
  async clear() {
    if (this.running) throw new Error("Stop the batch and wait for the current operation before clearing.");
    await this._release(); this._retainedJob = null; this.jobs = [];
    this.manifestName = null; this._manifest = null; this.error = null; this._change();
  }
  async _release() {
    const engine = this._engine; this._engine = null;
    if (engine) { try { await engine.dispose(); } finally { engine.device?.destroy(); } }
  }
  async _preflight(jobs) {
    const dir = this.directory;
    if (!dir || typeof dir.getDirectoryHandle !== "function" || typeof dir.getFileHandle !== "function" || typeof dir.queryPermission !== "function") {
      throw new Error("Batch export requires the File System Access API and a writable output folder.");
    }
    if (await dir.queryPermission({ mode: "readwrite" }) !== "granted") throw new Error("Choose the output folder again to grant write permission.");
    for (const job of jobs) for (const file of [...job.sourceFiles, ...job.masterFiles]) {
      if (!file || typeof file.slice !== "function" || !Number.isFinite(file.size) || file.size <= 0) throw new Error(`Unreadable input: ${file?.name || "file"}.`);
      // Confirm that each retained File is still readable without materializing patterns.
      const bytes = await file.slice(0, Math.min(16, file.size)).arrayBuffer();
      if (!bytes.byteLength) throw new Error(`Unreadable input: ${file.name}.`);
    }
  }
  async _exists(directory, name) {
    try { await directory.getFileHandle(name); return true; } catch (error) { if (error.name !== "NotFoundError" && error.name !== "TypeMismatchError") throw error; if (error.name === "TypeMismatchError") return true; }
    try { await directory.getDirectoryHandle(name); return true; } catch (error) { if (error.name === "NotFoundError") return false; if (error.name === "TypeMismatchError") return true; throw error; }
  }
  async _unique(base, extension = "") {
    for (let i = 1; ; i++) { const name = `${base}${i === 1 ? "" : `-${String(i).padStart(2, "0")}`}${extension}`; if (!await this._exists(this.directory, name)) return name; }
  }
  async _reserveManifest() {
    if (this._manifest) return;
    await reserveOutput(async () => {
      this.manifestName = await this._unique("batch", ".json");
      this._manifest = await this.directory.getFileHandle(this.manifestName, { create: true });
    });
  }
  async _reserveFolder(job) {
    if (job._folder) return;
    await reserveOutput(async () => {
      let name = job.outputStem;
      for (let i = 2; await this._exists(this.directory, name); i++) name = `${job.outputStem}_${String(i).padStart(2, "0")}`;
      job._folder = await this.directory.getDirectoryHandle(name, { create: true });
      job.folderName = name; job.directoryName = name; job.outputStem = name;
    });
  }
  async _artifactHandle(job, entry) {
    return reserveOutput(async () => {
      if (entry.handle) {
        const current = await job._folder.getFileHandle(entry.name);
        const same = typeof entry.handle.isSameEntry === "function" ? await entry.handle.isSameEntry(current) : entry.handle === current;
        if (!same) throw new Error(`Output changed since the earlier attempt; inspect it before continuing: ${entry.name}.`);
        // Native isSameEntry identifies a path, so a replacement at that path can pass it.
        // Unclosed atomic writes must still leave our original empty reservation intact.
        const file = await current.getFile();
        if (file.size !== 0 || entry.placeholder && (file.size !== entry.placeholder.size || file.lastModified !== entry.placeholder.lastModified)) {
          throw new Error(`Output changed since the earlier attempt; inspect it before continuing: ${entry.name}.`);
        }
        // An initial metadata-read failure never started a write. Adopt the first readable
        // empty reservation, while preserving any nonempty content placed at this path.
        entry.placeholder ??= { size: file.size, lastModified: file.lastModified };
      } else {
        // A recorded intent is not ownership: only creation returns the handle we own.
        if (await this._exists(job._folder, entry.name)) throw new Error(`Output already exists: ${entry.name}.`);
        entry.handle = await job._folder.getFileHandle(entry.name, { create: true });
        const file = await entry.handle.getFile();
        if (file.size !== 0) throw new Error(`Output reservation is no longer empty: ${entry.name}.`);
        entry.placeholder = { size: file.size, lastModified: file.lastModified };
      }
      return entry.handle;
    });
  }
  _manifestData(completingJob = null) {
    return { schemaVersion: 1, jobs: this.jobs.map((job) => ({ id: job.id, status: job === completingJob ? "complete" : job.status, outputStem: job.outputStem,
      directory: job.folderName, sources: job.sources, masters: job.masters, settings: job.settings, geometry: job.geometry, openGeometry: job.openGeometry,
      openHints: job.openHints, scanInfo: job.scanInfo, phases: job.phases, version: job.version,
      outputs: job.outputs.map(({ name, complete }) => ({ name, complete })), error: job === completingJob ? null : job.error })) };
  }
  async _write(handle, data) {
    const stream = await handle.createWritable();
    try { await stream.write(data); await stream.close(); }
    catch (error) { try { await stream.abort?.(); } catch {} throw error; }
  }
  async _saveManifest(completingJob = null) { await this._write(this._manifest, JSON.stringify(this._manifestData(completingJob), null, 2)); }
  _checkStop() { if (this._stop) throw STOPPED; }
  _checkDevice() { if (this._engine.deviceLost) throw new Error(`GPU device lost: ${message(this._engine.deviceLost)}`); }
  async _index(job) {
    const engine = this._engine;
    this._checkStop(); this._checkDevice();
    await engine.closeScan?.();
    engine.configure(clone(job.openHints));
    const opened = await engine.openScan(job.sourceFiles);
    const actual = engine.scan || engine.scanInfo || opened;
    for (const key of SCAN_FIELDS) if (job.scanInfo[key] !== actual?.[key]) throw new Error(`Scan ${key} changed: expected ${job.scanInfo[key]}, got ${actual?.[key]}.`);
    engine.configure(clone(job.settings));
    engine.setGeometry(clone(job.geometry));
    await engine.clearPhases();
    for (const file of job.masterFiles) { this._checkStop(); await engine.addPhase(file); }
    this._checkStop(); this._checkDevice();
    await engine.indexBegin();
    this._checkStop(); this._checkDevice();
    const started = performance.now(); let visits = 0;
    for (;;) {
      this._checkStop(); this._checkDevice();
      const before = engine.indexStatus;
      const status = await engine.indexStep();
      this._checkStop(); this._checkDevice();
      visits += Math.min(Math.max(0, before.total - before.done), job.settings.batch || before.total);
      const pps = visits / Math.max((performance.now() - started) / 1000, 0.001);
      job.progress = { done: status.done, total: status.total, sweep: status.sweep, sweeps: status.sweeps, pps,
        etaSeconds: status.left === 0 ? 0 : Math.max(0, status.total - status.done) / pps };
      this._change();
      if (status.left === 0) break;
    }
  }
  async _export(job) {
    job.status = "exporting"; this._change();
    await this._saveManifest();
    const completed = new Set(job.outputs.filter((item) => item.complete).map((item) => item.name));
    for await (const artifact of this.artifacts(this._engine, job, completed)) {
      if (!artifact?.name || /[/\\]/.test(artifact.name) || artifact.name === "." || artifact.name === "..") throw new Error("Invalid batch output filename.");
      if (completed.has(artifact.name)) continue;
      let entry = job.outputs.find((item) => item.name === artifact.name);
      if (!entry) {
        entry = { name: artifact.name, complete: false }; job.outputs.push(entry);
        await this._saveManifest();
      }
      const handle = await this._artifactHandle(job, entry);
      await this._write(handle, artifact.data);
      entry.complete = true; completed.add(entry.name);
      await this._saveManifest(); this._change();
    }
    // Keep the live status exporting until the completed manifest has closed.
    await this._saveManifest(job);
    job.status = "complete"; job.error = null; this._change();
  }
  async run() {
    if (this.running) throw new Error("The batch is already running.");
    const retained = this._retainedJob;
    const exports = this.jobs.filter((job) => job.status === "export-pending");
    if (retained ? !this.jobs.includes(retained) || retained.status !== "export-pending" || !this._engine || !retained._folder || exports.length !== 1
      : exports.length > 0) throw new Error("Pending export state is invalid. Clear the batch before continuing.");
    // The retained engine belongs to this export, regardless of the visible queue order.
    const queued = this.jobs.filter((job) => job.status === "queued");
    const pending = retained ? [retained, ...queued] : queued;
    if (!pending.length) return;
    this.running = true; this._stop = false; this.error = null; this._change();
    try {
      // Retained CPU results need only the output folder. Check new inputs after saving them.
      await this._preflight(retained ? [] : queued); await this._reserveManifest();
      let queuedPreflighted = !retained;
      for (const job of pending) {
        if (this._stop) break;
        if (!queuedPreflighted && job !== retained) {
          await this._preflight(queued); queuedPreflighted = true;
        }
        this._active = job;
        let exporting = job.status === "export-pending";
        try {
          await this._reserveFolder(job); await this._saveManifest();
          if (!exporting) {
            this._checkStop();
            if (!this._engine) this._engine = await this.createEngine();
            job.status = "indexing"; job.error = null; this._change();
            await this._saveManifest(); await this._index(job);
            exporting = true;
          } else if (this._retainedJob !== job || !this._engine) throw new Error("Pending export results are no longer available.");
          await this._export(job);
          this._retainedJob = null;
          if (this._engine.deviceLost) this._stop = true;
        } catch (error) {
          job.error = error === STOPPED ? "Stopped by user." : message(error);
          job.status = exporting ? "export-pending" : error === STOPPED ? "cancelled" : "failed";
          if (exporting) this._retainedJob = job;
          this.error = job.error;
          try { await this._saveManifest(); } catch (manifestError) { job.error += ` Manifest: ${message(manifestError)}`; this.error = job.error; }
          this._change(); break;
        }
      }
    } catch (error) { this.error = message(error); }
    finally {
      this._active = null;
      if (!this._retainedJob) { try { await this._release(); } catch (error) { this.error ||= message(error); } }
      this.running = false; this._change();
    }
  }
}
