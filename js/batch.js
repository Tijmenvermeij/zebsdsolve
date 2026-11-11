import { isOxfordH5, OXFORD_SOURCE_PC_WARNING } from "./oxford-export-policy.js";
import { BatchDownloads } from "./batch-downloads.js";
// Batch indexing owns its engine and GPU device. It never borrows the interactive engine.
const HINTS = ["map_rows", "map_cols", "pattern_height", "pattern_width"];
const SCAN_FIELDS = ["rows", "cols", "patH", "patW", "dtype", "format"];
const clone = (value) => structuredClone(value);
const message = (error) => String(error?.message || error);
const descriptor = (file) => ({ name: file.name, size: file.size, lastModified: file.lastModified ?? null });
const RESERVATION_LOCK = "zebsdsolve-batch-output-reservations-v1";
const MAX_JOURNAL_BYTES = 4 * 1024 * 1024;
const MAX_STEM_BYTES = 200; // Leaves room for every result extension on 255-byte filesystems.
const MAX_STEM_POINTS = 160;
const BANDWIDTH_TAIL = /(_ZEBSDsolve_BW[-+\deE.]{1,32}(?:_\d{1,7})?)$/;
const utf8 = new TextEncoder();
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
function fitStem(base, tail = "") {
  const byteBudget = MAX_STEM_BYTES - utf8.encode(tail).byteLength;
  const pointBudget = MAX_STEM_POINTS - Array.from(tail).length;
  let prefix = "", bytes = 0, points = 0;
  for (const point of base) {
    const size = utf8.encode(point).byteLength;
    if (bytes + size > byteBudget || points >= pointBudget) break;
    prefix += point; bytes += size; points++;
  }
  return prefix.replace(/[. ]+$/g, "") + tail;
}
function collisionStem(base, suffix = "") {
  const preserved = base.match(BANDWIDTH_TAIL)?.[0] || "";
  return fitStem(preserved ? base.slice(0, -preserved.length) : base, preserved + suffix);
}
function artifactSize(data) {
  const size = typeof data === "string" ? utf8.encode(data).byteLength : data?.size ?? data?.byteLength;
  if (!Number.isSafeInteger(size) || size < 0) throw new Error("Unsupported batch artifact data.");
  return size;
}
export function outputStem(value) {
  let name = String(value || "scan").replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").replace(/[. ]+$/g, "").trim();
  if (!name) name = "scan";
  if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(name)) name = `_${name}`;
  const bounded = collisionStem(name) || "scan";
  return /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(bounded) ? fitStem(`_${bounded}`) : bounded;
}
export function defaultOutputStem(file, bandwidth) {
  const suffix = `_ZEBSDsolve_BW${Number.isFinite(Number(bandwidth)) ? Math.trunc(Number(bandwidth)) : 0}`;
  const base = outputStem(String(file?.name || file || "scan").replace(/\.[^.]+$/, ""));
  return fitStem(base, suffix);
}
function snapshot(spec) {
  const sourceFiles = Object.freeze(Array.from(spec.sourceFiles || []));
  const masterFiles = Object.freeze(Array.from(spec.masterFiles || []));
  if (!sourceFiles.length || !masterFiles.length) throw new Error("A batch job needs scan files and master patterns.");
  if (!spec.settings || !spec.geometry || !spec.scanInfo) throw new Error("A batch job needs settings, geometry and scan metadata.");
  const warnings = isOxfordH5(spec.scanInfo.format) && spec.geometry.planeKind ? [OXFORD_SOURCE_PC_WARNING] : [];
  const openHints = Object.fromEntries(HINTS.map((key) => [key, spec.openHints?.[key] ?? spec.settings[key] ?? 0]));
  return {
    sourceFiles, masterFiles, warnings: Object.freeze(warnings), directory: spec.directory || null,
    settings: freeze(clone(spec.settings)), geometry: freeze(clone(spec.geometry)),
    openGeometry: spec.openGeometry == null ? null : freeze(clone(spec.openGeometry)),
    openHints: freeze(openHints), scanInfo: freeze(clone(spec.scanInfo)),
    phases: freeze(clone(spec.phases || [])), version: clone(spec.version ?? null),
    sources: freeze(sourceFiles.map(descriptor)), masters: freeze(masterFiles.map(descriptor)),
  };
}
const STOPPED = Symbol("stopped");

export class BatchQueue {
  constructor({ createEngine, artifacts, onChange = () => {}, downloadsAdapter = new BatchDownloads(), onDownloadReady = () => {} }) {
    this.createEngine = createEngine; this.artifacts = artifacts; this.onChange = onChange;
    this._downloads = downloadsAdapter; this.onDownloadReady = onDownloadReady; this.downloadsEnabled = false;
    this.jobs = []; this.running = false; this.directory = null; this.error = null;
    this._nextId = 1; this._stop = false;
    this._engine = null; this._retainedJob = null; this._active = null;
  }
  _change() { this.onChange(); }
  add(spec) {
    const automatic = defaultOutputStem(spec.sourceFiles[0], spec.settings?.bandwidth);
    const stem = spec.outputStem ? outputStem(spec.outputStem) : automatic;
    const job = { id: `job-${this._nextId++}`, ...snapshot(spec), outputStem: stem, autoName: spec.autoName ?? stem === automatic,
      status: "queued", progress: null, error: null, outputs: [], folderName: null, directoryName: null };
    this.jobs.push(job); this._change(); return job;
  }
  replace(id, spec) {
    const job = this._job(id);
    if (this.running || job.status !== "queued" || job.folderName) throw new Error("Only an unstarted queued job can be updated.");
    const automatic = defaultOutputStem(spec.sourceFiles[0], spec.settings?.bandwidth);
    const autoName = spec.autoName ?? (job.autoName && (spec.outputStem == null || outputStem(spec.outputStem) === job.outputStem || outputStem(spec.outputStem) === automatic));
    const directory = Object.hasOwn(spec, "directory") ? spec.directory : job.directory;
    Object.assign(job, snapshot(spec), { directory, autoName, outputStem: autoName ? automatic : outputStem(spec.outputStem ?? job.outputStem) });
    this._change(); return job;
  }
  _job(id) { const job = this.jobs.find((item) => item.id === id); if (!job) throw new Error("Unknown batch job."); return job; }
  remove(id) {
    if (this.running) throw new Error("Stop the batch before removing jobs.");
    const job = this._job(id);
    if (job === this._retainedJob) throw new Error("Clear the batch to discard pending exports.");
    if (this.modeFor(job) === "downloads") { this._downloads.release(job); job.archive = null; }
    job.resultCheckpoint = null; job.resultChance = null;
    this.jobs.splice(this.jobs.indexOf(job), 1); this._change();
  }
  rename(id, name) {
    const job = this._job(id);
    if (this.running || job.folderName) throw new Error("Only an unstarted job can be renamed.");
    job.outputStem = outputStem(name); job.autoName = false; this._change();
  }
  retry(id) {
    if (this.running) throw new Error("Stop the batch before retrying.");
    const job = this._job(id);
    if (["failed", "cancelled"].includes(job.status)) { job.resultCheckpoint = null; job.resultChance = null; job.status = "queued"; job.error = null; job.progress = null; }
    else if (job.status !== "export-pending") throw new Error("This job does not need a retry.");
    this._change(); return job;
  }
  setDirectory(handle) {
    if (this.running || this._retainedJob) throw new Error("Finish or clear pending work before changing the output folder.");
    if (this.jobs.some((job) => job.folderName)) throw new Error("Clear the batch before changing its output folder.");
    this.directory = handle; this._change();
  }
  setJobDirectory(id, handle) {
    const job = this._job(id);
    if (this.running || job.status !== "queued" || job._folder) throw new Error("Only an unstarted queued job can change its output folder.");
    job.directory = handle || null; this._change();
  }
  setDownloads(enabled) {
    enabled = !!enabled;
    if (enabled === this.downloadsEnabled) return;
    if (this.running || this._retainedJob || this.jobs.some(job => job._folder && job.status !== "complete")) throw new Error("Finish or clear reserved jobs before changing the output mode.");
    this.downloadsEnabled = enabled; this._change();
  }
  modeFor(job) { return job._transport || (this.downloadsEnabled ? "downloads" : "folder"); }
  directoryFor(job) { return job._root || (this.modeFor(job) === "downloads" ? job._downloadRoot || null : job.directory || this.directory); }
  get hasDestinations() { return this.jobs.filter((job) => ["queued", "export-pending"].includes(job.status)).every(job => this.modeFor(job) === "downloads" || !!this.directoryFor(job)); }
  stop() {
    this._stop = true;
    if (this._active?.status === "indexing") this._engine?.cancel?.();
    this._change();
  }
  async clear() {
    if (this.running) throw new Error("Stop the batch and wait for the current operation before clearing.");
    await this._release();
    for (const job of this.jobs) { job.resultCheckpoint = null; job.resultChance = null; }
    this._downloads.clear(this.jobs); this._retainedJob = null; this.jobs = [];
    this.error = null; this._change();
  }
  async _release() {
    const engine = this._engine; this._engine = null;
    if (engine) { try { await engine.dispose(); } finally { engine.device?.destroy(); } }
  }
  async _preflight(jobs, readInputs = true) {
    for (const job of jobs) {
      const dir = this.directoryFor(job);
      if (!dir || typeof dir.getDirectoryHandle !== "function" || typeof dir.getFileHandle !== "function" || typeof dir.queryPermission !== "function") {
        throw new Error(`Batch export requires the File System Access API and an output folder for ${job.sourceFiles[0].name}.`);
      }
      if (await dir.queryPermission({ mode: "readwrite" }) !== "granted") throw new Error(`Choose the output folder again to grant write permission for ${job.sourceFiles[0].name}.`);
      if (!readInputs) continue;
      for (const file of [...job.sourceFiles, ...job.masterFiles]) {
        if (!file || typeof file.slice !== "function" || !Number.isFinite(file.size) || file.size <= 0) throw new Error(`Unreadable input: ${file?.name || "file"}.`);
        // Confirm that each retained File is still readable without materializing patterns.
        const bytes = await file.slice(0, Math.min(16, file.size)).arrayBuffer();
        if (!bytes.byteLength) throw new Error(`Unreadable input: ${file.name}.`);
      }
    }
  }
  async _prepareDownloads(jobs) {
    const pending = jobs.filter(job => this.modeFor(job) === "downloads" && !job._root && !job._downloadRoot);
    if (pending.length) {
      const directory = await this._downloads.directory();
      for (const job of pending) job._downloadRoot = directory;
    }
  }
  async _exists(directory, name) {
    try { await directory.getFileHandle(name); return true; } catch (error) { if (error.name !== "NotFoundError" && error.name !== "TypeMismatchError") throw error; if (error.name === "TypeMismatchError") return true; }
    try { await directory.getDirectoryHandle(name); return true; } catch (error) { if (error.name === "NotFoundError") return false; if (error.name === "TypeMismatchError") return true; throw error; }
  }
  async _reserveFolder(job) {
    await reserveOutput(async () => {
      if (!job._folder) {
        const root = this.directoryFor(job);
        let name = job.outputStem;
        for (let i = 2; await this._exists(root, name); i++) {
          if (i > 1_000_000) throw new Error("Too many output folders share this name.");
          name = collisionStem(job.outputStem, `_${String(i).padStart(2, "0")}`);
        }
        job._folder = await root.getDirectoryHandle(name, { create: true });
        job._transport = this.modeFor(job); job._root = root; job.folderName = name; job.directoryName = name; job.outputStem = name;
        job.journalName = `${name}.json`;
        job.outputs = [".h5", isOxfordH5(job.scanInfo.format) ? ".h5oina" : ".ang", "_IPF-Z.png", "_NCC.png", ".json"].map((suffix) => ({ name: name + suffix, complete: false }));
      }
      if (!job._journal) {
        if (await this._exists(job._folder, job.journalName)) throw new Error(`Output already exists: ${job.journalName}.`);
        const handle = await job._folder.getFileHandle(job.journalName, { create: true });
        job._journal = { handle };
        const file = await handle.getFile();
        if (file.size !== 0) throw new Error(`Output reservation is no longer empty: ${job.journalName}.`);
        job._journal.placeholder = { size: 0, lastModified: file.lastModified };
      }
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
  _journalData(job, completing = false) {
    return { zebsdsolve: job.version, scan: job.sourceFiles[0].name,
      geometry_convention: isOxfordH5(job.scanInfo.format) ? "oxford-calibrated-v1" : "generic-v1",
      geometry: job.geometry, settings: job.settings, source_files: job.sources, master_files: job.masters,
      phases: job.phases, opening_hints: job.openHints, acquisition_geometry: job.openGeometry, scan_info: job.scanInfo,
      warnings: job.warnings, map_source: "native engine map: IPF-Z and detector NCC; same frame and colors as the GUI",
      batch: { schemaVersion: 1, id: job.id, outputStem: job.outputStem, directory: job.folderName, transport: this.modeFor(job),
        status: completing ? this.modeFor(job) === "downloads" ? "archive-ready" : "complete" : job.status, error: completing ? null : job.error,
        outputs: job.outputs.map(({ name, complete, error }) => ({ name, complete: completing && name === job.journalName ? true : complete,
          ...(error ? { error } : {}) })) } };
  }
  async _write(handle, data) {
    const stream = await handle.createWritable();
    try { await stream.write(data); await stream.close(); }
    catch (error) { try { await stream.abort?.(); } catch {} throw error; }
  }
  async _saveJournal(job, completing = false) {
    const journal = job._journal;
    if (!journal) throw new Error("The job journal is not reserved.");
    const text = JSON.stringify(this._journalData(job, completing), null, 2);
    if (new TextEncoder().encode(text).byteLength > MAX_JOURNAL_BYTES) throw new Error("The job journal exceeds its size limit.");
    await reserveOutput(async () => {
      const current = await job._folder.getFileHandle(job.journalName);
      const same = typeof journal.handle.isSameEntry === "function" ? await journal.handle.isSameEntry(current) : journal.handle === current;
      const file = await current.getFile();
      let unchanged = same && file.size <= MAX_JOURNAL_BYTES;
      if (journal.expected !== undefined) unchanged &&= file.size === new TextEncoder().encode(journal.expected).byteLength && await file.text() === journal.expected;
      else {
        unchanged &&= file.size === 0 && (!journal.placeholder || file.lastModified === journal.placeholder.lastModified);
        if (unchanged) journal.placeholder ??= { size: 0, lastModified: file.lastModified };
      }
      if (!unchanged) throw new Error(`Job journal changed or a failed close may have committed; inspect it before continuing: ${job.journalName}.`);
      await this._write(journal.handle, text);
      journal.expected = text;
    });
  }
  async _validateCompleted(job) {
    const errors = [];
    for (const entry of job.outputs) {
      if (entry.name === job.journalName) continue;
      if (entry.invalid) { errors.push(entry.error); continue; }
      if (!entry.complete) continue;
      let changed = false;
      try {
        if (!entry.handle || (!entry.saved && !entry.closed)) throw new Error("Saved output receipt is unavailable.");
        const current = await job._folder.getFileHandle(entry.name);
        const same = typeof entry.handle.isSameEntry === "function" ? await entry.handle.isSameEntry(current) : entry.handle === current;
        if (!same) changed = true;
        else {
          const file = await current.getFile();
          changed = file.size !== (entry.saved?.size ?? entry.closed.size) || !!entry.saved && file.lastModified !== entry.saved.lastModified;
          // A successful close is already an ownership receipt. If its first stat failed,
          // adopt metadata later only after checking the same handle and expected byte size.
          if (!changed) entry.saved ??= { size: file.size, lastModified: file.lastModified };
        }
      } catch (error) {
        if (error.name !== "NotFoundError" && error.name !== "TypeMismatchError") throw error;
        changed = true;
      }
      if (changed) {
        entry.complete = false; entry.invalid = true;
        entry.error = `Saved output is missing or changed; inspect it before continuing: ${entry.name}.`;
        errors.push(entry.error);
      }
    }
    if (errors.length) throw new Error(errors.join(" "));
  }
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
    // Capture once, before artifacts or reuse of this private engine. Downloads may
    // reclaim their staged files; this immutable Blob belongs to the retained row.
    if (!job.resultCheckpoint) {
      const chance = this._engine.resultChance?.() ?? null;
      job.resultCheckpoint = new Blob([this._engine.exportResultCheckpoint()], { type: "application/octet-stream" });
      job.resultChance = chance == null ? null : Object.freeze(clone(chance));
    }
    await this._validateCompleted(job);
    job.status = "exporting"; this._change();
    await this._saveJournal(job);
    const completed = new Set(job.outputs.filter((item) => item.complete).map((item) => item.name));
    for await (const artifact of this.artifacts(this._engine, job, completed)) {
      if (!artifact?.name || artifact.name === job.journalName || !job.outputs.some((entry) => entry.name === artifact.name)) throw new Error("Unexpected batch output filename.");
      if (completed.has(artifact.name)) continue;
      const entry = job.outputs.find((item) => item.name === artifact.name);
      const size = artifactSize(artifact.data);
      const handle = await this._artifactHandle(job, entry);
      await this._write(handle, artifact.data);
      // Only an acknowledged close creates this receipt. A thrown close keeps the
      // original empty-placeholder guard and cannot adopt possibly committed content.
      entry.closed = { size };
      entry.complete = true; completed.add(entry.name);
      await this._validateCompleted(job);
      await this._saveJournal(job); this._change();
    }
    if (job.outputs.some((entry) => entry.name !== job.journalName && !entry.complete)) throw new Error("Required batch result files are missing.");
    await this._validateCompleted(job);
    // Keep the live status exporting until the completed project journal has closed.
    await this._saveJournal(job, true);
    job.outputs.find((entry) => entry.name === job.journalName).complete = true;
    if (this.modeFor(job) === "downloads") {
      await this._validateCompleted(job);
      job.archive = await this._downloads.archive(job);
    }
    job.status = "complete"; job.error = null; this._change();
    if (job.archive && !job._downloadNotified) {
      job._downloadNotified = true;
      try { await this.onDownloadReady(job); } catch (error) { job.downloadError = message(error); }
      this._change();
    }
    if (job.archive) {
      try { await this._downloads.reclaim?.(job); } catch (error) { job.cleanupError = message(error); }
      this._change();
    }
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
      await this._prepareDownloads(retained ? [retained] : queued);
      await this._preflight(retained ? [retained] : queued, !retained);
      let queuedPreflighted = !retained;
      for (const job of pending) {
        if (this._stop) break;
        if (!queuedPreflighted && job !== retained) {
          await this._prepareDownloads(queued);
          await this._preflight(queued); queuedPreflighted = true;
        }
        this._active = job;
        let exporting = job.status === "export-pending";
        try {
          await this._reserveFolder(job); await this._saveJournal(job);
          if (!exporting) {
            this._checkStop();
            job.status = "indexing"; job.error = null; this._change();
            await this._saveJournal(job);
            if (!this._engine) this._engine = await this.createEngine();
            await this._index(job);
            exporting = true;
          } else if (this._retainedJob !== job || !this._engine) throw new Error("Pending export results are no longer available.");
          await this._export(job);
          this._retainedJob = null;
          if (this._engine.deviceLost) this._stop = true;
        } catch (error) {
          job.error = error === STOPPED ? "Stopped by user." : message(error);
          job.status = exporting ? "export-pending" : error === STOPPED ? "cancelled" : "failed";
          if (exporting && this.modeFor(job) === "downloads") job.outputs.find(entry => entry.name === job.journalName).complete = false;
          if (exporting) this._retainedJob = job;
          this.error = job.error;
          try { if (job._journal) await this._saveJournal(job); } catch (journalError) { job.error += ` Journal: ${message(journalError)}`; this.error = job.error; }
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
