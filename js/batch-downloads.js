import { createStoreZip } from "./zip-store.js";

const PREFIX = ".zebsdsolve-downloads-";
const MARKER = ".owner.json";
const MAGIC = "zebsdsolve-downloads";
const VERSION = 1;
const MARKER_LIMIT = 4096;
const UUID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
const WORKSPACE_NAME = new RegExp("^" + PREFIX.replaceAll(".", "\\.") + "(" + UUID + ")$", "i");
const lockName = id => "zebsdsolve-downloads-workspace:" + id;
const utf8 = new TextEncoder();

function privateDirectory(handle) {
  // OPFS belongs to this origin; its wrapper never asks for external file permissions.
  return { name: handle.name, kind: "directory", queryPermission: async () => "granted",
    getDirectoryHandle: (...args) => handle.getDirectoryHandle(...args),
    getFileHandle: (...args) => handle.getFileHandle(...args) };
}
async function readMarker(handle, id) {
  const file = await (await handle.getFileHandle(MARKER)).getFile();
  if (file.size < 1 || file.size > MARKER_LIMIT) return null;
  const marker = JSON.parse(await file.text());
  return marker && marker.magic === MAGIC && marker.version === VERSION && marker.id === id
    && typeof marker.lockProtected === "boolean"
    && Number.isSafeInteger(marker.created) && marker.created >= 0
    && Number.isSafeInteger(marker.lastUsed) && marker.lastUsed >= marker.created ? marker : null;
}

export class BatchDownloads {
  constructor({ getDirectory = async () => {
    if (!globalThis.navigator?.storage?.getDirectory) throw new Error("Downloads staging requires origin-private file storage in this browser.");
    return globalThis.navigator.storage.getDirectory();
  }, createObjectURL = blob => URL.createObjectURL(blob), revokeObjectURL = url => URL.revokeObjectURL(url),
  scheduleCleanup = (operation, delay) => { const timer = setTimeout(operation, delay); timer.unref?.(); },
  cleanupDelayMs = 60_000, staleAfterMs = 24 * 60 * 60 * 1000, now = () => Date.now(),
  locks = globalThis.navigator?.locks || null, randomId = () => crypto.randomUUID(), pack = createStoreZip } = {}) {
    this.getDirectory = getDirectory; this.createObjectURL = createObjectURL; this.revokeObjectURL = revokeObjectURL;
    this.scheduleCleanup = scheduleCleanup; this.cleanupDelayMs = Math.max(60_000, cleanupDelayMs);
    this.staleAfterMs = Math.max(0, staleAfterMs); this.now = now; this.locks = locks; this.randomId = randomId; this.pack = pack;
    this._archives = new Map(); this._workspaces = new WeakMap(); this._rootPromise = null;
  }
  async _hold(id) {
    if (!this.locks?.request) return { protected: false, release() {} };
    let ready, reject, release;
    const acquired = new Promise((resolve, fail) => { ready = resolve; reject = fail; });
    const lifetime = new Promise(resolve => { release = resolve; });
    const request = Promise.resolve().then(() => this.locks.request(lockName(id), { mode: "exclusive" }, async lock => {
      if (!lock) throw new Error("Cannot own the Downloads workspace lock.");
      ready();
      await lifetime;
    }));
    request.catch(reject);
    await acquired;
    return { protected: true, release, request };
  }
  async _reap(parent) {
    if (!this.locks?.request || typeof parent.entries !== "function") return;
    try {
      for await (const [name, handle] of parent.entries()) {
        const match = name.match(WORKSPACE_NAME);
        if (!match || handle.kind !== "directory") continue;
        try {
          await this.locks.request(lockName(match[1]), { mode: "exclusive", ifAvailable: true }, async lock => {
            if (!lock) return;
            const marker = await readMarker(handle, match[1]);
            const now = this.now();
            if (marker?.lockProtected && now - marker.lastUsed >= this.staleAfterMs && now - marker.created >= this.staleAfterMs) {
              await parent.removeEntry(name, { recursive: true });
            }
          });
        } catch {} // Unknown, unreadable or active workspaces are never reclaimed.
      }
    } catch {} // A failed cache scan must not prevent a fresh, privately owned workspace.
  }
  async _writeMarker(workspace) {
    const handle = await workspace.handle.getFileHandle(MARKER, { create: true });
    const file = await handle.getFile(), expected = workspace.markerText ?? "";
    if (file.size > MARKER_LIMIT || await file.text() !== expected) throw new Error("The Downloads ownership marker changed.");
    const marker = { magic: MAGIC, version: VERSION, id: workspace.id, lockProtected: workspace.lock.protected,
      created: workspace.created, lastUsed: Math.max(workspace.created, this.now()) };
    const text = JSON.stringify(marker), stream = await handle.createWritable();
    try { await stream.write(text); await stream.close(); workspace.markerText = text; }
    catch (error) { try { await stream.abort?.(); } catch {} throw error; }
  }
  async directory() {
    if (!this._rootPromise) this._rootPromise = (async () => {
      const parent = await this.getDirectory();
      await this._reap(parent);
      for (let attempt = 0; attempt < 32; attempt++) {
        const id = this.randomId(), name = PREFIX + id, lock = await this._hold(id);
        let created = false;
        try {
          try { await parent.getDirectoryHandle(name); lock.release(); continue; }
          catch (error) {
            if (error.name === "TypeMismatchError") { lock.release(); continue; }
            if (error.name !== "NotFoundError") throw error;
          }
          const handle = await parent.getDirectoryHandle(name, { create: true }); created = true;
          const root = privateDirectory(handle), workspace = { parent, name, id, root, handle, lock, created: this.now() };
          await this._writeMarker(workspace);
          this._workspace = workspace; this._workspaces.set(root, workspace);
          return root;
        } catch (error) {
          if (created) { try { await parent.removeEntry(name, { recursive: true }); } catch {} }
          lock.release();
          throw error;
        }
      }
      throw new Error("Cannot reserve a unique Downloads workspace.");
    })().catch(error => { this._rootPromise = null; throw error; });
    return this._rootPromise;
  }
  async archive(job) {
    if (this._archives.has(job)) return this._archives.get(job).archive;
    const workspace = this._workspaces.get(job._root);
    if (!workspace) throw new Error("The Downloads staging folder is not owned by this batch.");
    const members = [];
    for (const entry of job.outputs) {
      const handle = await job._folder.getFileHandle(entry.name), blob = await handle.getFile();
      const owned = entry.name === job.journalName ? job._journal.handle : entry.handle;
      const same = typeof owned?.isSameEntry === "function" ? await owned.isSameEntry(handle) : owned === handle;
      if (!same) throw new Error("A staged Downloads file changed: " + entry.name + ".");
      if (entry.name === job.journalName) {
        const expected = job._journal.expected;
        if (expected === undefined || blob.size !== utf8.encode(expected).byteLength || await blob.text() !== expected) throw new Error("The staged Downloads journal changed.");
      } else if (!entry.complete || !entry.saved || blob.size !== entry.saved.size || blob.lastModified !== entry.saved.lastModified) throw new Error("A staged Downloads result changed: " + entry.name + ".");
      members.push({ name: job.outputStem + "/" + entry.name, blob });
    }
    if (members.length !== 5) throw new Error("A Downloads archive must contain exactly five job files.");
    const blob = await this.pack(members);
    const archive = { blob, url: this.createObjectURL(blob), name: job.outputStem + ".zip", size: blob.size,
      members: members.map(member => ({ name: member.name, size: member.blob.size })) };
    this._archives.set(job, { archive, workspace });
    return archive;
  }
  async reclaim(job) {
    const record = this._archives.get(job);
    if (!record || record.reclaimed) return;
    // Archive + URL are already published. Cache reclamation cannot change that result.
    await this._writeMarker(record.workspace);
    for (const entry of job.outputs) {
      let handle;
      try { handle = await job._folder.getFileHandle(entry.name); }
      catch (error) { if (error.name === "NotFoundError") continue; throw error; }
      const file = await handle.getFile();
      const owned = entry.name === job.journalName ? job._journal.handle : entry.handle;
      const same = typeof owned?.isSameEntry === "function" ? await owned.isSameEntry(handle) : owned === handle;
      if (!same) throw new Error("A staged Downloads file changed before cleanup.");
      if (entry.name === job.journalName) {
        if (file.size !== utf8.encode(job._journal.expected).byteLength || await file.text() !== job._journal.expected) throw new Error("The Downloads journal changed before cleanup.");
      } else if (file.size !== entry.saved.size || file.lastModified !== entry.saved.lastModified) throw new Error("A Downloads result changed before cleanup.");
      await job._folder.removeEntry(entry.name);
    }
    // Keep the empty job directory as its collision reservation until the queue is cleared.
    record.reclaimed = true;
  }
  _later(operation) {
    this.scheduleCleanup(() => Promise.resolve().then(operation).catch(() => {}), this.cleanupDelayMs);
  }
  _retire(record) {
    if (record.retired) return;
    record.retired = true;
    this._later(() => this.revokeObjectURL(record.archive.url));
  }
  release(job) {
    const record = this._archives.get(job);
    if (record) { this._retire(record); this._archives.delete(job); }
  }
  clear(jobs) {
    for (const job of jobs) this.release(job);
    const workspace = this._workspace;
    this._workspace = null; this._rootPromise = null;
    if (workspace) this._later(async () => {
      try {
        const marker = await readMarker(workspace.handle, workspace.id);
        if (!marker || marker.created !== workspace.created) return;
        try { await workspace.parent.removeEntry(workspace.name, { recursive: true }); }
        catch (error) { if (error.name !== "NotFoundError") throw error; }
      } finally { workspace.lock.release(); }
    });
  }
}
