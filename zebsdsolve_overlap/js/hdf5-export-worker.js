// WORKERFS reads Blob/File slices synchronously on demand; acquisition patterns
// never enter MEMFS or the JavaScript heap. Only the pattern-free output is built.
import * as h from "./vendor/h5wasm/hdf5_hl.js";
import { exportOxfordFromPath } from "./hdf5-export.js";
self.onmessage = async ({ data }) => {
  let mounted = false;
  try {
    await h.ready;
    h.FS.mkdir("/acquisition");
    h.FS.mount(h.FS.filesystems.WORKERFS, { blobs: [{ name: "source.h5oina", data: data.source }] }, "/acquisition");
    mounted = true;
    const bytes = await exportOxfordFromPath(data.snapshot, "/acquisition/source.h5oina", data.frame);
    self.postMessage({ bytes }, [bytes.buffer]);
  } catch (error) {
    self.postMessage({ error: error?.message || String(error) });
  } finally {
    if (mounted) h.FS.unmount("/acquisition");
    if (h.FS?.analyzePath("/acquisition").exists) h.FS.rmdir("/acquisition");
    self.close();
  }
};
