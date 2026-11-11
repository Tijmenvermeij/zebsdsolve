import { isOxfordH5 } from "./oxford-export-policy.js";
// Uses native engine snapshots and map pixels; no orientation or frame calculations.
import { exportH5, exportOxford } from "./hdf5-export.js";
// putImageData copies the native RGBA raster at scan resolution without filtering.
async function mapPNG(engine, kind) {
  const { rows, cols } = engine.scan;
  const pixels = engine.map(kind);
  if (!Number.isSafeInteger(rows) || rows <= 0 || !Number.isSafeInteger(cols) || cols <= 0 || pixels.length !== rows * cols * 4) throw new Error("Invalid native map dimensions.");
  const canvas = typeof OffscreenCanvas === "function" ? new OffscreenCanvas(cols, rows) : Object.assign(document.createElement("canvas"), { width: cols, height: rows });
  const context = canvas.getContext("2d");
  if (!context) throw new Error("PNG map canvas is unavailable.");
  context.putImageData(new ImageData(new Uint8ClampedArray(pixels), cols, rows), 0, 0);
  if (canvas.convertToBlob) return canvas.convertToBlob({ type: "image/png" });
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("PNG map encoding failed.")), "image/png"));
}
export async function* batchArtifacts(engine, job, completed = new Set()) {
  const prefix = job.outputStem, oxford = isOxfordH5(job.scanInfo.format);
  let name = prefix + ".h5";
  if (!completed.has(name)) {
    const snapshot = engine.exportSnapshot();
    snapshot.maps = { ipf_z: engine.map("ipf_z"), ncc: engine.map("ncc") };
    yield { name, data: await exportH5(snapshot) };
  }
  name = prefix + (oxford ? ".h5oina" : ".ang");
  if (!completed.has(name)) {
    if (oxford) { const snapshot = engine.exportOxfordSnapshot(); yield { name, data: await exportOxford(snapshot, job.sourceFiles[0], snapshot.oxford_frame) }; }
    else yield { name, data: engine.exportText("ang") };
  }
  for (const [kind, suffix] of [["ipf_z", "_IPF-Z.png"], ["ncc", "_NCC.png"]]) {
    name = prefix + suffix;
    if (!completed.has(name)) yield { name, data: await mapPNG(engine, kind) };
  }
}
