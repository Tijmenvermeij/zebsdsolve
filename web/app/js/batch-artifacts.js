// Uses native engine snapshots and map pixels; no orientation or frame calculations.
import { exportH5, exportOxford } from "./hdf5-export.js";
const descriptor = (file) => ({ name: file.name, size: file.size, lastModified: file.lastModified });
async function png(rgba, scan) {
  const canvas = document.createElement("canvas"); canvas.width = scan.cols; canvas.height = scan.rows;
  canvas.getContext("2d").putImageData(new ImageData(rgba, scan.cols, scan.rows), 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("PNG map encoding failed.")), "image/png"));
}
export async function* batchArtifacts(engine, job, completed = new Set()) {
  const prefix = job.outputStem, oxford = /Oxford h5oina/i.test(job.scanInfo.format);
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
  name = prefix + "_settings.json";
  if (!completed.has(name)) yield { name, data: JSON.stringify({ zebsdsolve: engine.version, scan: job.sourceFiles[0].name,
    geometry_convention: oxford ? "oxford-calibrated-v1" : "generic-v1", geometry: job.geometry, settings: job.settings,
    source_files: job.sourceFiles.map(descriptor), master_files: job.masterFiles.map(descriptor), phases: job.phases,
    opening_hints: job.openHints, acquisition_geometry: job.openGeometry, scan_info: job.scanInfo,
    map_source: "native engine map: IPF-Z and detector NCC; same frame and colors as the GUI",
  }, null, 2) };
  for (const [kind, suffix] of [["ipf_z", "_IPF-Z.png"], ["ncc", "_NCC.png"]]) {
    name = prefix + suffix;
    if (!completed.has(name)) yield { name, data: await png(engine.map(kind), job.scanInfo) };
  }
}
