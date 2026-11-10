// HDF5 serialization only: orientations and coordinate transformations come from the engine.
// h5wasm 0.10.3 is vendored verbatim; see vendor/h5wasm/PROVENANCE.json and LICENSE.txt.
import { copyOxfordMetadata } from "./hdf5-metadata-copy.js";
let runtime;
async function hdf5() {
  runtime ||= import("./vendor/h5wasm/hdf5_hl.js").then(async (h) => { await h.ready; return h; });
  return runtime;
}
const TSL = [0, 1, 2, 22, 4, 42, 3, 32, 6, 62, 23, 43];
// Limit only the non-browser test fallback; browsers use random-access WORKERFS.
export const OXFORD_SOURCE_LIMIT = 256 * 1024 * 1024;
export function oxfordFrame(value) {
  const q = value?.engine_to_cs1;
  if (!Array.isArray(q) || q.length !== 4 || !q.every(Number.isFinite) || Math.abs(q.reduce((s, x) => s + x * x, 0) - 1) > 1e-6)
    throw new Error("Oxford frame requires a verified unit engine_to_cs1 quaternion [w,x,y,z].");
  if (typeof value.provenance !== "string" || !value.provenance.trim()) throw new Error("Oxford frame requires nonempty provenance.");
  const result = { ...value, engine_to_cs1: [...q], provenance: value.provenance.trim() };
  if (value.phase_id_map !== undefined) {
    if (!Array.isArray(value.phase_id_map) || !value.phase_id_map.every((x) => Number.isInteger(x) && x > 0) || new Set(value.phase_id_map).size !== value.phase_id_map.length)
      throw new Error("Oxford phase_id_map must contain unique positive source phase IDs in master order.");
    result.phase_id_map = [...value.phase_id_map];
  }
  return result;
}
function snapshot(s) {
  if (s?.schema_version !== 1) throw new Error("Unsupported HDF5 export snapshot version.");
  const n = s.scan.rows * s.scan.cols;
  if (!Number.isSafeInteger(n) || n <= 0 || s.results.phase.length !== n || s.results.euler.length !== n * 3 || s.results.quaternions.length !== n * 4)
    throw new Error("Incomplete HDF5 export snapshot.");
  if (!s.phases.length || s.phases.length > 255) throw new Error("HDF5 export supports 1–255 master phases.");
  if (s.phases.some((p) => !Number.isInteger(p.laue) || p.laue < 1 || p.laue > 11 || !Array.isArray(p.lattice) || p.lattice.length !== 6 || !p.lattice.every((v) => Number.isFinite(v) && v > 0))) throw new Error("HDF5 export requires complete phase lattice and Laue metadata.");
  return n;
}
function currentSnapshot(s) {
  const n = snapshot(s), results = structuredClone(s.results), candidates = structuredClone(s.candidates);
  const first = selected(s, n), indexed = first.map((v) => valid(v, s.phases.length));
  const clear = (values, i, width, fill) => { for (let k = 0; k < width; k++) values[i * width + k] = fill; };
  for (let i = 0; i < n; i++) if (!indexed[i]) for (const key of Object.keys(results)) clear(results[key], i, key === "quaternions" ? 4 : key === "euler" ? 3 : 1, key === "phase" || key === "sweep" ? -1 : 0);
  for (let phase = 0; phase < candidates.phase_count; phase++) for (let i = 0; i < n; i++) {
    const j = phase * n + i, v = { phase, euler: candidates.euler.slice(3*j, 3*j+3), q: candidates.quaternions.slice(4*j, 4*j+4), ncc: candidates.ncc[j], nmi: candidates.nmi[j], z: candidates.z[j], sweep: candidates.sweep[j] };
    if (!(v.sweep >= 0) || !valid(v, s.phases.length)) for (const key of Object.keys(candidates)) if (key !== "phase_count") clear(candidates[key], j, key === "quaternions" ? 4 : key === "euler" ? 3 : 1, key === "sweep" ? -1 : 0);
  }
  return { ...s, results, candidates };
}
function group(f, path) {
  let g = f;
  for (const part of path.split("/").filter(Boolean)) g = g.get(part) || g.create_group(part);
  return g;
}
function put(g, name, data, shape) { return g.create_dataset({ name, data, ...(shape ? { shape } : {}) }); }
function number(g, name, fallback = NaN) {
  const d = g?.get(name); if (!d) return fallback;
  const v = d.value; return Number(typeof v === "number" ? v : v?.[0] ?? v);
}
function text(g, name) {
  const v = g?.get(name)?.value;
  return typeof v === "string" ? v : String(v?.[0] ?? "");
}
function provenance(f, s, frame) {
  const g = group(f, "zebsdsolve");
  if (g.get("Producer")) throw new Error("Source already contains a zebsdsolve export; use the original acquisition file.");
  put(g, "Producer", `zebsdsolve ${s.version}`);
  put(g, "CandidateScope", "Best computed candidate per phase; independent of acceptance; no within-phase secondary peaks");
  put(g, "Frame", frame ? "Oxford CS1; q_out = conjugate(crystal_to_master) * q_engine * conjugate(engine_to_cs1)" : "ANG setting 2 compatible");
  put(g, "SettingsJSON", JSON.stringify(s.settings));
  put(g, "GeometryJSON", JSON.stringify(s.geometry));
  put(g, "ProvenanceJSON", JSON.stringify({ schema_version: 1, version: s.version, source_format: s.scan.format, units: { euler: "radians", lattice: "angstrom, degrees", map: "micrometres" }, ...(frame ? { oxford_frame: frame } : {}), ...(s.calibration ? { calibration: s.calibration } : {}) }));
  put(g, "PhasesJSON", JSON.stringify(s.phases));
  const selected = g.create_group("Selected"), n = s.scan.rows * s.scan.cols;
  for (const [key, width] of [["quaternions", 4], ["euler", 3], ["phase", 1], ["ncc", 1], ["nmi", 1], ["z", 1], ["sweep", 1]]) {
    if (s.results[key]?.length !== n * width) throw new Error(`Incomplete selected ${key} snapshot.`);
    put(selected, key, Float64Array.from(s.results[key], (v) => v ?? NaN), width === 1 ? [n] : [n, width]);
  }
  const c = s.candidates;
  if (c?.phase_count) {
    const n = s.scan.rows * s.scan.cols, cg = group(f, "zebsdsolve/Candidates");
    put(cg, "PhaseCount", new Uint32Array([c.phase_count]));
    for (const [name, width] of [["quaternions", 4], ["euler", 3], ["ncc", 1], ["nmi", 1], ["z", 1], ["sweep", 1]]) {
      if (c[name]?.length !== c.phase_count * n * width) throw new Error(`Incomplete ${name} candidate snapshot.`);
      put(cg, name, new Float64Array(c[name].map((x) => x ?? NaN)), width === 1 ? [c.phase_count, n] : [c.phase_count, n, width]);
    }
  }
  return g;
}
let serial = 0;
async function fileBytes(source, write) {
  const h = await hdf5(), { FS } = h;
  const path = `/zebsdsolve-export-${++serial}.h5`;
  let f;
  try {
    if (source) FS.writeFile(path, new Uint8Array(await source.arrayBuffer()));
    f = new h.File(path, source ? "a" : "w");
    write(f, h);
    f.flush(); f.close(); f = null;
    return FS.readFile(path);
  } finally {
    if (f) f.close();
    if (FS.analyzePath(path).exists) FS.unlink(path);
  }
}
function selected(s, n) {
  const r = s.results;
  return Array.from({ length: n }, (_, i) => ({ phase: r.phase[i], euler: r.euler.slice(i * 3, i * 3 + 3), q: r.quaternions.slice(i * 4, i * 4 + 4), ncc: r.ncc[i], nmi: r.nmi[i], z: r.z[i], sweep: r.sweep[i] }));
}
function valid(v, p) { return Number.isInteger(v.phase) && v.phase >= 0 && v.phase < p && v.euler.every(Number.isFinite) && v.q.every(Number.isFinite) && v.q.some((x) => x !== 0); }
function alternatives(s, n) {
  const c = s.candidates, p = s.phases.length;
  return Array.from({ length: n }, (_, i) => {
    const choices = [];
    for (let phase = 0; phase < (c?.phase_count || 0); phase++) {
      if (phase === s.results.phase[i]) continue;
      const j = phase * n + i;
      const v = { phase, euler: c.euler.slice(j * 3, j * 3 + 3), q: c.quaternions.slice(j * 4, j * 4 + 4), ncc: c.ncc[j], nmi: c.nmi[j], z: c.z[j], sweep: c.sweep[j] };
      if (v.sweep >= 0 && valid(v, p)) choices.push(v);
    }
    const metric = s.settings.phase_by === "ncc" ? "ncc" : "nmi";
    const score = (v) => Number.isFinite(v[metric]) ? v[metric] : -Infinity;
    choices.sort((a, b) => score(b) - score(a) || a.phase - b.phase);
    return choices;
  });
}
export async function exportH5(s) {
  s = currentSnapshot(s);
  const n = snapshot(s), first = selected(s, n), rest = alternatives(s, n);
  return fileBytes(null, (f) => {
    put(f, "Manufacturer", "EMSphInx"); put(f, "Version", String(s.version));
    provenance(f, s);
    const ranks = rest.reduce((r, v) => Math.max(r, v.length + 1), 1);
    for (let rank = 0; rank < ranks; rank++) {
      const g = group(f, `Scan ${rank + 1}/EBSD`), hd = g.create_group("Header"), d = g.create_group("Data"), geo = s.geometry;
      put(hd, "nRows", new Uint32Array([s.scan.rows])); put(hd, "nColumns", new Uint32Array([s.scan.cols]));
      put(hd, "Step X", new Float64Array([geo.step_x_um])); put(hd, "Step Y", new Float64Array([geo.step_y_um]));
      put(hd, "Grid Type", "SqrGrid"); put(hd, "GridType", "SqrGrid");
      put(hd, "Sample Tilt", new Float64Array([geo.sample_tilt_deg])); put(hd, "sampleTilt", new Float64Array([geo.sample_tilt_deg]));
      const pc = hd.create_group("Pattern Center Calibration");
      const aspect = s.scan.pat_w / s.scan.pat_h;
      put(pc, "x-star", new Float64Array([geo.pc[0]])); put(pc, "y-star", new Float64Array([(1 - geo.pc[1]) / aspect])); put(pc, "z-star", new Float64Array([geo.pc[2] / aspect]));
      const phases = hd.create_group("Phase");
      s.phases.forEach((p, i) => {
        const pg = phases.create_group(String(i + 1)); put(pg, "MaterialName", p.name); put(pg, "Symmetry", new Uint32Array([TSL[p.laue]]));
        put(pg, "Space Group", new Uint32Array([p.space_group]));
        ["a", "b", "c", "alpha", "beta", "gamma"].forEach((key, j) => put(pg, `Lattice Constant ${key}`, new Float64Array([p.lattice[j]])));
      });
      const values = Array.from({ length: n }, (_, i) => rank === 0 ? first[i] : rest[i][rank - 1]);
      const good = values.map((v) => v && valid(v, s.phases.length));
      put(d, "Phase", Uint8Array.from(values, (v, i) => good[i] ? v.phase : 255));
      for (const [name, j] of [["Phi1", 0], ["Phi", 1], ["Phi2", 2]]) put(d, name, Float64Array.from(values, (v, i) => good[i] ? v.euler[j] : 0));
      for (const [name, key] of [["Metric", "ncc"], ["IQ", "nmi"], ["CI", "ncc"], ["NMI", "nmi"], ["Z", "z"], ["Sweep", "sweep"]]) put(d, name, Float64Array.from(values, (v, i) => good[i] ? v[key] ?? NaN : key === "sweep" ? -1 : 0));
      put(d, "PixelID", Uint32Array.from({ length: n }, (_, i) => i));
      put(d, "X Position", Float64Array.from({ length: n }, (_, i) => i % s.scan.cols * geo.step_x_um));
      put(d, "Y Position", Float64Array.from({ length: n }, (_, i) => Math.floor(i / s.scan.cols) * geo.step_y_um));
      put(d, "Quaternion", Float64Array.from(values.flatMap((v, i) => good[i] ? v.q : [0, 0, 0, 0])), [n, 4]);
    }
  });
}
function unit(dataset) {
  const value = dataset?.attrs?.Unit?.value;
  return String(typeof value === "string" ? value : value?.[0] ?? "").trim().toLowerCase();
}
function lattice(pg) {
  const dims = pg.get("Lattice Dimensions"), angles = pg.get("Lattice Angles");
  const du = unit(dims), au = unit(angles);
  const ds = ({ angstrom: 1, angstroms: 1, "å": 1, nm: 10 })[du];
  const as = ({ rad: 180 / Math.PI, radians: 180 / Math.PI, deg: 1, degrees: 1 })[au];
  if (!ds || !as || dims?.value?.length !== 3 || angles?.value?.length !== 3) return null;
  return [...Array.from(dims.value, (v) => v * ds), ...Array.from(angles.value, (v) => v * as)];
}
function sameLattice(a, b) { return a && b?.length === 6 && a.every((v, i) => Number.isFinite(v) && Math.abs(v - b[i]) <= 1e-4 + 1e-5 * Math.abs(b[i])); }
function phaseMapping(pg, s, frame) {
  const ids = pg.keys().filter((id) => /^\d+$/.test(id) && Number(id) > 0).map(Number);
  if (frame.phase_id_map && frame.phase_id_map.length !== s.phases.length) throw new Error("Oxford phase_id_map length must equal the number of master phases.");
  const used = new Set(ids), pending = [], map = s.phases.map((p, i) => {
    const matches = (id, explicit) => {
      const source = pg.get(String(id)); if (!source) return false;
      const sg = number(source, "Space Group"), laue = number(source, "Laue Group");
      return ((sg > 0 && sg === p.space_group) || (explicit && sg === 0 && laue === p.laue)) && sameLattice(lattice(source), p.lattice);
    };
    if (frame.phase_id_map) {
      const id = frame.phase_id_map[i];
      if (!ids.includes(id) || !matches(id, true)) throw new Error(`Oxford source phase ${id} does not match master ${p.name} (space/Laue group, lattice and units).`);
      return id;
    }
    const matching = ids.filter((id) => matches(id, false));
    if (matching.length > 1) throw new Error(`Ambiguous Oxford phase for ${p.name}; supply an explicit phase_id_map.`);
    if (matching.length === 1) return matching[0];
    let id = Math.max(0, ...used) + 1;
    used.add(id); pending.push([id, p]); return id;
  });
  if (new Set(map).size !== map.length) throw new Error("Multiple master phases map to one Oxford phase; provide distinct phase_id_map entries.");
  return { map, pending };
}
function copyDataset(g, name, d) {
  const result = g.create_dataset({ name, data: d.value, shape: d.shape, dtype: d.dtype });
  for (const [key, a] of Object.entries(d.attrs)) result.create_attribute(key, a.value, a.shape, a.dtype);
  return result;
}
const INDEX_METRICS = ["Mean Angular Deviation", "MAD", "Mean Angular Error", "Error", "Bands", "Band Count", "Number of Bands", "Bands Detected", "Bands Indexed", "Confidence Index", "Fit", "Pattern Fit"];
const PC_FIELDS = ["Pattern Center X", "Pattern Center Y", "Detector Distance"];
export async function exportOxford(s, source, frameValue) {
  if (!source || !Number.isFinite(source.size)) throw new Error("Oxford export requires the original H5OINA acquisition file.");
  const frame = oxfordFrame(frameValue);
  snapshot(s);
  if (!/oxford|h5oina/i.test(s.scan.format)) throw new Error("Oxford export requires an original Oxford H5OINA scan.");
  if (typeof Worker !== "undefined") {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL("./hdf5-export-worker.js", import.meta.url), { type: "module" });
      const finish = (error, bytes) => { worker.terminate(); error ? reject(error) : resolve(bytes); };
      worker.onmessage = ({ data }) => finish(data.error ? new Error(data.error) : null, data.bytes);
      worker.onerror = (event) => { event.preventDefault(); finish(new Error(event.message || "Oxford HDF5 export worker failed.")); };
      worker.onmessageerror = () => finish(new Error("Oxford HDF5 export worker returned an unreadable response."));
      try { worker.postMessage({ source, snapshot: s, frame }); } catch (error) { finish(error); }
    });
  }
  // Node validation uses MEMFS for small synthetic fixtures. Production browser
  // exports must have a worker, never a whole-acquisition arrayBuffer fallback.
  if (typeof process === "undefined" || !process.versions?.node) throw new Error("Oxford export requires browser Web Worker support.");
  if (source.size > OXFORD_SOURCE_LIMIT) throw new Error("Oxford test fallback is limited to 256 MiB; browser export requires a Web Worker.");
  const h = await hdf5(), path = `/zebsdsolve-source-${++serial}.h5`;
  try {
    h.FS.writeFile(path, new Uint8Array(await source.arrayBuffer()));
    return await exportOxfordFromPath(s, path, frame);
  } finally {
    if (h.FS.analyzePath(path).exists) h.FS.unlink(path);
  }
}
// The browser worker passes a WORKERFS path; exposed separately for real-runtime
// Node checks without substituting a mock HDF5 implementation.
export async function exportOxfordFromPath(s, sourcePath, frameValue) {
  s = currentSnapshot(s);
  const frame = oxfordFrame(frameValue), n = snapshot(s), h = await hdf5();
  const source = new h.File(sourcePath, "r");
  try {
    return await fileBytes(null, (f, h) => {
      copyOxfordMetadata(source, f, h);
      writeOxford(f, h, s, frame, n);
    });
  } finally { source.close(); }
}
function writeOxford(f, h, s, frame, n) {
    const roots = f.keys().filter((key) => {
      if (f.get_link(`/${key}`) !== null || f.get_external_link(`/${key}`).filename !== undefined) return false;
      const root = f.get(key); if (!(root instanceof h.Group)) return false;
      const ebsd = root.get("EBSD"); return ebsd instanceof h.Group && ebsd.keys().includes("Header") && ebsd.keys().includes("Data");
    });
    if (!roots.length) throw new Error("Source has no Oxford EBSD scan.");
    // Native scan selection opens the first acquisition scan; no fitted spatial alignment.
    const path = `${roots[0]}/EBSD`, hd = f.get(`${path}/Header`), data = f.get(`${path}/Data`);
    if (number(hd, "X Cells") !== s.scan.cols || number(hd, "Y Cells") !== s.scan.rows) throw new Error("Oxford source grid differs from the indexed map.");
    for (const [key, value] of [["Pattern Height", s.scan.pat_h], ["Pattern Width", s.scan.pat_w]]) if (hd.get(key) && number(hd, key) !== value) throw new Error("Oxford source detector dimensions differ from the indexed patterns.");
    const euler = data.get("Euler"), phase = data.get("Phase"), x = data.get("X"), y = data.get("Y");
    if (!(euler instanceof h.Dataset) || !(phase instanceof h.Dataset) || euler.shape?.join() !== `${n},3` || phase.shape?.join() !== `${n}` || !/^Float(32|64)Array$/.test(euler.value?.constructor?.name) || !/^(Uint(8|16|32)|Int(8|16|32)|Uint8Clamped)Array$/.test(phase.value?.constructor?.name))
      throw new Error("Oxford Euler/Phase layout is unsupported; use Python export.");
    if (!["rad", "radians"].includes(unit(euler))) throw new Error("Oxford Euler angles must explicitly declare radians.");
    if (x?.shape?.join() !== `${n}` || y?.shape?.join() !== `${n}` || !["um", "µm", "micrometres", "micrometers"].includes(unit(x)) || !["um", "µm", "micrometres", "micrometers"].includes(unit(y))) throw new Error("Oxford pixel coordinates and micrometre units are required to verify raster order.");
    const xs = x.value, ys = y.value, sx = number(hd, "X Step"), sy = number(hd, "Y Step");
    for (const key of ["X Step", "Y Step"]) if (!["um", "µm", "micrometres", "micrometers"].includes(unit(hd.get(key)))) throw new Error("Oxford source raster steps must declare micrometre units.");
    if (!(sx > 0 && sy > 0)) throw new Error("Oxford source raster steps are invalid.");
    if (Math.abs(sx - s.geometry.step_x_um) > Math.max(1e-8, sx * 1e-5) || Math.abs(sy - s.geometry.step_y_um) > Math.max(1e-8, sy * 1e-5)) throw new Error("Oxford source steps differ from the current geometry; use Python export with an explicit raster mapping.");
    if (s.pattern_centres?.length !== 3 * n || !s.pattern_centres.every(Number.isFinite)) throw new Error("Oxford export requires native per-pixel pattern centres.");
    for (let i = 0; i < n; i++) {
      const ex = xs[0] + (i % s.scan.cols) * sx, ey = ys[0] + Math.floor(i / s.scan.cols) * sy;
      if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i]) || Math.abs(xs[i] - ex) > Math.max(1e-5, sx * 1e-3) || Math.abs(ys[i] - ey) > Math.max(1e-5, sy * 1e-3))
        throw new Error("Oxford source pixels do not match the native row-major scan order.");
    }
    const pg = hd.get("Phases"); if (!(pg instanceof h.Group)) throw new Error("Oxford source phase metadata is missing.");
    const { map, pending } = phaseMapping(pg, s, frame);
    const phaseValues = phase.value, eulerValues = euler.value;
    const maxID = phaseValues instanceof Uint8Array ? 255 : phaseValues instanceof Int8Array ? 127 : phaseValues instanceof Uint16Array ? 65535 : phaseValues instanceof Int16Array ? 32767 : phaseValues instanceof Int32Array ? 2147483647 : 4294967295;
    if (map.some((id) => id > maxID)) throw new Error("New Oxford phase IDs exceed the source phase dataset type; use Python export.");
    const selectedValues = selected(s, n), indexed = Uint8Array.from(selectedValues, (v) => valid(v, s.phases.length) ? 1 : 0), replaced = new Uint8Array(n).fill(1);
    const z = provenance(f, s, frame);
    for (const [name, field, count] of [["Detector Orientation Euler", "effective_detector_euler_radians", 3], ["Tilt Angle", "effective_tilt_radians", 1]]) {
      const effective = s.calibration?.[field];
      if (effective === undefined) continue;
      const values = Array.isArray(effective) ? effective : [effective], d = hd.get(name);
      if (values.length !== count || !values.every(Number.isFinite)) throw new Error(`Invalid native effective Oxford calibration: ${field}.`);
      if (!(d instanceof h.Dataset) || d.value.length !== count || !/^Float(32|64)Array$/.test(d.value?.constructor?.name) || !["rad", "radians"].includes(unit(d))) throw new Error(`Unsupported Oxford calibration layout or units: ${name}.`);
      copyDataset(group(f, "zebsdsolve/OriginalCalibration"), name, d);
      d.write_slice(d.shape.map((size) => [0, size]), new d.value.constructor(values));
    }
    for (const [id, p] of pending) {
      const g = pg.create_group(String(id));
      put(g, "Phase Name", [p.name]); put(g, "Reference", ["zebsdsolve master pattern"]); put(g, "Space Group", new Uint32Array([p.space_group])); put(g, "Laue Group", new Uint32Array([p.laue]));
      put(g, "Lattice Dimensions", new Float64Array(p.lattice.slice(0, 3)), [1, 3]).create_attribute("Unit", "angstrom");
      put(g, "Lattice Angles", Float64Array.from(p.lattice.slice(3), (v) => v * Math.PI / 180), [1, 3]).create_attribute("Unit", "rad");
    }
    for (let i = 0; i < n; i++) {
      phaseValues[i] = indexed[i] ? map[s.results.phase[i]] : 0;
      for (let j = 0; j < 3; j++) eulerValues[3 * i + j] = indexed[i] ? s.results.euler[3 * i + j] : 0;
    }
    euler.write_slice([[0, n], [0, 3]], eulerValues); phase.write_slice([[0, n]], phaseValues);
    for (let j = 0; j < PC_FIELDS.length; j++) {
      const name = PC_FIELDS[j], d = data.get(name);
      if (d && (d.shape?.join() !== `${n}` || !/^Float(32|64)Array$/.test(d.value?.constructor?.name))) throw new Error(`Unsupported Oxford ${name} layout; use Python export.`);
      const values = d ? d.value : new Float64Array(n).fill(NaN);
      for (let i = 0; i < n; i++) {
        const v = s.pattern_centres[3 * i + j], ratio = s.scan.pat_h / s.scan.pat_w;
        values[i] = indexed[i] ? (j === 0 ? v : (j === 1 ? 1 - v : v) * ratio) : NaN;
      }
      if (d) d.write_slice([[0, n]], values);
      else put(data, name, values).create_attribute("Unit", "fraction");
    }
    for (const name of INDEX_METRICS) {
      const d = data.get(name); if (!(d instanceof h.Dataset)) continue;
      if (d.shape?.[0] !== n) throw new Error(`Unsupported Oxford indexing metric ${name} layout; use Python export.`);
      const values = d.value;
      if (!ArrayBuffer.isView(values) || values instanceof DataView || values instanceof BigInt64Array || values instanceof BigUint64Array) throw new Error(`Unsupported Oxford indexing metric ${name}; use Python export.`);
      const width = values.length / n;
      for (let i = 0; i < n; i++) if (replaced[i]) values.fill(values instanceof Float32Array || values instanceof Float64Array ? NaN : 0, i * width, (i + 1) * width);
      d.write_slice(d.shape.map((size) => [0, size]), values);
    }
    // Solver metrics live under provenance, without relabelling NCC as Oxford MAD.
    put(z, "IndexedMask", indexed); put(z, "ReplacedMask", replaced); put(z, "PhaseIDMap", new Uint32Array(map));
    for (const name of ["ncc", "nmi", "z", "sweep"]) put(z, name.toUpperCase(), Float64Array.from(s.results[name], (v, i) => indexed[i] ? v ?? NaN : NaN));
}
