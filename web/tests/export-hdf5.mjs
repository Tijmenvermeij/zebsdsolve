// Run with Node 22+: node --experimental-default-type=module web/tests/export-hdf5.mjs
// Uses the same vendored browser HDF5 runtime. No GPU, network, Python or custom HDF5 code.
import assert from "node:assert/strict";
import * as h from "../app/js/vendor/h5wasm/hdf5_hl.js";
import { exportH5, exportOxford, oxfordFrame, OXFORD_SOURCE_LIMIT } from "../app/js/hdf5-export.js";
await h.ready;
const n = 4;
const s = {
  schema_version: 1, version: "export-test", scan: { rows: 2, cols: 2, pat_h: 8, pat_w: 10, format: "h5_oxford" },
  geometry: { pc: [.4, .5, .6], step_x_um: 1, step_y_um: 2, sample_tilt_deg: 70 }, settings: { phase_by: "nmi" },
  phases: [{ name: "Fe", space_group: 229, laue: 11, lattice: [3, 3, 3, 90, 90, 90] }, { name: "Ni", space_group: 225, laue: 11, lattice: [3.5, 3.5, 3.5, 90, 90, 90] }],
  results: { quaternions: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], euler: [.1, .2, .3, 0, 0, 0, .2, .4, .6, 0, 0, 0], phase: [0, -1, 1, -1], ncc: [.8, 0, .9, 0], nmi: [.5, 0, .7, 0], z: [12, 0, 15, 0], sweep: [0, -1, 1, -1] },
  candidates: { phase_count: 2, quaternions: Array.from({ length: 8 }, () => [1, 0, 0, 0]).flat(), euler: new Array(24).fill(0), ncc: [.8, .6, .4, 0, .7, 0, .9, 0], nmi: [.5, .3, .2, 0, .4, 0, .7, 0], z: [12, 9, 8, 0, 10, 0, 15, 0], sweep: [0, 0, 0, -1, 0, -1, 1, -1] },
  pattern_centres: Array.from({ length: n }, () => [.4, .5, .6]).flat(),
};
function put(g, name, data, shape) { return g.create_dataset({ name, data, ...(shape ? { shape } : {}) }); }
function read(bytes, check) {
  h.FS.writeFile("/check.h5", bytes); const f = new h.File("/check.h5");
  try { check(f); } finally { f.close(); h.FS.unlink("/check.h5"); }
}
const bytes = await exportH5(s);
assert.deepEqual(Array.from(bytes.slice(0, 8)), [137, 72, 68, 70, 13, 10, 26, 10]);
read(bytes, (f) => {
  assert.equal(f.get("Manufacturer").value, "EMSphInx");
  assert.deepEqual(Array.from(f.get("Scan 1/EBSD/Data/Phase").value), [0, 255, 1, 255]);
  assert.deepEqual(Array.from(f.get("Scan 2/EBSD/Data/Phase").value), [1, 0, 0, 255]);
  assert.equal(f.get("Scan 1/EBSD/Header/Phase/2/Symmetry").value[0], 43);
  assert.equal(f.get("Scan 1/EBSD/Header/Phase/2/Lattice Constant a").value[0], 3.5);
  assert.deepEqual(f.get("zebsdsolve/Candidates/euler").shape, [2, 4, 3]);
});
const frame = { engine_to_cs1: [1, 0, 0, 0], provenance: "Independent test fixture reference" };
// Indexed-but-unaccepted results belong in both ANG and HDF5.
const unaccepted = structuredClone(s); unaccepted.results.sweep[0] = -1;
read(await exportH5(unaccepted), (f) => assert.equal(f.get('Scan 1/EBSD/Data/Phase').value[0], s.results.phase[0]));
const missingMetrics = structuredClone(s);
missingMetrics.results.sweep[0] = -1;
missingMetrics.results.ncc[0] = null; missingMetrics.results.nmi[0] = -10; missingMetrics.results.z[0] = null;
missingMetrics.candidates.ncc[4] = null; missingMetrics.candidates.nmi[4] = null; missingMetrics.candidates.z[4] = null;
read(await exportH5(missingMetrics), (f) => {
  assert.equal(f.get("Scan 1/EBSD/Data/Phase").value[0], 0);
  assert.equal(f.get("Scan 1/EBSD/Data/Sweep").value[0], -1);
  assert.ok(Number.isNaN(f.get("Scan 1/EBSD/Data/CI").value[0]));
  assert.equal(f.get("Scan 2/EBSD/Data/Phase").value[0], 1);
  assert.ok(Number.isNaN(f.get("zebsdsolve/Selected/ncc").value[0]));
  assert.ok(Number.isNaN(f.get("zebsdsolve/Candidates/ncc").value[4]));
});
const candidatesOnly = structuredClone(s);
candidatesOnly.results.phase[0] = -1;
read(await exportH5(candidatesOnly), (f) => {
  assert.equal(f.get("Scan 1/EBSD/Data/Phase").value[0], 255);
  assert.equal(f.get("Scan 2/EBSD/Data/Phase").value[0], 0);
  assert.equal(f.get("Scan 3/EBSD/Data/Phase").value[0], 1);
  assert.deepEqual(Array.from(f.get("zebsdsolve/Selected/quaternions").value.slice(0,4)), [0,0,0,0]);
});
assert.throws(() => oxfordFrame({ engine_to_cs1: [1, 0, 0, 0] }), /provenance/);
assert.throws(() => oxfordFrame({ engine_to_cs1: [2, 0, 0, 0], provenance: "bad" }), /unit/);
let didRead = false;
await assert.rejects(() => exportOxford(s, { size: OXFORD_SOURCE_LIMIT + 1, arrayBuffer() { didRead = true; } }, frame), /256 MiB/);
assert.equal(didRead, false);
// A resolvable external link must stay external, even when its target has an
// unsupported datatype that would fail a local metadata copy.
const external = new h.File("/external.h5", "w"), vendor = external.create_group("vendor");
const externalValue = put(vendor, "value", new Uint8Array([9]));
vendor.create_dataset({ name: "data", data: [externalValue.create_reference()], dtype: "Reference" }); external.close();
const f = new h.File("/source.h5oina", "w");
put(f, "Manufacturer", "Oxford Instruments"); const root = f.create_group("1"), eb = root.create_group("EBSD"), hd = eb.create_group("Header"), d = eb.create_group("Data");
for (const [key, value] of [["X Cells", 2], ["Y Cells", 2], ["Pattern Width", 10], ["Pattern Height", 8]]) put(hd, key, new Uint32Array([value]));
for (const [key, value] of [["X Step", 1], ["Y Step", 2]]) put(hd, key, new Float64Array([value])).create_attribute("Unit", "um");
const pg = hd.create_group("Phases"), fe = pg.create_group("1");
put(fe, "Space Group", new Uint32Array([229])); put(fe, "Laue Group", new Uint32Array([11])); put(fe, "Phase Name", ["Fe"]);
put(fe, "Lattice Dimensions", new Float64Array([3, 3, 3]), [1, 3]).create_attribute("Unit", "angstrom");
put(fe, "Lattice Angles", new Float64Array([Math.PI / 2, Math.PI / 2, Math.PI / 2]), [1, 3]).create_attribute("Unit", "rad");
put(d, "Euler", new Float64Array(n * 3).fill(.8), [n, 3]).create_attribute("Unit", "rad");
put(d, "Phase", new Uint8Array(n).fill(1)); put(d, "Mean Angular Deviation", new Float32Array(n).fill(.2)); put(d, "Bands", new Uint8Array(n).fill(7));
put(d, "X", new Float64Array([0, 1, 0, 1])).create_attribute("Unit", "um"); put(d, "Y", new Float64Array([0, 0, 2, 2])).create_attribute("Unit", "um");
put(d, "Pattern Center X", new Float64Array(n).fill(.2));
put(root, "UnknownVendorData", new Float64Array([5, 6])).create_attribute("VendorUnit", "opaque");
root.create_attribute("Acquisition", "unknown vendor metadata");
put(root, "Integer64", new BigUint64Array([2n ** 60n + 3n]));
root.create_dataset({ name: "FixedASCII", data: ["vendor"], dtype: "A16" });
put(d, "Processed Patterns", new Uint8Array(n * 8 * 10).fill(44), [n, 8, 10]);
put(d, "Unprocessed Patterns", new Uint8Array(n * 8 * 10).fill(33), [n, 8, 10]);
put(d, "Raw Patterns", new Uint8Array(n * 8 * 10).fill(22), [n, 8, 10]);
put(d, "Processed Static Background", new Uint16Array(8 * 10).fill(17), [8, 10]).create_attribute("Unit", "counts");
f.create_hard_link("/1/EBSD/Data/Processed Patterns", "0PatternAlias");
f.create_soft_link("/1/EBSD/Data/Processed Patterns", "SoftPatternAlias");
f.create_soft_link("/1/UnknownVendorData", "VendorSoftLink");
f.create_soft_link("/missing/vendor", "BrokenSoftLink");
f.create_external_link("external.h5", "/vendor/data", "VendorExternalLink");
root.create_hard_link("/1", "GroupCycle");
f.create_hard_link("/1/UnknownVendorData", "VendorHardLink"); f.close();
const sourceBytes = h.FS.readFile("/source.h5oina").slice(); h.FS.unlink("/source.h5oina"); const source = new Blob([sourceBytes]);
const ox = await exportOxford(s, source, frame);
read(ox, (f) => {
  assert.deepEqual(Array.from(f.get("1/EBSD/Data/Phase").value), [1, 0, 2, 0]);
  assert.equal(f.get("1/EBSD/Header/Phases/2/Space Group").value[0], 225);
  assert.deepEqual(f.get("1/EBSD/Header/Phases/2/Reference").value, ["zebsdsolve master pattern"]);
  assert.equal(f.get("zebsdsolve/OriginalIndexing"), null);
  assert.equal(f.get("1/EBSD/Data/Euler").value[9], 0);
  assert.equal(f.get("1/EBSD/Data/Phase").value[3], 0);
  assert.deepEqual(Array.from(f.get("1/EBSD/Data/Bands").value), [0, 0, 0, 0]);
  assert.ok(Number.isNaN(f.get("1/EBSD/Data/Mean Angular Deviation").value[0]));
  assert.equal(f.get("1/UnknownVendorData").attrs.VendorUnit.value, "opaque");
  assert.equal(f.get("1").attrs.Acquisition.value, "unknown vendor metadata");
  assert.deepEqual(Array.from(f.get("1/Integer64").value), [2n ** 60n + 3n]);
  assert.equal(f.get("1/FixedASCII").dtype, "A16");
  assert.deepEqual(f.get("1/FixedASCII").value, ["vendor"]);
  for (const name of ["Processed Patterns", "Unprocessed Patterns", "Raw Patterns"]) assert.equal(f.get(`1/EBSD/Data/${name}`), null);
  assert.equal(f.get("0PatternAlias"), null); assert.equal(f.get("SoftPatternAlias"), null);
  assert.equal(f.get("1/EBSD/Data/Processed Static Background").value[0], 17);
  assert.equal(f.get("1/EBSD/Data/Processed Static Background").attrs.Unit.value, "counts");
  assert.deepEqual(f.get("VendorHardLink").create_reference().ref_data, f.get("1/UnknownVendorData").create_reference().ref_data);
  assert.deepEqual(f.get("1/GroupCycle").create_reference().ref_data, f.get("1").create_reference().ref_data);
  assert.equal(f.get_link("/VendorSoftLink"), "/1/UnknownVendorData");
  assert.equal(f.get_link("/BrokenSoftLink"), "/missing/vendor");
  assert.deepEqual(f.get_external_link("/VendorExternalLink"), { filename: "external.h5", obj_path: "/vendor/data" });
  assert.deepEqual(Array.from(f.get("VendorHardLink").value), [5, 6]);
  assert.equal(f.get("1/EBSD/Data/Pattern Center X").value[0], .4);
  assert.ok(Number.isNaN(f.get("1/EBSD/Data/Pattern Center X").value[3]));
  assert.ok(Math.abs(f.get("1/EBSD/Data/Pattern Center Y").value[0] - .4) < 1e-12);
  assert.ok(Number.isNaN(f.get("1/EBSD/Data/Pattern Center Y").value[3]));
});
read(await exportOxford(missingMetrics, source, frame), (f) => {
  assert.equal(f.get("1/EBSD/Data/Phase").value[0], 1);
  assert.equal(f.get("zebsdsolve/SWEEP").value[0], -1);
  assert.ok(Number.isNaN(f.get("zebsdsolve/NCC").value[0]));
  assert.deepEqual(Array.from(f.get("zebsdsolve/ReplacedMask").value), [1,1,1,1]);
});
assert.deepEqual(new Uint8Array(await source.arrayBuffer()), sourceBytes);
await assert.rejects(() => exportOxford(s, source, { ...frame, phase_id_map: [1, 1] }), /unique/);
await assert.rejects(() => exportOxford({ ...s, phases: [s.phases[1]] }, source, { ...frame, phase_id_map: [1] }), /does not match/);
await assert.rejects(() => exportOxford({ ...s, geometry: { ...s.geometry, step_x_um: 9 } }, source, frame), /steps differ/);
h.FS.writeFile("/processing.h5oina", sourceBytes);
const dp = new h.File("/processing.h5oina", "a"), processing = dp.get("1").create_group("Data Processing");
put(processing, "OriginalFit", new Float32Array([.1]));
// Processing may alias the live acquisition Data group. Its canonical fields
// must survive processing omission and receive solver results.
processing.create_hard_link("/1/EBSD/Data", "SharedData");
dp.create_soft_link("/1/Data Processing/OriginalFit", "ProcessingSoftLink");
dp.create_soft_link("/2/Data Processing/OriginalFit", "AliasedProcessingSoftLink");
processing.create_soft_link("../EBSD/Data/Euler", "BrokenParentLink");
processing.create_soft_link("/1/EBSD/Data/Phase", "AcquisitionPhaseLink");
const oldMap = processing.create_group("Data"); put(oldMap, "Euler", new Float32Array([.4,.5,.6]), [1,3]);
processing.create_soft_link("Data/Euler", "RelativeEulerLink");
processing.create_soft_link("OriginalFit", "RelativeLink"); dp.create_hard_link("/1", "2"); dp.close();
const dpSource = new Blob([h.FS.readFile("/processing.h5oina").slice()]); h.FS.unlink("/processing.h5oina");
read(await exportOxford(s, dpSource, frame), (f) => {
  assert.deepEqual(Array.from(f.get("1/EBSD/Data/Phase").value), [1,0,2,0]);
  assert.deepEqual(Array.from(f.get("1/EBSD/Data/X").value), [0,1,0,1]);
  assert.equal(f.get("1/Data Processing"), null);
  assert.equal(f.get("2/Data Processing"), null);
  assert.deepEqual(f.get("1").create_reference().ref_data, f.get("2").create_reference().ref_data);
  assert.equal(f.get("zebsdsolve/OriginalDataProcessing"), null);
  assert.equal(f.get("ProcessingSoftLink"), null);
  assert.equal(f.get("AliasedProcessingSoftLink"), null);

});
h.FS.writeFile("/references.h5oina", sourceBytes);
const references = new h.File("/references.h5oina", "a");
references.create_dataset({ name: "References", data: [references.get("1/UnknownVendorData").create_reference()], dtype: "Reference" }); references.close();
const refSource = new Blob([h.FS.readFile("/references.h5oina")]); h.FS.unlink("/references.h5oina");
await assert.rejects(() => exportOxford(s, refSource, frame), /unsupported HDF5 datatype.*references/);
// Structural discovery must reject links before HDF5 can open an external
// ancestor. The available external file above contains an unsupported reference.
for (const location of ["zebsdsolve", "EBSD", "Data Processing", "soft-zebsdsolve"]) {
  const path = "/structural.h5oina";
  if (location !== "EBSD") h.FS.writeFile(path, sourceBytes);
  const structural = new h.File(path, location === "EBSD" ? "w" : "a");
  if (location === "EBSD") structural.create_group("1").create_external_link("external.h5", "/vendor", "EBSD");
  else if (location === "Data Processing") structural.get("1").create_external_link("external.h5", "/vendor", location);
  else if (location === "soft-zebsdsolve") structural.create_soft_link("/1", "zebsdsolve");
  else structural.create_external_link("external.h5", "/vendor", location);
  structural.close();
  const linkedSource = new Blob([h.FS.readFile(path)]); h.FS.unlink(path);
  const fsOpen = h.FS.open; let externalOpens = 0;
  h.FS.open = function (path, ...args) {
    if (typeof path === "string" && path.split("/").pop() === "external.h5") {
      externalOpens++; throw new Error("Structural discovery opened an external target");
    }
    return fsOpen.call(this, path, ...args);
  };
  try {
    await assert.rejects(() => exportOxford(s, linkedSource, frame), /structural metadata.*soft\/external structural links are unsupported/);
    assert.equal(externalOpens, 0, `${location} must not open its external target`);
  } finally { h.FS.open = fsOpen; }
}
// Active planes preserve every source PC field, including unindexed pixels.
for (const plane_kind of [1, 2]) {
  const fitted = structuredClone(s); fitted.geometry.plane_kind = plane_kind;
  const warnings = [], originalWarn = console.warn;
  console.warn = (message) => warnings.push(message);
  let output;
  try { output = await exportOxford(fitted, source, frame); }
  finally { console.warn = originalWarn; }
  assert.equal(warnings.length, 1);
  read(output, (f) => {
    assert.deepEqual(Array.from(f.get("1/EBSD/Data/Pattern Center X").value), [.2,.2,.2,.2]);
    assert.equal(f.get("1/EBSD/Data/Pattern Center X").dtype, "<d");
    assert.equal(f.get("1/EBSD/Data/Pattern Center Y"), null);
    assert.equal(f.get("1/EBSD/Data/Detector Distance"), null);
    assert.deepEqual(Array.from(f.get("zebsdsolve/PatternCentres").value), fitted.pattern_centres);
    const policy = JSON.parse(f.get("zebsdsolve/ProvenanceJSON").value).oxford_frame;
    assert.equal(policy.preserve_source_pattern_centres, true);
    assert.deepEqual(policy.warnings, warnings);
  });
}
h.FS.unlink("/external.h5");
console.log("HDF5 browser export checks passed: EMS selected/alternative phases, Oxford phase append, current solutions/metadata attributes/links, ROI, pattern centres, pattern omission and aliases, static background, 64-bit/fixed-string types, processing omission, frame/phase/memory/reference/structural-link guards.");
