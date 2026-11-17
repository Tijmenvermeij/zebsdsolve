// The policy follows the frozen indexing snapshot, never the current controls.
export const OXFORD_SOURCE_PC_WARNING = "Orientations were indexed with a fitted source plane. Standard Oxford pattern-centre fields retain acquisition values. Fitted values are stored in /zebsdsolve/PatternCentres.";
export function preservesSourcePC(snapshot) {
  return snapshot?.oxford_frame?.preserve_source_pattern_centres === true || Number(snapshot?.geometry?.plane_kind) > 0;
}
export function oxfordExportWarnings(snapshot) {
  return preservesSourcePC(snapshot) ? [OXFORD_SOURCE_PC_WARNING] : [];
}

export function isOxfordH5(format) {
  return /^(?:Oxford h5oina|h5_oxford)$/i.test(format || "");
}

// Saved fitted geometry is tied to the detector convention used during fitting.
export function geometryConvention(scan) {
  if (!isOxfordH5(scan?.format)) return "generic-v1";
  if (scan.geometryFrame === 2) return "oxford-calibrated-v3";
  if (scan.geometryFrame === 3) return "oxford-fallback-v1";
  return "oxford-calibrated-v1";
}

export function acceptsGeometryConvention(scan, saved) {
  return saved === geometryConvention(scan) ||
    (isOxfordH5(scan?.format) && scan.geometryFrame === 3 && saved === "oxford-calibrated-v1");
}
