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

// Geometry residuals are meaningful only relative to the camera baseline used
// while fitting them. Unversioned or v1 Oxford residuals cannot be reinterpreted.
export function geometryConvention(format, detectorAttitudeVersion=null) {
  if(!isOxfordH5(format))return "generic-v1";
  return detectorAttitudeVersion===2?"oxford-calibrated-v2":detectorAttitudeVersion===0?"oxford-fallback-v1":"oxford-unversioned";
}
export function canRestoreGeometry(convention, format, detectorAttitudeVersion=null) {
  if(!isOxfordH5(format))return !(typeof convention==="string" && convention.startsWith("oxford-"));
  const current=geometryConvention(format,detectorAttitudeVersion);
  return current!=="oxford-unversioned" && convention===current;
}
