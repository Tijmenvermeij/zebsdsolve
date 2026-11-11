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
