// Lossless value/type/attribute/link copy through the unmodified h5wasm public API.
// Diffraction patterns are excluded by object identity, including hard/soft aliases.
const PATTERNS = new Set(["Processed Patterns", "Unprocessed Patterns", "Raw Patterns"]);
function identity(object) { return Array.from(object.create_reference().ref_data).join(","); }
function join(parent, name) { return `${parent === "/" ? "" : parent}/${name}`; }
function absolute(target, parent = "/") {
  const parts = (target.startsWith("/") ? target : join(parent, target)).split("/"), result = [];
  // HDF5 has no parent-directory traversal: ".." is a literal link name.
  for (const part of parts) if (part && part !== ".") result.push(part);
  return `/${result.join("/")}`;
}
function parent(path) { return path.slice(0, path.lastIndexOf("/")) || "/"; }
function basename(path) { return path.slice(path.lastIndexOf("/") + 1); }
function ensureGroup(file, path, h) {
  let result = file;
  for (const name of path.split("/").filter(Boolean)) {
    const next = result.get(name);
    if (next && !(next instanceof h.Group)) throw new Error(`Oxford metadata copy path collision at ${path}.`);
    result = next || result.create_group(name);
  }
  return result;
}
function supported(value, path, h) {
  const m = value.metadata, types = h.Module.H5T_class_t;
  const fail = (detail) => { throw new Error(`Oxford metadata ${path}: ${detail}; use Python export for this file.`); };
  if (!Array.isArray(m.shape)) fail("null dataspaces cannot be preserved by the browser HDF5 writer");
  if (m.type === types.H5T_INTEGER.value) {
    if (![1, 2, 4, 8].includes(m.size) || !m.littleEndian) fail("unsupported integer dtype or byte order");
  } else if (m.type === types.H5T_FLOAT.value) {
    if (![4, 8].includes(m.size) || !m.littleEndian) fail("unsupported float dtype or byte order");
  } else if (m.type === types.H5T_STRING.value) {
    // h5wasm's dtype API retains charset/width/VLEN, but fixes padding by charset.
    if (![0, 1].includes(m.cset) || (!m.vlen && m.strpad !== (m.cset === 0 ? 0 : 2))) fail("unsupported string padding or character set");
  } else {
    fail("unsupported HDF5 datatype (references, enums, compounds, arrays and opaque types require native copying)");
  }
  return value.dtype;
}
function attributes(source, target, h) {
  for (const [name, attr] of Object.entries(source.attrs)) {
    const dtype = supported(attr, `${source.path}@${name}`, h);
    target.create_attribute(name, attr.value, attr.shape, dtype);
  }
}
export function copyOxfordMetadata(source, destination, h) {
  // Structural discovery must never dereference a symbolic/external ancestor.
  // Ordinary metadata links are handled separately by the inventory below.
  function localHardObject(g, relativePath) {
    let object = g;
    for (const name of relativePath.split("/")) {
      if (!(object instanceof h.Group) || !object.keys().includes(name)) return null;
      const path = join(object.path, name);
      if (object.get_link(path) !== null || object.get_external_link(path).filename !== undefined)
        throw new Error(`Oxford structural metadata ${path}: soft/external structural links are unsupported; use Python export for this file.`);
      object = object.get(name);
    }
    return object;
  }
  if (localHardObject(source, "zebsdsolve/Producer")) throw new Error("Source already contains a zebsdsolve export; use the original acquisition file.");
  // Only a single acquisition is selectable in the browser and the native importer.
  const scans = new Map();
  for (const name of source.keys()) {
    if (source.get_link(join("/", name)) !== null || source.get_external_link(join("/", name)).filename !== undefined) continue;
    const root = source.get(name);
    if (root instanceof h.Group && localHardObject(root, "EBSD/Header") instanceof h.Group && localHardObject(root, "EBSD/Data") instanceof h.Group) {
      const id = identity(root);
      if (!scans.has(id)) scans.set(id, name);
    }
  }
  if (scans.size !== 1) throw new Error("Oxford browser export requires one unambiguous EBSD acquisition scan; use Python export for this file.");
  const scan = [...scans.values()][0], processing = `/${scan}/Data Processing`;
  const relocate = (path) => path === processing || path.startsWith(`${processing}/`) ? `/zebsdsolve/OriginalDataProcessing${path.slice(processing.length)}` : path;
  if (localHardObject(localHardObject(source, scan), "Data Processing") && localHardObject(source, "zebsdsolve/OriginalDataProcessing")) throw new Error("Oxford source already contains archived Data Processing metadata.");
  const rootID = identity(source), objects = new Map([[rootID, source]]), links = [], edges = new Map(), omitted = new Set(), omittedPaths = new Set();
  function inventory(g, path) {
    const children = new Map(); edges.set(identity(g), children);
    function add(name, link) { links.push(link); children.set(name, link); }
    for (const name of g.keys()) {
      const p = join(path, name), symbolic = source.get_link(p);
      if (symbolic !== null) {
        add(name, { kind: "soft", path: p, target: symbolic });
        if (PATTERNS.has(name)) omittedPaths.add(p);
        continue;
      }
      const external = source.get_external_link(p);
      if (external.filename !== undefined) {
        add(name, { kind: "external", path: p, object: external });
        if (PATTERNS.has(name) || PATTERNS.has(basename(external.obj_path))) omittedPaths.add(p);
        continue;
      }
      const object = source.get(p);
      if (object instanceof h.Group || object instanceof h.Dataset) {
        const id = identity(object);
        add(name, { kind: "hard", path: p, id });
        if (object instanceof h.Dataset && PATTERNS.has(name)) { omitted.add(id); omittedPaths.add(p); }
        if (!objects.has(id)) {
          objects.set(id, object);
          if (object instanceof h.Group) inventory(object, p);
        }
      } else {
        throw new Error(`Oxford metadata ${p}: unsupported named datatype or link; use Python export for this file.`);
      }
    }
  }
  inventory(source, "/");
  // Resolve symbolic links from the inventory. get_type() in this runtime can
  // misclassify broken links with missing parents, and following external links
  // could silently import data from outside the acquisition.
  function resolve(path, depth = 0) {
    if (depth > 64) return null;
    const parts = absolute(path).split("/").filter(Boolean);
    let id = rootID, prefix = "/";
    for (let i = 0; i < parts.length; i++) {
      const edge = edges.get(id)?.get(parts[i]);
      if (!edge || edge.kind === "external") return null;
      if (edge.kind === "soft") return resolve(join(absolute(edge.target, prefix), parts.slice(i + 1).join("/")), depth + 1);
      id = edge.id; prefix = join(prefix, parts[i]);
    }
    return id;
  }
  for (const link of links) if (link.kind === "soft") link.id = resolve(link.path);
  // A named pattern soft link must also exclude its underlying hard-linked object.
  for (const link of links) if (omittedPaths.has(link.path) && link.id && objects.get(link.id) instanceof h.Dataset) omitted.add(link.id);
  for (const link of links) if (link.id && omitted.has(link.id)) omittedPaths.add(link.path);
  function skip(link) {
    if (omittedPaths.has(link.path) || (link.id && omitted.has(link.id))) return true;
    if (link.kind !== "soft") return false;
    const target = absolute(link.target, parent(link.path));
    return omittedPaths.has(target) || PATTERNS.has(basename(target));
  }
  // Check all retained types before reading any dataset values or writing output.
  for (const [id, object] of objects) {
    if (omitted.has(id)) continue;
    if (object instanceof h.Dataset) supported(object, object.path, h);
    for (const [name, attr] of Object.entries(object.attrs)) supported(attr, `${object.path}@${name}`, h);
  }
  // Record archived paths by identity, so a soft target reached through a hard
  // alias of the acquisition still follows the archived processing object.
  const archived = new Map();
  function archive(id, path) {
    if (!id || archived.has(id) || omitted.has(id)) return;
    archived.set(id, path);
    for (const [name, edge] of edges.get(id) || []) if (edge.kind === "hard") archive(edge.id, join(path, name));
  }
  archive(resolve(processing), "/zebsdsolve/OriginalDataProcessing");
  attributes(source, destination, h);
  const copied = new Map([[identity(source), "/"]]);
  for (const link of links) {
    if (link.kind !== "hard" || skip(link)) continue;
    const path = relocate(link.path), g = ensureGroup(destination, parent(path), h), name = basename(path);
    if (copied.has(link.id)) {
      if (path !== copied.get(link.id) && g.create_hard_link(copied.get(link.id), name) < 0) throw new Error(`Cannot preserve Oxford hard link ${link.path}.`);
      continue;
    }
    const object = objects.get(link.id);
    let result;
    if (object instanceof h.Group) result = ensureGroup(destination, path, h);
    else result = g.create_dataset({ name, data: object.value, shape: object.shape, dtype: object.dtype });
    copied.set(link.id, path);
    attributes(object, result, h);
  }
  for (const link of links) {
    if (link.kind === "hard" || skip(link)) continue;
    const path = relocate(link.path), g = ensureGroup(destination, parent(path), h), name = basename(path);
    let status;
    if (link.kind === "external") status = g.create_external_link(link.object.filename, link.object.obj_path, name);
    else {
      // Rebase relative targets only when either endpoint moved during archival.
      const target = absolute(link.target, parent(link.path)), relocated = relocate(target);
      const archivedTarget = archived.get(link.id);
      const destinationTarget = archivedTarget === path ? copied.get(link.id) : archivedTarget;
      status = g.create_soft_link(destinationTarget || (relocated !== target || path !== link.path ? relocated : link.target), name);
    }
    if (status < 0) throw new Error(`Cannot preserve Oxford ${link.kind} link ${link.path}.`);
  }
}
