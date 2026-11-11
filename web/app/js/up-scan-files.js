// Selection only. Grid parsing and pixel-order validation belong to the native scan reader.
const isUP = (file) => /\.up[12]$/i.test(file.name);
const isANG = (file) => /\.ang$/i.test(file.name);
export function validateUPProjectSources(expectedSources, files) {
  if (!expectedSources?.some(isUP) || !expectedSources.some(isANG)) return;
  if (files.filter(isUP).length !== 1 || files.filter(isANG).length !== 1) {
    throw new Error("This project requires its original UP1/UP2 and ANG files. Open them together with the project.");
  }
  for (const expected of expectedSources.filter(file => isUP(file) || isANG(file))) {
    if (!files.some(file => file.name === expected.name && file.size === expected.size &&
      (!Number.isFinite(expected.lastModified) || !Number.isFinite(file.lastModified) || file.lastModified === expected.lastModified))) {
      throw new Error(`The project requires the original ${expected.name} (${expected.size} bytes). Select the original UP1/UP2 and ANG files.`);
    }
  }
}
export async function pairUPScan(files, chooseANG, expectedSources = null) {
  const patterns = files.filter(isUP), scans = files.filter(isANG);
  if (!patterns.length) {
    validateUPProjectSources(expectedSources, files);
    if (scans.length) throw new Error("Open the ANG scan file together with its UP1/UP2 pattern file.");
    return files;
  }
  if (patterns.length !== 1 || scans.length > 1 || files.some(file => !isUP(file) && !isANG(file))) throw new Error("Select one UP1/UP2 pattern file and its corresponding ANG scan file.");
  const ang = scans[0] || await chooseANG(patterns[0]);
  if (!ang) return null;
  if (!isANG(ang)) throw new Error("Select an ANG scan file.");
  const paired = [patterns[0], ang];
  validateUPProjectSources(expectedSources, paired);
  return paired;
}

export function chooseANG(pattern) {
  return new Promise((resolve, reject) => {
    const dialog = document.createElement("dialog"); dialog.id = "scan-info-dialog";
    dialog.className = "scan-info-dialog"; dialog.setAttribute("aria-labelledby", "scan-info-title");
    const title = document.createElement("h3"); title.id = "scan-info-title"; title.textContent = `Select the ANG scan file for ${pattern.name}`;
    const row = document.createElement("div"); row.className = "row";
    const open = document.createElement("button"); open.type = "button"; open.className = "primary"; open.textContent = "Choose ANG…";
    const cancel = document.createElement("button"); cancel.type = "button"; cancel.textContent = "Cancel";
    const input = document.createElement("input"); input.type = "file"; input.accept = ".ang"; input.hidden = true; input.id = "file-scan-info";
    row.append(open, cancel); dialog.append(title, row, input); document.body.append(dialog);
    let done = false, choosing = false;
    function finish(file, error = null) {
      if (done) return; done = true; dialog.close(); dialog.remove();
      error ? reject(error) : resolve(file);
    }
    input.addEventListener("change", () => finish(input.files[0] || null));
    input.addEventListener("cancel", () => finish(null));
    cancel.addEventListener("click", () => finish(null));
    dialog.addEventListener("cancel", (event) => { event.preventDefault(); finish(null); });
    async function pick() {
      if (choosing || done) return; choosing = true;
      try {
        if (typeof window.showOpenFilePicker === "function") {
          try {
            const handles = await window.showOpenFilePicker({ id: "zebsdsolve-scan", multiple: false,
              types: [{description: "ANG scan information", accept: {"text/plain": [".ang"]}}] });
            finish(handles.length ? await handles[0].getFile() : null); return;
          } catch (error) {
            if (error.name === "AbortError") { finish(null); return; }
            if (!["SecurityError", "NotAllowedError"].includes(error.name)) throw error;
          }
        }
        // Fall back to the standard input where the browser disallows this API.
        input.click();
      } catch (error) { finish(null, error); }
      finally { choosing = false; }
    }
    open.addEventListener("click", pick);
    dialog.showModal(); open.focus();
  });
}
