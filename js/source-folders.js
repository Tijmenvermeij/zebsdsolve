// File identity and folder grants stay in the browser, never in the native engine.
const fileHandles = new WeakMap();
export function rememberFileHandle(file, handle) { if (file && handle?.kind === 'file') fileHandles.set(file, handle); return file; }
export function fileHandleFor(file) { return fileHandles.get(file); }
export async function filesFromHandles(handles) { return Promise.all(handles.map(async handle => rememberFileHandle(await handle.getFile(), handle))); }

async function sameEntry(a, b) { return a === b || !!(a?.isSameEntry && await a.isSameEntry(b)); }
export async function verifyScanFolder(directory, file) {
  let entry;
  try { entry = await directory.getFileHandle(file.name); }
  catch (error) { if (!['NotFoundError', 'TypeMismatchError'].includes(error.name)) throw error; throw new Error(`Choose the folder containing ${file.name}.`); }
  const original = fileHandles.get(file);
  if (original && !await sameEntry(entry, original)) throw new Error(`This folder does not contain the selected ${file.name}.`);
  const current = await entry.getFile();
  if (current.name !== file.name || current.size !== file.size || Number.isFinite(file.lastModified) && current.lastModified !== file.lastModified) {
    throw new Error(`The folder's ${file.name} differs from the loaded file.`);
  }
  return directory;
}
export class SourceFolders {
  constructor() { this.roots = []; this.exact = new WeakMap(); }
  async add(directory) {
    for (const root of this.roots) if (await sameEntry(root, directory)) return root;
    this.roots.push(directory); return directory;
  }
  async accept(directory, file, required = false) {
    if (required) await verifyScanFolder(directory, file);
    directory = await this.add(directory);
    if (required) { this.exact.set(file, directory); }
    else {
      try { await verifyScanFolder(directory, file); this.exact.set(file, directory); } catch { /* An explicitly chosen custom destination need not contain the scan. */ }
    }
    return directory;
  }
  async find(file) {
    const exact = this.exact.get(file);
    if (exact) { await verifyScanFolder(exact, file); return exact; }
    if (!fileHandles.has(file)) return null;
    for (const root of this.roots) {
      try { await verifyScanFolder(root, file); this.exact.set(file, root); return root; }
      catch (error) { if (!/Choose the folder|does not contain|differs from/.test(error.message)) throw error; }
    }
    return null;
  }
}
export const sourceFolders = new SourceFolders();

export function chooseScanFolder(file, { current = null, permissionOnly = false } = {}) {
  return new Promise((resolve, reject) => {
    const dialog = document.createElement('dialog'); dialog.className = 'scan-info-dialog'; dialog.id = 'batch-folder-dialog';
    const title = document.createElement('h3'); title.id = 'batch-folder-title'; title.textContent = permissionOnly ? `Allow output access to ${current.name}` : `Choose the scan folder for ${file.name}`;
    dialog.setAttribute('aria-labelledby', title.id);
    const errorText = document.createElement('p'); errorText.className = 'folder-error'; errorText.setAttribute('role', 'alert'); errorText.hidden = true;
    const row = document.createElement('div'); row.className = 'row';
    const choose = document.createElement('button'); choose.type = 'button'; choose.className = 'primary'; choose.textContent = permissionOnly ? 'Grant write access' : 'Choose scan folder…';
    const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = 'Cancel';
    row.append(choose, cancel); dialog.append(title, errorText, row); document.body.append(dialog);
    let settled = false, selecting = false;
    const finish = value => { if (settled) return; settled = true; dialog.close(); dialog.remove(); resolve(value); };
    cancel.onclick = () => finish(null); dialog.oncancel = event => { event.preventDefault(); finish(null); };
    choose.onclick = async () => {
      if (selecting) return; selecting = true; choose.disabled = true;
      try {
        if (permissionOnly) {
          if (await current.requestPermission({mode:'readwrite'}) !== 'granted') throw new Error('Write access was not granted.');
          finish(current);
        } else {
          if (!window.showDirectoryPicker) throw new Error('Folder saving requires Chrome or Edge.');
          const startIn = current || fileHandleFor(file);
          const directory = await window.showDirectoryPicker({ id:'zebsdsolve-scan-output', mode:'readwrite', ...(startIn ? {startIn} : {}) });
          await sourceFolders.accept(directory, file, true); finish(directory);
        }
      } catch (error) {
        if (error.name === 'AbortError') finish(null);
        else { errorText.textContent = error.message || String(error); errorText.hidden = false; }
      } finally { selecting = false; choose.disabled = false; }
    };
    try { dialog.showModal(); choose.focus(); } catch (error) { dialog.remove(); reject(error); }
  });
}
