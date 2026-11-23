import { filesFromHandles, sourceFolders } from './source-folders.js';

const scanName = /\.(?:up1|up2|h5|hdf5|h5ebsd|h5oina|ebsp|dat)$/i;
const h5Name = /\.(?:h5|hdf5)$/i;
const isUP = name => /\.up[12]$/i.test(name);
const compareNames = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }) || a.name.localeCompare(b.name);

// Enumerating handles does not open or read file contents.
export async function listScanFolder(directory) {
  const entries = [];
  for await (const entry of directory.values()) if (entry.kind === 'file') entries.push(entry);
  entries.sort(compareNames);
  return {
    scans: entries.filter(entry => scanName.test(entry.name) && !(h5Name.test(entry.name) && /master/i.test(entry.name))),
    angs: entries.filter(entry => /\.ang$/i.test(entry.name)),
    setting: entries.find(entry => /^setting\.txt$/i.test(entry.name)) || null,
  };
}

// A null default with available ANG files means the user must choose explicitly.
export function defaultANG(scan, angs) {
  if (!isUP(scan.name)) return null;
  const base = scan.name.replace(/\.up[12]$/i, '').toLowerCase();
  const exact = angs.filter(entry => entry.name.replace(/\.ang$/i, '').toLowerCase() === base);
  return exact.length === 1 ? exact[0] : exact.length === 0 && angs.length === 1 ? angs[0] : null;
}
export function selectedHandles(scan, ang, setting) {
  return [scan, ...(isUP(scan.name) && ang ? [ang] : []), ...(/\.dat$/i.test(scan.name) && setting ? [setting] : [])];
}

export async function selectScanInFolder(directory, { chooseFolder } = {}) {
  return new Promise((resolve, reject) => {
    const dialog = document.createElement('dialog'); dialog.className = 'scan-info-dialog scan-folder-dialog';
    const title = document.createElement('h3'); title.id = 'scan-folder-title'; title.textContent = directory.name;
    dialog.setAttribute('aria-labelledby', title.id);
    const errorText = document.createElement('p'); errorText.className = 'folder-error'; errorText.setAttribute('role', 'alert'); errorText.hidden = true;
    const scanLabel = document.createElement('label'); scanLabel.textContent = 'Scan';
    const scans = document.createElement('select'); scans.size = 7; scanLabel.append(scans);
    const angLabel = document.createElement('label'); angLabel.textContent = 'ANG';
    const angSelect = document.createElement('select'); angLabel.append(angSelect); angLabel.hidden = true;
    const row = document.createElement('div'); row.className = 'row';
    const open = document.createElement('button'); open.type = 'button'; open.className = 'primary'; open.textContent = 'Open'; open.disabled = true;
    const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = 'Cancel';
    const change = document.createElement('button'); change.type = 'button'; change.textContent = 'Change folder…'; change.hidden = !chooseFolder;
    row.append(open, cancel, change); dialog.append(title, errorText, scanLabel, angLabel, row); document.body.append(dialog);
    let settled = false, busy = false, choosing = false, generation = 0, listing = null;
    const finish = value => { if (settled) return; settled = true; dialog.close(); dialog.remove(); resolve(value); };
    const showError = error => { errorText.textContent = error.message || String(error); errorText.hidden = false; };
    const updateOpen = () => { open.disabled = busy || choosing || !listing?.scans.length || (!angLabel.hidden && angSelect.value === ''); change.disabled = busy || choosing; };
    const updateANG = () => {
      const scan = listing.scans[Number(scans.value)];
      angLabel.hidden = !scan || !isUP(scan.name); angSelect.replaceChildren();
      if (!angLabel.hidden) {
        const chosen = defaultANG(scan, listing.angs);
        if (listing.angs.length && !chosen) angSelect.add(new Option('Choose ANG…', ''));
        listing.angs.forEach((entry, index) => angSelect.add(new Option(entry.name, String(index))));
        angSelect.add(new Option('Choose ANG outside this folder…', 'outside'));
        angSelect.value = chosen ? String(listing.angs.indexOf(chosen)) : listing.angs.length ? '' : 'outside';
      }
      updateOpen();
    };
    cancel.onclick = () => finish(null); dialog.oncancel = event => { event.preventDefault(); finish(null); };
    scans.onchange = updateANG; angSelect.onchange = updateOpen;
    open.onclick = async () => {
      if (busy || open.disabled) return;
      busy = true; updateOpen(); errorText.hidden = true;
      try {
        const scan = listing.scans[Number(scans.value)];
        const ang = !angLabel.hidden && angSelect.value !== 'outside' ? listing.angs[Number(angSelect.value)] : null;
        const files = await filesFromHandles(selectedHandles(scan, ang, listing.setting));
        if (settled) return;
        await sourceFolders.accept(directory, files[0], true);
        if (!settled) finish(files);
      } catch (error) { if (!settled) showError(error); }
      finally { busy = false; if (!settled) updateOpen(); }
    };
    const load = nextDirectory => {
      directory = nextDirectory;
      const current = ++generation;
      title.textContent = directory.name; listing = null; errorText.hidden = true;
      scans.replaceChildren(new Option('Loading…', '')); scans.disabled = true;
      angLabel.hidden = true; updateOpen();
      listScanFolder(directory).then(result => {
        if (settled || current !== generation) return;
        listing = result; scans.replaceChildren(); scans.disabled = !result.scans.length;
        result.scans.forEach((entry, index) => scans.add(new Option(entry.name, String(index))));
        if (!result.scans.length) showError(new Error('No scans found in this folder.'));
        else { scans.value = '0'; updateANG(); scans.focus(); }
        updateOpen();
      }, error => { if (!settled && current === generation) { scans.replaceChildren(); showError(error); updateOpen(); } });
    };
    change.onclick = () => {
      if (busy || choosing) return;
      choosing = true; updateOpen();
      let selection;
      try { selection = chooseFolder(); }
      catch (error) { choosing = false; showError(error); updateOpen(); return; }
      Promise.resolve(selection).then(next => {
        if (settled) return;
        choosing = false;
        if (next) load(next); else updateOpen();
      }, error => {
        if (settled) return;
        choosing = false;
        if (error.name !== 'AbortError') showError(error);
        updateOpen();
      });
    };
    try { dialog.showModal(); } catch (error) { dialog.remove(); reject(error); return; }
    load(directory);
  });
}
