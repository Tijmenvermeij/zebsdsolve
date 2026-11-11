// ZIP STORE records follow https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT.
// Snapshot each bounded CRC chunk once into owned Blob storage. The resulting ZIP
// stays readable after origin-private source files are reclaimed.
const ZIP32_MAX = 0xffffffff;
const MAX_CHUNK = 1024 * 1024;
const utf8 = new TextEncoder();
const crcTable = Uint32Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit++) value = value & 1 ? 0xedb88320 ^ value >>> 1 : value >>> 1;
  return value >>> 0;
});
function zip32(value) {
  if (!Number.isSafeInteger(value) || value < 0 || value >= ZIP32_MAX) throw new Error("Downloads ZIP exceeds ZIP32 limits. Clear/requeue this job in folder mode.");
  return value;
}
function dosDateTime(timestamp) {
  let date = new Date(timestamp);
  if (!Number.isFinite(date.getTime()) || date.getFullYear() < 1980) date = new Date(1980, 0, 1);
  if (date.getFullYear() > 2107) date = new Date(2107, 11, 31, 23, 59, 58);
  return { date: (date.getFullYear() - 1980) << 9 | (date.getMonth() + 1) << 5 | date.getDate(),
    time: date.getHours() << 11 | date.getMinutes() << 5 | Math.floor(date.getSeconds() / 2) };
}
async function crc32(blob, chunkSize, yieldControl) {
  let crc = 0xffffffff; const chunks = [];
  for (let offset = 0; offset < blob.size; offset += chunkSize) {
    const end = Math.min(offset + chunkSize, blob.size);
    const bytes = new Uint8Array(await blob.slice(offset, end).arrayBuffer());
    if (bytes.byteLength !== end - offset) throw new Error("A staged ZIP member changed while being read.");
    for (const byte of bytes) crc = crcTable[(crc ^ byte) & 255] ^ crc >>> 8;
    chunks.push(new Blob([bytes]));
    await yieldControl();
  }
  return { crc: (crc ^ 0xffffffff) >>> 0, chunks };
}
export async function createStoreZip(members, { chunkSize = MAX_CHUNK, yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)) } = {}) {
  if (!members.length || members.length >= 0xffff) throw new Error("Unsupported ZIP member count.");
  if (!Number.isSafeInteger(chunkSize) || chunkSize < 1 || chunkSize > MAX_CHUNK) throw new Error("ZIP CRC chunks must be between 1 byte and 1 MiB.");
  const names = new Set(); let offset = 0, centralSize = 0;
  const entries = members.map(({ name, blob }) => {
    if (typeof name !== "string" || !name || /[\\:\x00]/.test(name) || name.startsWith("/") || name.split("/").some(part => !part || part === "." || part === "..") || names.has(name)) throw new Error("Invalid ZIP member path.");
    const encoded = utf8.encode(name);
    if (encoded.length > 0xffff || !blob || typeof blob.slice !== "function") throw new Error("Invalid ZIP member.");
    names.add(name);
    const size = zip32(blob.size), start = zip32(offset);
    offset = zip32(offset + 30 + encoded.length + size);
    centralSize = zip32(centralSize + 46 + encoded.length);
    return { encoded, blob, size, start, ...dosDateTime(blob.lastModified ?? Date.now()) };
  });
  const centralStart = zip32(offset);
  zip32(centralStart + centralSize + 22); // Reject before any large member read or field overflow.
  const parts = [], central = [];
  for (const entry of entries) {
    const { crc, chunks } = await crc32(entry.blob, chunkSize, yieldControl);
    const local = new Uint8Array(30 + entry.encoded.length), l = new DataView(local.buffer);
    l.setUint32(0, 0x04034b50, true); l.setUint16(4, 20, true); l.setUint16(6, 0x0800, true);
    l.setUint16(10, entry.time, true); l.setUint16(12, entry.date, true); l.setUint32(14, crc, true);
    l.setUint32(18, entry.size, true); l.setUint32(22, entry.size, true); l.setUint16(26, entry.encoded.length, true);
    local.set(entry.encoded, 30); parts.push(local);
    for (const chunk of chunks) parts.push(chunk);
    const header = new Uint8Array(46 + entry.encoded.length), c = new DataView(header.buffer);
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
    c.setUint16(12, entry.time, true); c.setUint16(14, entry.date, true); c.setUint32(16, crc, true);
    c.setUint32(20, entry.size, true); c.setUint32(24, entry.size, true); c.setUint16(28, entry.encoded.length, true); c.setUint32(42, entry.start, true);
    header.set(entry.encoded, 46); central.push(header);
  }
  const end = new Uint8Array(22), e = new DataView(end.buffer);
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, entries.length, true); e.setUint16(10, entries.length, true);
  e.setUint32(12, centralSize, true); e.setUint32(16, centralStart, true);
  return new Blob([...parts, ...central, end], { type: "application/zip" });
}
