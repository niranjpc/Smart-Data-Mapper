/* Dependency-free .xlsx reader: ArrayBuffer -> { sheetName: rows[][] } (strings, numbers, booleans; dates stay Excel serials). */
async function readXlsx(buffer) {
  const u8 = new Uint8Array(buffer), dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), utf8 = new TextDecoder('utf-8');
  let eocd = -1;
  for (let p = u8.length - 22; p >= Math.max(0, u8.length - 66000); p--) if (dv.getUint32(p, true) === 0x06054b50) { eocd = p; break; }
  if (eocd < 0) throw new Error('The tracker is not an .xlsx workbook (no zip directory found).');
  const files = new Map();
  let p = dv.getUint32(eocd + 16, true);
  for (let n = dv.getUint16(eocd + 10, true); n > 0 && dv.getUint32(p, true) === 0x02014b50; n--) {
    const nameLen = dv.getUint16(p + 28, true);
    files.set(utf8.decode(u8.subarray(p + 46, p + 46 + nameLen)), { method: dv.getUint16(p + 10, true), size: dv.getUint32(p + 20, true), at: dv.getUint32(p + 42, true) });
    p += 46 + nameLen + dv.getUint16(p + 30, true) + dv.getUint16(p + 32, true);
  }
  const text = async (name) => {
    const f = files.get(name) || files.get(name.replace(/^\//, ''));
    if (!f) return null;
    if (f.size === 0xffffffff || f.at === 0xffffffff) throw new Error('Workbook too large for the built-in reader (zip64).');
    const start = f.at + 30 + dv.getUint16(f.at + 26, true) + dv.getUint16(f.at + 28, true);
    const raw = u8.subarray(start, start + f.size);
    if (f.method === 0) return utf8.decode(raw);
    if (f.method !== 8) throw new Error('Unsupported compression in workbook part ' + name);
    return utf8.decode(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer());
  };
  const unesc = (s) => s.replace(/&(#x?[0-9a-fA-F]+|amp|lt|gt|quot|apos);/g, (m, e) => e === 'amp' ? '&' : e === 'lt' ? '<' : e === 'gt' ? '>' : e === 'quot' ? '"' : e === 'apos' ? "'"
    : String.fromCodePoint(e.charAt(1) === 'x' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10))).replace(/_x000D_/g, '');
  const attr = (tag, name) => { const m = new RegExp('(?:^|\\s)' + name + '="([^"]*)"').exec(tag); return m ? unesc(m[1]) : null; };
  const runs = (xml) => { let s = ''; xml.replace(/<rPh\b[\s\S]*?<\/rPh>/g, '').replace(/<t\b[^>]*>([\s\S]*?)<\/t>/g, (m, t) => { s += t; return m; }); return unesc(s); };

  const wb = await text('xl/workbook.xml');
  if (!wb) throw new Error('The tracker is not an .xlsx workbook (xl/workbook.xml missing).');
  const rels = new Map();
  ((await text('xl/_rels/workbook.xml.rels')) || '').replace(/<Relationship\b([^>]*)>/g, (m, a) => { rels.set(attr(a, 'Id'), attr(a, 'Target')); return m; });
  const shared = [];
  ((await text('xl/sharedStrings.xml')) || '').replace(/<si\b[^>]*>([\s\S]*?)<\/si>/g, (m, x) => { shared.push(runs(x)); return m; });

  const out = {}, sheetTags = wb.match(/<sheet\b[^>]*>/g) || [];
  for (const tag of sheetTags) {
    const target = rels.get(attr(tag, 'r:id')) || '';
    const xml = await text(target.charAt(0) === '/' ? target.slice(1) : 'xl/' + target);
    const rows = [];
    let width = 0;
    if (xml) xml.replace(/<c\b([^>]*?)(?:\/>|>([\s\S]*?)<\/c>)/g, (m, a, inner) => {
      const ref = /^([A-Z]+)(\d+)$/.exec(attr(a, 'r') || '');
      if (!ref || inner === undefined) return m;
      const t = attr(a, 't'), vm = /<v\b[^>]*>([\s\S]*?)<\/v>/.exec(inner);
      let v = '';
      if (t === 'inlineStr') v = runs(inner);
      else if (!vm || vm[1] === '') return m; /* formula without a stored result = blank */
      else if (t === 's') v = shared[+vm[1]] === undefined ? '' : shared[+vm[1]];
      else if (t === 'b') v = vm[1] === '1';
      else if (t === 'str' || t === 'e') v = unesc(vm[1]);
      else v = Number(vm[1]);
      let c = 0;
      for (let k = 0; k < ref[1].length; k++) c = c * 26 + ref[1].charCodeAt(k) - 64;
      const r = +ref[2] - 1;
      while (rows.length <= r) rows.push([]);
      rows[r][c - 1] = v;
      if (c > width) width = c;
      return m;
    });
    out[attr(tag, 'name')] = rows.map((row) => { const full = new Array(width).fill(''); row.forEach((v, k) => { full[k] = v; }); return full; });
  }
  return out;
}
