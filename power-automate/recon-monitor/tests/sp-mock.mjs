// A strict stand-in for the SharePoint REST endpoints the flows use, to check the generated $batch requests offline.
// It parses the multipart/mixed batches exactly as written by the Office Script and rejects anything SharePoint would
// reject for the operations used here: unknown lists / fields / views, wrong item entity type, wrong value types,
// malformed CAML, formatter JSON that does not parse or uses unsupported style properties.

const STYLE_OK = new Set(['background-color', 'fill', 'border', 'border-top', 'border-left', 'border-right', 'border-bottom', 'border-color', 'border-style',
  'border-width', 'border-radius', 'box-shadow', 'box-sizing', 'overflow', 'opacity', 'cursor', 'height', 'max-height', 'max-width', 'min-height', 'min-width',
  'width', 'flex', 'flex-grow', 'flex-shrink', 'flex-flow', 'flex-direction', 'flex-wrap', 'justify-content', 'align-items', 'align-self', 'font', 'font-family',
  'font-size', 'font-style', 'font-weight', 'letter-spacing', 'line-height', 'text-align', 'text-decoration', 'text-overflow', 'text-transform', 'vertical-align',
  'white-space', 'word-break', 'display', 'color', 'padding', 'padding-top', 'padding-bottom', 'padding-left', 'padding-right', 'margin', 'margin-top',
  'margin-bottom', 'margin-left', 'margin-right', 'position', 'top', 'left', 'right', 'bottom', 'z-index', 'visibility']);
const SEVERITY = /^sp-field-severity--(good|low|warning|severeWarning|blocked)$/;

export function createMockSharePoint(site) {
  const lists = new Map(); // title -> { title, type, fields: Map(internal -> {type, title, formatter}), views: Map, items: Map, nextId }
  const entity = (t) => 'SP.Data.' + t.replace(/ /g, '_x0020_') + 'ListItem';
  const fail = (code, msg) => { const e = new Error(msg); e.code = code; throw e; };
  const getList = (t) => lists.get(t) || fail(404, `List '${t}' does not exist`);
  const field = (l, n) => l.fields.get(n) || [...l.fields.values()].find((f) => f.title === n) || fail(400, `Field '${n}' does not exist in '${l.title}'`);
  const checkFormatter = (json, kind) => {
    const f = JSON.parse(json); // throws on bad JSON
    if (!/json-schemas\/sp\/v2\/(row|column)-formatting\.schema\.json$/.test(f.$schema || '')) fail(400, 'formatter without a v2 $schema');
    const walk = (e) => {
      if (!e || typeof e !== 'object') return;
      if (Array.isArray(e)) return e.forEach(walk);
      if (e.style) Object.keys(e.style).forEach((k) => { if (!STYLE_OK.has(k)) fail(400, `style property '${k}' is not supported`); });
      if (e.attributes && e.attributes.class && !String(e.attributes.class).startsWith('=') && !SEVERITY.test(e.attributes.class)) fail(400, 'unknown class');
      Object.values(e).forEach(walk);
    };
    walk(kind === 'row' ? f.rowFormatter || {} : f);
    // every [$Field] used must exist
    return f;
  };
  const fieldsUsed = (json) => [...json.matchAll(/\[\$([A-Za-z0-9_]+)\]/g)].map((m) => m[1]);
  const checkCaml = (l, q) => {
    const tags = [...q.matchAll(/<(\/?)([A-Za-z]+)[^>]*?(\/?)>/g)];
    const stack = [];
    tags.forEach(([, close, name, self]) => { if (self) return; if (close) { if (stack.pop() !== name) fail(400, 'CAML not well formed: ' + q); } else stack.push(name); });
    if (stack.length) fail(400, 'CAML not closed: ' + q);
    [...q.matchAll(/FieldRef Name='([^']+)'/g)].forEach((m) => field(l, m[1]));
  };

  function apply(method, url, headers, body) {
    if (!url.startsWith(site.replace(/\/$/, '') + '/_api/')) fail(400, 'URL not on the site: ' + url);
    if (/\s/.test(url)) fail(400, 'unencoded space in URL');
    const path = decodeURIComponent(url.slice(site.replace(/\/$/, '').length + 6));
    const xm = (headers['x-http-method'] || method).toUpperCase();
    const json = body ? JSON.parse(body) : null;
    let m;
    if (path === 'web/lists' && xm === 'POST') {
      if (json.__metadata.type !== 'SP.List' || json.BaseTemplate !== 100) fail(400, 'bad list body');
      if (lists.has(json.Title)) fail(409, `A list '${json.Title}' already exists`);
      lists.set(json.Title, { title: json.Title, type: entity(json.Title), fields: new Map([['Title', { type: 'Text', title: 'Title' }]]), views: new Map([['All Items', { fields: ['LinkTitle'] }]]), items: new Map(), nextId: 1 });
      return 201;
    }
    if (!(m = /^web\/lists\/getbytitle\('((?:[^']|'')+)'\)(.*)$/.exec(path))) fail(400, 'unsupported path ' + path);
    const l = getList(m[1].replace(/''/g, "'")), rest = m[2];
    if ((m = /^\/fields\/getbyinternalnameortitle\('([^']+)'\)$/.exec(rest)) && xm === 'MERGE') {
      if (headers['if-match'] !== '*') fail(412, 'IF-MATCH missing');
      const f = field(l, m[1]);
      if (json.__metadata.type !== 'SP.Field') fail(400, 'bad field metadata');
      if ('Title' in json) f.title = json.Title;
      if ('CustomFormatter' in json) { checkFormatter(json.CustomFormatter, 'column'); f.formatter = json.CustomFormatter; }
      return 204;
    }
    if (rest === '/fields/createfieldasxml' && xm === 'POST') {
      const p = json.parameters;
      if (p.__metadata.type !== 'SP.XmlSchemaFieldCreationInformation' || p.Options !== 8) fail(400, 'bad field creation info');
      const a = Object.fromEntries([...p.SchemaXml.matchAll(/(\w+)="([^"]*)"/g)].map((z) => [z[1], z[2]]));
      if (!/^<Field [^<>]*\/>$/.test(p.SchemaXml)) fail(400, 'SchemaXml malformed');
      if (!['Text', 'Number', 'Note'].includes(a.Type) || !/^[A-Za-z][A-Za-z0-9]*$/.test(a.Name || '')) fail(400, 'bad field ' + p.SchemaXml);
      if (l.fields.has(a.Name)) fail(409, 'duplicate field ' + a.Name);
      l.fields.set(a.Name, { type: a.Type, title: a.DisplayName.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&amp;/g, '&') });
      return 201;
    }
    if (rest === '/views' && xm === 'POST') {
      if (json.__metadata.type !== 'SP.View' || json.PersonalView !== false) fail(400, 'bad view body');
      if (l.views.has(json.Title)) fail(409, 'duplicate view ' + json.Title);
      checkCaml(l, json.ViewQuery);
      l.views.set(json.Title, { query: json.ViewQuery, rowLimit: json.RowLimit, paged: json.Paged, fields: ['LinkTitle'] });
      return 201;
    }
    if ((m = /^\/views\/getbytitle\('([^']+)'\)(.*)$/.exec(rest))) {
      const vn = m[1], v = l.views.get(vn) || fail(404, `View '${vn}' does not exist`), r = m[2];
      if (r === '/viewfields/removeallviewfields' && xm === 'POST') { v.fields = []; return 200; }
      if ((m = /^\/viewfields\/addviewfield\('([^']+)'\)$/.exec(r)) && xm === 'POST') { field(l, m[1]); v.fields.push(m[1]); return 200; }
      if (r === '' && xm === 'MERGE') {
        const f = checkFormatter(json.CustomFormatter, 'row');
        fieldsUsed(json.CustomFormatter).forEach((n) => { field(l, n); if (!v.fields.includes(n)) fail(400, `formatter uses [$${n}] which is not in view '${vn}'`); });
        v.formatter = f; return 204;
      }
    }
    if (rest === '/items' && xm === 'POST') {
      if (json.__metadata.type !== l.type) fail(400, `entity type '${json.__metadata.type}' does not match '${l.type}'`);
      const item = {};
      Object.entries(json).forEach(([k, val]) => {
        if (k === '__metadata') return;
        const f = l.fields.get(k) || fail(400, `Column '${k}' does not exist in '${l.title}'`);
        if (f.type === 'Number' && typeof val !== 'number') fail(400, `'${k}' expects a number, got ${JSON.stringify(val)}`);
        if (f.type !== 'Number' && typeof val !== 'string') fail(400, `'${k}' expects text`);
        if (f.type === 'Text' && val.length > 255) fail(400, `'${k}' is longer than 255 characters`);
        item[k] = val;
      });
      item.Id = l.nextId++; l.items.set(item.Id, item); return 201;
    }
    if ((m = /^\/items\((\d+)\)$/.exec(rest)) && xm === 'DELETE') {
      if (headers['if-match'] !== '*') fail(412, 'IF-MATCH missing');
      if (!l.items.delete(+m[1])) fail(404, 'Item does not exist'); return 200;
    }
    fail(400, `unsupported ${xm} ${path}`);
  }

  /** Handles POST _api/$batch with the given Content-Type header; returns { status, body } like SharePoint (200 + per-part statuses). */
  function batch(contentType, body) {
    const bm = /^multipart\/mixed; boundary=([^;]+)$/.exec(contentType);
    if (!bm) return { status: 400, body: 'bad content type' };
    const bnd = bm[1];
    if (!body.startsWith('--' + bnd + '\r\n') || !body.trimEnd().endsWith('--' + bnd + '--')) return { status: 400, body: 'batch boundary mismatch' };
    const cm = /boundary="([^"]+)"/.exec(body.slice(0, 300));
    const parts = body.split('--' + cm[1]).slice(1, -1);
    let resp = '';
    for (const part of parts) {
      const blocks = part.replace(/^\r\n/, '').split('\r\n\r\n');
      if (!/^Content-Type: application\/http\r\nContent-Transfer-Encoding: binary$/.test(blocks[0])) return { status: 400, body: 'bad part headers' };
      const [reqLine, ...hl] = blocks[1].split('\r\n');
      const [method, url, ver] = reqLine.split(' ');
      if (ver !== 'HTTP/1.1' || reqLine.split(' ').length !== 3) return { status: 400, body: 'bad request line: ' + reqLine };
      const headers = Object.fromEntries(hl.map((h) => [h.slice(0, h.indexOf(':')).toLowerCase(), h.slice(h.indexOf(':') + 1).trim()]));
      const reqBody = blocks.slice(2).join('\r\n\r\n').replace(/(\r\n)+$/, '');
      if (reqBody && headers['content-type'] !== 'application/json;odata=verbose') return { status: 400, body: 'body without verbose content type' };
      let code;
      try { code = apply(method, url, headers, reqBody); } catch (e) { code = e.code || 400; resp += `HTTP/1.1 ${code} Error\r\n\r\n{"error":{"message":{"value":${JSON.stringify(e.message)}}}}\r\n`; continue; }
      resp += `HTTP/1.1 ${code} OK\r\n\r\n`;
    }
    return { status: 200, body: resp };
  }

  /** What the daily flow's "Get list" + "Get ids" steps return, as the sharepointJson array. */
  const info = (titles) => titles.filter((t) => lists.has(t)).map((t) => ({ title: t, type: lists.get(t).type, items: [...lists.get(t).items.keys()].map((Id) => ({ Id })) }));
  return { lists, batch, info };
}
