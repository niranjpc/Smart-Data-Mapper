/* Loader. In SharePoint: reads recon-config.json and the tracker workbooks with the signed-in user's session, then renders.
   Opened from disk (file://) or with ?source=local: asks for the workbooks (and optionally recon-config.json) instead. */
(async function () {
  const box = document.getElementById('body');
  const safe = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const show = (title, lines) => { box.innerHTML = '<div class="card"><h2>' + safe(title) + '</h2>' + lines.map((l) => '<div class="hint">' + safe(l) + '</div>').join('') + '</div>'; };
  const get = async (url, what) => {
    const r = await fetch(url, { credentials: 'same-origin', cache: 'no-store' });
    if (!r.ok) throw new Error(what + ' could not be read (HTTP ' + r.status + '): ' + url);
    return r;
  };
  const qs = new URLSearchParams(location.search);
  const inspect = qs.get('mode') === 'inspect';

  /* books: [{ name, sheets }] oldest first — the last one is the current workbook and wins where days overlap */
  const render = (cfg, books) => {
    const worksheets = [];
    books.forEach((b) => Object.keys(b.sheets).forEach((n) => worksheets.push({ getName: () => n, getUsedRange: () => (b.sheets[n].length ? { getValues: () => b.sheets[n] } : undefined) })));
    /* ?asOf=yyyy-MM-dd shows the page as of an earlier day */
    const today = qs.get('asOf') || new Intl.DateTimeFormat('en-CA', cfg.timeZone ? { timeZone: cfg.timeZone } : {}).format(new Date());
    const res = main({ getWorksheets: () => worksheets }, JSON.stringify(cfg), today, inspect ? 'inspect' : 'build');
    if (inspect || !res.ok) {
      show(inspect ? 'Tracker inspection' : 'No tracker rows could be read',
        ['Workbooks read: ' + books.map((b) => b.name).join(' · '), 'Sheets: ' + Object.keys(books[books.length - 1].sheets).join(' · ')].concat(res.warnings));
      return;
    }
    startDashboard(JSON.parse(res.dataJson));
    if (res.warnings.length) {
      const d = document.createElement('details');
      d.className = 'note';
      d.innerHTML = '<summary>' + res.warnings.length + ' tracker warning' + (res.warnings.length > 1 ? 's' : '') + '</summary>' + res.warnings.map((w) => '<div>' + safe(w) + '</div>').join('');
      document.querySelector('.wrap').appendChild(d);
    }
  };

  const fail = (e) => { show('The Reconciliation Monitor could not load', [e && e.message ? e.message : String(e)]); throw e; };

  /* ---- on your PC: pick the files ---- */
  if (location.protocol === 'file:' || qs.get('source') === 'local') {
    box.innerHTML = '<div class="card"><h2>Open the recon workbooks</h2>' +
      '<div class="hint">Select the current Reconciliation Report workbook, any archived weekly workbooks, and recon-config.json (for customer name mapping). Nothing is uploaded: the files are read in this browser.</div>' +
      '<div class="bar"><input type="file" id="pick" multiple accept=".xlsx,.xlsm,.json"></div>' +
      '<div class="hint">The newest file (by date modified) is treated as the current week. Add ?mode=inspect to the address to see which sheets and columns were matched.</div></div>';
    document.getElementById('pick').onchange = async (ev) => {
      try {
        const files = Array.from(ev.target.files || []);
        const cfgFile = files.find((f) => /\.json$/i.test(f.name));
        const cfg = cfgFile ? JSON.parse(await cfgFile.text()) : {};
        const xl = files.filter((f) => /\.xls[xm]$/i.test(f.name) && f.name.charAt(0) !== '~').sort((a, b) => a.lastModified - b.lastModified || a.name.localeCompare(b.name));
        if (!xl.length) throw new Error('Select at least one .xlsx workbook.');
        show('Loading…', ['Reading ' + xl.length + ' workbook' + (xl.length > 1 ? 's' : '') + '.']);
        const books = [];
        for (const f of xl) books.push({ name: f.name, sheets: await readXlsx(await f.arrayBuffer()) });
        render(cfg, books);
      } catch (e) { fail(e); }
    };
    return;
  }

  /* ---- in SharePoint ---- */
  try {
    show('Loading…', ['Reading the recon tracker from SharePoint.']);
    const here = location.href.replace(/[?#].*$/, '').replace(/[^\/]*$/, '');
    const cfg = await (await get(qs.get('config') || here + 'recon-config.json', 'recon-config.json')).json();
    if (cfg.dataUrl) { startDashboard(await (await get(cfg.dataUrl, 'The published dataset')).json()); return; }
    const site = (cfg.siteUrl || '').replace(/\/$/, '');
    const api = (kind, p) => site + "/_api/web/Get" + kind + "ByServerRelativePath(decodedurl='" + encodeURIComponent(String(p).replace(/'/g, "''")) + "')";
    if (!cfg.trackerUrl && !(site && cfg.trackerPath)) throw new Error('recon-config.json needs siteUrl and trackerPath (or trackerUrl / dataUrl).');
    /* archived weekly workbooks (optional) first, current workbook last */
    const list = [];
    if (cfg.archiveFolder) {
      const r = await fetch(api('Folder', cfg.archiveFolder) + '/Files?$select=Name,ServerRelativeUrl,TimeLastModified&$top=5000', { credentials: 'same-origin', cache: 'no-store', headers: { Accept: 'application/json;odata=nometadata' } });
      if (!r.ok) throw new Error('Archive folder could not be listed (HTTP ' + r.status + '): ' + cfg.archiveFolder);
      const current = String(cfg.trackerPath || '').toLowerCase(), only = cfg.archivePattern ? new RegExp(cfg.archivePattern, 'i') : null;
      ((await r.json()).value || [])
        .filter((f) => /\.xls[xm]$/i.test(f.Name) && f.Name.charAt(0) !== '~' && f.ServerRelativeUrl.toLowerCase() !== current && (!only || only.test(f.Name)))
        .sort((a, b) => String(a.TimeLastModified || '').localeCompare(String(b.TimeLastModified || '')) || a.Name.localeCompare(b.Name))
        .forEach((f) => list.push({ name: f.Name, url: api('File', f.ServerRelativeUrl) + '/$value' }));
    }
    list.push({ name: 'current workbook', url: cfg.trackerUrl || api('File', cfg.trackerPath) + '/$value' });
    const books = await Promise.all(list.map(async (b) => ({ name: b.name, sheets: await readXlsx(await (await get(b.url, 'Workbook "' + b.name + '"')).arrayBuffer()) })));
    render(cfg, books);
  } catch (e) { fail(e); }
})();
