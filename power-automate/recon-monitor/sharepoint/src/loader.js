/* Live loader: reads recon-config.json and the tracker from SharePoint with the signed-in user's session, then renders. */
(async function () {
  const box = document.getElementById('body');
  const safe = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const show = (title, lines) => { box.innerHTML = '<div class="card"><h2>' + safe(title) + '</h2>' + lines.map((l) => '<div class="hint">' + safe(l) + '</div>').join('') + '</div>'; };
  const get = async (url, what) => {
    const r = await fetch(url, { credentials: 'same-origin', cache: 'no-store' });
    if (!r.ok) throw new Error(what + ' could not be read (HTTP ' + r.status + '): ' + url);
    return r;
  };
  try {
    show('Loading…', ['Reading the recon tracker from SharePoint.']);
    const qs = new URLSearchParams(location.search);
    const here = location.href.replace(/[?#].*$/, '').replace(/[^\/]*$/, '');
    const cfg = await (await get(qs.get('config') || here + 'recon-config.json', 'recon-config.json')).json();
    let D, warnings = [];
    if (cfg.dataUrl) D = await (await get(cfg.dataUrl, 'The published dataset')).json();
    else {
      const url = cfg.trackerUrl || (cfg.siteUrl || '').replace(/\/$/, '') + "/_api/web/GetFileByServerRelativePath(decodedurl='" + encodeURIComponent(String(cfg.trackerPath || '').replace(/'/g, "''")) + "')/$value";
      if (!cfg.trackerUrl && !(cfg.siteUrl && cfg.trackerPath)) throw new Error('recon-config.json needs siteUrl and trackerPath (or trackerUrl / dataUrl).');
      const sheets = await readXlsx(await (await get(url, 'The recon tracker workbook')).arrayBuffer());
      const workbook = { getWorksheets: () => Object.keys(sheets).map((n) => ({ getName: () => n, getUsedRange: () => (sheets[n].length ? { getValues: () => sheets[n] } : undefined) })) };
      /* ?asOf=yyyy-MM-dd shows the page as of an earlier day */
      const today = qs.get('asOf') || new Intl.DateTimeFormat('en-CA', cfg.timeZone ? { timeZone: cfg.timeZone } : {}).format(new Date());
      const inspect = qs.get('mode') === 'inspect';
      const res = main(workbook, JSON.stringify(cfg), today, inspect ? 'inspect' : 'build');
      warnings = res.warnings;
      if (inspect || !res.ok) { show(inspect ? 'Tracker inspection' : 'No tracker rows could be read', ['Sheets: ' + Object.keys(sheets).join(' · ')].concat(warnings)); return; }
      D = JSON.parse(res.dataJson);
    }
    startDashboard(D);
    if (warnings.length) {
      const d = document.createElement('details');
      d.className = 'note';
      d.innerHTML = '<summary>' + warnings.length + ' tracker warning' + (warnings.length > 1 ? 's' : '') + '</summary>' + warnings.map((w) => '<div>' + safe(w) + '</div>').join('');
      document.querySelector('.wrap').appendChild(d);
    }
  } catch (e) {
    show('The Reconciliation Monitor could not load', [e && e.message ? e.message : String(e)]);
    throw e;
  }
})();
