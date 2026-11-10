// SharePoint-lists dashboard, end to end against the mock SharePoint, on the data of an existing dashboard HTML:
//   node tests/sharepoint-lists.mjs "<Reconciliation-Monitor.html>"
// setup flow → daily flow (archived weeks + current week) twice → every list row checked against the original page.
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { loadScript, mockWorkbook, datasetToWorkbook, runPage, scriptOf } from './harness.mjs';
import { createMockSharePoint } from './sp-mock.mjs';

const src = process.argv[2];
if (!src) { console.error('usage: node tests/sharepoint-lists.mjs <original dashboard html>'); process.exit(2); }
const body = scriptOf(readFileSync(src, 'utf8'));
const line = body.split('\n').find((l) => l.startsWith('const D='));
const cut = line.lastIndexOf(';D.x=');
const raw = JSON.parse(line.slice(8, cut)); raw.x = JSON.parse(line.slice(cut + 5).replace(/;\s*$/, ''));
const NM = new Function('return ' + /const NM=(\{.*?\});/.exec(body)[1])();

const SITE = 'https://contoso.sharepoint.com/sites/TechOps';
const TITLES = ['Recon Summary', 'Recon Interfaces', 'Recon Issues', 'Recon Daily Trend', 'Recon Weekly Scorecard', 'Recon Recurring Issues'];
const cfg = JSON.stringify({ startDate: raw.start, siteUrl: SITE, accountAliases: NM, ignoreAccounts: raw.acc.filter((n) => NM[n] === undefined), varianceWhenMissing: 'none' });
const os = loadScript();
const sp = createMockSharePoint(SITE);
const send = (batches, what) => {
  let ops = 0;
  for (const b of batches) {
    const r = sp.batch('multipart/mixed; boundary=' + b.boundary, b.body);   // what the flow's HTTP action sends
    assert.equal(r.status, 200, what + ': ' + r.body);
    const bad = /HTTP\/1\.1 [45]\d\d[^]*?"value":("[^"]*")/.exec(r.body);      // the flow's error check
    assert.ok(!bad, `${what} / ${b.label}: ${bad && bad[1]}`);
    ops += b.operations;
  }
  return ops;
};

// ---- setup flow ----
const setup = os.main(mockWorkbook({}), cfg, '', 'setup');
assert.deepEqual(setup.warnings, []);
const setupOps = send(setup.spBatches, 'setup');
assert.deepEqual([...sp.lists.keys()], TITLES);
const summary = sp.lists.get('Recon Summary');
assert.equal(summary.fields.get('Title').title, 'Customer');
assert.ok(summary.views.get('This week KPI').formatter.rowFormatter, 'KPI card formatter');
console.log(`setup: ${setup.spBatches.length} batches, ${setupOps} operations, ${[...sp.lists.values()].reduce((n, l) => n + l.views.size - 1, 0)} views`);
// running setup twice must fail loudly (lists exist), not duplicate anything
assert.throws(() => send(os.main(mockWorkbook({}), cfg, '', 'setup').spBatches, 'setup again'), /already exists/);

// ---- daily flow: archived weeks via extract, current week with archiveJson + sharepointJson ----
const part = (a, b) => mockWorkbook(datasetToWorkbook({ ...raw, f: raw.f.filter((r) => r[0] >= a && r[0] <= b), x: raw.x.filter((r) => r[0] >= a && r[0] <= b) }));
const archiveJson = '[' + [part(0, 62), part(63, 125)].map((w) => os.main(w, cfg, raw.asOf, 'extract').extractJson).join(',') + ']';
const daily = () => {
  const r = os.main(part(126, 999), cfg, raw.asOf, 'build', archiveJson, JSON.stringify(sp.info(TITLES)));
  assert.deepEqual(r.warnings, []);
  assert.equal(r.dataJson, ''); assert.equal(r.digestJson, '');
  return r;
};
const r1 = daily();
const ops1 = send(r1.spBatches, 'daily 1');
const count = () => Object.fromEntries(TITLES.map((t) => [t, sp.lists.get(t).items.size]));
const c1 = count();
const ids1 = [...summary.items.keys()];
const r2 = daily();
const ops2 = send(r2.spBatches, 'daily 2');
assert.deepEqual(count(), c1, 'second run replaces the rows instead of adding to them');
assert.ok(ids1.every((id) => !summary.items.has(id)), 'old rows removed');
const size = (r) => Math.round(JSON.stringify(r).length / 1024);
console.log(`daily: ${r1.spBatches.length} batches / ${ops1} ops (first), ${r2.spBatches.length} / ${ops2} (with deletes); script output ${size(r2)} KB`);
console.log('rows per list', c1);
assert.ok(size(r2) < 4500, 'Run script output must stay well under the 5 MB limit');

// ---- every row against the original page ----
const orig = runPage(body);
const items = (t) => [...sp.lists.get(t).items.values()];
const ALL = '(All customers)';
const ai = (name) => (name === ALL ? -1 : orig.ACC.indexOf(name));
const months = [];
{ const MS = 86400000, a = new Date(orig.T0), b = new Date(orig.T0 + orig.LAST * MS); let y = a.getUTCFullYear(), m = a.getUTCMonth(); while (y < b.getUTCFullYear() || (y === b.getUTCFullYear() && m <= b.getUTCMonth())) { months.push([y, m]); if (++m > 11) { m = 0; y++; } } }
const dayOf = (y, m, d) => Math.round((Date.UTC(y, m, d) - orig.T0) / 86400000);
const [my, mm] = months[months.length - 1];
const wk = Math.floor(orig.DEF / 7) * 7;
const P = { 'Latest day': ['d', orig.DEF, orig.DEF], 'This week': ['w', wk, Math.min(orig.LAST, wk + 6)], 'This month': ['m', Math.max(0, dayOf(my, mm, 1)), Math.min(orig.LAST, dayOf(my, mm + 1, 0))] };
const texts = (h) => [...h.matchAll(/<div class="fi"><i>\d+<\/i><div>(.*?)<\/div><\/div>/g)].map((m) => m[1].replace(/<\/?b>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&'));
let checks = 0;
// the original page runs in a vm sandbox: its arrays have another prototype, so compare plain JSON copies
const plainEq = (a, b, msg) => assert.deepEqual(JSON.parse(JSON.stringify(a)), JSON.parse(JSON.stringify(b)), msg);
for (const row of items('Recon Summary')) {
  const [v, d1, d2] = P[row.Period];
  Object.assign(orig.S, { p: v, a: ai(row.Title), f: -1, mo: months.length - 1 });
  const o = orig.scan(d1, d2, ai(row.Title));
  assert.deepEqual([row.InScope, row.Reconciled, row.NotReconciled, row.Ticketed, row.NotTicketed, row.NotReported, row.VarianceSum, row.Rate ?? null],
    [o.scope, o.g, o.att, o.attT, o.attN, o.miss, Math.round(o.sum), o.rate], `summary ${row.Period} ${row.Title}`);
  assert.equal(row.Highlights || '', texts(orig.findings(d1, d2, ai(row.Title), o)).map((h, k) => `${k + 1}. ${h}`).join('\n'), `highlights ${row.Period} ${row.Title}`);
  checks++;
}
for (const row of items('Recon Interfaces')) {
  const [v, d1, d2] = P[row.Period];
  Object.assign(orig.S, { p: v, a: ai(row.Customer), f: -1 });
  const b = orig.scan(d1, d2, ai(row.Customer), 1).byI[orig.IFS.indexOf(row.Title)];
  assert.deepEqual([row.InScope, row.Received, row.Loaded, row.Errors, row.Pending, row.VarianceSum, row.Reconciled, row.Ticketed, row.NotTicketed, row.NotReported],
    [b.scope, b.rec, b.ld, b.err, b.pend, Math.round(b.sum), b.g, b.attT, b.attN, b.miss], `interface ${row.Period} ${row.Customer} ${row.Title}`);
  checks++;
}
for (const [name, [v, d1, d2]] of Object.entries(P)) {
  Object.assign(orig.S, { p: v, a: -1, f: -1 });
  const o = orig.scan(d1, d2, -1);
  const mine = items('Recon Issues').filter((x) => x.Period === name);
  const vs = mine.filter((x) => x.IssueType === 'Variance').sort((p, q) => p.SortKey - q.SortKey);
  plainEq(vs.map((x) => [x.Customer, x.Title, x.Amount, x.Ticket]), o.items.slice(0, 250).map((x) => [orig.ACC[x.a], orig.IFS[x.i], Math.round(Math.abs(x.v)), x.t ? 'Tracked' : 'Not tracked']), 'variances ' + name);
  assert.equal(mine.filter((x) => x.IssueType === 'Not reported').length, Math.min(300, o.missL.length), 'not reported ' + name);
  plainEq(mine.filter((x) => x.IssueType === 'Volume anomaly').sort((p, q) => p.SortKey - q.SortKey).map((x) => [x.Customer, x.Title, x.Amount, x.Usual]),
    o.anom.slice(0, 50).map((x) => [orig.ACC[x.a], orig.IFS[x.i], x.v, x.m]), 'anomalies ' + name);
  checks += 3;
}
const trend = items('Recon Daily Trend').filter((x) => x.Title === ALL).sort((p, q) => p.DayDate.localeCompare(q.DayDate));
trend.forEach((x) => { const d = Math.round((Date.parse(x.DayDate) - orig.T0) / 86400000); Object.assign(orig.S, { f: -1 }); const o = orig.scan(d, d, -1, 1); assert.deepEqual([x.Rate ?? null, x.NotReconciled, x.NotReported], [o.rate, o.att, o.miss], 'trend ' + x.DayDate); checks++; });
const card = items('Recon Weekly Scorecard').find((x) => x.Title === ALL);
const lastW = Math.floor(orig.LAST / 7);
for (let k = 1; k <= 12; k++) { const w = lastW - 12 + k; if (w < 0) continue; const o = orig.scan(w * 7, Math.min(orig.LAST, w * 7 + 6), -1, 1); assert.equal(card['W' + String(k).padStart(2, '0')] ?? null, o.rate, 'scorecard week ' + k); checks++; }
assert.deepEqual(sp.lists.get('Recon Weekly Scorecard').fields.get('W12').title, 'Wk ' + orig.md(lastW * 7));
console.log(`content: ${checks} rows and figures identical to the original dashboard`);
