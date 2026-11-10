// Parity check against an existing Reconciliation Monitor HTML (the hand-built dashboard with embedded data):
//   node tests/parity.mjs "<path to Reconciliation-Monitor.html>"
// 1. rebuilds a tracker workbook from the dashboard's raw data, 2. runs the Office Script on it,
// 3. replays the original page code and compares every number, list and highlight sentence.
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import path from 'node:path';
import { ROOT, loadScript, mockWorkbook, datasetToWorkbook, runPage, scriptOf, modelOf } from './harness.mjs';

const src = process.argv[2];
if (!src) { console.error('usage: node tests/parity.mjs <original dashboard html>'); process.exit(2); }
const html = readFileSync(src, 'utf8');
const body = scriptOf(html);

// raw (pre-alignment) dataset + alias map, straight from the original page source
const line = body.split('\n').find((l) => l.startsWith('const D='));
const cut = line.lastIndexOf(';D.x=');
const raw = JSON.parse(line.slice('const D='.length, cut));
raw.x = JSON.parse(line.slice(cut + 5).replace(/;\s*$/, ''));
const NM = new Function('return ' + /const NM=(\{.*?\});/.exec(body)[1])();

const os = loadScript();
// the original data has rows without a variance value; keep them blank instead of deriving one from the header formula
const config = { startDate: raw.start, accountAliases: NM, ignoreAccounts: raw.acc.filter((n) => NM[n] === undefined), varianceWhenMissing: 'none' };
const res = os.main(mockWorkbook(datasetToWorkbook(raw)), JSON.stringify(config), raw.asOf);
assert.deepEqual(res.warnings, [], 'script warnings');
const data = JSON.parse(res.dataJson);
const digest = JSON.parse(res.digestJson);

const orig = runPage(body);                                       // original page, original data
const tpl = runPage(scriptOf(readFileSync(path.join(ROOT, 'dashboard/recon-monitor.template.html'), 'utf8')).replace('"__RECON_DATA__"', () => res.dataJson));
const M = modelOf(data, os.KEY);
let checks = 0;
// compares canonical JSON (sorted keys); reports only the label, a full diff of 14k rows is useless
const canon = (v) => JSON.stringify(v, (k, x) => (x && typeof x === 'object' && !Array.isArray(x) ? Object.fromEntries(Object.keys(x).sort().map((n) => [n, x[n]])) : x));
const same = (a, b, msg) => { if (canon(a) !== canon(b)) throw new Error('MISMATCH: ' + msg + '\n  script  : ' + canon(a).slice(0, 300) + '\n  original: ' + canon(b).slice(0, 300)); checks++; };

// ---- dataset ----
same(data.acc, orig.ACC, 'accounts'); same(data.ifs, orig.IFS, 'interfaces'); same(data.sch, orig.D.sch, 'schedule');
const key = (r) => r[0] + '|' + r[1] + '|' + r[2];
same(Object.fromEntries(data.f.map((r) => [key(r), r])), Object.fromEntries(orig.D.f.map((r) => [key(r), r])), 'fact rows');
same(data.x.map(key).sort(), [...new Set(orig.D.x.filter((x) => x[0] <= orig.LAST).map(key))].sort(), 'explained rows');
same([data.def, M.LAST], [orig.DEF, orig.LAST], 'default day / last day');

// ---- every period, portfolio + every customer ----
const slim = (o) => ({ g: o.g, att: o.att, attT: o.attT, attN: o.attN, miss: o.miss, sum: o.sum, scope: o.scope, rate: o.rate,
  items: o.items.map((x) => [x.a, x.i, x.d, x.v, !!x.t]), missL: o.missL.map((x) => [x.a, x.i, x.d]), anom: o.anom.map((x) => [x.a, x.i, x.d, x.v, x.m]),
  byI: o.byI.map((b) => [b.g, b.att, b.attT, b.attN, b.miss, b.sum, b.scope, b.rec, b.ld, b.err, b.pend]),
  byA: o.byA.map((b) => [b.g, b.att, b.attT, b.attN, b.miss, b.sum, b.scope, b.top || null, b.topv, b.missI, b.an]) });
const texts = (h) => [...h.matchAll(/<div class="fi"><i>\d+<\/i><div>(.*?)<\/div><\/div>/g)].map((m) => m[1]);
const months = os.monthList(M);
const periods = [];
for (let d = 0; d <= M.LAST; d++) periods.push(['d', d, d, 0]);
for (let w = 0; w <= M.LAST; w += 7) periods.push(['w', w, Math.min(M.LAST, w + 6), 0]);
months.forEach(([y, m], k) => periods.push(['m', Math.max(0, os.dayIdx(M, y, m, 1)), Math.min(M.LAST, os.dayIdx(M, y, m + 1, 0)), k]));

for (const page of [orig, tpl]) for (const [V, d1, d2, mo] of periods) for (let a0 = -1; a0 < M.NA; a0++) {
  if (a0 >= 0 && V === 'd' && d1 % 9) continue;                    // per-customer daily: sample every 9th day
  const mine = os.scan(M, d1, d2, a0, -1, false);
  Object.assign(page.S, { p: V, a: a0, f: -1, mo });
  const theirs = page.scan(d1, d2, a0);
  const tag = `${page === orig ? 'original' : 'template'} ${V} ${d1}-${d2} a=${a0}`;
  same(slim(mine), slim(theirs), 'scan ' + tag);
  same(os.findings(M, V, d1, d2, a0, mine, months, mo), texts(page.findings(d1, d2, a0, theirs)), 'highlights ' + tag);
}

// ---- what the flow receives for the default day / week / month ----
Object.assign(orig.S, { p: 'd', a: -1, f: -1 });
const oD = orig.scan(orig.DEF, orig.DEF, -1);
same(res.daily, { inScope: oD.scope, fullyReconciled: oD.g, notFullyReconciled: oD.att, trackedViaTicket: oD.attT, notTrackedViaTicket: oD.attN, notReported: oD.miss, varianceSum: oD.sum, reconciledRate: oD.rate }, 'daily tiles');
same(digest.daily.highlightsHtml, texts(orig.findings(orig.DEF, orig.DEF, -1, oD)), 'daily highlights');
same(digest.daily.topVariances.map((x) => x.variance), oD.items.slice(0, 12).map((x) => x.v), 'daily top variances');

console.log(`parity OK — ${checks} comparisons, ${data.f.length} feed rows, ${M.NA} customers, ${M.LAST + 1} days`);
console.log(res.subject);
console.log('daily  ', JSON.stringify(res.daily));
console.log('weekly ', JSON.stringify(res.weekly), '·', res.weeklyLabel);
console.log('monthly', JSON.stringify(res.monthly), '·', res.monthlyLabel);
digest.daily.highlights.forEach((h, k) => console.log(`  ${k + 1}. ${h}`));
