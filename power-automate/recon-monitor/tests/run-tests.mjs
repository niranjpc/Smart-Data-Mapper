// Self-contained tests on synthetic data (no customer data needed):  node tests/run-tests.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { ROOT, loadScript, mockWorkbook, datasetToWorkbook, serial, addDays } from './harness.mjs';

const os = loadScript();
let n = 0;
const test = (name, fn) => { fn(); n++; console.log('ok  ' + name); };
const START = '2026-06-29'; // Monday
const day = (k) => serial(addDays(START, k));
const run = (sheets, cfg = {}, asOf = '') => {
  const r = os.main(mockWorkbook(sheets), JSON.stringify({ startDate: START, minRowsForDefaultDay: 1, ...cfg }), asOf);
  return { r, data: JSON.parse(r.dataJson), digest: JSON.parse(r.digestJson) };
};
const HDR = ['Date', 'Account', 'Received', 'Loaded', 'Error', 'Pending', 'Variance', 'Ticket', 'Comments'];
const cfg2 = { interfaces: [{ name: 'Enrollment IB' }, { name: 'Claims IB', columns: { received: ['paper', '\\bedi\\b'], error: ['edifecs', 'i-?plus'] } }] };

test('schedule expressions', () => {
  const f = os.dayFlags;
  assert.equal(f('Daily'), '1111111'); assert.equal(f('Mon-Fri'), '1111100'); assert.equal(f('Weekdays'), '1111100');
  assert.equal(f('Tue, Thu'), '0101000'); assert.equal(f('Monday to Wednesday'), '1110000'); assert.equal(f('Sat - Mon'), '1000011');
  assert.equal(f('N/A'), '0000000'); assert.equal(f(''), '0000000'); assert.equal(f('0111000'), '0111000'); assert.equal(f('Tuesdays'), '0100000');
  assert.equal(f('Weekly'), null); assert.equal(f('1st of month'), null);
});

test('date notations', () => {
  const d = Date.UTC(2026, 6, 4) / 86400000;
  assert.equal(os.parseDay('2026-07-04', 'MDY'), d); assert.equal(os.parseDay('7/4/2026', 'MDY'), d); assert.equal(os.parseDay('4/7/2026', 'DMY'), d);
  assert.equal(os.parseDay('04-Jul-26', 'MDY'), d); assert.equal(os.parseDay('July 4, 2026', 'MDY'), d); assert.equal(os.parseDay('n/a', 'MDY'), null);
});

test('feed states, tiles, merging, aliases and explained rows', () => {
  const sheets = {
    'Enrollment IB': [['Tracker'], HDR,
      [day(0), 'Plan A', 100, 100, 0, 0, 0, '', ''],                 // reconciled
      [day(0), 'Plan B - East', 50, 40, 0, 0, 10, 'INC1', ''],       // variance, ticketed
      [day(0), 'Plan B - West', 30, 20, 0, 0, -5, '', ''],           // merged into Plan B: |10| + |-5|, ticket kept
      [day(0), 'Plan C', '', '', '', '', '', '', 'Holiday - no run'],// explained -> not "missing"
      [day(0), 'Old Plan', 9, 9, 0, 0, 0, '', ''],                   // ignored
      [day(1), 'Plan A', '1,200', '1,150', 0, 0, '(50)', 'N/A', ''], // text numbers, "N/A" is not a ticket
      [day(9), 'Plan A', 1, 1, 0, 0, 0, '', '']],                    // beyond asOf -> ignored
    'Claims IB': [['Date', 'Customer', 'Paper', 'EDI', 'Loaded', 'Edifecs errors', 'I-Plus errors', 'Pending', 'Variance', 'Jira'],
      [day(0), 'Plan A', 10, 90, 95, 2, 3, 0, 0, '']],
    'Recon Report Schedule': [['Account', 'Enrollment IB', 'Claims IB'], ['Plan A', 'Daily', 'Mon-Fri'], ['Plan B - East', 'Mon-Fri', 'N/A'], ['Plan C', 'Daily', 'Daily'], ['Plan D', 'Mon', '']],
  };
  const { r, data, digest } = run(sheets, { ...cfg2, accountAliases: { 'Plan B - East': 'Plan B', 'Plan B - West': 'Plan B' }, ignoreAccounts: ['old plan'] }, addDays(START, 1));
  assert.equal(data.asOf, addDays(START, 1));
  assert.deepEqual(data.acc, ['Plan B', 'Plan A', 'Plan C', 'Plan D']);
  const fact = (a, i, d) => data.f.find((x) => x[0] === d && x[1] === data.acc.indexOf(a) && x[2] === i);
  assert.deepEqual(fact('Plan B', 0, 0).slice(3), [80, 60, 0, 0, 15, 1]);
  assert.deepEqual(fact('Plan A', 1, 0).slice(3), [100, 95, 5, 0, 0, 0]);          // paper + EDI, Edifecs + I-Plus
  assert.deepEqual(fact('Plan A', 0, 1).slice(3), [1200, 1150, 0, 0, 50, 0]);
  assert.deepEqual(data.x, [[0, data.acc.indexOf('Plan C'), 0]]);
  // Mon: in scope = A enr, A claims, B enr, C enr, C claims, D enr = 6; reconciled A enr + A claims; B variance (ticket); C enr explained; C claims + D enr missing
  const day0 = JSON.parse(os.main(mockWorkbook(sheets), JSON.stringify({ startDate: START, minRowsForDefaultDay: 3, ...cfg2, accountAliases: { 'Plan B - East': 'Plan B', 'Plan B - West': 'Plan B' }, ignoreAccounts: ['Old Plan'] }), addDays(START, 1)).digestJson).daily;
  assert.deepEqual(day0.tiles, { inScope: 6, fullyReconciled: 2, notFullyReconciled: 1, trackedViaTicket: 1, notTrackedViaTicket: 0, notReported: 2, varianceSum: 15, reconciledRate: 67 });
  assert.deepEqual(day0.notReported.map((x) => x.customer + '/' + x.interface), ['Plan C/Claims IB', 'Plan D/Enrollment IB']);
  assert.equal(digest.daily.tiles.notTrackedViaTicket, 1);                          // Tue: Plan A 50 without ticket
  assert.equal(r.needsAttention, true);
  assert.match(r.subject, /Tue 6\/30 · 0% reconciled · 1 without ticket · 4 not reported/); // Tue: A claims, B enr, C enr, C claims
  assert.ok(r.warnings.some((w) => /Plan A/.test(w) && /accountAliases/.test(w)), 'unmapped account warning');
  assert.ok(!/<script|<\/script/i.test(r.emailHtml));
});

test('schedule layouts A and B, filter column, missing sheets', () => {
  const rows = [HDR.concat('Entity'), [day(0), 'P', 5, 5, 0, 0, 0, '', '', 'Supplier'], [day(0), 'P', 7, 6, 0, 0, 1, '', '', 'Practitioner']];
  const ifs = { interfaces: [{ name: 'Provider IB – Supplier', sheet: 'Provider', filterColumn: 'entity', filterEquals: 'supplier' }, { name: 'Provider IB – Practitioner', sheet: 'Provider', filterColumn: 'entity', filterEquals: 'practitioner' }, { name: 'Gone' }] };
  const A = run({ Provider: rows, 'Recon Report Schedule': [['Customer', 'Interface', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], ['P', 'Provider IB', 'Y', 'Y', '', '', 'x', '', '']] }, ifs);
  assert.deepEqual(A.data.sch, { P: { 'Provider IB': '1100100' } });
  assert.deepEqual(A.data.f.map((x) => x.slice(2, 4).concat(x[7])), [[0, 5, 0], [1, 7, 1]]);
  assert.ok(A.r.warnings.some((w) => /'Gone' not found/.test(w)));
  const B = run({ Provider: rows, Sched: [['Account', 'Feed', 'Frequency'], ['P', 'Provider IB', 'Mon, Wed'], ['P', 'Unknown feed', 'Daily']] }, { ...ifs, scheduleSheet: 'sched' });
  assert.deepEqual(B.data.sch, { P: { 'Provider IB': '1010000' } });
  assert.ok(B.r.warnings.some((w) => /Unknown feed/.test(w)));
});

test('volume anomaly, inspect mode and script-safe JSON', () => {
  const rows = [HDR];
  for (let d = 0; d < 10; d++) rows.push([day(d), 'Big <Plan>', d === 8 ? 5000 : 1000, d === 8 ? 5000 : 1000, 0, 0, 0, '', '']);
  const { r, digest } = run({ 'Enrollment IB': rows, 'Recon Report Schedule': [['Account', 'Enrollment IB'], ['Big <Plan>', 'Daily']] }, { interfaces: [{ name: 'Enrollment IB' }] });
  assert.deepEqual(digest.monthly.volumeAnomalies.map((x) => [x.date, x.received, x.usual, x.changePct]), [[addDays(START, 8), 5000, 1000, 400]]);
  assert.ok(!r.dataJson.includes('<'), 'dataJson must not contain "<"');
  assert.equal(digest.history.scorecard[0].rates.length, 2);
  const insp = os.main(mockWorkbook({ 'Enrollment IB': rows }), JSON.stringify({ startDate: START, interfaces: [{ name: 'Enrollment IB' }] }), '', 'inspect');
  assert.ok(insp.warnings.some((w) => /^\[inspect\] Enrollment IB .* received: Received .* variance: Variance/.test(w)));
});

test('sample config parses and the template still holds its placeholder', () => {
  const c = JSON.parse(readFileSync(path.join(ROOT, 'config/recon-config.sample.json'), 'utf8'));
  assert.equal(c.interfaces.length, 20);
  const t = readFileSync(path.join(ROOT, 'dashboard/recon-monitor.template.html'), 'utf8');
  assert.equal(t.split('"__RECON_DATA__"').length, 2);
});

// ---- the real report layout (default configuration) ----
test('header formulas', () => {
  const v = { A: 10, B: 2, C: 3, D: 4, E: 1 };
  const f = (e) => os.evalFormula(e, (L) => (L in v ? v[L] : null));
  assert.equal(f('A-(B+C+D+E)'), 0); assert.equal(f('(A + B) - (C+D+E)'), 4); assert.equal(f('B-(D+E)'), -3);
  assert.equal(f('A-(B+Z)'), null); assert.equal(f('A-(B'), null); assert.equal(f(''), null);
});

test('report layout: lettered headers, template rows, empty days, blank variance, two ticket columns', () => {
  const T = ['', '', '', 'Transaction Counts'];
  const H = ['Account', 'Date', 'File Name', 'Received\n(A)', 'EDIFECS Errors\n(B)', 'I-Plus Errors\n(C)', 'Workbasket\n(D)', 'Loaded into HRP\n(E)', 'Variance\nA-(B+C+D+E)', 'Jira Ticket Number\n\n(to resolve Errors)', 'Jira Ticket Number\n\n(to investigate transaction variance)', 'Comments '];
  const C = ['Account ', 'Date', 'System', 'Received - Paper\n(A)', 'Received - EDI\n(B)', 'EDIFECS Error\n( C )', 'I-Plus Errors\n(D)', 'Loaded into HRP\n( E )\n\nE = F+G+I', 'Finalized\n(F)', 'Denied\n(G)', 'Workbasket\n(I) \nI = J + K', 'Need Repair \n(J)', 'Need Review\n(K)', 'Variance\n\n(A + B) - (C+D+E)', 'Jira Ticket Number\n(to resolve Errors', 'Comments'];
  const sheets = {
    'Enrollment IB': [['', 'Back'], ['', 'Load Summary Report'], T, H,
      ['PlanOne', '', 'MD', '', '', '', '', '', 0, '', '', ''],                 // template row without a date: ignored
      ['PlanOne', day(0), 'f1', 100, 5, 0, 0, 95, '', '', '', ''],              // variance blank -> header formula -> 0
      ['Planone ', day(0), 'f2', 50, 0, 0, 0, 40, '', 'ERR-1', '', ''],         // second file same day: summed, 10 short, ticket
      ['Plan Two', day(0), 'No file for the day', '', '', '', '', '', 0, '', '', ''], // explained
      ['PlanTwo', day(1), '', '', '', '', '', '', 0, '', '', ''],            // dated but nothing entered: not reported
      ['Planone', day(1), 'f', 7, 0, 0, 0, 7, 3, '', '', '']],                  // a stored variance wins over the formula
    'Claims IB': [['', 'Back'], T, C,
      ['Planone', day(0), 'SDS', 10, 90, 2, 3, '', 70, 20, '', 4, 1, '', '', '']], // E and I blank -> derived: I=5, E=95, variance 0
    'Recon Report Schedule': [['Recon Report Schedule — Weekly Calendar (by Account)'], ['Account', 'Reconciliation Type', 'Report Type', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Monthly', 'Notes'],
      ['Planone', 'Transactional Integrity', 'Enrollment IB', 'X', 'X', 'X', 'X', 'X', ' ', ' ', ' ', ''], ['', '', 'Claims IB', 'X', ' ', '', '', '', '', '', '', ''], ['', '', 'TM Aging Summary', 'X', 'X', 'X', 'X', 'X', '', '', '', ''],
      ['PlanTwo', 'Transactional Integrity', 'Enrollment IB', 'X', 'X', '', '', '', '', '', 'Monthly', '']],
  };
  const { r, data } = run(sheets, { interfaces: undefined, accountAliases: { Planone: 'Plan One Health' } }, addDays(START, 1));
  const w = r.warnings.filter((x) => !/not found|accountAliases/.test(x));
  assert.deepEqual(w, []);
  const med = data.acc.indexOf('Plan One Health'), zing = data.acc.indexOf('Plan Two'); // first spelling seen names the account; 'PlanTwo' joins it
  assert.equal(data.acc.length, 2);
  const f = Object.fromEntries(data.f.map((x) => [x.slice(0, 3).join('|'), x.slice(3)]));
  assert.deepEqual(f[`0|${med}|0`], [150, 135, 5, 0, 10, 1]);
  assert.deepEqual(f[`1|${med}|0`], [7, 7, 0, 0, 3, 0]);
  assert.deepEqual(f[`0|${med}|2`], [100, 95, 5, 5, 0, 0]);
  assert.equal(data.f.length, 3);
  assert.deepEqual(data.x, [[0, zing, 0]]);
  assert.deepEqual(data.sch, { 'Plan One Health': { 'Enrollment IB': '1111100', 'Claims IB': '1000000' }, 'Plan Two': { 'Enrollment IB': '1100000' } });
  assert.deepEqual(r.weekly, { inScope: 5, fullyReconciled: 1, notFullyReconciled: 2, trackedViaTicket: 1, notTrackedViaTicket: 1, notReported: 1, varianceSum: 13, reconciledRate: 33 });
});

test('stacked tables and several workbooks round-trip', () => {
  const ifs = ['Enrollment IB', 'Provider IB – Tax entity', 'Provider IB – Practitioner', 'Provider IB – Supplier', 'Encounter Submission', 'Encounter Rejections'];
  const D = { start: START, acc: ['Plan A', 'Plan B'], ifs, f: [], x: [[2, 1, 4], [3, 0, 5]],
    sch: { 'Plan A': { 'Enrollment IB': '1111111', 'Provider IB': '1111100', 'Encounter Submission': '0100000' }, 'Plan B': { 'Enrollment IB': '1111100', 'Provider IB': '0000000', 'Encounter Submission': '1000001' } } };
  for (let d = 0; d < 12; d++) for (let a = 0; a < 2; a++) for (let i = 0; i < ifs.length; i++) if ((d + a + i) % 3) D.f.push([d, a, i, 100 + d + i, 90 + d, 4, 6 + i, (d * i) % 4 ? 0 : i + 1, (d + i) % 2]);
  const wb = (from, to) => mockWorkbook(datasetToWorkbook({ ...D, f: D.f.filter((r) => r[0] >= from && r[0] <= to), x: D.x.filter((r) => r[0] >= from && r[0] <= to) }));
  const two = { getWorksheets: () => wb(0, 6).getWorksheets().concat(wb(7, 11).getWorksheets()) };   // archived week + current week
  const r = os.main(two, JSON.stringify({ startDate: START, accountAliases: { 'Plan A': 'Plan A', 'Plan B': 'Plan B' } }), addDays(START, 11));
  assert.deepEqual(r.warnings.filter((x) => !/not found|accountAliases|no table with a date column/.test(x)), []); // the made-up tracker has 6 of the 20 interfaces
  const data = JSON.parse(r.dataJson), key = (x) => x.slice(0, 3).join('|');
  const remap = (x) => [x[0], data.acc.indexOf(D.acc[x[1]]), data.ifs.indexOf(ifs[x[2]]), ...x.slice(3)];
  assert.deepEqual(Object.fromEntries(data.f.map((x) => [key(x), x])), Object.fromEntries(D.f.map(remap).map((x) => [key(x), x])));
  assert.deepEqual(data.x.map(key).sort(), D.x.map(remap).map(key).sort());
  assert.deepEqual(data.sch, D.sch);
});

// ---- SharePoint live page: built-in .xlsx reader + generated page ----
const reader = new Function(readFileSync(path.join(ROOT, 'sharepoint/src/xlsx-reader.js'), 'utf8') + '\nreturn readXlsx;')();
const buf = readFileSync(path.join(ROOT, 'tests/fixtures/tracker.xlsx'));
const sheets = await reader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
test('xlsx reader returns sheets, serial dates, text, numbers, booleans and blanks', () => {
  assert.deepEqual(Object.keys(sheets), ['Enrollment IB', 'Recon Report Schedule']);
  const e = sheets['Enrollment IB'];
  assert.deepEqual(e[2], HDR);
  assert.deepEqual(e[4], [day(0), 'Plan B & <Co>', 50, 40, 0, 0, 10, 'INC1', '']);
  assert.deepEqual(e[6], [day(1), 'Plan B & <Co>', '', '', '', '', '', '', 'Holiday']);
  assert.ok(e.every((r) => r.length === 9));
  assert.deepEqual(sheets['Recon Report Schedule'][1], ['Plan A', 'Daily', true]);
});
test('workbook read from .xlsx gives the expected dashboard numbers', () => {
  const { data, digest } = run(sheets, { interfaces: [{ name: 'Enrollment IB' }] });
  assert.deepEqual(data.x, [[1, 1, 0]]);
  assert.deepEqual(digest.weekly.tiles, { inScope: 4, fullyReconciled: 1, notFullyReconciled: 2, trackedViaTicket: 1, notTrackedViaTicket: 1, notReported: 0, varianceSum: 60, reconciledRate: 33 });
});
test('SharePoint page is built from the current sources', async () => {
  const aspx = readFileSync(path.join(ROOT, 'sharepoint/Reconciliation-Monitor.aspx'), 'utf8');
  const js = aspx.slice(aspx.indexOf('<script>') + 8, aspx.lastIndexOf('</script>'));
  new Function(js); // parses
  for (const part of ['sharepoint/src/xlsx-reader.js', 'sharepoint/src/loader.js']) assert.ok(js.includes(readFileSync(path.join(ROOT, part), 'utf8')), part + ' changed — run node sharepoint/build.mjs');
  assert.ok(js.includes('function startDashboard(D){') && !js.includes('__RECON_DATA__'));
});

console.log(`\n${n} tests passed`);
