// Test helpers: load the Office Script outside Excel, fake a workbook, run dashboard page code headlessly.
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MS = 86400000;

/** Compiles office-scripts/ReconMonitor.ts and returns its functions. */
export function loadScript() {
  const ts = readFileSync(path.join(ROOT, 'office-scripts/ReconMonitor.ts'), 'utf8');
  const js = stripTypeScriptTypes(ts);
  return new Function(js + '\nreturn { main, scan, findings, cell, monthList, dayIdx, dayFlags, parseDay, evalFormula, KEY };')();
}

/** sheets: { name: rows[][] } -> object with the ExcelScript.Workbook methods the script uses. */
export function mockWorkbook(sheets) {
  return {
    getWorksheets: () => Object.entries(sheets).map(([name, values]) => ({
      getName: () => name,
      getUsedRange: () => (values.length ? { getValues: () => values } : undefined),
    })),
  };
}

export const serial = (isoDate) => Date.parse(isoDate + 'T00:00:00Z') / MS + 25569;
export const addDays = (isoDate, n) => new Date(Date.parse(isoDate + 'T00:00:00Z') + n * MS).toISOString().slice(0, 10);

const HEAD = ['Account', 'Date', 'File Name', 'Received\n(A)', 'I-Plus Errors\n(B)', 'Workbasket\n(C)', 'Loaded into HRP\n(D)', 'Variance\nA-(B+C+D)',
  'Jira Ticket Number\n\n(to resolve Errors)', 'Jira Ticket Number\n\n(to investigate transaction variance)', 'Comments '];

/**
 * Turns a dashboard dataset {start, acc, ifs, f, sch, x} into a tracker workbook laid out like the real report:
 * one sheet per interface with title rows and lettered headers, "Provider IB" and "Encounter Submission" holding
 * several stacked tables, template rows without dates, dated rows without counts, and a schedule tab with merged
 * account cells and X marks.
 */
export function datasetToWorkbook(D, scheduleSheet = 'Recon Report Schedule') {
  const sheets = {}, tables = [];
  D.ifs.forEach((n, i) => {
    const prov = /^Provider IB – (.*)$/.exec(n), rej = n === 'Encounter Rejections';
    const name = prov ? 'Provider IB' : rej ? 'Encounter Submission' : n;
    const head = HEAD.slice();
    if (rej) { head[1] = 'From Date'; head[2] = 'To Date'; head[3] = 'Extracted to File to be sent to Encounter Submission System\n(A)'; head[4] = 'Rejected in Encounter Submission System\n(B)'; head[6] = 'Loaded in Encounter Submission System\n(D)'; } // dated by "To Date"
    if (prov && i % 2) head[0] = '';                                          // some blocks have no "Account" header
    const rows = [[], ['', '', prov ? prov[1] : n === 'Encounter Submission' ? 'Transaction Counts - Extraction to Encounter Submission System' : rej ? 'Transaction Counts - Rejections in Encounter Submission System' : ''], ['', '', '', 'Transaction Counts'], head,
      [D.acc[0], '', '', '', '', '', '', 0, '', '', 'template row']]; // no date -> ignored
    tables[i] = { name, rows, dc: rej ? 2 : 1 };
  });
  const row = (i, date, acc, cells) => { const r = [acc, '', '', ...cells]; r[tables[i].dc] = date; if (tables[i].dc === 2) r[1] = 'NA'; tables[i].rows.push(r); };
  D.f.forEach((r, k) => {
    const day = addDays(D.start, r[0]);
    const [y, m, d] = day.split('-');
    const date = k % 5 === 0 ? `${m}-${d}-${y}` : k % 7 === 0 ? day : serial(day);
    const acc = k % 4 === 0 ? D.acc[r[1]].toUpperCase() : k % 9 === 0 ? ' ' + D.acc[r[1]] + ' ' : D.acc[r[1]];
    row(r[2], date, acc, [r[3], r[5], r[6], r[4], r[7] == null ? '' : r[7], r[8] && k % 2 ? 'INC' + k : '', r[8] && !(k % 2) ? 'INC' + k : '', k % 11 ? '' : 'late file']);
    if (k % 50 === 0) row(r[2], serial(addDays(day, 1)), D.acc[r[1]], ['', '', '', '', 0, '', '', '']);   // dated but not filled in
  });
  (D.x || []).forEach(([d, a, i], k) => {
    const note = ['N/A', 'No run today', 'Holiday', 'On hold - client request', 'Not Applicable', 'no load for the day'][k % 6];
    row(i, serial(addDays(D.start, d)), D.acc[a], ['', '', '', '', 0, '', '', note]);
  });
  tables.forEach((t) => { sheets[t.name] = (sheets[t.name] || [['', 'Back'], ['', 'Load Summary Report']]).concat(t.rows); }); // stacked tables
  const types = [...new Set(Object.values(D.sch).flatMap((s) => Object.keys(s)))];
  const sch = [['Recon Report Schedule — Weekly Calendar (by Account)'], ['HOW TO USE: every interface is listed for every account.'],
    ['Account', 'Reconciliation Type', 'Report Type', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Monthly', 'Notes', 'comments']];
  Object.entries(D.sch).forEach(([acc, s]) => {
    types.forEach((t, c) => sch.push([c ? '' : acc, c ? '' : 'Transactional Integrity', t, ...[...(s[t] || '0000000')].map((f) => (f === '1' ? 'X' : ' ')), ' ', s[t] == null ? 'NOT IN SCOPE' : ' ', ' ']));
    sch.push(['', '', 'TM Aging Summary', 'X', 'X', 'X', 'X', 'X', ' ', ' ', ' ', ' ', ' ']);
  });
  sheets[scheduleSheet] = sch;
  return sheets;
}

/** Runs a dashboard page <script> body with a stub DOM and returns its internals. */
export function runPage(scriptBody) {
  const el = () => ({ classList: { toggle() {} }, style: {}, dataset: {}, set innerHTML(v) {}, get innerHTML() { return ''; }, set textContent(v) {}, querySelectorAll: () => [], href: '' });
  const els = new Map();
  const get = (s) => { if (!els.has(s)) els.set(s, el()); return els.get(s); };
  const ctx = vm.createContext({ document: { querySelector: get, getElementById: get, querySelectorAll: () => [] }, window: { scrollTo() {} }, console });
  vm.runInContext(scriptBody + '\n;globalThis.__x={scan,findings,cell,S,ACC,IFS,D,MOS,DEF,LAST,T0,MS,md,sync};', ctx);
  return ctx.__x;
}

export const scriptOf = (html) => html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'));

/** Model object for calling the Office Script's scan()/findings() directly from a dataJson dataset. */
export function modelOf(D, KEY) {
  const NI = D.ifs.length;
  const F = new Map(D.f.map((r) => [(r[1] * NI + r[2]) * KEY + r[0], r]));
  const X = new Set(D.x.map(([d, a, i]) => (a * NI + i) * KEY + d));
  const T0 = Date.parse(D.start + 'T00:00:00Z') / MS;
  return { acc: D.acc, ifs: D.ifs, typ: D.typ, sch: D.sch, F, X, NA: D.acc.length, NI, T0, LAST: Date.parse(D.asOf + 'T00:00:00Z') / MS - T0, DEF: D.def, anomMin: 200, anomFactor: 3 };
}
