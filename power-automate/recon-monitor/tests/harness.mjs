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
  return new Function(js + '\nreturn { main, scan, findings, cell, monthList, dayIdx, dayFlags, parseDay, KEY };')();
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

const FLAG_WORDS = { '1111111': 'Daily', '1111100': 'Mon-Fri', '0000000': 'N/A', '1111110': 'Mon - Sat' };
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const flagsToText = (f, k) => FLAG_WORDS[f] && k % 3 ? FLAG_WORDS[f] : k % 3 === 1 ? [...f].map((c, j) => (c === '1' ? DAYS[j] : '')).filter(Boolean).join(', ') || 'N/A' : f;

/**
 * Turns a dashboard dataset {start, acc, ifs, f, sch, x} into tracker sheets (one per interface + schedule tab),
 * mixing date and schedule notations so the readers are exercised.
 */
export function datasetToWorkbook(D, scheduleSheet = 'Recon Report Schedule') {
  const sheets = {};
  D.ifs.forEach((n) => { sheets[n] = [['Recon tracker — ' + n], [], ['Date', 'Account', 'Received', 'Loaded', 'Error', 'Pending / WB', 'Variance', 'Ticket #', 'Comments']]; });
  D.f.forEach((r, k) => {
    const day = addDays(D.start, r[0]);
    const [y, m, d] = day.split('-');
    const date = k % 5 === 0 ? `${+m}/${+d}/${y}` : k % 7 === 0 ? day : serial(day);
    sheets[D.ifs[r[2]]].push([date, D.acc[r[1]], r[3], r[4], r[5], r[6], r[7] == null ? '' : r[7], r[8] ? 'INC' + (1000 + k) : '', '']);
  });
  (D.x || []).forEach(([d, a, i], k) => {
    sheets[D.ifs[i]].push([serial(addDays(D.start, d)), D.acc[a], '', '', '', '', '', '', ['N/A', 'No run today', 'Holiday', 'On hold - client request'][k % 4]]);
  });
  const types = [...new Set(Object.values(D.sch).flatMap((s) => Object.keys(s)))];
  const sch = [['Recon Report Schedule'], ['Account', ...types]];
  Object.entries(D.sch).forEach(([acc, s], r) => sch.push([acc, ...types.map((t, c) => (s[t] == null ? '' : flagsToText(s[t], r + c)))]));
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
