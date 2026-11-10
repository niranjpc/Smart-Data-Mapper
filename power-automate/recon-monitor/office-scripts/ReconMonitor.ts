/**
 * Reconciliation Monitor — Office Script (run from Power Automate with "Run script").
 *
 * Reads the recon tracker workbook, applies the Reconciliation Monitor rules and returns
 *   - dataJson   : the dataset the dashboard page renders (injected into the HTML template by the flow)
 *   - digestJson : every number the dashboard shows for the current day / week / month + history
 *   - subject / emailHtml / needsAttention : ready-made daily digest for mail or Teams
 *   - warnings   : anything in the workbook that did not match the configuration
 *
 * Parameters (all optional):
 *   configJson : contents of recon-config.json ("" = built-in defaults)
 *   asOf       : yyyy-MM-dd upper bound for the data (normally "today" in the business time zone)
 *   mode       : "build" (default), "inspect" (lists sheets + detected columns in warnings) or
 *                "extract" (reads this workbook only and returns its rows in extractJson — run on each archived week)
 *   archiveJson: JSON array of the extractJson outputs of the archived weeks, oldest first ("" = none).
 *                Rows of this workbook replace archived rows for the same customer, interface and day.
 *   sharepointJson: for the SharePoint-lists dashboard: JSON array [{ title, type, items: [{ Id }] }] — each list's
 *                item entity type and current item ids. The result then carries spBatches: $batch requests that
 *                replace the lists' contents. With mode "setup", spBatches create the lists, columns and views.
 */

type Cell = string | number | boolean;
/** [day, account, interface, received, loaded, error, pending, variance|null, ticket 0/1] */
type Fact = [number, number, number, number, number, number, number, number | null, number];

interface ColumnMap {
  date?: string[]; account?: string[]; received?: string[]; loaded?: string[]; error?: string[];
  pending?: string[]; variance?: string[]; ticket?: string[]; comment?: string[];
}
interface InterfaceCfg {
  name: string; scheduleType?: string; sheet?: string; block?: string; columns?: ColumnMap;
  variance?: string; filterColumn?: string; filterEquals?: string;
}
interface Segment { hr: number; end: number; titles: string[]; accCol: number }
interface ReconConfig {
  startDate?: string; homeUrl?: string; dashboardUrl?: string; dateOrder?: string; siteUrl?: string; spBatchSize?: number;
  minRowsForDefaultDay?: number; scheduleSheet?: string; scheduleIgnoreTypes?: string[];
  accountAliases?: { [raw: string]: string }; ignoreAccounts?: string[]; unmappedAccounts?: string;
  explainedCommentPattern?: string; varianceWhenMissing?: string;
  anomalyMinMedian?: number; anomalyFactor?: number;
  columns?: ColumnMap; interfaces?: InterfaceCfg[];
}
interface Tiles {
  inScope: number; fullyReconciled: number; notFullyReconciled: number; trackedViaTicket: number;
  notTrackedViaTicket: number; notReported: number; varianceSum: number; reconciledRate: number;
}
/**
 * Rows of one workbook, compact: f = [day, customer, interface, received, loaded, error, pending, |variance| | null, ticket 0/1],
 * x = [day, customer, interface] (explained, not reported); day counts from start, customer / interface index acc / ifs.
 */
interface WeekRows { start: string; acc: string[]; ifs: string[]; f: (number | null)[][]; x: number[][] }
interface ReconResult {
  ok: boolean; asOf: string; defaultDay: string; generated: string;
  dataJson: string; digestJson: string;
  daily: Tiles; weekly: Tiles; monthly: Tiles;
  dailyLabel: string; weeklyLabel: string; monthlyLabel: string;
  needsAttention: boolean; subject: string; emailHtml: string; warnings: string[]; extractJson: string; spBatches: SpBatch[];
}

interface Bucket {
  g: number; att: number; attT: number; attN: number; miss: number; sum: number; scope: number;
  rec: number; ld: number; err: number; pend: number;
  top: string; topv: number; missI: string[]; an: number;
}
interface Item { a: number; i: number; d: number; v: number; t: boolean }
interface Miss { a: number; i: number; d: number }
interface Anom { a: number; i: number; d: number; v: number; m: number; dev: number }
interface Scan {
  g: number; att: number; attT: number; attN: number; miss: number; sum: number; scope: number;
  rate: number | null; items: Item[]; missL: Miss[]; anom: Anom[]; byI: Bucket[]; byA: Bucket[];
}
interface CellState { s: string; r: Fact | null; v: number; t: boolean }
interface Model {
  acc: string[]; ifs: string[]; typ: string[]; sch: { [acc: string]: { [typ: string]: string } };
  F: Map<number, Fact>; X: Set<number>; NA: number; NI: number; T0: number; LAST: number; DEF: number;
  anomMin: number; anomFactor: number;
}
interface Group { k: string; n: number; sum: number; x: Item }
interface Rec { a: number; i: number; wk: Set<number>; n: number; sum: number; t: number; last: number; str: number; cur: number; prev: number }

const MS_DAY = 86400000;
const KEY = 100000; // day slots per (account, interface) key
const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const DEFAULT_COLUMNS: ColumnMap = {
  date: ["^date$", "^run date$"],
  account: ["^account", "^customer", "^client"],
  received: ["^received", "^triggered", "sent to print vendor", "^eligible claims", "^payments inserted", "^sent to external"],
  loaded: ["^loaded", "^extracted", "status received from print vendor", "^submitted by clearing house", "^ack\\w*ment received", "^total payment sent"],
  error: ["error", "^rejected by client", "^failed count", "rejections in edi"],
  pending: ["pending", "^work ?basket"],
  variance: ["^variance"],
  ticket: ["jira", "ticket"],
  comment: ["^comments?", "^file name", "remark"]
};
const DEFAULT_INTERFACES: InterfaceCfg[] = [
  {"name": "Enrollment IB"},
  {"name": "Claims Clearing House"},
  {"name": "Claims IB"},
  {"name": "Provider IB – Tax entity", "sheet": "Provider IB", "block": "^Tax entity$"},
  {"name": "Provider IB – Practitioner", "sheet": "Provider IB", "block": "^Practitioner$"},
  {"name": "Provider IB – Supplier", "sheet": "Provider IB", "block": "^Supplier$"},
  {"name": "Provider IB – Supplier Location", "sheet": "Provider IB", "block": "^Supplier Location$"},
  {"name": "Provider IB – Practitioner Role", "sheet": "Provider IB", "block": "^Practitioner Role$"},
  {"name": "Authorization IB"},
  {"name": "Accumulator IB"},
  {"name": "ID Cards-Extract"},
  {"name": "ID Cards-Acknowledgement", "variance": "A-(B+C)"},
  {"name": "Correspondences- Extract"},
  {"name": "Member Packets-Extract"},
  {"name": "Member Packets-Acknowledgement", "variance": "A-(B+C)"},
  {"name": "Claims Payment Extract"},
  {"name": "External Pricing"},
  {"name": "External Claims Editing"},
  {"name": "Encounter Submission", "block": "Extraction"},
  {"name": "Encounter Rejections", "sheet": "Encounter Submission", "block": "Rejections", "columns": {"date": ["^to date$"], "received": ["^extracted"], "loaded": ["^loaded"], "error": ["^rejected"]}}
];

function main(workbook: ExcelScript.Workbook, configJson: string = "", asOf: string = "", mode: string = "build", archiveJson: string = "",
  sharepointJson: string = ""): ReconResult {
  const warnings: string[] = [];
  const cfg: ReconConfig = configJson && configJson.trim() ? JSON.parse(configJson) : {};
  const none: Tiles = { inScope: 0, fullyReconciled: 0, notFullyReconciled: 0, trackedViaTicket: 0, notTrackedViaTicket: 0, notReported: 0, varianceSum: 0, reconciledRate: -1 };
  const empty = (ok: boolean, w: string[], extractJson: string, spBatches: SpBatch[]): ReconResult => ({
    ok, asOf: "", defaultDay: "", generated: new Date().toISOString(), dataJson: "", digestJson: "",
    daily: none, weekly: none, monthly: none, dailyLabel: "", weeklyLabel: "", monthlyLabel: "",
    needsAttention: false, subject: "", emailHtml: "", warnings: w, extractJson, spBatches
  });
  // one-off: create the SharePoint lists, columns, views and formatting
  if (mode.toLowerCase() === "setup") {
    if (!cfg.siteUrl) return empty(false, ["recon-config.json needs siteUrl (the site that will hold the lists)."], "", []);
    return empty(true, [], "", spSetup(cfg.siteUrl, cfg.spBatchSize || 100));
  }
  const interfaces: InterfaceCfg[] = cfg.interfaces && cfg.interfaces.length ? cfg.interfaces : DEFAULT_INTERFACES;
  const ifs = interfaces.map(x => x.name);
  const typ = interfaces.map(x => x.scheduleType ? x.scheduleType
    : /^Provider IB/.test(x.name) ? "Provider IB" : x.name === "Encounter Rejections" ? "Encounter Submission" : x.name);

  // ---- calendar: day 0 is the Monday on/before startDate ----
  let T0 = parseDay(cfg.startDate || "2026-06-29", "YMD");
  if (T0 === null) throw new Error("startDate in the configuration is not a valid yyyy-MM-dd date.");
  T0 -= ((T0 + 3) % 7 + 7) % 7; // 1970-01-01 was a Thursday -> (n+3)%7 is 0 on Mondays
  const order = (cfg.dateOrder || "MDY").toUpperCase();
  const asOfDay = asOf && asOf.trim() ? parseDay(asOf.trim(), "YMD") : null;
  const cap = asOfDay === null ? Number.MAX_SAFE_INTEGER : asOfDay - T0;

  // ---- account names: tracker names -> dashboard names ----
  const alias = new Map<string, string>();
  const acc: string[] = [];
  const accIdx = new Map<string, number>();
  const byLoose = new Map<string, number>(); // spelling variants of one name share an account
  const addAcc = (n: string): number => {
    let k = accIdx.get(n);
    if (k === undefined) { k = acc.length; acc.push(n); accIdx.set(n, k); if (!byLoose.has(loose(n))) byLoose.set(loose(n), k); }
    return k;
  };
  const aliases = cfg.accountAliases || {};
  Object.keys(aliases).forEach(raw => { alias.set(loose(raw), aliases[raw]); addAcc(aliases[raw]); });
  const ignore = new Set<string>((cfg.ignoreAccounts || []).map(loose));
  const dropUnmapped = (cfg.unmappedAccounts || "keep").toLowerCase() === "drop";
  const unmapped = new Set<string>();
  const resolve = (raw: string): number => {
    const n = loose(raw);
    if (!n || ignore.has(n)) return -1;
    const m = alias.get(n);
    if (m !== undefined) return addAcc(m);
    const known = byLoose.get(n);
    if (known !== undefined) return known;
    if (!unmapped.has(n)) { unmapped.add(n); warnings.push("Account '" + raw.trim() + "' is not in accountAliases — " + (dropUnmapped ? "dropped." : "kept under its tracker name.")); }
    return dropUnmapped ? -1 : addAcc(raw.trim());
  };

  // ---- read the tracker ----
  const sheets = new Map<string, Cell[][][]>();
  workbook.getWorksheets().forEach(ws => {
    const rng = ws.getUsedRange(true); // values only — ignores cells that are merely formatted
    const k = norm(ws.getName());
    const list = sheets.get(k) || [];
    list.push(rng ? rng.getValues() : []);
    sheets.set(k, list);
  });
  const explained = new RegExp(cfg.explainedCommentPattern || "\\b(n\\/?a|not applicable|no run|no file|no load|no claims|holiday|on hold|(out of|not in) scope)\\b", "i");
  const facts = new Map<number, Fact>();
  const owner = new Map<number, number>(); // feed key -> workbook that supplied it
  const X = new Set<number>();
  const NI = ifs.length;
  let maxDay = -1, badDates = 0;
  const inspect = mode.toLowerCase() === "inspect";

  // ---- archived weeks (outputs of mode "extract"), oldest first; the workbook being read wins over them ----
  const archives: WeekRows[] = archiveJson && archiveJson.trim() ? JSON.parse(archiveJson) : [];
  archives.forEach((ex, n) => {
    const book = n - archives.length; // negative: below every table of this workbook
    const base = parseDay(ex.start || "", "YMD");
    if (base === null) { warnings.push("Archived week " + (n + 1) + " has no start date — ignored."); return; }
    const ai = (ex.acc || []).map(name => resolve(name)), ii = (ex.ifs || []).map(name => ifs.indexOf(name));
    const val = (v: number | null): number => Number(v) || 0;
    (ex.f || []).forEach(r => {
      const d = base + val(r[0]) - T0, a = ai[val(r[1])], i = ii[val(r[2])];
      if (a === undefined || a < 0 || i === undefined || i < 0 || d < 0 || d > cap || d >= KEY) return;
      const k = (a * NI + i) * KEY + d;
      const v = r[7] === null || r[7] === undefined ? null : val(r[7]);
      const z = owner.get(k) === book ? facts.get(k) : undefined;
      owner.set(k, book);
      if (!z) facts.set(k, [d, a, i, val(r[3]), val(r[4]), val(r[5]), val(r[6]), v, val(r[8]) ? 1 : 0]);
      else { z[3] += val(r[3]); z[4] += val(r[4]); z[5] += val(r[5]); z[6] += val(r[6]); z[7] = z[7] === null && v === null ? null : (z[7] || 0) + (v || 0); z[8] = z[8] || (val(r[8]) ? 1 : 0); }
      if (d > maxDay) maxDay = d;
    });
    (ex.x || []).forEach(r => {
      const d = base + r[0] - T0, a = ai[r[1]], i = ii[r[2]];
      if (a === undefined || a < 0 || i === undefined || i < 0 || d < 0 || d > cap || d >= KEY) return;
      X.add((a * NI + i) * KEY + d);
      if (d > maxDay) maxDay = d;
    });
  });

  const noFormula = (cfg.varianceWhenMissing || "formula").toLowerCase() === "none";
  interfaces.forEach((ic, i) => {
    const sheetName = ic.sheet || ic.name;
    const tables = sheets.get(norm(sheetName));
    if (!tables) { warnings.push("Interface '" + ic.name + "': sheet '" + sheetName + "' not found."); return; }
    const col = (role: string): string[] => {
      const own = ic.columns ? pick(ic.columns, role) : undefined;
      const shared = cfg.columns ? pick(cfg.columns, role) : undefined;
      return own || shared || pick(DEFAULT_COLUMNS, role) || [];
    };
    const blockRe = ic.block ? new RegExp(ic.block, "i") : null;
    let used = 0;
    tables.forEach((values, book) => segments(values, col("account"), col("date")).forEach(seg => {
      if (blockRe && !seg.titles.some(t => blockRe.test(t))) return;
      const H = values[seg.hr];
      const cDate = findCols(H, col("date"));
      if (!cDate.length) return;
      used++;
      const cVar = findCols(H, col("variance")), cTkt = findCols(H, col("ticket")), cCom = findCols(H, col("comment"));
      // count roles never take a ticket, comment, variance or date column ("Jira Ticket Number (to resolve Errors)")
      const countCols = (role: string): number[] => findCols(H, col(role)).filter(c => cTkt.indexOf(c) < 0 && cCom.indexOf(c) < 0 && cVar.indexOf(c) < 0 && c !== cDate[0]);
      const cRec = countCols("received"), cLd = countCols("loaded"), cErr = countCols("error"), cPend = countCols("pending");
      const cFil = ic.filterColumn ? findCols(H, [ic.filterColumn]) : [];
      // lettered count columns, e.g. "Received (A)" -> A; used for the variance formula and to tell filled rows from empty ones
      const letters = new Map<string, number>();
      const derived = new Map<number, string>(); // column -> its own header formula, e.g. "Loaded into HRP (E) E = F+G+H"
      const counts: number[] = cRec.concat(cLd, cErr, cPend);
      for (let c = 0; c < H.length; c++) {
        if (cVar.indexOf(c) >= 0) continue;
        const text = String(H[c]).replace(/\s+/g, " ").trim();
        const m = /\(\s*([A-Z])\s*\)/.exec(text);
        if (!m) continue;
        if (!letters.has(m[1])) letters.set(m[1], c);
        if (counts.indexOf(c) < 0) counts.push(c);
        const def = new RegExp("\\b" + m[1] + "\\s*=\\s*([A-Z+\\s]+)$").exec(text);
        if (def) derived.set(c, def[1]);
      }
      const tail = cVar.length ? /(?:\b[A-Z]\b|[()+\-\s])+$/.exec(String(H[cVar[0]]).replace(/\s+/g, " ").trim()) : null;
      const formula = noFormula ? "" : ic.variance || (tail ? tail[0].trim() : "");
      const names = (c: number[]): string => c.length ? c.map(k => String(H[k]).replace(/\s+/g, " ").trim()).join(" + ") : "—";
      if (inspect) warnings.push("[inspect] " + ic.name + " <- sheet '" + sheetName + "' header row " + (seg.hr + 1) + (seg.titles.length ? " [" + seg.titles.join(" / ") + "]" : "") +
        " | date: " + names(cDate.slice(0, 1)) + " | account: " + String(H[seg.accCol] || "column " + (seg.accCol + 1)).trim() + " | received: " + names(cRec) + " | loaded: " + names(cLd) +
        " | error: " + names(cErr) + " | pending: " + names(cPend) + " | variance: " + names(cVar.slice(0, 1)) + (formula ? " (if blank: " + formula + ")" : "") + " | ticket: " + names(cTkt) + " | comment: " + names(cCom));
      if (!cRec.length) warnings.push("Interface '" + ic.name + "': no 'received' column matched.");
      if (!cVar.length && !formula) warnings.push("Interface '" + ic.name + "': no variance column — every reported row counts as reconciled.");

      for (let r = seg.hr + 1; r < seg.end; r++) {
        const row = values[r];
        const dv = row[cDate[0]];
        // rows without a date are template rows: ignored
        if (dv === "" || dv === null || dv === undefined) continue;
        let day: number | null;
        if (typeof dv === "number") day = Math.floor(dv) - 25569;
        else { const txt = String(dv).trim(); if (!txt || /^(n\/?a|-|tbd)$/i.test(txt)) continue; day = parseDay(txt, order); }
        if (day === null) { badDates++; continue; }
        const d = day - T0;
        if (d < 0 || d > cap || d >= KEY) continue;
        if (cFil.length && norm(String(row[cFil[0]])) !== norm(ic.filterEquals || "")) continue;
        const a = resolve(String(row[seg.accCol]));
        if (a < 0) continue;
        let isExplained = cCom.some(k => explained.test(String(row[k])));
        // a row is "reported" only when at least one count has been entered
        let hasNum = false;
        counts.forEach(k => { if (num(row[k]) !== null) hasNum = true; else if (explained.test(String(row[k]))) isExplained = true; });
        // value of a count cell; a blank cell with a header formula is derived from its parts
        const cellVal = (c: number, depth: number): number => {
          const n = num(row[c]);
          if (n !== null) return n;
          const f = derived.get(c);
          return f && depth < 4 ? evalFormula(f, L => letterVal(L, depth + 1)) || 0 : 0;
        };
        const letterVal = (L: string, depth: number): number | null => { const c = letters.get(L); return c === undefined ? null : cellVal(c, depth); };
        const total = (cols: number[]): number => { let t = 0; cols.forEach(k => { t += cellVal(k, 0); }); return t; };
        const k = (a * NI + i) * KEY + d;
        if (!hasNum) { if (isExplained) { X.add(k); if (d > maxDay) maxDay = d; } continue; }
        const rec = total(cRec), ld = total(cLd), er = total(cErr), pe = total(cPend);
        let v: number | null = cVar.length ? num(row[cVar[0]]) : null;
        if (v === null && formula) v = evalFormula(formula, L => letterVal(L, 0));
        const tk = cTkt.some(c => isTicket(row[c])) ? 1 : 0;
        // rows of one workbook add up; a later workbook (the current week) replaces what an archived one said for the same feed
        const z = owner.get(k) === book ? facts.get(k) : undefined;
        owner.set(k, book);
        if (!z) facts.set(k, [d, a, i, rec, ld, er, pe, v === null ? null : Math.abs(v), tk]);
        else {
          z[3] += rec; z[4] += ld; z[5] += er; z[6] += pe;
          z[7] = z[7] === null && v === null ? null : (z[7] || 0) + Math.abs(v || 0);
          z[8] = z[8] || tk;
        }
        if (d > maxDay) maxDay = d;
      }
    }));
    if (!used) warnings.push("Interface '" + ic.name + "': no table with a date column found in sheet '" + sheetName + "'" + (ic.block ? " under a title matching '" + ic.block + "'." : "."));
  });
  if (badDates) warnings.push(badDates + " row(s) skipped because the date could not be read (check dateOrder).");

  // ---- schedule ----
  const sch: { [acc: string]: { [t: string]: string } } = {};
  const schName = cfg.scheduleSheet || "Recon Report Schedule";
  const sv = sheets.get(norm(schName));
  if (!sv) warnings.push("Schedule sheet '" + schName + "' not found — nothing can be flagged as scheduled but not reported.");
  else readSchedule(sv[sv.length - 1], typ, resolve, acc, sch, warnings, inspect, (cfg.scheduleIgnoreTypes || ["TM Aging Summary"]).map(norm));

  if (maxDay < 0) warnings.push("No tracker rows were read. Run with mode = inspect and adjust recon-config.json.");

  if (mode.toLowerCase() === "extract") {
    const ex: WeekRows = { start: new Date(T0 * MS_DAY).toISOString().slice(0, 10), acc, ifs, f: [], x: [] };
    facts.forEach(r => ex.f.push(r.slice()));
    X.forEach(k => ex.x.push([k % KEY, Math.floor(Math.floor(k / KEY) / NI), Math.floor(k / KEY) % NI]));
    return empty(maxDay >= 0, warnings, JSON.stringify(ex), []);
  }
  const LAST = Math.max(0, asOfDay === null ? maxDay : Math.min(maxDay, cap));
  const NA = acc.length;

  const model: Model = { acc, ifs, typ, sch, F: facts, X, NA, NI, T0, LAST, DEF: LAST, anomMin: cfg.anomalyMinMedian || 200, anomFactor: cfg.anomalyFactor || 3 };
  const perDayRows: number[] = [];
  facts.forEach(r => { perDayRows[r[0]] = (perDayRows[r[0]] || 0) + 1; });
  // default day = latest day that looks complete: at least 60% of the busiest day's rows (or the configured minimum)
  let busiest = 0;
  perDayRows.forEach(n => { if (n > busiest) busiest = n; });
  const minRows = cfg.minRowsForDefaultDay === undefined ? Math.max(1, Math.ceil(busiest * 0.6)) : cfg.minRowsForDefaultDay;
  while (model.DEF > 0 && (perDayRows[model.DEF] || 0) < minRows) model.DEF--;

  // ---- dataset for the dashboard page ----
  const generated = new Date().toISOString();
  const fOut: Fact[] = []; facts.forEach(r => fOut.push(r));
  const xOut: number[][] = []; X.forEach(k => xOut.push([k % KEY, Math.floor(Math.floor(k / KEY) / NI), Math.floor(k / KEY) % NI]));
  const data = { asOf: iso(model, LAST), start: iso(model, 0), gen: generated, home: cfg.homeUrl || "", acc, ifs, typ, def: model.DEF, f: fOut, sch, x: xOut };
  // "<" and the JS line separators are escaped so the JSON is safe inside a <script> block
  const dataJson = JSON.stringify(data).replace(/</g, "\\u003c")
    .split(String.fromCharCode(0x2028)).join("\\u2028").split(String.fromCharCode(0x2029)).join("\\u2029");

  // ---- digest: the numbers behind every dashboard section ----
  const DEF = model.DEF, wk = Math.floor(DEF / 7) * 7;
  const months = monthList(model);
  const mo = months.length - 1;
  const mFrom = Math.max(0, dayIdx(model, months[mo][0], months[mo][1], 1)), mTo = Math.min(LAST, dayIdx(model, months[mo][0], months[mo][1] + 1, 0));
  const oD = scan(model, DEF, DEF, -1, -1, false), oW = scan(model, wk, Math.min(LAST, wk + 6), -1, -1, false), oM = scan(model, mFrom, mTo, -1, -1, false);
  const wTo = Math.min(LAST, wk + 6);
  const dailyLabel = DOW[DEF % 7] + " " + md(model, DEF) + " run";
  const weeklyLabel = "Week of " + md(model, wk) + " – " + md(model, wk + 6) + (wTo < wk + 6 ? " · updated through " + DOW[wTo % 7] + " " + md(model, wTo) : "");
  const monthlyLabel = MONTHS[months[mo][1]] + " " + months[mo][0];
  const hD = findings(model, "d", DEF, DEF, -1, oD, months, mo), hW = findings(model, "w", wk, wTo, -1, oW, months, mo), hM = findings(model, "m", mFrom, mTo, -1, oM, months, mo);
  const digest = {
    asOf: data.asOf, generated, defaultDay: iso(model, DEF),
    daily: section(model, dailyLabel, DEF, DEF, oD, hD),
    weekly: section(model, weeklyLabel, wk, wTo, oW, hW),
    monthly: section(model, monthlyLabel, mFrom, mTo, oM, hM),
    history: history(model)
  };
  // ---- SharePoint lists: rows for every list, as $batch requests that replace the lists' contents ----
  let spBatches: SpBatch[] = [];
  const spMode = !!(sharepointJson && sharepointJson.trim());
  if (spMode) {
    if (!cfg.siteUrl) warnings.push("recon-config.json needs siteUrl to write the SharePoint lists.");
    else {
      const info: SpListInfo[] = JSON.parse(sharepointJson);
      const periods: SpPeriod[] = [
        { name: SP_PERIODS[0], v: "d", d1: DEF, d2: DEF, label: dailyLabel },
        { name: SP_PERIODS[1], v: "w", d1: wk, d2: wTo, label: weeklyLabel },
        { name: SP_PERIODS[2], v: "m", d1: mFrom, d2: mTo, label: monthlyLabel }];
      const rows = spData(model, months, mo, periods, "Data loaded through " + md(model, LAST) + " · refreshed " + generated.slice(0, 16).replace("T", " ") + " UTC");
      spBatches = spDaily(cfg.siteUrl, info, rows, spWeekTitles(model), cfg.spBatchSize || 100, warnings);
    }
  }
  const needsAttention = oD.attN > 0 || oD.miss > 0;
  const subject = "Reconciliation Monitor · " + DOW[DEF % 7] + " " + md(model, DEF) + " · " + (oD.rate === null ? "no feeds reported" : oD.rate + "% reconciled") +
    (oD.attN ? " · " + oD.attN + " without ticket" : "") + (oD.miss ? " · " + oD.miss + " not reported" : "");

  return {
    ok: maxDay >= 0, asOf: data.asOf, defaultDay: iso(model, DEF), generated,
    // the SharePoint-lists flow does not need the page dataset or the digest; leaving them out keeps the output small
    dataJson: spMode ? "" : dataJson, digestJson: spMode ? "" : JSON.stringify(digest),
    daily: tiles(oD), weekly: tiles(oW), monthly: tiles(oM),
    dailyLabel, weeklyLabel, monthlyLabel, needsAttention, subject,
    emailHtml: emailHtml(model, dailyLabel, oD, hD, weeklyLabel, oW, cfg.dashboardUrl || ""), warnings, extractJson: "", spBatches
  };
}

/* ======================= reading helpers ======================= */

/** Account key: case, spacing and punctuation do not matter ("PlanOne" = "Planone", "Plan Two" = "PlanTwo"). */
function loose(s: string): string { return norm(s).replace(/[^a-z0-9]/g, ""); }

/** Tables inside a sheet: one per header row (a row with a Variance column, or with account + date columns). */
function segments(values: Cell[][], accPat: string[], datePat: string[]): Segment[] {
  const out: Segment[] = [];
  for (let r = 0; r < values.length; r++) {
    const acc = findCols(values[r], accPat);
    if (!findCols(values[r], ["^variance"]).length && !(acc.length && findCols(values[r], datePat.concat(["date"])).length)) continue;
    const titles: string[] = [];
    for (let k = Math.max(0, r - 3); k < r; k++) values[k].forEach(c => { const t = String(c === null || c === undefined ? "" : c).replace(/\s+/g, " ").trim(); if (t && typeof c === "string") titles.push(t); });
    if (out.length) out[out.length - 1].end = r;
    out.push({ hr: r, end: values.length, titles, accCol: acc.length ? acc[0] : out.length ? out[out.length - 1].accCol : 0 });
  }
  return out;
}

/** Evaluates a header formula such as "A-(B+C+D)" or "(A + B) - (C+D+E)". null when it cannot be evaluated. */
function evalFormula(expr: string, val: (letter: string) => number | null): number | null {
  const t = expr.replace(/\s+/g, "");
  let p = 0, ok = t.length > 0;
  const term = (): number => {
    const ch = t.charAt(p);
    if (ch === "(") { p++; const v = sum(); if (t.charAt(p) === ")") p++; else ok = false; return v; }
    if (ch === "-") { p++; return -term(); }
    if (/[A-Z]/.test(ch)) { p++; const v = val(ch); if (v === null) ok = false; return v || 0; }
    ok = false; p++; return 0;
  };
  const sum = (): number => {
    let v = term();
    while (ok && (t.charAt(p) === "+" || t.charAt(p) === "-")) { const op = t.charAt(p); p++; const w = term(); v = op === "+" ? v + w : v - w; }
    return v;
  };
  const v = sum();
  return ok && p === t.length ? v : null;
}

function norm(s: string): string { return String(s === null || s === undefined ? "" : s).replace(/\s+/g, " ").trim().toLowerCase(); }

function pick(m: ColumnMap, role: string): string[] | undefined {
  switch (role) {
    case "date": return m.date; case "account": return m.account; case "received": return m.received;
    case "loaded": return m.loaded; case "error": return m.error; case "pending": return m.pending;
    case "variance": return m.variance; case "ticket": return m.ticket; case "comment": return m.comment;
  }
  return undefined;
}

/** Indexes of the header cells matching any pattern (case-insensitive regular expressions). */
function findCols(header: Cell[], patterns: string[]): number[] {
  const out: number[] = [];
  const res = patterns.map(p => new RegExp(p, "i"));
  for (let c = 0; c < header.length; c++) {
    const h = String(header[c] === null || header[c] === undefined ? "" : header[c]).replace(/\s+/g, " ").trim();
    if (h && res.some(re => re.test(h))) out.push(c);
  }
  return out;
}

function num(v: Cell): number | null {
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (typeof v !== "string") return null;
  let s = v.replace(/[,\s]/g, "");
  let neg = false;
  if (/^\(.*\)$/.test(s)) { neg = true; s = s.slice(1, -1); }
  if (!/^-?\d+(\.\d+)?$/.test(s)) return null;
  return neg ? -Number(s) : Number(s);
}

function isTicket(v: Cell): boolean {
  const s = norm(String(v === null || v === undefined ? "" : v));
  return s !== "" && !/^(n\/?a|no|none|nil|-|0|false)$/.test(s);
}

/** Text date -> days since 1970-01-01 (UTC). order: YMD | MDY | DMY for all-numeric dates. */
function parseDay(s: string, order: string): number | null {
  const t = s.trim();
  let y = 0, m = 0, d = 0;
  let p = /^(\d{4})[-\/.](\d{1,2})[-\/.](\d{1,2})/.exec(t);
  if (p) { y = +p[1]; m = +p[2]; d = +p[3]; }
  else if ((p = /^(\d{1,2})[-\/.](\d{1,2})[-\/.](\d{2,4})/.exec(t))) {
    if (order === "DMY") { d = +p[1]; m = +p[2]; } else { m = +p[1]; d = +p[2]; }
    y = +p[3];
  } else if ((p = /^(\d{1,2})[-\s]([A-Za-z]{3})[A-Za-z]*[-\s,]+(\d{2,4})/.exec(t))) {
    d = +p[1]; m = monthNo(p[2]); y = +p[3];
  } else if ((p = /^([A-Za-z]{3})[A-Za-z]*\s+(\d{1,2}),?\s+(\d{2,4})/.exec(t))) {
    m = monthNo(p[1]); d = +p[2]; y = +p[3];
  } else return null;
  if (y < 100) y += 2000;
  if (m < 1 || m > 12 || d < 1 || d > 31) return null;
  return Math.round(Date.UTC(y, m - 1, d) / MS_DAY);
}
function monthNo(s: string): number { return MONTHS.map(x => x.toLowerCase()).indexOf(s.slice(0, 3).toLowerCase()) + 1; }

/** "Mon-Fri", "Daily", "Tue, Thu", "1111100", "Y" ... -> seven 0/1 flags, Monday first. null = not understood. */
function dayFlags(v: Cell): string | null {
  const s = norm(String(v === null || v === undefined ? "" : v));
  if (/^[01]{7}$/.test(s)) return s;
  if (s === "" || /^(n\/?a|no|none|nil|-|0|false|not applicable)$/.test(s)) return "0000000";
  if (/^(y|yes|x|true|1|daily|every ?day|all days|7 days)$/.test(s)) return "1111111";
  if (/^(weekdays?|business days|week days)$/.test(s)) return "1111100";
  const idx = (w: string): number => /^(mon(day)?|tue(s|sday)?|wed(nesday)?|thu(r|rs|rsday)?|fri(day)?|sat(urday)?|sun(day)?)s?\.?$/.test(w)
    ? ["mo", "tu", "we", "th", "fr", "sa", "su"].indexOf(w.slice(0, 2)) : -1;
  const f = [0, 0, 0, 0, 0, 0, 0];
  let any = false;
  const parts = s.split(/[,;\/&]|\band\b/);
  for (let k = 0; k < parts.length; k++) {
    const part = parts[k].trim();
    if (!part) continue;
    const r = /^([a-z]+)\.?\s*(?:-|–|to|through|thru)\s*([a-z]+\.?)$/.exec(part);
    if (r) {
      const a = idx(r[1]), b = idx(r[2]);
      if (a < 0 || b < 0) return null;
      for (let j = a; ; j = (j + 1) % 7) { f[j] = 1; if (j === b) break; }
      any = true;
    } else {
      const a = idx(part);
      if (a < 0) return null;
      f[a] = 1; any = true;
    }
  }
  return any ? f.join("") : null;
}

/**
 * Schedule tab. Three layouts are recognised:
 *   A) Account | Interface | Mon | Tue | ... | Sun      (any mark = scheduled)
 *   B) Account | Interface | Schedule ("Mon-Fri", "Daily", ...)
 *   C) Account | <one column per interface type>         (cell = "Mon-Fri", "Daily", "N/A", ...)
 */
function readSchedule(v: Cell[][], typ: string[], resolve: (raw: string) => number, acc: string[],
  sch: { [acc: string]: { [t: string]: string } }, warnings: string[], inspect: boolean, ignoreTypes: string[]): void {
  const types: string[] = [];
  typ.forEach(t => { if (types.indexOf(t) < 0) types.push(t); });
  const typeOf = (label: string): string | null => {
    const n = norm(label).replace(/[–—]/g, "-");
    for (let k = 0; k < types.length; k++) if (norm(types[k]).replace(/[–—]/g, "-") === n) return types[k];
    return null;
  };
  let hr = -1, cAcc: number[] = [];
  for (let r = 0; r < Math.min(v.length, 25) && hr < 0; r++) { cAcc = findCols(v[r], ["^account", "customer", "client", "health ?plan"]); if (cAcc.length) hr = r; }
  if (hr < 0) { warnings.push("Schedule sheet: no header row with an account column."); return; }
  const H = v[hr];
  const cIf = findCols(H, ["interface", "feed", "report", "^type$"]);
  const cDays: number[] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].map(dn => { const c = findCols(H, ["^" + dn]); return c.length ? c[0] : -1; });
  const cExpr = findCols(H, ["schedule", "frequency", "^days?$"]);
  const layout = cIf.length && cDays.filter(c => c >= 0).length >= 5 ? "A" : cIf.length && cExpr.length ? "B" : "C";
  if (inspect) warnings.push("[inspect] schedule layout " + layout + ", header row " + (hr + 1));
  const unknown = new Set<string>();
  const put = (a: number, t: string, flags: string): void => {
    const s = sch[acc[a]] || (sch[acc[a]] = {});
    const o = s[t] || "0000000";
    let m = "";
    for (let k = 0; k < 7; k++) m += flags.charAt(k) === "1" || o.charAt(k) === "1" ? "1" : "0";
    s[t] = m;
  };
  let lastAcc = "";
  for (let r = hr + 1; r < v.length; r++) {
    let raw = String(v[r][cAcc[0]]);
    // merged account cells: only the first row of each account carries the name
    if (raw.trim()) lastAcc = raw; else if (layout !== "C") raw = lastAcc;
    if (!raw.trim()) continue;
    const a = resolve(raw);
    if (a < 0) continue;
    if (layout === "C") {
      for (let c = 0; c < H.length; c++) {
        if (c === cAcc[0] || !String(H[c]).trim()) continue;
        const t = typeOf(String(H[c]));
        if (!t) { if (ignoreTypes.indexOf(norm(String(H[c]))) < 0) unknown.add(String(H[c]).trim()); continue; }
        const fl = dayFlags(v[r][c]);
        if (fl === null) warnings.push("Schedule: '" + String(v[r][c]) + "' (" + raw.trim() + " / " + t + ") not understood — treated as not scheduled.");
        put(a, t, fl || "0000000");
      }
    } else {
      const t = typeOf(String(v[r][cIf[0]]));
      if (!t) { const u = String(v[r][cIf[0]]).trim(); if (u && ignoreTypes.indexOf(norm(u)) < 0) unknown.add(u); continue; }
      let fl: string | null = "";
      if (layout === "A") cDays.forEach(c => { fl += c >= 0 && dayFlags(v[r][c]) === "1111111" ? "1" : "0"; });
      else fl = dayFlags(v[r][cExpr[0]]);
      if (fl === null) warnings.push("Schedule: '" + String(v[r][cExpr[0]]) + "' (" + raw.trim() + " / " + t + ") not understood — treated as not scheduled.");
      put(a, t, fl || "0000000");
    }
  }
  unknown.forEach(u => warnings.push("Schedule: '" + u + "' does not match any interface scheduleType — ignored."));
}

/* ======================= dashboard rules ======================= */

function iso(M: Model, i: number): string { return new Date((M.T0 + i) * MS_DAY).toISOString().slice(0, 10); }
function md(M: Model, i: number): string { const s = iso(M, i); return (+s.slice(5, 7)) + "/" + (+s.slice(8, 10)); }
function dayIdx(M: Model, y: number, m: number, d: number): number { return Math.round(Date.UTC(y, m, d) / MS_DAY) - M.T0; }
function full(n: number | null): string {
  if (n === null) return "–";
  const r = Math.round(n);
  return (r < 0 ? "-" : "") + String(Math.abs(r)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
function esc(s: string): string { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function median(a: number[]): number { const s = a.slice().sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; }
function pct(g: number, att: number): number | null { return g + att ? Math.round(100 * g / (g + att)) : null; }

/** true = scheduled that weekday, false = explicitly not scheduled, null = no schedule known. */
function sched(M: Model, a: number, i: number, d: number): boolean | null {
  const s = M.sch[M.acc[a]];
  if (!s) return null;
  const t = s[M.typ[i]];
  return t === undefined || t === null ? null : t.charAt(d % 7) === "1";
}

/**
 * State of one feed (account × interface × day):
 *   g    fully reconciled  — a row exists and |variance| <= 0.5
 *   att  not fully reconciled (t = tracked via ticket)
 *   miss scheduled, no row, and no explanatory comment
 *   null not applicable
 */
function cell(M: Model, a: number, i: number, d: number): CellState | null {
  const k = (a * M.NI + i) * KEY + d, r = M.F.get(k), s = sched(M, a, i, d);
  if (r) {
    const v = r[7];
    if (v !== null && Math.abs(v) > 0.5) return { s: "att", r, v: Math.abs(v), t: !!r[8] };
    return s === false ? null : { s: "g", r, v: 0, t: false };
  }
  if (s === true && !M.X.has(k)) return { s: "miss", r: null, v: 0, t: false };
  return null;
}

function bucket(): Bucket { return { g: 0, att: 0, attT: 0, attN: 0, miss: 0, sum: 0, scope: 0, rec: 0, ld: 0, err: 0, pend: 0, top: "", topv: 0, missI: [], an: 0 }; }

/** Aggregates days d1..d2 for one account (a0) / interface (i0), or all when -1. lite skips lists and anomalies. */
function scan(M: Model, d1: number, d2: number, a0: number, i0: number, lite: boolean): Scan {
  const o: Scan = { g: 0, att: 0, attT: 0, attN: 0, miss: 0, sum: 0, scope: 0, rate: null, items: [], missL: [], anom: [], byI: M.ifs.map(() => bucket()), byA: M.acc.map(() => bucket()) };
  for (let d = d1; d <= d2; d++) for (let a = 0; a < M.NA; a++) {
    if (a0 >= 0 && a !== a0) continue;
    for (let i = 0; i < M.NI; i++) {
      if (i0 >= 0 && i !== i0) continue;
      const A = o.byA[a], B = o.byI[i];
      if (sched(M, a, i, d) === true) { o.scope++; A.scope++; B.scope++; }
      const c = cell(M, a, i, d);
      if (!c) continue;
      if (c.r) { B.rec += c.r[3]; B.ld += c.r[4]; B.err += c.r[5]; B.pend += c.r[6]; }
      if (c.s === "g") { o.g++; A.g++; B.g++; }
      else if (c.s === "att") {
        o.att++; A.att++; B.att++; B.sum += c.v; A.sum += c.v; o.sum += c.v;
        if (c.t) { o.attT++; A.attT++; B.attT++; } else { o.attN++; A.attN++; B.attN++; }
        if (!lite) o.items.push({ a, i, d, v: c.v, t: c.t });
        if (c.v > A.topv) { A.topv = c.v; A.top = M.ifs[i] + " (" + md(M, d) + ")"; }
      } else {
        o.miss++; A.miss++; B.miss++;
        if (A.missI.length < 3 && A.missI.indexOf(M.ifs[i]) < 0) A.missI.push(M.ifs[i]);
        if (!lite) o.missL.push({ a, i, d });
      }
      // volume anomaly: received vs median of the previous (up to 20) reported days in the last 45
      if (!lite && c.r && c.r[3] > 0) {
        const base: number[] = [];
        for (let e = d - 1; e >= Math.max(0, d - 45) && base.length < 20; e--) {
          const q = M.F.get((a * M.NI + i) * KEY + e);
          if (q && q[3] > 0) base.push(q[3]);
        }
        if (base.length >= 6) {
          const m = median(base), v = c.r[3];
          if (m >= M.anomMin && (v > M.anomFactor * m || (v < m / M.anomFactor && d < M.LAST))) { o.anom.push({ a, i, d, v, m, dev: Math.abs(v - m) }); A.an++; }
        }
      }
    }
  }
  o.items.sort((p, q) => Math.abs(q.v) - Math.abs(p.v));
  o.anom.sort((p, q) => q.dev - p.dev);
  o.rate = pct(o.g, o.att);
  return o;
}

function monthList(M: Model): number[][] {
  const out: number[][] = [];
  const a = new Date(M.T0 * MS_DAY), b = new Date((M.T0 + M.LAST) * MS_DAY);
  let y = a.getUTCFullYear(), m = a.getUTCMonth();
  while (y < b.getUTCFullYear() || (y === b.getUTCFullYear() && m <= b.getUTCMonth())) { out.push([y, m]); m++; if (m > 11) { m = 0; y++; } }
  return out;
}

function groupBy(items: Item[], key: (x: Item) => string, numeric: boolean): Group[] {
  const g = new Map<string, Group>();
  items.forEach(x => {
    const k = key(x);
    let z = g.get(k);
    if (!z) { z = { k, n: 0, sum: 0, x }; g.set(k, z); }
    z.n++; z.sum += Math.abs(x.v);
  });
  const out: Group[] = []; g.forEach(z => out.push(z));
  if (numeric) out.sort((p, q) => (+p.k) - (+q.k));
  return out;
}

/** "Highlights" panel: up to four sentences for a day (d), week (w) or month (m). HTML with <b>. */
function findings(M: Model, V: string, d1: number, d2: number, a0: number, o: Scan, months: number[][], mo: number): string[] {
  const out: string[] = [];
  const all = a0 < 0;
  const name = (k: string): string => esc(all ? M.acc[+k] : M.ifs[+k]);
  const by = (x: Item): string => String(all ? x.a : x.i);
  let pr: Scan | null = null;
  if (V === "d") pr = scan(M, Math.max(0, d1 - 1), Math.max(0, d1 - 1), a0, -1, true);
  else if (V === "w") pr = scan(M, Math.max(0, d1 - 7), Math.max(0, d1 - 1), a0, -1, true);
  else if (mo > 0) pr = scan(M, Math.max(0, dayIdx(M, months[mo - 1][0], months[mo - 1][1], 1)), dayIdx(M, months[mo - 1][0], months[mo - 1][1] + 1, 0), a0, -1, true);
  const vs = (lbl: string): void => {
    if (!pr) return;
    const a = pct(o.g, o.att), b = pct(pr.g, pr.att);
    if (a === null || b === null) return;
    out.push("Reconciled rate <b>" + a + "%</b> vs " + b + "% in the " + lbl + " (" + (a > b ? "▲ better" : a < b ? "▼ worse" : "unchanged") + "); not fully reconciled " + o.att + " vs " + pr.att + ", not reported " + o.miss + " vs " + pr.miss + ".");
  };
  if (V === "d") {
    const t = o.items[0];
    if (t) out.push("Biggest variance today: <b>" + (all ? esc(M.acc[t.a]) + " · " : "") + esc(M.ifs[t.i]) + "</b> (" + full(t.v) + "), " + (t.t ? "tracked via a ticket" : "<b>no ticket</b>") + ".");
    const un = groupBy(o.items.filter(x => !x.t), by, true).sort((p, q) => q.n - p.n);
    if (o.attN) out.push("<b>" + o.attN + "</b> variance" + (o.attN > 1 ? "s" : "") + " need a ticket today" + (un.length ? " — " + un.slice(0, 3).map(z => name(z.k) + " (" + z.n + ")").join(", ") : "") + ".");
    else if (o.att) out.push("Every variance today is tracked via a ticket.");
    if (o.miss) {
      const ms = new Set<number>(o.missL.map(x => x.a)).size;
      out.push("<b>" + o.miss + "</b> scheduled feed" + (o.miss > 1 ? "s" : "") + " not reported today" + (all ? " across " + ms + " customer" + (ms > 1 ? "s" : "") : "") +
        " (e.g. " + o.missL.slice(0, 2).map(x => (all ? esc(M.acc[x.a]) + " · " : "") + esc(M.ifs[x.i])).join(", ") + ").");
    }
    if (o.anom.length) { const x = o.anom[0]; out.push("Volume check: " + (all ? "<b>" + esc(M.acc[x.a]) + "</b> · " : "") + esc(M.ifs[x.i]) + " received " + full(x.v) + " vs usual ~" + full(x.m) + "."); }
    vs("previous day");
  } else if (V === "w") {
    const g = groupBy(o.items, by, true).sort((p, q) => q.sum - p.sum);
    if (g.length && o.sum) out.push("<b>" + name(g[0].k) + "</b> accounts for " + Math.round(100 * g[0].sum / o.sum) + "% of the week’s variance (Σ|variance| " + full(g[0].sum) + ").");
    const rp = groupBy(o.items, x => x.a + "|" + x.i, false).filter(z => z.n >= 2).sort((p, q) => q.n - p.n);
    if (rp.length) out.push("<b>" + rp.length + "</b> feed" + (rp.length > 1 ? "s are" : " is") + " not reconciled on 2+ days this week; longest: " + (all ? esc(M.acc[rp[0].x.a]) + " · " : "") + esc(M.ifs[rp[0].x.i]) + " (" + rp[0].n + " days).");
    let bi = 0;
    const R: Scan[] = [];
    for (let d = d1; d <= d2; d++) R.push(scan(M, d, d, a0, -1, true));
    R.forEach((x, i) => { if (d1 + i <= M.DEF && x.att + x.miss > R[bi].att + R[bi].miss) bi = i; });
    if (R.length) out.push("Worst day: <b>" + DOW[(d1 + bi) % 7] + " " + md(M, d1 + bi) + "</b> with " + R[bi].att + " not fully reconciled and " + R[bi].miss + " not reported.");
    vs("previous week");
  } else {
    vs("previous month");
    const wk = new Map<number, number>();
    o.items.forEach(x => { const k = Math.floor(x.d / 7) * 7; wk.set(k, (wk.get(k) || 0) + Math.abs(x.v)); });
    const wl: number[][] = []; wk.forEach((s, k) => wl.push([k, s]));
    wl.sort((p, q) => p[0] - q[0]).sort((p, q) => q[1] - p[1]);
    if (wl.length) out.push("Heaviest week: <b>week of " + md(M, wl[0][0]) + "</b> (Σ|variance| " + full(wl[0][1]) + ").");
    const g = groupBy(o.items, by, true).sort((p, q) => q.n - p.n);
    if (g.length) out.push("Most days not reconciled: <b>" + name(g[0].k) + "</b> (" + g[0].n + " feed-days).");
    if (o.att) out.push("Ticket coverage: <b>" + Math.round(100 * o.attT / o.att) + "%</b> of variances this month are tracked via a ticket" + (pr && pr.att ? " (previous month " + Math.round(100 * pr.attT / pr.att) + "%)" : "") + ".");
  }
  return out.slice(0, 4);
}

/* ======================= output shaping ======================= */

function tiles(o: Scan): Tiles {
  return { inScope: o.scope, fullyReconciled: o.g, notFullyReconciled: o.att, trackedViaTicket: o.attT, notTrackedViaTicket: o.attN, notReported: o.miss, varianceSum: o.sum, reconciledRate: o.rate === null ? -1 : o.rate };
}

function section(M: Model, label: string, d1: number, d2: number, o: Scan, hl: string[]) {
  const day = (d: number): string => DOW[d % 7] + " " + md(M, d);
  return {
    label, from: iso(M, d1), to: iso(M, d2), tiles: tiles(o),
    highlights: hl.map(h => h.replace(/<\/?b>/g, "")), highlightsHtml: hl,
    customers: o.byA.map((x, a) => ({ a, x })).filter(z => z.x.g + z.x.att + z.x.miss + z.x.scope > 0)
      .sort((p, q) => q.x.sum - p.x.sum || q.x.miss - p.x.miss)
      .map(z => ({ customer: M.acc[z.a], inScope: z.x.scope, fullyReconciled: z.x.g, notFullyReconciled: z.x.att, trackedViaTicket: z.x.attT, notTrackedViaTicket: z.x.attN, notReported: z.x.miss, varianceSum: z.x.sum, reconciledRate: pct(z.x.g, z.x.att), topVariance: z.x.top, volumeAnomalies: z.x.an })),
    interfaces: o.byI.map((b, i) => ({ i, b })).filter(z => z.b.scope > 0 || z.b.g + z.b.att + z.b.miss > 0 || z.b.rec !== 0)
      .sort((p, q) => q.b.sum - p.b.sum || q.b.miss - p.b.miss || q.b.scope - p.b.scope)
      .map(z => ({ interface: M.ifs[z.i], inScope: z.b.scope, received: z.b.rec, loaded: z.b.ld, error: z.b.err, pending: z.b.pend, varianceSum: z.b.sum, fullyReconciled: z.b.g, trackedViaTicket: z.b.attT, notTrackedViaTicket: z.b.attN, notReported: z.b.miss })),
    topVariances: o.items.slice(0, 12).map(x => ({ customer: M.acc[x.a], interface: M.ifs[x.i], date: iso(M, x.d), day: day(x.d), variance: x.v, ticket: x.t })),
    varianceCount: o.items.length,
    notReported: o.missL.slice().sort((p, q) => q.d - p.d || p.a - q.a).slice(0, 200).map(x => ({ customer: M.acc[x.a], interface: M.ifs[x.i], date: iso(M, x.d), day: day(x.d) })),
    volumeAnomalies: o.anom.slice(0, 15).map(x => ({ customer: M.acc[x.a], interface: M.ifs[x.i], date: iso(M, x.d), day: day(x.d), received: x.v, usual: x.m, changePct: Math.round((x.v - x.m) / x.m * 100) }))
  };
}

/** Feeds not fully reconciled again and again: weeks affected, days, longest run of consecutive days, ticketed. */
function recurring(M: Model): Rec[] {
  const pr = new Map<string, Rec>();
  for (let d = 0; d <= M.LAST; d++) for (let a = 0; a < M.NA; a++) for (let i = 0; i < M.NI; i++) {
    const c = cell(M, a, i, d);
    if (!c || c.s !== "att") continue;
    const k = a + "|" + i;
    let p = pr.get(k);
    if (!p) { p = { a, i, wk: new Set<number>(), n: 0, sum: 0, t: 0, last: 0, str: 0, cur: 0, prev: -9 }; pr.set(k, p); }
    p.wk.add(Math.floor(d / 7)); p.n++; p.sum += c.v; if (c.t) p.t++;
    p.last = d; p.cur = d - p.prev <= 1 ? p.cur + 1 : 1; p.str = Math.max(p.str, p.cur); p.prev = d;
  }
  const recs: Rec[] = []; pr.forEach(p => recs.push(p));
  recs.sort((p, q) => q.wk.size - p.wk.size || q.sum - p.sum);
  return recs;
}

/** Historical view: weekly scorecard per customer, recurring issues, quality over time. */
function history(M: Model) {
  const weeks: number[] = [];
  for (let w = 0; w * 7 <= M.LAST; w++) weeks.push(w * 7);
  const wEnd = (w: number): number => Math.min(M.LAST, w + 6);
  const avg = (a: number[]): number => a.reduce((x, y) => x + y, 0) / a.length;
  const notNull = (v: number | null): boolean => v !== null;
  const scorecard: { customer: string; rates: (number | null)[]; last4VsPrior4: number | null }[] = [];
  M.acc.forEach((n, a) => {
    const rs = weeks.map(w => { const x = scan(M, w, wEnd(w), a, -1, true); return pct(x.g, x.att); });
    if (!rs.some(notNull)) return;
    const rec: number[] = [], old: number[] = [];
    rs.slice(-4).forEach(v => { if (v !== null) rec.push(v); });
    rs.slice(-8, -4).forEach(v => { if (v !== null) old.push(v); });
    scorecard.push({ customer: n, rates: rs, last4VsPrior4: rec.length && old.length ? Math.round(avg(rec) - avg(old)) : null });
  });
  const W = weeks.map(w => scan(M, w, wEnd(w), -1, -1, true));
  return {
    weeks: weeks.map(w => iso(M, w)),
    scorecard,
    recurringIssues: recurring(M).slice(0, 12).map(p => ({ customer: M.acc[p.a], interface: M.ifs[p.i], weeksAffected: p.wk.size, days: p.n, longestRunDays: p.str, varianceSum: p.sum, ticketedPct: Math.round(100 * p.t / p.n), lastSeen: iso(M, p.last) })),
    quality: W.map((x, k) => {
      const t = x.g + x.att + x.miss;
      return { week: iso(M, weeks[k]), reconciledRate: pct(x.g, x.att), ticketCoverage: x.att ? Math.round(100 * x.attT / x.att) : null, scheduleAdherence: t ? Math.round(100 * (1 - x.miss / t)) : null };
    })
  };
}

function emailHtml(M: Model, dLabel: string, oD: Scan, hD: string[], wLabel: string, oW: Scan, url: string): string {
  const td = "padding:6px 10px;border-bottom:1px solid #e1e7ee;font:13px Segoe UI,Arial,sans-serif;";
  const th = td + "color:#5b6578;font-size:11px;text-transform:uppercase;text-align:left;";
  const kpi = (l: string, n: string, c: string): string => "<td style=\"padding:10px 14px;border:1px solid #e1e7ee;border-top:3px solid " + c + ";text-align:center;font-family:Segoe UI,Arial,sans-serif\"><div style=\"font-size:11px;color:#5b6578;text-transform:uppercase\">" + l + "</div><div style=\"font-size:26px;font-weight:600;color:" + c + "\">" + n + "</div></td>";
  const row = (o: Scan): string => "<table cellspacing=\"0\" cellpadding=\"0\" style=\"border-collapse:collapse;margin:8px 0 14px\"><tr>" +
    kpi("In scope", String(o.scope), "#161d3f") + kpi("Fully reconciled", String(o.g), "#1e9e6a") + kpi("Not fully reconciled", String(o.att), "#c77700") +
    kpi("Tracked via ticket", String(o.attT), "#2f6fed") + kpi("Not tracked", String(o.attN), "#c2410c") + kpi("Not reported", String(o.miss), "#c73a3a") +
    kpi("Reconciled rate", o.rate === null ? "–" : o.rate + "%", "#1e9e6a") + "</tr></table>";
  let h = "<div style=\"font-family:Segoe UI,Arial,sans-serif;color:#1c2433;font-size:14px\">" +
    "<div style=\"background:#161d3f;color:#fff;padding:14px 18px;border-radius:8px\"><div style=\"font-size:11px;letter-spacing:.1em;color:#6fd9d4\">TECHNOLOGY OPERATIONS · RECONCILIATION REPORT</div><div style=\"font-size:20px;font-weight:600\">All customers · " + esc(dLabel) + "</div></div>" + row(oD);
  if (hD.length) h += "<div style=\"font-weight:600;margin:4px 0\">Highlights</div><ol style=\"margin:4px 0 12px;padding-left:20px\">" + hD.map(x => "<li style=\"margin:3px 0\">" + x + "</li>").join("") + "</ol>";
  if (oD.items.length) h += "<div style=\"font-weight:600;margin:4px 0\">Highest variances</div><table cellspacing=\"0\" style=\"border-collapse:collapse\"><tr><th style=\"" + th + "\">Customer</th><th style=\"" + th + "\">Interface</th><th style=\"" + th + "text-align:right\">Variance</th><th style=\"" + th + "\">Ticket</th></tr>" +
    oD.items.slice(0, 10).map(x => "<tr><td style=\"" + td + "\"><b>" + esc(M.acc[x.a]) + "</b></td><td style=\"" + td + "\">" + esc(M.ifs[x.i]) + "</td><td style=\"" + td + "text-align:right\"><b>" + full(x.v) + "</b></td><td style=\"" + td + "color:" + (x.t ? "#2f6fed" : "#c2410c") + "\">" + (x.t ? "tracked" : "not tracked") + "</td></tr>").join("") + "</table>";
  if (oD.missL.length) h += "<div style=\"font-weight:600;margin:12px 0 4px\">Scheduled but not reported (" + oD.miss + ")</div><table cellspacing=\"0\" style=\"border-collapse:collapse\">" +
    oD.missL.slice(0, 20).map(x => "<tr><td style=\"" + td + "\"><b>" + esc(M.acc[x.a]) + "</b></td><td style=\"" + td + "\">" + esc(M.ifs[x.i]) + "</td></tr>").join("") + "</table>";
  h += "<div style=\"font-weight:600;margin:14px 0 0\">" + esc(wLabel) + "</div>" + row(oW);
  if (url) h += "<p><a href=\"" + esc(url) + "\" style=\"color:#1fb5ad;font-weight:600\">Open the Reconciliation Monitor</a></p>";
  return h + "<div style=\"color:#5b6578;font-size:12px\">Data loaded through " + md(M, M.LAST) + " · generated by Power Automate.</div></div>";
}

/* ======================= SharePoint lists dashboard =======================
 * Six lists hold the dashboard; a modern SharePoint page shows them with List web parts. Selecting a customer in the
 * "Customers" web part filters the other web parts (SharePoint dynamic filtering) — the drill-down. Formatting uses
 * SharePoint's JSON view / column formatting, which needs no custom script.
 */

type SpValue = string | number | null;
interface SpRow { [field: string]: SpValue }
interface SpField { n: string; t: string; d: string }   // internal name, type (Text | Number | Note), display name
interface SpList { key: string; title: string; titleName: string; desc: string; fields: SpField[] }
interface SpView { list: string; title: string; query: string; rowLimit: number; paged: boolean; fields: string[]; formatter: string }
interface SpListInfo { title: string; type: string; items: { Id: number }[] }
interface SpBatch { boundary: string; body: string; operations: number; label: string }
interface SpPeriod { name: string; v: string; d1: number; d2: number; label: string }

const SP_PERIODS = ["Latest day", "This week", "This month"];
const SP_ALL = "(All customers)";
const SP_WEEKS = 12;
const SP_CAP_VARIANCES = 250, SP_CAP_NOT_REPORTED = 300, SP_CAP_ANOMALIES = 50, SP_CAP_RECURRING = 100, SP_TREND_DAYS = 28;

function spLists(): SpList[] {
  const num = (n: string, d: string): SpField => ({ n, t: "Number", d });
  const txt = (n: string, d: string): SpField => ({ n, t: "Text", d });
  const counts: SpField[] = [num("InScope", "In scope"), num("Reconciled", "Fully reconciled"), num("NotReconciled", "Not fully reconciled"),
    num("Ticketed", "Tracked via ticket"), num("NotTicketed", "Not tracked via ticket"), num("NotReported", "Not reported"), num("VarianceSum", "Variance Σ|v|")];
  const weeks: SpField[] = [];
  for (let k = 1; k <= SP_WEEKS; k++) weeks.push(num(spWeekField(k), "Week " + k));
  return [
    { key: "summary", title: "Recon Summary", titleName: "Customer", desc: "Reconciliation Monitor: tiles and highlights per period and customer (refreshed by Power Automate).",
      fields: [txt("Period", "Period"), txt("Covers", "Covers"), num("SortKey", "Sort"), txt("Status", "Status")].concat(counts,
        [num("Rate", "Reconciled rate"), { n: "Highlights", t: "Note", d: "Highlights" }, txt("DataThrough", "Data through")]) },
    { key: "interfaces", title: "Recon Interfaces", titleName: "Interface", desc: "Reconciliation Monitor: interface summary per period and customer.",
      fields: [txt("Customer", "Customer"), txt("Period", "Period"), txt("Status", "Status"), num("Received", "Received"), num("Loaded", "Loaded"),
        num("Errors", "Error"), num("Pending", "Pending / WB")].concat(counts) },
    { key: "issues", title: "Recon Issues", titleName: "Interface", desc: "Reconciliation Monitor: variances, feeds not reported and volume anomalies.",
      fields: [txt("Customer", "Customer"), txt("Period", "Period"), txt("IssueType", "Type"), txt("Day", "Day"), txt("DayDate", "Date"),
        num("Amount", "Value"), num("Usual", "Usual"), num("ChangePct", "Change %"), txt("Ticket", "Ticket"), num("SortKey", "Sort")] },
    { key: "trend", title: "Recon Daily Trend", titleName: "Customer", desc: "Reconciliation Monitor: reconciled rate per day, last " + SP_TREND_DAYS + " days.",
      fields: [txt("Day", "Day"), txt("DayDate", "Date"), num("Rate", "Reconciled rate"), num("NotReconciled", "Not fully reconciled"),
        num("NotReported", "Not reported"), num("InScope", "In scope")] },
    { key: "scorecard", title: "Recon Weekly Scorecard", titleName: "Customer", desc: "Reconciliation Monitor: reconciled rate by week (column titles are the week start dates).",
      fields: [num("SortKey", "Sort")].concat(weeks, [num("Change", "Last 4 vs prior 4")]) },
    { key: "recurring", title: "Recon Recurring Issues", titleName: "Interface", desc: "Reconciliation Monitor: feeds that keep coming back.",
      fields: [txt("Customer", "Customer"), num("WeeksAffected", "Weeks affected"), num("Days", "Days"), num("LongestRun", "Longest run (days)"),
        num("VarianceSum", "Variance Σ|v|"), num("TicketedPct", "Ticketed %"), txt("LastSeen", "Last seen"), num("SortKey", "Sort")] }
  ];
}
function spWeekField(k: number): string { return "W" + (k < 10 ? "0" : "") + k; }

/* ---------- formatting (SharePoint JSON formatting) ---------- */

const SP_ROW_SCHEMA = "https://developer.microsoft.com/json-schemas/sp/v2/row-formatting.schema.json";
const SP_COL_SCHEMA = "https://developer.microsoft.com/json-schemas/sp/v2/column-formatting.schema.json";

/** KPI card: the five dashboard tiles + highlights, for the one row the view shows. */
function spKpiFormatter(): string {
  const tile = (label: string, value: string, color: string, sub: string) => ({
    elmType: "div",
    style: { "flex": "1 1 150px", "min-width": "140px", "margin": "4px", "padding": "10px 12px", "border": "1px solid #e1e7ee",
      "border-top": "3px solid " + color, "border-radius": "12px", "background-color": "#ffffff", "text-align": "center" },
    children: [
      { elmType: "div", txtContent: label, style: { "font-size": "11px", "font-weight": "600", "text-transform": "uppercase", "letter-spacing": "0.06em", "color": "#5b6578" } },
      { elmType: "div", txtContent: value, style: { "font-size": "30px", "font-weight": "600", "line-height": "1.2", "color": color } },
      { elmType: "div", txtContent: sub, style: { "font-size": "12px", "color": "#5b6578" } }]
  });
  const f = {
    $schema: SP_ROW_SCHEMA, hideSelection: true, hideColumnHeader: true,
    rowFormatter: {
      elmType: "div", style: { "display": "block", "width": "100%", "padding": "4px 0" },
      children: [
        { elmType: "div", style: { "background-color": "#161d3f", "color": "#ffffff", "border-radius": "12px", "padding": "14px 18px", "margin": "4px" },
          children: [
            { elmType: "div", txtContent: "=if([$Title] == '" + SP_ALL + "', 'TECHNOLOGY OPERATIONS · PORTFOLIO', 'TECHNOLOGY OPERATIONS · CUSTOMER') + ' · RECONCILIATION REPORT'",
              style: { "font-size": "11px", "font-weight": "600", "letter-spacing": "0.12em", "color": "#6fd9d4" } },
            { elmType: "div", txtContent: "=if([$Title] == '" + SP_ALL + "', 'All customers · Portfolio', [$Title] + ' · Reconciliation')",
              style: { "font-size": "24px", "font-weight": "600" } },
            { elmType: "div", txtContent: "=[$Covers] + ' · ' + [$DataThrough]", style: { "font-size": "13px", "color": "#c9d0e6" } }] },
        { elmType: "div", style: { "display": "flex", "flex-wrap": "wrap" },
          children: [
            tile("In scope", "=toLocaleString([$InScope])", "#161d3f", "Feeds scheduled"),
            tile("Fully reconciled", "=toLocaleString([$Reconciled])", "#1e9e6a", "No variance"),
            tile("Not fully reconciled", "=toLocaleString([$NotReconciled])", "#c77700",
              "='Tracked ' + toLocaleString([$Ticketed]) + ' · not tracked ' + toLocaleString([$NotTicketed]) + ' · Σ|v| ' + toLocaleString([$VarianceSum])"),
            tile("Scheduled but not reported", "=toLocaleString([$NotReported])", "#c73a3a", "Expected, no row"),
            tile("Reconciled rate", "=if([$Rate] == '', '–', toString([$Rate]) + '%')", "#1e9e6a", "Of feeds reported")] },
        { elmType: "div", style: { "margin": "4px", "padding": "10px 14px", "border": "1px solid #e1e7ee", "border-left": "4px solid #1fb5ad",
            "border-radius": "12px", "background-color": "#f6f8fb", "display": "=if([$Highlights] == '', 'none', 'block')" },
          children: [
            { elmType: "div", txtContent: "Highlights", style: { "font-size": "14px", "font-weight": "700", "margin-bottom": "4px" } },
            { elmType: "div", txtContent: "[$Highlights]", style: { "white-space": "pre-wrap", "font-size": "13px", "line-height": "1.5" } }] }]
    }
  };
  return JSON.stringify(f);
}

/** Row colours by status, as on the dashboard (amber = variance without ticket, red = not reported, green = reconciled). */
function spStatusRows(): string {
  return JSON.stringify({ $schema: SP_ROW_SCHEMA,
    additionalRowClass: "=if([$Status] == 'Variance without ticket', 'sp-field-severity--severeWarning', if([$Status] == 'Variance with ticket', 'sp-field-severity--warning', if([$Status] == 'Not reported', 'sp-field-severity--blocked', if([$Status] == 'Reconciled', 'sp-field-severity--good', ''))))" });
}
function spTicketRows(): string {
  return JSON.stringify({ $schema: SP_ROW_SCHEMA,
    additionalRowClass: "=if([$Ticket] == 'Not tracked', 'sp-field-severity--severeWarning', if([$Ticket] == 'Tracked', 'sp-field-severity--low', 'sp-field-severity--blocked'))" });
}
function spPercent(): string {
  return JSON.stringify({ $schema: SP_COL_SCHEMA, elmType: "div", txtContent: "=if(@currentField == '', '–', toString(@currentField) + '%')" });
}
function spRateBar(): string {
  return JSON.stringify({ $schema: SP_COL_SCHEMA, elmType: "div", style: { "display": "flex", "align-items": "center", "width": "100%" },
    children: [
      { elmType: "div", style: { "height": "12px", "border-radius": "3px", "margin-right": "6px",
        "background-color": "=if(@currentField >= 95, '#8cc0a8', if(@currentField >= 85, '#b9d49a', if(@currentField >= 70, '#e3a977', '#dc9a9a')))",
        "width": "=if(@currentField == '', '0px', toString(@currentField) + 'px')" } },
      { elmType: "span", txtContent: "=if(@currentField == '', '–', toString(@currentField) + '%')" }] });
}
function spHeat(): string {
  return JSON.stringify({ $schema: SP_COL_SCHEMA, elmType: "div", txtContent: "=if(@currentField == '', '·', toString(@currentField))",
    style: { "width": "100%", "text-align": "center", "padding": "2px 0",
      "background-color": "=if(@currentField == '', '#f3f5f8', if(@currentField >= 95, '#d3eadd', if(@currentField >= 85, '#e6efd9', if(@currentField >= 70, '#f8ebc9', '#f2d4cf'))))" } });
}
function spChange(): string {
  return JSON.stringify({ $schema: SP_COL_SCHEMA, elmType: "div",
    txtContent: "=if(@currentField == '', '–', if(@currentField > 0, '▲ +' + toString(@currentField), if(@currentField < 0, '▼ ' + toString(@currentField), '● 0')))",
    style: { "font-weight": "600", "color": "=if(@currentField > 0, '#1e9e6a', if(@currentField < 0, '#c73a3a', '#5b6578'))" } });
}

/** Views (one set per period) and the column formatters, by list key. */
function spViews(): SpView[] {
  const x = (v: string): string => v.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/'/g, "&apos;");
  const eq = (f: string, v: string): string => "<Eq><FieldRef Name='" + f + "'/><Value Type='Text'>" + x(v) + "</Value></Eq>";
  const and = (a: string, b: string): string => "<And>" + a + b + "</And>";
  const where = (c: string): string => "<Where>" + c + "</Where>";
  const order = (...f: string[]): string => "<OrderBy>" + f.map(s => "<FieldRef Name='" + s.replace(/ desc$/, "") + "' Ascending='" + (/ desc$/.test(s) ? "FALSE" : "TRUE") + "'/>").join("") + "</OrderBy>";
  const group = (f: string): string => "<GroupBy Collapse='TRUE' GroupLimit='100'><FieldRef Name='" + f + "'/></GroupBy>";
  const counts = ["InScope", "Reconciled", "NotReconciled", "Ticketed", "NotTicketed", "NotReported", "VarianceSum"];
  const v: SpView[] = [];
  SP_PERIODS.forEach(P => {
    v.push({ list: "summary", title: P + " KPI", query: where(eq("Period", P)) + order("SortKey"), rowLimit: 1, paged: false,
      fields: ["Title", "Covers", "DataThrough", "Highlights", "Rate"].concat(counts), formatter: spKpiFormatter() });
    v.push({ list: "summary", title: P + " Customers", query: where(and(eq("Period", P), "<Gt><FieldRef Name='SortKey'/><Value Type='Number'>0</Value></Gt>")) + order("SortKey"),
      rowLimit: 100, paged: true, fields: ["Title"].concat(counts, ["Rate", "Status"]), formatter: spStatusRows() });
    v.push({ list: "interfaces", title: P + " Interfaces", query: group("Customer") + where(eq("Period", P)) + order("Customer", "VarianceSum desc", "NotReported desc"),
      rowLimit: 100, paged: true, fields: ["Title", "InScope", "Received", "Loaded", "Errors", "Pending", "VarianceSum", "Reconciled", "Ticketed", "NotTicketed", "NotReported", "Status"], formatter: spStatusRows() });
    v.push({ list: "issues", title: P + " Variances", query: where(and(eq("Period", P), eq("IssueType", "Variance"))) + order("SortKey"),
      rowLimit: 30, paged: true, fields: ["Customer", "Title", "Day", "Amount", "Ticket"], formatter: spTicketRows() });
    v.push({ list: "issues", title: P + " Not reported", query: where(and(eq("Period", P), eq("IssueType", "Not reported"))) + order("SortKey"),
      rowLimit: 30, paged: true, fields: ["Customer", "Title", "Day"], formatter: "" });
    v.push({ list: "issues", title: P + " Anomalies", query: where(and(eq("Period", P), eq("IssueType", "Volume anomaly"))) + order("SortKey"),
      rowLimit: 15, paged: true, fields: ["Customer", "Title", "Day", "Amount", "Usual", "ChangePct"], formatter: "" });
  });
  v.push({ list: "trend", title: "Last " + SP_TREND_DAYS + " days", query: group("Title") + order("Title", "DayDate"), rowLimit: 100, paged: true,
    fields: ["Day", "Rate", "NotReconciled", "NotReported", "InScope"], formatter: "" });
  const weeks: string[] = [];
  for (let k = 1; k <= SP_WEEKS; k++) weeks.push(spWeekField(k));
  v.push({ list: "scorecard", title: "Scorecard", query: order("SortKey"), rowLimit: 100, paged: true, fields: ["Title"].concat(weeks, ["Change"]), formatter: "" });
  v.push({ list: "recurring", title: "Recurring", query: order("SortKey"), rowLimit: 30, paged: true,
    fields: ["Customer", "Title", "WeeksAffected", "Days", "LongestRun", "VarianceSum", "TicketedPct", "LastSeen"], formatter: "" });
  return v;
}
function spColumnFormatters(): { list: string; field: string; formatter: string }[] {
  const out = [
    { list: "summary", field: "Rate", formatter: spPercent() },
    { list: "trend", field: "Rate", formatter: spRateBar() },
    { list: "scorecard", field: "Change", formatter: spChange() },
    { list: "recurring", field: "TicketedPct", formatter: spPercent() },
    { list: "issues", field: "ChangePct", formatter: spPercent() }];
  for (let k = 1; k <= SP_WEEKS; k++) out.push({ list: "scorecard", field: spWeekField(k), formatter: spHeat() });
  return out;
}

/* ---------- $batch requests ---------- */

function spUrl(site: string, list: string, rest: string): string {
  return site.replace(/\/$/, "") + "/_api/web/lists/getbytitle('" + encodeURIComponent(list.replace(/'/g, "''")) + "')" + rest;
}
function spOp(method: string, url: string, body: string, extraHeaders: string[]): string {
  const h = ["Accept: application/json;odata=verbose"].concat(body ? ["Content-Type: application/json;odata=verbose"] : [], extraHeaders);
  return method + " " + url + " HTTP/1.1\r\n" + h.join("\r\n") + "\r\n\r\n" + body;
}
/** Splits operations into $batch requests of at most `size` operations, keeping their order. */
function spBatch(ops: string[], size: number, label: string, start: number): SpBatch[] {
  const out: SpBatch[] = [];
  for (let k = 0; k < ops.length; k += size) {
    const id = (start + out.length + 1) + "_" + Math.floor(Math.random() * 1e9).toString(16);
    const b = "batch_" + id, c = "changeset_" + id;
    const part = ops.slice(k, k + size);
    let body = "--" + b + "\r\nContent-Type: multipart/mixed; boundary=\"" + c + "\"\r\nContent-Transfer-Encoding: binary\r\n\r\n";
    part.forEach(op => { body += "--" + c + "\r\nContent-Type: application/http\r\nContent-Transfer-Encoding: binary\r\n\r\n" + op + "\r\n\r\n"; });
    body += "--" + c + "--\r\n\r\n--" + b + "--\r\n";
    out.push({ boundary: b, body, operations: part.length, label: label + " " + (Math.floor(k / size) + 1) + "/" + Math.ceil(ops.length / size) });
  }
  return out;
}

/** One-off: lists, columns, views and formatting, in dependency order (lists → columns → views → view columns). */
function spSetup(site: string, size: number): SpBatch[] {
  const lists = spLists(), byKey = new Map<string, SpList>();
  lists.forEach(l => byKey.set(l.key, l));
  const x = (v: string): string => v.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  const merge = ["IF-MATCH: *", "X-HTTP-Method: MERGE"];
  const phase1: string[] = [], phase2: string[] = [], phase3: string[] = [], phase4: string[] = [];
  lists.forEach(l => {
    phase1.push(spOp("POST", site.replace(/\/$/, "") + "/_api/web/lists", JSON.stringify({ __metadata: { type: "SP.List" }, BaseTemplate: 100, Title: l.title, Description: l.desc, AllowContentTypes: true, ContentTypesEnabled: false }), []));
    phase2.push(spOp("POST", spUrl(site, l.title, "/fields/getbyinternalnameortitle('Title')"), JSON.stringify({ __metadata: { type: "SP.Field" }, Title: l.titleName }), merge));
    l.fields.forEach(f => {
      const xml = "<Field Type=\"" + f.t + "\" Name=\"" + f.n + "\" StaticName=\"" + f.n + "\" DisplayName=\"" + x(f.d) + "\"" +
        (f.t === "Number" ? " Decimals=\"0\"" : "") + (f.t === "Note" ? " NumLines=\"6\" RichText=\"FALSE\"" : "") + " />";
      // Options 8 = AddFieldInternalNameHint: keep Name as the internal name
      phase2.push(spOp("POST", spUrl(site, l.title, "/fields/createfieldasxml"),
        JSON.stringify({ parameters: { __metadata: { type: "SP.XmlSchemaFieldCreationInformation" }, SchemaXml: xml, Options: 8 } }), []));
    });
  });
  spColumnFormatters().forEach(c => phase3.push(spOp("POST", spUrl(site, (byKey.get(c.list) as SpList).title, "/fields/getbyinternalnameortitle('" + c.field + "')"),
    JSON.stringify({ __metadata: { type: "SP.Field" }, CustomFormatter: c.formatter }), merge)));
  spViews().forEach(v => {
    const list = (byKey.get(v.list) as SpList).title, view = "/views/getbytitle('" + encodeURIComponent(v.title) + "')";
    phase3.push(spOp("POST", spUrl(site, list, "/views"), JSON.stringify({ __metadata: { type: "SP.View" }, Title: v.title, PersonalView: false, ViewQuery: v.query, RowLimit: v.rowLimit, Paged: v.paged }), []));
    phase4.push(spOp("POST", spUrl(site, list, view + "/viewfields/removeallviewfields"), "", []));
    v.fields.forEach(f => phase4.push(spOp("POST", spUrl(site, list, view + "/viewfields/addviewfield('" + f + "')"), "", [])));
    if (v.formatter) phase4.push(spOp("POST", spUrl(site, list, view), JSON.stringify({ __metadata: { type: "SP.View" }, CustomFormatter: v.formatter }), merge));
  });
  return spBatch(phase1, size, "Create lists", 0)
    .concat(spBatch(phase2, size, "Create columns", 100), spBatch(phase3, size, "Create views", 200), spBatch(phase4, size, "View columns", 300));
}

/** Daily: delete the lists' current items, add the new rows, retitle the scorecard week columns. */
function spDaily(site: string, info: SpListInfo[], rows: { [key: string]: SpRow[] }, weekTitles: string[], size: number, warnings: string[]): SpBatch[] {
  const del: string[] = [], add: string[] = [], meta: string[] = [];
  const merge = ["IF-MATCH: *", "X-HTTP-Method: MERGE"];
  spLists().forEach(l => {
    const i = info.filter(z => z.title === l.title)[0];
    if (!i || !i.type) { warnings.push("SharePoint list '" + l.title + "' not found — run the setup flow first."); return; }
    (i.items || []).forEach(it => del.push(spOp("POST", spUrl(site, l.title, "/items(" + it.Id + ")"), "", ["IF-MATCH: *", "X-HTTP-Method: DELETE"])));
    (rows[l.key] || []).forEach(r => {
      const body: SpRow = {};
      Object.keys(r).forEach(k => { if (r[k] !== null && r[k] !== undefined) body[k] = r[k]; });
      add.push(spOp("POST", spUrl(site, l.title, "/items"), JSON.stringify({ __metadata: { type: i.type } }).slice(0, -1) + "," + JSON.stringify(body).slice(1), []));
    });
    if (l.key === "scorecard") weekTitles.forEach((t, k) => meta.push(spOp("POST", spUrl(site, l.title, "/fields/getbyinternalnameortitle('" + spWeekField(k + 1) + "')"),
      JSON.stringify({ __metadata: { type: "SP.Field" }, Title: t }), merge)));
  });
  return spBatch(del, size, "Remove old rows", 0).concat(spBatch(meta.concat(add), size, "Add rows", 500));
}

/* ---------- the rows ---------- */

/** Scorecard column titles: start date of each of the last SP_WEEKS weeks, oldest first ("—" padding when fewer). */
function spWeekTitles(M: Model): string[] {
  const t: string[] = [];
  const lastWeek = Math.floor(M.LAST / 7);
  for (let k = SP_WEEKS - 1; k >= 0; k--) { const w = lastWeek - k; t.push(w >= 0 ? "Wk " + md(M, w * 7) : "— " + (k + 1)); }
  return t;
}

function spData(M: Model, months: number[][], mo: number, periods: SpPeriod[], through: string): { [key: string]: SpRow[] } {
  const out: { [key: string]: SpRow[] } = { summary: [], interfaces: [], issues: [], trend: [], scorecard: [], recurring: [] };
  const plain = (h: string): string => h.replace(/<\/?b>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
  const status = (attN: number, att: number, miss: number, g: number): string =>
    attN ? "Variance without ticket" : att ? "Variance with ticket" : miss ? "Not reported" : g ? "Reconciled" : "";
  const day = (d: number): string => DOW[d % 7] + " " + md(M, d);
  const counts = (o: { scope: number; g: number; att: number; attT: number; attN: number; miss: number; sum: number }): SpRow =>
    ({ InScope: o.scope, Reconciled: o.g, NotReconciled: o.att, Ticketed: o.attT, NotTicketed: o.attN, NotReported: o.miss, VarianceSum: Math.round(o.sum) });

  periods.forEach(P => {
    const o = scan(M, P.d1, P.d2, -1, -1, false);
    const who: { a: number; o: Scan }[] = [{ a: -1, o }];
    o.byA.map((b, a) => ({ a, b })).filter(z => z.b.g + z.b.att + z.b.miss + z.b.scope > 0)
      .sort((p, q) => q.b.sum - p.b.sum || q.b.miss - p.b.miss || M.acc[p.a].localeCompare(M.acc[q.a]))
      .forEach(z => who.push({ a: z.a, o: scan(M, P.d1, P.d2, z.a, -1, false) }));
    who.forEach((w, rank) => {
      const name = w.a < 0 ? SP_ALL : M.acc[w.a];
      const hl = findings(M, P.v, P.d1, P.d2, w.a, w.o, months, mo).map((h, k) => (k + 1) + ". " + plain(h)).join("\n");
      const r: SpRow = { Title: name, Period: P.name, Covers: P.label, SortKey: rank, Status: status(w.o.attN, w.o.att, w.o.miss, w.o.g), Rate: w.o.rate, Highlights: hl, DataThrough: through };
      const c = counts(w.o);
      Object.keys(c).forEach(k => { r[k] = c[k]; });
      out.summary.push(r);
      w.o.byI.forEach((b, i) => {
        if (!(b.scope || b.g + b.att + b.miss || b.rec)) return;
        const ri: SpRow = { Title: M.ifs[i], Customer: name, Period: P.name, Status: status(b.attN, b.att, b.miss, b.g), Received: b.rec, Loaded: b.ld, Errors: b.err, Pending: b.pend };
        const ci = counts(b);
        Object.keys(ci).forEach(k => { ri[k] = ci[k]; });
        out.interfaces.push(ri);
      });
    });
    o.items.slice(0, SP_CAP_VARIANCES).forEach((x, k) => out.issues.push({ Title: M.ifs[x.i], Customer: M.acc[x.a], Period: P.name, IssueType: "Variance",
      Day: day(x.d), DayDate: iso(M, x.d), Amount: Math.round(x.v), Ticket: x.t ? "Tracked" : "Not tracked", SortKey: k + 1 }));
    o.missL.slice().sort((p, q) => q.d - p.d || M.acc[p.a].localeCompare(M.acc[q.a]) || p.i - q.i).slice(0, SP_CAP_NOT_REPORTED)
      .forEach((x, k) => out.issues.push({ Title: M.ifs[x.i], Customer: M.acc[x.a], Period: P.name, IssueType: "Not reported", Day: day(x.d), DayDate: iso(M, x.d), SortKey: k + 1 }));
    o.anom.slice(0, SP_CAP_ANOMALIES).forEach((x, k) => out.issues.push({ Title: M.ifs[x.i], Customer: M.acc[x.a], Period: P.name, IssueType: "Volume anomaly",
      Day: day(x.d), DayDate: iso(M, x.d), Amount: x.v, Usual: x.m, ChangePct: Math.round((x.v - x.m) / x.m * 100), SortKey: k + 1 }));
  });

  // daily trend, portfolio + each customer
  const from = Math.max(0, M.LAST - SP_TREND_DAYS + 1);
  for (let a = -1; a < M.NA; a++) for (let d = from; d <= M.LAST; d++) {
    const x = scan(M, d, d, a, -1, true);
    if (!(x.scope + x.g + x.att + x.miss)) continue;
    out.trend.push({ Title: a < 0 ? SP_ALL : M.acc[a], Day: day(d), DayDate: iso(M, d), Rate: x.rate, NotReconciled: x.att, NotReported: x.miss, InScope: x.scope });
  }

  // weekly scorecard (last SP_WEEKS weeks; change = average of the last 4 weeks vs the 4 before, over all weeks)
  const lastWeek = Math.floor(M.LAST / 7);
  const wEnd = (w: number): number => Math.min(M.LAST, w * 7 + 6);
  const avg = (v: (number | null)[]): number | null => { const n = v.filter(z => z !== null) as number[]; return n.length ? n.reduce((p, q) => p + q, 0) / n.length : null; };
  const card = (title: string, sort: number, val: (x: Scan) => number | null, a: number): void => {
    const all: (number | null)[] = [];
    for (let w = 0; w <= lastWeek; w++) all.push(val(scan(M, w * 7, wEnd(w), a, -1, true)));
    if (!all.some(z => z !== null)) return;
    const r: SpRow = { Title: title, SortKey: sort };
    for (let k = 1; k <= SP_WEEKS; k++) { const w = lastWeek - SP_WEEKS + k; r[spWeekField(k)] = w >= 0 ? all[w] : null; }
    const rec = avg(all.slice(-4)), old = avg(all.slice(-8, -4));
    r.Change = rec !== null && old !== null && all.length >= 8 ? Math.round(rec - old) : null;
    out.scorecard.push(r);
  };
  card(SP_ALL, 0, x => pct(x.g, x.att), -1);
  M.acc.forEach((n, a) => card(n, a + 1, x => pct(x.g, x.att), a));
  card(SP_ALL + " – ticket coverage", 900, x => x.att ? Math.round(100 * x.attT / x.att) : null, -1);
  card(SP_ALL + " – schedule adherence", 901, x => { const t = x.g + x.att + x.miss; return t ? Math.round(100 * (1 - x.miss / t)) : null; }, -1);

  recurring(M).slice(0, SP_CAP_RECURRING).forEach((p, k) => out.recurring.push({ Title: M.ifs[p.i], Customer: M.acc[p.a], WeeksAffected: p.wk.size, Days: p.n,
    LongestRun: p.str, VarianceSum: Math.round(p.sum), TicketedPct: Math.round(100 * p.t / p.n), LastSeen: iso(M, p.last), SortKey: k + 1 }));
  return out;
}
