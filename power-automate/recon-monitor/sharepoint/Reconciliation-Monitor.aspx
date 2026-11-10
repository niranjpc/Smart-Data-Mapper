<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reconciliation Monitor</title>
<style>
:root{--navy:#161d3f;--teal:#1fb5ad;--teal2:#6fd9d4;--ink:#1c2433;--mut:#5b6578;--line:#e1e7ee;--bg:#f6f8fb;--g:#1e9e6a;--a:#c77700;--r:#c73a3a;--b:#2f6fed;--o:#c2410c}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:"Segoe UI","Segoe UI Web (West European)",-apple-system,BlinkMacSystemFont,Roboto,"Helvetica Neue",sans-serif;font-size:14px}
.wrap{max-width:1280px;margin:0 auto;padding:18px 22px 56px}
.pv{background:#fff7e0;border:1px solid #f0d58a;color:#6b4e00;border-radius:8px;padding:5px 12px;margin-bottom:10px;font-size:12px}
.hero{background:var(--navy);color:#fff;border-radius:14px;padding:22px 26px;display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
.kick{color:var(--teal2);font-size:12px;font-weight:600;letter-spacing:.12em;text-transform:uppercase}
.hero h1{margin:4px 0 2px;font-size:32px;font-weight:600}.hero .sub{color:#c9d0e6}
.hr{display:flex;flex-direction:column;align-items:flex-end;gap:10px}
.seg{display:flex;border-radius:8px;overflow:hidden;border:1px solid rgba(255,255,255,.25)}
.seg button{background:rgba(255,255,255,.12);color:#fff;border:0;padding:9px 18px;font:inherit;font-weight:600;cursor:pointer;border-right:1px solid rgba(255,255,255,.2)}
.seg button:last-child{border:0}.seg button.on{background:var(--teal);color:#06231f}
.ghost{background:transparent;color:#fff;border:1px solid rgba(255,255,255,.4);border-radius:8px;padding:8px 16px;font:inherit;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block}
.bar{display:flex;gap:10px;align-items:center;margin:14px 2px 4px;flex-wrap:wrap;color:var(--mut)}
.bar input,.bar select,.bar button{border:1px solid var(--line);border-radius:8px;padding:6px 10px;font:inherit;background:#fff;color:var(--ink)}.bar button{cursor:pointer}
h2.sec{font-size:20px;font-weight:600;margin:22px 0 10px}
.tiles{display:grid;grid-template-columns:.8fr .8fr 1.7fr .9fr .9fr;gap:14px;margin:12px 0}
@media(max-width:1000px){.tiles{grid-template-columns:1fr 1fr}}
.tile{background:#fff;border:1px solid var(--line);border-top:3px solid var(--tc,#cfd8e3);border-radius:14px;padding:16px 18px;text-align:center}
.tile .l{font-size:11.5px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--mut)}
.tile .n{font-size:36px;font-weight:600;margin:4px 0;line-height:1.1}.tile .d{font-size:12.5px;color:var(--mut)}
.tile.nf .subs{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
.sub2{border:1px solid var(--line);border-radius:10px;padding:8px;background:#fafbfd}.sub2 .n{font-size:26px;margin:0}.sub2 .l{font-size:10.5px}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0}
.card h2{margin:0 0 2px;font-size:17px;font-weight:600}.card .hint{color:var(--mut);font-size:12.5px;margin-bottom:10px}
.split{display:grid;grid-template-columns:minmax(0,2.3fr) minmax(260px,1fr);gap:16px;align-items:start;margin:14px 0}
@media(max-width:1000px){.split{grid-template-columns:1fr}}
.dark{background:var(--navy);color:#fff;border-radius:12px;padding:16px 18px}.dark h2{margin:0 0 2px;font-size:17px;font-weight:700}.dark small{color:#aeb7d6}
.side{background:#f6f8fb;border:1px solid var(--line);border-left:4px solid var(--teal);border-radius:12px;padding:14px 16px}
.side h2{margin:0 0 8px;font-size:15px;font-weight:700}
.fi{display:flex;gap:10px;padding:8px 0;border-top:1px solid var(--line);font-size:13px;line-height:1.45}.fi:first-of-type{border-top:0}
.fi i{flex:none;width:22px;height:22px;border-radius:50%;background:var(--navy);color:#fff;font-style:normal;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center;margin-top:1px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child,.l{text-align:left}th{color:var(--mut);font-size:11px;letter-spacing:.06em;text-transform:uppercase;font-weight:600}
.lt th,.lt td{text-align:left}.cmp th,.cmp td{padding:4px 8px;font-size:12.5px}.cmp th{white-space:normal;line-height:1.15;vertical-align:bottom}.cmp td{line-height:1.2}.cmp .dot{width:8px;height:8px;margin-right:6px}
.dark table th{color:#aeb7d6;border-color:rgba(255,255,255,.14)}.dark td{border-color:rgba(255,255,255,.08)}
.dark tr.cl{cursor:pointer}.dark tr.cl:hover{background:rgba(255,255,255,.07)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}
.sc{overflow:auto;max-height:520px}
tr.o td{background:#fff7f1}tr.r td{background:#fff4f4}tr.tot td{background:#f3f5f8;border-top:2px solid var(--line)}
.chip{display:inline-block;border-radius:10px;padding:1px 9px;font-size:12px;font-weight:600;background:#eef2f7;color:#3b4660}
.note{color:var(--mut);font-size:12.5px;border-left:3px solid #e3c27a;padding:2px 9px;margin:10px 0 0}
.vm{display:flex;flex-wrap:wrap;gap:8px}
.vt{flex:1 1 var(--w,160px);min-width:150px;border-radius:12px;padding:10px 12px;background:rgba(214,140,90,var(--al,.2));border:1px solid rgba(214,140,90,.35)}
.vt b{font-size:20px;display:block}.vt small{color:#6a4a2f;display:block;line-height:1.35}.vt .t{font-size:11px;font-weight:600}
.lg{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:6px 0 0}.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
svg.ch{width:100%;height:auto}.gl{stroke:#e6ebf1}.ax{fill:var(--mut);font-size:10px}
.hm td{text-align:center;padding:5px 6px;min-width:34px}.hm td.z{color:#cfd6df}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}
</style></head><body><div class="wrap">
<div class="pv" id="pv0" style="display:none"></div>
<div class="hero"><div><div class="kick" id="kick"></div><h1 id="ttl"></h1><div class="sub" id="sub"></div></div>
<div class="hr"><div class="seg" id="seg"><button data-p="d">Daily</button><button data-p="w">Weekly</button><button data-p="m">Monthly</button><button data-p="h">Historical</button></div><div style="display:flex;gap:8px"><button class="ghost" id="back">← All customers</button><a class="ghost" id="home" href="#" target="_top" style="display:none">TechOps home</a></div></div></div>
<div class="bar" id="bar"></div>
<div id="body"></div>
</div>
<script>
/* GENERATED by sharepoint/build.mjs — edit the sources, not this file. */
/* ===== rules engine (same code as office-scripts/ReconMonitor.ts) ===== */
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
 *   mode       : "build" (default) or "inspect" (lists sheets + detected columns in warnings)
 */

/** [day, account, interface, received, loaded, error, pending, variance|null, ticket 0/1] */

const MS_DAY = 86400000;
const KEY = 100000; // day slots per (account, interface) key
const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const DEFAULT_COLUMNS            = {
  date: ["^date$", "run date", "recon date", "report date", "process(ed|ing)? date", "^day$"],
  account: ["^account", "customer", "client", "health ?plan"],
  received: ["received", "triggered"],
  loaded: ["loaded", "extracted"],
  error: ["error", "rejected"],
  pending: ["pending", "work ?basket", "\\bwb\\b"],
  variance: ["variance", "difference"],
  ticket: ["ticket", "jira", "incident"],
  comment: ["comment", "remark", "notes?$"]
};
const DEFAULT_INTERFACES           = [
  "Enrollment IB", "Claims Clearing House", "Claims IB", "Provider IB – Tax entity", "Provider IB – Practitioner",
  "Provider IB – Supplier", "Provider IB – Supplier Location", "Provider IB – Practitioner Role", "Authorization IB",
  "Accumulator IB", "ID Cards-Extract", "ID Cards-Acknowledgement", "Correspondences- Extract", "Member Packets-Extract",
  "Member Packets-Acknowledgement", "Claims Payment Extract", "External Pricing", "External Claims Editing",
  "Encounter Submission", "Encounter Rejections"
];

function main(workbook                      , configJson         = "", asOf         = "", mode         = "build")              {
  const warnings           = [];
  const cfg              = configJson && configJson.trim() ? JSON.parse(configJson) : {};
  const interfaces                 = cfg.interfaces && cfg.interfaces.length ? cfg.interfaces : DEFAULT_INTERFACES.map(n => {
    const c               = { name: n }; return c;
  });
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
  const alias = new Map                ();
  const acc           = [];
  const accIdx = new Map                ();
  const addAcc = (n        )         => {
    let k = accIdx.get(n);
    if (k === undefined) { k = acc.length; acc.push(n); accIdx.set(n, k); }
    return k;
  };
  const aliases = cfg.accountAliases || {};
  Object.keys(aliases).forEach(raw => { alias.set(norm(raw), aliases[raw]); addAcc(aliases[raw]); });
  const ignore = new Set        ((cfg.ignoreAccounts || []).map(norm));
  const dropUnmapped = (cfg.unmappedAccounts || "keep").toLowerCase() === "drop";
  const unmapped = new Set        ();
  const resolve = (raw        )         => {
    const n = norm(raw);
    if (!n || ignore.has(n)) return -1;
    const m = alias.get(n);
    if (m !== undefined) return addAcc(m);
    if (!unmapped.has(n)) { unmapped.add(n); warnings.push("Account '" + raw.trim() + "' is not in accountAliases — " + (dropUnmapped ? "dropped." : "kept under its tracker name.")); }
    return dropUnmapped ? -1 : addAcc(raw.trim());
  };

  // ---- read the tracker ----
  const sheets = new Map                  ();
  workbook.getWorksheets().forEach(ws => {
    const rng = ws.getUsedRange();
    sheets.set(norm(ws.getName()), rng ? rng.getValues() : []);
  });
  const explained = new RegExp(cfg.explainedCommentPattern || "\\b(n\\/?a|not applicable|no run|no file|holiday|on hold)\\b", "i");
  const facts = new Map              ();
  const X = new Set        ();
  const NI = ifs.length;
  let maxDay = -1, badDates = 0;
  const inspect = mode.toLowerCase() === "inspect";

  interfaces.forEach((ic, i) => {
    const sheetName = ic.sheet || ic.name;
    const values = sheets.get(norm(sheetName));
    if (!values) { warnings.push("Interface '" + ic.name + "': sheet '" + sheetName + "' not found."); return; }
    const col = (role        )           => {
      const own = ic.columns ? pick(ic.columns, role) : undefined;
      const shared = cfg.columns ? pick(cfg.columns, role) : undefined;
      return own || shared || pick(DEFAULT_COLUMNS, role) || [];
    };
    let hr = -1, cDate           = [], cAcc           = [];
    for (let r = 0; r < Math.min(values.length, 25) && hr < 0; r++) {
      cDate = findCols(values[r], col("date")); cAcc = findCols(values[r], col("account"));
      if (cDate.length && cAcc.length) hr = r;
    }
    if (hr < 0) { warnings.push("Interface '" + ic.name + "': no header row with a date and an account column in sheet '" + sheetName + "'."); return; }
    const H = values[hr];
    const cRec = findCols(H, col("received")), cLd = findCols(H, col("loaded")), cErr = findCols(H, col("error")),
      cPend = findCols(H, col("pending")), cVar = findCols(H, col("variance")), cTkt = findCols(H, col("ticket")),
      cCom = findCols(H, col("comment"));
    const cFil = ic.filterColumn ? findCols(H, [ic.filterColumn]) : [];
    const names = (c          )         => c.length ? c.map(k => String(H[k]).trim()).join(" + ") : "—";
    if (inspect) warnings.push("[inspect] " + ic.name + " <- sheet '" + sheetName + "' header row " + (hr + 1) + " | date: " + names(cDate.slice(0, 1)) +
      " | account: " + names(cAcc.slice(0, 1)) + " | received: " + names(cRec) + " | loaded: " + names(cLd) + " | error: " + names(cErr) +
      " | pending: " + names(cPend) + " | variance: " + names(cVar.slice(0, 1)) + " | ticket: " + names(cTkt.slice(0, 1)) + " | comment: " + names(cCom.slice(0, 1)));
    if (!cRec.length) warnings.push("Interface '" + ic.name + "': no 'received' column matched.");
    const computeVar = !cVar.length && (cfg.varianceWhenMissing || "none").toLowerCase() !== "none";
    if (!cVar.length) warnings.push("Interface '" + ic.name + "': no variance column matched — " + (computeVar ? "using received − loaded − error − pending." : "every reported row will count as reconciled."));

    for (let r = hr + 1; r < values.length; r++) {
      const row = values[r];
      const dv = row[cDate[0]];
      if (dv === "" || dv === null || dv === undefined) continue;
      const day = typeof dv === "number" ? Math.floor(dv) - 25569 : parseDay(String(dv), order);
      if (day === null) { badDates++; continue; }
      const d = day - T0;
      if (d < 0 || d > cap || d >= KEY) continue;
      if (cFil.length && norm(String(row[cFil[0]])) !== norm(ic.filterEquals || "")) continue;
      const a = resolve(String(row[cAcc[0]]));
      if (a < 0) continue;
      let hasNum = false, isExplained = cCom.length > 0 && explained.test(String(row[cCom[0]]));
      const total = (cols          )         => {
        let s = 0;
        cols.forEach(k => {
          const n = num(row[k]);
          if (n !== null) { s += n; hasNum = true; } else if (explained.test(String(row[k]))) isExplained = true;
        });
        return s;
      };
      const rec = total(cRec), ld = total(cLd), er = total(cErr), pe = total(cPend);
      let v                = cVar.length ? num(row[cVar[0]]) : null;
      if (v !== null) hasNum = true;
      if (v === null && computeVar && hasNum) v = rec - ld - er - pe;
      const k = (a * NI + i) * KEY + d;
      if (isExplained && !hasNum) { X.add(k); if (d > maxDay) maxDay = d; continue; }
      const tk = cTkt.length && isTicket(row[cTkt[0]]) ? 1 : 0;
      const z = facts.get(k);
      if (!z) facts.set(k, [d, a, i, rec, ld, er, pe, v === null ? null : Math.abs(v), tk]);
      else {
        z[3] += rec; z[4] += ld; z[5] += er; z[6] += pe;
        z[7] = z[7] === null && v === null ? null : (z[7] || 0) + Math.abs(v || 0);
        z[8] = z[8] || tk;
      }
      if (d > maxDay) maxDay = d;
    }
  });
  if (badDates) warnings.push(badDates + " row(s) skipped because the date could not be read (check dateOrder).");

  // ---- schedule ----
  const sch                                             = {};
  const schName = cfg.scheduleSheet || "Recon Report Schedule";
  const sv = sheets.get(norm(schName));
  if (!sv) warnings.push("Schedule sheet '" + schName + "' not found — nothing can be flagged as scheduled but not reported.");
  else readSchedule(sv, typ, resolve, acc, sch, warnings, inspect);

  if (maxDay < 0) warnings.push("No tracker rows were read. Run with mode = inspect and adjust recon-config.json.");
  const LAST = Math.max(0, asOfDay === null ? maxDay : Math.min(maxDay, cap));
  const NA = acc.length;

  const model        = { acc, ifs, typ, sch, F: facts, X, NA, NI, T0, LAST, DEF: LAST, anomMin: cfg.anomalyMinMedian || 200, anomFactor: cfg.anomalyFactor || 3 };
  const perDayRows           = [];
  facts.forEach(r => { perDayRows[r[0]] = (perDayRows[r[0]] || 0) + 1; });
  const minRows = cfg.minRowsForDefaultDay === undefined ? 100 : cfg.minRowsForDefaultDay;
  while (model.DEF > 0 && (perDayRows[model.DEF] || 0) < minRows) model.DEF--;

  // ---- dataset for the dashboard page ----
  const generated = new Date().toISOString();
  const fOut         = []; facts.forEach(r => fOut.push(r));
  const xOut             = []; X.forEach(k => xOut.push([k % KEY, Math.floor(Math.floor(k / KEY) / NI), Math.floor(k / KEY) % NI]));
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
  const needsAttention = oD.attN > 0 || oD.miss > 0;
  const subject = "Reconciliation Monitor · " + DOW[DEF % 7] + " " + md(model, DEF) + " · " + (oD.rate === null ? "no feeds reported" : oD.rate + "% reconciled") +
    (oD.attN ? " · " + oD.attN + " without ticket" : "") + (oD.miss ? " · " + oD.miss + " not reported" : "");

  return {
    ok: maxDay >= 0, asOf: data.asOf, defaultDay: iso(model, DEF), generated,
    dataJson, digestJson: JSON.stringify(digest),
    daily: tiles(oD), weekly: tiles(oW), monthly: tiles(oM),
    dailyLabel, weeklyLabel, monthlyLabel, needsAttention, subject,
    emailHtml: emailHtml(model, dailyLabel, oD, hD, weeklyLabel, oW, cfg.dashboardUrl || ""), warnings
  };
}

/* ======================= reading helpers ======================= */

function norm(s        )         { return String(s === null || s === undefined ? "" : s).replace(/\s+/g, " ").trim().toLowerCase(); }

function pick(m           , role        )                       {
  switch (role) {
    case "date": return m.date; case "account": return m.account; case "received": return m.received;
    case "loaded": return m.loaded; case "error": return m.error; case "pending": return m.pending;
    case "variance": return m.variance; case "ticket": return m.ticket; case "comment": return m.comment;
  }
  return undefined;
}

/** Indexes of the header cells matching any pattern (case-insensitive regular expressions). */
function findCols(header        , patterns          )           {
  const out           = [];
  const res = patterns.map(p => new RegExp(p, "i"));
  for (let c = 0; c < header.length; c++) {
    const h = String(header[c] === null || header[c] === undefined ? "" : header[c]).replace(/\s+/g, " ").trim();
    if (h && res.some(re => re.test(h))) out.push(c);
  }
  return out;
}

function num(v      )                {
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (typeof v !== "string") return null;
  let s = v.replace(/[,\s]/g, "");
  let neg = false;
  if (/^\(.*\)$/.test(s)) { neg = true; s = s.slice(1, -1); }
  if (!/^-?\d+(\.\d+)?$/.test(s)) return null;
  return neg ? -Number(s) : Number(s);
}

function isTicket(v      )          {
  const s = norm(String(v === null || v === undefined ? "" : v));
  return s !== "" && !/^(n\/?a|no|none|nil|-|0|false)$/.test(s);
}

/** Text date -> days since 1970-01-01 (UTC). order: YMD | MDY | DMY for all-numeric dates. */
function parseDay(s        , order        )                {
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
function monthNo(s        )         { return MONTHS.map(x => x.toLowerCase()).indexOf(s.slice(0, 3).toLowerCase()) + 1; }

/** "Mon-Fri", "Daily", "Tue, Thu", "1111100", "Y" ... -> seven 0/1 flags, Monday first. null = not understood. */
function dayFlags(v      )                {
  const s = norm(String(v === null || v === undefined ? "" : v));
  if (/^[01]{7}$/.test(s)) return s;
  if (s === "" || /^(n\/?a|no|none|nil|-|0|false|not applicable)$/.test(s)) return "0000000";
  if (/^(y|yes|x|true|1|daily|every ?day|all days|7 days)$/.test(s)) return "1111111";
  if (/^(weekdays?|business days|week days)$/.test(s)) return "1111100";
  const idx = (w        )         => /^(mon(day)?|tue(s|sday)?|wed(nesday)?|thu(r|rs|rsday)?|fri(day)?|sat(urday)?|sun(day)?)s?\.?$/.test(w)
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
function readSchedule(v          , typ          , resolve                         , acc          ,
  sch                                            , warnings          , inspect         )       {
  const types           = [];
  typ.forEach(t => { if (types.indexOf(t) < 0) types.push(t); });
  const typeOf = (label        )                => {
    const n = norm(label).replace(/[–—]/g, "-");
    for (let k = 0; k < types.length; k++) if (norm(types[k]).replace(/[–—]/g, "-") === n) return types[k];
    return null;
  };
  let hr = -1, cAcc           = [];
  for (let r = 0; r < Math.min(v.length, 25) && hr < 0; r++) { cAcc = findCols(v[r], ["^account", "customer", "client", "health ?plan"]); if (cAcc.length) hr = r; }
  if (hr < 0) { warnings.push("Schedule sheet: no header row with an account column."); return; }
  const H = v[hr];
  const cIf = findCols(H, ["interface", "feed", "report", "^type$"]);
  const cDays           = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"].map(dn => { const c = findCols(H, ["^" + dn]); return c.length ? c[0] : -1; });
  const cExpr = findCols(H, ["schedule", "frequency", "^days?$"]);
  const layout = cIf.length && cDays.filter(c => c >= 0).length >= 5 ? "A" : cIf.length && cExpr.length ? "B" : "C";
  if (inspect) warnings.push("[inspect] schedule layout " + layout + ", header row " + (hr + 1));
  const unknown = new Set        ();
  const put = (a        , t        , flags        )       => {
    const s = sch[acc[a]] || (sch[acc[a]] = {});
    const o = s[t] || "0000000";
    let m = "";
    for (let k = 0; k < 7; k++) m += flags.charAt(k) === "1" || o.charAt(k) === "1" ? "1" : "0";
    s[t] = m;
  };
  for (let r = hr + 1; r < v.length; r++) {
    const raw = String(v[r][cAcc[0]]);
    if (!raw.trim()) continue;
    const a = resolve(raw);
    if (a < 0) continue;
    if (layout === "C") {
      for (let c = 0; c < H.length; c++) {
        if (c === cAcc[0] || !String(H[c]).trim()) continue;
        const t = typeOf(String(H[c]));
        if (!t) { unknown.add(String(H[c]).trim()); continue; }
        const fl = dayFlags(v[r][c]);
        if (fl === null) warnings.push("Schedule: '" + String(v[r][c]) + "' (" + raw.trim() + " / " + t + ") not understood — treated as not scheduled.");
        put(a, t, fl || "0000000");
      }
    } else {
      const t = typeOf(String(v[r][cIf[0]]));
      if (!t) { if (String(v[r][cIf[0]]).trim()) unknown.add(String(v[r][cIf[0]]).trim()); continue; }
      let fl                = "";
      if (layout === "A") cDays.forEach(c => { fl += c >= 0 && dayFlags(v[r][c]) === "1111111" ? "1" : "0"; });
      else fl = dayFlags(v[r][cExpr[0]]);
      if (fl === null) warnings.push("Schedule: '" + String(v[r][cExpr[0]]) + "' (" + raw.trim() + " / " + t + ") not understood — treated as not scheduled.");
      put(a, t, fl || "0000000");
    }
  }
  unknown.forEach(u => warnings.push("Schedule: '" + u + "' does not match any interface scheduleType — ignored."));
}

/* ======================= dashboard rules ======================= */

function iso(M       , i        )         { return new Date((M.T0 + i) * MS_DAY).toISOString().slice(0, 10); }
function md(M       , i        )         { const s = iso(M, i); return (+s.slice(5, 7)) + "/" + (+s.slice(8, 10)); }
function dayIdx(M       , y        , m        , d        )         { return Math.round(Date.UTC(y, m, d) / MS_DAY) - M.T0; }
function full(n               )         {
  if (n === null) return "–";
  const r = Math.round(n);
  return (r < 0 ? "-" : "") + String(Math.abs(r)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
function esc(s        )         { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function median(a          )         { const s = a.slice().sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; }
function pct(g        , att        )                { return g + att ? Math.round(100 * g / (g + att)) : null; }

/** true = scheduled that weekday, false = explicitly not scheduled, null = no schedule known. */
function sched(M       , a        , i        , d        )                 {
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
function cell(M       , a        , i        , d        )                   {
  const k = (a * M.NI + i) * KEY + d, r = M.F.get(k), s = sched(M, a, i, d);
  if (r) {
    const v = r[7];
    if (v !== null && Math.abs(v) > 0.5) return { s: "att", r, v: Math.abs(v), t: !!r[8] };
    return s === false ? null : { s: "g", r, v: 0, t: false };
  }
  if (s === true && !M.X.has(k)) return { s: "miss", r: null, v: 0, t: false };
  return null;
}

function bucket()         { return { g: 0, att: 0, attT: 0, attN: 0, miss: 0, sum: 0, scope: 0, rec: 0, ld: 0, err: 0, pend: 0, top: "", topv: 0, missI: [], an: 0 }; }

/** Aggregates days d1..d2 for one account (a0) / interface (i0), or all when -1. lite skips lists and anomalies. */
function scan(M       , d1        , d2        , a0        , i0        , lite         )       {
  const o       = { g: 0, att: 0, attT: 0, attN: 0, miss: 0, sum: 0, scope: 0, rate: null, items: [], missL: [], anom: [], byI: M.ifs.map(() => bucket()), byA: M.acc.map(() => bucket()) };
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
        const base           = [];
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

function monthList(M       )             {
  const out             = [];
  const a = new Date(M.T0 * MS_DAY), b = new Date((M.T0 + M.LAST) * MS_DAY);
  let y = a.getUTCFullYear(), m = a.getUTCMonth();
  while (y < b.getUTCFullYear() || (y === b.getUTCFullYear() && m <= b.getUTCMonth())) { out.push([y, m]); m++; if (m > 11) { m = 0; y++; } }
  return out;
}

function groupBy(items        , key                     , numeric         )          {
  const g = new Map               ();
  items.forEach(x => {
    const k = key(x);
    let z = g.get(k);
    if (!z) { z = { k, n: 0, sum: 0, x }; g.set(k, z); }
    z.n++; z.sum += Math.abs(x.v);
  });
  const out          = []; g.forEach(z => out.push(z));
  if (numeric) out.sort((p, q) => (+p.k) - (+q.k));
  return out;
}

/** "Highlights" panel: up to four sentences for a day (d), week (w) or month (m). HTML with <b>. */
function findings(M       , V        , d1        , d2        , a0        , o      , months            , mo        )           {
  const out           = [];
  const all = a0 < 0;
  const name = (k        )         => esc(all ? M.acc[+k] : M.ifs[+k]);
  const by = (x      )         => String(all ? x.a : x.i);
  let pr              = null;
  if (V === "d") pr = scan(M, Math.max(0, d1 - 1), Math.max(0, d1 - 1), a0, -1, true);
  else if (V === "w") pr = scan(M, Math.max(0, d1 - 7), Math.max(0, d1 - 1), a0, -1, true);
  else if (mo > 0) pr = scan(M, Math.max(0, dayIdx(M, months[mo - 1][0], months[mo - 1][1], 1)), dayIdx(M, months[mo - 1][0], months[mo - 1][1] + 1, 0), a0, -1, true);
  const vs = (lbl        )       => {
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
      const ms = new Set        (o.missL.map(x => x.a)).size;
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
    const R         = [];
    for (let d = d1; d <= d2; d++) R.push(scan(M, d, d, a0, -1, true));
    R.forEach((x, i) => { if (d1 + i <= M.DEF && x.att + x.miss > R[bi].att + R[bi].miss) bi = i; });
    if (R.length) out.push("Worst day: <b>" + DOW[(d1 + bi) % 7] + " " + md(M, d1 + bi) + "</b> with " + R[bi].att + " not fully reconciled and " + R[bi].miss + " not reported.");
    vs("previous week");
  } else {
    vs("previous month");
    const wk = new Map                ();
    o.items.forEach(x => { const k = Math.floor(x.d / 7) * 7; wk.set(k, (wk.get(k) || 0) + Math.abs(x.v)); });
    const wl             = []; wk.forEach((s, k) => wl.push([k, s]));
    wl.sort((p, q) => p[0] - q[0]).sort((p, q) => q[1] - p[1]);
    if (wl.length) out.push("Heaviest week: <b>week of " + md(M, wl[0][0]) + "</b> (Σ|variance| " + full(wl[0][1]) + ").");
    const g = groupBy(o.items, by, true).sort((p, q) => q.n - p.n);
    if (g.length) out.push("Most days not reconciled: <b>" + name(g[0].k) + "</b> (" + g[0].n + " feed-days).");
    if (o.att) out.push("Ticket coverage: <b>" + Math.round(100 * o.attT / o.att) + "%</b> of variances this month are tracked via a ticket" + (pr && pr.att ? " (previous month " + Math.round(100 * pr.attT / pr.att) + "%)" : "") + ".");
  }
  return out.slice(0, 4);
}

/* ======================= output shaping ======================= */

function tiles(o      )        {
  return { inScope: o.scope, fullyReconciled: o.g, notFullyReconciled: o.att, trackedViaTicket: o.attT, notTrackedViaTicket: o.attN, notReported: o.miss, varianceSum: o.sum, reconciledRate: o.rate === null ? -1 : o.rate };
}

function section(M       , label        , d1        , d2        , o      , hl          ) {
  const day = (d        )         => DOW[d % 7] + " " + md(M, d);
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

/** Historical view: weekly scorecard per customer, recurring issues, quality over time. */
function history(M       ) {
  const weeks           = [];
  for (let w = 0; w * 7 <= M.LAST; w++) weeks.push(w * 7);
  const wEnd = (w        )         => Math.min(M.LAST, w + 6);
  const avg = (a          )         => a.reduce((x, y) => x + y, 0) / a.length;
  const notNull = (v               )          => v !== null;
  const scorecard                                                                                 = [];
  M.acc.forEach((n, a) => {
    const rs = weeks.map(w => { const x = scan(M, w, wEnd(w), a, -1, true); return pct(x.g, x.att); });
    if (!rs.some(notNull)) return;
    const rec           = [], old           = [];
    rs.slice(-4).forEach(v => { if (v !== null) rec.push(v); });
    rs.slice(-8, -4).forEach(v => { if (v !== null) old.push(v); });
    scorecard.push({ customer: n, rates: rs, last4VsPrior4: rec.length && old.length ? Math.round(avg(rec) - avg(old)) : null });
  });
  const pr = new Map             ();
  for (let d = 0; d <= M.LAST; d++) for (let a = 0; a < M.NA; a++) for (let i = 0; i < M.NI; i++) {
    const c = cell(M, a, i, d);
    if (!c || c.s !== "att") continue;
    const k = a + "|" + i;
    let p = pr.get(k);
    if (!p) { p = { a, i, wk: new Set        (), n: 0, sum: 0, t: 0, last: 0, str: 0, cur: 0, prev: -9 }; pr.set(k, p); }
    p.wk.add(Math.floor(d / 7)); p.n++; p.sum += c.v; if (c.t) p.t++;
    p.last = d; p.cur = d - p.prev <= 1 ? p.cur + 1 : 1; p.str = Math.max(p.str, p.cur); p.prev = d;
  }
  const recs        = []; pr.forEach(p => recs.push(p));
  recs.sort((p, q) => q.wk.size - p.wk.size || q.sum - p.sum);
  const W = weeks.map(w => scan(M, w, wEnd(w), -1, -1, true));
  return {
    weeks: weeks.map(w => iso(M, w)),
    scorecard,
    recurringIssues: recs.slice(0, 12).map(p => ({ customer: M.acc[p.a], interface: M.ifs[p.i], weeksAffected: p.wk.size, days: p.n, longestRunDays: p.str, varianceSum: p.sum, ticketedPct: Math.round(100 * p.t / p.n), lastSeen: iso(M, p.last) })),
    quality: W.map((x, k) => {
      const t = x.g + x.att + x.miss;
      return { week: iso(M, weeks[k]), reconciledRate: pct(x.g, x.att), ticketCoverage: x.att ? Math.round(100 * x.attT / x.att) : null, scheduleAdherence: t ? Math.round(100 * (1 - x.miss / t)) : null };
    })
  };
}

function emailHtml(M       , dLabel        , oD      , hD          , wLabel        , oW      , url        )         {
  const td = "padding:6px 10px;border-bottom:1px solid #e1e7ee;font:13px Segoe UI,Arial,sans-serif;";
  const th = td + "color:#5b6578;font-size:11px;text-transform:uppercase;text-align:left;";
  const kpi = (l        , n        , c        )         => "<td style=\"padding:10px 14px;border:1px solid #e1e7ee;border-top:3px solid " + c + ";text-align:center;font-family:Segoe UI,Arial,sans-serif\"><div style=\"font-size:11px;color:#5b6578;text-transform:uppercase\">" + l + "</div><div style=\"font-size:26px;font-weight:600;color:" + c + "\">" + n + "</div></td>";
  const row = (o      )         => "<table cellspacing=\"0\" cellpadding=\"0\" style=\"border-collapse:collapse;margin:8px 0 14px\"><tr>" +
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

/* ===== workbook reader ===== */
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
      else if (!vm) return m;
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

/* ===== dashboard ===== */
function startDashboard(D){
/* Dataset injected by the Power Automate flow (output "dataJson" of the ReconMonitor Office Script).
   Account names are already aligned and merged by the script. */

const ACC=D.acc,IFS=D.ifs,NA=ACC.length,NI=IFS.length;
const $=s=>document.querySelector(s);
const KEY=100000,MS=86400000,T0=Date.parse(D.start+'T00:00:00Z');
const iso=i=>new Date(T0+i*MS).toISOString().slice(0,10),md=i=>iso(i).slice(5).replace('-','/').replace(/^0/,'').replace(/\/0/,'/');
const idxOf=s=>Math.round((Date.parse(s+'T00:00:00Z')-T0)/MS);
const DOW=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
const LAST=idxOf(D.asOf);
const typ=D.typ||IFS.map(n=>/^Provider IB/.test(n)?'Provider IB':n==='Encounter Rejections'?'Encounter Submission':n);
const full=n=>n==null?'–':Math.round(n).toLocaleString();
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const sum=a=>a.reduce((x,y)=>x+(y||0),0);
const CG='#8cc0a8',CA='#e3a977',CR='#dc9a9a',CB='#8fb0d9',CN='#161d3f';
const F=new Map();D.f.forEach(r=>F.set((r[1]*NI+r[2])*KEY+r[0],r));
const X=new Set(D.x.map(([d,a,i])=>(a*NI+i)*KEY+d));
const sched=(a,i,d)=>{const s=D.sch[ACC[a]];if(!s)return null;const t=s[typ[i]];return t==null?null:t[d%7]==='1';};
/* g fully reconciled | att not fully reconciled (t = ticket) | miss scheduled but not reported; null = n/a (incl. rows explained by a comment) */
function cell(a,i,d){const k=(a*NI+i)*KEY+d,r=F.get(k),s=sched(a,i,d);
 if(r){const v=r[7];if(v!=null&&Math.abs(v)>0.5)return{s:'att',r,v:Math.abs(v),t:!!r[8]};return s===false?null:{s:'g',r,v:0};}
 if(s===true&&!X.has(k))return{s:'miss'};return null;}
const DC={};D.f.forEach(r=>DC[r[0]]=(DC[r[0]]||0)+1);let DEF=LAST;if(typeof D.def==='number')DEF=D.def;else while(DEF>0&&(DC[DEF]||0)<100)DEF--;
const S={p:'w',a:-1,f:-1,from:0,to:DEF};
const median=a=>{const s=a.slice().sort((x,y)=>x-y);return s.length?s[s.length>>1]:0;};
function scan(d1,d2,a0,lite){const i0=S.f;const o={g:0,att:0,attT:0,attN:0,miss:0,sum:0,scope:0,items:[],missL:[],anom:[],byI:IFS.map(()=>({g:0,att:0,attT:0,attN:0,miss:0,sum:0,scope:0,rec:0,ld:0,err:0,pend:0})),byA:ACC.map(()=>({g:0,att:0,attT:0,attN:0,miss:0,sum:0,scope:0,top:null,topv:0,missI:[],an:0}))};
 for(let d=d1;d<=d2;d++)for(let a=0;a<NA;a++){if(a0>=0&&a!==a0)continue;for(let i=0;i<NI;i++){if(i0>=0&&i!==i0)continue;const A=o.byA[a],B=o.byI[i];if(sched(a,i,d)===true){o.scope++;A.scope++;B.scope++;}
  const c=cell(a,i,d);if(!c)continue;if(c.r){B.rec+=c.r[3];B.ld+=c.r[4];B.err+=c.r[5];B.pend+=c.r[6];}
  if(c.s==='g'){o.g++;A.g++;B.g++;}
  else if(c.s==='att'){o.att++;A.att++;B.att++;B.sum+=c.v;A.sum+=c.v;o.sum+=c.v;if(c.t){o.attT++;A.attT++;B.attT++}else{o.attN++;A.attN++;B.attN++}if(!lite)o.items.push({a,i,d,v:c.r[7],t:c.t});if(c.v>A.topv){A.topv=c.v;A.top=IFS[i]+' ('+md(d)+')';}}
  else{o.miss++;A.miss++;B.miss++;if(A.missI.length<3&&!A.missI.includes(IFS[i]))A.missI.push(IFS[i]);if(!lite)o.missL.push({a,i,d});}
  if(!lite&&c.r&&c.r[3]>0){const base=[];for(let e=d-1;e>=Math.max(0,d-45)&&base.length<20;e--){const q=F.get((a*NI+i)*KEY+e);if(q&&q[3]>0)base.push(q[3]);}
   if(base.length>=6){const m=median(base),v=c.r[3];if(m>=200&&(v>3*m||(v<m/3&&d<LAST))){o.anom.push({a,i,d,v,m,dev:Math.abs(v-m)});A.an++;}}}}}
 o.items.sort((p,q)=>Math.abs(q.v)-Math.abs(p.v));o.anom.sort((p,q)=>q.dev-p.dev);o.rate=(o.g+o.att)?Math.round(100*o.g/(o.g+o.att)):null;return o;}
/* ---------- components ---------- */
const tile=(lab,n,col,d,cls)=>`<div class="tile ${cls||''}" style="--tc:${col}"><div class="l">${lab}</div><div class="n" style="color:${col}">${n}</div>${d?`<div class="d">${d}</div>`:''}</div>`;
const tiles=o=>`<div class="tiles">`+tile('In scope',o.scope,CN,'Feeds scheduled')+tile('Fully reconciled',o.g,'var(--g)','No variance')
 +`<div class="tile nf" style="--tc:var(--a)"><div class="l">Not fully reconciled</div><div class="n" style="color:var(--a)">${o.att}</div><div class="d">Counts do not reconcile · Σ|variance| ${full(o.sum)}</div><div class="subs"><div class="sub2"><div class="l">Tracked via ticket</div><div class="n" style="color:var(--b)">${o.attT}</div></div><div class="sub2"><div class="l">Not tracked via ticket</div><div class="n" style="color:var(--o)">${o.attN}</div></div></div></div>`
 +tile('Scheduled but not reported',o.miss,'var(--r)','Expected, no row')+tile('Reconciled rate',o.rate==null?'–':o.rate+'%','var(--g)','Of feeds reported')+'</div>';
function stack(labels,series,h0){const w=900,h=h0||210,m={l:38,r:8,t:8,b:24},n=labels.length,bw=(w-m.l-m.r)/n,mx=Math.max(1,...labels.map((_,i)=>series.reduce((s,x)=>s+x.data[i],0))),Y=v=>m.t+(h-m.t-m.b)*(1-v/mx);let s=`<svg viewBox="0 0 ${w} ${h}" class="ch">`;
 for(let k=0;k<=4;k++){const v=mx*k/4,y=Y(v);s+=`<line x1="${m.l}" x2="${w-m.r}" y1="${y}" y2="${y}" class="gl"/><text x="${m.l-5}" y="${y+4}" class="ax" text-anchor="end">${Math.round(v)}</text>`;}
 const st=Math.ceil(n/16);labels.forEach((l,i)=>{let acc=0;series.forEach(x=>{const v=x.data[i];if(!v)return;const y0=Y(acc+v),y1=Y(acc);s+=`<rect x="${m.l+i*bw+2}" y="${y0}" width="${Math.max(1,bw-4)}" height="${y1-y0}" rx="2" fill="${x.color}"><title>${l} · ${x.name}: ${v}</title></rect>`;acc+=v;});if(i%st===0)s+=`<text x="${m.l+i*bw+bw/2}" y="${h-7}" class="ax" text-anchor="middle">${l}</text>`;});
 return s+'</svg><div class="lg">'+series.map(x=>`<span><i style="background:${x.color}"></i>${x.name}</span>`).join('')+'</div>';}
function lines(labels,series,h0){const w=900,h=h0||220,m={l:42,r:10,t:10,b:24},n=labels.length,X=i=>m.l+(i+.5)*(w-m.l-m.r)/n,Y=v=>m.t+(h-m.t-m.b)*(1-v/100);let s=`<svg viewBox="0 0 ${w} ${h}" class="ch">`;
 for(let k=0;k<=4;k++){const v=25*k,y=Y(v);s+=`<line x1="${m.l}" x2="${w-m.r}" y1="${y}" y2="${y}" class="gl"/><text x="${m.l-5}" y="${y+4}" class="ax" text-anchor="end">${v}%</text>`;}
 labels.forEach((l,i)=>s+=`<text x="${X(i)}" y="${h-7}" class="ax" text-anchor="middle">${l}</text>`);
 series.forEach(x=>{let d='';x.data.forEach((v,i)=>{if(v==null)return;d+=(d?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)});s+=`<path d="${d}" fill="none" stroke="${x.color}" stroke-width="2.2" stroke-linejoin="round"/>`+x.data.map((v,i)=>v==null?'':`<circle cx="${X(i)}" cy="${Y(v)}" r="3" fill="#fff" stroke="${x.color}" stroke-width="1.8"><title>${labels[i]} · ${x.name}: ${v}%</title></circle>`).join('');});
 return s+'</svg><div class="lg">'+series.map(x=>`<span><i style="background:${x.color}"></i>${x.name}</span>`).join('')+'</div>';}
const SER=rows=>[{name:'Fully reconciled',color:CG,data:rows.map(x=>x.g)},{name:'Not fully reconciled',color:CA,data:rows.map(x=>x.att)},{name:'Scheduled but not reported',color:CR,data:rows.map(x=>x.miss)}];
const perDay=(d1,d2,a0)=>{const o=[];for(let d=d1;d<=d2;d++){const x=scan(d,d,a0,1);o.push({g:x.g,att:x.att,miss:x.miss});}return o;};
const perWeek=a0=>{const o=[];for(let w=0;w*7<=LAST;w++){const x=scan(w*7,Math.min(LAST,w*7+6),a0,1);o.push(x);}return o;};
function trend(s,e,a0,div){const R=perDay(s,e,a0),n=R.length,w=900,h=190,m={l:44,r:12,t:12,b:26},bw=(w-m.l-m.r)/n,X=i=>m.l+(i+.5)*bw;
 const rate=R.map(x=>(x.g+x.att)?Math.round(1000*x.g/(x.g+x.att))/10:null),rv=rate.filter(v=>v!=null),lo=Math.max(0,Math.floor(Math.min(...(rv.length?rv:[100]),100)/10)*10-10),Y=v=>m.t+(h-m.t-m.b)*(1-(v-lo)/(100-lo));
 let q=`<svg viewBox="0 0 ${w} ${h}" class="ch">`;
 [lo,(lo+100)/2,100].forEach(v=>{q+=`<line x1="${m.l}" x2="${w-m.r}" y1="${Y(v)}" y2="${Y(v)}" class="gl"/><text x="${m.l-6}" y="${Y(v)+3}" class="ax" text-anchor="end">${Math.round(v)}%</text>`;});
 let d='',pts=[];rate.forEach((v,i)=>{if(v==null)return;d+=(d?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1);pts.push(i)});
 if(pts.length>1)q+=`<path d="${d}L${X(pts[pts.length-1]).toFixed(1)} ${Y(lo)}L${X(pts[0]).toFixed(1)} ${Y(lo)}Z" fill="rgba(31,181,173,.12)"/>`;
 q+=`<path d="${d}" fill="none" stroke="#1fb5ad" stroke-width="2.4" stroke-linejoin="round"/>`+pts.map(i=>`<circle cx="${X(i).toFixed(1)}" cy="${Y(rate[i]).toFixed(1)}" r="3.6" fill="#fff" stroke="#1fb5ad" stroke-width="2"><title>${DOW[(s+i)%7]} ${md(s+i)} — ${rate[i]}% reconciled · ${R[i].att} not fully reconciled · ${R[i].miss} not reported</title></circle>`).join('');
 if(div!=null&&div>s){const x=m.l+(div-s)*bw;q+=`<line x1="${x}" x2="${x}" y1="${m.t}" y2="${h-m.b}" stroke="#b7c0cf" stroke-dasharray="4 4"/><text x="${x+5}" y="${m.t+8}" class="ax">this week →</text>`;}
 const st=n>16?Math.ceil(n/16):1;for(let k=0;k<n;k++){if(k%st===0)q+=`<text x="${X(k)}" y="${h-8}" class="ax" text-anchor="middle">${n<=16?DOW[(s+k)%7].slice(0,3)+' '+md(s+k):md(s+k)}</text>`;}
 return q+'</svg><div class="lg"><span><i style="background:#1fb5ad"></i>Reconciled rate — share of reported feeds with no variance. Hover a point for the day’s counts.</span></div>';}
function findings(d1,d2,a0,o){const V=S.p,out=[],ent=x=>a0<0?esc(ACC[x.a]):esc(IFS[x.i]),entN=a0<0?'customer':'interface';
 const pr=V==='d'?scan(Math.max(0,d1-1),Math.max(0,d1-1),a0,1):V==='w'?scan(Math.max(0,d1-7),Math.max(0,d1-1),a0,1):(S.mo>0?(()=>{const[y,m]=MOS[S.mo-1];return scan(Math.max(0,Math.round((Date.UTC(y,m,1)-T0)/MS)),Math.round((Date.UTC(y,m+1,0)-T0)/MS),a0,1);})():null);
 const grp=(items,f)=>{const g={};items.forEach(x=>{const k=f(x);const z=g[k]||(g[k]={k,n:0,sum:0,x});z.n++;z.sum+=Math.abs(x.v);});return Object.values(g);};
 const rt=(x)=>x&&(x.g+x.att)?Math.round(100*x.g/(x.g+x.att)):null;
 const vs=(lbl)=>{if(!pr)return;const a=rt(o),b=rt(pr);if(a==null||b==null)return;out.push(`Reconciled rate <b>${a}%</b> vs ${b}% in the ${lbl} (${a>b?'▲ better':a<b?'▼ worse':'unchanged'}); not fully reconciled ${o.att} vs ${pr.att}, not reported ${o.miss} vs ${pr.miss}.`);};
 if(V==='d'){const t=o.items[0];if(t)out.push(`Biggest variance today: <b>${a0<0?esc(ACC[t.a])+' · ':''}${esc(IFS[t.i])}</b> (${full(t.v)}), ${t.t?'tracked via a ticket':'<b>no ticket</b>'}.`);
  const un=grp(o.items.filter(x=>!x.t),x=>a0<0?x.a:x.i).sort((p,q)=>q.n-p.n);if(o.attN)out.push(`<b>${o.attN}</b> variance${o.attN>1?'s':''} need a ticket today${un.length?' — '+un.slice(0,3).map(z=>(a0<0?esc(ACC[z.k]):esc(IFS[z.k]))+' ('+z.n+')').join(', '):''}.`);else if(o.att)out.push('Every variance today is tracked via a ticket.');
  if(o.miss){const ms=new Set(o.missL.map(x=>x.a)).size;out.push(`<b>${o.miss}</b> scheduled feed${o.miss>1?'s':''} not reported today${a0<0?' across '+ms+' customer'+(ms>1?'s':''):''} (e.g. ${o.missL.slice(0,2).map(x=>(a0<0?esc(ACC[x.a])+' · ':'')+esc(IFS[x.i])).join(', ')}).`);}
  if(o.anom.length){const x=o.anom[0];out.push(`Volume check: ${a0<0?'<b>'+esc(ACC[x.a])+'</b> · ':''}${esc(IFS[x.i])} received ${full(x.v)} vs usual ~${full(x.m)}.`);}
  vs('previous day');}
 else if(V==='w'){const g=grp(o.items,x=>a0<0?x.a:x.i).sort((p,q)=>q.sum-p.sum);if(g.length&&o.sum)out.push(`<b>${a0<0?esc(ACC[g[0].k]):esc(IFS[g[0].k])}</b> accounts for ${Math.round(100*g[0].sum/o.sum)}% of the week’s variance (Σ|variance| ${full(g[0].sum)}).`);
  const rp=grp(o.items,x=>x.a+'|'+x.i).filter(z=>z.n>=2).sort((p,q)=>q.n-p.n);if(rp.length)out.push(`<b>${rp.length}</b> feed${rp.length>1?'s are':' is'} not reconciled on 2+ days this week; longest: ${a0<0?esc(ACC[rp[0].x.a])+' · ':''}${esc(IFS[rp[0].x.i])} (${rp[0].n} days).`);
  const R=perDay(d1,d2,a0);let bi=0;R.forEach((x,i)=>{if(d1+i<=DEF&&x.att+x.miss>R[bi].att+R[bi].miss)bi=i;});if(R.length)out.push(`Worst day: <b>${DOW[(d1+bi)%7]} ${md(d1+bi)}</b> with ${R[bi].att} not fully reconciled and ${R[bi].miss} not reported.`);
  vs('previous week');}
 else{vs('previous month');
  const wk={};o.items.forEach(x=>{const k=Math.floor(x.d/7)*7;wk[k]=(wk[k]||0)+Math.abs(x.v);});const wl=Object.entries(wk).sort((p,q)=>q[1]-p[1]);if(wl.length)out.push(`Heaviest week: <b>week of ${md(+wl[0][0])}</b> (Σ|variance| ${full(wl[0][1])}).`);
  const g=grp(o.items,x=>a0<0?x.a:x.i).sort((p,q)=>q.n-p.n);if(g.length)out.push(`Most days not reconciled: <b>${a0<0?esc(ACC[g[0].k]):esc(IFS[g[0].k])}</b> (${g[0].n} feed-days).`);
  if(o.att)out.push(`Ticket coverage: <b>${Math.round(100*o.attT/o.att)}%</b> of variances this month are tracked via a ticket${pr&&pr.att?' (previous month '+Math.round(100*pr.attT/pr.att)+'%)':''}.`);}
 if(!out.length)return '';
 return `<div class="side"><h2>Highlights</h2>${out.slice(0,4).map((x,n)=>`<div class="fi"><i>${n+1}</i><div>${x}</div></div>`).join('')}</div>`;}
function glance(o){const rows=o.byA.map((x,a)=>({a,x})).filter(z=>z.x.g+z.x.att+z.x.miss+z.x.scope).sort((p,q)=>q.x.sum-p.x.sum||q.x.miss-p.x.miss);
 return `<div class="dark"><h2>Customers at a glance</h2><small>Sorted by largest variance. Select a customer to drill into daily, weekly, monthly and historical insights.</small><div style="margin-top:6px"><table class="cmp"><tr><th>Customer</th><th>In scope</th><th>Fully reconciled</th><th>Not fully reconciled</th><th>Tracked via ticket</th><th>Not tracked via ticket</th><th>Not reported</th><th>Rate</th></tr>`+rows.map(({a,x})=>{const rt=(x.g+x.att)?Math.round(100*x.g/(x.g+x.att)):null,col=x.attN?'#e3a977':x.att?'#e8c48a':x.miss?'#dc9a9a':'#8cc0a8';return `<tr class="cl" data-a="${a}"><td><span class="dot" style="background:${col}"></span><b>${esc(ACC[a])}</b></td><td>${x.scope}</td><td style="color:#9fd8bb">${x.g}</td><td style="color:#f2c98f"><b>${x.att}</b></td><td style="color:#9dbcf2">${x.attT}</td><td style="color:#f1a982">${x.attN}</td><td style="color:#f0a3a3">${x.miss}</td><td>${rt==null?'–':rt+'%'}</td></tr>`}).join('')+'</table></div></div>';}
function varMap(o){const it=o.items.slice(0,12);if(!it.length)return '<div class="card"><h2>Where to focus first</h2><div class="note">No variances in this period.</div></div>';const mx=Math.abs(it[0].v)||1;
 return `<div class="card"><h2>Where to focus first — highest variances</h2><div class="hint">Bigger and darker = larger variance. Top ${it.length} of ${o.items.length}.</div><div class="vm">`+it.map(x=>{const r=Math.abs(x.v)/mx;return `<div class="vt" style="--w:${Math.round(150+Math.sqrt(r)*340)}px;--al:${(.14+.5*r).toFixed(2)}"><div class="t">${S.a<0?esc(ACC[x.a])+' · ':''}${esc(IFS[x.i])}</div><b>${full(x.v)}</b><small>${DOW[x.d%7]} ${md(x.d)} · ${x.t?'tracked via ticket':'not tracked via ticket'}</small></div>`}).join('')+'</div></div>';}
const notRep=(o,acc)=>`<div class="card"><h2>Schedule deviations (${o.miss}) — expected but not reported</h2><div class="hint">Feeds the Recon Report Schedule expects that day, with no row in the tracker.</div>`+(o.missL.length?`<div class="sc"><table class="lt"><tr>${acc?'<th>Customer</th>':''}<th>Interface</th><th>Expected on</th></tr>`+o.missL.slice().sort((p,q)=>q.d-p.d||p.a-q.a).slice(0,200).map(x=>`<tr>${acc?`<td><b>${esc(ACC[x.a])}</b></td>`:''}<td>${esc(IFS[x.i])}</td><td><span class="chip">${DOW[x.d%7]} ${md(x.d)}</span></td></tr>`).join('')+'</table></div>':'<div class="note">Everything scheduled has been reported.</div>')+'<div class="note">Schedule = latest “Recon Report Schedule” tab applied to all weeks [TO CONFIRM]. Rows with an explanatory comment (N/A, no run, holiday, on hold…) are not counted.</div></div>';
const anomT=(o,acc)=>`<div class="card"><h2>Volume anomalies — received vs usual</h2><div class="hint">Received more than 3× or less than ⅓ of the usual level (median of the previous ~20 reported days).</div>`+(o.anom.length?`<div class="sc"><table class="lt"><tr>${acc?'<th>Customer</th>':''}<th>Interface</th><th>Day</th><th style="text-align:right">Received</th><th style="text-align:right">Usual</th><th style="text-align:right">Change</th></tr>`+o.anom.slice(0,15).map(x=>`<tr>${acc?`<td><b>${esc(ACC[x.a])}</b></td>`:''}<td>${esc(IFS[x.i])}</td><td><span class="chip">${DOW[x.d%7]} ${md(x.d)}</span></td><td style="text-align:right"><b>${full(x.v)}</b></td><td style="text-align:right">${full(x.m)}</td><td style="text-align:right;color:var(--r)"><b>${x.v>x.m?'+':''}${Math.round((x.v-x.m)/x.m*100)}%</b></td></tr>`).join('')+'</table></div>':'<div class="note">No volume anomalies.</div>')+'<div class="note">Median ≥ 200; the latest day is only checked for spikes as it may still be filling. Thresholds [TO CONFIRM].</div></div>';
function ifaceSummary(o,d1,d2){const rows=IFS.map((n,i)=>({n,b:o.byI[i]})).filter(x=>x.b.scope||x.b.g+x.b.att+x.b.miss||x.b.rec);const T=rows.reduce((s,{b})=>{for(const k in b)s[k]=(s[k]||0)+b[k];return s;},{});
 const row=(n,b,bold)=>`<tr class="${bold?'tot':b.attN?'o':b.miss?'r':''}"><td>${bold?'<b>'+esc(n)+'</b>':esc(n)}</td><td>${b.scope||''}</td><td>${full(b.rec)}</td><td>${full(b.ld)}</td><td><b>${full(b.err)}</b></td><td>${full(b.pend)}</td><td>${b.sum?'<b>'+full(b.sum)+'</b>':''}</td><td>${b.g||''}</td><td>${b.attT||''}</td><td>${b.attN||''}</td><td>${b.miss||''}</td></tr>`;
 return `<div class="card" style="margin-top:0"><h2>Interfaces — ${d1===d2?DOW[d1%7]+' '+md(d1):md(d1)+' – '+md(d2)}</h2><div class="sc"><table><tr><th>Interface</th><th>In scope</th><th>Received</th><th>Loaded</th><th>Error</th><th>Pending / WB</th><th>Variance Σ|v|</th><th>Fully reconciled</th><th>Tracked via ticket</th><th>Not tracked</th><th>Not reported</th></tr>`+rows.sort((p,q)=>q.b.sum-p.b.sum||q.b.miss-p.b.miss||q.b.scope-p.b.scope).map(x=>row(x.n,x.b)).join('')+row('Total',T,true)+'</table></div><div class="note">Received / Loaded / Error / Pending are summed from each interface’s tracker columns (Claims IB: received = paper + EDI, error = Edifecs + I-Plus; extracts: received = triggered, loaded = extracted).</div></div>';}
const rcol=p=>p==null?'#f3f5f8':p>=95?'#d3eadd':p>=85?'#e6efd9':p>=70?'#f8ebc9':'#f2d4cf';
/* ---------- historical ---------- */
const MN=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const compact=n=>n==null?'–':Math.abs(n)>=1e6?(n/1e6).toFixed(2)+'M':Math.abs(n)>=1e4?Math.round(n/1e3)+'k':Math.abs(n)>=1e3?(n/1e3).toFixed(1)+'k':String(Math.round(n));
function spark(vals,partial){const w=330,h=92,m={l:4,r:4,t:8,b:14},n=vals.length,mx=Math.max(1,...vals),X=i=>m.l+i*(w-m.l-m.r)/(n-1),Y=v=>m.t+(h-m.t-m.b)*(1-v/mx);
 const pts=vals.map((v,i)=>[X(i),Y(v)]),cut=partial?n-1:n,d=pts.slice(0,cut).map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join('');
 let s=`<svg viewBox="0 0 ${w} ${h}" class="ch"><line x1="${m.l}" x2="${w-m.r}" y1="${h-m.b}" y2="${h-m.b}" class="gl"/>`;
 if(cut>1)s+=`<path d="${d}L${pts[cut-1][0].toFixed(1)} ${h-m.b}L${pts[0][0].toFixed(1)} ${h-m.b}Z" fill="rgba(31,181,173,.13)"/><path d="${d}" fill="none" stroke="#1fb5ad" stroke-width="2" stroke-linejoin="round"/>`;
 if(partial&&n>1)s+=`<path d="M${pts[n-2][0].toFixed(1)} ${pts[n-2][1].toFixed(1)}L${pts[n-1][0].toFixed(1)} ${pts[n-1][1].toFixed(1)}" stroke="#1fb5ad" stroke-width="2" stroke-dasharray="3 3" fill="none"/>`;
 s+=vals.map((v,i)=>`<circle cx="${pts[i][0].toFixed(1)}" cy="${pts[i][1].toFixed(1)}" r="${i===cut-1?3.4:2}" fill="${i===cut-1?'#161d3f':'#1fb5ad'}"><title>Week of ${md(i*7)}${partial&&i===n-1?' (partial)':''}: ${full(v)}</title></circle>`).join('');
 return s+`<text x="${m.l}" y="${h-2}" class="ax">${md(0)}</text><text x="${w-m.r}" y="${h-2}" class="ax" text-anchor="end">${md((n-1)*7)}</text></svg>`;}
function histAll(a0){const NW=Math.floor(LAST/7)+1,partial=LAST%7!==6,cw=partial?NW-2:NW-1;let h='';
 /* 2 customer / interface scorecard */
 const W=perWeek(a0),lab=W.map((_,w)=>md(w*7));
 const rate=x=>(x.g+x.att)?Math.round(100*x.g/(x.g+x.att)):null,tk=x=>x.att?Math.round(100*x.attT/x.att):null,ad=x=>{const t=x.g+x.att+x.miss;return t?Math.round(100*(1-x.miss/t)):null;};
 const rows=a0<0?ACC.map((n,a)=>({n,get:(d1,d2)=>scan(d1,d2,a,1)})):IFS.map((n,i)=>({n,get:(d1,d2)=>{const f=S.f;S.f=i;const r=scan(d1,d2,a0,1);S.f=f;return r;}}));
 const body=rows.map(r=>{const ws=W.map((_,w)=>r.get(w*7,Math.min(LAST,w*7+6)));const rs=ws.map(rate);if(!rs.some(v=>v!=null))return '';const rec=rs.slice(-4).filter(v=>v!=null),old=rs.slice(-8,-4).filter(v=>v!=null);const dl=rec.length&&old.length?Math.round(sum(rec)/rec.length-sum(old)/old.length):null;
  return `<tr><td class="l">${esc(r.n)}</td>`+rs.map((v,k)=>`<td style="background:${rcol(v)}" title="${lab[k]}: ${v==null?'no data':v+'%'}">${v==null?'·':v}</td>`).join('')+`<td>${dl==null?'–':`<b style="color:${dl>0?'var(--g)':dl<0?'var(--r)':'var(--mut)'}">${dl>0?'▲ +':dl<0?'▼ ':'● '}${dl}</b>`}</td></tr>`}).join('');
 h+=`<div class="card"><h2>${a0<0?'Customer':'Interface'} scorecard — reconciled rate by week</h2><div class="hint">Who is improving and who is slipping: the last column compares the latest 4 weeks with the 4 before (points).</div><div class="sc"><table class="hm"><tr><th></th>${lab.map(l=>`<th>${l}</th>`).join('')}<th>Last 4 vs prior 4</th></tr>${body}</table></div><div class="lg"><span><i style="background:#d3eadd"></i>≥95%</span><span><i style="background:#e6efd9"></i>85–94</span><span><i style="background:#f8ebc9"></i>70–84</span><span><i style="background:#f2d4cf"></i>&lt;70</span></div></div>`;
 /* 3 recurring issues */
 const pr={};for(let d=0;d<=LAST;d++)for(let a=0;a<NA;a++){if(a0>=0&&a!==a0)continue;for(let i=0;i<NI;i++){if(S.f>=0&&i!==S.f)continue;const c=cell(a,i,d);if(!c||c.s!=='att')continue;const k=a+'|'+i,p=pr[k]||(pr[k]={a,i,wk:new Set(),n:0,sum:0,t:0,last:0,str:0,cur:0,prev:-9});p.wk.add(Math.floor(d/7));p.n++;p.sum+=c.v;if(c.t)p.t++;p.last=d;p.cur=(d-p.prev<=1)?p.cur+1:1;p.str=Math.max(p.str,p.cur);p.prev=d;}}
 const rec=Object.values(pr).sort((p,q)=>q.wk.size-p.wk.size||q.sum-p.sum).slice(0,12);
 h+=`<div class="card"><h2>Recurring issues</h2><div class="hint">Feeds that keep coming back: weeks affected, longest run of consecutive days, and how many were ticketed.</div>`+(rec.length?`<div class="sc"><table class="lt"><tr>${a0<0?'<th>Customer</th>':''}<th>Interface</th><th style="text-align:right">Weeks affected</th><th style="text-align:right">Days</th><th style="text-align:right">Longest run</th><th style="text-align:right">Σ|variance|</th><th style="text-align:right">Ticketed</th><th>Last seen</th></tr>`+rec.map(p=>`<tr>${a0<0?`<td><b>${esc(ACC[p.a])}</b></td>`:''}<td>${esc(IFS[p.i])}</td><td style="text-align:right"><b>${p.wk.size}</b> / ${W.length}</td><td style="text-align:right">${p.n}</td><td style="text-align:right">${p.str} d</td><td style="text-align:right">${full(p.sum)}</td><td style="text-align:right;color:${p.t?'var(--b)':'var(--o)'}">${Math.round(100*p.t/p.n)}%</td><td><span class="chip">${md(p.last)}</span></td></tr>`).join('')+'</table></div>':'<div class="note">No recurring issues.</div>')+'</div>';
 /* 4 quality over time (lower priority) */
 h+=`<div class="card"><h2>Reconciliation quality over time</h2><div class="hint">Weekly reconciled rate, ticket coverage of variances and schedule adherence.</div>${lines(lab,[{name:'Reconciled rate',color:CG,data:W.map(rate)},{name:'Ticket coverage of variances',color:CB,data:W.map(tk)},{name:'Schedule adherence',color:CA,data:W.map(ad)}],190)}<div class="note">Last point is a partial week; early weeks track fewer feeds.</div></div>`;
 return h;}
/* ---------- navigation state ---------- */
const WKS=[];for(let w=0;w*7<=LAST;w++)WKS.push(w*7);
const MOS=[];{const tl=new Date(T0+LAST*MS);let y=new Date(T0).getUTCFullYear(),m=new Date(T0).getUTCMonth();while(y<tl.getUTCFullYear()||(y===tl.getUTCFullYear()&&m<=tl.getUTCMonth())){MOS.push([y,m]);m++;if(m>11){m=0;y++}}}
Object.assign(S,{d:DEF,wk:Math.floor(DEF/7)*7,mo:MOS.length-1});
function sync(){if(S.p==='d'){S.from=S.to=S.d}else if(S.p==='w'){S.from=S.wk;S.to=Math.min(LAST,S.wk+6)}else if(S.p==='m'){const[y,m]=MOS[S.mo];S.from=Math.max(0,Math.round((Date.UTC(y,m,1)-T0)/MS));S.to=Math.min(LAST,Math.round((Date.UTC(y,m+1,0)-T0)/MS))}else{S.from=0;S.to=LAST}}
/* ---------- render ---------- */
function render(){sync();const d1=S.from,d2=S.to,a0=S.a,V=S.p,o=V==='h'?null:scan(d1,d2,a0);
 const per=V==='d'?`${DOW[d1%7]} ${md(d1)} run`:V==='w'?`Week of ${md(d1)} – ${md(d1+6)}${d2<d1+6?' · updated through '+DOW[d2%7]+' '+md(d2):''}`:V==='m'?`${MN[MOS[S.mo][1]]} ${MOS[S.mo][0]}`:`Full archive · ${md(0)} → ${md(LAST)} · ${WKS.length} weeks`;
 $('#kick').textContent='TECHNOLOGY OPERATIONS · '+(a0<0?'PORTFOLIO':'CUSTOMER')+' · RECONCILIATION REPORT';
 $('#ttl').textContent=(a0<0?'All customers · Portfolio':ACC[a0]+' · Reconciliation')+(S.f>=0?' · '+IFS[S.f]:'');
 $('#sub').textContent=per+' · Data loaded through '+md(LAST)+' · Archive from '+md(0);
 document.querySelectorAll('#seg button').forEach(b=>b.classList.toggle('on',b.dataset.p===V));$('#back').style.display=a0<0?'none':'';
 let nav='';
 if(V==='d')nav=`<button id="pv">◀</button><input type="date" id="dt" min="${D.start}" max="${D.asOf}" value="${iso(S.d)}"><button id="nx">▶</button>`;
 else if(V==='w')nav=`<button id="pv">◀</button><select id="wk">${WKS.map(w=>`<option value="${w}" ${w===S.wk?'selected':''}>Week of ${md(w)} – ${md(w+6)}</option>`).join('')}</select><button id="nx">▶</button>`;
 else if(V==='m')nav=`<button id="pv">◀</button><select id="mo">${MOS.map((x,k)=>`<option value="${k}" ${k===S.mo?'selected':''}>${MN[x[1]]} ${x[0]}</option>`).join('')}</select><button id="nx">▶</button>`;
 else nav=`<span>Trends across the full archive</span>`;
 $('#bar').innerHTML=nav+`<span style="margin-left:auto;display:flex;gap:8px"><select id="fi"><option value="-1">All interfaces</option>${IFS.map((n,i)=>`<option value="${i}" ${i===S.f?'selected':''}>${esc(n)}</option>`).join('')}</select><select id="ac"><option value="-1">All customers</option>${ACC.map((a,i)=>`<option value="${i}" ${i===a0?'selected':''}>${esc(a)}</option>`).join('')}</select></span>`;
 let h='';
 if(V==='h')h=`<h2 class="sec">History · ${a0<0?'all customers':esc(ACC[a0])}</h2>`+histAll(a0);
 else{h=`<h2 class="sec">${V==='d'?'On this day':V==='w'?'This week':'This month'}, ${a0<0?'all customers':esc(ACC[a0])}</h2>`+tiles(o)+`<div class="split">${a0<0?glance(o):ifaceSummary(o,d1,d2)}${findings(d1,d2,a0,o)}</div>`;
  let tr='';if(V==='d'){const s=Math.max(0,d1-13);tr+=`<div class="card"><h2>Reconciled rate — last 14 days</h2>${trend(s,d1,a0)}</div>`;}
  else{const s=V==='w'?Math.max(0,d1-7):d1;tr+=`<div class="card"><h2>Reconciled rate — day by day${V==='w'?', this week vs previous':''}</h2>${trend(s,d2,a0,V==='w'?d1:null)}</div>`;}
  h+=varMap(o)+anomT(o,a0<0)+notRep(o,a0<0)+tr;}
 $('#body').innerHTML=h;
 $('#body').querySelectorAll('tr.cl').forEach(r=>r.onclick=()=>{S.a=+r.dataset.a;render();window.scrollTo(0,0)});
 const step=n=>{if(V==='d')S.d=Math.max(0,Math.min(LAST,S.d+n));else if(V==='w')S.wk=Math.max(0,Math.min(WKS[WKS.length-1],S.wk+7*n));else S.mo=Math.max(0,Math.min(MOS.length-1,S.mo+n));render();};
 if($('#pv')){$('#pv').onclick=()=>step(-1);$('#nx').onclick=()=>step(1);}
 if($('#dt'))$('#dt').oninput=e=>{if(e.target.value){S.d=Math.max(0,Math.min(LAST,idxOf(e.target.value)));render();}};
 if($('#wk'))$('#wk').oninput=e=>{S.wk=+e.target.value;render();};
 if($('#mo'))$('#mo').oninput=e=>{S.mo=+e.target.value;render();};
 $('#ac').oninput=e=>{S.a=+e.target.value;render()};$('#fi').oninput=e=>{S.f=+e.target.value;render()};}
document.querySelectorAll('#seg button').forEach(b=>b.onclick=()=>{S.p=b.dataset.p;render()});
$('#back').onclick=()=>{S.a=-1;render()};
if(D.home){$('#home').href=D.home;$('#home').style.display='';}
if(D.gen){$('#pv0').textContent='Refreshed automatically from the recon tracker · '+D.gen.slice(0,16).replace('T',' ')+' UTC';$('#pv0').style.display='';}
S.p='w';render();
}

/* ===== live loader ===== */
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
</script></body></html>
