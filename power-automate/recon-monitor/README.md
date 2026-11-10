# Reconciliation Monitor on Power Automate

Rebuilds the Reconciliation Monitor dashboard automatically. A scheduled cloud flow reads the recon
tracker workbook, applies the dashboard's rules, republishes the page to SharePoint and mails a daily digest.

```
Recurrence (daily)
  ├─ Get recon-config.json + page template        SharePoint
  ├─ Run script  ReconMonitor                     Excel Online (Business)  ← all dashboard logic
  │     returns dataJson · digestJson · subject · emailHtml · tiles · warnings
  ├─ Compose page  = template with dataJson injected
  ├─ Publish page + recon-digest.json             SharePoint (overwrite)
  └─ Send daily digest                            Outlook
```

The rules live in one Office Script because a flow cannot loop over ~50,000 customer × interface × day
cells in reasonable time; the script does it in well under a second and is called with a single action.

| File | Purpose |
|---|---|
| `office-scripts/ReconMonitor.ts` | The logic: read tracker, align names, classify feeds, KPIs, highlights, anomalies, history, e-mail body |
| `dashboard/recon-monitor.template.html` | The dashboard page without data (`"__RECON_DATA__"` placeholder) |
| `config/recon-config.sample.json` | Settings: sheet/column mapping, customer name alignment, thresholds |
| `flow/definition.json` | Reference definition of the flow (trigger, actions, expressions) |
| `tests/` | Synthetic tests and a parity check against a hand-built dashboard |

## What the script decides

Each **feed** is one customer × interface × day.

| Rule | Definition |
|---|---|
| In scope | The schedule tab marks the interface type as running on that weekday for the customer |
| Fully reconciled | A tracker row exists and \|variance\| ≤ 0.5 (not counted when the schedule explicitly says "not scheduled") |
| Not fully reconciled | A tracker row exists and \|variance\| > 0.5 — split into tracked / not tracked via ticket |
| Scheduled but not reported | In scope, no tracker row, and no row with an explanatory comment (N/A, no run, no file, holiday, on hold) |
| Reconciled rate | fully reconciled ÷ (fully + not fully reconciled), rounded |
| Σ\|variance\| | Sum of absolute variances of the not-fully-reconciled feeds |
| Volume anomaly | Received > 3× or < ⅓ of the median of the previous ≤ 20 reported days (last 45 days, ≥ 6 days of history, median ≥ 200); the latest day is only checked for spikes |
| Name alignment | Tracker names are mapped through `accountAliases`; names that map to the same customer are merged (counts and \|variance\| summed, ticket kept if any row has one) |
| Schedule type | `Provider IB – *` interfaces follow the `Provider IB` schedule, `Encounter Rejections` follows `Encounter Submission` |
| Default day | Latest day with at least `minRowsForDefaultDay` rows, so a half-filled "today" does not become the headline |
| Highlights | Day: biggest variance, variances needing a ticket, feeds not reported, volume check, vs previous day. Week: top contributor, feeds failing 2+ days, worst day, vs previous week. Month: vs previous month, heaviest week, most days not reconciled, ticket coverage |
| History | Weekly reconciled rate per customer with last-4 vs prior-4 weeks, recurring issues (weeks affected, longest run, % ticketed), weekly quality (rate, ticket coverage, schedule adherence) |

## Set-up

### 1. Files in SharePoint

Create a folder (for example `SiteAssets/ReconMonitor`) and upload

- `recon-monitor.template.html`
- `recon-config.json` — a copy of `config/recon-config.sample.json` with your values

### 2. Office Script

Open the recon tracker in Excel for the web → **Automate → New script**, paste `ReconMonitor.ts`,
rename it **ReconMonitor**, save.

### 3. Match the configuration to the tracker

The script expects one sheet per interface with a header row containing a date and an account column,
plus a schedule tab. Run the script from Excel once with `mode` = `inspect` (leave the other
parameters empty or paste your config): the `warnings` output lists, per interface, the sheet, header
row and the columns it matched. Adjust `recon-config.json` until every interface reports the right columns.

| Setting | Meaning |
|---|---|
| `startDate` | First day of the archive (snapped back to a Monday) |
| `interfaces[]` | `name` (as shown on the dashboard), optional `sheet` (default = name), `scheduleType`, `columns` overrides, `filterColumn` / `filterEquals` when several interfaces share one sheet |
| `columns` | Header patterns per role (`date`, `account`, `received`, `loaded`, `error`, `pending`, `variance`, `ticket`, `comment`). Case-insensitive regular expressions. Count roles add up every matching column, e.g. Claims IB received = `paper` + `edi` |
| `accountAliases` | Tracker name → dashboard name. Several tracker names may share one dashboard name |
| `ignoreAccounts` | Tracker names to leave out |
| `unmappedAccounts` | `keep` (default: show under the tracker name, with a warning) or `drop` |
| `scheduleSheet` | Schedule tab. Layouts: Account + one column per interface type; Account, Interface, Mon…Sun; or Account, Interface, Schedule. Cells may read `Daily`, `Mon-Fri`, `Tue, Thu`, `N/A`, `Y`, or `1111100` |
| `explainedCommentPattern` | Comment text that explains a missing feed |
| `varianceWhenMissing` | `none`, or `received-loaded-error-pending` for sheets without a variance column |
| `dateOrder` | `MDY` or `DMY` for dates stored as text |
| `minRowsForDefaultDay`, `anomalyMinMedian`, `anomalyFactor` | Thresholds described above |
| `homeUrl`, `dashboardUrl` | "TechOps home" button on the page; link in the e-mail |

### 4. Build the flow

Create a **Scheduled cloud flow** and add these steps (`flow/definition.json` has the same content as code).

| # | Action | Values |
|---|---|---|
| 1 | **Recurrence** | Daily, your time zone, after the tracker is normally complete |
| 2 | **Compose** → rename `Settings` | Object with `siteUrl`, `configPath`, `templatePath`, `outputFolder` (server-relative, e.g. `/sites/TechOps/SiteAssets/ReconMonitor`), `pageFile`, `digestFile`, `timeZone`, `mailTo` |
| 3 | SharePoint **Get file content using path** → `Get config` | Site = `siteUrl`, path = `configPath`, *Infer content type* = **No** |
| 4 | SharePoint **Get file content using path** → `Get template` | As above with `templatePath`, *Infer content type* = **No** |
| 5 | Excel Online (Business) **Run script** → `Run script` | Pick the tracker workbook and script `ReconMonitor`. `configJson` = `base64ToString(body('Get_config')?['$content'])` · `asOf` = `convertTimeZone(utcNow(), 'UTC', outputs('Settings')?['timeZone'], 'yyyy-MM-dd')` · `mode` = `build` |
| 6 | **Condition** `result/ok` is equal to `false` → **Terminate** (Failed) | Keeps the last good page when nothing could be read |
| 7 | **Compose** → `Compose page` | `replace(base64ToString(body('Get_template')?['$content']), '"__RECON_DATA__"', body('Run_script')?['result']?['dataJson'])` |
| 8 | SharePoint **Send an HTTP request to SharePoint** → `Publish page` | POST `_api/web/GetFolderByServerRelativeUrl('<outputFolder>')/Files/add(url='<pageFile>',overwrite=true)`, body = output of `Compose page` |
| 9 | Same action → `Publish digest` | `url='<digestFile>'`, body = `result/digestJson` |
| 10 | Outlook **Send an email (V2)** | To = `mailTo`, Subject = `result/subject`, Body (code view) = `result/emailHtml`, Importance = `if(body('Run_script')?['result']?['needsAttention'], 'High', 'Normal')` |

Publish the page under the same name and location the Batch Monitor page links to (or embeds), so the
existing navigation keeps working. For Teams, add *Post message in a chat or channel* with
`result/subject` and the `daily` tile values.

### Script outputs available to the flow

`ok`, `asOf`, `defaultDay`, `generated`, `dataJson`, `digestJson`, `subject`, `emailHtml`,
`needsAttention` (variances without a ticket or feeds not reported on the default day), `warnings[]`,
`dailyLabel` / `weeklyLabel` / `monthlyLabel`, and for each of `daily`, `weekly`, `monthly`:
`inScope`, `fullyReconciled`, `notFullyReconciled`, `trackedViaTicket`, `notTrackedViaTicket`,
`notReported`, `varianceSum`, `reconciledRate` (−1 when no feed was reported).

`digestJson` adds, per period: highlights, customers at a glance, interface summary, top variances,
not-reported list and volume anomalies; plus the `history` block.

## Tests

```bash
node tests/run-tests.mjs                                   # synthetic data
node tests/parity.mjs "<existing Reconciliation-Monitor.html>"   # compare with a hand-built dashboard
tsc --noEmit --strict --target ES2019 --lib ES2019 office-scripts/ReconMonitor.ts tests/excelscript.d.ts
```

`parity.mjs` rebuilds a tracker from the data embedded in an existing dashboard file, runs the script on
it, and compares every period (each day, week and month, portfolio and per customer) with the numbers,
lists and highlight sentences the original page computes — both for the script's own engine and for
the template rendered with the script's data.

## Notes

- Run script limits: 120 s per call, 5 MB of output, 1,600 calls per user per day. The current archive
  (≈ 19 weeks) produces ≈ 0.4 MB and runs in a fraction of a second.
- The page template differs from the hand-built page in four places: data comes from the placeholder,
  the name-alignment block is gone (the script does it), the preview banner became a "refreshed" stamp,
  and the per-feed key now allows archives longer than 400 days.
- The schedule tab is applied to every week of the archive, as on the original dashboard.
