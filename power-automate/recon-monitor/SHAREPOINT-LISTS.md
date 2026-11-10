# Reconciliation Monitor as SharePoint lists and pages

The dashboard lives on ordinary modern SharePoint pages. A Power Automate flow refreshes six SharePoint lists
every morning; the pages show those lists with **List** web parts. Selecting a customer filters the other web
parts on the page to that customer — the drill-down. No custom script, no admin setting, nothing to download.

```
Power Automate (daily)                                  SharePoint site
  Run script on each archived week  (extract)            Recon Summary          tiles + highlights per period and customer
  Run script on the current week    (build)    ──▶       Recon Interfaces       interface table per period and customer
  Send the $batch requests it returns                    Recon Issues           variances · not reported · volume anomalies
  E-mail the daily summary with a link                   Recon Daily Trend      reconciled rate per day, last 28 days
                                                         Recon Weekly Scorecard reconciled rate by week, last 12 weeks
                                                         Recon Recurring Issues feeds that keep coming back
                                                         4 pages: Latest day · This week · This month · History
```

The numbers come from the same Office Script as before. On the original data every list row was checked
against the original dashboard (`tests/sharepoint-lists.mjs`: 602 rows and figures identical).

**What is different from the HTML dashboard**

- It looks like SharePoint: KPI card and status colours are reproduced with SharePoint's JSON formatting,
  tables are SharePoint list views, the trend is a bar per day instead of a line chart.
- Each page shows the current period (latest complete day, this week, this month); there is no ◀ ▶ to step
  back. Earlier weeks are on the History page (scorecard, recurring issues, trend).
- Up to 250 variances, 300 not-reported feeds and 50 anomalies per period are listed (the tiles count all).

## What you need

- A work account with Power Automate, Excel for the web with the **Automate** tab, and **Edit** (member)
  permission on the SharePoint site — that is enough to create lists and pages. No admin setting.
- The files: `office-scripts/ReconMonitor.ts`, `recon-config.json` (with `siteUrl` = the site that will hold
  the lists and pages), `flow/setup-lists.json`, `flow/daily-refresh.json` (references for the designer).

## 1. Script and configuration

1. Upload `recon-config.json` to the site, e.g. `Shared Documents/Recon/Monitor/recon-config.json`.
   Check `siteUrl` is the site's address, e.g. `https://contoso.sharepoint.com/sites/TechOps`.
2. Open the current Reconciliation Report in Excel for the web → **Automate → New script** → paste
   `ReconMonitor.ts` → name it **ReconMonitor** → Save.

## Quickest: import the two flows

`flow/packages/Recon-Setup-Lists.zip` and `flow/packages/Recon-Daily-Refresh.zip` contain the flows of
steps 2 and 3 below, ready to import (rebuild them with `python3 flow/make-packages.py` after changing the
definitions).

For each package: **make.powerautomate.com → My flows → Import → Import Package (Legacy)** → upload the zip →
under *Related resources* click **Select during import** for each connection (SharePoint, Excel Online
(Business), Office 365 Outlook) and pick yours → **Import**. Then open the flow → **Edit** and set what is
specific to your tenant:

| Flow | Step | Set |
|---|---|---|
| both | `Settings` | `siteUrl`, `configPath` (and in the daily flow `archiveFolder`, `pageUrl`, `timeZone`, `mailTo`) |
| setup | `Run_script_setup` | **Location** (your site), **Document Library**, **File** = the current Reconciliation Report, **Script** = `ReconMonitor` |
| daily | `Get_archived_weeks` | **Library Name** = the library holding the archive folder |
| daily | `Run_script_on_archived_week` (inside the first loop) | **Location**, **Document Library** (of the archive), **Script** = `ReconMonitor`; leave **File** as the expression `items('For_each_archived_week')?['{Identifier}']` |
| daily | `Run_script` | **Location**, **Document Library**, **File** = the current Reconciliation Report, **Script** = `ReconMonitor` |

After picking the script, check its input boxes still hold the expressions (re-selecting a script can clear
them); re-enter any that are empty from the tables in steps 2 and 3. Save. The setup flow is run by hand; the
daily flow runs on its schedule (07:15 Eastern by default, change it in the trigger).

If the import itself is refused, build the flows by hand from the tables below — the result is the same.

## 2. Setup flow (run once)

Create an **Instant cloud flow** (*Manually trigger a flow*) with these steps; names matter for the expressions.

| # | Step name | Action | Settings |
|---|---|---|---|
| 1 | `Settings` | **Compose** | `{ "siteUrl": "https://…/sites/TechOps", "configPath": "/Shared Documents/Recon/Monitor/recon-config.json" }` |
| 2 | `Get_config` | SharePoint · **Get file content using path** | Site = your site, File path = `outputs('Settings')?['configPath']`, *Infer content type* = **No** |
| 3 | `Initialize_errors` | **Initialize variable** | `errors`, Array, `[]` |
| 4 | `Run_script_setup` | Excel Online (Business) · **Run script** | File = the current report (only used to run the script), Script = `ReconMonitor`, `configJson` = `base64ToString(body('Get_config')?['$content'])`, `mode` = `setup` |
| 5 | `For_each_batch` | **Apply to each** over `body('Run_script_setup')?['result']?['spBatches']` | Settings → Concurrency control **On**, degree **1** (order matters) |
| 5a | `Send_batch` | SharePoint · **Send an HTTP request to SharePoint** | Method **POST**, Uri `_api/$batch`, Headers: `Content-Type` = `multipart/mixed; boundary=@{items('For_each_batch')?['boundary']}`, `Accept` = `application/json;odata=verbose`; Body = `items('For_each_batch')?['body']` |
| 5b | `Batch_response` | **Compose** | `if(empty(body('Send_batch')?['$content']), string(body('Send_batch')), base64ToString(body('Send_batch')?['$content']))` |
| 5c | `If_any_request_failed` | **Condition**: `outputs('Batch_response')` contains `HTTP/1.1 4` **or** contains `HTTP/1.1 5` | Yes: **Append to array variable** `errors` = `@{items('For_each_batch')?['label']}: @{take(outputs('Batch_response'), 800)}` |
| 6 | `If_setup_failed` | **Condition** `length(variables('errors'))` is not equal to `0` | Yes: **Terminate**, Failed, message `join(variables('errors'), ' || ')` |

Run it once. In **Site contents** you should see the six *Recon …* lists. A SharePoint `$batch` answers
"200 OK" even when a request inside it fails, which is why steps 5b–5c read the response; if the run fails,
the message says which request and why. To start over, delete the six lists and run it again (running it on
existing lists fails with "already exists" and changes nothing).

## 3. Daily flow

Create a **Scheduled cloud flow** (daily, after the report is normally filled in) following
`flow/daily-refresh.json`:

| # | Step name | Action | Settings |
|---|---|---|---|
| 1 | — | **Recurrence** | Daily, your time zone |
| 2 | `Settings` | **Compose** | `siteUrl`, `configPath`, `archiveFolder`, `pageUrl` (the This week page, for the e-mail link), `timeZone` (Windows name, e.g. `Eastern Standard Time`), `mailTo`, `lists` = `["Recon Summary","Recon Interfaces","Recon Issues","Recon Daily Trend","Recon Weekly Scorecard","Recon Recurring Issues"]` |
| 3 | `Get_config` | As in the setup flow | |
| 4 | `Get_archived_weeks` | SharePoint · **Get files (properties only)** | Library = archive library, *Limit entries to folder* = archive folder, Filter `FSObjType eq 0`, Order By `Modified asc` |
| 5 | `Only_workbooks` | **Filter array** | From `body('Get_archived_weeks')?['value']`, advanced: `@and(or(endsWith(toLower(item()?['{FilenameWithExtension}']), '.xlsx'), endsWith(toLower(item()?['{FilenameWithExtension}']), '.xlsm')), not(startsWith(item()?['{FilenameWithExtension}'], '~')))` |
| 6 | `Initialize_archive`, `Initialize_lists`, `Initialize_errors` | **Initialize variable** ×3 | each Array, `[]` |
| 7 | `For_each_archived_week` | **Apply to each** `body('Only_workbooks')`, concurrency **1** | inside: **Run script** `Run_script_on_archived_week` (File = `items('For_each_archived_week')?['{Identifier}']`, `configJson` as above, `asOf` = `convertTimeZone(utcNow(), 'UTC', outputs('Settings')?['timeZone'], 'yyyy-MM-dd')`, `mode` = `extract`), then **Append to array variable** `archive` = `body('Run_script_on_archived_week')?['result']?['extractJson']` |
| 8 | `For_each_list` | **Apply to each** `outputs('Settings')?['lists']`, concurrency **1** | inside: `Get_list` = HTTP **GET** `_api/web/lists/getbytitle('@{encodeUriComponent(items('For_each_list'))}')?$select=ListItemEntityTypeFullName`; `Get_item_ids` = HTTP **GET** `_api/web/lists/getbytitle('@{encodeUriComponent(items('For_each_list'))}')/items?$select=Id&$top=5000` (both with header `Accept` = `application/json;odata=nometadata`); **Append to array variable** `lists` = `{ "title": "@{items('For_each_list')}", "type": "@{body('Get_list')?['ListItemEntityTypeFullName']}", "items": @{body('Get_item_ids')?['value']} }` |
| 9 | `Run_script` | **Run script** on the current report | `configJson`, `asOf` as above, `mode` = `build`, `archiveJson` = `concat('[', join(variables('archive'), ','), ']')`, `sharepointJson` = `string(variables('lists'))` |
| 10 | `If_nothing_was_read` | **Condition** `body('Run_script')?['result']?['ok']` = `false` | Yes: **Terminate** Failed — the lists keep yesterday's data |
| 11 | `For_each_batch` | Steps 5–5c of the setup flow, over `body('Run_script')?['result']?['spBatches']` | concurrency **1** |
| 12 | `Send_daily_summary` | Outlook · **Send an email (V2)** | Subject `body('Run_script')?['result']?['subject']`, Body (code view) = link to `pageUrl` + `body('Run_script')?['result']?['emailHtml']` |
| 13 | `If_list_updates_failed` | **Condition** `length(variables('errors'))` ≠ 0 | Yes: **Terminate** Failed with the errors |

Each refresh removes the previous rows and adds the new ones (~1,600 rows, ~30 batch requests, a few minutes).
During those minutes the page may show partly filled lists, so schedule it before people look.

## 4. Pages

Create four pages (**New → Page**, blank): **Recon – Latest day**, **Recon – This week**, **Recon – This month**,
**Recon – History**. Build *This week* first, then **Copy page** for the other two periods and change the views.

On every page, first a **Quick links** web part with links to the four pages (this replaces the
Daily / Weekly / Monthly / Historical buttons).

**Period page** (views shown for *This week*; use *Latest day …* / *This month …* on the other pages).
Add **List** web parts; in each web part's settings pick the list and the view, give it the title below, and turn *Show command bar* off.

| Section | Web part title | List · View | Dynamic filtering (web part settings → *Dynamic filtering* On) |
|---|---|---|---|
| One column | *(none)* | Recon Summary · **This week KPI** | Field in this list **Customer** ← web part **Customers at a glance**, field **Customer** |
| Two-thirds left | **Customers at a glance** — select a customer to drill down | Recon Summary · **This week Customers** | — (this is the source) |
| One-third right | **Where to focus first** | Recon Issues · **This week Variances** | **Customer** ← Customers at a glance · **Customer** |
| One column | **Interfaces** | Recon Interfaces · **This week Interfaces** | **Customer** ← Customers at a glance · **Customer** |
| Two columns | **Scheduled but not reported** / **Volume anomalies** | Recon Issues · **This week Not reported** / **This week Anomalies** | **Customer** ← Customers at a glance · **Customer** |
| One column | **Reconciled rate — last 28 days** | Recon Daily Trend · **Last 28 days** | **Customer** ← Customers at a glance · **Customer** |

How it behaves: with no customer selected the KPI card shows the portfolio ("All customers · Portfolio"),
the tables show every customer (interfaces and trend grouped by customer, portfolio group first). Select a
customer row in *Customers at a glance* and the card, highlights and every table switch to that customer;
clear the selection to go back to the portfolio.

**History page**

| Web part title | List · View | Dynamic filtering |
|---|---|---|
| **Customer scorecard — reconciled rate by week** | Recon Weekly Scorecard · **Scorecard** | — (source) |
| **Recurring issues** | Recon Recurring Issues · **Recurring** | **Customer** ← Customer scorecard · **Customer** |
| **Reconciled rate — last 28 days** | Recon Daily Trend · **Last 28 days** | **Customer** ← Customer scorecard · **Customer** |

The scorecard's column headings are the week start dates, renamed by the daily flow. The last rows are the
portfolio's ticket coverage and schedule adherence.

Publish the pages and add them to the site navigation or link them from the Batch Monitor page.

## 5. Check it

1. After the setup flow: open each list's views (*All items* drop-down) — they exist and are empty.
2. Run the daily flow once manually (**Test → Manually**). Open *Recon Summary*: rows for Latest day,
   This week, This month, one per customer plus *(All customers)*.
3. Open the *This week* page: compare the KPI card with the e-mail and with the report; select a customer.
4. If a batch failed, the run shows `If_list_updates_failed` → Terminate with the message SharePoint returned.

## Notes

- Lists are replaced, not appended, so they stay small (~1,600 rows) and do not need indexing.
- Column names inside the lists are fixed (`InScope`, `NotReconciled`, …); the display names can be changed in
  list settings without breaking the flow. Do not rename the lists or views — the flow and pages use their titles.
- If a request in a batch is refused, nothing else in that batch is rolled back; the next daily run replaces
  everything again.
- SharePoint behaviour assumed here and not yet verified in a real tenant: the `$batch` format, creating
  fields with internal names via `CreateFieldAsXml` (option 8), and that dynamic filtering shows all rows while
  nothing is selected. The first setup run and the first daily run will show it; send the error text if a step fails.
