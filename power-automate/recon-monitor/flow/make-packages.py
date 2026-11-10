"""Builds Power Automate import packages (My flows → Import → Import Package (Legacy)) from the flow definitions.

    python3 flow/make-packages.py        → flow/packages/Recon-Setup-Lists.zip, flow/packages/Recon-Daily-Refresh.zip

Each package holds one flow and the connections it uses (SharePoint, Excel Online (Business), Office 365 Outlook);
during import you pick your own connection for each. Ids are derived from the flow name so rebuilding gives the same
package.
"""
import json, os, uuid, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
APIS = {
    'shared_sharepointonline': 'SharePoint',
    'shared_excelonlinebusiness': 'Excel Online (Business)',
    'shared_office365': 'Office 365 Outlook',
}
NS = uuid.UUID('6f1c1d2e-8a51-4c7b-9a43-0d6c2f1e5b11')
gid = lambda *k: str(uuid.uuid5(NS, '/'.join(k)))


def apis_used(defn):
    used = set()
    def walk(acts):
        for a in acts.values():
            host = (a.get('inputs') or {}).get('host') if isinstance(a.get('inputs'), dict) else None
            if host: used.add(host['connectionName'])
            for k in ('actions',):
                if k in a: walk(a[k])
            if 'else' in a: walk(a['else'].get('actions', {}))
    walk(defn['actions'])
    return [k for k in APIS if k in used]


def package(src, display, zip_name, description):
    defn = json.load(open(os.path.join(HERE, src), encoding='utf-8'))
    defn.pop('metadata', None)
    flow_id = gid(display, 'flow')
    used = apis_used(defn)
    res_flow = gid(display, 'resource')
    resources = {res_flow: {
        'id': '/providers/Microsoft.Flow/flows/' + flow_id, 'name': flow_id, 'type': 'Microsoft.Flow/flows',
        'suggestedCreationType': 'New', 'creationType': 'Existing, New, Update',
        'details': {'displayName': display}, 'configurableBy': 'User', 'hierarchy': 'Root', 'dependsOn': []}}
    apis_map, conns_map, conn_refs = {}, {}, {}
    for api in used:
        r_api, r_conn = gid(display, api, 'api'), gid(display, api, 'connection')
        resources[r_api] = {'id': '/providers/Microsoft.PowerApps/apis/' + api, 'name': api, 'type': 'Microsoft.PowerApps/apis',
                            'suggestedCreationType': 'Existing', 'details': {'displayName': APIS[api], 'iconUri': ''},
                            'configurableBy': 'System', 'hierarchy': 'Child', 'dependsOn': []}
        resources[r_conn] = {'type': 'Microsoft.PowerApps/apis/connections', 'suggestedCreationType': 'Existing', 'creationType': 'Existing',
                             'details': {'displayName': APIS[api], 'iconUri': ''}, 'configurableBy': 'User', 'hierarchy': 'Child', 'dependsOn': [r_api]}
        resources[res_flow]['dependsOn'] += [r_api, r_conn]
        apis_map[api], conns_map[api] = r_api, r_conn
        conn_refs[api] = {'connectionName': api.replace('_', '-') + '-' + gid(display, api)[:8], 'source': 'Embedded',
                          'id': '/providers/Microsoft.PowerApps/apis/' + api, 'tier': 'NotSpecified'}
    manifest = {'schema': '1.0', 'details': {'displayName': display, 'description': description, 'createdTime': '2026-11-10T00:00:00.0000000Z',
                'packageTelemetryId': gid(display, 'telemetry'), 'creator': 'N/A', 'sourceEnvironment': ''}, 'resources': resources}
    flow = {'name': flow_id, 'id': '/providers/Microsoft.Flow/flows/' + flow_id, 'type': 'Microsoft.Flow/flows',
            'properties': {'apiId': '/providers/Microsoft.PowerApps/apis/shared_logicflows', 'displayName': display, 'definition': defn,
                           'connectionReferences': conn_refs, 'flowFailureAlertSubscribed': False, 'isManaged': False}}
    os.makedirs(os.path.join(HERE, 'packages'), exist_ok=True)
    out = os.path.join(HERE, 'packages', zip_name)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        put = lambda name, obj: z.writestr(zipfile.ZipInfo(name, (2026, 11, 10, 0, 0, 0)), json.dumps(obj, indent=2, ensure_ascii=False))
        put('manifest.json', manifest)
        put('Microsoft.Flow/flows/manifest.json', {'packageSchemaVersion': '1.0', 'flowAssets': {'assetPaths': [flow_id]}})
        put(f'Microsoft.Flow/flows/{flow_id}/definition.json', flow)
        put(f'Microsoft.Flow/flows/{flow_id}/apisMap.json', apis_map)
        put(f'Microsoft.Flow/flows/{flow_id}/connectionsMap.json', conns_map)
    print(out, ', '.join(APIS[a] for a in used))


package('setup-lists.json', 'Recon Monitor - Setup SharePoint lists', 'Recon-Setup-Lists.zip',
        'Run once: creates the six Recon lists with columns, views and formatting.')
package('daily-refresh.json', 'Recon Monitor - Daily refresh', 'Recon-Daily-Refresh.zip',
        'Daily: reads the archived weeks and the current Reconciliation Report and refreshes the Recon lists.')
