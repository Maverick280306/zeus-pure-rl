#!/usr/bin/env python3
"""Check expected actuated joint names and asset-preparation artifacts against team manifest."""
import json, argparse,xml.etree.ElementTree as ET
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',default='robot_manifest.json');a=p.parse_args()
m=json.loads(Path(a.manifest).read_text())
source=Path(m['source']).resolve()
if not source.is_file():raise SystemExit('FAIL: missing source model '+str(source))
if source.suffix=='.xml':
 root=ET.parse(source).getroot()
 names={x.get('name') for x in root.findall('.//joint')}
 absent=[n for n in m['actuated_joint_names'] if n not in names]
 print('Joint names expected:',m['actuated_joint_names']);print('Missing joints:',absent)
 if absent:raise SystemExit('FAIL: source joint names missing')
else:print('Non-MJCF model: verify joint names inside Isaac stage manually.')
assert len(m['actuated_joint_names'])==m['action_dimension'], 'action dimension != declared controlled joints'
print('PASS: manifest/source static checks. Physics mapping remains untested.')
