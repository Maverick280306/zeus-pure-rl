#!/usr/bin/env python3
"""Read-only model intake: MJCF, URDF or USD. Python stdlib only.
Does not pretend that syntax alone proves a valid PhysX articulation.
"""
import argparse, json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET

def main():
    ap=argparse.ArgumentParser(description='Inspect and register a robot model supplied by any team')
    ap.add_argument('source',type=Path)
    ap.add_argument('--out',type=Path,default=Path('robot_intake.json'))
    a=ap.parse_args(); p=a.source.expanduser().resolve()
    if not p.is_file(): ap.error('File not found: '+str(p))
    kind={'.xml':'mjcf','.urdf':'urdf','.usd':'usd','.usda':'usd','.usdc':'usd'}.get(p.suffix.lower())
    if kind is None: ap.error('Supported: .xml MJCF, .urdf URDF, .usd/.usda/.usdc USD. For CAD, request URDF/MJCF/USD with link transforms and inertial properties.')
    report={'source':str(p),'format':kind,'status':'INSPECTION_ONLY','checks':{},'warnings':[]}
    if kind in ('mjcf','urdf'):
        root=ET.parse(p).getroot()
        if kind=='mjcf':
            if root.tag!='mujoco': ap.error('Expected <mujoco> root for MJCF')
            meshes=root.findall('.//asset/mesh')
            names=[]; missing=[]; inline=0
            for m in meshes:
                name=m.get('name',''); names.append(name)
                if m.get('vertex'): inline+=1
                if m.get('file'):
                    f=p.parent/m.get('file')
                    if not f.is_file(): missing.append(str(f))
            joints=[j.get('name') for j in root.findall('.//joint') if j.get('name')]
            actuators=[j.get('joint') for act in root.findall('.//actuator') for j in act if j.get('joint')]
            report['checks']={'meshes':len(meshes),'inline_meshes':inline,'missing_mesh_files':missing,'named_joints':joints,'actuated_joints':actuators,'actuated_count':len(set(actuators))}
            if inline: report['warnings'].append('MJCF inline meshes may fail in Isaac Sim importer: export external triangulated assets into a separate MJCF')
            if not actuators: report['warnings'].append('No actuator joint mappings declared; training joint names must be supplied manually')
        else:
            if root.tag!='robot': ap.error('Expected <robot> root for URDF')
            links=root.findall('link'); joints=root.findall('joint'); missing=[]
            for mesh in root.findall('.//mesh'):
                fname=mesh.get('filename','')
                if not fname or fname.startswith('package://'): report['warnings'].append('Resolve package:// mesh resources using ROS package paths before import: '+fname)
                elif not Path(fname).is_absolute() and not (p.parent/fname).exists(): missing.append(fname)
            lacking=[x.get('name') for x in links if x.find('inertial') is None]
            report['checks']={'links':len(links),'joints':len(joints),'revolute_joints':[j.get('name') for j in joints if j.get('type') in ('revolute','continuous')], 'missing_relative_meshes':missing, 'links_without_inertial':lacking}
            if lacking: report['warnings'].append('Some links lack inertial data. Inspect mass/inertia before physics.')
    else:
        report['checks']['usd_runtime']='Needs Isaac Sim Kit or pxr USD Python runtime for payloads, variant sets and articulation inspection'
        report['warnings'].append('Run tools/inspect_stage.py inside Isaac Sim after loading USD. File presence does not prove visual or physics validity.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Report saved:',a.out)

if __name__=='__main__':main()
