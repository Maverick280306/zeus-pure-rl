#!/usr/bin/env python3
"""Convert MJCF inline vertex assets into triangulated binary STL.
Avoid double scaling: bake mesh scale into STL positions; strip original scale.
Never edits source MJCF. Requires numpy and scipy. No IsaacSim dependency.
"""
import argparse, struct, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

def main():
 ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 source=a.source.resolve();out=a.out.resolve()
 if source==out:ap.error('Output MUST differ from source')
 root=ET.parse(source).getroot()
 if root.tag!='mujoco':ap.error('Not MJCF')
 folder=out.parent/(out.stem+'_meshes');folder.mkdir(parents=True,exist_ok=True)
 count=0
 for m in root.findall('.//asset/mesh'):
  raw=m.get('vertex');name=m.get('name')
  if not raw: continue
  if not name or '/' in name or '..' in name:raise ValueError('Unsafe mesh name: '+repr(name))
  values=np.fromstring(raw,sep=' ')
  if len(values)%3:raise ValueError('Malformed mesh vertices '+name)
  v=values.reshape(-1,3)
  scale=np.fromstring(m.get('scale','1 1 1'),sep=' ')
  if len(scale)!=3:raise ValueError('Malformed scale '+name)
  v=v*scale
  hull=ConvexHull(v)
  filename=folder/(name+'.stl')
  with filename.open('wb') as f:
   f.write(b'Robot convex hull'.ljust(80,b'\x00'))
   f.write(struct.pack('<I',len(hull.simplices)))
   for ids,eq in zip(hull.simplices,hull.equations):
    ids=list(map(int,ids));pts=v[ids]
    normal=np.cross(pts[1]-pts[0],pts[2]-pts[0]);norm=np.linalg.norm(normal)
    if norm<1e-16:raise ValueError('Degenerate triangle '+name)
    if np.dot(normal,eq[:3])<0:
     ids[1],ids[2]=ids[2],ids[1];pts=v[ids];normal=-normal
    normal=normal/norm
    f.write(struct.pack('<12fH',*(list(normal)+list(pts.reshape(-1))),0))
  m.attrib.pop('vertex',None);m.attrib.pop('face',None);m.attrib.pop('scale',None)
  m.set('file',f'{folder.name}/{name}.stl');count+=1
 out.parent.mkdir(parents=True,exist_ok=True)
 ET.ElementTree(root).write(out,encoding='utf-8',xml_declaration=True)
 print(f'Converted {count} inline meshes to {folder}')
 print(f'MJCF copy: {out}; original unchanged: {source}')
 if not count:print('No inline meshes. The copied MJCF may still reference meshes with relative paths; check intake report.')

if __name__=='__main__':main()
