"""Execute INSIDE Isaac Sim > Window > Script Editor after opening robot stage.
Prints a pass/fail report. Does not modify any USD or physics property.
"""
import json
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError('No active USD stage. Open the imported robot first.')
# Important: a missing payload or unloaded reference can make geometry disappear.
stage.Load()
prims = list(stage.Traverse())
meshes=[p for p in prims if p.IsA(UsdGeom.Mesh)]
point_counts=[]; rendered=[]
for p in meshes:
    pts=UsdGeom.Mesh(p).GetPointsAttr().Get()
    n=len(pts) if pts is not None else 0
    point_counts.append(n)
    imageable=UsdGeom.Imageable(p)
    vis=str(imageable.ComputeVisibility())
    rendered.append({'path':str(p.GetPath()),'points':n,'visibility':vis})
revs=[p for p in prims if p.IsA(UsdPhysics.RevoluteJoint)]
arts=[p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
refs=[str(p.GetPath()) for p in prims if p.HasAuthoredReferences() or p.HasAuthoredPayloads()]
report={'stage':stage.GetRootLayer().identifier,'total_prims':len(prims), 'meshes':len(meshes), 'nonempty_meshes':sum(x>0 for x in point_counts), 'revolute_joints':len(revs),'articulation_roots':[str(p.GetPath()) for p in arts], 'references_or_payloads':refs[:20], 'sample_meshes':rendered[:25], 'sample_revolute_joints':[str(p.GetPath()) for p in revs[:30]], 'visuals_pass':any(x>0 for x in point_counts), 'articulation_root_pass':len(arts)>0}
print('=== ROBOT STAGE AUDIT ===')
print(json.dumps(report,indent=2))
if not report['visuals_pass']:
    print('FAIL: No nonempty USD Mesh prims. Xform/joints alone are not a visible robot. Do not launch training.')
if not report['articulation_root_pass']:
    print('FAIL: No ArticulationRootAPI found. Do not launch training.')
print('NOTE: collision contacts, limits, inertias, stability and JOINT ORDER require additional dynamic tests.')
