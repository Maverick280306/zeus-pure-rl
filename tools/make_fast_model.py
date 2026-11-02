"""
make_fast_model.py -- build model/zeus_fast.xml from the full-mesh CAD model.

    python tools/make_fast_model.py --src /path/to/model   # folder with zeus.xml + meshes/

MuJoCo collides a mesh geom as the convex hull of its vertices, but it still builds (and, on
every collision pass, clears) a bounding-volume tree over every triangle of the CAD mesh. For
this robot that tree is the largest single cost of a physics step. This script replaces each
mesh by the vertices of its own convex hull, written inline in the XML, so the result is:
  * the same collision shape (convex hull is unchanged),
  * the same mass / inertia (the XML sets them explicitly: inertiafromgeom="false"),
  * a ~40x smaller tree, no STL files, and a ~300 KB model instead of 40+ MB.
Needs numpy + scipy (dev dependency only; the trained/runtime code does not).
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull


def read_stl_points(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = int(raw[80:84].view(np.uint32)[0])
    if raw.size != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL")
    tri = raw[84:].reshape(n, 50)[:, 12:48].copy().view("<f4")
    return tri.reshape(-1, 3).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder holding zeus.xml and its meshes/")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "model", "zeus_fast.xml"))
    a = ap.parse_args()
    tree = ET.parse(os.path.join(a.src, "zeus.xml"))
    root = tree.getroot()
    comp = root.find("compiler")
    meshdir = os.path.join(a.src, comp.attrib.pop("meshdir", ""))
    nin = nout = 0
    for mesh in root.iter("mesh"):
        pts = read_stl_points(os.path.join(meshdir, mesh.attrib.pop("file")))
        mesh.attrib.pop("content_type", None)
        hull = pts[ConvexHull(pts).vertices].astype(np.float32).astype(np.float64)
        mesh.set("vertex", " ".join(f"{v:.9g}" for v in hull.ravel()))
        nin += len(np.unique(pts, axis=0)); nout += len(hull)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ET.indent(tree)
    tree.write(a.out, encoding="utf-8")
    print(f"{nin} unique mesh vertices -> {nout} hull vertices -> {os.path.abspath(a.out)} "
          f"({os.path.getsize(a.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
