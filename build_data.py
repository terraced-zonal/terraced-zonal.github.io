"""Builds the data files for the TerracedZonal web demo.

For every construction and every target N, computes the spherical Voronoi
diagram of the point set and, for each cell,

  * ``deg``:   number of Voronoi edges (= Delaunay neighbors),
  * ``aniso``: anisotropy sqrt(lambda_max / lambda_min) - 1 of the cell's
               second-moment tensor in the tangent plane (as in the paper's
               Figure 5; 0 for a regular hexagon),
  * ``perim``: perimeter excess P / P_hex - 1, with P_hex the perimeter of the
               regular spherical hexagon of the same area (paper, Figure 1).

Writes data/<key>_<target>.json and data/manifest.json next to this script.

Usage:
  PYTHONPATH=src python3 web/terraced_zonal_demo/build_data.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.spatial import SphericalVoronoi

from smales7.constructions import (
    equal_area_icosahedral, fibonacci_lattice, hammersley, terraced_zonal_dp,
    zonal_equal_area)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
TARGETS = [250, 500, 1000, 2000, 4000, 10000, 30000, 100000]
FULL_MAX = 10000  # targets above this: binary per-point data plus only the cells the zoom can show
ELEV = math.radians(22.0)  # must match ELEV in app.js: the zoom center stays at this latitude
ZOOM_R = 1.8 * math.hypot(3.2, 2.4)  # zoom cell-selection radius (app.js), in units of unit area per point
CONSTRUCTIONS = [  # (key, display name, family, module)
    ("fib", "Fibonacci lattice", "Spiral (Swinbank\u2013Purser, odd N)", fibonacci_lattice),
    ("zonaleq", "ZonalEQ", "Zonal rings", zonal_equal_area),
    ("eqareaico", "EqAreaIco", "Polyhedral", equal_area_icosahedral),
    ("hammersley", "Hammersley", "Quasi-Monte Carlo", hammersley),
    ("tz", "TerracedZonal", "This paper", terraced_zonal_dp),
]


def tangent_frame(c):
  if abs(c[2]) > 0.999:
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
  e = np.cross([0.0, 0.0, 1.0], c)
  e /= np.linalg.norm(e)
  return e, np.cross(c, e)


def cell_metrics(points, sv):
  areas = sv.calculate_areas()
  n = len(points)
  deg = np.empty(n, dtype=int)
  perim = np.empty(n)
  aniso = np.empty(n)
  for k, reg in enumerate(sv.regions):
    v = sv.vertices[reg]
    deg[k] = len(reg)
    w = np.roll(v, -1, axis=0)
    p = np.arccos(np.clip(np.sum(v * w, axis=1), -1.0, 1.0)).sum()
    # Reference: the regular spherical hexagon of the same area (interior angle
    # alpha, side a), so that a perfect cell scores exactly 0 at every N.
    alpha = (areas[k] + 4.0 * math.pi) / 6.0
    side = 2.0 * math.acos(min(1.0, math.cos(math.pi / 6.0) / math.sin(alpha / 2.0)))
    perim[k] = p / (6.0 * side) - 1.0
    e, nn = tangent_frame(points[k])
    x, y = v @ e, v @ nn
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
      x, y = x[::-1], y[::-1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2
    cx, cy = ((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)
    x, y, x1, y1 = x - cx, y - cy, x1 - cx, y1 - cy
    cr = x * y1 - x1 * y
    ixx = ((y * y + y * y1 + y1 * y1) * cr).sum() / 12
    iyy = ((x * x + x * x1 + x1 * x1) * cr).sum() / 12
    ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24
    ev = np.linalg.eigvalsh(np.array([[ixx, ixy], [ixy, iyy]]))
    aniso[k] = math.sqrt(ev[1] / ev[0]) - 1.0
  return deg, perim, aniso


def main():
  os.makedirs(OUT, exist_ok=True)
  manifest = {"targets": TARGETS, "constructions": []}
  for key, name, family, mod in CONSTRUCTIONS:
    entry = {"key": key, "name": name, "family": family, "files": {}}
    for target in TARGETS:
      n = int(mod.allowed_n(target))
      pts = np.asarray(mod.construct(n), dtype=np.float64)
      pts /= np.linalg.norm(pts, axis=1, keepdims=True)
      sv = SphericalVoronoi(pts)
      sv.sort_vertices_of_regions()
      deg, perim, aniso = cell_metrics(pts, sv)
      summary = {
              "median_perim": float(np.median(perim)),
              "median_aniso": float(np.median(aniso)),
              "mean_perim": float(np.mean(perim)),
              "mean_aniso": float(np.mean(aniso)),
              "frac_perim_gt1": float(np.mean(perim > 0.01)),
              "n5": int(np.sum(deg == 5)), "n7": int(np.sum(deg == 7)),
              "n_other": int(np.sum((deg != 5) & (deg != 6) & (deg != 7))),
      }
      fname = f"{key}_{target}.json"
      if target <= FULL_MAX:  # by target, so that every construction switches at the same slider position
        rec = {
            "key": key, "name": name, "family": family, "n": n, "large": False,
            "pts": np.round(pts, 5).ravel().tolist(),
            "verts": np.round(sv.vertices, 5).ravel().tolist(),
            "regions": [list(map(int, r)) for r in sv.regions],
            "deg": deg.tolist(),
            "perim": np.round(perim, 5).tolist(),
            "aniso": np.round(aniso, 5).tolist(),
            "summary": summary,
        }
      else:
        base = f"{key}_{target}"
        pts.astype("<f4").tofile(os.path.join(OUT, base + "_pts.f32"))
        perim.astype("<f4").tofile(os.path.join(OUT, base + "_perim.f32"))
        aniso.astype("<f4").tofile(os.path.join(OUT, base + "_aniso.f32"))
        np.clip(deg, 0, 255).astype("u1").tofile(os.path.join(OUT, base + "_deg.u8"))
        # Cells near latitude ELEV, the only ones the zoom panel can show.
        scale = math.sqrt(n / (4.0 * math.pi))
        lat = np.arcsin(np.clip(pts[:, 2], -1, 1))
        band = np.flatnonzero(np.abs(lat - ELEV) < ZOOM_R / scale + 0.01)
        used = sorted({v for k in band for v in sv.regions[k]})
        remap = {v: i for i, v in enumerate(used)}
        rec = {
            "key": key, "name": name, "family": family, "n": n, "large": True,
            "bin": base,
            "band_cells": band.tolist(),
            "band_verts": np.round(sv.vertices[used], 6).ravel().tolist(),
            "band_regions": [[remap[v] for v in sv.regions[k]] for k in band],
            "summary": summary,
        }
      with open(os.path.join(OUT, fname), "w") as f:
        json.dump(rec, f, separators=(",", ":"))
      entry["files"][str(target)] = {"file": fname, "n": n}
      s = summary
      print(f"{name:18s} N={n:5d}  median aniso {100*s['median_aniso']:5.2f}%  "
            f"perim {100*s['median_perim']:5.2f}%  5:{s['n5']} 7:{s['n7']} other:{s['n_other']}")
    manifest["constructions"].append(entry)
  with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)


if __name__ == "__main__":
  main()
