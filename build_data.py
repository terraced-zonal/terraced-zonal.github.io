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
The manifest also records, for every point set, its exact logarithmic energy
in the single-counted convention E = sum_{i<j} -log ||x_i - x_j|| (each pair
counted once; half of the double-counted sum over i != j) and its gap per
point (E - E_BHS) / N above the Brauchart-Hardin-Saff asymptotic energy.

For HIGH_TARGETS (N = 1e6, 1e7, 1e8) only the energies are recorded (no cell
data; the page keeps showing the largest TARGETS picture), computed by
  * the exact ring evaluator for the ring constructions (TerracedZonal, ZonalEQ),
  * the O(N) Ewald estimator for the Fibonacci lattice and EqAreaIco (validated
    against exact sums in the large-N benchmark), and
  * the exact O(N^2) sum for Hammersley up to EXACT_MAX (no validated estimator
    exists for its uneven density, so larger N are left as null).

Usage:
  PYTHONPATH=src python3 web/terraced_zonal_demo/build_data.py
  PYTHONPATH=src python3 web/terraced_zonal_demo/build_data.py --energies-only
      (recomputes only the energies and gaps in data/manifest.json)
  PYTHONPATH=src python3 web/terraced_zonal_demo/build_data.py --high-n
      (computes the HIGH_TARGETS energies into data/manifest.json)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import mpmath
import numpy as np
from scipy.spatial import SphericalVoronoi

from smales7.constructions import (
    equal_area_icosahedral, fibonacci_lattice, hammersley, terraced_zonal_dp,
    zonal_equal_area)
from smales7.evaluators import (
    Ring, bhs_residual, ewald_centered_sum, exact_centered_sum, ring_centered_sum)
from smales7.evaluators.ring import rings_terraced_zonal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
# Slider positions: the R10 preferred numbers (10 per decade, roughly log-spaced) from 10 to 1e5.
R10 = [1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0]
TARGETS = sorted({int(math.floor(m * 10 ** e + 0.5)) for e in range(1, 5) for m in R10} | {100_000})
HIGH_TARGETS = [1_000_000, 10_000_000, 100_000_000]  # energies only
EXACT_MAX = 1_000_000  # largest N for the exact O(N^2) pair sum (~12 min on 12 cores at 1e6)
EWALD_C = 1.5  # as in smales7.large_n_benchmark
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


def point_set(mod, target):
  n = int(mod.allowed_n(target))
  pts = np.asarray(mod.construct(n), dtype=np.float64)
  pts /= np.linalg.norm(pts, axis=1, keepdims=True)
  return n, pts


def energies(pts):
  """Exact energy E = sum_{i<j} -log ||x_i - x_j|| and gap per point (E - E_BHS) / N.

  Every pair is counted once. The evaluator returns the centered sum
  S = E - MU N (N - 1) / 2 with MU = 1/2 - log 2; the N^2 term is added back,
  and E_BHS = MU N^2 / 2 - N log(N) / 4 + C_BHS N / 2 subtracted, in mpmath.
  """
  return from_centered_sum(exact_centered_sum(pts), len(pts), "exact")


def from_centered_sum(s, n, method):
  """Energy record from the centered sum S = E - MU N (N - 1) / 2 (E and E - E_BHS in mpmath)."""
  with mpmath.workdps(40):
    e = mpmath.mpf(s) + (mpmath.mpf(1) / 2 - mpmath.log(2)) * n * (n - 1) / 2
  return {"energy": float(e), "excess": bhs_residual(s, n) / n, "method": method}


def rings_zonal_equal_area(n):
  """Uniform rings of zonal_equal_area.construct(n) (poles excluded), north to south.

  Mirrors that construction: collar j has counts[j] points at longitudes
  2 pi ((i + 1/2) / y + offset) (a single point sits at 2 pi offset).
  """
  counts = zonal_equal_area.collar_counts(n)
  colats = zonal_equal_area.cap_colatitudes(n)
  rings, offset = [], 0.0
  for j in range(1, len(counts) - 1):
    y = int(counts[j])
    colat = 0.5 * (colats[j - 1] + colats[j])
    rings.append(Ring(y, math.log(math.tan(colat / 2.0)), alpha=offset + (0.0 if y == 1 else 0.5 / y)))
    y_next = int(counts[j + 1])
    offset += (1.0 / y_next - 1.0 / y) / 2.0 + math.gcd(y, y_next) / (2.0 * y * y_next)
    offset -= math.floor(offset)
  return rings


RING_ADAPTERS = {"tz": rings_terraced_zonal, "zonaleq": rings_zonal_equal_area}
EWALD_KEYS = {"fib", "eqareaico"}


def high_n_energy(key, mod, target):
  """Energy record at a HIGH_TARGETS size (None values where no reliable method applies)."""
  n = int(mod.allowed_n(target))
  if key in RING_ADAPTERS:
    return from_centered_sum(ring_centered_sum(RING_ADAPTERS[key](n)), n, "ring")
  if key in EWALD_KEYS:
    _, pts = point_set(mod, target)
    return from_centered_sum(ewald_centered_sum(pts, c=EWALD_C), n, "ewald")
  if n <= EXACT_MAX:
    _, pts = point_set(mod, target)
    return from_centered_sum(exact_centered_sum(pts), n, "exact")
  return {"n": n, "energy": None, "excess": None, "method": None}


def check_ring_adapters(target=100000):
  """The ring evaluators must reproduce the exact pair sum on the TARGETS point sets."""
  for key, _, _, mod in CONSTRUCTIONS:
    if key in RING_ADAPTERS:
      n, pts = point_set(mod, target)
      ring = from_centered_sum(ring_centered_sum(RING_ADAPTERS[key](n)), n, "ring")
      exact = energies(pts)
      diff = abs(ring["excess"] - exact["excess"])
      print(f"ring check {key} N={n}: (E - E_BHS)/N ring {ring['excess']:.12e} "
            f"exact {exact['excess']:.12e}  |diff| {diff:.1e}", flush=True)
      assert diff < 1e-9, key


def high_n():
  """Adds (or refreshes) the HIGH_TARGETS energies in data/manifest.json."""
  check_ring_adapters()
  path = os.path.join(OUT, "manifest.json")
  with open(path) as f:
    manifest = json.load(f)
  mods = {key: mod for key, _, _, mod in CONSTRUCTIONS}
  for target in HIGH_TARGETS:
    for entry in manifest["constructions"]:
      t0 = time.time()
      info = high_n_energy(entry["key"], mods[entry["key"]], target)
      info["n"] = int(mods[entry["key"]].allowed_n(target))
      entry["files"][str(target)] = info
      gap = "n/a" if info["excess"] is None else f"{info['excess']:.6e}"
      print(f"{entry['name']:18s} N={info['n']:10d}  (E - E_BHS)/N = {gap}  "
            f"[{info['method']}, {time.time() - t0:.0f} s]", flush=True)
      manifest["targets"] = sorted(set(manifest["targets"]) | {target})
      with open(path, "w") as f:  # save after each value: the large sums take minutes
        json.dump(manifest, f, indent=1)


def energies_only():
  """Recomputes the energies in an existing manifest without touching the cell data."""
  path = os.path.join(OUT, "manifest.json")
  with open(path) as f:
    manifest = json.load(f)
  mods = {key: mod for key, _, _, mod in CONSTRUCTIONS}
  for entry in manifest["constructions"]:
    for target, info in entry["files"].items():
      if "file" not in info:  # HIGH_TARGETS entry: see high_n()
        continue
      n, pts = point_set(mods[entry["key"]], int(target))
      assert n == info["n"], (entry["key"], target, n, info["n"])
      info.update(energies(pts))
      print(f"{entry['name']:18s} N={n:6d}  E = {info['energy']:.15e}  "
            f"(E - E_BHS)/N = {info['excess']:.6e}", flush=True)
  with open(path, "w") as f:
    json.dump(manifest, f, indent=1)


def main():
  os.makedirs(OUT, exist_ok=True)
  manifest = {"targets": TARGETS, "constructions": []}
  for key, name, family, mod in CONSTRUCTIONS:
    entry = {"key": key, "name": name, "family": family, "files": {}}
    by_n = {}  # several targets can share one admissible N (e.g. EqAreaIco at small N)
    for target in TARGETS:
      n = int(mod.allowed_n(target))
      if n in by_n:
        entry["files"][str(target)] = dict(by_n[n])
        continue
      n, pts = point_set(mod, target)
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
      entry["files"][str(target)] = by_n[n] = {"file": fname, "n": n, **energies(pts)}
      s = summary
      print(f"{name:18s} N={n:5d}  median aniso {100*s['median_aniso']:5.2f}%  "
            f"perim {100*s['median_perim']:5.2f}%  5:{s['n5']} 7:{s['n7']} other:{s['n_other']}")
    manifest["constructions"].append(entry)
  # Keep the HIGH_TARGETS energies of an existing manifest (recompute them with --high-n).
  path = os.path.join(OUT, "manifest.json")
  if os.path.exists(path):
    with open(path) as f:
      old = {e["key"]: e for e in json.load(f)["constructions"]}
    for entry in manifest["constructions"]:
      for target in HIGH_TARGETS:
        info = old.get(entry["key"], {}).get("files", {}).get(str(target))
        if info is not None:
          entry["files"][str(target)] = info
          manifest["targets"] = sorted(set(manifest["targets"]) | {target})
  with open(path, "w") as f:
    json.dump(manifest, f, indent=1)
  # Remove data files of earlier target lists that the manifest no longer references.
  keep = {"manifest.json"}
  for entry in manifest["constructions"]:
    for info in entry["files"].values():
      if "file" in info:
        keep.add(info["file"])
        base = info["file"][:-len(".json")]
        keep.update(f"{base}_{s}" for s in ("pts.f32", "perim.f32", "aniso.f32", "deg.u8"))
  for fname in sorted(os.listdir(OUT)):
    if fname not in keep:
      os.remove(os.path.join(OUT, fname))
      print("removed stale", fname)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--energies-only", action="store_true",
                      help="only recompute the energies in data/manifest.json")
  parser.add_argument("--high-n", action="store_true",
                      help="compute the HIGH_TARGETS energies into data/manifest.json")
  args = parser.parse_args()
  if args.high_n:
    high_n()
  elif args.energies_only:
    energies_only()
  else:
    main()
