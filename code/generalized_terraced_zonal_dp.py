"""TerracedZonal_Q: linear-time terraced zonal construction of N points on S^2.

Reference implementation of Algorithm 1 of the paper ("A Linear-Time Construction
on S^2 Achieving a Sublinear Energy Gap on Smale's Seventh Problem"). Every
reference below (Section, eq., Algorithm 1 line, Lemma, Proposition, Theorem,
Table, Appendix) is to the paper and names the part that the code implements.

  * moment_degree Q = 0 (default) gives TerracedZonal_0 (TZ_0), the construction
    for Smale's problem that the paper analyzes and calls simply TerracedZonal.
  * Q = 96 gives TerracedZonal_96 (TZ_96), the equal-weight integration rule of
    Section 6; it differs from TZ_0 only in Algorithm 1, line 7.

Three stages (Algorithm 1; spelled out line by line in Algorithms 1a and 1b,
Appendix C):

  Stage 1  How many points on each ring (Section 2.1; lines 1-3)   Theta(N^{5/6})
           `_dp_m_rings`: K ~ 0.952 N^{1/2} rings, then integer populations m_k
           chosen by a 1D dynamic program that trades stretching inside terraces
           against one seam cost per population change, then repaired so that
           sum_k m_k = N - 2.
  Stage 2  Height and rotation of each ring (Section 2.2; lines 4-7) Theta(N^{1/2})
           `_optimize_z_and_phases`: closed-form zonal heights, a half-spacing
           stagger between rings, 4 multipole corrections to the heights, and,
           for Q > 0 only, `_project_legendre_moments`.
  Stage 3  Points (Appendix C.4; line 8)                            Theta(N)
           `generate_sphere_points`: the poles, and m_k equally spaced points at
           height z_k on ring k, rotated by alpha_k.

Total: Theta(N) time and memory for every N >= 1 (Proposition 3.1; proof in
Appendix D). Accuracy: for TZ_0 without the 4 relaxation steps of Stage 2,
E - E_BHS = C_1 N^{2/3} + o(N^{2/3}) with C_1 ~ 0.0262 under the locality
assumption (A) (Theorem 4.3); the relaxation steps lower the excess further.
All constants are listed with their origin in Table 3 (Appendix C.5).

Usage:
  import generalized_terraced_zonal_dp as tz
  x = tz.construct(10_000)                     # TZ_0: (10000, 3) array of unit vectors
  x96 = tz.construct(10_000, moment_degree=96) # TZ_96 (integration rule)
"""

from __future__ import annotations

import math

import numpy as np

EQUIDISTRIBUTED = True
STOCHASTIC = False


def allowed_n(target: int) -> int:
  """Every N >= 1 is admissible (Section 2: "Given any integer N >= 1")."""
  if target < 1:
    raise ValueError("target must be positive")
  return int(target)


def _ring_stretch_cost(
    m: np.ndarray | float, y: np.ndarray | float, c_stretch: float = 0.25
) -> np.ndarray | float:
  """Stretching cost psi_stretch(m, y) of putting m points on a ring that ideally holds y.

  Paper: eq. (5). c_stretch = 1/4 is the shear stiffness of the triangular
  lattice (Lemma 4.2); the exponents 1/3, 2/3 in y^{1/3} m^{2/3} make a terrace's
  best population the mean of its ideal populations to second order
  (Lemma E.4, "unbiased"). Table 3.
  """
  return c_stretch * ((m - y) ** 2) / ((y ** (1.0 / 3.0)) * (m ** (2.0 / 3.0)))


def _dp_m_rings(
    n: int,
    c_step: float = 0.004324,
    c_stretch: float = 0.25,
) -> np.ndarray:
  """Stage 1 (Section 2.1; Algorithm 1, lines 1-3): integer ring populations m_0..m_{K-1}.

  c_step = log(1 + exp(-pi sqrt 3)) ~ 0.004324 is the binding energy per point
  that two equal, staggered adjacent rings lose when their populations differ
  (Section 1.2; Table 3).
  """
  # --- Line 1: ring discretization (Appendix C.1, eqs. (18)-(19)) ---------------
  # Candidate window half-width w = max(20, round(0.65 N^{1/3})), eq. (4).
  window = max(20, int(round(0.65 * (n ** (1.0 / 3.0)))))
  # Polar caps of area 4 pi / N (one point each) and the triangular-lattice row
  # spacing h_0 = sqrt(2 pi sqrt(3) / N), giving K ~ 0.952 N^{1/2} rings.
  theta_cap = math.acos(1.0 - 2.0 / n)
  h0 = math.sqrt(2.0 * math.pi * math.sqrt(3.0) / n)
  k_rings = max(1, int(round((math.pi - 2.0 * theta_cap) / h0)))
  # Ideal real populations y_k: N - 2 times the relative area of each strip.
  theta_edges = np.linspace(theta_cap, math.pi - theta_cap, k_rings + 1)
  dz_ideal = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
  y_ideal = (n - 2) * dz_ideal / np.sum(dz_ideal)

  # --- Line 2: exact 1D dynamic program (eqs. (5)-(7)) ---------------------------
  # Candidates m_k in round(y_k) +- w: a K x W table, W = 2w + 1 = Theta(N^{1/3}).
  y_base = np.maximum(1, np.rint(y_ideal).astype(np.int64))
  w_size = 2 * window + 1
  offsets = np.arange(-window, window + 1, dtype=np.int64)
  cand = y_base[:, None] + offsets[None, :]
  valid_cand = cand >= 1
  cand_safe = np.maximum(1, cand).astype(np.float64)

  # Node cost: stretching psi_stretch(m_k, y_k), eq. (5).
  node_cost = np.where(
      valid_cand,
      _ring_stretch_cost(cand_safe, y_ideal[:, None], c_stretch),
      1e30,
  )

  dp = np.full((k_rings, w_size), 1e30, dtype=np.float64)
  back = np.zeros((k_rings, w_size), dtype=np.int32)
  dp[0] = node_cost[0]
  idx_arange = np.arange(w_size, dtype=np.int32)

  # Viterbi recursion, eq. (7). The seam cost, eq. (6), is
  # c_step (y_{k-1} + y_k) / 2 whenever m_k != m_{k-1}, independent of WHICH
  # populations differ, so each step needs only (a) the best previous value plus
  # one seam ("change") and (b) the same population on the previous ring
  # ("stay"): O(W) work per ring instead of O(W^2) (proof of Proposition 3.1).
  for k in range(1, k_rings):
    step_pen = c_step * 0.5 * (y_ideal[k - 1] + y_ideal[k])
    best_prev_idx = int(np.argmin(dp[k - 1]))
    best_prev_val = dp[k - 1, best_prev_idx]

    # (a) change population: best previous value + seam cost.
    dp_k = best_prev_val + step_pen + node_cost[k]
    back_k = np.full(w_size, best_prev_idx, dtype=np.int32)

    # (b) stay (same m, no seam): cand[k, i] == cand[k-1, j] iff
    # j = i + (y_base[k] - y_base[k-1]), a shifted 1D slice.
    shift = int(y_base[k] - y_base[k - 1])
    i_lo = max(0, -shift)
    i_hi = min(w_size, w_size - shift)
    if i_lo < i_hi:
      j_slice = slice(i_lo + shift, i_hi + shift)
      i_slice = slice(i_lo, i_hi)
      stay_val = dp[k - 1, j_slice] + node_cost[k, i_slice]
      better = stay_val < dp_k[i_slice]
      dp_k[i_slice] = np.where(better, stay_val, dp_k[i_slice])
      back_k[i_slice] = np.where(
          better, idx_arange[j_slice], back_k[i_slice]
      )

    dp[k] = dp_k
    back[k] = back_k

  # Backtracking: the optimal populations, piecewise constant on terraces.
  m_rings = np.empty(k_rings, dtype=np.int64)
  idx = int(np.argmin(dp[k_rings - 1]))
  for k in range(k_rings - 1, -1, -1):
    m_rings[k] = max(1, int(cand[k, idx]))
    idx = int(back[k, idx])

  # --- Line 3: restore sum_k m_k = N - 2 (Appendix C.2) -------------------------
  # Repeatedly apply the move with the least stretching cost per point that
  # neither adds a seam nor splits a terrace: move a terrace boundary by one ring,
  # or change a whole terrace's population by +-1; only if neither fits, change
  # one end ring of a terrace ("peel"). |diff_0| = O(N^{1/3}), so this costs
  # O(N^{5/6}) (Proposition 3.1; Table 6 lists diff_0 for N = 1e3..1e8).
  diff = (n - 2) - int(np.sum(m_rings))
  if diff != 0:
    runs = []  # terraces as [start, end) ring ranges
    r_s = 0
    for k in range(1, k_rings + 1):
      if k == k_rings or m_rings[k] != m_rings[r_s]:
        runs.append([r_s, k])
        r_s = k

    while diff != 0:
      n_runs = len(runs)
      best_cost_per_pt = 1e30
      best_move = None

      # Move 1: shift the boundary between two terraces by one ring.
      for b_idx in range(n_runs - 1):
        s1, e1 = runs[b_idx]
        s2, e2 = runs[b_idx + 1]
        m1 = int(m_rings[s1])
        m2 = int(m_rings[s2])
        if m1 == m2:
          continue
        # Shift right: ring s2 changes from m2 to m1.
        if e2 - s2 > 1:
          d_sum = m1 - m2
          if (diff > 0 and 0 < d_sum <= diff) or (
              diff < 0 and diff <= d_sum < 0
          ):
            yk = float(y_ideal[s2])
            d_cost = float(
                _ring_stretch_cost(m1, yk, c_stretch)
                - _ring_stretch_cost(m2, yk, c_stretch)
            )
            cpp = d_cost / abs(d_sum)
            if cpp < best_cost_per_pt:
              best_cost_per_pt = cpp
              best_move = ("b_right", b_idx, d_sum)
        # Shift left: ring e1 - 1 changes from m1 to m2.
        if e1 - s1 > 1:
          d_sum = m2 - m1
          if (diff > 0 and 0 < d_sum <= diff) or (
              diff < 0 and diff <= d_sum < 0
          ):
            yk = float(y_ideal[e1 - 1])
            d_cost = float(
                _ring_stretch_cost(m2, yk, c_stretch)
                - _ring_stretch_cost(m1, yk, c_stretch)
            )
            cpp = d_cost / abs(d_sum)
            if cpp < best_cost_per_pt:
              best_cost_per_pt = cpp
              best_move = ("b_left", b_idx, d_sum)

      # Move 2: change a whole terrace's population by +-1.
      step_dir = 1 if diff > 0 else -1
      for t_idx in range(n_runs):
        s, e = runs[t_idx]
        t_len = e - s
        if t_len <= 0 or t_len > abs(diff):
          continue
        m_cur = int(m_rings[s])
        m_new = m_cur + step_dir
        if m_new < 1:
          continue
        yk_slice = y_ideal[s:e]
        d_cost = float(
            np.sum(
                _ring_stretch_cost(float(m_new), yk_slice, c_stretch)
                - _ring_stretch_cost(float(m_cur), yk_slice, c_stretch)
            )
        )
        cpp = d_cost / t_len
        if cpp < best_cost_per_pt:
          best_cost_per_pt = cpp
          best_move = ("terrace", t_idx, step_dir * t_len)

      # Move 3 (fallback): change one end ring of a terrace by +-1.
      if best_move is None:
        for t_idx in range(n_runs):
          s, e = runs[t_idx]
          m_cur = int(m_rings[s])
          m_new = m_cur + step_dir
          if e - s > 1 and m_new >= 1:
            yk = float(y_ideal[s])
            d_cost = float(
                _ring_stretch_cost(m_new, yk, c_stretch)
                - _ring_stretch_cost(m_cur, yk, c_stretch)
            )
            if d_cost < best_cost_per_pt:
              best_cost_per_pt = d_cost
              best_move = ("b_peel", t_idx, step_dir)

      # (No admissible move has never occurred: tested exhaustively for
      # N = 1..20,000 and at 1,647 random N up to 1e8.)

      kind, idx_m, d_sum = best_move
      if kind == "b_right":
        k_ring = runs[idx_m + 1][0]
        m_rings[k_ring] = m_rings[runs[idx_m][0]]
        runs[idx_m][1] += 1
        runs[idx_m + 1][0] += 1
      elif kind == "b_left":
        k_ring = runs[idx_m][1] - 1
        m_rings[k_ring] = m_rings[runs[idx_m + 1][0]]
        runs[idx_m][1] -= 1
        runs[idx_m + 1][0] -= 1
      elif kind == "terrace":
        s, e = runs[idx_m]
        m_rings[s:e] += 1 if d_sum > 0 else -1
      else:
        s, e = runs[idx_m]
        m_rings[s] += d_sum
        runs[idx_m][0] = s + 1
        runs.insert(idx_m, [s, s + 1])
      diff -= d_sum

  return m_rings


def _project_legendre_moments(
    z_init: np.ndarray, m_rings: np.ndarray, moment_degree: int
) -> np.ndarray:
  """Algorithm 1, line 7 (Q > 0 only; Section 6, eq. (15)): match height moments.

  Adjusts the heights z so that P_1(z), ..., P_{Q'}(z), Q' = min(K - 1, Q), have
  their exact sphere averages, i.e. the rule integrates every polynomial in z of
  degree <= Q' exactly. Up to 6 damped minimum-norm Gauss-Newton steps, eq. (34)
  (Appendix H); safeguards as in Table 3.
  """
  k_rings = len(m_rings)
  if moment_degree <= 0 or k_rings < 2:
    return z_init
  l_max = min(k_rings - 1, int(moment_degree))
  n = int(np.sum(m_rings)) + 2
  w_rings = m_rings.astype(np.float64) / float(n)
  w_pole = 1.0 / float(n)
  degs = np.arange(1, l_max + 1, dtype=np.int64)
  pole_contrib = w_pole * (1.0 + (-1.0) ** degs)

  z_rings = z_init.copy()
  p_all = np.empty((l_max + 1, k_rings), dtype=np.float64)
  p_all[0] = 1.0
  for _ in range(6):
    # Legendre three-term recurrence and residuals r_l = (1/N)(1 + (-1)^l)
    # + sum_k (m_k / N) P_l(z_k), which vanish for the exact sphere average.
    p_all[1] = z_rings
    for l in range(1, l_max):
      p_all[l + 1] = (
          (2 * l + 1) * z_rings * p_all[l] - l * p_all[l - 1]
      ) / (l + 1)
    p_mat = p_all[1 : l_max + 1]
    p_prev = p_all[0:l_max]
    res = pole_contrib + p_mat @ w_rings
    if float(np.max(np.abs(res))) < 1e-14:
      break
    # Jacobian J_{l,k} = (m_k / N) P'_l(z_k) and minimum-norm step, eq. (34).
    denom = np.minimum(-1e-12, z_rings * z_rings - 1.0)
    dp_mat = degs[:, None] * (z_rings[None, :] * p_mat - p_prev) / denom
    jac = dp_mat * w_rings[None, :]
    gram = jac @ jac.T
    gram.flat[:: l_max + 1] += 1e-14
    dz = -jac.T @ np.linalg.solve(gram, res)
    # Step cap 0.2 / K (a tenth of the mean ring spacing in height), so the
    # rings stay ordered; a step that would reorder them is rejected.
    max_step = float(np.max(np.abs(dz)))
    if max_step > 0.2 / k_rings:
      dz *= (0.2 / k_rings) / max_step
    z_new = np.clip(z_rings + dz, -1.0 + 1e-9, 1.0 - 1e-9)
    if np.all(np.diff(z_new) < 0.0):
      z_rings = z_new
    else:
      break

  return z_rings


def _optimize_z_and_phases(
    m_rings: np.ndarray,
    max_band: int = 6,
    moment_degree: int = 0,
    eta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
  """Stage 2 (Section 2.2; Algorithm 1, lines 4-7): ring heights z_k and rotations alpha_k.

  Theta(N^{1/2}) time (Proposition 3.1).

  Args:
    m_rings: Integer ring populations (length K) from Stage 1.
    max_band: Rings at most this far apart enter the multipole corrections
      (d_max = 6; Table 3).
    moment_degree: Q, the number of matched height moments (line 7); 0 for
      TZ_0 (Smale's problem), 96 for TZ_96 (integration, Section 6).
    eta: Interpolation weight in [0, 1] between TZ_0's heights (0.0) and the
      moment-matched heights (1.0); the paper always uses 1.0.
  """
  k_rings = len(m_rings)
  n = int(np.sum(m_rings)) + 2
  m = m_rings.astype(np.float64)
  cum_end = np.cumsum(m_rings)
  cum_start = cum_end - m_rings
  m_above = 1.0 + cum_start.astype(np.float64)
  m_below = 1.0 + (n - 2 - cum_end).astype(np.float64)

  # --- Line 4: heights that minimize the zonal (azimuthally averaged) energy, in
  # closed form, eq. (8): z_k = (w_+ - w_-) / (w_+ + w_-) from the point counts
  # above and below ring k.
  w_minus = 0.5 * m * (m_above + 0.5 * (m - 1.0))
  w_plus = 0.5 * m * (m_below + 0.5 * (m - 1.0))
  w_sum = w_plus + w_minus
  z_zonal = (w_plus - w_minus) / w_sum

  # --- Line 5: stagger, eq. (9): alpha_k = alpha_{k-1} + 1 / (2 lcm(m_{k-1}, m_k))
  # (mod 1). Inside a terrace lcm(m, m) = m, a half spacing 1/(2m), which
  # interlocks equal rings into a triangular lattice.
  phases = np.zeros(k_rings, dtype=np.float64)
  if k_rings > 1:
    ma = m_rings[:-1]
    mb = m_rings[1:]
    g = np.gcd(ma, mb)
    l_adj = (ma // g) * mb
    d_alpha = 0.5 / l_adj.astype(np.float64)
    phases[1:] = np.cumsum(d_alpha) % 1.0

  # --- Line 6: 4 fixed-point corrections from the multipole forces between rings
  # at most d_max = 6 apart (Appendix C.3, eqs. (20)-(21)). By the cyclotomic
  # identity (Lemma 4.1), the alignment-dependent interaction of rings a, b
  # depends on r = (zeta_a / zeta_b)^{lcm(m_a, m_b)}, where zeta is the
  # stereographic radius; pairs with log r <= -36 contribute below double
  # precision and are skipped (Table 3).
  band_data = []
  for d in range(1, min(max_band + 1, k_rings)):
    ma = m_rings[:-d]
    mb = m_rings[d:]
    g = np.gcd(ma, mb)
    l_ab = ((ma // g) * mb).astype(np.float64)  # lcm(m_a, m_b)
    m_prod = ma.astype(np.float64) * mb.astype(np.float64)
    c_ab = np.cos(2.0 * math.pi * l_ab * (phases[:-d] - phases[d:]))
    band_data.append((d, l_ab, m_prod, c_ab))

  z = z_zonal.copy()
  for _ in range(4):
    log_zeta = 0.5 * (np.log(1.0 - z) - np.log(1.0 + z))
    grad_xi = np.zeros(k_rings, dtype=np.float64)
    for d, l_ab, m_prod, c_ab in band_data:
      log_r = l_ab * (log_zeta[:-d] - log_zeta[d:])
      mask = log_r > -36.0
      if not np.any(mask):
        continue
      r = np.where(mask, np.exp(np.clip(log_r, -36.0, 0.0)), 0.0)
      denom = 1.0 - 2.0 * r * c_ab + r * r
      # Multipole force between rings a and b, eq. (20).
      g_val = np.where(
          mask, m_prod * r * (r - c_ab) / np.maximum(denom, 1e-30), 0.0
      )
      grad_xi[:-d] -= g_val
      grad_xi[d:] += g_val
    # Fixed-point update, eq. (21); kept only if the rings stay ordered.
    z_new = (w_plus - w_minus + grad_xi) / w_sum
    if np.all(np.diff(z_new) < 0.0) and z_new[0] < 1.0 and z_new[-1] > -1.0:
      z = z_new

  # --- Line 7 (Q > 0 only): match the first Q height moments (Section 6).
  if moment_degree > 0 and eta != 0.0:
    z_proj = _project_legendre_moments(z, m_rings, moment_degree)
    z = (1.0 - eta) * z + eta * z_proj

  return z, phases


def generate_sphere_points(
    n: int, *, moment_degree: int = 0, eta: float = 1.0
) -> np.ndarray:
  """TerracedZonal_Q: returns n points on the unit sphere as an (n, 3) float64 array.

  Runs Stages 1-3 of Algorithm 1. For n = 1, 2 it returns the north pole or both
  poles (Section 2).
  """
  if int(n) != n or n < 1:
    raise ValueError("n must be a positive integer")
  n = int(n)
  if n <= 2:
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[0, 2] = 1.0
    if n == 2:
      pts[1, 2] = -1.0
    return pts

  m_rings = _dp_m_rings(n)                     # Stage 1
  z_rings, phases = _optimize_z_and_phases(    # Stage 2
      m_rings, moment_degree=moment_degree, eta=eta
  )

  # --- Stage 3 (Appendix C.4; Algorithm 1, line 8; eq. (22)): the poles, and on
  # ring k the m_k points (r_k cos phi, r_k sin phi, z_k) with
  # phi = 2 pi (j / m_k + alpha_k), j = 0..m_k - 1. Theta(N) time and memory.
  r_rings = np.sqrt(np.maximum(0.0, 1.0 - z_rings * z_rings))

  pts = np.empty((n, 3), dtype=np.float64)
  pts[0] = (0.0, 0.0, 1.0)
  pts[n - 1] = (0.0, 0.0, -1.0)

  k_rings = len(m_rings)
  cum_end = np.cumsum(m_rings)
  cum_start = cum_end - m_rings

  for k in range(k_rings):
    m = int(m_rings[k])
    s_idx = 1 + int(cum_start[k])
    e_idx = 1 + int(cum_end[k])
    j = np.arange(m, dtype=np.float64)
    phi = 2.0 * math.pi * (j / m + phases[k])
    pts[s_idx:e_idx, 0] = r_rings[k] * np.cos(phi)
    pts[s_idx:e_idx, 1] = r_rings[k] * np.sin(phi)
    pts[s_idx:e_idx, 2] = z_rings[k]

  return pts


def construct(
    n: int, *, seed: int = 0, moment_degree: int = 0, eta: float = 1.0
) -> np.ndarray:
  """Construct an N-point TerracedZonal_Q configuration (deterministic; `seed` is ignored)."""
  del seed  # deterministic
  return generate_sphere_points(n, moment_degree=moment_degree, eta=eta)
