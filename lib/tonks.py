"""
Analytical solution for an N-component hard-sphere + Coulomb fluid
in a single-file (1D) cylindrical pore — generalised Tonks gas model.

Core computation: ion densities are found by solving det(Theta) = 0,
where Theta_ij = eta_ij * sqrt(A_i A_j) * exp(0.5*(mu_el_i + mu_el_j)) - delta_ij.
The pair integrals eta_ij capture hard-sphere exclusion and screened Coulomb
interactions along the pore axis.

Public API
----------
Ion                    : ion species descriptor
f_density(ions, ...)   : solve for densities -> ([rho_i], sv)
f_charge(dens, ions)   : sum q_i * rho_i
f_charge_full(...)     : set elchempot then return charge (used for dQ/du)
"""

import numpy as np
from math import log, exp, sqrt
from scipy.integrate import quad
from scipy.optimize import brentq

S_MAX_VALUE = 5.0   # initial upper bound for the packing density search


class Ion:
    """Ion species in the single-file pore model.

    Parameters
    ----------
    charge      : valency (e.g. +1, -1, +2)
    radius_A    : hard-sphere radius [Å]
    chempot_kBT : chemical potential [kBT]  (NOT including log(rr0_A))
    rr0_A       : reference length scale = r_pore / 2.4  [Å]
    """
    def __init__(self, charge, radius_A, chempot_kBT, rr0_A):
        self.charge = charge
        self.radius = radius_A
        self.rr0_A  = rr0_A
        self.area   = 1.0
        self.chempot   = chempot_kBT + log(rr0_A)
        self.elchempot = chempot_kBT + log(rr0_A)

    def set_chempot(self, mu_kBT):
        """Update chemical potential [kBT]; resets elchempot to same value."""
        self.chempot   = mu_kBT + log(self.rr0_A)
        self.elchempot = self.chempot

    def set_elchempot(self, u_kBT):
        """Set electrochemical potential: mu_chem + q * u  [kBT]."""
        self.elchempot = self.chempot + self.charge * u_kBT


# ---------- pair partition function integrals ----------

_SV_BOLTZMANN = 8.0  # switch from Mayer-f to Boltzmann form at this sv (repulsive pairs only)


def _eta(i, j, ions, sv, A, r0_A):
    """Pair partition function integral eta_ij.

    For repulsive pairs (qi*qj > 0) at large sv the Mayer-f split
    exp(−sv·d)/sv + ∫(exp(−U)−1) suffers catastrophic cancellation: the two
    terms nearly cancel to give η << each term, so the quadrature error in the
    integral alone can exceed η itself.  Above _SV_BOLTZMANN we switch to the
    Boltzmann form  ∫_d^∞ exp(−sv·x − U(x)) dx,  which is algebraically
    identical but avoids the cancellation.  The integral is split at the
    integrand maximum (x_m = ln(C/sv)) so that scipy quad does not warn about
    a slowly convergent tail.  Mayer-f is kept for small sv and for attractive
    or neutral pairs where cancellation is absent.
    """
    d  = (ions[i].radius + ions[j].radius) / r0_A
    qi, qj = ions[i].charge, ions[j].charge
    C  = qi * qj * A                          # >0 repulsive, <0 attractive
    if C > 0 and sv > _SV_BOLTZMANN:
        # x_m must be >= d; if the peak falls before the integration domain the
        # integrand is monotone-decreasing on [d, inf) and no split is needed.
        x_m = max(d, log(C / sv) if C > sv else d)
        if x_m == d:
            v, _ = quad(lambda x: exp(-sv * x - C * exp(-x)), d, np.inf, limit=200)
            return v
        v1, _ = quad(lambda x: exp(-sv * x - C * exp(-x)), d,   x_m,    limit=200)
        v2, _ = quad(lambda x: exp(-sv * x - C * exp(-x)), x_m, np.inf, limit=200)
        return v1 + v2
    v, _ = quad(
        lambda x, sv, A: exp(-sv * x) * (exp(-C * exp(-x)) - 1.0),
        d, np.inf, args=(sv, A), limit=50)
    return exp(-sv * d) / sv + v


def _d_eta(i, j, ions, sv, A, r0_A):
    """Derivative d(eta_ij)/d(sv).  Same hybrid Mayer-f / Boltzmann strategy as _eta."""
    d  = (ions[i].radius + ions[j].radius) / r0_A
    qi, qj = ions[i].charge, ions[j].charge
    C  = qi * qj * A
    if C > 0 and sv > _SV_BOLTZMANN:
        x_m = max(d, log(C / sv) if C > sv else d)
        if x_m == d:
            v, _ = quad(lambda x: -x * exp(-sv * x - C * exp(-x)), d, np.inf, limit=200)
            return v
        v1, _ = quad(lambda x: -x * exp(-sv * x - C * exp(-x)), d,   x_m,    limit=200)
        v2, _ = quad(lambda x: -x * exp(-sv * x - C * exp(-x)), x_m, np.inf, limit=200)
        return v1 + v2
    v, _ = quad(
        lambda x, sv, A: x * exp(-sv * x) * (exp(-C * exp(-x)) - 1.0),
        d, np.inf, args=(sv, A), limit=50)
    return -(1.0 / sv + d) * exp(-sv * d) / sv - v


# ---------- equilibrium condition ----------

def _build_theta(ions, sv, bV0, rr0_A):
    N = len(ions)
    T = np.empty((N, N))
    for i in range(N):
        T[i, i] = _eta(i, i, ions, sv, bV0, rr0_A) * ions[i].area * exp(ions[i].elchempot) - 1.0
    for i in range(N):
        for j in range(i + 1, N):
            v = _eta(i, j, ions, sv, bV0, rr0_A) * sqrt(ions[i].area * ions[j].area) \
                * exp(0.5 * (ions[i].elchempot + ions[j].elchempot))
            T[i, j] = T[j, i] = v
    return T


def _eq_s(sv, ions, bV0, rr0_A):
    return np.linalg.det(_build_theta(ions, sv, bV0, rr0_A))


def _find_bracket(s_max, ions, bV0, rr0_A, sv_hint=None):
    """Find [s0, s1] bracketing the root of _eq_s nearest to sv_hint.

    sv_hint : solution from the previous scan step (continuation tracking).
              When provided, a tight bracket around sv_hint is tried first so
              the solver stays on the same physical branch even when multiple
              roots exist.  Falls back to a deterministic log-spaced scan.
    Returns (None, None) when no root can be found (effectively empty pore).
    """
    N = len(ions)
    s_min = 1e-11
    # expand s_max until it has the correct asymptotic sign  (-1)^N
    v1 = _eq_s(s_max, ions, bV0, rr0_A)
    while abs(1.0 - v1 * (-1) ** N) > 1e-3:
        s_max *= 1.5
        v1 = _eq_s(s_max, ions, bV0, rr0_A)

    # Continuation: try a tight bracket centred on sv_hint first.
    # Expand the half-width in steps until a sign change is detected.
    if sv_hint is not None and sv_hint > s_min:
        for spread in (0.05, 0.10, 0.20, 0.50):
            sa = max(s_min, sv_hint * (1.0 - spread))
            sb = min(s_max, sv_hint * (1.0 + spread))
            va = _eq_s(sa, ions, bV0, rr0_A)
            vb = _eq_s(sb, ions, bV0, rr0_A)
            if va * vb < 0:
                return sa, sb
        # tight bracket failed; try hint-fraction vs s_max
        for factor in (0.9, 0.7, 0.5, 0.2, 0.05):
            sa = max(s_min, sv_hint * factor)
            va = _eq_s(sa, ions, bV0, rr0_A)
            if va * v1 < 0:
                return sa, s_max

    # Deterministic fallback: log-spaced scan from s_min to s_max.
    for sa in np.logspace(np.log10(max(s_min, 1e-8)), np.log10(s_max), 200):
        va = _eq_s(sa, ions, bV0, rr0_A)
        if va * v1 < 0:
            return sa, s_max

    return None, None   # empty pore


# ---------- public solver ----------

def f_density(ions, bV0, rr0_A, d_max_ren, sv_hint=None):
    """Solve for ion densities at the current electrochemical potentials.

    sv_hint : sv from the previous scan step, passed to _find_bracket so the
              solver tracks the same physical root across successive steps.

    Returns
    -------
    (dens, sv) : list of per-species densities [d_max_ren units] and packing density sv
    """
    s0, s1 = _find_bracket(S_MAX_VALUE, ions, bV0, rr0_A, sv_hint=sv_hint)
    if s0 is None:                      # empty pore (bracket not found)
        return [0.0] * len(ions), 0.0
    if _eq_s(s0, ions, bV0, rr0_A) * _eq_s(s1, ions, bV0, rr0_A) < 0:
        sv = brentq(lambda s: _eq_s(s, ions, bV0, rr0_A), s0, s1)
    else:
        sv = s0

    N = len(ions)
    theta = _build_theta(ions, sv, bV0, rr0_A)

    # d(det)/d(sv)
    ddelta_ds = 0.0
    for i in range(N):
        dt = np.copy(theta)
        for j in range(N):
            dt[i, j] = _d_eta(i, j, ions, sv, bV0, rr0_A) \
                       * sqrt(ions[i].area * ions[j].area) \
                       * exp(0.5 * (ions[i].elchempot + ions[j].elchempot))
        ddelta_ds += np.linalg.det(dt)

    dens = []
    for l in range(N):
        ddelta_dmu = 0.0
        for i in range(N):
            dt = np.copy(theta)
            if l == i:
                for j in range(N):
                    dt[i, j] = dt[i, j] * 0.5 if j != i else dt[i, j] + 1.0
            else:
                for j in range(N):
                    dt[i, j] = dt[i, j] * 0.5 if j == l else 0.0
            ddelta_dmu += np.linalg.det(dt)
        dens.append(-d_max_ren * ddelta_dmu / ddelta_ds)

    return dens, sv


def f_charge(dens, ions):
    """Total charge density: sum q_i * rho_i."""
    return sum(d * ion.charge for d, ion in zip(dens, ions))


def f_charge_full(ions, bV0, rr0_A, u_kBT, d_max_ren, sv_hint=None):
    """Set elchempot to u_kBT, compute density, return charge.
    sv_hint is passed to f_density for branch continuity.
    NOTE: modifies ions[*].elchempot as a side effect.
    """
    for ion in ions:
        ion.set_elchempot(u_kBT)
    dens, _ = f_density(ions, bV0, rr0_A, d_max_ren, sv_hint=sv_hint)
    return f_charge(dens, ions)
