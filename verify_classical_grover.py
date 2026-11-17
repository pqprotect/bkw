#!/usr/bin/env python3
"""Recompute classical/Grover BKW coefficients from the schedule equations.

Python 3 with NumPy, SciPy and mpmath is required. Run, for example::

    python3 verify_classical_grover.py --panels 32
    python3 verify_classical_grover.py --panels 64 --model classical

No data files or other project modules are used, and no files are written.
The local-polynomial method integrates the diagonal logarithm analytically
and splits at the cap boundary. Polynomial moments are built at 80 decimal
places; the subsequent schedule calculation uses double precision. Degree
8, grading 4, and 64-point Gauss quadrature are used throughout.

The carry correction is evaluated on 501 round positions. Agreement with
the paper is a numerical consistency check, not a rigorous error bound or
a proof of the assumptions on intermediate lists.
"""

import argparse
import time

import mpmath as mp
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import solve
from scipy.optimize import brentq, root
from scipy.special import lqmn, roots_legendre

LN2 = np.log(2.0)
RHO = 0.75
B = 2 * LN2

def nodes_weights(p):
    """Polynomial integrals and endpoint logarithmic moments, formed at 80 dps."""
    mp.mp.dps = 80
    xx = [(1 - mp.cos(mp.pi * k / p)) / 2 for k in range(p + 1)]
    polys = []
    for j in range(p + 1):
        co = [mp.mpf(1)]
        den = mp.mpf(1)
        for k in range(p + 1):
            if k == j:
                continue
            nxt = [mp.mpf(0)] * (len(co) + 1)
            for i, a in enumerate(co):
                nxt[i] -= xx[k] * a
                nxt[i + 1] += a
            co = nxt
            den *= xx[j] - xx[k]
        polys.append([v / den for v in co])
    basic = []
    logw = []
    for r in xx[1:]:
        basic.append([sum((co[k] * r ** k / (k + 1) for k in range(p + 1))) for co in polys])
        logw.append([-sum((co[k] * r ** k * mp.harmonic(k + 1) / (k + 1) for k in range(p + 1))) for co in polys])
    return (np.array(xx, float), np.array(basic, float), np.array(logw, float))

def classical_beta(k):
    return (k * np.log(k) - (k - 1) * np.log(4 * (k - 1))) / (2 * LN2)

def classical_beta_prime(k):
    return np.log(k / (4 * (k - 1))) / (2 * LN2)

def classical_kernels(r, k, need_phi=True):
    ac = r >= k * (2 - k)
    xi = np.where(ac, np.log(k * (k - r) / (4 * (k - 1) ** 2)), np.log((2 * k - r) ** 2 / (4 * k * (k - r)))) / (2 * LN2)
    ph = k * xi + np.where(ac, -np.log1p(-r) / 2 + np.log(k - 1), 0) / LN2 if need_phi else np.zeros_like(r)
    xk = np.where(ac, 1 / k + 1 / (k - r) - 2 / (k - 1), 4 / (2 * k - r) - 1 / k - 1 / (k - r)) / (2 * LN2)
    pk = xi + k * xk + np.where(ac, 1 / (k - 1), 0) / LN2
    return (xi, ph, xk, pk)

class PanelSolver:

    def __init__(self, N=32, p=8, grade=4, gauss=64):
        self.N = N
        self.p = p
        self.gauss = gauss
        self.grade = grade
        self.xx, self.basic, self.logw = nodes_weights(p)
        self.qr, self.qw = roots_legendre(gauss)
        self.qr = (self.qr + 1) / 2
        self.qw = self.qw / 2
        self.bary = BarycentricInterpolator(self.xx, np.eye(p + 1), axis=0)
        self.leginv = np.linalg.inv(np.polynomial.legendre.legvander(2 * self.xx - 1, p))
        self.logcache = {}
        x = np.arange(N + 1) / N
        self.cuts = RHO + (1 - RHO) * x ** grade / (x ** grade + (1 - x) ** grade)
        self.h = np.diff(self.cuts)
        self.times = self.cuts[:-1, None] + self.h[:, None] * self.xx
        self.nn = np.zeros((N, p + 1))
        self.kk = self.nn.copy()
        self.mres = 0.0
        self.capres = 0.0

    def logweights(self, d):
        """Integral on [0,1] of Lagrange basis times log(d-t), d>=1."""
        key = float(d)
        if key in self.logcache:
            return self.logcache[key]
        if d > 10:
            t = self.qr
            L = self.bary(t)
            val = np.full_like(t, np.log(d))
            z = t / d
            powz = z.copy()
            for j in range(1, 25):
                val -= powz / j
                powz *= z
            out = np.einsum('q,qk->k', self.qw * val, L)
        else:
            mom = np.zeros(self.p + 1)
            if abs(d - 1) < 1e-13:
                mom[0] = -1
                for j in range(1, self.p + 1):
                    mom[j] = -1 / (j * (j + 1))
            else:
                D = 2 * d - 1
                Q = lqmn(0, self.p + 1, D)[0][0]
                mom[0] = d * np.log(d) - (d - 1) * np.log(d - 1) - 1
                for j in range(1, self.p + 1):
                    mom[j] = (Q[j + 1] - Q[j - 1]) / (2 * j + 1)
            out = mom @ self.leginv
        self.logcache[key] = out
        return out

    def old_integrals(self, panel, u, k):
        if panel == 0:
            return np.zeros(4)
        a = self.cuts[:panel]
        b = self.cuts[1:panel + 1]
        cap = u * k * (2 - k)
        mid = np.maximum(a, np.minimum(b, cap))
        sums = np.zeros(4)
        for lo, hi in [(a, mid), (mid, b)]:
            z = lo[:, None] + (hi - lo)[:, None] * self.qr
            tt = (z - a[:, None]) / self.h[:panel, None]
            L = self.bary(tt)
            n = np.einsum('iqk,ik->iq', L, self.nn[:panel])
            xi, ph, xk, pk = classical_kernels(z / u, k)
            ph += np.log(u - z) / (2 * LN2)
            weight = (hi - lo)[:, None] * self.qw * n
            sums += np.array([np.sum(weight * v) for v in [xi, ph, xk, pk]])
        for j in range(panel):
            lw = self.h[j] * (np.log(self.h[j]) * self.basic[-1] + self.logweights((u - self.cuts[j]) / self.h[j]))
            sums[1] -= lw @ self.nn[j] / (2 * LN2)
        return sums

    def current_weights(self, panel, j, k):
        r = self.xx[j + 1]
        u = self.times[panel, j + 1]
        ell = self.h[panel] * r
        z = self.cuts[panel] + ell * self.qr
        L = self.bary(r * self.qr)
        xi, _, xk, pk = classical_kernels(z / u, k, False)
        reg = (0.5 * np.log(u) + np.log(k - 1)) / LN2 + k * xi
        w0 = ell * np.einsum('q,qk->k', self.qw * xi, L)
        w1 = ell * np.einsum('q,qk->k', self.qw * reg, L) - ell * (np.log(ell) * self.basic[j] + self.logw[j]) / (2 * LN2)
        w2 = ell * np.einsum('q,qk->k', self.qw * xk, L)
        w3 = ell * np.einsum('q,qk->k', self.qw * pk, L)
        return (w0, w1, w2, w3)

    def forward(self):
        n0 = 1 / (RHO * classical_beta(4 / 3))
        k0 = 4 / 3
        for panel in range(self.N):
            startn = n0 if panel == 0 else self.nn[panel - 1, -1]
            startk = k0 if panel == 0 else self.kk[panel - 1, -1]

            def fun(y, jac=False):
                ns = np.r_[startn, y[:self.p]]
                ks = y[self.p:]
                F = np.empty(2 * self.p)
                J = np.zeros((2 * self.p, 2 * self.p))
                for j, k in enumerate(ks):
                    u = self.times[panel, j + 1]
                    scale = u - RHO
                    xi, ph, xk, pk = self.current_weights(panel, j, k)
                    ox, op, oxk, opk = self.old_integrals(panel, u, k)
                    F[j] = u * classical_beta(k) * ns[j + 1] + op + ph @ ns - 1
                    F[self.p + j] = (ox + xi @ ns) / scale
                    J[j, :self.p] = ph[1:]
                    J[j, j] += u * classical_beta(k)
                    J[self.p + j, :self.p] = xi[1:] / scale
                    J[j, self.p + j] = u * classical_beta_prime(k) * ns[j + 1] + opk + pk @ ns
                    J[self.p + j, self.p + j] = (oxk + xk @ ns) / scale
                return J if jac else F
            guess = np.r_[np.full(self.p, startn), startk + 0.5 * (self.times[panel, 1:] - self.cuts[panel])]
            sol = root(fun, guess, jac=lambda y: fun(y, True), tol=1e-10)
            for refine in range(4):
                ff = fun(sol.x)
                if np.max(np.abs(ff)) < 2e-15:
                    break
                delta = solve(fun(sol.x, True), ff)
                sol.x -= delta
            err = fun(sol.x)
            if not sol.success and np.max(np.abs(err)) > 2e-10:
                raise RuntimeError((panel, sol.message, np.max(abs(err))))
            self.nn[panel] = np.r_[startn, sol.x[:self.p]]
            self.kk[panel] = np.r_[startk, sol.x[self.p:]]
            self.mres = max(self.mres, float(np.max(abs(err[:self.p]))))
            self.capres = max(self.capres, float(np.max(abs(err[self.p:]))))

def grover_beta(a):
    c = 1 - a
    return (-c * np.log(4) - a * np.log(a) - c * np.log(c)) / B

def grover_beta_prime(a):
    return np.log(4 * (1 - a) / a) / B

def grover_kernels(r, a):
    c = 1 - a
    t = 1 - r
    active = t <= 2 * c * c / a
    den = t + 2 * c
    b = 2 - a
    d = np.where(active, t / den, (b - r) / (2 * b - r))
    h = np.where(active, np.log(16 * c ** 3 / (a * den ** 2)), np.log(4 * d * (1 - d))) / B
    hd = np.where(active, -3 / c - 1 / a + 4 / den, -r * r / (b * (b - r) * (2 * b - r))) / B
    ell_a = np.where(active, 2 / den, -r / ((b - r) * (2 * b - r))) / B
    u = np.minimum(1, d / c)
    mu = -np.log(4 * d * (1 - d)) / B
    w = -np.log((u - d) / (1 - d)) / B
    C = -np.log(u) / B
    phi = (2 - a) * mu + a * w - C
    phid = h - r * ell_a
    return (h, phi, hd, phid)

class GroverPanels(PanelSolver):

    def old_integrals(self, panel, u, a):
        if panel == 0:
            return np.zeros(4)
        lo0 = self.cuts[:panel]
        hi0 = self.cuts[1:panel + 1]
        threshold = u * (1 - 2 * (1 - a) ** 2 / a)
        mid = np.maximum(lo0, np.minimum(hi0, threshold))
        out = np.zeros(4)
        for lo, hi in [(lo0, mid), (mid, hi0)]:
            z = lo[:, None] + (hi - lo)[:, None] * self.qr
            L = self.bary((z - lo0[:, None]) / self.h[:panel, None])
            density = np.einsum('iqk,ik->iq', L, self.nn[:panel])
            h, phi, ha, phia = grover_kernels(z / u, a)
            phi += np.log(u - z) / B
            weights = (hi - lo)[:, None] * self.qw * density
            out += np.array([np.sum(weights * v) for v in [h, phi, ha, phia]])
        for j in range(panel):
            lw = self.h[j] * (np.log(self.h[j]) * self.basic[-1] + self.logweights((u - self.cuts[j]) / self.h[j]))
            out[1] -= lw @ self.nn[j] / B
        return out

    def current_weights(self, panel, j, a):
        rnode = self.xx[j + 1]
        u = self.times[panel, j + 1]
        length = self.h[panel] * rnode
        z = self.cuts[panel] + length * self.qr
        L = self.bary(rnode * self.qr)
        r = z / u
        c = 1 - a
        t = 1 - r
        h, _, ha, phia = grover_kernels(r, a)
        regular = ((1 + 2 * c) * np.log(t + 2 * c) - 2 * c * np.log(2 * c) - (1 + c) * np.log(4) - a * np.log(a) - c * np.log(c) + np.log(u)) / B
        w0 = length * np.einsum('q,qk->k', self.qw * h, L)
        w1 = length * np.einsum('q,qk->k', self.qw * regular, L) - length * (np.log(length) * self.basic[j] + self.logw[j]) / B
        w2 = length * np.einsum('q,qk->k', self.qw * ha, L)
        w3 = length * np.einsum('q,qk->k', self.qw * phia, L)
        return (w0, w1, w2, w3)

    def forward(self):
        n0 = 1 / (RHO * grover_beta(0.8))
        a0 = 0.8
        for panel in range(self.N):
            startn = n0 if panel == 0 else self.nn[panel - 1, -1]
            starta = a0 if panel == 0 else self.kk[panel - 1, -1]

            def fun(y, jac=False):
                ns = np.r_[startn, y[:self.p]]
                caps = y[self.p:]
                F = np.empty(2 * self.p)
                J = np.zeros((2 * self.p, 2 * self.p))
                for j, a in enumerate(caps):
                    u = self.times[panel, j + 1]
                    scale = u - RHO
                    h, phi, ha, phia = self.current_weights(panel, j, a)
                    oh, op, oha, opa = self.old_integrals(panel, u, a)
                    F[j] = u * grover_beta(a) * ns[j + 1] + op + phi @ ns - 1
                    F[self.p + j] = (oh + h @ ns) / scale
                    J[j, :self.p] = phi[1:]
                    J[j, j] += u * grover_beta(a)
                    J[self.p + j, :self.p] = h[1:] / scale
                    J[j, self.p + j] = u * grover_beta_prime(a) * ns[j + 1] + opa + phia @ ns
                    J[self.p + j, self.p + j] = (oha + ha @ ns) / scale
                return J if jac else F
            guess = np.r_[np.full(self.p, startn), starta - 0.4 * (self.times[panel, 1:] - self.cuts[panel])]
            sol = root(fun, guess, jac=lambda y: fun(y, True), tol=1e-10)
            for refine in range(4):
                ff = fun(sol.x)
                if np.max(abs(ff)) < 3e-15:
                    break
                sol.x -= solve(fun(sol.x, True), ff)
            residual = fun(sol.x)
            self.mres = max(self.mres, float(np.max(abs(residual[:self.p]))))
            self.capres = max(self.capres, float(np.max(abs(residual[self.p:]))))
            if np.max(abs(residual)) > 2e-09:
                raise RuntimeError((panel, np.max(abs(residual))))
            self.nn[panel] = np.r_[startn, sol.x[:self.p]]
            self.kk[panel] = np.r_[starta, sol.x[self.p:]]

REFERENCE = {
    "classical": {"fresh": 0.888251733, "amplified": 1.635883551},
    "grover": {"fresh": 0.873673376, "amplified": 1.612982610},
}


def verify_model(model, panels):
    """Solve once, then evaluate both the depth-two and depth-one budgets."""
    started = time.perf_counter()
    solver_type = PanelSolver if model == "classical" else GroverPanels
    sol = solver_type(N=panels, p=8, grade=4, gauss=64)
    sol.forward()
    # Solver coordinate v=3x/4: N(x) dx=n(v) dv and alpha=2 integral v*n(v) dv.
    polys = [Polynomial.fit(sol.xx, row, deg=sol.p,
                            domain=[0, 1]).convert() for row in sol.nn]
    mass_integrals = [p.integ() for p in polys]
    depth_integrals = [
        (p * Polynomial([a, h])).integ()
        for p, a, h in zip(polys, sol.cuts, sol.h)
    ]
    masses = np.array([
        h * (p(1) - p(0)) for h, p in zip(sol.h, mass_integrals)
    ])
    depths = np.array([
        2 * h * (p(1) - p(0)) for h, p in zip(sol.h, depth_integrals)
    ])
    cumulative = np.r_[0, np.cumsum(depths)]

    def overhead(v):
        if v <= RHO:
            return 0.0
        index = min(np.searchsorted(sol.cuts, v, side="right") - 1,
                    panels - 1)
        parameter = float(sol.bary((v - sol.cuts[index]) / sol.h[index])
                          @ sol.kk[index])
        cuts = list(sol.cuts[:index + 1]) + [v]
        if model == "classical":
            cap = v * parameter * (2 - parameter)
        else:
            cap = v * (1 - 2 * (1 - parameter)**2 / parameter)
        if RHO < cap < v:
            cuts.append(cap)
        total = 0.0
        cuts = sorted(set(cuts))
        for low, high in zip(cuts[:-1], cuts[1:]):
            if high <= low:
                continue
            points = low + (high - low) * sol.qr
            j = min(np.searchsorted(sol.cuts, (low + high) / 2,
                                    side="right") - 1, panels - 1)
            density = sol.bary((points - sol.cuts[j]) / sol.h[j]) @ sol.nn[j]
            ratio = points / v
            if model == "classical":
                cost = np.maximum(0, np.log(parameter * (parameter - 1)
                                           / (parameter - ratio)) / (2 * LN2))
            else:
                d = np.where(
                    ratio >= 1 - 2 * (1 - parameter)**2 / parameter,
                    (1 - ratio) / (3 - 2 * parameter - ratio),
                    (2 - parameter - ratio) / (4 - 2 * parameter - ratio),
                )
                cost = np.maximum(0, np.log((1 - d) / parameter) / (2 * LN2))
            total += (high - low) * (sol.qw @ (density * cost))
        return float(total)

    rows = []
    for regime, budget in [("fresh", 2.0), ("amplified", 1.0)]:
        if cumulative[-1] <= budget:
            vend, mass, depth = 1.0, float(np.sum(masses)), float(cumulative[-1])
        else:
            j = min(np.searchsorted(cumulative, budget, side="right") - 1,
                    panels - 1)
            z = brentq(
                lambda z: cumulative[j]
                + 2 * sol.h[j] * (depth_integrals[j](z) - depth_integrals[j](0))
                - budget, 0, 1, xtol=2e-14,
            )
            vend = sol.cuts[j] + sol.h[j] * z
            mass = float(np.sum(masses[:j]) + sol.h[j]
                         * (mass_integrals[j](z) - mass_integrals[j](0)))
            depth = budget
        overheads = np.array([overhead(v) for v in np.linspace(RHO, vend, 501)])
        carry = float(np.max(overheads
                             - np.minimum.accumulate(overheads[::-1])[::-1]))
        dimension = (budget - depth) / 2 + mass
        coefficient = (1 + carry) / dimension
        expected = REFERENCE[model][regime]
        agrees = abs(coefficient - expected) <= 5e-9
        rows.append((regime, coefficient, vend / RHO, depth, carry, agrees))
    return rows, sol.mres, sol.capres, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", type=int, default=32,
                        help="number of local polynomial panels (default: 32; use 64 to refine)")
    parser.add_argument("--model", choices=["classical", "grover", "all"],
                        default="all", help="algorithm to check (default: all)")
    args = parser.parse_args()
    if args.panels < 4:
        parser.error("--panels must be at least 4")
    passed = True
    for model in (["classical", "grover"] if args.model == "all" else [args.model]):
        print(f"{model.capitalize()}: {args.panels} panels", flush=True)
        rows, forward_residual, cap_residual, elapsed = verify_model(model, args.panels)
        for regime, coefficient, endpoint, depth, carry, agrees in rows:
            print(f"  {regime:9s} a={coefficient:.12f}  X={endpoint:.10f}"
                  f"  coded_depth={depth:.10f}  sampled_carry={carry:.3e}"
                  f"  {'PASS' if agrees else 'MISMATCH'}", flush=True)
            passed &= agrees
        print(f"  forward residual={forward_residual:.3e};"
              f" cap residual={cap_residual:.3e}; {elapsed:.2f} s", flush=True)
    print("Comparison tolerance: 5e-9 against the paper's nine-decimal coefficients.")
    print("Numerical agreement does not certify the heuristic list assumptions.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
