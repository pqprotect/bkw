"""Numerically verify the fresh-sample and amplified quantum-walk BKW constants.

Requires Python 3 and mpmath. No local data files or network access are used.
Run: python3 verify_walk.py
For a finer convergence check, use --degree 48 --quadrature 110 --grade 7 --dps 56.
The numerical method evaluates the equations in the paper; it does not establish
its heuristic list assumptions.
"""
import argparse
import time

import mpmath as mp


def chebyshev_quadrature(degree):
    """Lobatto nodes and weights for 1 and log(1-v) on [0,1]."""
    one = mp.mpf(1)
    half = one / 2
    cosine = [mp.cos(mp.pi * j / degree) for j in range(2 * degree)]
    endpoint = [half] + [one] * (degree - 1) + [half]
    plain = [one / (1 - j*j) if j % 2 == 0 else mp.mpf(0)
             for j in range(degree + 1)]
    aux = [mp.mpf(0), mp.mpf(2)]
    for j in range(2, degree + 2):
        aux.append(aux[j - 2] + (4 * one / (j - 1) if j % 2 == 0
                               else 2 * (one / j + one / (j - 2))))
    logs = [-one, -half] + [(aux[j - 1] / (j - 1) - aux[j + 1] / (j + 1)) / 4
                            for j in range(2, degree + 1)]

    def weights(moment):
        return [2 * endpoint[i] / degree * mp.fsum(
            endpoint[j] * moment[j] * cosine[(i*j) % (2*degree)]
            for j in range(degree + 1)) for i in range(degree + 1)]

    nodes = [one] + [(1 + cosine[j]) / 2 for j in range(1, degree)] + [mp.mpf(0)]
    return nodes, weights(plain), weights(logs)


def solve(degree, quadrature, grade, dps, coordinate="x", check=False,
          quadrature_rule="chebyshev", clock_check=False):
    if quadrature < degree + 2*grade - 1:
        raise ValueError("quadrature degree must cover the first moment of the interpolant")
    mp.mp.dps = dps
    zero, one = mp.mpf(0), mp.mpf(1)
    two_thirds, four_thirds = 2 * one / 3, 4 * one / 3
    B0 = 2 * mp.log(2)
    gaussian = quadrature_rule == "gauss"
    if gaussian:
        nodes, weights = mp.gauss_quadrature(quadrature + 1, "legendre")
        v, plain = [(1 + t) / 2 for t in nodes], [w / 2 for w in weights]
        logs = [zero] * (quadrature + 1)
    else:
        v, plain, logs = chebyshev_quadrature(quadrature)
    z = [zero] + [(1 - mp.cos(mp.pi*j/degree)) / 2
                  for j in range(1, degree)] + [one]
    logarithmic_time = coordinate == "log-x"
    extent = mp.log(four_thirds) if logarithmic_time else one / 3
    times = [extent * t**grade for t in z]
    x = [mp.exp(t) if logarithmic_time else 1 + t for t in times]
    diagonal_scale = [one] * len(z) if logarithmic_time else x
    barycentric = [mp.mpf((-1)**j) / (2 if j in (0, degree) else 1)
                   for j in range(degree + 1)]
    n = degree + 1
    v_power = [t**grade for t in v]
    jacobian = [grade * t**(grade - 1) for t in v]
    regular = [mp.log(mp.fsum(t**i for i in range(grade))) for t in v]

    def basis(t):
        differences = [t - u for u in z]
        for j, value in enumerate(differences):
            if value == 0:
                return [one if i == j else zero for i in range(n)]
        raw = [b / d for b, d in zip(barycentric, differences)]
        total = mp.fsum(raw)
        return [b / total for b in raw]

    if gaussian:
        # Independent log weights in the ordinary power basis. The exact
        # moments are int_0^1 v^(m+p-1) log(1-v) dv = -H_(m+p)/(m+p).
        power_basis = []
        for j in range(n):
            polynomial = [one]
            divisor = one
            for k in range(n):
                if k == j:
                    continue
                polynomial = [(-z[k] * polynomial[i] if i < len(polynomial) else zero)
                              + (polynomial[i-1] if i else zero)
                              for i in range(len(polynomial) + 1)]
                divisor *= z[j] - z[k]
            power_basis.append([c / divisor for c in polynomial])
        power_moments = [-grade * mp.harmonic(m + grade)/(m + grade) for m in range(n)]

    def parameters(k):
        a = (3*k - 2) / (k + 2)
        a_prime = 8 / (k + 2)**2
        beta = 2 * (mp.log(k) - (k - 1) * mp.log(4*(k-1)/k)) / (k + 2)
        cap0 = mp.log(4 * (k - 1)) - two_thirds * mp.log(k)
        return a, a_prime, beta, cap0

    k0 = mp.findroot(lambda k: parameters(k)[3], (mp.mpf('1.28'), mp.mpf('1.31')))
    n0 = B0 / parameters(k0)[2]
    density = [n0 / (1 + 8*(s - 1)) for s in x]
    cap = [k0 + mp.mpf('.36')*(s - 1) for s in x]
    starts = time.monotonic()
    rows = [None]

    for i in range(1, n):
        # Divide history integrals by x_i-1 before constructing the rows.
        # This also normalizes the cap residual at the start.
        samples = [basis(z[i] * t) for t in v]
        weighted = [[plain[l] * jacobian[l] * value for value in samples[l]]
                    for l in range(quadrature + 1)]
        if logarithmic_time:
            gap = [times[i] * (1-t) for t in v_power]
            w = [-mp.expm1(-t) for t in gap]
            regular_i = [base + (mp.log(distance/t) if t else zero)
                         for base, distance, t in zip(regular, w, gap)]
        else:
            w = [times[i] / x[i] * (1-t) for t in v_power]
            regular_i = regular
        log_weights = [jacobian[l] * (logs[l] + plain[l] * regular_i[l])
                       for l in range(quadrature + 1)]
        mass = [mp.fsum(row[j] for row in weighted) for j in range(n)]
        log_row = [mp.fdot(log_weights, (row[j] for row in samples)) for j in range(n)]
        if gaussian:
            target_weights = [z[i]**m * power_moments[m] for m in range(n)]
            log_row = [current + mp.fdot(polynomial, target_weights)
                       for current, polynomial in zip(log_row, power_basis)]
        rows.append((weighted, mass, log_row, w))

    for iteration in range(12):
        residual = mp.matrix(2*n, 1)
        matrix = mp.matrix(2*n)
        matrix[0, 0] = matrix[n, n] = 1
        residual[0], residual[n] = density[0] - n0, cap[0] - k0
        for i in range(1, n):
            weighted, mass, log_row, w = rows[i]
            s, k = diagonal_scale[i], cap[i]
            a, a_prime, beta, cap0 = parameters(k)
            delta = times[i]
            q = a * (k - 1)
            log_scale = mp.log(delta / (s*q))
            mass_at_s = mp.fdot(mass, density)
            beta_prime = -6 * cap0 / (k + 2)**2
            q_rel_prime = a_prime / a + 1 / (k - 1)
            residual[i] = s * beta * density[i] - B0 - delta * (
                mp.fdot(log_row, density) + log_scale * mass_at_s)
            matrix[i, n+i] = s * beta_prime * density[i] + delta * q_rel_prime * mass_at_s
            log_cap = [cap0 - four_thirds * mp.log1p(t/a) for t in w]
            for j in range(n):
                matrix[i, j] = (s * beta if i == j else zero) - delta * (
                    log_row[j] + log_scale * mass[j])
                matrix[n+i, j] = mp.fdot(log_cap, (row[j] for row in weighted))
            residual[n+i] = mp.fdot((matrix[n+i, j] for j in range(n)), density)
            cap_prime0 = 1 / (k - 1) - two_thirds / k
            matrix[n+i, n+i] = cap_prime0 * mass_at_s + four_thirds * a_prime / a * mp.fsum(
                t/(a+t) * mp.fdot(row, density) for t, row in zip(w, weighted) if t)
        update = mp.lu_solve(matrix, residual)
        for i in range(n):
            density[i] -= update[i]
            cap[i] -= update[n+i]
        error = max(abs(t) for t in update)
        if error < mp.mpf(10)**(-dps + 8):
            break
    else:
        raise ArithmeticError("Newton iteration did not converge")

    def moment(z_end, order):
        total = zero
        for t, weight, d in zip(v, plain, jacobian):
            local = z_end * t
            t_physical = extent * local**grade
            x_physical = mp.exp(t_physical) if logarithmic_time else 1 + t_physical
            total += weight * d * x_physical**order * mp.fdot(basis(local), density)
        return total * extent * z_end**grade

    mass = moment(one, 0)
    coded = 3 * moment(one, 1) / 2
    fresh = 1 / (mass + (2 - coded) / 2)
    z_amp = mp.findroot(lambda t: 3 * moment(t, 1) / 2 - 1,
                        (mp.mpf('.8'), mp.mpf('.95')))
    amp_mass = moment(z_amp, 0)
    amplified = 1 / amp_mass
    cap_margin = min((k - 1) * parameters(k)[0] - (1 - 1/s)
                     for k, s in zip(cap, x))
    def evaluate_off_grid(t):
        tau = extent * t**grade
        row = basis(t)
        n_t = mp.fdot(row, density)
        k_t = mp.fdot(row, cap)
        a, _, beta, cap0 = parameters(k_t)
        q = a * (k_t - 1)
        s = one if logarithmic_time else 1 + tau

        def values(v):
            local = t*v
            age = tau * (1-v**grade)
            w = -mp.expm1(-age) if logarithmic_time else age/s
            weight = grade * tau * v**(grade-1) * mp.fdot(basis(local), density)
            return w, weight

        def cap_function(v):
            w, weight = values(v)
            return weight * (cap0 - four_thirds * mp.log1p(w/a))

        def log_function(v):
            w, weight = values(v)
            return weight * mp.log(w/q) if w else zero

        cap_error = mp.quad(cap_function, [0, .5, 1]) / tau
        density_error = s * beta * n_t - B0 - mp.quad(log_function, [0, .5, 1])
        return max(abs(cap_error), abs(density_error))

    def backward_clock(endpoint):
        # A separate linear solve checks both clock moments. Symmetric grading
        # at both ends handles the two endpoint logarithmic derivatives.
        coeff = [(-1)**j * mp.binomial(grade-1+j, j)
                 * mp.binomial(2*grade-1, grade-1-j) for j in range(grade)]
        norm = mp.factorial(2*grade-1) / mp.factorial(grade-1)**2

        def phi(t):
            if t > mp.mpf('.5'):
                return 1 - phi(1-t)
            return t**grade * mp.fsum(c*t**j for j, c in enumerate(coeff))

        def derivative(t):
            return norm * (t*(1-t))**(grade-1)

        def price(s):
            elapsed = mp.log(s) if logarithmic_time else s-1
            loc = (elapsed/extent)**(one/grade)
            return mp.fdot(basis(loc), cap)

        x_end = endpoint-1
        backward_nodes, back_weights, back_logs = chebyshev_quadrature(quadrature)
        # Reverse weights so that the singularity is log u, not log(1-u).
        back_logs.reverse()
        wmat = mp.matrix(n)
        for i, t in enumerate(z):
            x_here = 1 + x_end*phi(t)
            beta_here = parameters(price(x_here))[2]
            wmat[i, i] = beta_here
            if t == 1:
                continue
            for v, weight, log_weight in zip(backward_nodes, back_weights, back_logs):
                y = t+(1-t)*v
                jac = x_end*(1-t)*derivative(y)
                if not jac:
                    continue
                s = 1+x_end*phi(y)
                a_s = parameters(price(s))[0]
                order = grade if t == 0 else 1
                if t == 0:
                    regular_gap = x_end * mp.fsum(c*v**j for j,c in enumerate(coeff))
                else:
                    regular_gap = ((s-x_here)/v if v
                                   else x_end*(1-t)*derivative(t))
                regular_log = mp.log(4*regular_gap/((1+a_s)*s-x_here))
                factor = jac/s*(weight*regular_log+order*log_weight)
                for j, val in enumerate(basis(y)):
                    wmat[i, j] -= factor*val
        values = mp.lu_solve(wmat, mp.matrix([B0]*n))
        depth_from_clock, mass_from_clock = zero, zero
        for t, weight in zip(backward_nodes, back_weights):
            s = 1 + x_end*phi(t)
            val = x_end*derivative(t)*weight*mp.fdot(basis(t), values)
            depth_from_clock += val/s
            mass_from_clock += val
        return dict(coded_depth=str(3*depth_from_clock/2),
                    mass=str(mass_from_clock/endpoint),
                    min_sampled_clock=str(min(values)))

    endpoint_tau = extent * z_amp**grade
    endpoint = mp.exp(endpoint_tau) if logarithmic_time else 1 + endpoint_tau
    result = dict(degree=degree, quadrature=quadrature, grade=grade, dps=dps,
                coordinate=coordinate, quadrature_rule=quadrature_rule, fresh=str(fresh), amplified=str(amplified), fresh_coded_depth=str(coded),
                fresh_mass=str(mass), amplified_mass=str(amp_mass),
                amplified_endpoint=str(endpoint),
                start_u=str(1/k0), start_beta=str(1/n0),
                final_u=str(1/cap[-1]), min_sampled_clip_margin=str(cap_margin),
                newton_iterations=iteration + 1, last_newton_step=str(error),
                elapsed_seconds=time.monotonic() - starts)
    if clock_check:
        result["fresh_backward_clock"] = backward_clock(four_thirds)
        result["amplified_backward_clock"] = backward_clock(endpoint)
    if check:
        test_nodes = [mp.mpf(t) for t in (".13", ".31", ".57", ".83", ".96")]
        result["sampled_direct_quadrature_residual"] = str(max(
            evaluate_off_grid(t) for t in test_nodes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degree", type=int, default=24)
    parser.add_argument("--quadrature", type=int, default=56)
    parser.add_argument("--grade", type=int, default=5)
    parser.add_argument("--dps", type=int, default=42)
    parser.add_argument("--no-checks", action="store_true",
                        help="skip off-grid residual and independent backward-clock checks")
    args = parser.parse_args()
    result = solve(args.degree, args.quadrature, args.grade, args.dps,
                   check=not args.no_checks, clock_check=not args.no_checks)
    targets = {"fresh": "0.864269772", "amplified": "1.597871236"}
    print("Quantum BKW with reusable walks")
    passed = True
    for key, expected in targets.items():
        value = mp.mpf(result[key])
        matches = abs(value - mp.mpf(expected)) < mp.mpf("5e-9")
        passed &= matches
        print(f"  {key:10s}: {mp.nstr(value, 18)}  "
              f"({'matches' if matches else 'DIFFERS from'} paper value {expected})")
    print(f"  coded depth: {mp.nstr(mp.mpf(result['fresh_coded_depth']), 18)}")
    print(f"  amplified endpoint: {mp.nstr(mp.mpf(result['amplified_endpoint']), 18)}")
    print(f"  minimum sampled active-cap margin: {result['min_sampled_clip_margin']}")
    if not args.no_checks:
        residual = mp.mpf(result["sampled_direct_quadrature_residual"])
        print(f"  off-grid equation residual: {mp.nstr(residual, 6)}")
        passed &= residual < mp.mpf("1e-7")
        for key, expected_depth, expected_mass in [
            ("fresh_backward_clock", mp.mpf(result["fresh_coded_depth"]),
             mp.mpf(result["fresh_mass"])),
            ("amplified_backward_clock", mp.mpf(1),
             mp.mpf(result["amplified_mass"])),
        ]:
            clock = result[key]
            error = max(abs(mp.mpf(clock["coded_depth"]) - expected_depth),
                        abs(mp.mpf(clock["mass"]) - expected_mass))
            print(f"  {key} moment error: {mp.nstr(error, 6)}")
            passed &= error < mp.mpf("1e-7") and mp.mpf(clock["min_sampled_clock"]) > 0
    passed &= mp.mpf(result["min_sampled_clip_margin"]) > 0
    print("PASS" if passed else "CHECK FAILED: refine the numerical resolution")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
