import os, pickle, argparse
import numpy as np
import gusrn_v2 as V
OUTDIR = './risk_binding_results'
GRID = 7
N_SAMPLES = 41
LAM_SCALES = (1, 2, 5, 10, 25, 50, 100, 250, 500)
K_RANGE = tuple(range(1, 25))
ENERGY_REFERENCE = {'S07': dict(K=8, H_star=247.012, tol=0.001, fail_K=(2, 3, 4, 6)), 'S08': dict(K=8, H_star=254.633, tol=0.001, fail_K=(2, 3, 4, 6)), 'S09': dict(K=10, H_star=257.5366402466316, tol=1e-06, fail_K=(2, 3, 4, 6, 8)), 'S10': dict(K=8, H_star=185.4525639640633, tol=1e-06, fail_K=(2, 3, 4, 6))}

def density_field(seed, size=GRID):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    field = np.full((size, size), 0.05)
    for _ in range(int(rng.integers(2, 4))):
        cy, cx = rng.uniform(0, size - 1, 2)
        sigma = rng.uniform(0.9, 1.6)
        amp = rng.uniform(0.5, 1.0)
        field += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return np.clip(field, 0.0, 1.0)

def _bilinear(field, y, x):
    size = field.shape[0]
    y = min(max(y, 0.0), size - 1.0)
    x = min(max(x, 0.0), size - 1.0)
    y0, x0 = (int(np.floor(y)), int(np.floor(x)))
    y1, x1 = (min(y0 + 1, size - 1), min(x0 + 1, size - 1))
    fy, fx = (y - y0, x - x0)
    return field[y0, x0] * (1 - fy) * (1 - fx) + field[y0, x1] * (1 - fy) * fx + field[y1, x0] * fy * (1 - fx) + field[y1, x1] * fy * fx

def mean_density_along(field, p, q, n=N_SAMPLES):
    ts = np.linspace(0.0, 1.0, n)
    return float(np.mean([_bilinear(field, p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])) for t in ts]))

def exposure_matrix(I, field):
    pos, N = (I.meta['pos'], I.N)
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            R[i, j] = R[j, i] = mean_density_along(field, pos[i], pos[j])
    Dmax = float(np.max(I.D))
    R = R * np.asarray(I.D, float) / Dmax
    np.fill_diagonal(R, 0.0)
    return R

def cost_matrix(I, R):
    a1, a2, a3, a4 = V.W_DEF
    N = I.N
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            C[i, j] = a1 * (2.0 * I.D[i, j] + 10.0) + a2 * I.D[i, j] / 60.0 + a3 * R[i, j] * I.D[i, j] + a4 * 0.5 * I.E[i, j]
    return C

def make_risk_twin(I, min_gap=0.1, max_tries=5000):
    for seed in range(max_tries):
        R = exposure_matrix(I, density_field(seed))
        C = cost_matrix(I, R)
        twin = V.TourInstance('R' + I.name[1:], 'R', I.D, I.e, I.l, C, I.E, R, None, None, dict(I.meta, source=I.name, density_seed=seed, budget_mode='risk-binding'))
        tours = [(p, m) for p, m in V.enumerate_tours(twin) if m['tw_ok']]
        if len(tours) < 3:
            continue
        best = min(tours, key=lambda z: z[1]['cost'])[1]
        r_min = min((m['risk'] for _, m in tours))
        r_max = max((m['risk'] for _, m in tours))
        if r_min <= 0 or (best['risk'] - r_min) / r_min < min_gap:
            continue
        twin.BR = r_min + 0.5 * (best['risk'] - r_min)
        twin.BE = np.inf
        under_budget = [(p, m) for p, m in tours if m['risk'] <= twin.BR + 1e-12]
        constrained_opt = min(under_budget, key=lambda z: z[1]['cost'])[1]
        twin.meta.update(n_tw_feasible=len(tours), risk_min=r_min, risk_costopt=best['risk'], risk_max=r_max, n_feasible_tours=len(under_budget), risk_constrained_opt_cost=constrained_opt['cost'], risk_constrained_opt_risk=constrained_opt['risk'])
        return twin
    return None

def resource_sums(n, r):
    idx = np.arange(1 << n, dtype=np.int64)
    S = np.zeros(1 << n)
    for a in range(n):
        S += (idx >> a & 1) * r[a]
    return S

def projected_energy(E_base, S, B, K, lam_b):
    w0 = B / (2 ** K - 1)
    j = np.clip(np.rint((B - S) / w0), 0, 2 ** K - 1)
    resid = S + j * w0 - B
    return E_base + lam_b * resid ** 2

def exact_projected(pr, E_base, S, B, K, lam_b):
    Ep = projected_energy(E_base, S, B, K, lam_b)
    bi = int(np.argmin(Ep))
    Hs = float(Ep[bi])
    info = pr.tour_states.get(bi)
    out = dict(H_star=Hs, argmin_state=bi, n_arcs=pr.n, n_slack=K, n_qubits=pr.n + K, argmin_feasible=bool(info and info['feas']), argmin_cost=info['cost'] if info else None, R_star_feas=pr.R_star, n_feasible_tours=pr.n_feasible_tours)
    if pr.feas_states:
        Ef = min((float(Ep[s]) for s in pr.feas_states))
        out['min_feasible_energy'] = Ef
        out['qubo_exact'] = bool(abs(Ef - Hs) < 1e-06 and info is not None and info['feas'])
    else:
        out['qubo_exact'] = None
    if pr.R_star is not None and info is not None and info['feas'] and (info['cost'] is not None):
        out['base_cost_match'] = bool(abs(info['cost'] - pr.R_star) < 1e-06)
    else:
        out['base_cost_match'] = False
    return out

def calibrate_projected(pr, E_base, S, B, lam_base, K_range=K_RANGE, lam_scales=LAM_SCALES):
    best_miss = None
    for K in K_range:
        for sc in lam_scales:
            ex = exact_projected(pr, E_base, S, B, K, lam_base * sc)
            if ex['qubo_exact']:
                return (K, sc, ex)
            gap = abs(ex.get('min_feasible_energy', np.inf) - ex['H_star'])
            if best_miss is None or gap < best_miss['gap']:
                best_miss = dict(K=K, scale=sc, gap=gap)
    return (None, None, best_miss)

def selfcheck(outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    reg = {I.name: I for I in V.build_registry(19)}
    rows, all_ok = ([], True)
    print('SELFCHECK: slack-projected exactness vs. completed brute-force ENERGY results')
    for name, ref in ENERGY_REFERENCE.items():
        I = reg[name]
        pr = V.TourProblem(I)
        q = pr.qubo
        r = np.array([I.E[a] for a in q.arcs])
        B = I.BE
        E_base = V.all_energies(q)
        S = resource_sums(q.n, r)
        ex = exact_projected(pr, E_base, S, B, ref['K'], q.lam)
        h_ok = abs(ex['H_star'] - ref['H_star']) <= ref['tol']
        exact_ok = ex['qubo_exact'] is True
        fails_ok = all((not exact_projected(pr, E_base, S, B, K, q.lam * sc)['qubo_exact'] for K in ref['fail_K'] for sc in LAM_SCALES))
        Kmin, sc, exm = calibrate_projected(pr, E_base, S, B, q.lam)
        ok = h_ok and exact_ok and fails_ok
        all_ok &= ok
        print(f"  [{name}] n_arcs={q.n} K={ref['K']}: H*={ex['H_star']:.6f} (ref {ref['H_star']}) H*match={h_ok} exact={exact_ok} smaller-K-all-fail={fails_ok} | fine-grid minimal K={Kmin} (lam x{sc}) -> {('PASS' if ok else 'FAIL')}")
        rows.append(dict(instance=name, n_arcs=q.n, K_ref=ref['K'], qubits_ref=q.n + ref['K'], H_star_projected=ex['H_star'], H_star_reference=ref['H_star'], H_match=h_ok, exact_at_K=exact_ok, smaller_K_fail=fails_ok, K_min_fine_grid=Kmin, qubits_at_K_min=q.n + Kmin if Kmin else None, passed=ok))
    print('SELFCHECK', 'PASSED' if all_ok else 'FAILED -- do not trust the risk table until this passes')
    with open(os.path.join(outdir, 'selfcheck.pkl'), 'wb') as f:
        pickle.dump(rows, f)
    V.write_table(rows, ['instance', 'n_arcs', 'K_ref', 'qubits_ref', 'H_star_projected', 'H_star_reference', 'H_match', 'exact_at_K', 'smaller_K_fail', 'K_min_fine_grid', 'qubits_at_K_min', 'passed'], os.path.join(outdir, 'tables', 'T10_projection_selfcheck'), 'Slack-projected exactness reproduces the brute-force energy results.', 'tab:proj_check')
    return all_ok

def run_risk(outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    reg = {I.name: I for I in V.build_registry(19)}
    rows, store, skipped = ([], {}, [])
    for src in ('S07', 'S08', 'S09', 'S10'):
        twin = make_risk_twin(reg[src])
        if twin is None:
            skipped.append(src)
            print(f'  [{src}] GENERATION FAILED after {5000} density seeds at min_gap=10% -- documented limitation, not retried with a weaker criterion')
            continue
        pr = V.TourProblem(twin)
        q = pr.qubo
        r = np.array([twin.R[a] for a in q.arcs])
        B = twin.BR
        E_base = V.all_energies(q)
        S = resource_sums(q.n, r)
        bi0 = int(np.argmin(E_base))
        info0 = pr.tour_states.get(bi0)
        K, sc, ex = calibrate_projected(pr, E_base, S, B, q.lam)
        m = twin.meta
        row = dict(instance=twin.name, source=src, density_seed=m['density_seed'], n_tw_feasible=m['n_tw_feasible'], n_feasible=pr.n_feasible_tours, risk_min=round(m['risk_min'], 4), risk_costopt=round(m['risk_costopt'], 4), risk_max=round(m['risk_max'], 4), BR=round(B, 4), risk_constrained_opt_cost=round(m['risk_constrained_opt_cost'], 4), risk_constrained_opt_risk=round(m['risk_constrained_opt_risk'], 4), base_argmin_feasible=bool(info0 and info0['feas']), n_arcs=q.n, K_min=K, lam_budget_scale=sc, qubits=q.n + K if K else None, H_star=ex['H_star'] if K else None, qubo_argmin_feasible=ex.get('argmin_feasible') if K else None, qubo_argmin_base_cost=ex.get('argmin_cost') if K else None, constrained_opt_cost=pr.R_star, base_cost_match=ex.get('base_cost_match') if K else False, qubo_exact=ex.get('qubo_exact') if K else False)
        rows.append(row)
        store[twin.name] = dict(row=row, twin=twin.to_dict(), calibration=ex)
        print(f"  [{twin.name}] (from {src}, seed={m['density_seed']}) risk min/costopt/max={m['risk_min']:.3f}/{m['risk_costopt']:.3f}/{m['risk_max']:.3f} BR={B:.3f} | constrained-opt cost={pr.R_star:.4f} | " + (f"K_min={K} (lamx{sc})->{q.n + K} qubits | argmin_feasible={ex['argmin_feasible']} | argmin_base_cost={ex['argmin_cost']:.4f} | base_cost_match={ex['base_cost_match']} | qubo_exact={ex['qubo_exact']}" if K else f'NOT calibrated within K<={K_RANGE[-1]}; closest={ex}'))
    if skipped:
        print(f'\nSkipped (no twin generated, documented not hidden): {skipped}')
    with open(os.path.join(outdir, 'risk_results.pkl'), 'wb') as f:
        pickle.dump(dict(rows=rows, skipped=skipped, store=store), f)
    V.write_table(rows, ['instance', 'source', 'density_seed', 'n_tw_feasible', 'n_feasible', 'risk_min', 'risk_costopt', 'risk_max', 'BR', 'risk_constrained_opt_cost', 'risk_constrained_opt_risk', 'base_argmin_feasible', 'n_arcs', 'K_min', 'lam_budget_scale', 'qubits', 'qubo_argmin_feasible', 'qubo_argmin_base_cost', 'constrained_opt_cost', 'base_cost_match', 'qubo_exact'], os.path.join(outdir, 'tables', 'T11_risk_binding_calibration'), 'Risk-binding twins: projected-slack calibration vs. the exact constrained routing optimum.', 'tab:risk_bind')
    return (rows, skipped)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['selfcheck', 'risk', 'all'])
    ap.add_argument('--outdir', default=OUTDIR)
    a, _ = ap.parse_known_args()
    if a.stage in ('selfcheck', 'all'):
        selfcheck(a.outdir)
    if a.stage in ('risk', 'all'):
        run_risk(a.outdir)
if __name__ == '__main__':
    main()
