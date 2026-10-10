import os, pickle, argparse
import numpy as np
import gusrn_v2 as V
OUTDIR = './csweep_transfer_results'
os.makedirs(OUTDIR, exist_ok=True)
WEIGHT_CONFIGS = {'baseline': (0.3, 0.2, 0.25, 0.25), 'travel-emph': (0.55, 0.15, 0.15, 0.15), 'time-emph': (0.15, 0.55, 0.15, 0.15), 'risk-emph': (0.15, 0.15, 0.55, 0.15), 'energy-emph': (0.15, 0.15, 0.15, 0.55)}
LAM_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)

def reweight_instance(I, weights, tag):
    a1, a2, a3, a4 = weights
    N = I.N
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            C[i, j] = a1 * (2.0 * I.D[i, j] + 10.0) + a2 * I.D[i, j] / 60.0 + a3 * I.R[i, j] * I.D[i, j] + a4 * 0.5 * I.E[i, j]
    return V.TourInstance(f'{I.name}_{tag}', I.family, I.D, I.e, I.l, C, I.E, I.R, I.BE, I.BR, dict(I.meta, weights=weights))

def run_csweep(max_qubits=19, verbose=True):
    reg = [I for I in V.build_registry(max_qubits) if I.family == 'S']
    rows = []
    for I in reg:
        base_feas = {p_: m for p_, m in V.enumerate_tours(I) if m['feas']}
        if not base_feas:
            continue
        for cfg_name, w in WEIGHT_CONFIGS.items():
            J = reweight_instance(I, w, cfg_name)
            q = V.build_qubo_v2(J)
            if q.n > max_qubits:
                continue
            pr = V.TourProblem(J)
            ex = V.exact_analysis(pr, V.FastEngine(q))
            best_feas_cost = min((m['cost'] for _, m in V.enumerate_tours(J) if m['feas']))
            row = dict(instance=I.name, config=cfg_name, weights=w, n_qubits=ex['n_qubits'], H_star=ex['H_star'], argmin_feasible=ex['argmin_feasible'], qubo_exact=ex.get('qubo_exact'), best_feasible_cost=best_feas_cost, n_feasible_tours=pr.n_feasible_tours)
            rows.append(row)
            if verbose:
                print(f"  {I.name:6s} {cfg_name:12s} H*={ex['H_star']:8.2f}  argmin_feasible={ex['argmin_feasible']}  qubo_exact={ex.get('qubo_exact')}  best_feasible_cost={best_feas_cost:.2f}", flush=True)
    with open(os.path.join(OUTDIR, 'csweep.pkl'), 'wb') as f:
        pickle.dump(rows, f)
    n_bad = sum((1 for r in rows if not r['qubo_exact']))
    print(f'\nC-sweep: {len(rows)} (instance, weight-config) pairs, {n_bad} where QUBO optimum was NOT the cheapest feasible tour under that weighting.')
    return rows

def run_transfer(max_qubits=19, verbose=True):
    reg = V.build_registry(max_qubits)
    rows = []
    for I in reg:
        pr_default = V.TourProblem(I)
        if not pr_default.feas_states:
            continue
        min_working = None
        for lam_scale in LAM_SCALES:
            q = V.build_qubo_v2(I, lam_scale=lam_scale)
            if q.n > max_qubits:
                continue
            pr = V.TourProblem(I)
            pr.qubo = q
            ex = V.exact_analysis(pr, V.FastEngine(q))
            ok = bool(ex.get('qubo_exact'))
            rows.append(dict(instance=I.name, lam_scale=lam_scale, qubo_exact=ok, H_star=ex['H_star']))
            if ok and min_working is None:
                min_working = lam_scale
            if verbose:
                print(f'  {I.name:6s} lam_scale={lam_scale:5.2f}  qubo_exact={ok}', flush=True)
        rows.append(dict(instance=I.name, lam_scale='min_working', qubo_exact=None, H_star=None, min_working_lam_scale=min_working))
    with open(os.path.join(OUTDIR, 'transfer.pkl'), 'wb') as f:
        pickle.dump(rows, f)
    mins = [r['min_working_lam_scale'] for r in rows if 'min_working_lam_scale' in r and r['min_working_lam_scale'] is not None]
    default_ok = [r for r in rows if r.get('lam_scale') == 2.0]
    n_default_ok = sum((1 for r in default_ok if r['qubo_exact']))
    print(f'\nTransfer/robustness sweep: default lam_scale=2.0 kept qubo_exact on {n_default_ok}/{len(default_ok)} instances (untuned, same constant everywhere).')
    if mins:
        print(f'Per-instance minimum working lam_scale ranged {min(mins):.2f}-{max(mins):.2f} (smaller than 2.0 everywhere means the default carries real margin; any instance needing >2.0 means the untuned default is NOT safe there).')
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['all', 'csweep', 'transfer'])
    ap.add_argument('--max-qubits', type=int, default=19)
    a = ap.parse_args()
    if a.stage in ('all', 'csweep'):
        print('=== C-SWEEP (weight sensitivity, family S) ===')
        run_csweep(a.max_qubits)
    if a.stage in ('all', 'transfer'):
        print('\n=== TRANSFER / ROBUSTNESS SWEEP (lam_scale, all instances) ===')
        run_transfer(a.max_qubits)
if __name__ == '__main__':
    main()
