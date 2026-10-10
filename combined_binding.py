import os, pickle, argparse, itertools
import numpy as np
import gusrn_v2 as V
import risk_binding as RB
OUTDIR = './combined_results'
LAM_SCALES = (1, 2, 5, 10, 25, 50)
K_RANGE = tuple(range(1, 17))

def resource_sums(n, r):
    idx = np.arange(1 << n, dtype=np.int64)
    S = np.zeros(1 << n)
    for a in range(n):
        S += (idx >> a & 1) * r[a]
    return S

def make_combined_twin(I, min_gap=0.1, max_tries=5000):
    for seed in range(max_tries):
        R = RB.exposure_matrix(I, RB.density_field(seed))
        C = RB.cost_matrix(I, R)
        twin = V.TourInstance('X' + I.name[1:], 'X', I.D, I.e, I.l, C, I.E, R, I.BE, None, dict(I.meta, source=I.name, density_seed=seed, budget_mode='combined-binding'))
        tours = [(p, m) for p, m in V.enumerate_tours(twin) if m['tw_ok']]
        if len(tours) < 3:
            continue
        energy_feasible = [(p, m) for p, m in tours if m['energy'] <= I.BE + 1e-12]
        if len(energy_feasible) < 2:
            continue
        ef_best = min(energy_feasible, key=lambda z: z[1]['cost'])[1]
        r_min_ef = min((m['risk'] for _, m in energy_feasible))
        if r_min_ef <= 0 or (ef_best['risk'] - r_min_ef) / r_min_ef < min_gap:
            continue
        twin.BR = r_min_ef + 0.5 * (ef_best['risk'] - r_min_ef)
        both_feasible = [(p, m) for p, m in energy_feasible if m['risk'] <= twin.BR + 1e-12]
        if not both_feasible:
            continue
        joint_opt = min(both_feasible, key=lambda z: z[1]['cost'])[1]
        global_best = min(tours, key=lambda z: z[1]['cost'])[1]
        twin.meta.update(n_tw_feasible=len(tours), n_energy_feasible=len(energy_feasible), n_both_feasible=len(both_feasible), global_opt_cost=global_best['cost'], global_opt_energy_ok=bool(global_best['energy'] <= I.BE + 1e-12), global_opt_risk_ok=bool(global_best['risk'] <= twin.BR + 1e-12), energy_opt_cost=ef_best['cost'], energy_opt_risk=ef_best['risk'], energy_opt_risk_ok=bool(ef_best['risk'] <= twin.BR + 1e-12), joint_opt_cost=joint_opt['cost'], joint_opt_energy=joint_opt['energy'], joint_opt_risk=joint_opt['risk'])
        if abs(joint_opt['cost'] - ef_best['cost']) < 1e-09:
            continue
        return twin
    return None

def projected_combined(E_base, S_E, S_R, BE, BR, K_E, K_R, lam_E, lam_R):
    w0E = BE / (2 ** K_E - 1)
    jE = np.clip(np.rint((BE - S_E) / w0E), 0, 2 ** K_E - 1)
    residE = S_E + jE * w0E - BE
    w0R = BR / (2 ** K_R - 1)
    jR = np.clip(np.rint((BR - S_R) / w0R), 0, 2 ** K_R - 1)
    residR = S_R + jR * w0R - BR
    return E_base + lam_E * residE ** 2 + lam_R * residR ** 2

def exact_combined(pr, E_base, S_E, S_R, BE, BR, K_E, K_R, lam_E, lam_R):
    Ep = projected_combined(E_base, S_E, S_R, BE, BR, K_E, K_R, lam_E, lam_R)
    bi = int(np.argmin(Ep))
    Hs = float(Ep[bi])
    info = pr.tour_states.get(bi)
    out = dict(H_star=Hs, n_arcs=pr.n, K_E=K_E, K_R=K_R, n_qubits=pr.n + K_E + K_R, argmin_feasible=bool(info and info['feas']), argmin_cost=info['cost'] if info else None, R_star_feas=pr.R_star, n_feasible_tours=pr.n_feasible_tours)
    if pr.feas_states:
        Ef = min((float(Ep[s]) for s in pr.feas_states))
        out['min_feasible_energy'] = Ef
        out['qubo_exact'] = bool(abs(Ef - Hs) < 1e-06 and info is not None and info['feas'])
    else:
        out['qubo_exact'] = None
    out['base_cost_match'] = bool(pr.R_star is not None and info is not None and info['feas'] and (info['cost'] is not None) and (abs(info['cost'] - pr.R_star) < 1e-06))
    return out

def calibrate_combined(pr, E_base, S_E, S_R, BE, BR, lam_base, K_range=K_RANGE, lam_scales=LAM_SCALES):
    combos = sorted(itertools.product(K_range, K_range), key=lambda t: t[0] + t[1])
    best_miss = None
    for K_E, K_R in combos:
        for sc in lam_scales:
            ex = exact_combined(pr, E_base, S_E, S_R, BE, BR, K_E, K_R, lam_base * sc, lam_base * sc)
            if ex['qubo_exact']:
                return (K_E, K_R, sc, ex)
            gap = abs(ex.get('min_feasible_energy', np.inf) - ex['H_star'])
            if best_miss is None or gap < best_miss['gap']:
                best_miss = dict(K_E=K_E, K_R=K_R, scale=sc, gap=gap)
    return (None, None, None, best_miss)

def run_twins(outdir=OUTDIR, max_qubits_for_qaoa_hint=25):
    os.makedirs(outdir, exist_ok=True)
    reg = {I.name: I for I in V.build_registry(19)}
    rows, store, skipped = ([], {}, [])
    for src in ('S07', 'S08', 'S09', 'S10'):
        twin = make_combined_twin(reg[src])
        if twin is None:
            skipped.append(src)
            print(f'  [{src}] GENERATION FAILED (no density seed gave a joint optimum distinct from both single-constraint optima) -- documented limitation')
            continue
        pr = V.TourProblem(twin)
        q = pr.qubo
        r_E = np.array([twin.E[a] for a in q.arcs])
        r_R = np.array([twin.R[a] for a in q.arcs])
        E_base = V.all_energies(q)
        S_E = resource_sums(q.n, r_E)
        S_R = resource_sums(q.n, r_R)
        K_E, K_R, sc, ex = calibrate_combined(pr, E_base, S_E, S_R, twin.BE, twin.BR, q.lam)
        m = twin.meta
        qubits = q.n + K_E + K_R if K_E is not None else None
        row = dict(instance=twin.name, source=src, density_seed=m['density_seed'], n_tw_feasible=m['n_tw_feasible'], n_energy_feasible=m['n_energy_feasible'], n_both_feasible=m['n_both_feasible'], global_opt_cost=round(m['global_opt_cost'], 4), global_opt_violates='energy' if not m['global_opt_energy_ok'] else 'risk' if not m['global_opt_risk_ok'] else 'nothing', energy_opt_cost=round(m['energy_opt_cost'], 4), energy_opt_risk_ok=m['energy_opt_risk_ok'], joint_opt_cost=round(m['joint_opt_cost'], 4), BE=round(twin.BE, 4), BR=round(twin.BR, 4), n_arcs=q.n, K_E=K_E, K_R=K_R, lam_scale=sc, qubits=qubits, qubo_argmin_feasible=ex.get('argmin_feasible') if K_E is not None else None, base_cost_match=ex.get('base_cost_match') if K_E is not None else False, qubo_exact=ex.get('qubo_exact') if K_E is not None else False, qaoa_tractable_guess=bool(qubits is not None and qubits <= max_qubits_for_qaoa_hint))
        rows.append(row)
        store[twin.name] = dict(row=row, twin=twin.to_dict(), calibration=ex)
        print(f"  [{twin.name}] (from {src}, seed={m['density_seed']}) global-opt cost={m['global_opt_cost']:.2f} (violates {row['global_opt_violates']}) | energy-opt cost={m['energy_opt_cost']:.2f} (risk_ok={m['energy_opt_risk_ok']}) | joint-opt cost={m['joint_opt_cost']:.2f} (differs from both, by construction) | " + (f"K_E={K_E} K_R={K_R} (lamx{sc}) -> {qubits} qubits | base_cost_match={ex['base_cost_match']} qubo_exact={ex['qubo_exact']} | QAOA at this size: {('plausible' if row['qaoa_tractable_guess'] else 'UNLIKELY tractable (>' + str(max_qubits_for_qaoa_hint) + ' qubits)')}" if K_E is not None else f'NOT calibrated within K<={K_RANGE[-1]} each; closest={ex}'))
    if skipped:
        print(f'\nSkipped (documented, not hidden): {skipped}')
    with open(os.path.join(outdir, 'combined_results.pkl'), 'wb') as f:
        pickle.dump(dict(rows=rows, skipped=skipped, store=store), f)
    V.write_table(rows, ['instance', 'source', 'density_seed', 'n_tw_feasible', 'n_energy_feasible', 'n_both_feasible', 'global_opt_cost', 'global_opt_violates', 'energy_opt_cost', 'energy_opt_risk_ok', 'joint_opt_cost', 'BE', 'BR', 'n_arcs', 'K_E', 'K_R', 'lam_scale', 'qubits', 'qubo_argmin_feasible', 'base_cost_match', 'qubo_exact', 'qaoa_tractable_guess'], os.path.join(outdir, 'tables', 'T12_combined_binding'), 'Combined energy+risk binding: calibrated two-register slack encoding vs. the exact joint optimum.', 'tab:combined')
    return (rows, skipped)

def report(outdir=OUTDIR):
    path = os.path.join(outdir, 'combined_results.pkl')
    if not os.path.exists(path):
        print('no combined_results.pkl -- run --stage twins first')
        return
    d = pickle.load(open(path, 'rb'))
    for r in d['rows']:
        print(r)
    print(f"\nSkipped: {d['skipped']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='twins', choices=['twins', 'report'])
    ap.add_argument('--outdir', default=OUTDIR)
    a = ap.parse_args()
    if a.stage == 'twins':
        run_twins(a.outdir)
    else:
        report(a.outdir)
if __name__ == '__main__':
    main()
