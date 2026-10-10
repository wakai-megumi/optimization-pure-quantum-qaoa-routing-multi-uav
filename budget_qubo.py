import os, time, pickle, argparse
import numpy as np
import gusrn_v2 as V
OUTDIR = './budget_qubo_v3_results'
CALIBRATED_K = {'S07': 8, 'S08': 8, 'S09': 10, 'S10': 8}
CALIBRATED_LAM_BUDGET_SCALE = 1.0
EXPECTED_VARIANTS = ['Standard', 'CVaR', 'WS_eps1e-4', 'WS_eps0.1', 'WS_eps0.25']
_SEED_CALIBRATION = {(name, 'energy'): (K, CALIBRATED_LAM_BUDGET_SCALE) for name, K in CALIBRATED_K.items()}

def calibrate_budget_for(I, which, K_options=(2, 3, 4, 6, 8, 10, 12, 14, 16), lam_scales=(1, 2, 5, 10, 25, 50), max_qubits=35, verbose=True):
    best_miss = None
    for K in K_options:
        qb_probe = build_qubo_budget(I, n_slack_bits=K, which=which, lam_budget_scale=1.0)
        if qb_probe is None:
            return (None, None, dict(reason=f'{which} budget not finite/positive on {I.name}'))
        if qb_probe.n > max_qubits:
            if verbose:
                print(f'    [{I.name}/{which}] K={K}: qubits={qb_probe.n} > max_qubits={max_qubits}, stopping sweep')
            break
        for scale in lam_scales:
            qb = build_qubo_budget(I, n_slack_bits=K, which=which, lam_budget_scale=scale)
            ex = exact_analysis_budget(I, qb)
            if ex.get('qubo_exact'):
                if verbose:
                    print(f'    [{I.name}/{which}] CALIBRATED: K={K}, lam_budget_scale={scale} -> qubits={qb.n}, qubo_exact=True')
                return (K, scale, dict(qubits=qb.n))
            gap = abs(ex.get('min_feasible_energy', float('inf')) - ex['H_star'])
            if best_miss is None or gap < best_miss['gap']:
                best_miss = dict(K=K, scale=scale, qubits=qb.n, gap=gap)
            if scale > 1:
                continue
    if verbose and best_miss:
        print(f"    [{I.name}/{which}] NO combination worked. Closest: K={best_miss['K']}, lam_budget_scale={best_miss['scale']} (qubits={best_miss['qubits']}, gap={best_miss['gap']:.4f})")
    return (None, None, dict(reason='no combination in swept range worked', closest_miss=best_miss))

def get_or_calibrate(I, which, R, path, verbose=True):
    cal = R.setdefault('_calibration', dict(_SEED_CALIBRATION))
    key = (I.name, which)
    if key in cal:
        K, scale = cal[key]
        if verbose:
            print(f'  [{I.name}/{which}] using cached calibration: K={K}, lam_budget_scale={scale}')
        return (K, scale)
    if verbose:
        print(f"  [{I.name}/{which}] no calibration on record -- running sweep now (this is the real fix: {which} gets its OWN sweep, never borrows another resource's K)")
    K, scale, info = calibrate_budget_for(I, which, verbose=verbose)
    cal[key] = (K, scale)
    save_pkl(R, path)
    if K is None and verbose:
        print(f'  [{I.name}/{which}] calibration FAILED -- {info}')
    return (K, scale)

def build_qubo_budget(I, lam_scale=2.0, prune=True, pair_nogoods=True, n_slack_bits=4, which='energy', lam_budget_scale=1.0):
    B = I.BE if which == 'energy' else I.BR
    R = I.E if which == 'energy' else I.R
    if not np.isfinite(B) or B <= 0:
        return None
    q = V.build_qubo_v2(I, lam_scale=lam_scale, prune=prune, pair_nogoods=pair_nogoods)
    n_arcs = q.n
    K = int(n_slack_bits)
    r = np.array([R[a] for a in q.arcs], float)
    w = np.array([B / (2 ** K - 1) * 2 ** m for m in range(K)], float) if K > 0 else np.array([])
    n_total = n_arcs + K
    Q = np.zeros((n_total, n_total))
    g = np.zeros(n_total)
    C = q.C
    Q[:n_arcs, :n_arcs] = q.Q
    g[:n_arcs] = q.g
    lam_b = q.lam * lam_budget_scale
    for a in range(n_arcs):
        g[a] += lam_b * r[a] ** 2
        for b in range(a + 1, n_arcs):
            Q[a, b] += lam_b * 2 * r[a] * r[b]
    for a in range(n_arcs):
        g[a] += lam_b * (-2 * B * r[a])
    for a in range(n_arcs):
        for m in range(K):
            Q[a, n_arcs + m] += lam_b * 2 * r[a] * w[m]
    for m in range(K):
        g[n_arcs + m] += lam_b * w[m] ** 2
        for m2 in range(m + 1, K):
            Q[n_arcs + m, n_arcs + m2] += lam_b * 2 * w[m] * w[m2]
    for m in range(K):
        g[n_arcs + m] += lam_b * (-2 * B * w[m])
    C += lam_b * B ** 2
    qb = V.TinyQubo()
    qb.n = n_total
    qb.Q = Q
    qb.g = g
    qb.C = C
    qb.arcs = q.arcs
    qb.arc_index = q.arc_index
    qb.lam = q.lam
    qb.n_arcs = n_arcs
    qb.n_slack = K
    qb.slack_weights = w
    qb.budget_which = which
    qb.budget_value = B
    qb.n_arcs_full = getattr(q, 'n_arcs_full', n_arcs)
    return qb

def exact_analysis_budget(I, qb):
    engine = V.FastEngine(qb)
    E = engine.energies
    Hs = float(E.min())
    bi = int(np.argmin(E))
    arc_mask = (1 << qb.n_arcs) - 1
    argmin_arc_state = bi & arc_mask
    tour_states = {}
    for perm, m in V.enumerate_tours(I):
        tour = [0] + list(perm) + [0]
        arcs_ = list(zip(tour[:-1], tour[1:]))
        if all((a in qb.arc_index for a in arcs_)):
            s = sum((1 << qb.arc_index[a] for a in arcs_))
            tour_states[s] = m
    info = tour_states.get(argmin_arc_state)
    feas_states = {s: m for s, m in tour_states.items() if m['feas']}
    R_star = min((m['cost'] for m in feas_states.values()), default=None)
    out = dict(n_qubits=qb.n, n_arcs=qb.n_arcs, n_slack=qb.n_slack, H_star=Hs, argmin_feasible=bool(info and info['feas']), argmin_cost=info['cost'] if info else None, R_star_feas=R_star, n_feasible_tours=len(feas_states))
    if feas_states:
        sub_best = None
        for arc_state, m in feas_states.items():
            sub = [arc_state | s << qb.n_arcs for s in range(1 << qb.n_slack)]
            Ef_this = float(min((E[s] for s in sub)))
            if sub_best is None or Ef_this < sub_best:
                sub_best = Ef_this
        out['min_feasible_energy'] = sub_best
        out['qubo_exact'] = bool(abs(sub_best - Hs) < 1e-06 and info and info['feas'])
    else:
        out['qubo_exact'] = None
    return out

class BudgetProblem:

    def __init__(self, I, qb):
        self.I = I
        self.qubo = qb
        self.n = qb.n
        self.name = I.name
        self.family = I.family
        self.arc_mask = (1 << qb.n_arcs) - 1
        self.tour_states = {}
        for perm, m in V.enumerate_tours(I):
            tour = [0] + list(perm) + [0]
            arcs_ = list(zip(tour[:-1], tour[1:]))
            if all((a in qb.arc_index for a in arcs_)):
                s = sum((1 << qb.arc_index[a] for a in arcs_))
                self.tour_states[s] = dict(ham=True, cov=True, feas=m['feas'], cost=m['cost'], tw_ok=m['tw_ok'], batt_ok=m['batt_ok'], sora_ok=m['sora_ok'], energy=m['energy'], risk=m['risk'], tour=tour)
        feas = {s: i for s, i in self.tour_states.items() if i['feas']}
        self.R_star = min((i['cost'] for i in feas.values()), default=None)
        self.n_feasible_tours = len(feas)
        self._none = dict(ham=False, cov=False, feas=False, cost=None, tw_ok=False, batt_ok=False, sora_ok=False, energy=None, risk=None, tour=None)

    def state_info(self, idx):
        return self.tour_states.get(idx & self.arc_mask, self._none)

    def describe(self):
        return f'{self.name} budget-extended ({self.qubo.budget_which}, B={self.qubo.budget_value:.2f}, {self.qubo.n_slack} slack qubits)'

def run_problem_budget(problem, I, qb, cfg, verbose=True):
    t_all = time.time()
    engine = V.FastEngine(qb)
    ex = exact_analysis_budget(I, qb)
    Hs = ex['H_star']
    res = dict(name=problem.name, family=problem.family, desc=problem.describe(), exact=ex, cfg=dict(cfg))
    if verbose:
        print(f"  [{problem.name}] (budget-extended) n={qb.n} qubits | H*={Hs:.3f} | feasible tours={ex['n_feasible_tours']} | argmin feasible={ex['argmin_feasible']} | exact QUBO={ex.get('qubo_exact')}", flush=True)
    res['baselines'] = V.run_baselines(problem, engine, Hs, cfg['n_shots'], legacy_grad=False)
    uni = [V.summarize_idxs(problem, engine, np.random.default_rng(V.SEED + s).integers(0, engine.N_states, cfg['n_shots']), Hs) for s in range(cfg['n_seeds'])]
    res['baselines']['Uniform'] = uni
    res['variants'] = {}
    for vname, builder in V.variant_builders(engine, cfg['p'], False).items():
        runs = []
        for s in range(cfg['n_seeds']):
            runs.append(V.run_variational(problem, engine, vname, builder, cfg['p'], cfg['n_restarts'], V.SEED + 1000 * s, cfg['n_shots'], Hs))
        res['variants'][vname] = runs
        if verbose:
            r = [x['stats']['rho_qubo'] for x in runs]
            print(f"     {vname:14s} rho_QUBO(opt)={np.mean([x for x in r if x is not None]):.4f} reached_opt={np.mean([x['stats']['reached_opt'] for x in runs]):.2f} ({np.mean([x['elapsed'] for x in runs]):.0f}s/seed)", flush=True)
    res['elapsed'] = time.time() - t_all
    return res

def after_full_complete(entry):
    af = entry.get('after_full')
    return bool(af and af.get('baselines') is not None and (set(af.get('variants', {}).keys()) >= set(EXPECTED_VARIANTS)))

def run_after_full_incremental(I, qb, cfg, entry, R, key, path, verbose=True):
    problem = BudgetProblem(I, qb)
    engine = V.FastEngine(qb)
    if entry.get('after_full') is None:
        ex = exact_analysis_budget(I, qb)
        Hs = ex['H_star']
        entry['after_full'] = dict(name=problem.name, family=problem.family, desc=problem.describe(), exact=ex, cfg=dict(cfg), baselines=None, variants={})
        R[key] = entry
        save_pkl(R, path)
        if verbose:
            print(f"  [{problem.name}] (budget-extended) n={qb.n} qubits | H*={Hs:.3f} | feasible tours={ex['n_feasible_tours']} | argmin feasible={ex['argmin_feasible']} | exact QUBO={ex.get('qubo_exact')}", flush=True)
    res = entry['after_full']
    Hs = res['exact']['H_star']
    if res['baselines'] is None:
        t0 = time.time()
        res['baselines'] = V.run_baselines(problem, engine, Hs, cfg['n_shots'], legacy_grad=False)
        uni = [V.summarize_idxs(problem, engine, np.random.default_rng(V.SEED + s).integers(0, engine.N_states, cfg['n_shots']), Hs) for s in range(cfg['n_seeds'])]
        res['baselines']['Uniform'] = uni
        R[key] = entry
        save_pkl(R, path)
        if verbose:
            print(f'     baselines saved in {time.time() - t0:.0f}s', flush=True)
    elif verbose:
        print(f'     baselines already cached, skipping')
    for vname, builder in V.variant_builders(engine, cfg['p'], False).items():
        if vname in res['variants']:
            if verbose:
                print(f'     {vname:14s} already cached, skipping')
            continue
        t0 = time.time()
        runs = []
        for s in range(cfg['n_seeds']):
            runs.append(V.run_variational(problem, engine, vname, builder, cfg['p'], cfg['n_restarts'], V.SEED + 1000 * s, cfg['n_shots'], Hs))
        res['variants'][vname] = runs
        R[key] = entry
        save_pkl(R, path)
        if verbose:
            r = [x['stats']['rho_qubo'] for x in runs]
            print(f"     {vname:14s} rho_QUBO(opt)={np.mean([x for x in r if x is not None]):.4f} reached_opt={np.mean([x['stats']['reached_opt'] for x in runs]):.2f} ({np.mean([x['elapsed'] for x in runs]):.0f}s/seed) -- SAVED", flush=True)
    return entry

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    return pickle.load(open(path, 'rb')) if os.path.exists(path) else {}

def run(mode='full', preset='quick', which='energy', only=None, verbose=True, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'budget_results.pkl')
    R = load_pkl(path)
    cfg = V.PRESETS[preset]
    reg = [I for I in V.build_registry(19) if I.family == 'S' and I.meta.get('budget_mode') == 'binding']
    if only:
        reg = [I for I in reg if I.name in only]
    if not reg:
        print('No matching binding-mode S instances found -- nothing to check.')
        return R
    for I in reg:
        K, lam_scale_b = get_or_calibrate(I, which, R, path, verbose=verbose)
        if K is None:
            print(f'  [{I.name}] SKIP: {which} could not be calibrated exact within the swept range -- see calibration failure logged above, this is a real result (not a bug) if it persists after widening K_options/max_qubits.')
            continue
        key = f'{I.name}_{which}_{mode}'
        entry = R.get(key) or dict(mode=mode, which=which, calibrated=True, n_slack_bits=K, lam_budget_scale=lam_scale_b, exact=dict(before=None, after=None))
        qb = build_qubo_budget(I, n_slack_bits=K, which=which, lam_budget_scale=lam_scale_b)
        if qb is None:
            print(f'  [{I.name}] SKIP: {which} budget not finite/positive on this instance')
            continue
        if entry['exact']['before'] is None:
            entry['exact']['before'] = V.exact_analysis(V.TourProblem(I), V.FastEngine(V.build_qubo_v2(I)))
            R[key] = entry
            save_pkl(R, path)
        if entry['exact']['after'] is None:
            entry['exact']['after'] = exact_analysis_budget(I, qb)
            R[key] = entry
            save_pkl(R, path)
            print(f"  [{key}] exactness saved: qubo_exact {entry['exact']['before'].get('qubo_exact')} -> {entry['exact']['after'].get('qubo_exact')}")
        if mode == 'exact':
            continue
        if entry.get('before_full') is None:
            t0 = time.time()
            try:
                entry['before_full'] = V.run_problem(V.TourProblem(I), cfg, verbose=verbose)
                R[key] = entry
                save_pkl(R, path)
                print(f'  [{key}] before_full saved in {time.time() - t0:.0f}s')
            except Exception as exn:
                print(f'  [{key}] before_full FAILED: {exn}')
                import traceback
                traceback.print_exc()
                save_pkl(R, path)
                continue
        else:
            print(f'  [{key}] before_full already cached, skipping')
        if entry.get('after_full') is not None and after_full_complete(entry):
            print(f'  [{key}] after_full already fully complete (all 5 variants), skipping')
            continue
        try:
            entry = run_after_full_incremental(I, qb, cfg, entry, R, key, path, verbose=verbose)
            if after_full_complete(entry):
                print(f'  [{key}] after_full complete -- instance done')
            else:
                print(f'  [{key}] after_full partially done this session -- rerun the same command to continue from the next unfinished variant')
        except Exception as exn:
            print(f'  [{key}] after_full FAILED: {exn}')
            import traceback
            traceback.print_exc()
            save_pkl(R, path)
    return R

def report(outdir=OUTDIR):
    path = os.path.join(outdir, 'budget_results.pkl')
    R = load_pkl(path)
    if not R:
        print('no budget_results.pkl -- run first')
        return
    T = os.path.join(outdir, 'tables')
    os.makedirs(T, exist_ok=True)
    has_exact = {k: v for k, v in R.items() if v.get('exact', {}).get('before') and v.get('exact', {}).get('after')}
    complete_full = {k: v for k, v in R.items() if after_full_complete(v)}
    incomplete = {k: v for k, v in R.items() if k not in has_exact or (v['mode'] == 'full' and k not in complete_full)}
    if incomplete:
        print(f"Note: {len(incomplete)} instance(s) still incomplete: {list(incomplete)} -- excluded from the relevant table(s) below, rerun 'run' to finish them.")
    ex_rows = []
    for key, entry in sorted(has_exact.items()):
        b_ex, a_ex = (entry['exact']['before'], entry['exact']['after'])
        ex_rows.append(dict(instance=key, which=entry['which'], slack_bits=entry['n_slack_bits'], lam_budget_scale=entry['lam_budget_scale'], qubits_before=b_ex['n_qubits'], qubits_after=a_ex['n_qubits'], argmin_feasible_before=b_ex['argmin_feasible'], argmin_feasible_after=a_ex['argmin_feasible'], qubo_exact_before=b_ex.get('qubo_exact'), qubo_exact_after=a_ex.get('qubo_exact')))
    if ex_rows:
        V.write_table(ex_rows, ['instance', 'which', 'slack_bits', 'lam_budget_scale', 'qubits_before', 'qubits_after', 'argmin_feasible_before', 'argmin_feasible_after', 'qubo_exact_before', 'qubo_exact_after'], os.path.join(T, 'T7a_budget_exactness'), 'Exactness before vs after the calibrated slack-variable budget encoding.', 'tab:budget_exact')
        n_fixed = sum((1 for r in ex_rows if not r['qubo_exact_before'] and r['qubo_exact_after']))
        print(f'\nExactness: {n_fixed}/{len(ex_rows)} instances flipped qubo_exact False->True.')
    full_entries = complete_full
    if full_entries:
        rows = []
        for key, entry in sorted(full_entries.items()):
            for side, res in [('before', entry['before_full']), ('after', entry['after_full'])]:
                for vname, runs in res['variants'].items():
                    st = [x['stats'] for x in runs]
                    rows.append(dict(instance=key, side=side, method=vname, rho_qubo=V._mean([s['rho_qubo'] for s in st]), reached_opt=V._mean([s['reached_opt'] for s in st]), feasible_rate=V._mean([s['feas_rate'] for s in st]), best_shot_feasible=V._mean([s['best_shot_feasible'] for s in st]), rho_route=V._mean([s['rho_route'] for s in st])))
                b = res['baselines']
                for m in ['Greedy', 'RRLS']:
                    if m in b:
                        lst = b[m] if isinstance(b[m], list) else [b[m]]
                        rows.append(dict(instance=key, side=side, method=m, rho_qubo=V._mean([x.get('rho_qubo') for x in lst]), reached_opt=V._mean([x.get('reached_opt') for x in lst]), feasible_rate=None, best_shot_feasible=V._mean([x.get('feasible') for x in lst]), rho_route=V._mean([x.get('rho_route') for x in lst])))
        V.write_table(rows, ['instance', 'side', 'method', 'rho_qubo', 'reached_opt', 'feasible_rate', 'best_shot_feasible', 'rho_route'], os.path.join(T, 'T7b_budget_qaoa_comparison'), 'With (after) vs without (before) the calibrated budget-encoded QUBO.', 'tab:budget_qaoa')
        print(f'QAOA comparison table written for {len(full_entries)} complete instance(s).')
    else:
        print("No complete 'full' mode entries yet.")
    print(f'\nTables written to {T}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='full', choices=['exact', 'full'])
    ap.add_argument('--preset', default='quick', choices=list(V.PRESETS))
    ap.add_argument('--which', default='energy', choices=['energy', 'risk'])
    ap.add_argument('--only', default='', help='comma-separated instance names, e.g. S09,S10')
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--stage', default='all', choices=['all', 'run', 'report'])
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    if a.stage in ('all', 'run'):
        run(mode=a.mode, preset=a.preset, which=a.which, only=only, outdir=a.outdir)
    if a.stage in ('all', 'report'):
        report(outdir=a.outdir)
if __name__ == '__main__':
    main()
