import os, pickle, argparse, hashlib
import numpy as np
import gusrn_v2 as V
import budget_qubo as B
import joint_family as JF
OUTDIR = './joint_family_results'
RESULTS_FILENAME = 'joint_qaoa_results.pkl'

def _code_sha():
    try:
        return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:12]
    except Exception:
        return 'unknown'
CODE_SHA = _code_sha()
TIER = {'J02': 'robust_same_scale', 'J11': 'robust_same_scale', 'J29': 'robust_same_scale', 'J01': 'penalty_rescued', 'J07': 'penalty_rescued', 'J12': 'penalty_rescued', 'J16': 'penalty_rescued', 'J18': 'penalty_rescued', 'J19': 'penalty_rescued', 'J25': 'coarse_only', 'J26': 'coarse_only', 'J13': 'tw_gap_flagged', 'J20': 'tw_gap_flagged', 'J08': 'n/a_selected_K>1', 'J14': 'clean_highK', 'J27': 'n/a_selected_K>1', 'J28': 'clean_highK', 'J30': 'clean_highK'}
TIER_K1 = [n for n, t in TIER.items() if t in ('robust_same_scale', 'penalty_rescued', 'coarse_only')]
TIER_HIGHK = [n for n, t in TIER.items() if t in ('n/a_selected_K>1', 'clean_highK')]
TIER_TW_GAP = [n for n, t in TIER.items() if t == 'tw_gap_flagged']

def build_qubo_joint(I, K_E, K_R, lam_E_scale, lam_R_scale):
    BE, BR = (I.BE, I.BR)
    q = V.build_qubo_v2(I)
    n_arcs = q.n
    r_E = np.array([I.E[a] for a in q.arcs], float)
    r_R = np.array([I.R[a] for a in q.arcs], float)
    w_E = np.array([BE / (2 ** K_E - 1) * 2 ** m for m in range(K_E)], float) if K_E > 0 else np.array([])
    w_R = np.array([BR / (2 ** K_R - 1) * 2 ** m for m in range(K_R)], float) if K_R > 0 else np.array([])
    n_total = n_arcs + K_E + K_R
    Q = np.zeros((n_total, n_total))
    g = np.zeros(n_total)
    C = q.C
    Q[:n_arcs, :n_arcs] = q.Q
    g[:n_arcs] = q.g
    lam_E = q.lam * lam_E_scale
    lam_R = q.lam * lam_R_scale

    def add_penalty(r, w, B_, lam, slack_offset, K):
        nonlocal C
        for a in range(n_arcs):
            g[a] += lam * r[a] ** 2
            for b in range(a + 1, n_arcs):
                Q[a, b] += lam * 2 * r[a] * r[b]
        for a in range(n_arcs):
            g[a] += lam * (-2 * B_ * r[a])
        for a in range(n_arcs):
            for m in range(K):
                Q[a, slack_offset + m] += lam * 2 * r[a] * w[m]
        for m in range(K):
            g[slack_offset + m] += lam * w[m] ** 2
            for m2 in range(m + 1, K):
                Q[slack_offset + m, slack_offset + m2] += lam * 2 * w[m] * w[m2]
        for m in range(K):
            g[slack_offset + m] += lam * (-2 * B_ * w[m])
        C += lam * B_ ** 2
    add_penalty(r_E, w_E, BE, lam_E, n_arcs, K_E)
    add_penalty(r_R, w_R, BR, lam_R, n_arcs + K_E, K_R)
    qb = V.TinyQubo()
    qb.n = n_total
    qb.Q = Q
    qb.g = g
    qb.C = C
    qb.arcs = q.arcs
    qb.arc_index = q.arc_index
    qb.lam = q.lam
    qb.n_arcs = n_arcs
    qb.K_E = K_E
    qb.K_R = K_R
    qb.n_slack = K_E + K_R
    qb.slack_weights = np.concatenate([w_E, w_R])
    qb.budget_which = 'energy+risk'
    qb.budget_values = (BE, BR)
    qb.n_arcs_full = getattr(q, 'n_arcs_full', n_arcs)
    return qb

def exact_analysis_joint(I, qb):
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
    out = dict(n_qubits=qb.n, argmin_feasible=bool(info and info['feas']), argmin_cost=info['cost'] if info else None, R_star_feas=R_star, n_feasible_tours=len(feas_states))
    if feas_states:
        n_slack = qb.K_E + qb.K_R
        sub_best = min((float(E[arc_state | s << qb.n_arcs]) for arc_state in feas_states for s in range(1 << n_slack)))
        out['min_feasible_energy'] = sub_best
        out['qubo_exact'] = bool(abs(sub_best - Hs) < 1e-06 and info is not None and info['feas'])
    else:
        out['qubo_exact'] = None
    out['base_cost_match'] = bool(R_star is not None and info is not None and info['feas'] and (info['cost'] is not None) and (abs(info['cost'] - R_star) < 1e-06))
    out['strict_exact'] = bool(out['qubo_exact'] and out['base_cost_match'])
    out['H_star'] = Hs
    return out

class JointProblem(B.BudgetProblem):

    def describe(self):
        return f'{self.name} joint-budget (energy+risk, BE={self.I.BE:.2f}, BR={self.I.BR:.3f}, K_E={self.qubo.K_E}, K_R={self.qubo.K_R}, {self.qubo.n_slack} slack qubits)'

def run_after_joint_incremental(I, qb, cfg, entry, R, key, path, verbose=True, ex=None):
    import time
    problem = JointProblem(I, qb)
    engine = V.FastEngine(qb)
    if entry.get('after_full') is None:
        ex = ex or exact_analysis_joint(I, qb)
        entry['after_full'] = dict(name=problem.name, family=problem.family, desc=problem.describe(), exact=ex, cfg=dict(cfg), baselines=None, variants={})
        R[key] = entry
        B.save_pkl(R, path)
        if verbose:
            print(f"  [{problem.name}] (joint-budget) n={qb.n} qubits | H*={ex['H_star']:.3f} | feasible tours={ex['n_feasible_tours']} | argmin feasible={ex['argmin_feasible']} | strict_exact={ex.get('strict_exact')}", flush=True)
    res = entry['after_full']
    Hs = res['exact']['H_star']
    if res['baselines'] is None:
        t0 = time.time()
        res['baselines'] = V.run_baselines(problem, engine, Hs, cfg['n_shots'], legacy_grad=False)
        uni = [V.summarize_idxs(problem, engine, np.random.default_rng(V.SEED + s_).integers(0, engine.N_states, cfg['n_shots']), Hs) for s_ in range(cfg['n_seeds'])]
        res['baselines']['Uniform'] = uni
        R[key] = entry
        B.save_pkl(R, path)
        if verbose:
            print(f'     baselines saved in {time.time() - t0:.0f}s', flush=True)
    elif verbose:
        print('     baselines already cached, skipping')
    for vname, builder in V.variant_builders(engine, cfg['p'], False).items():
        if vname in res['variants']:
            if verbose:
                print(f'     {vname:14s} already cached, skipping')
            continue
        runs = []
        for s_ in range(cfg['n_seeds']):
            runs.append(V.run_variational(problem, engine, vname, builder, cfg['p'], cfg['n_restarts'], V.SEED + 1000 * s_, cfg['n_shots'], Hs))
        res['variants'][vname] = runs
        R[key] = entry
        B.save_pkl(R, path)
        if verbose:
            r = [x['stats']['rho_qubo'] for x in runs]
            print(f"     {vname:14s} rho_QUBO(opt)={np.mean([x for x in r if x is not None]):.4f} reached_opt={np.mean([x['stats']['reached_opt'] for x in runs]):.2f} ({np.mean([x['elapsed'] for x in runs]):.0f}s/seed) -- SAVED", flush=True)
    return entry

def pool_code_sha(pool_dir):
    try:
        return pickle.load(open(os.path.join(pool_dir, 'joint_results.pkl'), 'rb')).get('code_sha')
    except Exception:
        return None

def load_joint_instances(pool_dir=OUTDIR):
    store = pickle.load(open(os.path.join(pool_dir, 'joint_results.pkl'), 'rb'))['store']
    out = {}
    for name, entry in store.items():
        d = entry['instance']
        row = entry['row']
        I = V.TourInstance(d['name'], d['family'], d['D'], d['e'], d['l'], d['C'], d['E'], d['R'], d['BE'], d['BR'], d['meta'])
        out[name] = (I, row['K_E'], row['K_R'], row['lam_scale'])
    return out

def run(mode='full', preset='quick', tier='k1', only=None, verbose=True, outdir=OUTDIR, pool_dir=OUTDIR, results_name=RESULTS_FILENAME):
    os.makedirs(outdir, exist_ok=True)
    results_pkl = os.path.join(outdir, results_name)
    R = B.load_pkl(results_pkl)
    cfg = V.PRESETS[preset]
    all_inst = load_joint_instances(pool_dir)
    psha = pool_code_sha(pool_dir)
    if only:
        unknown = [n for n in only if n not in all_inst]
        if unknown:
            print(f'  WARNING: not in the pool, skipped: {unknown}')
        names = [n for n in only if n in all_inst]
    elif tier == 'k1':
        names = [n for n in TIER_K1 if n in all_inst]
    elif tier == 'highk':
        names = [n for n in TIER_HIGHK if n in all_inst]
    elif tier == 'all':
        names = list(all_inst)
    else:
        names = []
    if not names:
        print('No matching instances. Check --pool-dir, --tier/--only.')
        return R
    print(f'joint_qaoa.py code_sha={CODE_SHA}  joint_family.py code_sha={JF.CODE_SHA}  pool code_sha={psha}')
    for name in names:
        I, K_E, K_R, lam = all_inst[name]
        tier_label = TIER.get(name, 'unlabeled')
        key = f'{name}_joint_{mode}'
        entry = R.get(key) or dict(mode=mode, tier=tier_label, K_E=K_E, K_R=K_R, lam_scale=lam, exact=dict(before=None, after=None))
        entry.update(code_sha_joint_qaoa=CODE_SHA, code_sha_joint_family=JF.CODE_SHA, pool_code_sha=psha)
        qb = build_qubo_joint(I, K_E, K_R, lam, lam)
        lam_pool = V.TourProblem(I).qubo.lam
        if abs(qb.lam - lam_pool) > 1e-09:
            raise RuntimeError(f'[{name}] base lambda {qb.lam} differs from the one used in pool calibration {lam_pool}')
        if entry['exact']['before'] is None:
            entry['exact']['before'] = V.exact_analysis(V.TourProblem(I), V.FastEngine(V.build_qubo_v2(I)))
            R[key] = entry
            B.save_pkl(R, results_pkl)
        if entry['exact']['after'] is None or 'strict_exact' not in entry['exact']['after']:
            entry['exact']['after'] = exact_analysis_joint(I, qb)
            R[key] = entry
            B.save_pkl(R, results_pkl)
        ea = entry['exact']['after']
        print(f"  [{key}] tier={tier_label} ({qb.n} qubits) | baseline qubo_exact={entry['exact']['before'].get('qubo_exact')} | real-QUBO: qubo_exact={ea.get('qubo_exact')} base_cost_match={ea.get('base_cost_match')} STRICT={ea.get('strict_exact')}")
        if not ea.get('strict_exact'):
            print(f'  WARNING [{name}]: the explicit Hamiltonian is NOT strictly exact -- investigate before any QAOA')
        if mode == 'exact':
            continue
        if entry.get('before_full') is None:
            pr_before = V.TourProblem(I)
            pr_before.describe = lambda: f'{I.name} Family J (before budgets)'
            entry['before_full'] = V.run_problem(pr_before, cfg, verbose=verbose)
            R[key] = entry
            B.save_pkl(R, results_pkl)
            print(f'  [{key}] before_full saved')
        else:
            print(f'  [{key}] before_full already cached, skipping')
        if mode == 'before':
            continue
        if entry.get('after_full') is not None and B.after_full_complete(entry):
            print(f'  [{key}] after_full already complete, skipping')
            continue
        entry = run_after_joint_incremental(I, qb, cfg, entry, R, key, results_pkl, verbose=verbose, ex=ea)
        print(f"  [{key}] after_full {('complete' if B.after_full_complete(entry) else 'partial -- rerun to continue')}")
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='exact', choices=['exact', 'before', 'full'])
    ap.add_argument('--preset', default='quick', choices=list(V.PRESETS))
    ap.add_argument('--tier', default='k1', choices=['k1', 'highk', 'all'])
    ap.add_argument('--only', default='', help='comma-separated, order kept, overrides --tier, e.g. J02,J11')
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--pool-dir', default=OUTDIR)
    ap.add_argument('--results-name', default=RESULTS_FILENAME)
    a = ap.parse_args()
    only = [x.strip() for x in a.only.split(',') if x.strip()] or None
    run(mode=a.mode, preset=a.preset, tier=a.tier, only=only, outdir=a.outdir, pool_dir=a.pool_dir, results_name=a.results_name)
if __name__ == '__main__':
    main()
