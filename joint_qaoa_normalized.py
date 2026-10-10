import os, copy, time, argparse, hashlib
import numpy as np
import gusrn_v2 as V
import budget_qubo as B
import joint_family as JF
import joint_qaoa as J
P = V.P
OUTDIR = './joint_family_results'
RESULTS_FILENAME = 'joint_qaoa_results_normalized.pkl'
K1_INSTANCES = ['J01', 'J02', 'J07', 'J11', 'J12', 'J16', 'J18', 'J19', 'J25', 'J26', 'J29']
VARIANT_NAMES = ['Standard', 'CVaR', 'WS_eps1e-4', 'WS_eps0.1', 'WS_eps0.25']

def _code_sha():
    try:
        return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:12]
    except Exception:
        return 'unknown'
CODE_SHA = _code_sha()

class WSRaw(V.WarmStartEps):

    def __init__(self, engine, c_relax, p=2, eps=0.0001):
        self.eng = engine
        self.qubo = engine.qubo
        self.p = p
        self.n_params = 2 * p
        self.eps = eps
        self.c_relax = c_relax
        c = np.clip(self.c_relax, eps, 1 - eps)
        self.thetas = 2 * np.arcsin(np.sqrt(c))
        self.init_mode = f'relaxation(eps={eps},grad=correct,raw-qubo)'

def make_builders(ctx, p):
    engn, c = (ctx['engn'], ctx['relax_c'])
    return {'Standard': lambda: P.StandardQAOA(engn, p=p), 'CVaR': lambda: V.FastCVaR(engn, p=p, alpha=0.25), 'WS_eps1e-4': lambda: WSRaw(engn, c, p=p, eps=0.0001), 'WS_eps0.1': lambda: WSRaw(engn, c, p=p, eps=0.1), 'WS_eps0.25': lambda: WSRaw(engn, c, p=p, eps=0.25)}

def route_states(I, qubo):
    tour_states = {}
    for perm, m in V.enumerate_tours(I):
        tour = [0] + list(perm) + [0]
        arcs_ = list(zip(tour[:-1], tour[1:]))
        if all((a in qubo.arc_index for a in arcs_)):
            s = sum((1 << qubo.arc_index[a] for a in arcs_))
            tour_states[s] = m

    def cheapest(pred):
        c = [(m['cost'], s) for s, m in tour_states.items() if pred(m)]
        return min(c) if c else (None, None)
    cC, sC = cheapest(lambda m: m['tw_ok'])
    cE, sE = cheapest(lambda m: m['tw_ok'] and m['batt_ok'])
    cER, sER = cheapest(lambda m: m['feas'])
    for nm, cost in (('x_C', cC), ('x_E', cE), ('x_ER', cER)):
        ref = I.meta.get(f'{nm}_cost')
        if ref is not None and cost is not None and (abs(cost - ref) > 1e-06):
            print(f'  WARNING [{I.name}] {nm} cost {cost} differs from pool meta {ref}')
    return dict(x_C=sC, x_E=sE, x_ER=sER, feas=np.array([s for s, m in tour_states.items() if m['feas']], dtype=np.int64), tours=np.array(list(tour_states), dtype=np.int64))

def prepare(I, qraw):
    eng_raw = V.FastEngine(qraw)
    E = eng_raw.energies
    sigma = float(np.std(E))
    qs = copy.copy(qraw)
    qs.Q, qs.g, qs.C = (qraw.Q / sigma, qraw.g / sigma, qraw.C / sigma)
    qs.sigma_H = sigma
    engn = copy.copy(eng_raw)
    engn.qubo = qs
    engn.energies = E / sigma
    n_arcs = len(qraw.arcs)
    n_slack = qraw.n - n_arcs
    Hs_raw = float(E.min())
    ctx = dict(qraw=qraw, qs=qs, eng_raw=eng_raw, engn=engn, sigma=sigma, n_arcs=n_arcs, n_slack=n_slack, rs=route_states(I, qraw), Hs_raw=Hs_raw, Hs_norm=float(engn.energies.min()), argmin_idx=int(np.argmin(E)), tol_mask=(E - Hs_raw) / max(1.0, abs(Hs_raw)) <= V.TOL_SUCCESS, relax_c=V.relaxation(qraw, True))
    assert int(np.argmin(engn.energies)) == ctx['argmin_idx'], 'normalization changed the argmin'
    assert abs(float(np.std(engn.energies)) - 1.0) < 1e-09, 'normalized std is not 1'
    return ctx

def route_probs(psi, ctx):
    pr = np.abs(psi) ** 2
    pr = pr / pr.sum()
    n_arcs, n_slack, rs = (ctx['n_arcs'], ctx['n_slack'], ctx['rs'])
    marg = pr.reshape(1 << n_slack, 1 << n_arcs).sum(axis=0)
    base = 1.0 / (1 << n_arcs)
    out = {}
    for k in ('x_C', 'x_E', 'x_ER'):
        s = rs[k]
        out[f'P_{k}'] = None if s is None else float(marg[s])
        out[f'enrich_{k}'] = None if s is None else float(marg[s] / base)
    nf = len(rs['feas'])
    out['P_feas'] = float(marg[rs['feas']].sum()) if nf else None
    out['enrich_feas'] = out['P_feas'] / (nf * base) if nf else None
    out['P_tour'] = float(marg[rs['tours']].sum())
    out['P_argmin_full'] = float(pr[ctx['argmin_idx']])
    out['uniform_P_route'] = base
    return out

def run_variant_seed(problem, ctx, vname, builder, cfg, s_):
    cap = {}

    def b():
        cap['m'] = builder()
        return cap['m']
    ent = V.run_variational(problem, ctx['engn'], vname, b, cfg['p'], cfg['n_restarts'], V.SEED + 1000 * s_, cfg['n_shots'], ctx['Hs_norm'])
    model = cap['m']
    psi = model.run_circuit(ent['params'])
    psi0 = model.initial_psi() if ent['is_ws'] else ctx['engn'].uniform_init()
    ent['probs_after'] = route_probs(psi, ctx)
    ent['probs_before'] = route_probs(psi0, ctx)
    pr = np.abs(psi) ** 2
    pr = pr / pr.sum()
    ent['p_tol_raw'] = float(pr[ctx['tol_mask']].sum())
    ent['exp_before_raw'] = ent['exp_before'] * ctx['sigma']
    ent['exp_after_raw'] = ent['exp_after'] * ctx['sigma']
    ent['sigma_H'] = ctx['sigma']
    return ent

def _fmt(x, d=3):
    return 'n/a' if x is None else f'{x:.{d}g}'

def run_stage(tag, problem, ctx, cfg, section, flush, verbose=True):
    engn, Hs = (ctx['engn'], ctx['Hs_norm'])
    if section.get('baselines') is None:
        t0 = time.time()
        bl = V.run_baselines(problem, engn, Hs, cfg['n_shots'], legacy_grad=False)
        bl['Uniform'] = [V.summarize_idxs(problem, engn, np.random.default_rng(V.SEED + s).integers(0, engn.N_states, cfg['n_shots']), Hs) for s in range(cfg['n_seeds'])]
        section['baselines'] = bl
        flush()
        if verbose:
            print(f'     [{tag}] baselines (on normalized H) saved in {time.time() - t0:.0f}s', flush=True)
    builders = make_builders(ctx, cfg['p'])
    for vname in VARIANT_NAMES:
        runs = section['variants'].setdefault(vname, [])
        if len(runs) >= cfg['n_seeds'] and verbose:
            print(f'     [{tag}] {vname:12s} already has {len(runs)} seed(s), skipping')
        for s_ in range(len(runs), cfg['n_seeds']):
            ent = run_variant_seed(problem, ctx, vname, builders[vname], cfg, s_)
            runs.append(ent)
            flush()
            if verbose:
                pa = ent['probs_after']
                drop = 1.0 - ent['exp_after'] / ent['exp_before'] if ent['exp_before'] else float('nan')
                print(f"     [{tag}] {vname:12s} seed{s_} rho={_fmt(ent['stats']['rho_qubo'])} reached_opt={ent['stats']['reached_opt']} P(xER)={_fmt(pa['P_x_ER'])} (x{_fmt(pa['enrich_x_ER'])} vs uniform) P_feas={_fmt(pa['P_feas'])} exp_drop={drop:.1%} Neff={ent['N_eff_after']:.0f} ({ent['elapsed']:.0f}s) -- SAVED", flush=True)

def run(mode='full', preset='quick', only=None, n_seeds=None, with_before=False, verbose=True, outdir=OUTDIR, pool_dir=OUTDIR, results_name=RESULTS_FILENAME):
    os.makedirs(outdir, exist_ok=True)
    results_pkl = os.path.join(outdir, results_name)
    R = B.load_pkl(results_pkl)
    cfg = dict(V.PRESETS[preset])
    if n_seeds:
        cfg['n_seeds'] = int(n_seeds)
    all_inst = J.load_joint_instances(pool_dir)
    psha = J.pool_code_sha(pool_dir)
    if only:
        unknown = [n for n in only if n not in all_inst]
        if unknown:
            print(f'  WARNING: not in the pool, skipped: {unknown}')
        names = [n for n in only if n in all_inst]
    else:
        names = [n for n in K1_INSTANCES if n in all_inst]
    if not names:
        print('No matching instances.')
        return R
    print(f'joint_qaoa_normalized.py code_sha={CODE_SHA}  joint_qaoa.py code_sha={J.CODE_SHA}  joint_family.py code_sha={JF.CODE_SHA}  pool code_sha={psha}')
    print(f'cfg={cfg}  mode={mode}  with_before={with_before}')
    for name in names:
        I, K_E, K_R, lam = all_inst[name]
        key = f'{name}_joint_norm_{mode}'
        entry = R.get(key) or dict(mode=mode, tier=J.TIER.get(name, 'unlabeled'), K_E=K_E, K_R=K_R, lam_scale=lam, exact=dict(before=None, after=None))
        entry.update(code_sha_normalized=CODE_SHA, code_sha_joint_qaoa=J.CODE_SHA, code_sha_joint_family=JF.CODE_SHA, pool_code_sha=psha)
        flush = lambda: (R.__setitem__(key, entry), B.save_pkl(R, results_pkl))
        qb = J.build_qubo_joint(I, K_E, K_R, lam, lam)
        lam_pool = V.TourProblem(I).qubo.lam
        if abs(qb.lam - lam_pool) > 1e-09:
            raise RuntimeError(f'[{name}] base lambda {qb.lam} differs from pool calibration {lam_pool}')
        t0 = time.time()
        ctx = prepare(I, qb)
        entry['normalization'] = dict(rule='H/std(all 2^n energies)', sigma_H=ctx['sigma'], H_star_raw=ctx['Hs_raw'], H_star_norm=ctx['Hs_norm'])
        if entry['exact']['after'] is None:
            entry['exact']['after'] = J.exact_analysis_joint(I, qb)
        flush()
        ea = entry['exact']['after']
        print(f"  [{key}] {qb.n} qubits | sigma_H={ctx['sigma']:.4e} | H*_raw={ctx['Hs_raw']:.4f} | strict_exact(raw)={ea.get('strict_exact')} | prep {time.time() - t0:.0f}s", flush=True)
        if not ea.get('strict_exact'):
            print(f'  WARNING [{name}]: raw Hamiltonian NOT strictly exact -- skipping QAOA for this instance')
            continue
        if mode == 'exact':
            continue
        if with_before and entry.get('before_full') is None:
            pr_b = V.TourProblem(I)
            ctx_b = prepare(I, pr_b.qubo)
            pr_b.qubo = ctx_b['qs']
            pr_b.describe = lambda: f'{I.name} Family J (before budgets, normalized)'
            entry['before_full'] = dict(name=name, family='J', desc=pr_b.describe(), cfg=dict(cfg), exact=dict(H_star_raw=ctx_b['Hs_raw'], sigma_H=ctx_b['sigma']), baselines=None, variants={})
            run_stage('before', pr_b, ctx_b, cfg, entry['before_full'], flush, verbose)
        if entry.get('after_full') is None:
            problem = J.JointProblem(I, ctx['qs'])
            entry['after_full'] = dict(name=name, family='J', desc=problem.describe(), exact=ea, cfg=dict(cfg), baselines=None, variants={}, normalization=entry['normalization'])
            flush()
        problem = J.JointProblem(I, ctx['qs'])
        entry['after_full']['cfg'] = dict(cfg)
        run_stage('after', problem, ctx, cfg, entry['after_full'], flush, verbose)
        done = all((len(entry['after_full']['variants'].get(v, [])) >= cfg['n_seeds'] for v in VARIANT_NAMES))
        print(f"  [{key}] after_full {('complete' if done else 'partial -- rerun to continue')}", flush=True)
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='full', choices=['exact', 'full'])
    ap.add_argument('--preset', default='quick', choices=list(V.PRESETS))
    ap.add_argument('--n-seeds', type=int, default=0, help='override the preset seed count (resumes/adds seeds)')
    ap.add_argument('--only', default='', help='comma-separated, order kept, e.g. J02,J01,J12')
    ap.add_argument('--with-before', action='store_true', help='also run the no-budget QUBO, normalized')
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--pool-dir', default=OUTDIR)
    ap.add_argument('--results-name', default=RESULTS_FILENAME)
    a = ap.parse_args()
    only = [x.strip() for x in a.only.split(',') if x.strip()] or None
    run(mode=a.mode, preset=a.preset, only=only, n_seeds=a.n_seeds or None, with_before=a.with_before, outdir=a.outdir, pool_dir=a.pool_dir, results_name=a.results_name)
if __name__ == '__main__':
    main()
