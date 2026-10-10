import os, sys, time, pickle, argparse
import numpy as np
import gusrn_v2 as V
import gusrn_qaoa_patched as P
OUTDIR = './highp_results'
os.makedirs(OUTDIR, exist_ok=True)
PATH = os.path.join(OUTDIR, 'highp.pkl')

def load():
    return pickle.load(open(PATH, 'rb')) if os.path.exists(PATH) else {}

def save(d):
    tmp = PATH + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(d, f)
    os.replace(tmp, PATH)
METHOD_BUILDERS = [('Standard', lambda engine, p: P.StandardQAOA(engine, p=p)), ('CVaR', lambda engine, p: V.FastCVaR(engine, p=p, alpha=0.25)), ('WS_eps0.1', lambda engine, p: V.WarmStartEps(engine, p=p, eps=0.1))]

def run_one_incremental(problem, depths, n_restarts, n_shots, n_seeds, R, name, save_every=True, verbose=True):
    engine = V.FastEngine(problem.qubo)
    Hs = float(engine.energies.min())
    entry = R.setdefault(name, {})
    entry.setdefault('H_star', Hs)
    entry.setdefault('R_star', problem.R_star)
    entry.setdefault('rows', [])
    entry.setdefault('n_qubits', problem.n)
    entry.setdefault('n_feasible_tours', problem.n_feasible_tours)
    for mname, builder in METHOD_BUILDERS:
        for p in depths:
            existing = [r for r in entry['rows'] if r['method'] == mname and r['p'] == p]
            have = len(existing)
            if have >= n_seeds:
                if verbose:
                    print(f'     {mname:10s} p={p:3d}  cached ({have} seeds), skipping', flush=True)
                continue
            for s in range(have, n_seeds):
                seed = 42 + 1000 * s
                t0 = time.time()
                model = builder(engine, p)
                params, e, hist = model.optimize(n_restarts=n_restarts, seed=seed)
                psi = model.run_circuit(params)
                st = V.sample_psi(problem, engine, psi, Hs, n_shots, seed)
                entry['rows'].append(dict(method=mname, p=p, seed=seed, energy=float(e), rho_qubo=st['rho_qubo'], reached_opt=st['reached_opt'], best_shot_feasible=st['best_shot_feasible'], feas_rate=st['feas_rate'], elapsed=time.time() - t0))
                if save_every:
                    save(R)
            if not save_every:
                save(R)
            rr = [r for r in entry['rows'] if r['method'] == mname and r['p'] == p]
            if verbose:
                print(f"     {mname:10s} p={p:3d}  rho_qubo={np.mean([r['rho_qubo'] or 0 for r in rr]):.3f}  reached_opt={np.mean([r['reached_opt'] for r in rr]):.2f}  best_shot_feasible={np.mean([r['best_shot_feasible'] for r in rr]):.2f}  ({np.mean([r['elapsed'] for r in rr]):.1f}s/seed)", flush=True)
    return entry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depths', default='1,2,4,6,8,12,16,20')
    ap.add_argument('--only', default='')
    ap.add_argument('--n-restarts', type=int, default=3)
    ap.add_argument('--n-shots', type=int, default=2000)
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--max-qubits', type=int, default=17)
    ap.add_argument('--save-every', default='seed', choices=['seed', 'depth'], help="'seed' (default): save after every single seed run (safest). 'depth': save after every (method,depth) finishes all its seeds (less disk I/O, up to n_seeds of work at risk per crash).")
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(',')]
    only = set(a.only.split(',')) if a.only else None
    reg = V.build_registry(a.max_qubits)
    if only:
        reg = [I for I in reg if I.name in only]
    print(f'Running {len(reg)} instances x depths {depths} x 3 methods x {a.n_seeds} seeds (checkpoint granularity: per-{a.save_every})')
    R = load()
    if R:
        print(f'Loaded existing cache with instances: {sorted(R.keys())}')
    for I in reg:
        pr = V.TourProblem(I)
        print(f'  [{I.name}] n={pr.n} qubits, feasible_tours={pr.n_feasible_tours}', flush=True)
        t0 = time.time()
        try:
            run_one_incremental(pr, depths, a.n_restarts, a.n_shots, a.n_seeds, R, I.name, save_every=a.save_every == 'seed')
            print(f'  [{I.name}] DONE in {time.time() - t0:.0f}s', flush=True)
        except Exception as ex:
            print(f'  [{I.name}] INTERRUPTED/FAILED after {time.time() - t0:.0f}s: {ex}', flush=True)
            save(R)
            raise
    print('\n=== SUMMARY: reached_opt by method and depth (macro-avg over instances run so far) ===')
    print(f"{'p':>4} {'Standard':>10} {'CVaR':>10} {'WS_eps0.1':>10}")
    for p in depths:
        row = {}
        for m in ['Standard', 'CVaR', 'WS_eps0.1']:
            vals = [r['reached_opt'] for res in R.values() for r in res['rows'] if r['method'] == m and r['p'] == p]
            row[m] = np.mean(vals) if vals else float('nan')
        print(f"{p:>4} {row['Standard']:>10.3f} {row['CVaR']:>10.3f} {row['WS_eps0.1']:>10.3f}")
    print(f'\nRaw results in {PATH}')
if __name__ == '__main__':
    main()
