import os, time, pickle, argparse, itertools
import numpy as np
import gusrn_v2 as V
OUTDIR = './harder_results'
os.makedirs(OUTDIR, exist_ok=True)

def make_hard_candidate(seed, width=300, N=5):
    return V.make_synthetic(f'H{seed}', seed, N=N, width=width, budget_mode='loose')

def cost_gap(I):
    feas = sorted((m['cost'] for _, m in V.enumerate_tours(I) if m['feas']))
    if len(feas) < 2:
        return (None, len(feas))
    return ((feas[1] - feas[0]) / feas[0], len(feas))

def build_hard_registry(n_instances, min_feasible, max_gap, max_qubits, seed_start=9000, seed_span=3000):
    reg = []
    tried = 0
    for seed in range(seed_start, seed_start + seed_span):
        if len(reg) >= n_instances:
            break
        tried += 1
        for width in (200, 300, 400):
            I = make_hard_candidate(seed, width=width)
            gap, nfeas = cost_gap(I)
            if gap is None or nfeas < min_feasible or gap > max_gap:
                continue
            q = V.build_qubo_v2(I)
            if q.n > max_qubits:
                continue
            I.name = f'H{len(reg) + 1:02d}'
            I.meta['n_feasible_tours'] = nfeas
            I.meta['cost_gap'] = gap
            reg.append(I)
            break
    print(f'found {len(reg)}/{n_instances} candidates after checking {tried} seeds (min_feasible={min_feasible}, max_gap={max_gap})')
    return reg

def run_and_report(reg, cfg, outdir):
    path = os.path.join(outdir, 'hard_results.pkl')
    R = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    for I in reg:
        if I.name in R:
            print(f'  [{I.name}] cached, skipping')
            continue
        pr = V.TourProblem(I)
        print(f"  [{I.name}] n={pr.n} qubits, feasible_tours={pr.n_feasible_tours}, cost_gap={I.meta['cost_gap']:.3f}")
        t0 = time.time()
        try:
            R[I.name] = V.run_problem(pr, cfg)
            R[I.name]['meta'] = dict(I.meta)
            with open(path, 'wb') as f:
                pickle.dump(R, f)
            print(f'  [{I.name}] DONE in {time.time() - t0:.0f}s')
        except Exception as ex:
            print(f'  [{I.name}] FAILED: {ex}')
            with open(path, 'wb') as f:
                pickle.dump(R, f)
    return R

def report(outdir):
    path = os.path.join(outdir, 'hard_results.pkl')
    R = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    if not R:
        print('no results yet')
        return
    print(f"\n{'instance':10}{'qubits':>7}{'feas_tours':>11}{'cost_gap':>10}{'method':>14}{'rho_qubo':>10}{'reached_opt':>12}{'ctrl_reached':>13}")
    agg = {}
    for name, r in sorted(R.items()):
        for m, runs in r['variants'].items():
            rq = np.mean([x['stats']['rho_qubo'] or 0 for x in runs])
            ro = np.mean([x['stats']['reached_opt'] for x in runs])
            co = np.mean([x['control']['reached_opt'] for x in runs])
            print(f"{name:10}{r['exact']['n_qubits']:>7}{r['meta']['n_feasible_tours']:>11}{r['meta']['cost_gap']:>10.3f}{m:>14}{rq:>10.3f}{ro:>12.3f}{co:>13.3f}")
            agg.setdefault(m, []).append((ro, co))
    print(f'\n=== macro-average, family H (hard regime) ===')
    for m, v in sorted(agg.items()):
        ro = np.mean([x[0] for x in v])
        co = np.mean([x[1] for x in v])
        print(f'  {m:14s} reached_opt(optimized)={ro:.3f}  reached_opt(control/no-layers)={co:.3f}  delta={ro - co:+.3f}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-instances', type=int, default=10)
    ap.add_argument('--min-feasible', type=int, default=6, help='require at least this many feasible tours')
    ap.add_argument('--max-gap', type=float, default=0.08, help='max relative cost gap between best 2 feasible tours')
    ap.add_argument('--max-qubits', type=int, default=19)
    ap.add_argument('--preset', default='standard', choices=list(V.PRESETS))
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--stage', default='all', choices=['all', 'build', 'report'])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.stage in ('all', 'build'):
        reg = build_hard_registry(a.n_instances, a.min_feasible, a.max_gap, a.max_qubits)
        if not reg:
            print('No instances matched -- relax --min-feasible or raise --max-gap and retry.')
            return
        run_and_report(reg, V.PRESETS[a.preset], a.outdir)
    if a.stage in ('all', 'report'):
        report(a.outdir)
if __name__ == '__main__':
    main()
