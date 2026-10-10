import os, argparse
import gusrn_v2 as V
import budget_qubo as B
OUTDIR = './risk_binding_results'
RISK_PKL = os.path.join(OUTDIR, 'risk_results.pkl')
RESULTS_PKL = os.path.join(OUTDIR, 'risk_qaoa_results.pkl')

def load_twins():
    import pickle
    store = pickle.load(open(RISK_PKL, 'rb'))['store']
    twins = {}
    for name, entry in store.items():
        d = entry['twin']
        I = V.TourInstance(d['name'], d['family'], d['D'], d['e'], d['l'], d['C'], d['E'], d['R'], d['BE'], d['BR'], d['meta'])
        row = entry['row']
        twins[name] = (I, row['K_min'], row['lam_budget_scale'])
    return twins

def run(mode='full', preset='quick', only=None, verbose=True):
    os.makedirs(OUTDIR, exist_ok=True)
    R = B.load_pkl(RESULTS_PKL)
    cfg = V.PRESETS[preset]
    twins = load_twins()
    names = [n for n in twins if only is None or n in only]
    if not names:
        print('No matching risk twins found in risk_results.pkl (did risk_binding.py --stage risk run?)')
        return R
    for name in names:
        I, K, lam = twins[name]
        if K is None:
            print(f'  [{name}] SKIP: not calibrated exact (K is None in risk_results.pkl)')
            continue
        key = f'{name}_risk_{mode}'
        entry = R.get(key) or dict(mode=mode, which='risk', n_slack_bits=K, lam_budget_scale=lam, exact=dict(before=None, after=None))
        qb = B.build_qubo_budget(I, n_slack_bits=K, which='risk', lam_budget_scale=lam)
        if qb is None:
            print(f'  [{name}] SKIP: risk budget not finite/positive on this twin')
            continue
        if entry['exact']['before'] is None:
            entry['exact']['before'] = V.exact_analysis(V.TourProblem(I), V.FastEngine(V.build_qubo_v2(I)))
            R[key] = entry
            B.save_pkl(R, RESULTS_PKL)
        if entry['exact']['after'] is None:
            entry['exact']['after'] = B.exact_analysis_budget(I, qb)
            R[key] = entry
            B.save_pkl(R, RESULTS_PKL)
            print(f"  [{key}] exactness on the REAL full QUBO: qubo_exact {entry['exact']['before'].get('qubo_exact')} -> {entry['exact']['after'].get('qubo_exact')} (expect False -> True, matching risk_binding.py's projected result)")
        if mode == 'exact':
            continue
        if entry.get('before_full') is None:
            pr_before = V.TourProblem(I)
            pr_before.describe = lambda: f"{I.name} risk-binding twin (source={I.meta.get('source')}, density_seed={I.meta.get('density_seed')})"
            entry['before_full'] = V.run_problem(pr_before, cfg, verbose=verbose)
            R[key] = entry
            B.save_pkl(R, RESULTS_PKL)
            print(f'  [{key}] before_full saved')
        else:
            print(f'  [{key}] before_full already cached, skipping')
        if entry.get('after_full') is not None and B.after_full_complete(entry):
            print(f'  [{key}] after_full already complete, skipping')
            continue
        entry = B.run_after_full_incremental(I, qb, cfg, entry, R, key, RESULTS_PKL, verbose=verbose)
        if B.after_full_complete(entry):
            print(f'  [{key}] after_full complete -- instance done')
        else:
            print(f'  [{key}] after_full partial this session -- rerun the same command to continue')
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='full', choices=['exact', 'full'])
    ap.add_argument('--preset', default='quick', choices=list(V.PRESETS))
    ap.add_argument('--only', default='', help='comma-separated, e.g. R10')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    run(mode=a.mode, preset=a.preset, only=only)
if __name__ == '__main__':
    main()
