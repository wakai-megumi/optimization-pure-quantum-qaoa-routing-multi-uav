import os, csv, pickle, argparse, itertools, hashlib
import numpy as np
import gusrn_v2 as V
import risk_binding as RB
import combined_binding as CB

def _code_sha():
    try:
        return hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:12]
    except Exception:
        return 'unknown'
CODE_SHA = _code_sha()
OUTDIR = './joint_family_results'
MIN_TW_FEASIBLE = 5
MIN_ENERGY_FEASIBLE = 3
MIN_JOINT_FEASIBLE = 2
COST_TOL = 1e-09
WIDTHS = (100, 200, 300, 400)
MAX_BASE_QUBITS = 19
QAOA_MAX_QUBITS = 30

def select_budgets(tours, tol=COST_TOL):
    if len(tours) < MIN_TW_FEASIBLE:
        return None
    x_C = min(tours, key=lambda z: z[1]['cost'])[1]
    candidate_BEs = sorted(set((m['energy'] for _, m in tours)))
    BE = None
    for BE_try in candidate_BEs:
        ef = [(p, m) for p, m in tours if m['energy'] <= BE_try + tol]
        if len(ef) < MIN_ENERGY_FEASIBLE:
            continue
        if any((m is x_C for _, m in ef)):
            return None
        BE = BE_try
        break
    if BE is None:
        return None
    x_E = min(ef, key=lambda z: z[1]['cost'])[1]
    candidate_BRs = sorted(set((m['risk'] for _, m in ef)))
    BR = None
    for BR_try in candidate_BRs:
        both = [(p, m) for p, m in ef if m['risk'] <= BR_try + tol]
        if len(both) >= MIN_JOINT_FEASIBLE and (not any((m is x_E for _, m in both))):
            BR = BR_try
            x_ER = min(both, key=lambda z: z[1]['cost'])[1]
            break
    if BR is None:
        return None
    if not (x_E['cost'] > x_C['cost'] + tol and x_ER['cost'] > x_E['cost'] + tol):
        return None
    en_excl = [m['energy'] for _, m in tours if m['energy'] > BE + tol]
    rk_excl = [m['risk'] for _, m in ef if m['risk'] > BR + tol]
    margin_E = (min(en_excl) - BE) / BE if en_excl and BE > 0 else float('nan')
    margin_R = (min(rk_excl) - BR) / BR if rk_excl and BR > 0 else float('nan')
    en = np.array([m['energy'] for _, m in tours], float)
    rk = np.array([m['risk'] for _, m in tours], float)
    er_corr = float(np.corrcoef(en, rk)[0, 1]) if en.std() > 0 and rk.std() > 0 else float('nan')
    return dict(BE=BE, BR=BR, x_C=x_C, x_E=x_E, x_ER=x_ER, n_tw=len(tours), n_energy=len(ef), n_joint=len(both), rel_gap_CE=(x_E['cost'] - x_C['cost']) / x_C['cost'], rel_gap_EER=(x_ER['cost'] - x_E['cost']) / x_E['cost'], margin_E=margin_E, margin_R=margin_R, er_corr=er_corr)

def calibrate_strict(pr, E_base, S_E, S_R, BE, BR, lam_base, K_range=CB.K_RANGE, lam_scales=CB.LAM_SCALES):
    combos = sorted(itertools.product(K_range, K_range), key=lambda t: t[0] + t[1])
    first_loose, best_miss = (None, None)
    for K_E, K_R in combos:
        for sc in lam_scales:
            ex = CB.exact_combined(pr, E_base, S_E, S_R, BE, BR, K_E, K_R, lam_base * sc, lam_base * sc)
            if ex['qubo_exact'] and first_loose is None:
                first_loose = dict(K_E=K_E, K_R=K_R, scale=sc, base_cost_match=ex['base_cost_match'])
            if ex['qubo_exact'] and ex['base_cost_match']:
                ex['first_qubo_exact_only'] = first_loose
                return (K_E, K_R, sc, ex)
            gap = abs(ex.get('min_feasible_energy', np.inf) - ex['H_star'])
            if best_miss is None or gap < best_miss['gap']:
                best_miss = dict(K_E=K_E, K_R=K_R, scale=sc, gap=gap)
    return (None, None, None, dict(best_miss=best_miss, first_qubo_exact_only=first_loose))

def exposure_from_pos(pos, D, field):
    N = len(pos)
    Rm = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            Rm[i, j] = Rm[j, i] = RB.mean_density_along(field, pos[i], pos[j])
    Dmax = float(np.max(D))
    Rm = Rm * np.asarray(D, float) / Dmax
    np.fill_diagonal(Rm, 0.0)
    return Rm

def generate_candidate(seed, width, N=5, scale=20.0):
    rng = np.random.default_rng(seed)
    env = V.P.WindWaveEnvironment()
    cells = [(r, c) for r in range(7) for c in range(7)]
    pos = [cells[i] for i in rng.choice(len(cells), N, replace=False)]
    D = np.array([[round(scale * np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])) for j in range(N)] for i in range(N)], float)
    E = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            aec = env.arc_energy_cost(pos[i], pos[j]) * (1.0 if i < j else 1.1)
            E[i, j] = D[i, j] * 0.05 + aec
    perm = list(rng.permutation(np.arange(1, N)))
    planted = [0] + perm + [0]
    t = 0.0
    arr = {0: 0.0}
    for a, b in zip(planted[:-1], planted[1:]):
        t += D[a, b]
        arr[b] = t
    e = np.zeros(N)
    l = np.zeros(N)
    for j in range(1, N):
        e[j] = max(0.0, arr[j] - rng.uniform(0, width))
        l[j] = arr[j] + rng.uniform(0, width)
    l[0] = t + width
    field = RB.density_field(seed)
    R = exposure_from_pos(pos, D, field)
    a1, a2, a3, a4 = V.W_DEF
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            C[i, j] = a1 * (2.0 * D[i, j] + 10.0) + a2 * D[i, j] / 60.0 + a3 * R[i, j] * D[i, j] + a4 * 0.5 * E[i, j]
    I = V.TourInstance(f'Jcand{seed}', 'J', D, e, l, C, E, R, None, None, dict(seed=seed, width=width, pos=[list(p) for p in pos], density_seed=seed))
    tours = [(p, m) for p, m in V.enumerate_tours(I) if m['tw_ok']]
    sel = select_budgets(tours)
    if sel is None:
        return None
    q = V.build_qubo_v2(I)
    if q.n > MAX_BASE_QUBITS:
        return None
    x_C, x_E, x_ER, BE, BR = (sel['x_C'], sel['x_E'], sel['x_ER'], sel['BE'], sel['BR'])
    assert x_C['energy'] > BE + COST_TOL, 'x_C must violate energy'
    assert x_E['energy'] <= BE + COST_TOL and x_E['risk'] > BR + COST_TOL, 'x_E must pass energy, fail risk'
    assert x_ER['energy'] <= BE + COST_TOL and x_ER['risk'] <= BR + COST_TOL, 'x_ER must pass both'
    I.BE = BE
    I.BR = BR
    I.meta.update(n_tw_feasible=sel['n_tw'], n_energy_feasible=sel['n_energy'], n_joint_feasible=sel['n_joint'], rel_gap_CE=sel['rel_gap_CE'], rel_gap_EER=sel['rel_gap_EER'], margin_E=sel['margin_E'], margin_R=sel['margin_R'], er_corr=sel['er_corr'], x_C_cost=x_C['cost'], x_E_cost=x_E['cost'], x_ER_cost=x_ER['cost'], x_C_energy=x_C['energy'], x_E_energy=x_E['energy'], x_ER_energy=x_ER['energy'], x_C_risk=x_C['risk'], x_E_risk=x_E['risk'], x_ER_risk=x_ER['risk'], x_C_energy_ok=bool(x_C['energy'] <= BE + 1e-09), x_C_risk_ok=bool(x_C['risk'] <= BR + 1e-09), x_E_risk_ok=bool(x_E['risk'] <= BR + 1e-09))
    return I
POOL_COLS = ['instance', 'seed', 'width', 'n_tw_feasible', 'n_energy_feasible', 'n_joint_feasible', 'x_C_cost', 'x_E_cost', 'x_ER_cost', 'rel_gap_CE', 'rel_gap_EER', 'BE', 'BR', 'margin_E', 'margin_R', 'er_corr']

def _append_pool_row(outdir, I):
    path = os.path.join(outdir, 'candidate_pool.csv')
    m = I.meta
    row = dict(instance=I.name, seed=m['seed'], width=m['width'], n_tw_feasible=m['n_tw_feasible'], n_energy_feasible=m['n_energy_feasible'], n_joint_feasible=m['n_joint_feasible'], x_C_cost=round(m['x_C_cost'], 4), x_E_cost=round(m['x_E_cost'], 4), x_ER_cost=round(m['x_ER_cost'], 4), rel_gap_CE=round(m['rel_gap_CE'], 6), rel_gap_EER=round(m['rel_gap_EER'], 6), BE=round(I.BE, 6), BR=round(I.BR, 6), margin_E=round(m['margin_E'], 6), margin_R=round(m['margin_R'], 6), er_corr=round(m['er_corr'], 4))
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=POOL_COLS)
        if new:
            w.writeheader()
        w.writerow(row)

def build_family(n_instances=4, seed_start=20000, seed_span=20000, outdir=OUTDIR, verbose=True, widths=WIDTHS, prefix='J'):
    os.makedirs(outdir, exist_ok=True)
    pool_path = os.path.join(outdir, 'candidate_pool.csv')
    if os.path.exists(pool_path):
        os.remove(pool_path)
    accepted = []
    tried = 0
    for seed in range(seed_start, seed_start + seed_span):
        if len(accepted) >= n_instances:
            break
        for width in widths:
            tried += 1
            I = generate_candidate(seed, width)
            if I is not None:
                I.name = f'{prefix}{len(accepted) + 1:02d}'
                accepted.append(I)
                _append_pool_row(outdir, I)
                m = I.meta
                if verbose:
                    print(f"  [{I.name}] seed={seed} width={width} | tw_feasible={m['n_tw_feasible']} energy_feasible={m['n_energy_feasible']} joint_feasible={m['n_joint_feasible']} | C(x_C)={m['x_C_cost']:.2f} < C(x_E)={m['x_E_cost']:.2f} < C(x_ER)={m['x_ER_cost']:.2f} | BE={I.BE:.3f} BR={I.BR:.3f} | margin_E={m['margin_E']:.4f} margin_R={m['margin_R']:.4f}")
                break
    if verbose:
        print(f'\n{len(accepted)}/{n_instances} Family J instances accepted after {tried} (seed,width) tries')
    return accepted

def calibrate_and_report(instances, outdir=OUTDIR, max_qubits_for_qaoa_hint=QAOA_MAX_QUBITS):
    rows, store = ([], {})
    for I in instances:
        pr = V.TourProblem(I)
        q = pr.qubo
        r_E = np.array([I.E[a] for a in q.arcs])
        r_R = np.array([I.R[a] for a in q.arcs])
        E_base = V.all_energies(q)
        S_E = CB.resource_sums(q.n, r_E)
        S_R = CB.resource_sums(q.n, r_R)
        K_E, K_R, sc, ex = calibrate_strict(pr, E_base, S_E, S_R, I.BE, I.BR, q.lam)
        m = I.meta
        qubits = q.n + K_E + K_R if K_E is not None else None
        row = dict(instance=I.name, seed=m['seed'], width=m['width'], n_tw_feasible=m['n_tw_feasible'], n_energy_feasible=m['n_energy_feasible'], n_joint_feasible=m['n_joint_feasible'], rel_gap_CE=round(m['rel_gap_CE'], 6), rel_gap_EER=round(m['rel_gap_EER'], 6), margin_E=round(m['margin_E'], 6), margin_R=round(m['margin_R'], 6), er_corr=round(m['er_corr'], 4), x_C_cost=round(m['x_C_cost'], 3), x_E_cost=round(m['x_E_cost'], 3), x_ER_cost=round(m['x_ER_cost'], 3), BE=round(I.BE, 4), BR=round(I.BR, 4), x_C_energy=round(m['x_C_energy'], 4), x_E_energy=round(m['x_E_energy'], 4), x_ER_energy=round(m['x_ER_energy'], 4), x_C_risk=round(m['x_C_risk'], 4), x_E_risk=round(m['x_E_risk'], 4), x_ER_risk=round(m['x_ER_risk'], 4), n_arcs=q.n, K_E=K_E, K_R=K_R, lam_scale=sc, qubits=qubits, qubo_argmin_feasible=ex.get('argmin_feasible') if K_E is not None else None, base_cost_match=ex.get('base_cost_match') if K_E is not None else False, qubo_exact=ex.get('qubo_exact') if K_E is not None else False, qaoa_tractable_guess=bool(qubits is not None and qubits <= max_qubits_for_qaoa_hint))
        rows.append(row)
        store[I.name] = dict(row=row, instance=I.to_dict(), calibration=ex)
        fl = (ex or {}).get('first_qubo_exact_only')
        if K_E is not None and fl and ((fl['K_E'], fl['K_R']) != (K_E, K_R)):
            print(f"      note: qubo_exact alone was reached earlier at K_E={fl['K_E']} K_R={fl['K_R']} (base_cost_match={fl['base_cost_match']}); strict calibration needed more qubits")
        print(f"  [{I.name}] n_arcs={q.n} -> K_E={K_E} K_R={K_R} (lamx{sc}) -> qubits={qubits} | base_cost_match={row['base_cost_match']} qubo_exact={row['qubo_exact']} | QAOA: {('plausible' if row['qaoa_tractable_guess'] else 'likely intractable')}")
    with open(os.path.join(outdir, 'joint_results.pkl'), 'wb') as f:
        pickle.dump(dict(rows=rows, store=store, code_sha=CODE_SHA), f)
    V.write_table(rows, ['instance', 'seed', 'width', 'n_tw_feasible', 'n_energy_feasible', 'n_joint_feasible', 'rel_gap_CE', 'rel_gap_EER', 'margin_E', 'margin_R', 'er_corr', 'x_C_cost', 'x_E_cost', 'x_ER_cost', 'BE', 'BR', 'x_C_energy', 'x_E_energy', 'x_ER_energy', 'x_C_risk', 'x_E_risk', 'x_ER_risk', 'n_arcs', 'K_E', 'K_R', 'lam_scale', 'qubits', 'qubo_argmin_feasible', 'base_cost_match', 'qubo_exact', 'qaoa_tractable_guess'], os.path.join(outdir, 'tables', 'T13_joint_family'), 'Family J: purpose-built joint energy+risk interaction instances, calibrated combined encoding.', 'tab:joint_family')
    return rows

def report(outdir=OUTDIR):
    path = os.path.join(outdir, 'joint_results.pkl')
    if not os.path.exists(path):
        print('no joint_results.pkl -- run --stage generate first')
        return
    d = pickle.load(open(path, 'rb'))
    for r in d['rows']:
        print(r)

def describe_argmin(all_tours, argmin_cost, BE, BR, tol=1e-06):
    if argmin_cost is None:
        return 'argmin is not a Hamiltonian tour'
    hits = [(p, m) for p, m in all_tours if abs(m['cost'] - argmin_cost) < tol]
    if len(hits) != 1:
        return f'no unique cost match ({len(hits)} tours)'
    p, m = hits[0]
    route = list(p) if isinstance(p, (list, tuple, np.ndarray)) else p
    return f"route={route} tw_ok={m['tw_ok']} energy={m['energy']:.4f} ({('<=' if m['energy'] <= BE + 1e-09 else '>')}BE) risk={m['risk']:.4f} ({('<=' if m['risk'] <= BR + 1e-09 else '>')}BR)"

def select_eligible(outdir=OUTDIR, max_qubits=QAOA_MAX_QUBITS):
    path = os.path.join(outdir, 'joint_results.pkl')
    if not os.path.exists(path):
        print('no joint_results.pkl -- run --stage generate first')
        return []
    rows = pickle.load(open(path, 'rb'))['rows']
    valid = [r for r in rows if r['qubo_exact'] and r['base_cost_match'] and (r['qubits'] is not None)]
    elig = [r for r in valid if r['qubits'] <= max_qubits]
    print(f'pool={len(rows)}  strict-exact={len(valid)}  eligible (<= {max_qubits} qubits)={len(elig)}')
    print(f"failed strict calibration: {[r['instance'] for r in rows if r not in valid]}")
    print(f"eligible with K_E=K_R=1: {sum((1 for r in elig if r['K_E'] == 1 and r['K_R'] == 1))}/{len(elig)}  (strict-exact instances above the ceiling use finer slack and are excluded purely by qubit count)")
    cols = ['instance', 'seed', 'width', 'n_arcs', 'K_E', 'K_R', 'lam_scale', 'qubits', 'rel_gap_CE', 'rel_gap_EER']
    for r in elig:
        print('  ' + '  '.join((f'{c}={r[c]}' for c in cols)))
    with open(os.path.join(outdir, 'qaoa_eligible.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in elig:
            w.writerow({c: r[c] for c in cols})
    return elig
ROBUST_KS = (1, 2, 4, 8, 16)
ROBUST_SCALES = (1, 50)
FINE_KS = (8, 16)
STRONG_SCALE = 50
EXT_SCALES = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 2000, 5000)

def robustness(outdir=OUTDIR, max_qubits=QAOA_MAX_QUBITS):
    path = os.path.join(outdir, 'joint_results.pkl')
    if not os.path.exists(path):
        print('no joint_results.pkl -- run --stage generate first')
        return []
    rows = pickle.load(open(path, 'rb'))['rows']
    elig = [r for r in rows if r['qubo_exact'] and r['base_cost_match'] and (r['qubits'] is not None) and (r['qubits'] <= max_qubits)]
    out = []
    for r in elig:
        s_star = r['lam_scale']
        scales = sorted(set(ROBUST_SCALES) | {s_star})
        I = generate_candidate(r['seed'], r['width'])
        if I is None:
            print(f"  [{r['instance']}] could not be regenerated")
            continue
        pr = V.TourProblem(I)
        q = pr.qubo
        r_E = np.array([I.E[a] for a in q.arcs])
        r_R = np.array([I.R[a] for a in q.arcs])
        E_base = V.all_energies(q)
        S_E = CB.resource_sums(q.n, r_E)
        S_R = CB.resource_sums(q.n, r_R)
        grid = {}
        for k in ROBUST_KS:
            for sc in scales:
                ex = CB.exact_combined(pr, E_base, S_E, S_R, I.BE, I.BR, k, k, q.lam * sc, q.lam * sc)
                grid[k, sc] = bool(ex['qubo_exact'] and ex['base_cost_match'])
        k_fine = max(FINE_KS)
        min_scale = None
        for sc in EXT_SCALES:
            ex = CB.exact_combined(pr, E_base, S_E, S_R, I.BE, I.BR, k_fine, k_fine, q.lam * sc, q.lam * sc)
            if ex['qubo_exact'] and ex['base_cost_match']:
                min_scale = sc
                break
        ex_s = CB.exact_combined(pr, E_base, S_E, S_R, I.BE, I.BR, k_fine, k_fine, q.lam * s_star, q.lam * s_star)
        k16_label = _stage_label(ex_s['argmin_cost'], I.meta)
        k1 = (r['K_E'], r['K_R']) == (1, 1)
        repro_ok = bool(abs(round(I.BE, 4) - r['BE']) < 0.001 and (grid[1, s_star] if k1 else True))
        if not repro_ok:
            print(f"  WARNING [{r['instance']}]: regenerated instance does not reproduce the stored calibration")
        if k1:
            same = all((grid[k, s_star] for k in FINE_KS))
            strong = all((grid[k, STRONG_SCALE] for k in FINE_KS))
            tier = 'robust_same_scale' if same else 'penalty_rescued' if strong else 'coarse_only'
        else:
            same = strong = None
            tier = 'n/a_selected_K>1'
        rec = dict(instance=r['instance'], seed=r['seed'], width=r['width'], qubits=r['qubits'], K_E=r['K_E'], K_R=r['K_R'], lam_scale_selected=s_star, tier=tier, fine_same_scale=same, fine_strong_penalty=strong, min_scale_K16=min_scale, K16_argmin_at_selected=k16_label, repro_ok=repro_ok, code_sha=CODE_SHA)
        for k in ROBUST_KS:
            for sc in scales:
                rec[f'exact_K{k}_x{sc}'] = grid[k, sc]
        out.append(rec)
        yn = lambda b: 'Y' if b else 'n'
        print(f"  [{rec['instance']}] selected K=({r['K_E']},{r['K_R']}) x{s_star} | tier={tier} | same-scale " + ' '.join((f'K{k}={yn(grid[k, s_star])}' for k in FINE_KS)) + ' | strong ' + ' '.join((f'K{k}={yn(grid[k, STRONG_SCALE])}' for k in FINE_KS)) + ' | grid: ' + ' '.join((f'K{k}x{sc}={yn(grid[k, sc])}' for k in ROBUST_KS for sc in scales)) + f" | K16@s*->{k16_label}, min_scale_K16={(min_scale if min_scale is not None else '>' + str(EXT_SCALES[-1]))}")
    if out:
        fields = []
        for o in out:
            for kk in o:
                if kk not in fields:
                    fields.append(kk)
        with open(os.path.join(outdir, 'robustness.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, restval='')
            w.writeheader()
            w.writerows(out)
    print(f'\ncode_sha={CODE_SHA}')
    for t in ('robust_same_scale', 'penalty_rescued', 'coarse_only', 'n/a_selected_K>1'):
        names = [o['instance'] for o in out if o['tier'] == t]
        if names:
            print(f'{t}: {len(names)}/{len(out)} {names}')
    return out

def _stage_label(cost, meta, tol=1e-06):
    if cost is None:
        return 'non-tour'
    for nm in ('x_C', 'x_E', 'x_ER'):
        if abs(cost - meta[f'{nm}_cost']) < tol:
            return nm
    return 'other'

def hierarchy(outdir=OUTDIR, max_qubits=QAOA_MAX_QUBITS, fine_k=16, fine_scale=STRONG_SCALE):
    path = os.path.join(outdir, 'joint_results.pkl')
    if not os.path.exists(path):
        print('no joint_results.pkl -- run --stage generate first')
        return []
    rows = pickle.load(open(path, 'rb'))['rows']
    elig = [r for r in rows if r['qubo_exact'] and r['base_cost_match'] and (r['qubits'] is not None) and (r['qubits'] <= max_qubits)]
    expected = dict(baseline='x_C', energy_only='x_E', joint='x_ER')
    out = []
    for r in elig:
        s_star = r['lam_scale']
        I = generate_candidate(r['seed'], r['width'])
        if I is None:
            print(f"  [{r['instance']}] could not be regenerated")
            continue
        pr = V.TourProblem(I)
        q = pr.qubo
        r_E = np.array([I.E[a] for a in q.arcs])
        r_R = np.array([I.R[a] for a in q.arcs])
        E_base = V.all_energies(q)
        S_E = CB.resource_sums(q.n, r_E)
        S_R = CB.resource_sums(q.n, r_R)
        all_tours = V.enumerate_tours(I)
        for tag, (kE, kR, sc) in (('selected', (r['K_E'], r['K_R'], s_star)), ('fine_ref', (fine_k, fine_k, fine_scale))):
            lam = q.lam * sc
            stage_E = dict(baseline=E_base, energy_only=CB.projected_combined(E_base, S_E, S_R, I.BE, I.BR, kE, kR, lam, 0.0), joint=CB.projected_combined(E_base, S_E, S_R, I.BE, I.BR, kE, kR, lam, lam))
            rec = dict(instance=r['instance'], encoding=tag, K_E=kE, K_R=kR, lam_scale=sc, code_sha=CODE_SHA)
            detail = {}
            for st, Ep in stage_E.items():
                info = pr.tour_states.get(int(np.argmin(Ep)))
                cost = info['cost'] if info else None
                rec[f'{st}_argmin'] = _stage_label(cost, I.meta)
                if rec[f'{st}_argmin'] != expected[st]:
                    detail[st] = describe_argmin(all_tours, cost, I.BE, I.BR)
            rec['hierarchy_ok'] = all((rec[f'{st}_argmin'] == expected[st] for st in expected))
            out.append(rec)
            print(f"  [{r['instance']}] {tag:8s} K=({kE},{kR}) x{sc:<3} | baseline->{rec['baseline_argmin']:6s} energy_only->{rec['energy_only_argmin']:6s} joint->{rec['joint_argmin']:6s} | ok={rec['hierarchy_ok']}")
            for st, d in detail.items():
                print(f'        {st}: {d}')
    if out:
        with open(os.path.join(outdir, 'hierarchy.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    print(f'\ncode_sha={CODE_SHA}')
    for tag in ('selected', 'fine_ref'):
        sub = [o for o in out if o['encoding'] == tag]
        print(f"{tag}: full hierarchy reproduced on {sum((o['hierarchy_ok'] for o in sub))}/{len(sub)}")
    return out

def diagnose_failed(outdir=OUTDIR):
    path = os.path.join(outdir, 'joint_results.pkl')
    if not os.path.exists(path):
        print('no joint_results.pkl -- run --stage generate first')
        return
    failed = [r for r in pickle.load(open(path, 'rb'))['rows'] if not (r['qubo_exact'] and r['base_cost_match'])]
    print(f"{len(failed)} instances failed strict calibration: {[r['instance'] for r in failed]}")
    for r in failed:
        print(f"\n=== {r['instance']} ===")
        diagnose(r['seed'], r['width'])

def _xer_gap(pr, E_base, S_E, S_R, I, k, sc, q, ex):
    if pr.R_star is None:
        return 'H(x_ER)-H*: n/a'
    xer = [st for st, info in pr.tour_states.items() if info['feas'] and abs(info['cost'] - pr.R_star) < 1e-06]
    if not xer:
        return 'H(x_ER)-H*: x_ER state not found'
    Ep = CB.projected_combined(E_base, S_E, S_R, I.BE, I.BR, k, k, q.lam * sc, q.lam * sc)
    return f"H(x_ER)-H*={float(Ep[xer[0]]) - ex['H_star']:.4f}"

def diagnose(seed, width, ks=(1, 2, 4, 8, 16), scales=(1, 50)):
    I = generate_candidate(seed, width)
    if I is None:
        print(f'seed={seed} width={width}: not an accepted candidate')
        return
    pr = V.TourProblem(I)
    q = pr.qubo
    r_E = np.array([I.E[a] for a in q.arcs])
    r_R = np.array([I.R[a] for a in q.arcs])
    E_base = V.all_energies(q)
    S_E = CB.resource_sums(q.n, r_E)
    S_R = CB.resource_sums(q.n, r_R)
    all_tours = V.enumerate_tours(I)
    m = I.meta
    print(f"seed={seed} width={width} n_arcs={q.n} BE={I.BE:.4f} BR={I.BR:.4f} R_star={pr.R_star} margin_E={m['margin_E']:.5f} margin_R={m['margin_R']:.5f}")
    for k in ks:
        for sc in scales:
            ex = CB.exact_combined(pr, E_base, S_E, S_R, I.BE, I.BR, k, k, q.lam * sc, q.lam * sc)
            print(f"  K_E=K_R={k:2d} lam x{sc:<3} qubo_exact={ex['qubo_exact']} base_cost_match={ex['base_cost_match']} argmin_feasible={ex['argmin_feasible']} argmin_cost={ex['argmin_cost']}" + ('' if ex['base_cost_match'] else '\n        argmin tour: ' + describe_argmin(all_tours, ex['argmin_cost'], I.BE, I.BR) + f"\n        QUBO: H*={ex['H_star']:.4f}  " + _xer_gap(pr, E_base, S_E, S_R, I, k, sc, q, ex)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='generate', choices=['generate', 'report', 'diagnose', 'diagnose_failed', 'select', 'robustness', 'hierarchy'])
    ap.add_argument('--n-instances', type=int, default=4)
    ap.add_argument('--seed-start', type=int, default=20000)
    ap.add_argument('--seed-span', type=int, default=20000)
    ap.add_argument('--widths', default=','.join((str(w) for w in WIDTHS)), help='comma-separated window widths, tried in order (first accepted width wins per seed)')
    ap.add_argument('--seed', type=int, default=None, help='diagnose: geometry seed')
    ap.add_argument('--width', type=int, default=None, help='diagnose: window width')
    ap.add_argument('--max-qubits', type=int, default=QAOA_MAX_QUBITS, help='select: QAOA qubit ceiling')
    ap.add_argument('--name-prefix', default='J', help='generate: instance-name prefix (use a new one for a second batch)')
    ap.add_argument('--outdir', default=OUTDIR)
    a = ap.parse_args()
    print(f'joint_family.py code_sha={CODE_SHA}')
    if a.stage == 'generate':
        widths = tuple((int(w) for w in a.widths.split(',')))
        instances = build_family(n_instances=a.n_instances, seed_start=a.seed_start, seed_span=a.seed_span, outdir=a.outdir, widths=widths, prefix=a.name_prefix)
        if instances:
            calibrate_and_report(instances, outdir=a.outdir)
        else:
            print("No Family J instances found -- widen seed_span or relax MIN_TW_FEASIBLE/MIN_ENERGY_FEASIBLE/MIN_JOINT_FEASIBLE explicitly (and say so), don't silently loosen them.")
    elif a.stage == 'diagnose':
        if a.seed is None or a.width is None:
            ap.error('--stage diagnose requires --seed and --width')
        diagnose(a.seed, a.width)
    elif a.stage == 'select':
        select_eligible(a.outdir, a.max_qubits)
    elif a.stage == 'robustness':
        robustness(a.outdir, a.max_qubits)
    elif a.stage == 'diagnose_failed':
        diagnose_failed(a.outdir)
    elif a.stage == 'hierarchy':
        hierarchy(a.outdir, a.max_qubits)
    else:
        report(a.outdir)
if __name__ == '__main__':
    main()
