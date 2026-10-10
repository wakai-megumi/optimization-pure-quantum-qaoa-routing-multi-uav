import os, sys, json, time, pickle, argparse, itertools, traceback
import numpy as np
from scipy.optimize import minimize
import gusrn_qaoa_patched as P
SEED = 42
TOL_SUCCESS = 0.05
EPS_EXACT = 1e-09
W_DEF = (0.3, 0.2, 0.25, 0.25)
PRESETS = {'quick': dict(n_seeds=1, n_restarts=2, n_shots=1000, p=2, depth=False), 'standard': dict(n_seeds=3, n_restarts=3, n_shots=2000, p=2, depth=True), 'full': dict(n_seeds=5, n_restarts=3, n_shots=2000, p=2, depth=True)}
DEPTHS = (1, 2, 3, 4, 5)
DEPTH_MAX_QUBITS = 16
MAX_QUBITS = 19
DERIVED_INSTANCES = json.loads('[{"name": "D01", "source": "Dumas et al. (1995) n20w20.003 via Lopez-Ibanez & Blum collection", "file": "n20w20.003", "customers": [9, 7, 15, 11], "bk_tour_len": 394.0, "tw_width": 20, "D": [[0.0, 36.0, 32.0, 32.0, 45.0], [36.0, 0.0, 22.0, 28.0, 38.0], [32.0, 22.0, 0.0, 6.0, 16.0], [32.0, 28.0, 6.0, 0.0, 14.0], [45.0, 38.0, 16.0, 14.0, 0.0]], "e": [0.0, 76.0, 109.0, 110.0, 143.0], "l": [417.0, 102.0, 119.0, 141.0, 160.0], "n_feasible_tours": 2, "n_qubits": 15}, {"name": "D02", "source": "Dumas et al. (1995) n20w20.004 via Lopez-Ibanez & Blum collection", "file": "n20w20.004", "customers": [15, 9, 5, 6], "bk_tour_len": 396.0, "tw_width": 20, "D": [[0.0, 22.0, 28.0, 31.0, 28.0], [22.0, 0.0, 11.0, 40.0, 44.0], [28.0, 11.0, 0.0, 36.0, 44.0], [31.0, 40.0, 36.0, 0.0, 14.0], [28.0, 44.0, 44.0, 14.0, 0.0]], "e": [0.0, 55.0, 67.0, 115.0, 117.0], "l": [408.0, 84.0, 92.0, 126.0, 142.0], "n_feasible_tours": 2, "n_qubits": 15}, {"name": "D03", "source": "Dumas et al. (1995) n20w20.005 via Lopez-Ibanez & Blum collection", "file": "n20w20.005", "customers": [8, 2, 5, 4], "bk_tour_len": 352.0, "tw_width": 20, "D": [[0.0, 27.0, 21.0, 22.0, 20.0], [27.0, 0.0, 20.0, 23.0, 27.0], [21.0, 20.0, 0.0, 3.0, 8.0], [22.0, 23.0, 3.0, 0.0, 6.0], [20.0, 27.0, 8.0, 6.0, 0.0]], "e": [0.0, 88.0, 125.0, 143.0, 141.0], "l": [375.0, 109.0, 156.0, 165.0, 169.0], "n_feasible_tours": 5, "n_qubits": 17}, {"name": "D04", "source": "Dumas et al. (1995) n20w40.001 via Lopez-Ibanez & Blum collection", "file": "n20w40.001", "customers": [13, 15, 18, 10], "bk_tour_len": 254.0, "tw_width": 40, "D": [[0.0, 12.0, 30.0, 32.0, 36.0], [12.0, 0.0, 19.0, 21.0, 28.0], [30.0, 19.0, 0.0, 3.0, 13.0], [32.0, 21.0, 3.0, 0.0, 11.0], [36.0, 28.0, 13.0, 11.0, 0.0]], "e": [0.0, 5.0, 155.0, 180.0, 244.0], "l": [295.0, 56.0, 213.0, 183.0, 258.0], "n_feasible_tours": 2, "n_qubits": 15}, {"name": "D05", "source": "Dumas et al. (1995) n20w40.001 via Lopez-Ibanez & Blum collection", "file": "n20w40.001", "customers": [7, 18, 16, 14], "bk_tour_len": 254.0, "tw_width": 40, "D": [[0.0, 26.0, 32.0, 33.0, 29.0], [26.0, 0.0, 52.0, 49.0, 32.0], [32.0, 52.0, 0.0, 6.0, 28.0], [33.0, 49.0, 6.0, 0.0, 22.0], [29.0, 32.0, 28.0, 22.0, 0.0]], "e": [0.0, 53.0, 180.0, 167.0, 196.0], "l": [295.0, 105.0, 183.0, 214.0, 239.0], "n_feasible_tours": 2, "n_qubits": 15}, {"name": "D06", "source": "Dumas et al. (1995) n20w40.003 via Lopez-Ibanez & Blum collection", "file": "n20w40.003", "customers": [10, 13, 6, 12], "bk_tour_len": 317.0, "tw_width": 40, "D": [[0.0, 21.0, 15.0, 14.0, 37.0], [21.0, 0.0, 13.0, 20.0, 47.0], [15.0, 13.0, 0.0, 7.0, 34.0], [14.0, 20.0, 7.0, 0.0, 28.0], [37.0, 47.0, 34.0, 28.0, 0.0]], "e": [0.0, 73.0, 78.0, 142.0, 156.0], "l": [372.0, 98.0, 102.0, 166.0, 191.0], "n_feasible_tours": 2, "n_qubits": 15}, {"name": "D07", "source": "Dumas et al. (1995) n20w60.001 via Lopez-Ibanez & Blum collection", "file": "n20w60.001", "customers": [3, 11, 14, 15], "bk_tour_len": 335.0, "tw_width": 60, "D": [[0.0, 31.0, 20.0, 19.0, 24.0], [31.0, 0.0, 15.0, 19.0, 42.0], [20.0, 15.0, 0.0, 4.0, 38.0], [19.0, 19.0, 4.0, 0.0, 40.0], [24.0, 42.0, 38.0, 40.0, 0.0]], "e": [0.0, 19.0, 0.0, 53.0, 151.0], "l": [436.0, 87.0, 79.0, 155.0, 182.0], "n_feasible_tours": 5, "n_qubits": 17}, {"name": "D08", "source": "Dumas et al. (1995) n20w60.001 via Lopez-Ibanez & Blum collection", "file": "n20w60.001", "customers": [10, 5, 9, 12], "bk_tour_len": 335.0, "tw_width": 60, "D": [[0.0, 7.0, 15.0, 28.0, 24.0], [7.0, 0.0, 12.0, 35.0, 31.0], [15.0, 12.0, 0.0, 39.0, 35.0], [28.0, 35.0, 39.0, 0.0, 5.0], [24.0, 31.0, 35.0, 5.0, 0.0]], "e": [0.0, 0.0, 29.0, 154.0, 120.0], "l": [436.0, 22.0, 103.0, 236.0, 222.0], "n_feasible_tours": 2, "n_qubits": 15}, {"name": "D09", "source": "Dumas et al. (1995) n20w60.002 via Lopez-Ibanez & Blum collection", "file": "n20w60.002", "customers": [12, 10, 20, 16], "bk_tour_len": 244.0, "tw_width": 60, "D": [[0.0, 25.0, 40.0, 31.0, 23.0], [25.0, 0.0, 16.0, 6.0, 2.0], [40.0, 16.0, 0.0, 10.0, 18.0], [31.0, 6.0, 10.0, 0.0, 8.0], [23.0, 2.0, 18.0, 8.0, 0.0]], "e": [0.0, 12.0, 75.0, 227.0, 191.0], "l": [321.0, 75.0, 141.0, 287.0, 243.0], "n_feasible_tours": 2, "n_qubits": 15}, {"name": "D10", "source": "Dumas et al. (1995) n20w80.001 via Lopez-Ibanez & Blum collection", "file": "n20w80.001", "customers": [19, 3, 11, 4], "bk_tour_len": 329.0, "tw_width": 80, "D": [[0.0, 11.0, 2.0, 30.0, 23.0], [11.0, 0.0, 11.0, 41.0, 18.0], [2.0, 11.0, 0.0, 30.0, 25.0], [30.0, 41.0, 30.0, 0.0, 45.0], [23.0, 18.0, 25.0, 45.0, 0.0]], "e": [0.0, 0.0, 2.0, 28.0, 226.0], "l": [441.0, 67.0, 66.0, 115.0, 298.0], "n_feasible_tours": 2, "n_qubits": 16}, {"name": "D11", "source": "Dumas et al. (1995) n20w80.001 via Lopez-Ibanez & Blum collection", "file": "n20w80.001", "customers": [5, 14, 17, 20], "bk_tour_len": 329.0, "tw_width": 80, "D": [[0.0, 23.0, 32.0, 34.0, 15.0], [23.0, 0.0, 12.0, 18.0, 37.0], [32.0, 12.0, 0.0, 25.0, 47.0], [34.0, 18.0, 25.0, 0.0, 47.0], [15.0, 37.0, 47.0, 47.0, 0.0]], "e": [0.0, 40.0, 80.0, 138.0, 283.0], "l": [441.0, 118.0, 162.0, 173.0, 373.0], "n_feasible_tours": 2, "n_qubits": 15}, {"name": "D12", "source": "Dumas et al. (1995) n20w100.001 via Lopez-Ibanez & Blum collection", "file": "n20w100.001", "customers": [15, 4, 1, 18], "bk_tour_len": 237.0, "tw_width": 100, "D": [[0.0, 18.0, 25.0, 26.0, 21.0], [18.0, 0.0, 19.0, 23.0, 35.0], [25.0, 19.0, 0.0, 4.0, 28.0], [26.0, 23.0, 4.0, 0.0, 27.0], [21.0, 35.0, 28.0, 27.0, 0.0]], "e": [0.0, 44.0, 0.0, 0.0, 147.0], "l": [405.0, 126.0, 158.0, 133.0, 213.0], "n_feasible_tours": 6, "n_qubits": 17}, {"name": "D13", "source": "Dumas et al. (1995) n20w100.001 via Lopez-Ibanez & Blum collection", "file": "n20w100.001", "customers": [11, 16, 8, 18], "bk_tour_len": 237.0, "tw_width": 100, "D": [[0.0, 10.0, 18.0, 22.0, 21.0], [10.0, 0.0, 12.0, 13.0, 17.0], [18.0, 12.0, 0.0, 8.0, 27.0], [22.0, 13.0, 8.0, 0.0, 24.0], [21.0, 17.0, 27.0, 24.0, 0.0]], "e": [0.0, 0.0, 31.0, 0.0, 147.0], "l": [405.0, 25.0, 156.0, 83.0, 213.0], "n_feasible_tours": 2, "n_qubits": 15}]')

def bits_of(idx, n):
    return (idx >> np.arange(n) & 1).astype(float)

def idx_of(x):
    return int(sum((1 << i for i, b in enumerate(x) if b > 0.5)))

def all_energies(qubo, chunk=1 << 16):
    n = qubo.n
    Ns = 1 << n
    out = np.empty(Ns)
    ar = np.arange(n)
    for s in range(0, Ns, chunk):
        ii = np.arange(s, min(Ns, s + chunk))
        X = (ii[:, None] >> ar & 1).astype(float)
        out[s:s + len(ii)] = np.einsum('bi,ij,bj->b', X, qubo.Q, X) + X @ qubo.g + qubo.C
    return out

class TourInstance:

    def __init__(self, name, family, D, e, l, C, E, R, BE, BR, meta):
        self.name, self.family = (name, family)
        self.D, self.e, self.l = (np.array(D, float), np.array(e, float), np.array(l, float))
        self.C, self.E, self.R = (np.array(C, float), np.array(E, float), np.array(R, float))
        self.BE, self.BR = (np.inf if BE is None else BE, np.inf if BR is None else BR)
        self.N = len(self.D)
        self.meta = meta

    def to_dict(self):
        f = lambda v: None if not np.isfinite(v) else float(v)
        return dict(name=self.name, family=self.family, N=self.N, D=self.D.tolist(), e=self.e.tolist(), l=self.l.tolist(), C=self.C.tolist(), E=self.E.tolist(), R=self.R.tolist(), BE=f(self.BE), BR=f(self.BR), meta=self.meta)

def tour_eval(I, tour):
    t = 0.0
    tw_ok = True
    for a, b in zip(tour[:-1], tour[1:]):
        t += I.D[a, b]
        if b != 0:
            t = max(t, I.e[b])
        if t > I.l[b] + 1e-09:
            tw_ok = False
    arcs = list(zip(tour[:-1], tour[1:]))
    en = sum((I.E[a, b] for a, b in arcs))
    rk = sum((I.R[a, b] for a, b in arcs))
    cost = sum((I.C[a, b] for a, b in arcs))
    batt = en <= I.BE + 1e-09
    sora = rk <= I.BR + 1e-09
    return dict(tw_ok=tw_ok, batt_ok=batt, sora_ok=sora, cost=float(cost), energy=float(en), risk=float(rk), feas=bool(tw_ok and batt and sora))

def enumerate_tours(I):
    return [(perm, tour_eval(I, [0] + list(perm) + [0])) for perm in itertools.permutations(range(1, I.N))]

def make_synthetic(name, seed, N=5, width=100, budget_mode='loose', scale=20.0):
    rng = np.random.default_rng(seed)
    env = P.WindWaveEnvironment()
    cells = [(r, c) for r in range(7) for c in range(7)]
    pos = [cells[i] for i in rng.choice(len(cells), N, replace=False)]
    D = np.array([[round(scale * np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])) for j in range(N)] for i in range(N)], float)
    sora = np.clip(rng.uniform(0.1, 0.9, N), 0.05, 0.95)
    E = np.zeros((N, N))
    R = np.zeros((N, N))
    C = np.zeros((N, N))
    a1, a2, a3, a4 = W_DEF
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            aec = env.arc_energy_cost(pos[i], pos[j]) * (1.0 if i < j else 1.1)
            E[i, j] = D[i, j] * 0.05 + aec
            R[i, j] = 0.5 * (sora[i] + sora[j])
            C[i, j] = a1 * (2.0 * D[i, j] + 10.0) + a2 * D[i, j] / 60.0 + a3 * R[i, j] * D[i, j] + a4 * 0.5 * E[i, j]
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
    I = TourInstance(name, 'S', D, e, l, C, E, R, None, None, dict(seed=seed, width=width, budget_mode=budget_mode, planted=planted, pos=[list(p) for p in pos], sora=sora.tolist()))
    tw_feas = [(p_, m) for p_, m in enumerate_tours(I) if m['tw_ok']]
    pl = tour_eval(I, planted)
    if budget_mode == 'loose':
        I.BE, I.BR = (1.5 * pl['energy'], 1.5 * pl['risk'])
    else:
        tw_best0 = min(tw_feas, key=lambda z: z[1]['cost'])[1]
        e_min = min((m['energy'] for _, m in tw_feas))
        I.BR = 1.5 * pl['risk']
        I.BE = e_min + 0.5 * (tw_best0['energy'] - e_min) if tw_best0['energy'] > e_min + 1e-06 else 1.5 * pl['energy']
    I.meta['n_tw_feasible_tours'] = len(tw_feas)
    tw_best = min(tw_feas, key=lambda z: z[1]['cost'])
    I.meta['budget_bites'] = bool(not tour_eval(I, [0] + list(tw_best[0]) + [0])['feas'])
    return I

def build_registry(max_qubits=MAX_QUBITS):
    reg = []
    cells = [(100, 'loose', 2), (200, 'loose', 2), (400, 'loose', 2), (200, 'binding', 2), (400, 'binding', 2), (30, 'loose', 2)]
    k = 0
    for ci, (w, bm, cnt) in enumerate(cells):
        got = 0
        seed = 1000 * (ci + 1)
        while got < cnt and seed < 1000 * (ci + 1) + 900:
            seed += 1
            I = make_synthetic(f'S{k + 1:02d}', seed, width=w, budget_mode=bm)
            feas = [m for _, m in enumerate_tours(I) if m['feas']]
            if not feas:
                continue
            if bm == 'loose' and w > 30 and (len(feas) < 2):
                continue
            if bm == 'binding' and (not (I.meta['budget_bites'] and I.meta['n_tw_feasible_tours'] >= 3)):
                continue
            q = build_qubo_v2(I)
            if q.n > max_qubits:
                continue
            I.meta['n_feasible_tours'] = len(feas)
            reg.append(I)
            got += 1
            k += 1
    for d in DERIVED_INSTANCES:
        I = TourInstance(d['name'], 'D', d['D'], d['e'], d['l'], d['D'], np.zeros((5, 5)), np.zeros((5, 5)), None, None, dict(source=d['source'], file=d['file'], customers=d['customers'], tw_width=d['tw_width'], bk_tour_len=d['bk_tour_len']))
        I.meta['n_feasible_tours'] = d['n_feasible_tours']
        if build_qubo_v2(I).n <= max_qubits:
            reg.append(I)
    return reg

class TinyQubo:

    def evaluate(self, x):
        return float(x @ self.Q @ x + self.g @ x + self.C)

def build_qubo_v2(I, lam_scale=2.0, prune=True, pair_nogoods=True):
    N = I.N
    if N > 5:
        raise ValueError('per-node in=out=1 + 2-cycle penalties give a single Hamiltonian cycle only for N<=5')
    LB = np.full(N, np.inf)
    LB[0] = 0.0
    done = set()
    while len(done) < N:
        u = min((v for v in range(N) if v not in done), key=lambda v: LB[v])
        done.add(u)
        for v in range(1, N):
            if v in done:
                continue
            LB[v] = min(LB[v], max(LB[u] + I.D[u, v], I.e[v]))
    dep = lambda i: 0.0 if i == 0 else max(LB[i], I.e[i])
    arcs = [(i, j) for i in range(N) for j in range(N) if i != j]
    if prune:
        arcs = [(i, j) for i, j in arcs if dep(i) + I.D[i, j] <= I.l[j] + 1e-09]
    idx = {a: k for k, a in enumerate(arcs)}
    n = len(arcs)
    cmax = max((I.C[a] for a in arcs))
    lam = lam_scale * N * cmax
    q = TinyQubo()
    q.n = n
    q.Q = np.zeros((n, n))
    q.g = np.zeros(n)
    q.C = 0.0
    q.arcs = arcs
    q.arc_index = idx
    q.lam = lam
    for a in arcs:
        q.g[idx[a]] += I.C[a]

    def one_hot(S):
        q.C += lam
        for a in S:
            q.g[idx[a]] -= lam
        for x_ in range(len(S)):
            for y_ in range(x_ + 1, len(S)):
                i, j = sorted((idx[S[x_]], idx[S[y_]]))
                q.Q[i, j] += 2 * lam
    for v in range(N):
        one_hot([a for a in arcs if a[0] == v])
        one_hot([a for a in arcs if a[1] == v])
    for i, j in arcs:
        if i < j and (j, i) in idx:
            a, b = sorted((idx[i, j], idx[j, i]))
            q.Q[a, b] += lam
    q.n_pair = 0
    if pair_nogoods:
        for i, j in arcs:
            for j2, k in arcs:
                if j2 != j or k == i or j == 0:
                    continue
                aj = max(dep(i) + I.D[i, j], I.e[j])
                if aj + I.D[j, k] > I.l[k] + 1e-09:
                    a, b = sorted((idx[i, j], idx[j, k]))
                    q.Q[a, b] += lam
                    q.n_pair += 1
    q.n_arcs_full = N * (N - 1)
    return q

class TourProblem:

    def __init__(self, I):
        self.I = I
        self.name = I.name
        self.family = I.family
        self.qubo = build_qubo_v2(I)
        self.n = self.qubo.n
        self.tour_states = {}
        for perm, m in enumerate_tours(I):
            tour = [0] + list(perm) + [0]
            arcs = list(zip(tour[:-1], tour[1:]))
            if all((a in self.qubo.arc_index for a in arcs)):
                s = sum((1 << self.qubo.arc_index[a] for a in arcs))
                self.tour_states[s] = dict(ham=True, cov=True, feas=m['feas'], cost=m['cost'], tw_ok=m['tw_ok'], batt_ok=m['batt_ok'], sora_ok=m['sora_ok'], energy=m['energy'], risk=m['risk'], tour=tour)
        feas = {s: i for s, i in self.tour_states.items() if i['feas']}
        self.feas_states = feas
        self.R_star = min((i['cost'] for i in feas.values()), default=None)
        self.n_feasible_tours = len(feas)
        self._none = dict(ham=False, cov=False, feas=False, cost=None, tw_ok=False, batt_ok=False, sora_ok=False, energy=None, risk=None, tour=None)

    def state_info(self, idx):
        return self.tour_states.get(idx, self._none)

    def describe(self):
        m = self.I.meta
        if self.family == 'S':
            return f"planted, width={m['width']}, budgets={m['budget_mode']}, seed={m['seed']}"
        return f"{m['file']} customers {m['customers']}"

class LegacyProblem:

    def __init__(self, key):
        self.name = key
        self.family = 'L'
        net = P.UAVNetwork(key, seed=SEED, env=P.WindWaveEnvironment())
        self.qubo = P.GUSRNQuboBuilder(net, lambda_visit=P.LV_MAP[key])
        self.n = self.qubo.n
        self.dec = P.RouteDecoder(net, self.qubo)
        self._cache = {}
        self.R_star = None
        self.n_feasible_tours = 0
        self.feas_states = {}

    def state_info(self, idx):
        if idx not in self._cache:
            m = self.dec.compute_metrics(self.dec.decode(bits_of(idx, self.n)))
            self._cache[idx] = dict(ham=bool(m['structure_ok'] and m['reachable_ok']), cov=bool(m['coverage_complete']), feas=bool(m['fully_feasible']), cost=m['cost'], tw_ok=m.get('tw_ok', False), batt_ok=m['battery_ok'], sora_ok=m['sora_ok'], energy=m['energy'], risk=m['risk'], tour=None)
        return self._cache[idx]

    def describe(self):
        return f'legacy instance {self.name} (lambda_visit={P.LV_MAP[self.name]})'

class FastEngine:

    def __init__(self, qubo):
        self.qubo = qubo
        self.n = qubo.n
        self.N_states = 1 << self.n
        self.energies = all_energies(qubo)

    def uniform_init(self):
        return np.full(self.N_states, 1.0 / np.sqrt(self.N_states), dtype=complex)

    def warm_init(self, thetas):
        psi = np.ones(1, dtype=complex)
        for i, th in enumerate(thetas):
            psi = np.kron(np.array([np.cos(th / 2), np.sin(th / 2)], dtype=complex), psi)
        return psi / np.linalg.norm(psi)

    def apply_cost_unitary(self, psi, gamma):
        return psi * np.exp(-1j * gamma * self.energies)

    def apply_standard_mixer(self, psi, beta):
        cb, sb = (np.cos(beta), -1j * np.sin(beta))
        for i in range(self.n):
            x = psi.reshape(1 << self.n - 1 - i, 2, 1 << i)
            psi = (cb * x + sb * x[:, ::-1, :]).reshape(-1)
        return psi

    def apply_warm_mixer(self, psi, beta, thetas):
        eib, emib = (np.exp(1j * beta), np.exp(-1j * beta))
        for i, th in enumerate(thetas):
            ct, st = (np.cos(th / 2), np.sin(th / 2))
            u00 = ct * ct * eib + st * st * emib
            u01 = ct * st * (eib - emib)
            u10 = u01
            u11 = st * st * eib + ct * ct * emib
            x = psi.reshape(1 << self.n - 1 - i, 2, 1 << i)
            out = np.empty_like(x)
            out[:, 0, :] = u00 * x[:, 0, :] + u01 * x[:, 1, :]
            out[:, 1, :] = u10 * x[:, 0, :] + u11 * x[:, 1, :]
            psi = out.reshape(-1)
        return psi

class FastCVaR(P.CVaRQAOA):

    def __init__(self, engine, p=2, alpha=0.25):
        super().__init__(engine, p=p, alpha=alpha)
        self._order = np.argsort(engine.energies, kind='stable')
        self._es = engine.energies[self._order]

    def cvar_objective(self, params):
        psi = self.run_circuit(params)
        pr = np.abs(psi) ** 2
        pr /= pr.sum()
        ps = pr[self._order]
        cs = np.cumsum(ps)
        k = int(np.searchsorted(cs, self.alpha))
        if k >= len(ps):
            return float(np.dot(ps, self._es) / cs[-1])
        mass = cs[k - 1] if k > 0 else 0.0
        return float((np.dot(ps[:k], self._es[:k]) + (self.alpha - mass) * self._es[k]) / self.alpha)

def relaxation(qubo, correct_gradient=True, n_starts=5):
    Qs = qubo.Q + qubo.Q.T
    jac = (lambda x: Qs @ x + qubo.g) if correct_gradient else lambda x: 2 * qubo.Q @ x + qubo.g
    best_f, best_x = (np.inf, None)
    for s in range(n_starts):
        np.random.seed(s)
        x0 = np.random.uniform(0, 1, qubo.n)
        r = minimize(qubo.evaluate, x0, jac=jac, method='L-BFGS-B', bounds=[(0, 1)] * qubo.n, options={'maxiter': 500})
        if r.fun < best_f:
            best_f, best_x = (r.fun, r.x.copy())
    return best_x

class WarmStartEps(P.WarmStartQAOA):

    def __init__(self, engine, p=2, eps=0.0001, correct_gradient=True):
        self.eng = engine
        self.qubo = engine.qubo
        self.p = p
        self.n_params = 2 * p
        self.eps = eps
        self.c_relax = relaxation(engine.qubo, correct_gradient)
        c = np.clip(self.c_relax, eps, 1 - eps)
        self.thetas = 2 * np.arcsin(np.sqrt(c))
        self.init_mode = f"relaxation(eps={eps},grad={('correct' if correct_gradient else 'legacy')})"

def theta_hat(thetas):
    th = np.minimum(thetas, np.pi - thetas)
    return dict(min=float(th.min()), median=float(np.median(th)), max=float(th.max()))

def summarize_idxs(problem, engine, idxs, Hstar):
    E = engine.energies
    H = E[idxs]
    bi = int(np.argmin(H))
    best_idx = int(idxs[bi])
    Hb = float(H[bi])
    uniq, cnt = np.unique(idxs, return_counts=True)
    ns = len(idxs)
    infos = [problem.state_info(int(u)) for u in uniq]
    rate = lambda key: float(sum((c for i, c in zip(infos, cnt) if i[key])) / ns)
    feas_costs = [i['cost'] for i in infos if i['feas'] and i['cost'] is not None]
    rq, gap = P.rho_qubo(Hstar, Hb)
    info_b = problem.state_info(best_idx)
    rho_route = None
    if problem.R_star is not None and info_b['feas'] and info_b['cost']:
        rho_route = problem.R_star / info_b['cost']
    return dict(best_of_shots=Hb, mean_shots=float(H.mean()), std_shots=float(H.std()), p_exact=float(np.mean(H <= Hstar + EPS_EXACT)), p_tol=float(np.mean((H - Hstar) / max(1.0, abs(Hstar)) <= TOL_SUCCESS)), ham_rate=rate('ham'), cov_rate=rate('cov'), feas_rate=rate('feas'), unique_states=int(len(uniq)), rho_qubo=rq, rho_qubo_gap=gap, reached_opt=bool(Hb <= Hstar + EPS_EXACT), best_shot_ham=bool(info_b['ham']), best_shot_feasible=bool(info_b['feas']), rho_route=rho_route, best_feasible_shot_cost=min(feas_costs) if feas_costs else None, reached_feasible_opt=bool(feas_costs and problem.R_star is not None and (min(feas_costs) <= problem.R_star + 1e-07)), rank_best=int((E < Hb - EPS_EXACT).sum()), best_idx=best_idx)

def sample_psi(problem, engine, psi, Hstar, n_shots, seed):
    pr = np.abs(psi) ** 2
    pr = np.maximum(pr, 0)
    pr /= pr.sum()
    idxs = np.random.default_rng(seed).choice(engine.N_states, size=n_shots, p=pr)
    return summarize_idxs(problem, engine, idxs, Hstar)

def exact_analysis(problem, engine):
    E = engine.energies
    Hs = float(E.min())
    bi = int(E.argmin())
    info = problem.state_info(bi)
    tol_mask = (E - Hs) / max(1.0, abs(Hs)) <= TOL_SUCCESS
    out = dict(n_qubits=problem.n, n_qubits_unpruned=getattr(problem.qubo, 'n_arcs_full', problem.n), H_star=Hs, n_optimal_states=int((E <= Hs + EPS_EXACT).sum()), argmin_idx=bi, argmin_ham=info['ham'], argmin_feasible=info['feas'], argmin_cost=info['cost'], p_random_tol=float(tol_mask.mean()), R_star_feas=problem.R_star, n_feasible_tours=problem.n_feasible_tours, H_uniform_mean=float(E.mean()), H_median=float(np.median(E)))
    if problem.feas_states:
        Ef = min((E[s] for s in problem.feas_states))
        out['rank_of_best_feasible'] = int((E < Ef - EPS_EXACT).sum())
        out['qubo_exact'] = bool(abs(Ef - Hs) < 1e-07 and info['feas'])
        out['min_feasible_energy'] = float(Ef)
    return out

def rrls(engine, n, budget, seed):
    rng = np.random.default_rng(seed)
    best_i, best_e, used = (None, np.inf, 0)
    while used < budget:
        i = int(rng.integers(engine.N_states))
        e = engine.energies[i]
        used += 1
        improved = True
        while improved and used < budget:
            improved = False
            for b in rng.permutation(n):
                j = i ^ 1 << int(b)
                ej = engine.energies[j]
                used += 1
                if ej < e - 1e-12:
                    i, e, improved = (j, ej, True)
                    break
                if used >= budget:
                    break
        if e < best_e:
            best_i, best_e = (i, e)
    return (best_i, float(best_e))

def run_baselines(problem, engine, Hstar, n_shots, legacy_grad=False):
    out = {}
    bench = P.ClassicalBenchmarks(problem.qubo)

    def rec(idx, H, t=None):
        info = problem.state_info(int(idx))
        rq, gap = P.rho_qubo(Hstar, float(H))
        rr = problem.R_star / info['cost'] if problem.R_star and info['feas'] and info['cost'] else None
        return dict(H=float(H), idx=int(idx), rho_qubo=rq, gap=gap, ham=info['ham'], feasible=info['feas'], reached_opt=bool(H <= Hstar + EPS_EXACT), rho_route=rr, time=t)
    t = time.time()
    xg, Hg = bench.greedy()
    out['Greedy'] = rec(idx_of(xg), Hg, time.time() - t)
    sa = []
    for s in range(10):
        t = time.time()
        xs, Hs_, _ = bench.simulated_annealing(n_iter=3000, seed=SEED + s)
        sa.append(rec(idx_of(xs), Hs_, time.time() - t))
    out['SA'] = sa
    out['RRLS'] = [rec(*rrls(engine, problem.n, n_shots, SEED + s)) for s in range(10)]
    xr = np.round(relaxation(problem.qubo, True))
    out['Relax-round'] = rec(idx_of(xr), problem.qubo.evaluate(xr))
    if legacy_grad:
        xl = np.round(relaxation(problem.qubo, False))
        out['Relax-round(legacy grad)'] = rec(idx_of(xl), problem.qubo.evaluate(xl))
    return out

def variant_builders(engine, p, legacy_grad):
    v = {'Standard': lambda: P.StandardQAOA(engine, p=p), 'CVaR': lambda: FastCVaR(engine, p=p, alpha=0.25), 'WS_eps1e-4': lambda: WarmStartEps(engine, p=p, eps=0.0001), 'WS_eps0.1': lambda: WarmStartEps(engine, p=p, eps=0.1), 'WS_eps0.25': lambda: WarmStartEps(engine, p=p, eps=0.25)}
    if legacy_grad:
        v['WS_legacygrad'] = lambda: WarmStartEps(engine, p=p, eps=0.0001, correct_gradient=False)
    return v

def run_variational(problem, engine, name, builder, p, n_restarts, seed, n_shots, Hstar):
    t0 = time.time()
    model = builder()
    is_ws = isinstance(model, P.WarmStartQAOA)
    psi0 = model.initial_psi() if is_ws else engine.uniform_init()
    d0 = P.distribution_diagnostics(psi0, engine)
    params, e, hist = model.optimize(n_restarts=n_restarts, seed=seed)
    psi = model.run_circuit(params)
    d1 = P.distribution_diagnostics(psi, engine)
    st = sample_psi(problem, engine, psi, Hstar, n_shots, seed)
    ctl = sample_psi(problem, engine, psi0, Hstar, n_shots, seed)
    ent = dict(seed=seed, energy=float(e), params=np.array(params), history=list(hist), P_max_before=d0['P_max'], P_max_after=d1['P_max'], exp_before=d0['expectation'], exp_after=d1['expectation'], entropy_after=d1['entropy'], N_eff_after=d1['N_eff'], stats=st, control=ctl, elapsed=time.time() - t0, is_ws=is_ws)
    if is_ws:
        ent['theta_hat'] = theta_hat(model.thetas)
        ent['init_mode'] = model.init_mode
        ent['relax_H'] = float(problem.qubo.evaluate(np.round(model.c_relax)))
    return ent

def run_problem(problem, cfg, verbose=True):
    t_all = time.time()
    engine = FastEngine(problem.qubo)
    ex = exact_analysis(problem, engine)
    Hs = ex['H_star']
    res = dict(name=problem.name, family=problem.family, desc=problem.describe(), exact=ex, cfg=dict(cfg))
    if verbose:
        print(f"  [{problem.name}] n={problem.n} qubits | H*={Hs:.3f} | feasible tours={ex['n_feasible_tours']} | argmin feasible={ex['argmin_feasible']} | exact QUBO={ex.get('qubo_exact')} | chance(5%)={ex['p_random_tol']:.4f}", flush=True)
    res['baselines'] = run_baselines(problem, engine, Hs, cfg['n_shots'], legacy_grad=problem.family == 'L')
    uni = [summarize_idxs(problem, engine, np.random.default_rng(SEED + s).integers(0, engine.N_states, cfg['n_shots']), Hs) for s in range(cfg['n_seeds'])]
    res['baselines']['Uniform'] = uni
    res['variants'] = {}
    for vname, builder in variant_builders(engine, cfg['p'], problem.family == 'L').items():
        runs = []
        for s in range(cfg['n_seeds']):
            runs.append(run_variational(problem, engine, vname, builder, cfg['p'], cfg['n_restarts'], SEED + 1000 * s, cfg['n_shots'], Hs))
        res['variants'][vname] = runs
        if verbose:
            r = [x['stats']['rho_qubo'] for x in runs]
            c = [x['control']['rho_qubo'] for x in runs]
            print(f"     {vname:14s} rho_QUBO(opt)={np.mean([x for x in r if x is not None]):.4f} control={np.mean([x for x in c if x is not None]):.4f} reached_opt={np.mean([x['stats']['reached_opt'] for x in runs]):.2f} feas_rate={np.mean([x['stats']['feas_rate'] for x in runs]):.3f} ({np.mean([x['elapsed'] for x in runs]):.0f}s/seed)", flush=True)
    res['elapsed'] = time.time() - t_all
    return res

def run_depth(problem, cfg, verbose=True):
    engine = FastEngine(problem.qubo)
    Hs = float(engine.energies.min())
    rows = []
    for mname, mk in [('Standard', lambda p: P.StandardQAOA(engine, p=p)), ('CVaR', lambda p: FastCVaR(engine, p=p)), ('WS_eps0.1', lambda p: WarmStartEps(engine, p=p, eps=0.1))]:
        for p in DEPTHS:
            for s in range(2):
                model = mk(p)
                seed = SEED + 1000 * s
                params, e, _ = model.optimize(n_restarts=cfg['n_restarts'], seed=seed)
                psi = model.run_circuit(params)
                st = sample_psi(problem, engine, psi, Hs, cfg['n_shots'], seed)
                rows.append(dict(method=mname, p=p, seed=seed, energy=float(e), rho_qubo=st['rho_qubo'], reached_opt=st['reached_opt'], p_exact=st['p_exact'], feas_rate=st['feas_rate']))
            if verbose:
                print(f'     depth {problem.name} {mname} p={p} done', flush=True)
    return dict(name=problem.name, H_star=Hs, rows=rows)

def get_problems(families, only=None, max_qubits=MAX_QUBITS):
    probs = []
    reg = build_registry(max_qubits)
    for I in reg:
        if I.family in families:
            probs.append(TourProblem(I))
    if 'L' in families:
        for k in 'ABCDEF':
            probs.append(LegacyProblem(k))
    if only:
        probs = [p for p in probs if p.name in only]
    return probs

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    return pickle.load(open(path, 'rb')) if os.path.exists(path) else {}

def stage_instances(outdir, families, max_qubits):
    reg = [I for I in build_registry(max_qubits) if I.family in families]
    json.dump([I.to_dict() for I in reg], open(os.path.join(outdir, 'instances_v2.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f'registry: {len(reg)} instances written to instances_v2.json')
    for I in reg:
        pr = TourProblem(I)
        print(f"  {I.name} [{I.family}] qubits={pr.n:2d} (of {pr.qubo.n_arcs_full}) feasible tours={pr.n_feasible_tours} R*={(pr.R_star if pr.R_star is None else round(pr.R_star, 2))}  {pr.describe()}{('  budget-binding' if I.meta.get('budget_bites') else '')}")

def stage_main(outdir, probs, cfg):
    path = os.path.join(outdir, 'main_results.pkl')
    R = load_pkl(path)
    for pr in probs:
        if pr.name in R and R[pr.name]['cfg'] == cfg:
            print(f'  [{pr.name}] cached, skipping')
            continue
        try:
            R[pr.name] = run_problem(pr, cfg)
            save_pkl(R, path)
            print(f"  [{pr.name}] DONE in {R[pr.name]['elapsed']:.0f}s", flush=True)
        except Exception as ex:
            print(f'  [{pr.name}] FAILED {type(ex).__name__}: {ex}')
            traceback.print_exc()
            save_pkl(R, path)

def stage_depth(outdir, probs, cfg):
    path = os.path.join(outdir, 'depth_results.pkl')
    R = load_pkl(path)
    for pr in probs:
        if pr.n > DEPTH_MAX_QUBITS or pr.name in R:
            continue
        R[pr.name] = run_depth(pr, cfg)
        save_pkl(R, path)
        print(f'  [{pr.name}] depth DONE', flush=True)

def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None

def _std(v):
    v = [x for x in v if x is not None]
    return float(np.std(v)) if len(v) > 1 else 0.0 if v else None

def _fmt(x, d=3):
    return '--' if x is None else f'{x:.{d}f}' if isinstance(x, (float, np.floating)) else str(x)

def write_table(rows, cols, path_base, caption, label):
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    with open(path_base + '.csv', 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(('' if r.get(c) is None else str(r.get(c)) for c in cols)) + '\n')
    esc = lambda s: str(s).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
    with open(path_base + '.tex', 'w') as f:
        f.write('\\begin{table}[t]\n\\centering\\small\n\\caption{' + esc(caption) + '}\\label{' + label + '}\n')
        f.write('\\begin{tabular}{' + 'l' * len(cols) + '}\n\\toprule\n' + ' & '.join((esc(c) for c in cols)) + ' \\\\\n\\midrule\n')
        for r in rows:
            f.write(' & '.join((esc(_fmt(r.get(c))) for c in cols)) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

def stage_report(outdir):
    R = load_pkl(os.path.join(outdir, 'main_results.pkl'))
    if not R:
        print('no main_results.pkl')
        return
    T = os.path.join(outdir, 'tables')
    names = sorted(R)
    rows = []
    for n in names:
        ex = R[n]['exact']
        rows.append(dict(instance=n, family=R[n]['family'], qubits=ex['n_qubits'], of_unpruned=ex['n_qubits_unpruned'], feasible_tours=ex['n_feasible_tours'], R_star=ex['R_star_feas'], H_star=ex['H_star'], optimal_states=ex['n_optimal_states'], argmin_is_cycle=ex['argmin_ham'], argmin_feasible=ex['argmin_feasible'], qubo_exact=ex.get('qubo_exact'), chance_5pct=ex['p_random_tol'], description=R[n]['desc']))
    write_table(rows, ['instance', 'family', 'qubits', 'of_unpruned', 'feasible_tours', 'R_star', 'H_star', 'optimal_states', 'argmin_is_cycle', 'argmin_feasible', 'qubo_exact', 'chance_5pct'], os.path.join(T, 'T1_instances_exactness'), 'Instances: size, feasible tours, and whether the exact QUBO optimum is a fully feasible best tour.', 'tab:v2_instances')
    with open(os.path.join(T, 'T1_instances_exactness_descriptions.csv'), 'w') as f:
        f.write('instance,description\n')
        [f.write(f'''{r['instance']},"{r['description']}"\n''') for r in rows]
    per = []
    agg = {}
    for n in names:
        r = R[n]
        for vname, runs in r['variants'].items():
            st = [x['stats'] for x in runs]
            ct = [x['control'] for x in runs]
            row = dict(instance=n, method=vname, seeds=len(runs), rho_qubo=_mean([s['rho_qubo'] for s in st]), rho_qubo_std=_std([s['rho_qubo'] for s in st]), reached_opt=_mean([s['reached_opt'] for s in st]), p_exact=_mean([s['p_exact'] for s in st]), p_5pct=_mean([s['p_tol'] for s in st]), cycle_rate=_mean([s['ham_rate'] for s in st]), feasible_rate=_mean([s['feas_rate'] for s in st]), best_shot_feasible=_mean([s['best_shot_feasible'] for s in st]), rho_route=_mean([s['rho_route'] for s in st]), log2_rank=_mean([np.log2(1 + s['rank_best']) for s in st]), control_rho_qubo=_mean([s['rho_qubo'] for s in ct]), control_reached_opt=_mean([s['reached_opt'] for s in ct]), control_feasible_rate=_mean([s['feas_rate'] for s in ct]), exp_before=_mean([x['exp_before'] for x in runs]), exp_after=_mean([x['exp_after'] for x in runs]), Pmax_before=_mean([x['P_max_before'] for x in runs]), Pmax_after=_mean([x['P_max_after'] for x in runs]), N_eff_after=_mean([x['N_eff_after'] for x in runs]), runtime_s=_mean([x['elapsed'] for x in runs]))
            per.append(row)
            agg.setdefault(vname, []).append(row)
        b = r['baselines']

        def brow(m, lst):
            lst = lst if isinstance(lst, list) else [lst]
            return dict(instance=n, method=m, seeds=len(lst), rho_qubo=_mean([x.get('rho_qubo') for x in lst]), rho_qubo_std=_std([x.get('rho_qubo') for x in lst]), reached_opt=_mean([x.get('reached_opt') for x in lst]), best_shot_feasible=_mean([x.get('feasible', x.get('best_shot_feasible')) for x in lst]), rho_route=_mean([x.get('rho_route') for x in lst]), runtime_s=_mean([x.get('time') for x in lst]))
        for m in ['Greedy', 'SA', 'RRLS', 'Relax-round', 'Relax-round(legacy grad)']:
            if m in b:
                per.append(brow(m, b[m]))
                agg.setdefault(m, []).append(per[-1])
        per.append(dict(instance=n, method='Uniform', seeds=len(b['Uniform']), rho_qubo=_mean([x['rho_qubo'] for x in b['Uniform']]), reached_opt=_mean([x['reached_opt'] for x in b['Uniform']]), feasible_rate=_mean([x['feas_rate'] for x in b['Uniform']])))
        agg.setdefault('Uniform', []).append(per[-1])
    cols = ['instance', 'method', 'seeds', 'rho_qubo', 'rho_qubo_std', 'reached_opt', 'p_exact', 'p_5pct', 'cycle_rate', 'feasible_rate', 'best_shot_feasible', 'rho_route', 'log2_rank', 'control_rho_qubo', 'control_reached_opt', 'exp_before', 'exp_after', 'Pmax_before', 'Pmax_after', 'N_eff_after', 'runtime_s']
    write_table(per, cols, os.path.join(T, 'T2_per_instance_method'), 'Per-instance results (mean over seeds).', 'tab:v2_per')
    for fam in ['ALL', 'S', 'D', 'L']:
        ag = []
        for m, rs in agg.items():
            rs_f = [x for x in rs if fam == 'ALL' or R[x['instance']]['family'] == fam]
            if not rs_f:
                continue
            ag.append(dict(method=m, instances=len(rs_f), rho_qubo=_mean([x['rho_qubo'] for x in rs_f]), reached_opt=_mean([x['reached_opt'] for x in rs_f]), best_shot_feasible=_mean([x.get('best_shot_feasible') for x in rs_f]), feasible_shot_rate=_mean([x.get('feasible_rate') for x in rs_f]), rho_route=_mean([x.get('rho_route') for x in rs_f]), runtime_s=_mean([x.get('runtime_s') for x in rs_f])))
        if ag:
            write_table(ag, ['method', 'instances', 'rho_qubo', 'reached_opt', 'best_shot_feasible', 'feasible_shot_rate', 'rho_route', 'runtime_s'], os.path.join(T, f'T3_summary_by_method_{fam}'), f'Method summary (family {fam}); macro-average over instances.', f'tab:v2_sum_{fam}')
    eff = []
    for vname in ['Standard', 'CVaR', 'WS_eps1e-4', 'WS_eps0.1', 'WS_eps0.25', 'WS_legacygrad']:
        rs = [x for x in per if x['method'] == vname]
        if not rs:
            continue
        better = sum((1 for x in rs if x['reached_opt'] is not None and x['control_reached_opt'] is not None and (x['reached_opt'] > x['control_reached_opt'] + 1e-09)))
        worse = sum((1 for x in rs if x['reached_opt'] is not None and x['control_reached_opt'] is not None and (x['reached_opt'] < x['control_reached_opt'] - 1e-09)))
        eff.append(dict(method=vname, instances=len(rs), mean_exp_change=_mean([x['exp_after'] - x['exp_before'] for x in rs if x['exp_after'] is not None]), mean_Pmax_before=_mean([x['Pmax_before'] for x in rs]), mean_Pmax_after=_mean([x['Pmax_after'] for x in rs]), reached_opt_opt=_mean([x['reached_opt'] for x in rs]), reached_opt_control=_mean([x['control_reached_opt'] for x in rs]), instances_qaoa_better=better, instances_qaoa_worse=worse))
    write_table(eff, ['method', 'instances', 'mean_exp_change', 'mean_Pmax_before', 'mean_Pmax_after', 'reached_opt_opt', 'reached_opt_control', 'instances_qaoa_better', 'instances_qaoa_worse'], os.path.join(T, 'T4_optimizer_effect_vs_control'), 'Effect of the QAOA layers relative to sampling the initial state with no layers.', 'tab:v2_effect')
    ws = []
    for n in names:
        for vname in ['WS_eps1e-4', 'WS_eps0.1', 'WS_eps0.25']:
            runs = R[n]['variants'][vname]
            th = runs[0]['theta_hat']
            ws.append(dict(instance=n, variant=vname, theta_hat_min=th['min'], theta_hat_max=th['max'], relax_rounding_H=runs[0]['relax_H'], H_star=R[n]['exact']['H_star'], Pmax_before=_mean([x['P_max_before'] for x in runs]), Pmax_after=_mean([x['P_max_after'] for x in runs]), exp_before=_mean([x['exp_before'] for x in runs]), exp_after=_mean([x['exp_after'] for x in runs]), cobyla_first=_mean([x['history'][0] for x in runs]), cobyla_best=_mean([min(x['history']) for x in runs]), gamma_beta=str(np.round(runs[0]['params'], 3).tolist())))
    write_table(ws, ['instance', 'variant', 'theta_hat_min', 'theta_hat_max', 'relax_rounding_H', 'H_star', 'Pmax_before', 'Pmax_after', 'exp_before', 'exp_after', 'cobyla_first', 'cobyla_best', 'gamma_beta'], os.path.join(T, 'T5_warmstart_diagnostics'), 'Warm-start diagnostics: pole distance, probability concentration, optimiser trajectory and parameters.', 'tab:v2_ws')
    D = load_pkl(os.path.join(outdir, 'depth_results.pkl'))
    if D:
        dr = []
        for n, d in sorted(D.items()):
            for m in ['Standard', 'CVaR', 'WS_eps0.1']:
                for p in DEPTHS:
                    rr = [x for x in d['rows'] if x['method'] == m and x['p'] == p]
                    dr.append(dict(instance=n, method=m, p=p, rho_qubo=_mean([x['rho_qubo'] for x in rr]), reached_opt=_mean([x['reached_opt'] for x in rr]), p_exact=_mean([x['p_exact'] for x in rr]), feasible_rate=_mean([x['feas_rate'] for x in rr]), energy=_mean([x['energy'] for x in rr])))
        write_table(dr, ['instance', 'method', 'p', 'rho_qubo', 'reached_opt', 'p_exact', 'feasible_rate', 'energy'], os.path.join(T, 'T6_depth'), 'Depth sweep p=1..5.', 'tab:v2_depth')
    print('\n=== SUMMARY (family ALL): method | mean rho_QUBO | reached H* | best shot feasible | feasible-shot rate | runtime')
    for m, rs in sorted(agg.items()):
        print(f"  {m:24s} {_fmt(_mean([x['rho_qubo'] for x in rs]), 4):>7} {_fmt(_mean([x['reached_opt'] for x in rs]), 3):>7} {_fmt(_mean([x.get('best_shot_feasible') for x in rs]), 3):>7} {_fmt(_mean([x.get('feasible_rate') for x in rs]), 3):>7} {_fmt(_mean([x.get('runtime_s') for x in rs]), 1):>8}")
    print(f'\nTables (CSV + LaTeX) written to {T}')

def selftest():
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{('PASS' if cond else 'FAIL')}] {name}")
    reg = build_registry()
    chk(f'registry built ({len(reg)} instances)', len(reg) > 0)
    pr = TourProblem(reg[0])
    n = pr.n
    base = P.StatevectorEngine(pr.qubo)
    fast = FastEngine(pr.qubo)
    chk('energies equal to patched engine', np.allclose(base.energies, fast.energies))
    rng = np.random.default_rng(1)
    th = rng.uniform(0.05, 3.0, n)
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi /= np.linalg.norm(psi)
    chk('standard mixer == patched', np.allclose(base.apply_standard_mixer(psi.copy(), 0.37), fast.apply_standard_mixer(psi.copy(), 0.37), atol=1e-10))
    chk('warm mixer == patched', np.allclose(base.apply_warm_mixer(psi.copy(), 0.41, th), fast.apply_warm_mixer(psi.copy(), 0.41, th), atol=1e-10))
    chk('warm init == patched', np.allclose(base.warm_init(th), fast.warm_init(th), atol=1e-12))
    pr_ = rng.uniform(0, 2 * np.pi, 4)
    a = P.CVaRQAOA(base, p=2).cvar_objective(pr_)
    b = FastCVaR(fast, p=2).cvar_objective(pr_)
    chk(f'CVaR objective == patched ({a:.6f} vs {b:.6f})', abs(a - b) < 1e-08)
    from scipy.optimize import check_grad
    x = rng.uniform(0, 1, n)
    Qs = pr.qubo.Q + pr.qubo.Q.T
    chk('corrected relaxation gradient matches finite differences', check_grad(pr.qubo.evaluate, lambda z: Qs @ z + pr.qubo.g, x) < 0.001)
    chk('legacy gradient does NOT (documents the bug)', check_grad(pr.qubo.evaluate, lambda z: 2 * pr.qubo.Q @ z + pr.qubo.g, x) > 0.01)
    bad = 0
    for I in reg:
        p_ = TourProblem(I)
        for s, info in p_.tour_states.items():
            if abs(all_energies(p_.qubo)[s] - info['cost']) > 1e-06 if p_.n <= 14 else False:
                bad += 1
    chk('valid-tour energy == route cost (offset-free) on n<=14 instances', bad == 0)
    ex = {p_.name: exact_analysis(p_, FastEngine(p_.qubo)) for p_ in [TourProblem(I) for I in reg[:6]]}
    chk('exact QUBO optimum == best feasible tour (first 6 instances)', all((v.get('qubo_exact') for v in ex.values())))
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='standard', choices=list(PRESETS))
    ap.add_argument('--stage', default='all', choices=['all', 'instances', 'main', 'depth', 'report'])
    ap.add_argument('--families', default='S,D')
    ap.add_argument('--only', default='')
    ap.add_argument('--outdir', default='./gusrn_v2_results')
    ap.add_argument('--max-qubits', type=int, default=MAX_QUBITS)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.outdir, exist_ok=True)
    cfg = PRESETS[a.preset]
    fam = set(a.families.split(','))
    only = set(a.only.split(',')) if a.only else None
    t0 = time.time()
    print(f'GUSRN-QAOA v2 | preset={a.preset} {cfg} | families={sorted(fam)} | outdir={a.outdir}')
    if a.stage in ('all', 'instances'):
        stage_instances(a.outdir, fam, a.max_qubits)
    probs = get_problems(fam, only, a.max_qubits) if a.stage in ('all', 'main', 'depth') else []
    if a.stage in ('all', 'main'):
        print('\nSTAGE MAIN')
        stage_main(a.outdir, probs, cfg)
    if a.stage in ('all', 'depth') and cfg['depth']:
        print('\nSTAGE DEPTH')
        stage_depth(a.outdir, probs, cfg)
    if a.stage in ('all', 'report'):
        stage_report(a.outdir)
    print(f'\nTOTAL {time.time() - t0:.0f}s')
import sys
if __name__ == '__main__':
    main()
