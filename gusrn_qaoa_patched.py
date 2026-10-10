import time
import numpy as np
import networkx as nx
from scipy.optimize import minimize
SEED = 42
N_SHOTS = 2000
SUCCESS_TOL = 0.05
TW_EPS = 1e-09

class WindWaveEnvironment:

    def __init__(self, size=7):
        self.size = size
        self.wind_grid = np.zeros((size, size))
        self.wave_grid = np.zeros((size, size))
        self._build_wind_field()
        self._build_wave_field()
        self.combined = self.wind_grid + self.wave_grid
        self.convolved = self._apply_convolution(self.combined)

    def _build_wind_field(self):
        for i in range(self.size):
            for j in range(self.size):
                if i > j:
                    self.wind_grid[i, j] = 3
                elif i < j:
                    self.wind_grid[i, j] = -3

    def _build_wave_field(self):
        for i in range(self.size):
            for j in range(self.size):
                if i < self.size / 2:
                    self.wave_grid[j, i] = -1
                elif i > self.size / 2:
                    self.wave_grid[j, i] = 1

    def _apply_convolution(self, grid):
        conv = np.zeros_like(grid)
        s = grid.shape[0]
        for i in range(1, s - 1):
            for j in range(1, s - 1):
                conv[i, j] = grid[i - 1, j - 1] + grid[i - 1, j] + grid[i - 1, j + 1] + grid[i, j - 1] + grid[i, j] + grid[i, j + 1] + grid[i + 1, j - 1] + grid[i + 1, j] + grid[i + 1, j + 1]
        conv[0, :] = conv[1, :]
        conv[-1, :] = conv[-2, :]
        conv[:, 0] = conv[:, 1]
        conv[:, -1] = conv[:, -2]
        return conv

    def arc_energy_cost(self, node_i_pos, node_j_pos):
        ri, ci = node_i_pos
        rj, cj = node_j_pos
        ri = min(max(ri, 0), self.size - 1)
        ci = min(max(ci, 0), self.size - 1)
        rj = min(max(rj, 0), self.size - 1)
        cj = min(max(cj, 0), self.size - 1)
        wind_penalty = max(0, self.convolved[ri, ci]) + max(0, self.convolved[rj, cj])
        dist = np.sqrt((ri - rj) ** 2 + (ci - cj) ** 2)
        return dist + 0.1 * wind_penalty

class UAVNetwork:
    INSTANCE_TEMPLATES = {'A': {'nodes': {0: {'type': 'D', 'pos': (3, 1)}, 1: {'type': 'W', 'pos': (3, 3)}, 2: {'type': 'T', 'pos': (1, 5)}, 3: {'type': 'T', 'pos': (5, 5)}, 4: {'type': 'T', 'pos': (3, 6)}}, 'edges': [(0, 1, 30.0), (0, 2, 50.0), (1, 2, 35.0), (1, 3, 45.0), (1, 4, 59.0), (2, 4, 40.0), (3, 4, 25.0)], 'time_windows': {0: (0, 100), 1: (5, 80), 2: (15, 50), 3: (20, 60), 4: (30, 90)}, 'battery_demand': {0: 0, 1: 5, 2: 12, 3: 15, 4: 18}, 'sora_zone': {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.7, 4: 0.4}}, 'B': {'nodes': {0: {'type': 'D', 'pos': (3, 1)}, 1: {'type': 'W', 'pos': (1, 3)}, 2: {'type': 'W', 'pos': (5, 3)}, 3: {'type': 'T', 'pos': (1, 6)}, 4: {'type': 'T', 'pos': (5, 6)}}, 'edges': [(0, 1, 100.0), (0, 2, 100.0), (0, 3, 200.0), (1, 3, 150.0), (1, 4, 250.0), (2, 3, 250.0), (2, 4, 150.0), (3, 4, 160.0), (1, 2, 80.0), (0, 4, 400.0)], 'time_windows': {0: (0, 200), 1: (10, 150), 2: (10, 150), 3: (30, 120), 4: (40, 180)}, 'battery_demand': {0: 0, 1: 8, 2: 8, 3: 20, 4: 25}, 'sora_zone': {0: 0.1, 1: 0.3, 2: 0.2, 3: 0.8, 4: 0.6}}, 'C': {'nodes': {0: {'type': 'D', 'pos': (3, 1)}, 1: {'type': 'W', 'pos': (1, 3)}, 2: {'type': 'W', 'pos': (5, 3)}, 3: {'type': 'T', 'pos': (2, 6)}, 4: {'type': 'T', 'pos': (4, 6)}}, 'edges': [(0, 1, 230.0), (0, 2, 230.0), (1, 3, 220.0), (2, 4, 220.0), (1, 4, 350.0), (2, 3, 350.0), (3, 4, 300.0), (0, 3, 400.0), (0, 4, 400.0)], 'time_windows': {0: (0, 300), 1: (20, 160), 2: (20, 160), 3: (50, 140), 4: (60, 150)}, 'battery_demand': {0: 0, 1: 12, 2: 12, 3: 30, 4: 30}, 'sora_zone': {0: 0.1, 1: 0.3, 2: 0.25, 3: 0.85, 4: 0.9}}, 'D': {'nodes': {0: {'type': 'D', 'pos': (3, 1)}, 1: {'type': 'W', 'pos': (1, 3)}, 2: {'type': 'W', 'pos': (5, 3)}, 3: {'type': 'T', 'pos': (1, 5)}, 4: {'type': 'T', 'pos': (5, 5)}, 5: {'type': 'T', 'pos': (3, 7)}}, 'edges': [(0, 1, 80.0), (0, 2, 120.0), (1, 3, 90.0), (1, 4, 160.0), (2, 3, 160.0), (2, 4, 90.0), (3, 5, 110.0), (4, 5, 95.0), (0, 5, 280.0)], 'time_windows': {0: (0, 300), 1: (15, 160), 2: (15, 160), 3: (40, 130), 4: (40, 130), 5: (80, 200)}, 'battery_demand': {0: 0, 1: 10, 2: 10, 3: 22, 4: 22, 5: 35}, 'sora_zone': {0: 0.1, 1: 0.25, 2: 0.3, 3: 0.6, 4: 0.65, 5: 0.5}}, 'E': {'nodes': {0: {'type': 'D', 'pos': (3, 1)}, 1: {'type': 'W', 'pos': (2, 3)}, 2: {'type': 'W', 'pos': (4, 3)}, 3: {'type': 'T', 'pos': (1, 6)}, 4: {'type': 'T', 'pos': (5, 6)}}, 'edges': [(0, 1, 50.0), (0, 2, 70.0), (0, 3, 180.0), (0, 4, 300.0), (1, 2, 40.0), (1, 3, 110.0), (1, 4, 200.0), (2, 3, 130.0), (2, 4, 160.0), (3, 4, 80.0)], 'time_windows': {0: (0, 200), 1: (8, 60), 2: (8, 60), 3: (25, 80), 4: (35, 90)}, 'battery_demand': {0: 0, 1: 7, 2: 7, 3: 18, 4: 22}, 'sora_zone': {0: 0.1, 1: 0.4, 2: 0.45, 3: 0.85, 4: 0.9}}, 'F': {'nodes': {0: {'type': 'D', 'pos': (3, 1)}, 1: {'type': 'W', 'pos': (1, 3)}, 2: {'type': 'W', 'pos': (3, 3)}, 3: {'type': 'W', 'pos': (5, 3)}, 4: {'type': 'T', 'pos': (1, 6)}, 5: {'type': 'T', 'pos': (3, 6)}, 6: {'type': 'T', 'pos': (5, 6)}}, 'edges': [(0, 1, 100.0), (0, 3, 100.0), (1, 2, 80.0), (2, 4, 120.0), (2, 5, 90.0), (3, 5, 130.0), (3, 6, 95.0), (4, 5, 60.0), (5, 6, 70.0)], 'time_windows': {0: (0, 400), 1: (20, 180), 2: (20, 180), 3: (20, 180), 4: (60, 160), 5: (70, 170), 6: (80, 200)}, 'battery_demand': {0: 0, 1: 12, 2: 12, 3: 12, 4: 25, 5: 28, 6: 30}, 'sora_zone': {0: 0.1, 1: 0.3, 2: 0.35, 3: 0.3, 4: 0.7, 5: 0.75, 6: 0.8}}}

    def __init__(self, instance_key='A', seed=42, env=None, weights=(0.3, 0.2, 0.25, 0.25)):
        np.random.seed(seed)
        self.instance_key = instance_key
        self.env = env or WindWaveEnvironment(size=7)
        self.weights = weights
        tmpl = self.INSTANCE_TEMPLATES[instance_key]
        self.node_info = tmpl['nodes']
        self.all_nodes = sorted(self.node_info.keys())
        self.depot = 0
        self.waypoint_ids = [n for n, v in self.node_info.items() if v['type'] == 'W']
        self.target_ids = [n for n, v in self.node_info.items() if v['type'] == 'T']
        self.pos = {n: v['pos'] for n, v in self.node_info.items()}
        self.time_window = tmpl['time_windows']
        self.battery_demand = tmpl['battery_demand']
        self.sora_zone = {n: min(0.95, max(0.05, tmpl['sora_zone'][n] + np.random.uniform(-0.05, 0.05))) for n in self.all_nodes}
        self.sparse_edge_defs = tmpl['edges']
        self.G = self._build_sparse_graph()
        self.edges = list(self.G.edges())
        self.n_edges = len(self.edges)
        self.edge_index = {e: i for i, e in enumerate(self.edges)}

    def _build_sparse_graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.all_nodes)
        a1, a2, a3, a4 = self.weights
        c1_unit, c_fixed, CE, speed = (2.0, 10.0, 0.5, 60.0)
        for i, j, dist_km in self.sparse_edge_defs:
            pi, pj = (self.pos[i], self.pos[j])
            energy = dist_km * 0.05 + self.env.arc_energy_cost(pi, pj)
            sora = 0.5 * (self.sora_zone[i] + self.sora_zone[j])
            c_travel = c1_unit * dist_km
            c_energy = CE * energy
            c_time = dist_km / speed
            total = a1 * (c_travel + c_fixed) + a2 * c_time + a3 * sora * dist_km + a4 * c_energy
            G.add_edge(i, j, distance=dist_km, energy=energy, sora_risk=sora, travel_cost=c_travel, total_cost=total)
            if not G.has_edge(j, i):
                energy_rev = dist_km * 0.05 + self.env.arc_energy_cost(pj, pi) * 1.1
                sora_rev = 0.5 * (self.sora_zone[j] + self.sora_zone[i])
                total_rev = a1 * (c_travel + c_fixed) + a2 * c_time + a3 * sora_rev * dist_km + a4 * CE * energy_rev
                G.add_edge(j, i, distance=dist_km, energy=energy_rev, sora_risk=sora_rev, travel_cost=c_travel, total_cost=total_rev)
        return G

class GUSRNQuboBuilder:

    def __init__(self, network, lambda_flow=5.0, lambda_visit=60.0, lambda_tw=1.0, lambda_bat=2.0, lambda_sora=1.5, lambda_depot=20.0, battery_max=500.0, battery_min=10.0, risk_max=10.0, alpha_tw=0.5, beta_tw=1.0):
        self.net = network
        self.lf, self.lv, self.ltw = (lambda_flow, lambda_visit, lambda_tw)
        self.lb, self.ls, self.ld = (lambda_bat, lambda_sora, lambda_depot)
        self.bat_max, self.bat_min = (battery_max, battery_min)
        self.risk_max = risk_max
        self.alpha, self.beta = (alpha_tw, beta_tw)
        self.n = network.n_edges
        self.Q = np.zeros((self.n, self.n))
        self.g = np.zeros(self.n)
        self.C = 0.0
        self._build()

    def _ei(self, e):
        return self.net.edge_index[e]

    def _entering(self, v):
        return [e for e in self.net.edges if e[1] == v]

    def _leaving(self, v):
        return [e for e in self.net.edges if e[0] == v]

    def _add_arc_costs(self):
        for e in self.net.edges:
            self.g[self._ei(e)] += self.net.G.edges[e]['total_cost']

    def _add_visit_constraints(self):
        for t in self.net.target_ids:
            in_e = self._entering(t)
            if not in_e:
                continue
            self.C += self.lv
            for e in in_e:
                self.g[self._ei(e)] -= self.lv
            for a in range(len(in_e)):
                for b in range(a + 1, len(in_e)):
                    ia, ib = (self._ei(in_e[a]), self._ei(in_e[b]))
                    i, j = (min(ia, ib), max(ia, ib))
                    self.Q[i, j] += 2.0 * self.lv

    def _add_depot_constraints(self):
        for edge_set in [self._leaving(self.net.depot), self._entering(self.net.depot)]:
            if not edge_set:
                continue
            self.C += self.ld
            for e in edge_set:
                self.g[self._ei(e)] -= self.ld
            for a in range(len(edge_set)):
                for b in range(a + 1, len(edge_set)):
                    ia, ib = (self._ei(edge_set[a]), self._ei(edge_set[b]))
                    i, j = (min(ia, ib), max(ia, ib))
                    self.Q[i, j] += 2.0 * self.ld

    def _add_flow_conservation(self):
        for v in self.net.all_nodes:
            if v == self.net.depot:
                continue
            in_e = self._entering(v)
            out_e = self._leaving(v)
            all_e = in_e + out_e
            signs = [+1] * len(in_e) + [-1] * len(out_e)
            for ea, sa in zip(all_e, signs):
                self.g[self._ei(ea)] += self.lf * sa * sa
            for ai in range(len(all_e)):
                for bi in range(ai + 1, len(all_e)):
                    ia, ib = (self._ei(all_e[ai]), self._ei(all_e[bi]))
                    sa, sb = (signs[ai], signs[bi])
                    i, j = (min(ia, ib), max(ia, ib))
                    self.Q[i, j] += 2.0 * self.lf * sa * sb

    def _add_time_window_penalty(self):
        try:
            sp = nx.single_source_dijkstra_path_length(self.net.G, self.net.depot, weight='distance')
        except Exception:
            sp = {}
        for e in self.net.edges:
            i, j = e
            idx = self._ei(e)
            est = sp.get(j, self.net.G.edges[e]['distance'])
            ej, lj = self.net.time_window[j]
            early = self.alpha * max(0.0, ej - est)
            late = self.beta * max(0.0, est - lj)
            self.g[idx] += self.ltw * (early + late)

    def _add_battery_constraints(self):
        budget = self.bat_max - self.bat_min
        for e in self.net.edges:
            en = self.net.G.edges[e]['energy']
            if en > budget:
                self.g[self._ei(e)] += self.lb * (en - budget) ** 2
        for v in self.net.all_nodes:
            for ea in self._entering(v):
                for eb in self._leaving(v):
                    en_a = self.net.G.edges[ea]['energy']
                    en_b = self.net.G.edges[eb]['energy']
                    if en_a + en_b > budget:
                        over = en_a + en_b - budget
                        ia, ib = (self._ei(ea), self._ei(eb))
                        i, j = (min(ia, ib), max(ia, ib))
                        self.Q[i, j] += self.lb * over

    def _add_sora_constraints(self):
        per_arc_limit = self.risk_max / max(1, len(self.net.edges) // 3)
        for e in self.net.edges:
            sr = self.net.G.edges[e]['sora_risk']
            if sr > per_arc_limit:
                self.g[self._ei(e)] += self.ls * sr
        for ea in self.net.edges:
            for eb in self.net.edges:
                if ea >= eb:
                    continue
                sa = self.net.G.edges[ea]['sora_risk']
                sb = self.net.G.edges[eb]['sora_risk']
                if sa + sb > self.risk_max * 0.4:
                    ia, ib = (self._ei(ea), self._ei(eb))
                    i, j = (min(ia, ib), max(ia, ib))
                    self.Q[i, j] += self.ls * sa * sb * 0.3

    def _build(self):
        self._add_arc_costs()
        self._add_visit_constraints()
        self._add_depot_constraints()
        self._add_flow_conservation()
        self._add_time_window_penalty()
        self._add_battery_constraints()
        self._add_sora_constraints()
        for i in range(self.n):
            self.g[i] += self.Q[i, i]
            self.Q[i, i] = 0.0

    def evaluate(self, x):
        return float(x @ self.Q @ x + self.g @ x + self.C)

class StatevectorEngine:

    def __init__(self, qubo):
        self.qubo = qubo
        self.n = qubo.n
        self.N_states = 2 ** self.n
        self.bitstrings = np.array([[s >> i & 1 for i in range(self.n)] for s in range(self.N_states)], dtype=float)
        self.energies = self._compute_all_energies()
        all_idx = np.arange(self.N_states)
        self.flip_indices = [all_idx ^ 1 << i for i in range(self.n)]

    def _compute_all_energies(self):
        xQx = np.einsum('bi,ij,bj->b', self.bitstrings, self.qubo.Q, self.bitstrings)
        gx = self.bitstrings @ self.qubo.g
        return xQx + gx + self.qubo.C

    def uniform_init(self):
        return np.ones(self.N_states, dtype=complex) / np.sqrt(self.N_states)

    def warm_init(self, thetas):
        psi = np.ones(self.N_states, dtype=complex)
        for i, theta in enumerate(thetas):
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            bit_i = np.arange(self.N_states) >> i & 1
            psi *= np.where(bit_i == 0, c, s)
        norm = np.linalg.norm(psi)
        return psi / norm if norm > 1e-15 else psi

    def apply_cost_unitary(self, psi, gamma):
        return psi * np.exp(-1j * gamma * self.energies)

    def apply_standard_mixer(self, psi, beta):
        cb, sb = (np.cos(beta), np.sin(beta))
        for flip_idx in self.flip_indices:
            psi = cb * psi - 1j * sb * psi[flip_idx]
        return psi

    def apply_warm_mixer(self, psi, beta, thetas):
        for i, theta in enumerate(thetas):
            ct, st = (np.cos(theta / 2), np.sin(theta / 2))
            eib = np.exp(1j * beta)
            emib = np.exp(-1j * beta)
            u00 = ct * ct * eib + st * st * emib
            u01 = ct * st * (eib - emib)
            u10 = st * ct * (eib - emib)
            u11 = st * st * eib + ct * ct * emib
            bit_i = np.arange(self.N_states) >> i & 1
            idx0 = np.where(bit_i == 0)[0]
            idx1 = np.where(bit_i == 1)[0]
            psi_new = psi.copy()
            psi_new[idx0] = u00 * psi[idx0] + u01 * psi[idx1]
            psi_new[idx1] = u10 * psi[idx0] + u11 * psi[idx1]
            psi = psi_new
        return psi

def sample_shots(psi, engine, qubo, H_star, n_shots=N_SHOTS, tol=SUCCESS_TOL, seed=None):
    rng = np.random.default_rng(seed)
    probs = np.abs(psi) ** 2
    probs = np.maximum(probs, 0)
    probs = probs / probs.sum()
    idxs = rng.choice(engine.N_states, size=n_shots, p=probs)
    H_samples = engine.energies[idxs]
    best_idx_pos = np.argmin(H_samples)
    best_idx = idxs[best_idx_pos]
    best_x = engine.bitstrings[best_idx]
    best_H = H_samples[best_idx_pos]
    denom = max(1.0, abs(H_star))
    rel_gap = (H_samples - H_star) / denom
    success_prob = float(np.mean(rel_gap <= tol))
    return dict(best_of_shots=float(best_H), mean_shots=float(H_samples.mean()), std_shots=float(H_samples.std()), success_probability=success_prob, success_tol=tol, unique_states_sampled=int(len(np.unique(idxs))), best_x=best_x, best_idx=int(best_idx), all_idxs=idxs)

def distribution_diagnostics(psi, engine):
    probs = np.abs(psi) ** 2
    probs = np.maximum(probs, 0)
    probs = probs / probs.sum()
    P_max = float(probs.max())
    expectation = float(np.dot(probs, engine.energies))
    nz = probs[probs > 1e-15]
    entropy = float(-np.sum(nz * np.log(nz)))
    N_eff = float(1.0 / np.sum(probs ** 2))
    return dict(P_max=P_max, expectation=expectation, entropy=entropy, N_eff=N_eff)

def theta_hat_summary(thetas):
    theta_hat = np.minimum(thetas, np.pi - thetas)
    return dict(theta_hat_min=float(theta_hat.min()), theta_hat_median=float(np.median(theta_hat)), theta_hat_max=float(theta_hat.max()), theta_hat_all=theta_hat.tolist())

class RouteDecoder:

    def __init__(self, network, qubo, battery_max=500.0, battery_min=10.0, risk_max=10.0):
        self.net = network
        self.qubo = qubo
        self.battery_budget = battery_max - battery_min
        self.risk_max = risk_max

    def decode(self, x):
        return [e for idx, e in enumerate(self.net.edges) if x[idx] > 0.5]

    def compute_metrics(self, selected_edges):
        if not selected_edges:
            return dict(cost=0, time=0, risk=0, energy=0, n_edges=0, coverage_complete=False, fully_feasible=False, tw_violations=[], structure_ok=False, reachable_ok=False, battery_ok=False, sora_ok=False)
        total_cost = sum((self.net.G.edges[e]['travel_cost'] for e in selected_edges))
        total_dist = sum((self.net.G.edges[e]['distance'] for e in selected_edges))
        total_risk = sum((self.net.G.edges[e]['sora_risk'] for e in selected_edges))
        total_energy = sum((self.net.G.edges[e]['energy'] for e in selected_edges))
        visited = set((j for i, j in selected_edges if j in self.net.target_ids))
        coverage_complete = all((t in visited for t in self.net.target_ids))
        adj = {}
        for i, j in selected_edges:
            adj.setdefault(i, []).append(j)
        arrival = {self.net.depot: 0.0}
        stack = [self.net.depot]
        edge_dist = {(i, j): self.net.G.edges[i, j]['distance'] for i, j in selected_edges}
        visited_order = set()
        while stack:
            u = stack.pop()
            if u in visited_order:
                continue
            visited_order.add(u)
            for w in adj.get(u, []):
                cand = arrival.get(u, 0.0) + edge_dist.get((u, w), 0.0)
                if w not in arrival or cand < arrival[w]:
                    arrival[w] = cand
                stack.append(w)
        cum = 0.0
        twp = 0.0
        tw_violations = []
        for e in selected_edges:
            cum += self.net.G.edges[e]['distance']
            j = e[1]
            if j in self.net.time_window:
                ej, lj = self.net.time_window[j]
                aj = arrival.get(j, cum)
                early = max(0, ej - aj)
                late = 2 * max(0, aj - lj)
                twp += early + late
                if not ej - TW_EPS <= aj <= lj + TW_EPS:
                    tw_violations.append((j, aj, ej, lj))
        z2_route_time_surrogate = total_dist + twp
        src = np.array([e[0] for e in selected_edges])
        dst = np.array([e[1] for e in selected_edges])
        structure_ok = True
        for v in self.net.all_nodes:
            indeg = int((dst == v).sum())
            outdeg = int((src == v).sum())
            if v == self.net.depot:
                if indeg != 1 or outdeg != 1:
                    structure_ok = False
                    break
            else:
                if indeg == 0 and outdeg == 0:
                    continue
                if indeg != 1 or outdeg != 1:
                    structure_ok = False
                    break
        reachable_ok = False
        if structure_ok:
            adjr = {}
            for a, b in zip(src.tolist(), dst.tolist()):
                adjr.setdefault(a, []).append(b)
            seen = {self.net.depot}
            st = [self.net.depot]
            while st:
                u = st.pop()
                for w in adjr.get(u, []):
                    if w not in seen:
                        seen.add(w)
                        st.append(w)
            sel_nodes = set(src.tolist()) | set(dst.tolist())
            reachable_ok = sel_nodes.issubset(seen)
        battery_ok = total_energy <= self.battery_budget
        sora_ok = total_risk <= self.risk_max
        tw_ok = len(tw_violations) == 0
        fully_feasible = bool(coverage_complete and structure_ok and reachable_ok and battery_ok and sora_ok and tw_ok)
        return dict(cost=total_cost, time=z2_route_time_surrogate, risk=total_risk, energy=total_energy, n_edges=len(selected_edges), coverage_complete=coverage_complete, fully_feasible=fully_feasible, structure_ok=structure_ok, reachable_ok=reachable_ok, battery_ok=battery_ok, sora_ok=sora_ok, tw_ok=tw_ok, tw_violations=tw_violations)

def rho_qubo(H_star, H_alg):
    if H_star is None or H_alg is None:
        return (None, None)
    if H_star > 0 and H_alg > 0:
        return (H_star / H_alg, None)
    if H_star < 0 and H_alg < 0:
        return (H_alg / H_star, None)
    return (None, H_alg - H_star)

def best_feasible_travel_cost(engine, decoder):
    best_cost, best_idx = (None, None)
    for idx in range(engine.N_states):
        x = engine.bitstrings[idx]
        sel = decoder.decode(x)
        m = decoder.compute_metrics(sel)
        if m['fully_feasible']:
            if best_cost is None or m['cost'] < best_cost:
                best_cost, best_idx = (m['cost'], idx)
    return (best_cost, best_idx)

def rho_travel(R_star_feas_cost, alg_metrics):
    if R_star_feas_cost is None:
        return (None, 'no fully feasible state exists for this instance')
    if not alg_metrics['fully_feasible']:
        return (None, "algorithm's decoded route is not fully_feasible")
    if alg_metrics['cost'] <= 0:
        return (None, 'algorithm travel cost <= 0 (degenerate)')
    return (R_star_feas_cost / alg_metrics['cost'], None)

class StandardQAOA:
    name = 'Standard QAOA'

    def __init__(self, engine, p=2):
        self.eng = engine
        self.qubo = engine.qubo
        self.p = p
        self.n_params = 2 * p

    def run_circuit(self, params):
        gammas = params[:self.p]
        betas = params[self.p:]
        psi = self.eng.uniform_init()
        for k in range(self.p):
            psi = self.eng.apply_cost_unitary(psi, gammas[k])
            psi = self.eng.apply_standard_mixer(psi, betas[k])
        return psi

    def expectation(self, params):
        psi = self.run_circuit(params)
        probs = np.abs(psi) ** 2
        return float(np.dot(probs, self.eng.energies))

    def optimize(self, n_restarts=3, seed=0, verbose=False):
        best_params, best_e, best_hist = (None, np.inf, [])
        for r in range(n_restarts):
            np.random.seed(seed + r * 17)
            p0 = np.random.uniform(0, 2 * np.pi, self.n_params)
            hist = []

            def obj(p):
                e = self.expectation(p)
                hist.append(e)
                return e
            res = minimize(obj, p0, method='COBYLA', options={'maxiter': 300, 'rhobeg': 0.5})
            if res.fun < best_e:
                best_e, best_params, best_hist = (res.fun, res.x, hist)
        return (best_params, best_e, best_hist)

class CVaRQAOA:

    def __init__(self, engine, p=2, alpha=0.25):
        self.eng = engine
        self.qubo = engine.qubo
        self.p = p
        self.alpha = alpha
        self.name = f'CVaR QAOA (α={alpha})'
        self.n_params = 2 * p

    @staticmethod
    def compute_cvar(probabilities, values, alpha):
        probs = np.array(probabilities)
        vals = np.array(values)
        order = np.argsort(vals)
        probs_s = probs[order]
        vals_s = vals[order]
        cvar, total_prob = (0.0, 0.0)
        for p, v in zip(probs_s, vals_s):
            if total_prob + p > alpha:
                p = alpha - total_prob
            total_prob += p
            cvar += p * v
            if total_prob >= alpha - 1e-12:
                break
        return cvar / alpha

    def run_circuit(self, params):
        gammas = params[:self.p]
        betas = params[self.p:]
        psi = self.eng.uniform_init()
        for k in range(self.p):
            psi = self.eng.apply_cost_unitary(psi, gammas[k])
            psi = self.eng.apply_standard_mixer(psi, betas[k])
        return psi

    def cvar_objective(self, params):
        psi = self.run_circuit(params)
        probs = np.abs(psi) ** 2
        probs = np.maximum(probs, 0)
        probs /= probs.sum()
        return self.compute_cvar(probs, self.eng.energies, self.alpha)

    def optimize(self, n_restarts=3, seed=0, verbose=False):
        best_params, best_e, best_hist = (None, np.inf, [])
        for r in range(n_restarts):
            np.random.seed(seed + r * 17)
            p0 = np.random.uniform(0, 2 * np.pi, self.n_params)
            hist = []

            def obj(p):
                e = self.cvar_objective(p)
                hist.append(e)
                return e
            res = minimize(obj, p0, method='COBYLA', options={'maxiter': 300, 'rhobeg': 0.5})
            if res.fun < best_e:
                best_e, best_params, best_hist = (res.fun, res.x, hist)
        return (best_params, best_e, best_hist)

class WarmStartQAOA:
    name = 'Warm-Start QAOA'

    def __init__(self, engine, p=2, greedy_x=None):
        self.eng = engine
        self.qubo = engine.qubo
        self.p = p
        self.n_params = 2 * p
        self.thetas, self.init_mode = self._compute_thetas(greedy_x)

    def _compute_thetas(self, greedy_x):
        n = self.qubo.n
        best_x, best_c = (None, np.inf)
        for seed in range(5):
            np.random.seed(seed)
            x0 = np.random.uniform(0, 1, n)
            res = minimize(lambda x: self.qubo.evaluate(x), x0, jac=lambda x: 2 * self.qubo.Q @ x + self.qubo.g, method='L-BFGS-B', bounds=[(0, 1)] * n, options={'maxiter': 500})
            if res.fun < best_c:
                best_c, best_x = (res.fun, res.x.copy())
        c_stars = np.clip(best_x, 0.0001, 1 - 0.0001)
        if c_stars.max() < 0.05 and greedy_x is not None:
            np.random.seed(0)
            c_stars = np.clip(greedy_x + np.random.uniform(-0.1, 0.1, n), 0.0001, 1 - 0.0001)
            mode = 'greedy_fallback'
        else:
            mode = 'relaxation'
        thetas = 2 * np.arcsin(np.sqrt(c_stars))
        return (thetas, mode)

    def initial_psi(self):
        return self.eng.warm_init(self.thetas)

    def run_circuit(self, params):
        gammas = params[:self.p]
        betas = params[self.p:]
        psi = self.eng.warm_init(self.thetas)
        for k in range(self.p):
            psi = self.eng.apply_cost_unitary(psi, gammas[k])
            psi = self.eng.apply_warm_mixer(psi, betas[k], self.thetas)
        return psi

    def expectation(self, params):
        psi = self.run_circuit(params)
        probs = np.abs(psi) ** 2
        return float(np.dot(probs, self.eng.energies))

    def optimize(self, n_restarts=3, seed=0, verbose=False):
        best_params, best_e, best_hist = (None, np.inf, [])
        for r in range(n_restarts):
            np.random.seed(seed + r * 17)
            p0 = np.array([np.pi, np.pi / 2] * self.p) if r == 0 else np.random.uniform(0, 2 * np.pi, self.n_params)
            p0 += np.random.uniform(-0.1, 0.1, self.n_params)
            hist = []

            def obj(p):
                e = self.expectation(p)
                hist.append(e)
                return e
            res = minimize(obj, p0, method='COBYLA', options={'maxiter': 300, 'rhobeg': 0.5})
            if res.fun < best_e:
                best_e, best_params, best_hist = (res.fun, res.x, hist)
        return (best_params, best_e, best_hist)

class ClassicalBenchmarks:

    def __init__(self, qubo):
        self.qubo = qubo
        self.n = qubo.n

    def simulated_annealing(self, T_init=10.0, T_final=0.01, n_iter=3000, seed=0):
        np.random.seed(seed)
        x = np.random.randint(0, 2, self.n).astype(float)
        c = self.qubo.evaluate(x)
        best_x, best_c = (x.copy(), c)
        T = T_init
        alpha = (T_final / T_init) ** (1.0 / n_iter)
        hist = [c]
        for _ in range(n_iter):
            flip = np.random.randint(self.n)
            xn = x.copy()
            xn[flip] = 1 - xn[flip]
            cn = self.qubo.evaluate(xn)
            if cn < c or np.random.rand() < np.exp(-(cn - c) / T):
                x, c = (xn, cn)
                if c < best_c:
                    best_c = c
                    best_x = x.copy()
            T *= alpha
            hist.append(best_c)
        return (best_x, best_c, hist)

    def greedy(self):
        x = np.zeros(self.n)
        order = np.argsort(self.qubo.g)
        for idx in order:
            x[idx] = 1
            base = self.qubo.evaluate(x)
            x[idx] = 0
            if self.qubo.evaluate(x) > base:
                x[idx] = 1
        return (x, self.qubo.evaluate(x))

def theta_hat_summary(ws_qaoa):
    theta = ws_qaoa.thetas
    theta_hat = np.minimum(theta, np.pi - theta)
    return dict(theta_hat_min=float(theta_hat.min()), theta_hat_median=float(np.median(theta_hat)), theta_hat_max=float(theta_hat.max()), theta_hat_array=theta_hat)

def timed_greedy(bench):
    t0 = time.time()
    x, cost = bench.greedy()
    return (x, cost, time.time() - t0)

def timed_sa(bench, n_iter=3000, seed=SEED):
    t0 = time.time()
    x, cost, hist = bench.simulated_annealing(n_iter=n_iter, seed=seed)
    return (x, cost, hist, time.time() - t0)

def run_D_chain(engine, qubo, decoder, bench, p=2, n_restarts=3, seed=SEED):
    x_gr, gr_cost, gr_time = timed_greedy(bench)
    gr_metrics = decoder.compute_metrics(decoder.decode(x_gr))
    ws = WarmStartQAOA(engine, p=p, greedy_x=x_gr)
    psi_init = ws.initial_psi()
    diag_init = distribution_diagnostics(psi_init, engine)
    init_probs = np.abs(psi_init) ** 2
    init_best_idx = np.argmax(init_probs)
    init_metrics = decoder.compute_metrics(decoder.decode(engine.bitstrings[init_best_idx]))
    theta_diag = theta_hat_summary(ws)
    wp, we, wh = ws.optimize(n_restarts=n_restarts, seed=seed)
    psi_final = ws.run_circuit(wp)
    diag_final = distribution_diagnostics(psi_final, engine)
    final_probs = np.abs(psi_final) ** 2
    final_best_idx = np.argmax(final_probs)
    final_metrics = decoder.compute_metrics(decoder.decode(engine.bitstrings[final_best_idx]))
    return dict(H_greedy=gr_cost, greedy_metrics=gr_metrics, greedy_time=gr_time, H_WS_init=float(engine.energies[init_best_idx]), WS_init_metrics=init_metrics, diag_init=diag_init, theta_hat=theta_diag, H_WS_final=float(engine.energies[final_best_idx]), WS_final_metrics=final_metrics, diag_final=diag_final, ws_params=wp, ws_history=wh, init_mode=ws.init_mode, x_greedy=x_gr)
WEIGHT_CONFIGS = {'baseline': (0.3, 0.2, 0.25, 0.25), 'travel-emph': (0.55, 0.15, 0.15, 0.15), 'time-emph': (0.15, 0.55, 0.15, 0.15), 'risk-emph': (0.15, 0.15, 0.55, 0.15), 'energy-emph': (0.15, 0.15, 0.15, 0.55)}

def run_C_sweep(instance_key, seed=SEED, lv=None, p=2, n_restarts=3):
    results = {}
    for name, w in WEIGHT_CONFIGS.items():
        env = WindWaveEnvironment()
        net = UAVNetwork(instance_key=instance_key, seed=seed, env=env, weights=w)
        qubo = GUSRNQuboBuilder(net, lambda_visit=lv)
        engine = StatevectorEngine(qubo)
        decoder = RouteDecoder(net, qubo)
        bench = ClassicalBenchmarks(qubo)
        x_gr, _ = bench.greedy()
        ws = WarmStartQAOA(engine, p=p, greedy_x=x_gr)
        wp, we, wh = ws.optimize(n_restarts=n_restarts, seed=seed)
        psi = ws.run_circuit(wp)
        best_idx = np.argmax(np.abs(psi) ** 2)
        m = decoder.compute_metrics(decoder.decode(engine.bitstrings[best_idx]))
        results[name] = dict(weights=w, H=float(engine.energies[best_idx]), coverage_complete=m['coverage_complete'], fully_feasible=m['fully_feasible'], travel_cost=m['cost'], risk=m['risk'], energy=m['energy'])
    return results
KNOWN_NO_FULLY_FEASIBLE = {'A': True, 'B': True, 'C': True, 'D': True, 'E': True, 'F': True}
LV_MAP = {'A': 60, 'B': 300, 'C': 450, 'D': 500, 'E': 350, 'F': 550}
TRANSFER_PAIRS = [('A', 'D'), ('B', 'E'), ('C', 'F')]

def run_instance(instance_key, seed=SEED, p=2, n_restarts=3, n_shots=N_SHOTS, run_brute_force=True, run_C=False, verbose=True):
    if verbose:
        print(f"\n{'=' * 70}\nInstance {instance_key}\n{'=' * 70}")
    t_total = time.time()
    env = WindWaveEnvironment()
    net = UAVNetwork(instance_key=instance_key, seed=seed, env=env)
    qubo = GUSRNQuboBuilder(net, lambda_visit=LV_MAP[instance_key])
    engine = StatevectorEngine(qubo)
    decoder = RouteDecoder(net, qubo)
    bench = ClassicalBenchmarks(qubo)
    result = dict(instance=instance_key, n_qubits=qubo.n)
    H_star = None
    if run_brute_force and qubo.n <= 20:
        bi = int(np.argmin(engine.energies))
        H_star = float(engine.energies[bi])
        bf_metrics = decoder.compute_metrics(decoder.decode(engine.bitstrings[bi]))
        result['H_star'] = H_star
        result['bf_metrics'] = bf_metrics
        if verbose:
            print(f"  H* (BF) = {H_star:.4f}  coverage_complete={bf_metrics['coverage_complete']} fully_feasible={bf_metrics['fully_feasible']}")
    R_star_feas_cost = None if KNOWN_NO_FULLY_FEASIBLE.get(instance_key) else best_feasible_travel_cost(engine, decoder)[0]
    result['R_star_feas_cost'] = R_star_feas_cost
    x_gr, gr_cost, gr_time = timed_greedy(bench)
    gr_metrics = decoder.compute_metrics(decoder.decode(x_gr))
    x_sa, sa_cost, sa_hist, sa_time = timed_sa(bench, n_iter=3000, seed=seed)
    sa_metrics = decoder.compute_metrics(decoder.decode(x_sa))
    result['greedy'] = dict(cost=gr_cost, metrics=gr_metrics, time=gr_time)
    result['sa'] = dict(cost=sa_cost, metrics=sa_metrics, time=sa_time)
    if verbose:
        print(f'  Greedy: {gr_cost:.4f} ({gr_time:.2f}s)  SA: {sa_cost:.4f} ({sa_time:.2f}s)')
    variants = {}
    for name, builder in [('Standard', lambda: StandardQAOA(engine, p=p)), ('CVaR', lambda: CVaRQAOA(engine, p=p, alpha=0.25)), ('WarmStart', lambda: WarmStartQAOA(engine, p=p, greedy_x=x_gr))]:
        t0 = time.time()
        model = builder()
        if name == 'WarmStart':
            psi_before = model.initial_psi()
        else:
            psi_before = engine.uniform_init()
        diag_before = distribution_diagnostics(psi_before, engine)
        params, energy, hist = model.optimize(n_restarts=n_restarts, seed=seed)
        psi_after = model.run_circuit(params)
        diag_after = distribution_diagnostics(psi_after, engine)
        shots = sample_shots(psi_after, engine, qubo, H_star if H_star is not None else energy, n_shots=n_shots, seed=seed)
        best_x = shots['best_x']
        best_metrics = decoder.compute_metrics(decoder.decode(best_x))
        rq, gap = rho_qubo(H_star, shots['best_of_shots']) if H_star is not None else (None, None)
        rt, rt_reason = rho_travel(R_star_feas_cost, best_metrics)
        entry = dict(energy=energy, params=params, history=hist, elapsed=time.time() - t0, P_max_before=diag_before['P_max'], expectation_before=diag_before['expectation'], P_max_after=diag_after['P_max'], expectation_after=diag_after['expectation'], entropy_after=diag_after['entropy'], N_eff_after=diag_after['N_eff'], **shots, best_metrics=best_metrics, rho_qubo=rq, rho_qubo_gap=gap, rho_travel=rt, rho_travel_reason=rt_reason)
        if name == 'WarmStart':
            entry['theta_hat'] = theta_hat_summary(model)
            entry['init_mode'] = model.init_mode
        variants[name] = entry
        if verbose:
            print(f"  {name}: H={energy:.4f} best_of_shots={shots['best_of_shots']:.4f} P_max {diag_before['P_max']:.4f}->{diag_after['P_max']:.4f} success_prob={shots['success_probability']:.3f} ({entry['elapsed']:.1f}s)")
    result['variants'] = variants
    d_chain = run_D_chain(engine, qubo, decoder, bench, p=p, n_restarts=n_restarts, seed=seed)
    result['D_chain'] = d_chain
    if verbose:
        print(f"  D-chain: H_greedy={d_chain['H_greedy']:.4f} -> H_WS-init={d_chain['H_WS_init']:.4f} -> H_WS-final={d_chain['H_WS_final']:.4f}")
    result['elapsed_total'] = time.time() - t_total
    return result

def run_depth_sweep(instance_key, depths=(1, 2, 3, 4, 5), seed=SEED, n_restarts=2, n_shots=500, verbose=True):
    env = WindWaveEnvironment()
    net = UAVNetwork(instance_key=instance_key, seed=seed, env=env)
    qubo = GUSRNQuboBuilder(net, lambda_visit=LV_MAP[instance_key])
    engine = StatevectorEngine(qubo)
    decoder = RouteDecoder(net, qubo)
    bench = ClassicalBenchmarks(qubo)
    x_gr, _ = bench.greedy()
    H_star = None
    if qubo.n <= 20:
        H_star = float(engine.energies.min())
    out = {'instance': instance_key, 'H_star': H_star, 'depths': list(depths), 'by_method': {}}
    for name, builder in [('Standard', lambda p: StandardQAOA(engine, p=p)), ('CVaR', lambda p: CVaRQAOA(engine, p=p, alpha=0.25)), ('WarmStart', lambda p: WarmStartQAOA(engine, p=p, greedy_x=x_gr))]:
        rows = []
        for p in depths:
            model = builder(p)
            params, energy, hist = model.optimize(n_restarts=n_restarts, seed=seed)
            psi = model.run_circuit(params)
            shots = sample_shots(psi, engine, qubo, H_star if H_star is not None else energy, n_shots=n_shots, seed=seed)
            rq, _ = rho_qubo(H_star, shots['best_of_shots']) if H_star is not None else (None, None)
            rows.append(dict(p=p, energy=energy, best_of_shots=shots['best_of_shots'], rho_qubo=rq))
            if verbose:
                print(f'  {instance_key} {name} p={p}: H={energy:.4f} rho_qubo={rq}')
        out['by_method'][name] = rows
    return out

def run_transfer_test(calibrate_on, test_on, seed=SEED, p=2, n_restarts=3, n_shots=N_SHOTS, verbose=True):
    frozen_lambda = LV_MAP[calibrate_on]
    env = WindWaveEnvironment()
    net = UAVNetwork(instance_key=test_on, seed=seed, env=env)
    qubo = GUSRNQuboBuilder(net, lambda_visit=frozen_lambda)
    engine = StatevectorEngine(qubo)
    decoder = RouteDecoder(net, qubo)
    bench = ClassicalBenchmarks(qubo)
    x_gr, _ = bench.greedy()
    H_star = float(engine.energies.min()) if qubo.n <= 20 else None
    ws = WarmStartQAOA(engine, p=p, greedy_x=x_gr)
    wp, we, wh = ws.optimize(n_restarts=n_restarts, seed=seed)
    psi = ws.run_circuit(wp)
    shots = sample_shots(psi, engine, qubo, H_star if H_star is not None else we, n_shots=n_shots, seed=seed)
    best_metrics = decoder.compute_metrics(decoder.decode(shots['best_x']))
    rq, gap = rho_qubo(H_star, shots['best_of_shots']) if H_star is not None else (None, None)
    cov_count = 0
    full_count = 0
    for idx in np.unique(shots['all_idxs']):
        x = engine.bitstrings[idx]
        m = decoder.compute_metrics(decoder.decode(x))
        cnt = int((shots['all_idxs'] == idx).sum())
        if m['coverage_complete']:
            cov_count += cnt
        if m['fully_feasible']:
            full_count += cnt
    result = dict(calibrate_on=calibrate_on, test_on=test_on, frozen_lambda=frozen_lambda, H_star=H_star, best_of_shots=shots['best_of_shots'], rho_qubo=rq, rho_qubo_gap=gap, coverage_complete_rate=cov_count / len(shots['all_idxs']), fully_feasible_rate=full_count / len(shots['all_idxs']), best_metrics=best_metrics)
    if verbose:
        print(f"  Transfer {calibrate_on}->{test_on} (lambda={frozen_lambda}): rho_qubo={rq} coverage_rate={result['coverage_complete_rate']:.3f} fully_feasible_rate={result['fully_feasible_rate']:.3f}")
    return result

def run_all_transfer_tests(seed=SEED, **kwargs):
    return {pair: run_transfer_test(pair[0], pair[1], seed=seed, **kwargs) for pair in TRANSFER_PAIRS}
print('Patch module loaded: all §1-§11.7 components defined (incl. C/D experiments, timed benchmarks, theta_hat diagnostics, per-instance runner, depth sweep, transfer matrix).')
