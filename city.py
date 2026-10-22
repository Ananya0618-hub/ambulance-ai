"""
City road network model.

The city is a grid of intersections (nodes) connected by roads (edges).
- Arterial roads (every 4th row/column) are fast (50 km/h) with light traffic.
- Local roads are slower (25 km/h) with heavier, random traffic, and a few
  are blocked (construction / one-off closures).
- Each node has an emergency-call demand weight (population hotspots).
- A few nodes are hospitals.
Edge weight = travel time in minutes.
"""
import numpy as np
from collections import deque


class City:
    def __init__(self, size=15, spacing_km=1.0, n_hospitals=3,
                 block_prob=0.06, n_hotspots=4, seed=42):
        self.size = size
        self.spacing = spacing_km
        self.n = size * size
        self.max_speed = 50.0  # km/h, used by the A* heuristic
        self.seed = seed

        attempt = 0
        while True:
            rng = np.random.default_rng(seed + attempt)
            self._build_roads(rng, block_prob)
            if self._is_connected():
                break
            attempt += 1

        self._build_demand(rng, n_hotspots)
        self._place_hospitals(n_hospitals)

    # ---------- construction ----------
    def node_id(self, r, c):
        return r * self.size + c

    def rc(self, node):
        return divmod(node, self.size)

    def _build_roads(self, rng, block_prob):
        s = self.size
        self.coords = np.array([(c * self.spacing, r * self.spacing)
                                for r in range(s) for c in range(s)], dtype=float)
        self.adj = {i: [] for i in range(self.n)}
        self.edges = []  # (a, b, time, arterial)
        for r in range(s):
            for c in range(s):
                for dr, dc in ((0, 1), (1, 0)):
                    r2, c2 = r + dr, c + dc
                    if r2 >= s or c2 >= s:
                        continue
                    arterial = (dr == 0 and r % 4 == 0) or (dc == 0 and c % 4 == 0)
                    if not arterial and rng.random() < block_prob:
                        continue  # blocked road
                    speed = 50.0 if arterial else 25.0
                    traffic = rng.uniform(1.0, 1.4) if arterial else rng.uniform(1.0, 2.0)
                    t = self.spacing / speed * 60.0 * traffic  # minutes
                    a, b = self.node_id(r, c), self.node_id(r2, c2)
                    self.adj[a].append((b, t))
                    self.adj[b].append((a, t))
                    self.edges.append((a, b, t, arterial))

    def _is_connected(self):
        seen = {0}
        q = deque([0])
        while q:
            u = q.popleft()
            for v, _ in self.adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return len(seen) == self.n

    def _build_demand(self, rng, n_hotspots):
        extent = (self.size - 1) * self.spacing
        self.hotspots = rng.uniform(0.15 * extent, 0.85 * extent, size=(n_hotspots, 2))
        sigma = max(extent / 6.0, 1e-6)
        d = np.full(self.n, 0.15)
        for cx, cy in self.hotspots:
            w = rng.uniform(0.6, 1.5)
            d += w * np.exp(-((self.coords[:, 0] - cx) ** 2 +
                              (self.coords[:, 1] - cy) ** 2) / (2 * sigma ** 2))
        self.demand = d / d.sum()

    def _place_hospitals(self, k):
        """Spread hospitals out: pick the highest-demand node, then repeatedly
        pick the node farthest from existing hospitals (weighted by demand)."""
        hosp = [int(np.argmax(self.demand))]
        while len(hosp) < k:
            dmin = np.min([np.linalg.norm(self.coords - self.coords[h], axis=1)
                           for h in hosp], axis=0)
            score = dmin * (0.5 + self.demand / self.demand.max())
            score[hosp] = -1
            hosp.append(int(np.argmax(score)))
        self.hospitals = hosp

    def neighbors(self, node):
        return [v for v, _ in self.adj[node]]
