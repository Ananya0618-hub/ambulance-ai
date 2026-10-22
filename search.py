"""
Heuristic search: A* for ambulance routing, with Dijkstra (uniform-cost search)
as the uninformed baseline.

Heuristic h(n) = straight-line distance / max road speed (in minutes).
Admissible: no road is faster than max_speed and traffic multiplier >= 1,
so h never overestimates the true travel time -> A* returns optimal routes.
Consistent: h(u) <= cost(u,v) + h(v) by the triangle inequality.
"""
import heapq
import math
import numpy as np


def heuristic(city, a, b):
    dx, dy = city.coords[a] - city.coords[b]
    return math.hypot(dx, dy) / city.max_speed * 60.0


def astar(city, start, goal, use_heuristic=True):
    """Returns (path, cost_minutes, nodes_expanded).
    use_heuristic=False turns A* into Dijkstra / uniform-cost search."""
    h = (lambda n: heuristic(city, n, goal)) if use_heuristic else (lambda n: 0.0)
    open_heap = [(h(start), 0.0, start)]
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while open_heap:
        _, gc, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == goal:
            break
        for v, w in city.adj[u]:
            ng = gc + w
            if ng < g.get(v, math.inf):
                g[v] = ng
                parent[v] = u
                heapq.heappush(open_heap, (ng + h(v), ng, v))
    if goal not in closed:
        return None, math.inf, expanded
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1], g[goal], expanded


def dijkstra_all(city, source):
    """Single-source shortest travel times to every node."""
    dist = np.full(city.n, np.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in city.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def travel_time_matrix(city):
    """T[i, j] = fastest travel time (min) from node i to node j.
    Used by the GA fitness function so each evaluation is a fast numpy lookup."""
    return np.vstack([dijkstra_all(city, s) for s in range(city.n)])


def compare_astar_dijkstra(city, n_pairs=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_pairs):
        s, t = rng.choice(city.n, 2, replace=False)
        _, c1, e1 = astar(city, s, t, True)
        _, c2, e2 = astar(city, s, t, False)
        rows.append((c1, c2, e1, e2))
    a = np.array(rows)
    return {
        "pairs": n_pairs,
        "same_optimal_cost": bool(np.allclose(a[:, 0], a[:, 1])),
        "avg_expanded_astar": float(a[:, 2].mean()),
        "avg_expanded_dijkstra": float(a[:, 3].mean()),
        "reduction_pct": float(100 * (1 - a[:, 2].mean() / a[:, 3].mean())),
    }
