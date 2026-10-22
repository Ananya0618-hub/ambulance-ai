"""
Genetic Algorithm for ambulance station placement.

Chromosome : list of N distinct node ids, one per ambulance station.
             e.g. [17, 42, 108, 190, 203, 66]
Fitness    : demand-weighted average response time across the city
             + penalty for demand NOT covered within the target time.
             (lower cost = fitter)
Selection  : tournament (k = 3)
Crossover  : set-based - keep stations common to both parents, fill the rest
             randomly from the parents' remaining stations (always valid, no repair)
Mutation   : per gene - 70% move station to a neighbouring intersection (local tweak),
             30% jump to a random intersection (exploration)
Elitism    : best E chromosomes copied unchanged to next generation
"""
import numpy as np


class GAPlacement:
    def __init__(self, T, demand, n_ambulances, neighbors=None,
                 pop_size=60, generations=150, crossover_rate=0.9,
                 mutation_rate=0.15, tournament_k=3, elite=2,
                 target_minutes=8.0, dispatch_delay=1.0,
                 coverage_penalty=10.0, seed=0):
        self.T = T
        self.demand = demand
        self.n_nodes = len(demand)
        self.k = n_ambulances
        self.neighbors = neighbors
        self.pop_size = pop_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.tk = tournament_k
        self.elite = elite
        self.target = target_minutes
        self.delay = dispatch_delay
        self.penalty = coverage_penalty
        self.rng = np.random.default_rng(seed)
        self.evaluations = 0

    # ---------- fitness ----------
    def evaluate(self, chrom):
        """Returns (cost, avg_response_min, coverage_fraction)."""
        self.evaluations += 1
        rt = self.T[list(chrom)].min(axis=0) + self.delay   # nearest station to each node
        avg = float((rt * self.demand).sum())
        cov = float(self.demand[rt <= self.target].sum())
        return avg + self.penalty * (1 - cov), avg, cov

    # ---------- operators ----------
    def random_chrom(self):
        return list(self.rng.choice(self.n_nodes, self.k, replace=False))

    def tournament(self, pop, costs):
        idx = self.rng.choice(len(pop), self.tk, replace=False)
        return pop[min(idx, key=lambda i: costs[i])]

    def crossover(self, p1, p2):
        if self.rng.random() > self.cx_rate:
            return list(p1)
        common = list(set(p1) & set(p2))
        pool = list((set(p1) | set(p2)) - set(common))
        self.rng.shuffle(pool)
        return common + pool[: self.k - len(common)]

    def mutate(self, chrom):
        chrom = list(chrom)
        for i in range(self.k):
            if self.rng.random() < self.mut_rate:
                if self.neighbors is not None and self.rng.random() < 0.7:
                    options = [v for v in self.neighbors(chrom[i]) if v not in chrom]
                else:
                    options = [v for v in self.rng.choice(self.n_nodes, 5, replace=False)
                               if v not in chrom]
                if options:
                    chrom[i] = int(self.rng.choice(options))
        return chrom

    # ---------- main loop ----------
    def run(self, callback=None):
        pop = [self.random_chrom() for _ in range(self.pop_size)]
        history = {"best": [], "mean": [], "best_avg_rt": [], "best_cov": []}
        best, best_eval = None, (np.inf, None, None)

        for gen in range(self.generations):
            evals = [self.evaluate(c) for c in pop]
            costs = [e[0] for e in evals]
            order = np.argsort(costs)
            if costs[order[0]] < best_eval[0]:
                best, best_eval = list(pop[order[0]]), evals[order[0]]

            history["best"].append(best_eval[0])
            history["mean"].append(float(np.mean(costs)))
            history["best_avg_rt"].append(best_eval[1])
            history["best_cov"].append(best_eval[2])
            if callback:
                callback(gen, best_eval)

            new_pop = [list(pop[i]) for i in order[: self.elite]]
            while len(new_pop) < self.pop_size:
                p1 = self.tournament(pop, costs)
                p2 = self.tournament(pop, costs)
                new_pop.append(self.mutate(self.crossover(p1, p2)))
            pop = new_pop

        return {"stations": [int(s) for s in best], "cost": best_eval[0],
                "avg_response": best_eval[1], "coverage": best_eval[2],
                "history": history, "evaluations": self.evaluations}


def random_search(T, demand, n_ambulances, n_evals, target_minutes=8.0,
                  dispatch_delay=1.0, coverage_penalty=10.0, seed=1):
    """Baseline: try n_evals random placements (same budget as the GA)."""
    ga = GAPlacement(T, demand, n_ambulances, target_minutes=target_minutes,
                     dispatch_delay=dispatch_delay, coverage_penalty=coverage_penalty,
                     seed=seed)
    best, best_eval = None, (np.inf, None, None)
    for _ in range(n_evals):
        c = ga.random_chrom()
        e = ga.evaluate(c)
        if e[0] < best_eval[0]:
            best, best_eval = c, e
    return {"stations": [int(s) for s in best], "cost": best_eval[0],
            "avg_response": best_eval[1], "coverage": best_eval[2]}


def evaluate_placement(T, demand, stations, target_minutes=8.0,
                       dispatch_delay=1.0, coverage_penalty=10.0):
    ga = GAPlacement(T, demand, len(stations), target_minutes=target_minutes,
                     dispatch_delay=dispatch_delay, coverage_penalty=coverage_penalty)
    cost, avg, cov = ga.evaluate(stations)
    return {"stations": list(stations), "cost": cost, "avg_response": avg, "coverage": cov}
