"""
Run the full pipeline end to end and save results to ./results

  1. Build city road network
  2. Heuristic search : A* vs Dijkstra
  3. Genetic Algorithm: ambulance station placement (vs random search)
  4. Fuzzy logic      : triage priority examples
  5. Dispatch sim     : FIFO vs fuzzy priority, across increasing call load

Usage:  python main.py  [--ambulances 8] [--size 15] [--generations 150]
"""
import argparse
import json
import os
import numpy as np

from city import City
from search import travel_time_matrix, compare_astar_dijkstra, astar
from ga_placement import GAPlacement, random_search
from fuzzy_triage import FuzzyTriage
from dispatch_sim import generate_calls, simulate, summarize
import viz


def load_sweep(city, T, stations, ft, rates, seeds=5, target=10.0):
    out = {k: [] for k in ["fifo_critical", "fuzzy_critical", "fifo_minor",
                           "fuzzy_minor", "fifo_all", "fuzzy_all"]}
    for rate in rates:
        acc = {k: [] for k in out}
        for sd in range(seeds):
            calls = generate_calls(city, rate_per_hour=rate, seed=sd)
            for pol in ("fifo", "fuzzy"):
                s = summarize(simulate(city, T, stations, calls, pol, ft,
                                       target_minutes=target), target)
                acc[f"{pol}_critical"].append(s["avg_response_critical"])
                acc[f"{pol}_minor"].append(s["avg_response_minor"])
                acc[f"{pol}_all"].append(s["avg_response_all"])
        for k in out:
            out[k].append(float(np.mean(acc[k])))
    out["rates"] = list(rates)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--ambulances", type=int, default=8)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--target", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    print("1) Building city ...")
    city = City(size=args.size, seed=args.seed)
    T = travel_time_matrix(city)
    print(f"   {city.n} intersections, {len(city.edges)} roads, hospitals at {city.hospitals}")

    print("2) A* vs Dijkstra ...")
    sstats = compare_astar_dijkstra(city)
    print(f"   A* avg expanded {sstats['avg_expanded_astar']:.1f} vs Dijkstra "
          f"{sstats['avg_expanded_dijkstra']:.1f}  ({sstats['reduction_pct']:.0f}% fewer), "
          f"same optimal cost: {sstats['same_optimal_cost']}")
    viz.plot_search_comparison(sstats).savefig("results/astar_vs_dijkstra.png", dpi=150)

    print("3) Genetic Algorithm placement ...")
    ga = GAPlacement(T, city.demand, args.ambulances, neighbors=city.neighbors,
                     pop_size=args.pop, generations=args.generations,
                     target_minutes=args.target, seed=0)
    res = ga.run()
    rnd = random_search(T, city.demand, args.ambulances, res["evaluations"],
                        target_minutes=args.target)
    print(f"   GA     : avg response {res['avg_response']:.2f} min, "
          f"coverage<= {args.target:.0f} min {100*res['coverage']:.1f}%")
    print(f"   Random : avg response {rnd['avg_response']:.2f} min, "
          f"coverage<= {args.target:.0f} min {100*rnd['coverage']:.1f}%  "
          f"(same {res['evaluations']} evaluations)")
    viz.plot_convergence(res["history"], rnd["cost"]).savefig("results/ga_convergence.png", dpi=150)

    # example route: farthest high-demand node from its nearest station
    rt = T[res["stations"]].min(axis=0)
    target_node = int(np.argmax(rt * (city.demand > np.median(city.demand))))
    src = min(res["stations"], key=lambda s: T[s, target_node])
    path, cost, _ = astar(city, src, target_node)
    viz.plot_city(city, res["stations"], path=path, compare=rnd["stations"],
                  title=f"GA stations + example A* route ({cost:.1f} min)"
                  ).savefig("results/city_map.png", dpi=150)

    print("4) Fuzzy triage examples ...")
    ft = FuzzyTriage()
    examples = [("Cardiac arrest, 70y", 9.5, 70, 0), ("Fracture, 8y", 5, 8, 2),
                ("Fracture, 30y", 5, 30, 2), ("Minor cut, 30y", 2, 30, 0),
                ("Minor cut, 30y, waited 20m", 2, 30, 20), ("Fever, 80y", 3, 80, 5)]
    fuzzy_rows = []
    for name, s, a, w in examples:
        p = ft.priority(s, a, w)
        fuzzy_rows.append({"case": name, "severity": s, "age": a, "wait": w,
                           "priority": round(p, 1)})
        print(f"   {name:28s} -> {p:5.1f}")
    sc, info = ft.infer(9.5, 70, 0, explain=True)
    viz.plot_membership_functions(ft, {"severity": 9.5, "age": 70, "wait": 0,
                                       "aggregated": info["aggregated"]}
                                  ).savefig("results/fuzzy_mfs.png", dpi=150)

    print("5) Dispatch simulation (FIFO vs fuzzy) ...")
    calls = generate_calls(city, rate_per_hour=args.rate, seed=0)
    s_fifo = summarize(simulate(city, T, res["stations"], calls, "fifo", ft,
                                target_minutes=args.target), args.target)
    s_fuzzy = summarize(simulate(city, T, res["stations"], calls, "fuzzy", ft,
                                 target_minutes=args.target), args.target)
    for name, s in (("FIFO", s_fifo), ("Fuzzy", s_fuzzy)):
        print(f"   {name:5s}: critical avg {s['avg_response_critical']:.1f} min | "
              f"minor avg {s['avg_response_minor']:.1f} min | all {s['avg_response_all']:.1f} min")
    sweep = load_sweep(city, T, res["stations"], ft, rates=[3, 4, 5, 6, 7, 8],
                       target=args.target)
    viz.plot_load_sweep(sweep).savefig("results/dispatch_load_sweep.png", dpi=150)

    summary = {"astar_vs_dijkstra": sstats,
               "ga": {k: res[k] for k in ("stations", "avg_response", "coverage", "evaluations")},
               "random_search": rnd, "fuzzy_examples": fuzzy_rows,
               "dispatch_single_run": {"rate_per_hour": args.rate, "fifo": s_fifo, "fuzzy": s_fuzzy},
               "dispatch_load_sweep": sweep}
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone. Plots and summary.json saved in ./results")


if __name__ == "__main__":
    main()
