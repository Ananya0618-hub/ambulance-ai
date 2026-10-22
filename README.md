# AI Ambulance Placement & Dispatch

AISC project: Algorithmic Implementation of AI Approaches.

| Rubric approach | Where it is used | File |
|---|---|---|
| **Genetic Algorithm** | Choose where to station N ambulances | `ga_placement.py` |
| **Heuristic Search (A\*)** | Fastest route from station to emergency | `search.py` |
| **Fuzzy Logic** | Triage priority from severity, age, waiting time | `fuzzy_triage.py` |

All three algorithms are implemented from scratch (only numpy), so each step can be explained in the viva.

## Run

```bash
pip install -r requirements.txt
python main.py            # full pipeline -> plots + summary.json in ./results
streamlit run app.py      # interactive demo for the viva
```

Options: `python main.py --ambulances 8 --size 15 --generations 150 --rate 8`

## Files

```
city.py           City road network (grid, arterial/local roads, traffic, blocked roads,
                  demand hotspots, hospitals)
search.py         A*, Dijkstra, all-pairs travel-time matrix, A* vs Dijkstra comparison
ga_placement.py   GA for station placement + random-search baseline
fuzzy_triage.py   Mamdani fuzzy inference system (from scratch)
dispatch_sim.py   Shift simulation: FIFO vs fuzzy-priority dispatch
viz.py            Plots
main.py           Runs everything, saves results
app.py            Streamlit UI (4 tabs)
```

## How the system works

1. **City model.** A 15×15 km grid with 225 intersections. Arterial roads (every 4 km) run at 50 km/h. Local roads run at 25 km/h with a random traffic factor of 1–2×, and about 6% are blocked. Emergency demand is concentrated around random population hotspots.
2. **Placement (GA).** The GA finds the set of N intersections that minimizes demand-weighted average response time and maximizes the share of demand reachable within the target time (10 min).
3. **Triage (Fuzzy).** Each incoming call gets a 0–100 priority from symptom severity, patient age, and minutes already waiting.
4. **Dispatch.** When an ambulance is free, the highest-priority call is served by the nearest free ambulance, which is routed with A*. The ambulance stays busy for scene time, the hospital trip, handover, and the return to station.

## Results (default run)

| Experiment | Result |
|---|---|
| A* vs Dijkstra (200 random routes) | Same optimal routes; A* expands **~57% fewer nodes** |
| GA vs random search (same 9,000 evaluations) | Avg response **5.76 vs 6.17 min**; coverage ≤10 min **96.5% vs 91.6%** |
| Fuzzy vs FIFO dispatch at 8 calls/hr (avg of 5 runs) | Critical patients: **18.6 vs 46.4 min** avg response |

At low load (3–5 calls/hr) both dispatch policies perform the same, because no one has to wait. The fuzzy policy pays off when ambulances are scarce. The cost is that minor patients wait longer; the "wait is long" rules limit this so minor cases don't wait forever.

Plots in `results/`: `city_map.png`, `ga_convergence.png`, `fuzzy_mfs.png`, `dispatch_load_sweep.png`, `astar_vs_dijkstra.png`.

---

## Viva notes

### Genetic Algorithm
- **Chromosome:** list of N distinct node ids, e.g. `[17, 42, 108, ...]`, one gene per ambulance station.
- **Fitness (cost, minimized):** `avg_response + 10 × (1 − coverage)`
  - `avg_response` = Σ demand(j) × (min over stations of travel_time(station, j) + 1 min dispatch delay)
  - `coverage` = fraction of demand reachable within the target time
  - The travel-time matrix is precomputed once with Dijkstra from every node, so each fitness evaluation is a fast numpy lookup.
- **Selection:** tournament with k = 3. It is simple, needs no fitness scaling, and k controls selection pressure.
- **Crossover:** set-based. Keep the stations common to both parents, then fill the remaining slots from the other parent genes. Children are always valid (no duplicate stations), so no repair step is needed.
- **Mutation:** per gene with p = 0.15. 70% of the time the station moves to a neighbouring intersection (local fine-tuning); 30% of the time it jumps to a random node (exploration, avoids local optima).
- **Elitism:** the best 2 chromosomes pass unchanged, so the best solution never gets worse.
- **Why GA and not brute force?** Choosing 8 stations from 225 nodes gives C(225, 8) ≈ 10¹⁴ combinations.
- **This is the p-median / maximal covering location problem**, which is NP-hard.

### A* Search
- `f(n) = g(n) + h(n)`. g = travel time so far; h = straight-line distance ÷ 50 km/h.
- **Admissible:** no road is faster than 50 km/h and traffic is ≥ 1×, so h never overestimates. A* therefore returns optimal routes, verified against Dijkstra.
- **Consistent:** h(u) ≤ cost(u, v) + h(v) by the triangle inequality, so each node is expanded at most once.
- **Dijkstra = A* with h = 0.** It is uninformed and explores in all directions.
- Complexity: O(E log V) with a binary heap. A* has the same worst case, but expands far fewer nodes in practice.

### Fuzzy Logic (Mamdani)
1. **Fuzzification:** a crisp input (severity 6.5) becomes membership degrees (medium 0.25, high 0.25).
2. **Rule evaluation:** AND = min, OR = max. There are 7 IF-THEN rules (see the app's "Full rule base").
3. **Implication:** each output set is clipped at its rule's firing strength (min).
4. **Aggregation:** clipped sets are combined with max.
5. **Defuzzification:** centroid, `Σ y·μ(y) / Σ μ(y)`.
- **Why fuzzy?** Triage is vague: "fairly severe", "elderly", "waited a while". Fixed thresholds such as age > 60 cause unfair jumps, while fuzzy sets give smooth, explainable priorities.
- **Triangular vs trapezoidal MFs:** a trapezoid is used where a whole range is fully true (e.g. every age above 68 is fully "elderly").

### Likely viva questions
- *What if the GA converges prematurely?* Increase the mutation rate or population size, or reduce tournament size.
- *How would you use real data?* Load an OpenStreetMap road graph (`osmnx`), take historical 108-call locations as demand, and use real hospital coordinates.
- *Limitations?* Static traffic, ambulances only redeploy to their home station, and the fuzzy rules are hand-designed (they could be tuned by the GA, i.e. a genetic-fuzzy system).
- *Extensions:* dynamic relocation of idle ambulances, time-of-day traffic, or an NN to predict call demand per area and hour.
