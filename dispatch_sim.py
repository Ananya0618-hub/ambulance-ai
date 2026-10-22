"""
Dispatch simulation: emergency calls arrive over a shift; ambulances based at
the GA-chosen stations respond.

Each minute:
  1. New calls join the waiting queue.
  2. While a free ambulance and a waiting call exist:
       - choose which call to serve   (policy: FIFO or FUZZY priority)
       - send the nearest free ambulance
       - route it with A*
  3. Ambulance is busy for: travel to patient + on-scene time
     + travel to nearest hospital + handover + return to station.

Compares FIFO (first-come-first-served) vs fuzzy-priority dispatch.
"""
import numpy as np
from search import astar


def generate_calls(city, duration=720, rate_per_hour=9.0, seed=7):
    rng = np.random.default_rng(seed)
    calls, t, cid = [], 0.0, 0
    while True:
        t += rng.exponential(60.0 / rate_per_hour)   # Poisson arrivals
        if t > duration:
            break
        u = rng.random()
        if u < 0.2:
            sev = rng.uniform(7, 10)     # critical: cardiac, stroke, major trauma
        elif u < 0.6:
            sev = rng.uniform(4, 7)      # serious: fractures, breathing issues
        else:
            sev = rng.uniform(0, 4)      # minor
        calls.append({"id": cid, "time": t,
                      "node": int(rng.choice(city.n, p=city.demand)),
                      "severity": float(sev),
                      "age": float(rng.integers(1, 91))})
        cid += 1
    return calls


def simulate(city, T, stations, calls, policy="fuzzy", triage=None,
             dispatch_delay=1.0, scene_time=15.0, handover_time=10.0,
             target_minutes=8.0):
    ambulances = [{"id": i, "station": s, "free_at": 0.0} for i, s in enumerate(stations)]
    pending, records = [], []
    i, t = 0, 0.0
    calls = sorted(calls, key=lambda c: c["time"])

    while i < len(calls) or pending:
        while i < len(calls) and calls[i]["time"] <= t:
            pending.append(calls[i])
            i += 1
        free = [a for a in ambulances if a["free_at"] <= t]
        while free and pending:
            if policy == "fuzzy":
                call = max(pending, key=lambda c: triage.priority(
                    c["severity"], c["age"], t - c["time"]))
            else:
                call = min(pending, key=lambda c: c["time"])
            pending.remove(call)

            amb = min(free, key=lambda a: T[a["station"], call["node"]])
            free.remove(amb)
            path, travel, expanded = astar(city, amb["station"], call["node"])
            hosp = min(city.hospitals, key=lambda h: T[call["node"], h])

            wait = t - call["time"]
            response = wait + dispatch_delay + travel
            busy = (dispatch_delay + travel + scene_time + T[call["node"], hosp]
                    + handover_time + T[hosp, amb["station"]])
            amb["free_at"] = t + busy
            records.append({**call, "ambulance": amb["id"], "queue_wait": wait,
                            "travel": travel, "response": response,
                            "astar_expanded": expanded, "path": path,
                            "critical": call["severity"] >= 7})
        t += 1.0
    return records


def summarize(records, target_minutes=8.0):
    r = np.array([x["response"] for x in records])
    crit = np.array([x["response"] for x in records if x["critical"]])
    minor = np.array([x["response"] for x in records if x["severity"] < 4])
    return {
        "calls": len(records),
        "avg_response_all": float(r.mean()),
        "avg_response_critical": float(crit.mean()) if len(crit) else float("nan"),
        "critical_within_target_pct": float(100 * (crit <= target_minutes).mean()) if len(crit) else float("nan"),
        "avg_response_minor": float(minor.mean()) if len(minor) else float("nan"),
        "max_response": float(r.max()),
    }
