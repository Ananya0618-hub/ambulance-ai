"""
Streamlit demo UI.   Run:  streamlit run app.py
Tabs: City & Placement (GA) | Route (A*) | Triage (Fuzzy) | Dispatch Simulation
"""
import numpy as np
import pandas as pd
import streamlit as st

from city import City
from search import travel_time_matrix, compare_astar_dijkstra, astar
from ga_placement import GAPlacement, random_search
from fuzzy_triage import FuzzyTriage
from dispatch_sim import generate_calls, simulate, summarize
import viz

st.set_page_config(page_title="AI Ambulance Placement & Dispatch", layout="wide")
st.title("🚑 AI Ambulance Placement & Dispatch")
st.caption("Genetic Algorithm (placement) · A* Search (routing) · Fuzzy Logic (triage)")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("City")
    size = st.slider("Grid size (km)", 8, 20, 15)
    city_seed = st.number_input("City seed", 0, 999, 42)
    st.header("GA parameters")
    n_amb = st.slider("Ambulances", 2, 15, 8)
    pop = st.slider("Population size", 10, 150, 60, step=10)
    gens = st.slider("Generations", 10, 300, 150, step=10)
    mut = st.slider("Mutation rate", 0.0, 0.5, 0.15, step=0.05)
    target = st.slider("Target response time (min)", 5, 20, 10)


@st.cache_resource
def build_city(size, seed):
    c = City(size=size, seed=seed)
    return c, travel_time_matrix(c)


city, T = build_city(size, int(city_seed))
ft = FuzzyTriage()

tab1, tab2, tab3, tab4 = st.tabs(["1 · Placement (GA)", "2 · Routing (A*)",
                                  "3 · Triage (Fuzzy)", "4 · Dispatch simulation"])

# ---------------- tab 1: GA ----------------
with tab1:
    if st.button("Run Genetic Algorithm", type="primary"):
        bar = st.progress(0.0)
        ga = GAPlacement(T, city.demand, n_amb, neighbors=city.neighbors,
                         pop_size=pop, generations=gens, mutation_rate=mut,
                         target_minutes=target, seed=0)
        res = ga.run(callback=lambda g, _: bar.progress((g + 1) / gens))
        rnd = random_search(T, city.demand, n_amb, res["evaluations"], target_minutes=target)
        st.session_state["ga"] = res
        st.session_state["rnd"] = rnd
        st.session_state["ga_key"] = (size, city_seed, n_amb, target)

    if "ga" in st.session_state and st.session_state.get("ga_key") == (size, city_seed, n_amb, target):
        res, rnd = st.session_state["ga"], st.session_state["rnd"]
        c1, c2, c3 = st.columns(3)
        c1.metric("GA avg response", f"{res['avg_response']:.2f} min",
                  f"{res['avg_response'] - rnd['avg_response']:.2f} vs random", delta_color="inverse")
        c2.metric(f"GA coverage ≤ {target} min", f"{100 * res['coverage']:.1f}%",
                  f"{100 * (res['coverage'] - rnd['coverage']):.1f} pts vs random")
        c3.metric("Fitness evaluations", f"{res['evaluations']:,}")
        a, b = st.columns(2)
        a.pyplot(viz.plot_city(city, res["stations"], compare=rnd["stations"],
                               title="GA stations (▲) vs random search (×)"))
        b.pyplot(viz.plot_convergence(res["history"], rnd["cost"]))
        st.write("**GA stations (node ids):**", res["stations"])
    else:
        st.info("Set parameters in the sidebar and click **Run Genetic Algorithm**.")
        st.pyplot(viz.plot_city(city, title="City: demand heatmap, roads, hospitals"))

# ---------------- tab 2: A* ----------------
with tab2:
    st.subheader("Route an ambulance with A*")
    stations = st.session_state.get("ga", {}).get("stations")
    c1, c2, c3, c4 = st.columns(4)
    er = c1.number_input("Emergency row", 0, size - 1, size - 2)
    ec = c2.number_input("Emergency col", 0, size - 1, size - 2)
    goal = city.node_id(int(er), int(ec))
    if stations:
        start = min(stations, key=lambda s: T[s, goal])
        c3.write(f"Nearest GA station: node **{start}** {city.rc(start)}")
    else:
        sr = c3.number_input("Start row", 0, size - 1, 0)
        sc = c4.number_input("Start col", 0, size - 1, 0)
        start = city.node_id(int(sr), int(sc))
    path, cost, exp_a = astar(city, start, goal, True)
    _, cost_d, exp_d = astar(city, start, goal, False)
    m1, m2, m3 = st.columns(3)
    m1.metric("Travel time", f"{cost:.1f} min")
    m2.metric("Nodes expanded (A*)", exp_a)
    m3.metric("Nodes expanded (Dijkstra)", exp_d)
    a, b = st.columns(2)
    a.pyplot(viz.plot_city(city, stations, path=path, title="A* route"))
    b.pyplot(viz.plot_search_comparison(compare_astar_dijkstra(city)))
    st.caption("Heuristic h(n) = straight-line distance ÷ max road speed. "
               "Admissible because no road is faster than the max speed.")

# ---------------- tab 3: Fuzzy ----------------
with tab3:
    st.subheader("Fuzzy triage priority calculator")
    c1, c2, c3 = st.columns(3)
    sev = c1.slider("Symptom severity (0–10)", 0.0, 10.0, 6.5, 0.5)
    age = c2.slider("Age (years)", 0, 100, 72)
    wait = c3.slider("Minutes already waiting", 0, 30, 4)
    score, info = ft.infer(sev, age, wait, explain=True)
    st.metric("Priority (0–100)", f"{score:.1f}")
    l, r = st.columns([1, 2])
    with l:
        st.write("**Fuzzified inputs**")
        for var, terms in info["memberships"].items():
            st.write(f"*{var}*: " + ", ".join(f"{t}={v:.2f}" for t, v in terms.items() if v > 0))
        st.write("**Rules fired (strength)**")
        st.dataframe(pd.DataFrame(info["fired_rules"], columns=["Rule", "Strength"]),
                     hide_index=True)
    with r:
        st.pyplot(viz.plot_membership_functions(
            ft, {"severity": sev, "age": age, "wait": wait, "aggregated": info["aggregated"]}))
    with st.expander("Full rule base"):
        for d, _, _ in ft.rules:
            st.write("• " + d)

# ---------------- tab 4: Dispatch ----------------
with tab4:
    st.subheader("Shift simulation: FIFO vs fuzzy-priority dispatch")
    stations = st.session_state.get("ga", {}).get("stations")
    if not stations:
        st.warning("Run the GA in tab 1 first (ambulances are based at GA stations).")
    else:
        c1, c2 = st.columns(2)
        rate = c1.slider("Emergency calls per hour", 1.0, 12.0, 8.0, 0.5)
        hours = c2.slider("Shift length (hours)", 4, 24, 12)
        if st.button("Run simulation", type="primary"):
            calls = generate_calls(city, duration=hours * 60, rate_per_hour=rate, seed=0)
            rows = {}
            for pol in ("fifo", "fuzzy"):
                recs = simulate(city, T, stations, calls, pol, ft, target_minutes=target)
                rows["FIFO" if pol == "fifo" else "Fuzzy priority"] = summarize(recs, target)
                if pol == "fuzzy":
                    st.session_state["fuzzy_records"] = recs
            df = pd.DataFrame(rows).T.round(1)
            df.columns = ["Calls", "Avg response (all)", "Avg response (critical)",
                          f"Critical ≤ {target} min (%)", "Avg response (minor)", "Max response"]
            st.dataframe(df)
            st.caption("Fuzzy dispatch serves critical patients faster under load; "
                       "minor patients wait longer, which is the intended trade-off.")
        if "fuzzy_records" in st.session_state:
            recs = st.session_state["fuzzy_records"]
            log = pd.DataFrame([{k: r[k] for k in ("id", "time", "severity", "age",
                                                    "queue_wait", "travel", "response",
                                                    "ambulance")} for r in recs]).round(1)
            st.write("**Fuzzy dispatch log**")
            st.dataframe(log, hide_index=True, height=300)
