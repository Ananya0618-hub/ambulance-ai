"""Plotting helpers (matplotlib) used by main.py and the Streamlit app."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_city(city, stations=None, path=None, title="City map", compare=None, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 7))
    s = city.size
    heat = city.demand.reshape(s, s)
    ext = (s - 1) * city.spacing
    ax.imshow(heat, origin="lower", cmap="YlOrRd", alpha=0.55,
              extent=(-0.5, ext + 0.5, -0.5, ext + 0.5))
    for a, b, _, arterial in city.edges:
        (x1, y1), (x2, y2) = city.coords[a], city.coords[b]
        ax.plot([x1, x2], [y1, y2], color="#333" if arterial else "#999",
                lw=2.2 if arterial else 0.7, zorder=1)
    hx, hy = city.coords[city.hospitals].T
    ax.scatter(hx, hy, marker="P", s=260, c="#d62728", edgecolor="white",
               zorder=4, label="Hospital")
    if compare is not None:
        cx, cy = city.coords[compare].T
        ax.scatter(cx, cy, marker="x", s=120, c="#555", zorder=4, label="Random placement")
    if stations is not None:
        sx, sy = city.coords[stations].T
        ax.scatter(sx, sy, marker="^", s=220, c="#1f77b4", edgecolor="white",
                   zorder=5, label="GA ambulance station")
    if path:
        px, py = city.coords[path].T
        ax.plot(px, py, color="#2ca02c", lw=4, zorder=3, label="A* route")
        ax.scatter([px[0]], [py[0]], s=90, c="#2ca02c", zorder=6)
        ax.scatter([px[-1]], [py[-1]], marker="*", s=300, c="#ff7f0e",
                   edgecolor="black", zorder=6, label="Emergency")
    ax.set_title(title)
    ax.set_xlabel("km"); ax.set_ylabel("km")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    if own:
        fig.tight_layout()
        return fig


def plot_convergence(history, random_cost=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["best"], label="GA best cost", lw=2)
    ax.plot(history["mean"], label="GA population mean", lw=1, alpha=0.7)
    if random_cost is not None:
        ax.axhline(random_cost, color="gray", ls="--",
                   label="Random search best (same #evaluations)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Cost (lower is better)")
    ax.set_title("GA convergence: ambulance placement")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_membership_functions(ft, highlight=None):
    vars_ = ["severity", "age", "wait", "priority"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, var in zip(axes.flat, vars_):
        x = ft.universe[var]
        for term in ft.mfs[var]:
            ax.plot(x, ft.mf(var, term, x), label=term, lw=2)
        if highlight and var in highlight:
            ax.axvline(highlight[var], color="black", ls="--", lw=1.5)
        if var == "priority" and highlight and "aggregated" in highlight:
            ax.fill_between(x, highlight["aggregated"], color="gray", alpha=0.4,
                            label="aggregated output")
        ax.set_title(var.capitalize()); ax.set_ylim(-0.05, 1.08)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Fuzzy triage membership functions")
    fig.tight_layout()
    return fig


def plot_load_sweep(sweep):
    rates = sweep["rates"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(rates, sweep["fifo_critical"], "o-", label="FIFO")
    axes[0].plot(rates, sweep["fuzzy_critical"], "s-", label="Fuzzy priority")
    axes[0].set_title("Critical patients: avg response time")
    axes[0].set_xlabel("Calls per hour"); axes[0].set_ylabel("Minutes")
    axes[1].plot(rates, sweep["fifo_minor"], "o-", label="FIFO")
    axes[1].plot(rates, sweep["fuzzy_minor"], "s-", label="Fuzzy priority")
    axes[1].set_title("Minor patients: avg response time (the trade-off)")
    axes[1].set_xlabel("Calls per hour"); axes[1].set_ylabel("Minutes")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_search_comparison(stats):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Dijkstra (UCS)", "A*"],
           [stats["avg_expanded_dijkstra"], stats["avg_expanded_astar"]],
           color=["#999", "#2ca02c"])
    ax.set_ylabel("Avg nodes expanded per route")
    ax.set_title(f"A* expands {stats['reduction_pct']:.0f}% fewer nodes\n(same optimal routes)")
    fig.tight_layout()
    return fig
