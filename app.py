import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import solara

from model import InformationModel, INFO_LEVELS, COGNITIVE_CAPACITY, SIGNAL_NOISE, TASK_COMPLEXITY
from mesa.visualization import Slider, SolaraViz


def _safe_dataframe(model):
    """Trim all columns to the same length to avoid Mesa off-by-one bug."""
    raw = model.datacollector.model_vars
    if not raw:
        return pd.DataFrame()
    min_len = min(len(v) for v in raw.values())
    return pd.DataFrame({k: v[:min_len] for k, v in raw.items()})


def draw_line_chart(model, ax):
    """Decision quality per info-load group over time."""
    ax.cla()
    data = _safe_dataframe(model)
    if data.empty:
        return
    colors = cm.RdYlGn(np.linspace(0.15, 0.85, len(INFO_LEVELS)))
    for lvl, col in zip(INFO_LEVELS, colors):
        ax.plot(data.index, data[str(lvl)], label=f"{lvl} signals", color=col, linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Baseline")
    ax.set_title("Decision Quality Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg Decision Quality")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)


def draw_bar_chart(model, ax):
    """Mean quality per group; green = within capacity, red = overloaded."""
    ax.cla()
    data = _safe_dataframe(model)
    if data.empty:
        return
    means  = [data[str(lvl)].mean() for lvl in INFO_LEVELS]
    colors = ["#4caf50" if lvl <= model.cognitive_capacity else "#f44336" for lvl in INFO_LEVELS]
    bars   = ax.bar([str(l) for l in INFO_LEVELS], means, color=colors, edgecolor="white")
    cap_idx = min(range(len(INFO_LEVELS)), key=lambda i: abs(INFO_LEVELS[i] - model.cognitive_capacity))
    ax.axvline(cap_idx, color="black", linestyle="--", linewidth=1.2,
               label=f"Mean capacity ({model.cognitive_capacity})")
    ax.set_title("Mean Quality by Info Load\n[green] within capacity  [red] overload")
    ax.set_xlabel("Signals per step")
    ax.set_ylabel("Mean Decision Quality")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=8)


def draw_sharing_chart(model, ax):
    """Assigned vs effective load per group — shows network signal redistribution."""
    ax.cla()
    assigned  = INFO_LEVELS
    effective = []
    for lvl in INFO_LEVELS:
        agents = [a for a in model.agents if a.info_load == lvl]
        effective.append(np.mean([a.effective_load for a in agents]) if agents else lvl)

    x     = np.arange(len(INFO_LEVELS))
    width = 0.35
    ax.bar(x - width / 2, assigned,  width, label="Assigned load",  color="#90CAF9", edgecolor="white")
    ax.bar(x + width / 2, effective, width, label="Effective load", color="#1565C0", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in INFO_LEVELS])
    ax.axhline(model.cognitive_capacity, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean capacity ({model.cognitive_capacity})")
    ax.set_title("Network Sharing: Assigned vs Effective Load")
    ax.set_xlabel("Assigned signals per step")
    ax.set_ylabel("Signals")
    ax.legend(fontsize=7)


def draw_capacity_hist(model, ax):
    """Distribution of individual cognitive capacities across all agents."""
    ax.cla()
    capacities = [a.cognitive_capacity for a in model.agents]
    ax.hist(capacities, bins=range(1, max(capacities) + 2),
            color="#7E57C2", edgecolor="white", align="left")
    ax.axvline(model.cognitive_capacity, color="red", linestyle="--",
               linewidth=1.5, label=f"Model mean ({model.cognitive_capacity})")
    ax.set_title("Agent Capacity Distribution")
    ax.set_xlabel("Cognitive capacity")
    ax.set_ylabel("Number of agents")
    ax.legend(fontsize=7)


def LineChartComponent(model):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    draw_line_chart(model, ax)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


def BarChartComponent(model):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    draw_bar_chart(model, ax)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


def SharingChartComponent(model):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    draw_sharing_chart(model, ax)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


def CapacityHistComponent(model):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    draw_capacity_hist(model, ax)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)
    plt.close(fig)


model_params = {
    "cognitive_capacity": Slider("Cognitive Capacity (mean)", value=COGNITIVE_CAPACITY, min=2, max=15, step=1),
    "signal_noise":       Slider("Signal Noise",              value=SIGNAL_NOISE,       min=0.0, max=0.5, step=0.05),
    "task_complexity":    Slider("Task Complexity",           value=TASK_COMPLEXITY,    min=0.0, max=0.9, step=0.05),
}

model = InformationModel()

page = SolaraViz(
    model,
    components=[LineChartComponent, BarChartComponent, SharingChartComponent, CapacityHistComponent],
    model_params=model_params,
    name="Information Overload ABM",
    play_interval=150,
)

page