import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from model import InformationModel, INFO_LEVELS, COGNITIVE_CAPACITY, SIGNAL_NOISE, TASK_COMPLEXITY
from mesa.visualization import Slider, SolaraViz
from mesa.visualization.components.matplotlib_components import make_mpl_space_component

def draw_line_chart(model, ax):
    """Line chart: avg decision quality per info level over time."""
    ax.cla()
    data = model.datacollector.get_model_vars_dataframe()
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
    """Bar chart: mean quality per info level across all steps so far."""
    ax.cla()
    data = model.datacollector.get_model_vars_dataframe()
    if data.empty:
        return
    means  = [data[str(lvl)].mean() for lvl in INFO_LEVELS]
    colors = ["#4caf50" if lvl <= model.cognitive_capacity else "#f44336" for lvl in INFO_LEVELS]
    bars = ax.bar([str(l) for l in INFO_LEVELS], means, color=colors, edgecolor="white")
    cap_idx = min(range(len(INFO_LEVELS)), key=lambda i: abs(INFO_LEVELS[i] - model.cognitive_capacity))
    ax.axvline(cap_idx, color="black", linestyle="--", linewidth=1.2,
               label=f"Capacity ({model.cognitive_capacity})")
    ax.set_title("Mean Quality by Info Load\n🟢 within capacity  🔴 overload")
    ax.set_xlabel("Signals per step")
    ax.set_ylabel("Mean Decision Quality")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=8)


def make_combined_chart(draw_fn):
    """Wrap a draw function into a Mesa matplotlib space component."""
    def portrayal(model):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        draw_fn(model, ax)
        fig.tight_layout()
        return fig

    import solara

    @solara.component
    def Chart():
        import solara
        m = solara.use_state(None)

        # Re-render on each reactive update by depending on the model
        fig, ax = plt.subplots(figsize=(6, 3.5))
        draw_fn(model_instance, ax)
        fig.tight_layout()
        solara.FigureMatplotlib(fig)
        plt.close(fig)

    return Chart

import solara

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

# Model parameters
model_params = {
    "cognitive_capacity": Slider(
        "Cognitive Capacity", value=COGNITIVE_CAPACITY, min=2, max=15, step=1
    ),
    "signal_noise": Slider(
        "Signal Noise", value=SIGNAL_NOISE, min=0.0, max=0.5, step=0.05
    ),
    "task_complexity": Slider(
        "Task Complexity", value=TASK_COMPLEXITY, min=0.0, max=0.9, step=0.05
    ),
}

# Instantiate model
model = InformationModel()

page = SolaraViz(
    model,
    components=[LineChartComponent, BarChartComponent],
    model_params=model_params,
    name="Information Overload ABM",
    play_interval=150,
)

page