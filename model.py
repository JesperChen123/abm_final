import mesa
import numpy as np
from agents import DecisionAgent

INFO_LEVELS        = [1, 3, 5, 8, 12, 18]   # signals per step (fixed groups)
AGENTS_PER_LEVEL   = 10
COGNITIVE_CAPACITY = 6
SIGNAL_NOISE       = 0.0
TASK_COMPLEXITY    = 0.0


def _make_level_reporter(lvl):
    """Return a reporter function that computes avg decision quality for one info level."""
    def reporter(model):
        agents = [a for a in model.agents if a.info_load == lvl]
        if not agents:
            return 0.0
        return float(np.mean([a.decision_quality for a in agents]))
    reporter.__name__ = str(lvl)
    return reporter


class InformationModel(mesa.Model):
    """
    Simulates agents grouped by information load.
    Tracks how decision quality varies across info levels over time.

    Parameters
    cognitive_capacity-signals an agent handles without penalty
    signal_noise-how corrupted incoming signals are (0.0-0.5)
    task_complexity-fraction of signals that are irrelevant (0.0-0.9)
    """

    def __init__(
        self,
        cognitive_capacity: int   = COGNITIVE_CAPACITY,
        signal_noise:       float = SIGNAL_NOISE,
        task_complexity:    float = TASK_COMPLEXITY,
        **kwargs,
    ):
        super().__init__()
        self.cognitive_capacity = cognitive_capacity
        self.signal_noise       = signal_noise
        self.task_complexity    = task_complexity
        self.step_count         = 0
        self.history            = []

        for info_load in INFO_LEVELS:
            for _ in range(AGENTS_PER_LEVEL):
                DecisionAgent(
                    self,
                    info_load,
                    cognitive_capacity,
                    signal_noise,
                    task_complexity,
                )

        # This is what make_plot_component reads from
        self.datacollector = mesa.DataCollector(
            model_reporters={str(lvl): _make_level_reporter(lvl) for lvl in INFO_LEVELS}
        )
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        self.step_count += 1

    def run(self, steps: int = 30):
        for _ in range(steps):
            self.step()