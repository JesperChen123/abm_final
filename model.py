import mesa
import numpy as np
from agents import DecisionAgent

INFO_LEVELS        = [1, 3, 5, 8, 12, 18]  # signals per step per group
AGENTS_PER_LEVEL   = 10
COGNITIVE_CAPACITY = 6    # population mean; individual agents vary around this
SIGNAL_NOISE       = 0.0  # environmental: how corrupted signals are (0.0–0.5)
TASK_COMPLEXITY    = 0.0  # environmental: how demanding the decision is (0.0–0.9)
CAPACITY_SD        = 1.5  # std-dev for individual capacity variation


def _make_level_reporter(lvl):
    """Returns average decision quality for one info-load group."""
    def reporter(model):
        agents = [a for a in model.agents if a.info_load == lvl]
        if not agents:
            return 0.0
        return float(np.mean([a.decision_quality for a in agents]))
    reporter.__name__ = str(lvl)
    return reporter


class InformationModel(mesa.Model):
    """
    60 agents divided into 6 info-load groups.
    signal_noise and task_complexity are environmental conditions stored at
    model level; agents read them each step rather than owning them.
    Each agent gets an individual cognitive_capacity drawn from N(mean, SD).
    Effective loads are reset at the model level each step before agents run,
    ensuring consistent network sharing across all activation orders.
    """

    def __init__(
        self,
        cognitive_capacity: int   = COGNITIVE_CAPACITY,
        signal_noise:       float = SIGNAL_NOISE,
        task_complexity:    float = TASK_COMPLEXITY,
        **kwargs,
    ):
        super().__init__()
        self.cognitive_capacity = cognitive_capacity  # mean across population
        self.signal_noise       = signal_noise        # environmental condition
        self.task_complexity    = task_complexity     # environmental condition
        self.step_count         = 0

        # Heterogeneous agents: each gets a slightly different capacity
        for info_load in INFO_LEVELS:
            for _ in range(AGENTS_PER_LEVEL):
                individual_capacity = max(
                    1,
                    int(np.round(self.random.gauss(cognitive_capacity, CAPACITY_SD)))
                )
                DecisionAgent(self, info_load, individual_capacity)

        self.datacollector = mesa.DataCollector(
            model_reporters={str(lvl): _make_level_reporter(lvl) for lvl in INFO_LEVELS}
        )
        self.datacollector.collect(self)

    def step(self):
        # Reset all effective loads before any agent steps to ensure consistency
        for agent in self.agents:
            agent.effective_load = agent.info_load

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        self.step_count += 1

    def run(self, steps: int = 100):
        for _ in range(steps):
            self.step()