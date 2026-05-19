import mesa
import numpy as np


class DecisionAgent(mesa.Agent):
    """Agent representing a single decision-maker processing information signals."""

    SMOOTHING    = 0.8  # memory weight: 80% prior, 20% new
    NEIGHBOURHOOD = 5   # number of agents this agent can share load with

    def __init__(self, model, info_load: int, cognitive_capacity: int):
        super().__init__(model)
        self.info_load          = info_load           # signals received per step (environmental)
        self.cognitive_capacity = cognitive_capacity   # max signals agent can process (individual)
        self.effective_load     = info_load           # adjusted after network sharing
        self.decision_quality   = 0.5                 # neutral starting point

    def _share_load(self):
        """Overloaded agents offload excess signals to under-loaded neighbours.
        Sharing is limited to a small random neighbourhood, not the whole population."""
        my_overload = max(0, self.effective_load - self.cognitive_capacity)
        if my_overload == 0:
            return

        # Sample a small local neighbourhood rather than searching all agents
        all_others = [a for a in self.model.agents if a is not self]
        neighbours = self.random.sample(all_others, min(self.NEIGHBOURHOOD, len(all_others)))
        helpers    = [a for a in neighbours if a.effective_load < a.cognitive_capacity]

        if not helpers:
            return

        helper   = self.random.choice(helpers)
        spare    = helper.cognitive_capacity - helper.effective_load
        transfer = min(my_overload, spare)

        self.effective_load   -= transfer
        helper.effective_load += transfer

    def _process_signals(self) -> float:
        # Environmental conditions read from model, not stored on agent
        signal_noise    = self.model.signal_noise
        task_complexity = self.model.task_complexity

        useful   = min(self.effective_load, self.cognitive_capacity)
        overload = max(0, self.effective_load - self.cognitive_capacity)

        # Diminishing returns: complex tasks need more signals before quality improves
        required_signals = self.cognitive_capacity * (1 + task_complexity)
        signal_benefit   = 1 - np.exp(-useful / required_signals)

        # Noise penalty scales with both signal_noise and overload (dual-process theory)
        noise_scale   = useful / self.cognitive_capacity
        overload_amp  = 1 + overload / self.cognitive_capacity
        noise_penalty = np.random.normal(0, signal_noise * noise_scale * overload_amp)

        # Structural degradation from exceeding capacity
        overload_noise = np.random.normal(0, overload * 0.08)

        return float(np.clip(signal_benefit + noise_penalty + overload_noise, 0, 1))

    def step(self):
        self._share_load()        # network interaction before processing
        new_quality = self._process_signals()
        self.decision_quality = (
            self.SMOOTHING * self.decision_quality
            + (1 - self.SMOOTHING) * new_quality
        )