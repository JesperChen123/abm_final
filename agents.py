import mesa
import numpy as np

class DecisionAgent(mesa.Agent):
    """
    An agent making decisions under varying information loads.
    Parameters
    info_load - how many signals the agent receives each step
    cognitive_capacity - how many signals the agent can process effectively
    signal_noise - std-dev of noise added to each signal (0 = perfect signals)
    task_complexity - how cognitively demanding the decision is (0-0.9)
                    0 = simple daily choice (what to eat)
                    0.9 = high-stakes complex decision (buying a house)

    Decision quality degrades when:
      info_load > cognitive_capacity (overload)
      signal_noise is high (corrupted information)
      task_complexity is high (more evidence needed before quality judgment)
    """

    SMOOTHING = 0.8

    def __init__(
        self,
        model,
        info_load: int,
        cognitive_capacity: int,
        signal_noise: float = 0.0,
        task_complexity: float = 0.0,
    ):
        super().__init__(model)
        self.info_load          = info_load
        self.cognitive_capacity = cognitive_capacity
        self.signal_noise       = signal_noise
        self.task_complexity    = task_complexity
        self.decision_quality   = 0.5   # starts neutral [0, 1]

    def _process_signals(self) -> float:
        # cognitive capacity caps useful signals
        useful   = min(self.info_load, self.cognitive_capacity)
        overload = max(0, self.info_load - self.cognitive_capacity)

        # task complexity raises the evidence threshold
        # Complex task (complexity=0.9): agent needs around 2x more signals to reach
        required_signals = self.cognitive_capacity * (1 + self.task_complexity)
        signal_benefit   = 1 - np.exp(-useful / required_signals)

        # noise grows with overload (dual-process theory)
        noise_scale  = useful / self.cognitive_capacity
        overload_amp = 1 + overload / self.cognitive_capacity
        noise_penalty = np.random.normal(
            0, self.signal_noise * noise_scale * overload_amp
        )

        # Structural overload degradation
        overload_noise = np.random.normal(0, overload * 0.08)

        return float(np.clip(signal_benefit + noise_penalty + overload_noise, 0, 1))

    def step(self):
        new_quality = self._process_signals()
        self.decision_quality = (
            self.SMOOTHING * self.decision_quality
            + (1 - self.SMOOTHING) * new_quality
        )