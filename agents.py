import mesa
import numpy as np

class DecisionAgent(mesa.Agent):
    """
    An agent making decisions under varying information loads.

    Parameters
    info_load-how many signals the agent receives each step
    cognitive_capacity-how many signals the agent can process effectively
    signal_noise-std-dev of noise added to each signal (0 = perfect signals)
    task_complexity-raction of signals that are irrelevant distractors (0-0.9)
    Decision quality degrades when:
      info_load > cognitive_capacity (overload)
      signal_noise is high (corrupted information)
      task_complexity is high (most signals are distractors)
    """

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
        self.decision_quality   = 0.5  # starts neutral [0,1]

    def _process_signals(self) -> float:
        # Only a fraction of signals are relevant
        relevant_count = max(1, int(round(self.info_load * (1 - self.task_complexity))))

        # Agent can only process up to cognitive_capacity of those relevant signals
        useful   = min(relevant_count, self.cognitive_capacity)
        overload = max(0, relevant_count - self.cognitive_capacity)

        signal_benefit = 1 - np.exp(-useful / self.cognitive_capacity)

        # Signal noise corrupts the quality of each useful signal
        noise_penalty = np.random.normal(0, self.signal_noise * useful / self.cognitive_capacity)

        # Overload adds extra degradation on top
        overload_noise = np.random.normal(0, overload * 0.08)

        return float(np.clip(signal_benefit + noise_penalty + overload_noise, 0, 1))

    def step(self):
        self.decision_quality = self._process_signals()