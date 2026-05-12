# Does Information Abundance Help or Harm Decision-Making?

An agent-based model exploring the relationship between information load and decision quality
---

## Overview

It is tempting to assume that more information always leads to better decisions. This project challenges that assumption by simulating individual decision-makers with bounded cognitive capacity operating in environments of varying information abundance.

The central finding is that decision quality follows an **inverted-U** over information load: performance improves up to a point, then declines as cognitive overload sets in. Crucially, this peak is not fixed — it tracks each agent's cognitive capacity. The model also identifies two additional failure modes: **signal noise** (unreliable information) and **task complexity** (irrelevant distractors).

---

## Repository Structure

```
abm_final/
├── agents.py               # DecisionAgent class and signal-processing logic
├── model.py                # InformationModel class, scheduler, and data collection
├── app.py                  # Solara GUI with interactive sliders and visualizations
├── batch_analysis.ipynb    # Batch run experiments and figures
├── batch_capacity.csv      # Results: varying cognitive capacity
├── batch_noise.csv         # Results: varying signal noise
├── batch_complexity.csv    # Results: varying task complexity
└── README.md
```

---

## Model Description

### Agents

Each agent (`DecisionAgent`) represents an individual decision-maker initialized with four attributes:

| Attribute | Description |
|---|---|
| `info_load` | Number of information signals received per time step |
| `cognitive_capacity` | Maximum number of signals the agent can process effectively |
| `signal_noise` | How unreliable or corrupted incoming signals are (0.0–0.5) |
| `task_complexity` | Proportion of incoming signals that are irrelevant distractors (0.0–0.9) |

Agents are divided into six information-load groups (1, 3, 5, 8, 12, or 18 signals/step), with 10 agents per group — 60 agents total per run.