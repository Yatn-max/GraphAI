"""
Disruption Generator
====================
Job: ONLY create disruptions. It decides WHICH edge is blocked, FROM which time step and
for HOW LONG. It does not move any traffic (that is the Traffic Redistributor's job).

Right now one scenario = one blocked edge.

Reproducibility: the same `seed` always produces the same list of scenarios.

Saved layout (one folder per scenario):
    disruptions/
        scenario_001/disruption.json
        scenario_002/disruption.json
        ...
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class Disruption:
    scenario_id: int
    edge_id: str        # SUMO edge id that is blocked
    edge_index: int     # position of that edge in the [T, E, F] tensors (must match everywhere)
    start_step: int     # first blocked time-step index (0-based)
    end_step: int       # last blocked time-step index (inclusive)
    start_time: int     # same thing in seconds (from the 'time' column)
    end_time: int
    seed: int


def generate_disruptions(edge_ids, times, n_scenarios, seed=42,
                         min_start_step=12, min_duration=3, max_duration=15,
                         eligible=None, max_tries=1000):
    """
    Randomly create `n_scenarios` single-edge blocks.

    edge_ids       : list of edge ids, in the same order as the model/data tensors
    times          : array of time stamps in seconds, length T (e.g. 0, 60, ..., 5340)
    min_start_step : earliest step a block may start. The prediction model needs
                     `input_len` (12) steps of history, so keep this >= 12.
    min/max_duration : block length in time steps (a block is cut off at the end of the data)
    eligible       : optional boolean array [T, E]. eligible[t, e] = True means "edge e is a sensible
                     thing to block at step t" (for example it has traffic and has feeder edges).
                     If None, any edge can be blocked at any step.
    """
    rng = np.random.default_rng(seed)           # seeded generator -> reproducible scenarios
    T, E = len(times), len(edge_ids)
    last_start = T - min_duration               # leave room for at least min_duration blocked steps
    if min_start_step > last_start:
        raise ValueError("Not enough time steps for the chosen min_start_step / min_duration.")

    scenarios = []
    for sid in range(1, n_scenarios + 1):
        for _ in range(max_tries):
            start = int(rng.integers(min_start_step, last_start + 1))
            candidates = np.arange(E) if eligible is None else np.flatnonzero(eligible[start])
            if len(candidates) == 0:            # nothing sensible to block at this step, try another step
                continue
            edge = int(rng.choice(candidates))
            duration = int(rng.integers(min_duration, max_duration + 1))
            end = min(start + duration - 1, T - 1)
            scenarios.append(Disruption(
                scenario_id=sid, edge_id=str(edge_ids[edge]), edge_index=edge,
                start_step=start, end_step=end,
                start_time=int(times[start]), end_time=int(times[end]), seed=seed,
            ))
            break
        else:
            raise RuntimeError("Could not find an eligible edge/time. Relax the `eligible` mask.")
    return scenarios


def save_disruption(d, root="disruptions"):
    """Write disruptions/scenario_XXX/disruption.json and return that folder."""
    folder = Path(root) / f"scenario_{d.scenario_id:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "disruption.json", "w") as f:
        json.dump(asdict(d), f, indent=2)
    return folder


def load_disruption(folder):
    with open(Path(folder) / "disruption.json") as f:
        return Disruption(**json.load(f))
