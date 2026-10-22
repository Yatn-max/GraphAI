"""
Traffic Redistributor
=====================
Job: given a blocked edge, move the traffic that wanted to use it onto other edges,
using the edge weights from the prediction model (lower weight -> more traffic).

It does NOT predict weights and it does NOT generate disruptions. It only needs:
    traffic   [T, E]  vehicles per edge per time step (we use the `entered` column)
    weights   [E]     predicted congestion weight per edge (0..1, higher = more congested)
    A_up, A_down      road connections from gated_traffic_model.load_graph()

Weights are used as given for the whole disruption: they are NOT re-predicted after the
redistribution (the receiving edges get busier, but their weights stay as predicted).

How the traffic is moved, for each blocked time step t
------------------------------------------------------
1. volume  = vehicles that entered the blocked edge b at step t. These must go elsewhere.
2. feeders = edges that lead INTO b. The volume is split between them in proportion to the
             traffic on each feeder at step t (equal split if the feeders are all empty).
3. For each feeder p: its other exits (everything p connects to, except b) compete.
   Each exit gets a share  exp(-w / tau) / sum(exp(-w / tau))  -> low weight = more traffic.
4. Dead end: if b is p's ONLY exit, vehicles on p cannot turn around, so we walk back
   upstream to the nearest edge that does have a choice, and do step 3 there
   (up to `max_back` edges back). If no choice is found, the vehicles are counted as `unrouted`.

Level 1 limitation: traffic moves ONE hop onto the alternative edges. It is not followed all the
way to its final destination.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RedistributionResult:
    traffic_after: np.ndarray    # [T, E] traffic after the block + diversion
    added: np.ndarray            # [T, E] extra vehicles each edge received
    removed: np.ndarray          # [T]    vehicles taken off the blocked edge
    unrouted: np.ndarray         # [T]    vehicles that could not be moved anywhere


class TrafficRedistributor:
    def __init__(self, A_up, A_down, tau=0.1, max_back=3):
        """
        A_up[e, p]   > 0  means edge p feeds into edge e   (p is upstream of e)
        A_down[e, s] > 0  means edge e feeds into edge s   (s is downstream of e)
        tau          : softmax temperature (lower = nearly all traffic to the best edge)
        max_back     : how many edges to walk back upstream when an edge has no alternative exit
        """
        self.E = A_up.shape[0]
        self.preds = [np.flatnonzero(A_up[e]) for e in range(self.E)]
        self.succs = [np.flatnonzero(A_down[e]) for e in range(self.E)]
        self.tau = tau
        self.max_back = max_back

    # ---------------------------------------------------------------------------------
    def _shares(self, weights, alts):
        """Softmax over negative weights: lower weight -> bigger share. Shares sum to 1."""
        z = -np.asarray(weights[alts], dtype=np.float64) / self.tau
        z -= z.max()                                    # numerical stability
        p = np.exp(z)
        return p / p.sum()

    def _split_by_traffic(self, volume, edges, traffic_t):
        """Split `volume` across `edges` in proportion to their current traffic (equal if all empty)."""
        t = traffic_t[edges].astype(np.float64)
        frac = t / t.sum() if t.sum() > 0 else np.full(len(edges), 1.0 / len(edges))
        return volume * frac

    def _divert(self, u, came_from, volume, depth, blocked, weights, traffic_t, added_t):
        """
        Vehicles are on edge `u`, heading to `came_from` (which is blocked or leads to the block).
        Spread them over u's other exits. Returns the volume that could NOT be placed.
        """
        alts = np.array([s for s in self.succs[u] if s != came_from and s != blocked], dtype=int)
        if len(alts) > 0:
            added_t[alts] += volume * self._shares(weights, alts)
            return 0.0

        # No alternative here -> walk back upstream to find an edge that has a choice.
        if depth >= self.max_back:
            return volume
        ups = np.array([q for q in self.preds[u] if q != blocked and q != u], dtype=int)
        if len(ups) == 0:
            return volume
        lost = 0.0
        for q, v_q in zip(ups, self._split_by_traffic(volume, ups, traffic_t)):
            lost += self._divert(q, u, v_q, depth + 1, blocked, weights, traffic_t, added_t)
        return lost

    # ---------------------------------------------------------------------------------
    def redistribute(self, traffic, weights, blocked, start, end):
        """
        traffic : [T, E] vehicles per edge per step (not modified)
        weights : [E] predicted weights (not modified)
        blocked : index of the blocked edge
        start, end : first and last blocked step (inclusive)
        """
        traffic = np.asarray(traffic, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        T = traffic.shape[0]
        added = np.zeros_like(traffic)
        removed = np.zeros(T)
        unrouted = np.zeros(T)

        feeders = np.array([p for p in self.preds[blocked] if p != blocked], dtype=int)

        for t in range(start, min(end, T - 1) + 1):
            volume = traffic[t, blocked]
            removed[t] = volume
            if volume <= 0:
                continue
            if len(feeders) == 0:                       # nothing leads into the blocked edge
                unrouted[t] = volume
                continue
            for p, v_p in zip(feeders, self._split_by_traffic(volume, feeders, traffic[t])):
                unrouted[t] += self._divert(p, blocked, v_p, 0, blocked, weights, traffic[t], added[t])

        after = traffic + added
        after[start:end + 1, blocked] = 0.0             # the blocked edge carries nothing
        return RedistributionResult(after, added, removed, unrouted)
