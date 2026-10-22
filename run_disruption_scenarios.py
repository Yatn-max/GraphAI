"""
Run disruption scenarios
========================
Connects the three parts:
    disruption_generator   -> which edge is blocked, when, for how long
    gated_traffic_model    -> predicted edge weights (just before the block starts)
    traffic_redistributor  -> moves the blocked edge's traffic using those weights

For every scenario it saves:
    <out>/scenario_XXX/disruption.json   what was blocked
    <out>/scenario_XXX/weights_used.csv  the predicted weights used (NOT re-predicted afterwards)
    <out>/scenario_XXX/traffic.csv       traffic before / after, per edge and time step
    <out>/scenario_XXX/summary.json      totals and the edges that received the most traffic

Usage
-----
    python run_disruption_scenarios.py --data "normal_traffic_median_filled - Copy2.xlsx" ^
        --net osm.net.xml.gz --n 20 --seed 42
(On PowerShell use ` instead of ^ to continue a line, or put it all on one line.)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gated_traffic_model import (
    load_tensor, load_graph, load_for_inference, predict_edge_weights, build_label, FeatureScaler,
)
from disruption_generator import generate_disruptions, save_disruption
from traffic_redistributor import TrafficRedistributor


def load_entered(data_path, edge_ids, times):
    """
    Vehicles that ENTERED each edge in each 60 s step -> [T, E].
    (`entered` is a whole-number vehicle count, so it is the cleanest 'traffic on this edge' measure.)
    """
    path = str(data_path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype={"edge_id": str})
    else:
        df = pd.read_excel(path, dtype={"edge_id": str})
    table = df.pivot(index="time", columns="edge_id", values="entered")
    return table.reindex(index=times, columns=edge_ids).to_numpy(dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--ckpt", default="gated_traffic_model.pt")
    ap.add_argument("--out", default="disruptions")
    ap.add_argument("--n", type=int, default=10, help="number of scenarios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-duration", type=int, default=3)
    ap.add_argument("--max-duration", type=int, default=15)
    ap.add_argument("--min-entered", type=float, default=3,
                    help="only block edges that had at least this many vehicles entering at the start step")
    ap.add_argument("--tau", type=float, default=None, help="softmax temperature (default: the one saved in the checkpoint)")
    ap.add_argument("--use-formula-weights", action="store_true",
                    help="skip the trained model and use the formula label of the last observed step as weights "
                         "(only for testing without a checkpoint)")
    a = ap.parse_args()

    # ---- load data, graph, model -----------------------------------------------------
    raw, edge_ids, times = load_tensor(a.data)
    entered = load_entered(a.data, edge_ids, times)
    A_up, A_down, _ = load_graph(a.net, edge_ids)

    if a.use_formula_weights:
        scaler = FeatureScaler().fit(raw)
        formula = build_label(scaler.transform(raw))                  # [T, E]
        model, input_len, tau = None, 12, 0.1
    else:
        model, scaler, ck_edges, _, cfg = load_for_inference(a.ckpt, a.net)
        assert ck_edges == edge_ids, "Edge order in the data differs from the one used in training."
        input_len, tau = cfg.input_len, cfg.tau
    tau = a.tau if a.tau is not None else tau

    redistributor = TrafficRedistributor(A_up, A_down, tau=tau)

    # ---- generate scenarios ----------------------------------------------------------
    # Only block an edge that actually has traffic at the start step AND has at least one feeder edge.
    has_feeder = (A_up > 0).sum(axis=1) > 0
    eligible = (entered >= a.min_entered) & has_feeder[None, :]
    scenarios = generate_disruptions(
        edge_ids, times, a.n, seed=a.seed, min_start_step=input_len,
        min_duration=a.min_duration, max_duration=a.max_duration, eligible=eligible,
    )

    # ---- run each scenario -----------------------------------------------------------
    for d in scenarios:
        # Weights come from the 12 steps just BEFORE the block starts, and are used unchanged.
        if model is None:
            weights = formula[d.start_step - 1]
        else:
            window = raw[d.start_step - input_len:d.start_step]
            weights = predict_edge_weights(model, scaler, window)[:, 0]

        res = redistributor.redistribute(entered, weights, d.edge_index, d.start_step, d.end_step)

        folder = save_disruption(d, a.out)

        pd.DataFrame({"edge_id": edge_ids, "weight": weights}).to_csv(folder / "weights_used.csv", index=False)

        T, E = entered.shape
        pd.DataFrame({
            "time": np.repeat(times, E),
            "edge_id": np.tile(edge_ids, T),
            "traffic_before": entered.reshape(-1),
            "traffic_after": res.traffic_after.reshape(-1),
            "added": res.added.reshape(-1),
            "blocked": ((np.arange(T)[:, None] >= d.start_step) & (np.arange(T)[:, None] <= d.end_step)
                        & (np.arange(E)[None, :] == d.edge_index)).reshape(-1),
        }).to_csv(folder / "traffic.csv", index=False)

        received = res.added.sum(axis=0)
        top = np.argsort(-received)[:5]
        summary = {
            "blocked_edge": d.edge_id,
            "blocked_steps": [d.start_step, d.end_step],
            "vehicles_diverted": float(res.removed.sum() - res.unrouted.sum()),
            "vehicles_unrouted": float(res.unrouted.sum()),
            "top_receiving_edges": [{"edge_id": edge_ids[i], "extra_vehicles": float(received[i])}
                                    for i in top if received[i] > 0],
        }
        with open(folder / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"scenario {d.scenario_id:03d}: block {d.edge_id} steps {d.start_step}-{d.end_step} | "
              f"diverted {summary['vehicles_diverted']:.1f} veh, unrouted {summary['vehicles_unrouted']:.1f}")

    print(f"\nSaved {len(scenarios)} scenarios in '{a.out}'")


if __name__ == "__main__":
    main()
