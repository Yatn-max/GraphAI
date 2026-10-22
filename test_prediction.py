"""
Test the trained gated traffic model.

What it does
------------
1. Loads the trained checkpoint (gated_traffic_model.pt) and your data.
2. Takes 12 consecutive time steps as the input window.
3. Predicts the congestion weight of every edge for the next step.
4. Compares the prediction with the "true" label (from the weighted formula) when it is available.
5. Splits traffic at junctions (lower weight -> more traffic) and saves everything to a CSV.

Usage
-----
    python test_prediction.py --data "normal_traffic_median_filled - Copy2.xlsx" --net osm.net.xml.gz
    python test_prediction.py --data "..." --net osm.net.xml.gz --start 30     # window = steps 30..41
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from gated_traffic_model import (
    load_tensor, load_for_inference, predict_edge_weights, allocate_traffic, build_label,
)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--net", required=True)
ap.add_argument("--ckpt", default="gated_traffic_model.pt")
ap.add_argument("--start", type=int, default=None,
                help="first time-step index of the 12-step window (default: the last usable window)")
ap.add_argument("--out", default="predicted_edge_weights.csv")
args = ap.parse_args()

# ---- load model + data -------------------------------------------------------------------
model, scaler, edge_ids, from_nodes, cfg = load_for_inference(args.ckpt, args.net)
raw, data_edge_ids, times = load_tensor(args.data)
assert data_edge_ids == edge_ids, "Edge order in the data differs from the one used in training."

L, H = cfg.input_len, cfg.horizon
T = raw.shape[0]
start = args.start if args.start is not None else T - L - H
assert 0 <= start <= T - L, f"--start must be between 0 and {T - L}"

# ---- predict -----------------------------------------------------------------------------
window = raw[start:start + L]                              # [12, 369, 9] raw values
weights = predict_edge_weights(model, scaler, window)      # [369, H]
w = weights[:, 0]                                          # first predicted step
target_step = start + L
print(f"Input window : time steps {start}..{start + L - 1}  ({times[start]}s to {times[start + L - 1]}s)")
print(f"Predicting   : time step {target_step}" + (f"  ({times[target_step]}s)" if target_step < T else " (beyond the data)"))

# ---- accuracy check (only possible if that step exists in the data) ------------------------
if target_step < T:
    truth = build_label(scaler.transform(raw))[target_step]   # "true" weight from the formula
    mae = np.abs(w - truth).mean()
    corr = np.corrcoef(w, truth)[0, 1]
    print(f"\nAccuracy on that step:  MAE = {mae:.4f}   correlation with true weights = {corr:.3f}")
    print("  (MAE near 0 and correlation near 1 means the model is working well)")
else:
    truth = np.full_like(w, np.nan)

# ---- most / least congested edges ----------------------------------------------------------
order = np.argsort(-w)
print("\nTop 5 most congested edges (predicted):")
for i in order[:5]:
    print(f"  {edge_ids[i]:>20}  weight = {w[i]:.3f}")
print("Top 5 least congested edges (predicted):")
for i in order[-5:][::-1]:
    print(f"  {edge_ids[i]:>20}  weight = {w[i]:.3f}")

# ---- traffic allocation --------------------------------------------------------------------
shares = allocate_traffic(w, from_nodes, tau=cfg.tau)

groups = defaultdict(list)
for i, n in enumerate(from_nodes):
    groups[n].append(i)
biggest = max(groups, key=lambda n: len(groups[n]))
print(f"\nExample junction {biggest} ({len(groups[biggest])} outgoing edges):")
for i in sorted(groups[biggest], key=lambda i: w[i]):
    print(f"  {edge_ids[i]:>20}  weight = {w[i]:.3f}  ->  traffic share = {shares[i] * 100:5.1f}%")

# ---- save everything -----------------------------------------------------------------------
pd.DataFrame({
    "edge_id": edge_ids,
    "from_junction": from_nodes,
    "predicted_weight": w,
    "true_weight": truth,
    "traffic_share": shares,
}).to_csv(args.out, index=False)
print(f"\nSaved all {len(edge_ids)} edges to {args.out}")
