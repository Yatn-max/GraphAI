"""
Gated spatio-temporal traffic model (PyTorch)
=============================================

What this file does
-------------------
1. Loads your per-edge time-series (xlsx/csv) and the SUMO network file (.net.xml / .net.xml.gz).
2. Builds a "congestion weight" label (0..1) for every edge at every time step
   from a weighted combination of the 9 features (you have no ground-truth column).
3. Builds the road graph (which edge feeds which) from the SUMO <connection> tags.
4. Defines a gated model:
       - GRU gates   -> decide how much past congestion to remember / forget (time)
       - Spatial gate -> decides how much upstream / downstream neighbours influence an edge (space)
5. Trains it (you run this yourself) and saves a checkpoint.
6. Inference helpers:
       - predict_edge_weights(...)  : new feature values -> weight of every edge
       - allocate_traffic(...)      : weights -> share of traffic each edge should get
                                      (lower weight = more traffic)

Usage
-----
    python gated_traffic_model.py --data normal_traffic.xlsx --net osm_net_xml.gz --epochs 200

Tensor shape legend
-------------------
    T = time steps (90), E = edges (369), F = features (9)
    B = batch size, L = input length (window), H = horizon (steps predicted)
"""

import argparse
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------------------

# Input features, in the exact order used everywhere in this file.
FEATURES = [
    "density", "laneDensity", "occupancy", "waitingTime", "timeLoss",
    "speed", "speedRelative", "flow", "traveltime",
]

# Recipe for the congestion label:  feature -> (weight, flip)
#   flip=True  means "low value = bad" (e.g. low speed = congestion), so we use (1 - value).
#   After this, a HIGHER label always means a MORE congested edge.
# Weights sum to 1, so the label stays in [0, 1].
# density/laneDensity/occupancy are strongly correlated (0.77-0.94) and so are speed/speedRelative
# (0.96), which is why the "duplicate" signals get small weights.
LABEL_RECIPE = {
    "waitingTime":   (0.20, False),
    "timeLoss":      (0.15, False),
    "speedRelative": (0.15, True),
    "laneDensity":   (0.15, False),
    "occupancy":     (0.10, False),
    "traveltime":    (0.10, False),   # already log-compressed in the scaler below
    "density":       (0.05, False),
    "speed":         (0.05, True),
    "flow":          (0.05, True),
}
assert abs(sum(w for w, _ in LABEL_RECIPE.values()) - 1.0) < 1e-9
assert set(LABEL_RECIPE) == set(FEATURES)


@dataclass
class Config:
    input_len: int = 12        # how many past time steps the model sees
    horizon: int = 1           # how many future steps of weights it predicts
    hidden: int = 64           # size of the hidden state per edge
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 8
    patience: int = 25         # early stopping: stop if val loss doesn't improve this many epochs
    train_frac: float = 0.8    # first 80% of windows (in time order) for training
    tau: float = 0.1           # softmax temperature for traffic allocation (lower = more decisive)
    seed: int = 42


# --------------------------------------------------------------------------------------
# 2. DATA LOADING
# --------------------------------------------------------------------------------------

def load_tensor(path):
    """Read the xlsx/csv and return (raw [T,E,F] array, edge_ids list, times array)."""
    # IMPORTANT: read edge_id as TEXT. Excel silently turns ids like "-465398528" into numbers,
    # and then they no longer match the SUMO net file (69 of your 369 edges would be lost).
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path, dtype={"edge_id": str})
    else:
        df = pd.read_excel(path, dtype={"edge_id": str})

    edge_ids = sorted(df["edge_id"].unique())          # fixed, deterministic edge order
    times = np.sort(df["time"].unique())

    # One [T, E] table per feature, then stack them into [T, E, F].
    per_feature = []
    for f in FEATURES:
        table = df.pivot(index="time", columns="edge_id", values=f)
        table = table.reindex(index=times, columns=edge_ids)
        per_feature.append(table.to_numpy(dtype=np.float32))
    raw = np.stack(per_feature, axis=-1)
    if np.isnan(raw).any():
        raise ValueError("Missing (edge, time) rows found - every edge needs a row for every time step.")
    return raw, edge_ids, times


# --------------------------------------------------------------------------------------
# 3. SCALING + LABEL
# --------------------------------------------------------------------------------------

class FeatureScaler:
    """
    Turns raw features into 0..1 numbers.
      - traveltime is capped (your data has 100000 placeholders for stuck edges) and log-compressed
      - every feature is then min-max scaled
    Fitted on TRAINING time steps only, so validation data can't leak into the scaling.
    """

    def __init__(self):
        self.tt_idx = FEATURES.index("traveltime")
        self.tt_cap = None
        self.lo = None
        self.hi = None

    def _pre(self, raw):
        x = raw.copy()
        x[..., self.tt_idx] = np.log1p(np.minimum(x[..., self.tt_idx], self.tt_cap))
        return x

    def fit(self, raw):
        self.tt_cap = float(np.percentile(raw[..., self.tt_idx], 95))
        x = self._pre(raw)
        flat = x.reshape(-1, x.shape[-1])
        self.lo = flat.min(axis=0)
        self.hi = flat.max(axis=0)
        return self

    def transform(self, raw):
        x = self._pre(np.asarray(raw, dtype=np.float32))
        scaled = (x - self.lo) / np.maximum(self.hi - self.lo, 1e-8)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)   # clip handles unseen extreme values

    def state_dict(self):
        return {"tt_cap": self.tt_cap, "lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_state(cls, s):
        o = cls()
        o.tt_cap = s["tt_cap"]
        o.lo = np.array(s["lo"], dtype=np.float32)
        o.hi = np.array(s["hi"], dtype=np.float32)
        return o


def build_label(scaled):
    """scaled [T,E,F] (0..1)  ->  congestion label [T,E] (0..1) using LABEL_RECIPE."""
    label = np.zeros(scaled.shape[:2], dtype=np.float32)
    for f, (w, flip) in LABEL_RECIPE.items():
        v = scaled[..., FEATURES.index(f)]
        label += w * ((1.0 - v) if flip else v)
    return label


def make_windows(scaled, label, input_len, horizon):
    """
    Slide a window over time.
      x: features for steps [s, s+L)          -> [N, L, E, F]
      y: labels   for steps [s+L, s+L+H)      -> [N, E, H]
    """
    T = scaled.shape[0]
    xs, ys = [], []
    for s in range(T - input_len - horizon + 1):
        xs.append(scaled[s:s + input_len])
        ys.append(label[s + input_len:s + input_len + horizon].T)   # [H,E] -> [E,H]
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


# --------------------------------------------------------------------------------------
# 4. ROAD GRAPH FROM THE SUMO NET FILE
# --------------------------------------------------------------------------------------

def load_graph(net_path, edge_ids):
    """
    Returns
      A_up   [E,E]: row e averages the edges that FEED INTO e   (upstream neighbours)
      A_down [E,E]: row e averages the edges e FEEDS INTO       (downstream neighbours)
      from_nodes   : list, the junction each edge starts at (used to group edges for allocation)
    """
    opener = gzip.open if str(net_path).endswith(".gz") else open
    with opener(net_path, "rb") as fh:
        root = ET.parse(fh).getroot()

    index = {e: i for i, e in enumerate(edge_ids)}
    E = len(edge_ids)

    from_nodes = [None] * E
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":      # skip SUMO's internal junction lanes
            continue
        eid = edge.get("id")
        if eid in index:
            from_nodes[index[eid]] = edge.get("from")
    if any(n is None for n in from_nodes):
        raise ValueError("Some edge_ids in the data were not found in the net file.")

    A_up = np.zeros((E, E), dtype=np.float32)
    A_down = np.zeros((E, E), dtype=np.float32)
    for c in root.findall("connection"):
        a, b = c.get("from"), c.get("to")            # vehicles can drive from edge a onto edge b
        if a in index and b in index and a != b:
            A_up[index[b], index[a]] = 1.0
            A_down[index[a], index[b]] = 1.0

    # Row-normalise so each row is an average over neighbours (edges with no neighbours stay 0).
    for A in (A_up, A_down):
        A /= np.maximum(A.sum(axis=1, keepdims=True), 1.0)
    return A_up, A_down, from_nodes


# --------------------------------------------------------------------------------------
# 5. THE MODEL
# --------------------------------------------------------------------------------------

class SpatialGate(nn.Module):
    """
    Lets each edge look at its upstream and downstream neighbours.
    A learned gate (values 0..1, per hidden unit) decides how much neighbour information to let in.
    """

    def __init__(self, d, A_up, A_down):
        super().__init__()
        self.register_buffer("A_up", torch.tensor(A_up))
        self.register_buffer("A_down", torch.tensor(A_down))
        self.gate = nn.Linear(3 * d, d)
        self.msg = nn.Linear(2 * d, d)

    def forward(self, h):                                   # h: [B,E,d]
        m_up = torch.matmul(self.A_up, h)                   # average of upstream neighbours' states
        m_down = torch.matmul(self.A_down, h)               # average of downstream neighbours' states
        g = torch.sigmoid(self.gate(torch.cat([h, m_up, m_down], dim=-1)))   # how open the gate is
        m = torch.tanh(self.msg(torch.cat([m_up, m_down], dim=-1)))          # neighbour message
        return h + g * m                                    # own state + gated neighbour influence


class GatedTrafficModel(nn.Module):
    """
    For each time step in the window:
        1. SpatialGate : mix in neighbour information
        2. GRUCell     : update the edge's memory using the new features (time gates)
    After the last step, an MLP head turns each edge's memory into weight(s) in 0..1.
    """

    def __init__(self, n_edges, n_features, A_up, A_down, hidden=64, horizon=1, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.in_proj = nn.Linear(n_features, hidden)
        self.edge_emb = nn.Embedding(n_edges, hidden)      # lets each edge learn its own "personality"
        self.spatial = SpatialGate(hidden, A_up, A_down)
        self.cell = nn.GRUCell(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, horizon))

    def forward(self, x):                                   # x: [B,L,E,F]
        B, L, E, _ = x.shape
        d = self.hidden
        h = x.new_zeros(B, E, d)
        ids = torch.arange(E, device=x.device)
        for t in range(L):
            xt = self.in_proj(x[:, t]) + self.edge_emb(ids)                 # [B,E,d]
            h = self.spatial(h)                                             # spatial gating
            h = self.cell(xt.reshape(B * E, d), h.reshape(B * E, d)).view(B, E, d)   # temporal gating
        return torch.sigmoid(self.head(self.drop(h)))                      # [B,E,H] in 0..1


# --------------------------------------------------------------------------------------
# 6. TRAINING
# --------------------------------------------------------------------------------------

def train(data_path, net_path, cfg, ckpt_path="gated_traffic_model.pt", device=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    raw, edge_ids, _ = load_tensor(data_path)
    A_up, A_down, _ = load_graph(net_path, edge_ids)
    T, E, F = raw.shape
    print(f"Data: T={T} steps, E={E} edges, F={F} features | graph links: {int((A_up > 0).sum())}")

    # Split by TIME (never shuffle before splitting - that would leak the future).
    n_windows = T - cfg.input_len - cfg.horizon + 1
    n_train = int(n_windows * cfg.train_frac)
    train_end_t = n_train + cfg.input_len + cfg.horizon       # last time step touched by training windows

    scaler = FeatureScaler().fit(raw[:train_end_t])           # fit scaling on training period only
    scaled = scaler.transform(raw)
    label = build_label(scaled)

    X, Y = make_windows(scaled, label, cfg.input_len, cfg.horizon)
    train_dl = DataLoader(TensorDataset(X[:n_train], Y[:n_train]), batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X[n_train:], Y[n_train:]), batch_size=cfg.batch_size)
    print(f"Windows: {n_train} train / {n_windows - n_train} validation")

    model = GatedTrafficModel(E, F, A_up, A_down, cfg.hidden, cfg.horizon, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best, bad_epochs = float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # keeps GRU training stable
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= max(len(train_dl.dataset), 1)

        model.eval()
        va_loss, va_mae = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_loss += loss_fn(pred, yb).item() * len(xb)
                va_mae += (pred - yb).abs().mean().item() * len(xb)
        n_val = max(len(val_dl.dataset), 1)
        va_loss, va_mae = va_loss / n_val, va_mae / n_val

        print(f"epoch {epoch:3d} | train MSE {tr_loss:.5f} | val MSE {va_loss:.5f} | val MAE {va_mae:.4f}")

        if va_loss < best - 1e-6:
            best, bad_epochs = va_loss, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": asdict(cfg),
                "scaler": scaler.state_dict(),
                "edge_ids": edge_ids,
                "features": FEATURES,
                "n_features": F,
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping (best val MSE {best:.5f}). Checkpoint: {ckpt_path}")
                break
    return ckpt_path


# --------------------------------------------------------------------------------------
# 7. INFERENCE: new feature values -> edge weights -> traffic share per edge
# --------------------------------------------------------------------------------------

def load_for_inference(ckpt_path, net_path, device="cpu"):
    """Rebuild the trained model + scaler + graph info from a checkpoint."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ck["config"])
    edge_ids = ck["edge_ids"]
    A_up, A_down, from_nodes = load_graph(net_path, edge_ids)
    model = GatedTrafficModel(len(edge_ids), ck["n_features"], A_up, A_down,
                              cfg.hidden, cfg.horizon, cfg.dropout).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    scaler = FeatureScaler.from_state(ck["scaler"])
    return model, scaler, edge_ids, from_nodes, cfg


@torch.no_grad()
def predict_edge_weights(model, scaler, raw_window, device="cpu"):
    """
    raw_window: numpy array [L, E, F] of RAW feature values (same units as your xlsx,
                features in the order of FEATURES, edges in the order of the saved edge_ids).
    returns   : numpy array [E, H] of predicted congestion weights (0..1; higher = more congested).
    """
    x = torch.tensor(scaler.transform(raw_window)).unsqueeze(0).to(device)   # [1,L,E,F]
    return model(x)[0].cpu().numpy()


def allocate_traffic(weights, from_nodes, tau=0.1, node_volume=None):
    """
    Split traffic among the edges that LEAVE THE SAME JUNCTION. Lower weight -> larger share.

        share_e = exp(-w_e / tau) / sum over edges at the same junction of exp(-w / tau)

    weights    : [E] weights for one time step (e.g. predict_edge_weights(...)[:, 0])
    from_nodes : junction each edge starts at (returned by load_for_inference)
    tau        : temperature. Small = nearly all traffic to the best edge; large = more even split.
    node_volume: optional dict {junction_id: vehicles}. If given, returns vehicle counts instead of shares.
    returns    : [E] array of shares (each junction's shares sum to 1), or vehicle counts.
    """
    weights = np.asarray(weights, dtype=np.float64)
    groups = defaultdict(list)
    for i, n in enumerate(from_nodes):
        groups[n].append(i)

    out = np.zeros(len(weights))
    for node, idx in groups.items():
        z = -weights[idx] / tau
        z -= z.max()                                  # numerical stability
        p = np.exp(z)
        p /= p.sum()
        out[idx] = p * (node_volume.get(node, 0.0) if node_volume else 1.0)
    return out


# --------------------------------------------------------------------------------------
# 8. COMMAND LINE
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="xlsx or csv with the per-edge time series")
    ap.add_argument("--net", required=True, help="SUMO net file (.net.xml or .net.xml.gz)")
    ap.add_argument("--out", default="gated_traffic_model.pt")
    defaults = Config()
    for k, v in asdict(defaults).items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    cfg = Config(**{k: getattr(a, k) for k in asdict(defaults)})
    train(a.data, a.net, cfg, a.out)
