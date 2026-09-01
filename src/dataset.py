"""Load .npz gesture samples into PyTorch datasets, split by recording session."""

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.normalize import normalize_sequence, flatten_sequence


def load_all_samples(data_dir: str, gestures: list[str]):
    """Return (X, y, sessions): X (N, T, 63) float32, y (N,) int64, sessions (N,) str."""
    X, y, sessions = [], [], []
    for label_idx, gesture in enumerate(gestures):
        for path in sorted(glob.glob(f"{data_dir}/{gesture}/*.npz")):
            d = np.load(path)
            seq = flatten_sequence(normalize_sequence(d["landmarks"]))   # (T, 63)
            X.append(seq)
            y.append(label_idx)
            sessions.append(str(d["session"]))
    X = np.stack(X).astype(np.float32)          # (N, T, 63)
    y = np.array(y, dtype=np.int64)             # (N,)
    return X, y, np.array(sessions)


def split_by_session(sessions: np.ndarray, y: np.ndarray, seed: int = 0):
    """Hold out one whole session per class for validation (stratified, no leakage)."""
    rng = np.random.default_rng(seed)
    val_sessions = set()
    for label in np.unique(y):
        label_sessions = np.unique(sessions[y == label])   # the sessions of this class
        val_sessions.add(rng.choice(label_sessions))       # pick one at random
    is_val = np.array([s in val_sessions for s in sessions])
    return ~is_val, is_val                   # boolean masks (train, val)


class GestureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)            # (N, T, 63)
        self.y = torch.from_numpy(y)            # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]             # (T, 63), scalar


def make_dataloaders(cfg: dict, batch_size: int = 32):
    X, y, sessions = load_all_samples(cfg["data_dir"], cfg["gestures"])
    train_mask, val_mask = split_by_session(sessions, y)
    train_ds = GestureDataset(X[train_mask], y[train_mask])
    val_ds = GestureDataset(X[val_mask], y[val_mask])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("configs/config.yaml"))
    X, y, sessions = load_all_samples(cfg["data_dir"], cfg["gestures"])
    print("X:", X.shape, "y:", y.shape, "unique sessions:", len(np.unique(sessions)))
    train_mask, val_mask = split_by_session(sessions, y)
    print("train:", train_mask.sum(), "val:", val_mask.sum())
    print("val labels present:", np.unique(y[val_mask]))

    train_dl, val_dl = make_dataloaders(cfg)
    xb, yb = next(iter(train_dl))
    print("one batch  X:", xb.shape, "y:", yb.shape, "dtype:", xb.dtype)