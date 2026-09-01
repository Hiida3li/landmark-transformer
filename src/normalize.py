"""Landmark normalization: make gestures invariant to position and scale.

Each frame is a (21, 3) matrix of landmark vectors p_i in R^3.
    p_tilde_i = (p_i - p_0) / ||p_9 - p_0||
where p_0 is the wrist and p_9 is the base of the middle finger.
"""

import numpy as np

WRIST = 0
MIDDLE_MCP = 9          # base knuckle of the middle finger


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """(T, 21, 3) raw landmarks -> (T, 21, 3) centered and scaled landmarks."""
    assert seq.ndim == 3 and seq.shape[1:] == (21, 3), seq.shape

    wrist = seq[:, WRIST:WRIST + 1, :]            # (T, 1, 3)  keep dim for broadcasting
    centered = seq - wrist                        # (T, 21, 3) - (T, 1, 3) -> (T, 21, 3)

    ref = centered[:, MIDDLE_MCP, :]              # (T, 3)   vector wrist -> middle base
    scale = np.linalg.norm(ref, axis=-1)          # (T,)     its Euclidean length
    scale = scale[:, None, None]                  # (T, 1, 1) for broadcasting
    scale = np.maximum(scale, 1e-6)               # never divide by zero

    normalized = centered / scale                 # (T, 21, 3)
    assert normalized.shape == seq.shape
    return normalized.astype(np.float32)


def flatten_sequence(seq: np.ndarray) -> np.ndarray:
    """(T, 21, 3) -> (T, 63): one feature vector per frame (one token per frame)."""
    T = seq.shape[0]
    return seq.reshape(T, 21 * 3)


if __name__ == "__main__":
    import glob
    path = sorted(glob.glob("data/peace/*.npz"))[0]
    raw = np.load(path)["landmarks"]              # (30, 21, 3)
    norm = normalize_sequence(raw)

    np.set_printoptions(precision=3, suppress=True)
    print("raw frame 0, first 3 landmarks:\n", raw[0, :3])
    print("normalized frame 0, first 3 landmarks:\n", norm[0, :3])
    print("wrist after normalization:", norm[0, WRIST])
    print("|wrist->middle_mcp| after normalization:",
          np.linalg.norm(norm[0, MIDDLE_MCP]))
    print("flattened shape:", flatten_sequence(norm).shape)