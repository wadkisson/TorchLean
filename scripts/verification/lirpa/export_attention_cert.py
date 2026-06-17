#!/usr/bin/env python3
"""Export a deterministic attention-softmax interval certificate."""

from typing import Any

from common import centered_box, matmul_interval, softmax_interval, write_json

# Attention-softmax graph: 0=input(4) -> 1=matmul Wq (5) -> 2=softmax -> 3=matmul Wv (3)

nIn = 4
nScores = 5
nOut = 3


def seed_params():
    """Return deterministic query/value matrices for the attention fixture."""
    Wq = [[float(1 + (i + 2 * j)) for j in range(nIn)] for i in range(nScores)]
    Wv = [[float(2 + (i + j)) for j in range(nScores)] for i in range(nOut)]
    return Wq, Wv


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3, 4]`."""
    x0 = [float(i + 1) for i in range(nIn)]
    return centered_box(x0, eps)


def run_ibp() -> dict[str, Any]:
    """Compute the attention certificate payload consumed by Lean."""
    Wq, Wv = seed_params()
    x_lo, x_hi = seed_input_box(0.5)
    s_lo, s_hi = matmul_interval(Wq, x_lo, x_hi)    # node 1
    p_lo, p_hi = softmax_interval(s_lo, s_hi)       # node 2
    y_lo, y_hi = matmul_interval(Wv, p_lo, p_hi)    # node 3
    return {
        "graph": "attention_softmax_workflow_v1",
        "input_box": {"id": 0, "dim": nIn, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 3, "dim": nOut, "lo": y_lo, "hi": y_hi},
    }


def main():
    """Write the attention-softmax certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/attention_softmax_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
