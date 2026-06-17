#!/usr/bin/env python3
"""Export a deterministic MLP interval certificate matching the Lean fixture."""

from typing import Any

from common import affine_interval, centered_box, relu_interval, write_json

# MLP graph: 0=input(3) -> 1=linear(4) -> 2=relu -> 3=linear(2)

nIn = 3
nH = 4
nOut = 2


def seed_params():
    """Return deterministic weights and biases for the tiny MLP graph."""
    # Deterministic weights matching the Lean checker workflow.
    W1 = [[float(1 + (i + j)) for j in range(nIn)] for i in range(nH)]
    b1 = [float(i + 1) for i in range(nH)]
    W2 = [[float(2 + (i + j)) for j in range(nH)] for i in range(nOut)]
    b2 = [float(i) for i in range(nOut)]
    return W1, b1, W2, b2


def seed_input_box(eps: float = 1.0):
    """Return the input interval box centered at `[1, 2, 3]`."""
    x0 = [float(i + 1) for i in range(nIn)]
    return centered_box(x0, eps)


def run_ibp() -> dict[str, Any]:
    """Compute the certificate payload consumed by the Lean LiRPA checker."""
    W1, b1, W2, b2 = seed_params()
    x_lo, x_hi = seed_input_box(1.0)
    h_lo, h_hi = affine_interval(W1, b1, x_lo, x_hi)  # node 1
    h_lo, h_hi = relu_interval(h_lo, h_hi)             # node 2
    y_lo, y_hi = affine_interval(W2, b2, h_lo, h_hi)   # node 3
    return {
        "graph": "mlp_graph_workflow_v1",
        "input_box": {"id": 0, "dim": nIn, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 3, "dim": nOut, "lo": y_lo, "hi": y_hi},
    }


def main():
    """Write the MLP certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/mlp_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
