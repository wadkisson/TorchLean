#!/usr/bin/env python3
"""Export a deterministic MLP interval certificate matching the Lean fixture."""

from typing import Any

from common import affine_interval, centered_box, relu_interval, write_json

# MLP graph:
# input -> hidden linear -> ReLU -> output linear.

nIn = 3
nH = 4
nOut = 2


def seed_params():
    """Return deterministic weights and biases for the tiny MLP graph."""
    # Deterministic weights matching the Lean checker workflow.
    hidden_weight = [[float(1 + (i + j)) for j in range(nIn)] for i in range(nH)]
    hidden_bias = [float(i + 1) for i in range(nH)]
    output_weight = [[float(2 + (i + j)) for j in range(nH)] for i in range(nOut)]
    output_bias = [float(i) for i in range(nOut)]
    return hidden_weight, hidden_bias, output_weight, output_bias


def seed_input_box(eps: float = 1.0):
    """Return the input interval box centered at `[1, 2, 3]`."""
    input_center = [float(i + 1) for i in range(nIn)]
    return centered_box(input_center, eps)


def run_ibp() -> dict[str, Any]:
    """Compute the certificate payload consumed by the Lean LiRPA checker."""
    hidden_weight, hidden_bias, output_weight, output_bias = seed_params()
    x_lo, x_hi = seed_input_box(1.0)
    hidden_lo, hidden_hi = affine_interval(hidden_weight, hidden_bias, x_lo, x_hi)
    hidden_lo, hidden_hi = relu_interval(hidden_lo, hidden_hi)
    output_lo, output_hi = affine_interval(output_weight, output_bias, hidden_lo, hidden_hi)
    return {
        "graph": "mlp_graph_workflow_v1",
        "input_box": {"id": 0, "dim": nIn, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 3, "dim": nOut, "lo": output_lo, "hi": output_hi},
    }


def main():
    """Write the MLP certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/mlp_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
