#!/usr/bin/env python3
"""Export a deterministic GRU-gate interval certificate."""
import math
from typing import Any

from common import affine_interval, centered_box, write_json

# GRU gate graph: 0=input(3) -> 1=linear -> 2=sigmoid; 0 -> 3=linear -> 4=tanh; 5=mul_elem

n = 3


def seed_params():
    """Return deterministic shared gate weights and biases."""
    W = [[float(1 + (i + j)) for j in range(n)] for i in range(n)]
    b = [float(i) for i in range(n)]
    return W, b


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3]`."""
    x0 = [float(i + 1) for i in range(n)]
    return centered_box(x0, eps)


def ibp_sigmoid(lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through elementwise sigmoid."""
    def s(x: float) -> float:
        """Evaluate the logistic sigmoid at one scalar."""
        return 1.0 / (1.0 + math.exp(-x))
    out_lo = []
    out_hi = []
    for l, u in zip(lo, hi):
        sl, su = s(l), s(u)
        out_lo.append(min(sl, su))
        out_hi.append(max(sl, su))
    return out_lo, out_hi


def ibp_tanh(lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through elementwise tanh."""
    def t(x: float) -> float:
        """Evaluate tanh at one scalar."""
        return math.tanh(x)
    out_lo = []
    out_hi = []
    for l, u in zip(lo, hi):
        tl, tu = t(l), t(u)
        out_lo.append(min(tl, tu))
        out_hi.append(max(tl, tu))
    return out_lo, out_hi


def ibp_mul_elem(
    x_lo: list[float],
    x_hi: list[float],
    y_lo: list[float],
    y_hi: list[float],
) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through elementwise multiplication."""
    lo = []
    hi = []
    for lx, ux, ly, uy in zip(x_lo, x_hi, y_lo, y_hi):
        p1 = lx * ly
        p2 = lx * uy
        p3 = ux * ly
        p4 = ux * uy
        lo.append(min(p1, p2, p3, p4))
        hi.append(max(p1, p2, p3, p4))
    return lo, hi


def run_ibp() -> dict[str, Any]:
    """Compute the GRU-gate certificate payload consumed by Lean."""
    W, b = seed_params()
    x_lo, x_hi = seed_input_box(0.5)
    a_lo, a_hi = affine_interval(W, b, x_lo, x_hi)  # node 1
    s_lo, s_hi = ibp_sigmoid(a_lo, a_hi)       # node 2
    b_lo, b_hi = affine_interval(W, b, x_lo, x_hi)  # node 3
    t_lo, t_hi = ibp_tanh(b_lo, b_hi)          # node 4
    y_lo, y_hi = ibp_mul_elem(s_lo, s_hi, t_lo, t_hi)  # node 5
    return {
        "graph": "gru_gate_workflow_v1",
        "input_box": {"id": 0, "dim": n, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 5, "dim": n, "lo": y_lo, "hi": y_hi},
    }


def main():
    """Write the GRU-gate certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/gru_gate_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
