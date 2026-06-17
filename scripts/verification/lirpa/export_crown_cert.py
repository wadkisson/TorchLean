#!/usr/bin/env python3
"""Export a tiny transformer-encoder-style interval certificate."""
import math
from typing import Any

from common import affine_interval, centered_box, matmul_interval, relu_interval, softmax_interval, write_json

# Tiny transformer-encoder-like graph IBP, mirroring
# `NN.Verification.LiRPA.TransformerEncoder`.
# Node ids: 0=input(4), 1=linear Wq (5), 2=softmax, 3=matmul Wv (4), 4=add residual,
# 5=layernorm, 6=linear W1 (6), 7=relu, 8=linear W2 (4), 9=add, 10=layernorm

nModel = 4
scoresDim = 5
nHidden = 6

def seed_params():
    """Return deterministic weights for the transformer-like fixture graph."""
    # Wq[i,j] = 1 + (i + 2*j); bq[i] = 0.1 * i
    Wq = [[float(1 + (i + 2*j)) for j in range(nModel)] for i in range(scoresDim)]
    bq = [0.1 * float(i) for i in range(scoresDim)]
    # Wv[i,j] = 2 + (i + j)
    Wv = [[float(2 + (i + j)) for j in range(scoresDim)] for i in range(nModel)]
    # W1[i,j] = 1 + ((i + j) % 3); b1[i] = 0.05 * i
    W1 = [[float(1 + ((i + j) % 3)) for j in range(nModel)] for i in range(nHidden)]
    b1 = [0.05 * float(i) for i in range(nHidden)]
    # W2[i,j] = 2 + ((i + j) % 4); b2[i] = 0.02 * i
    W2 = [[float(2 + ((i + j) % 4)) for j in range(nHidden)] for i in range(nModel)]
    b2 = [0.02 * float(i) for i in range(nModel)]
    return Wq, bq, Wv, W1, b1, W2, b2


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3, 4]`."""
    x0 = [float(i + 1) for i in range(nModel)]
    return centered_box(x0, eps)


def ibp_add(lo1: list[float], hi1: list[float], lo2: list[float], hi2: list[float]):
    """Add two interval vectors elementwise."""
    return [a + c for a, c in zip(lo1, lo2)], [b + d for b, d in zip(hi1, hi2)]


def ibp_layernorm(lo: list[float], hi: list[float], eps: float = 1e-6):
    """Propagate coarse interval bounds through a layernorm-like normalization."""
    n = len(lo)
    sum_lo = sum(lo)
    sum_hi = sum(hi)
    mu_lo = sum_lo / n
    mu_hi = sum_hi / n
    # Upper bound on variance via worst-case deviations
    sum_abs_sq = 0.0
    for i in range(n):
        dl = abs(lo[i] - mu_hi)
        du = abs(hi[i] - mu_lo)
        a = max(dl, du)
        sum_abs_sq += a * a
    var_hi = sum_abs_sq / n
    den_lo = math.sqrt(eps)
    den_hi = math.sqrt(var_hi + eps)
    out_lo = []
    out_hi = []
    for i in range(n):
        dl = lo[i] - mu_hi
        du = hi[i] - mu_lo
        cands = [dl / den_lo, dl / den_hi, du / den_lo, du / den_hi]
        out_lo.append(min(cands))
        out_hi.append(max(cands))
    return out_lo, out_hi


def run_ibp() -> dict[str, Any]:
    """Compute the transformer-like certificate payload consumed by Lean."""
    Wq, bq, Wv, W1, b1, W2, b2 = seed_params()
    x_lo, x_hi = seed_input_box(0.5)
    s_lo, s_hi = affine_interval(Wq, bq, x_lo, x_hi)     # node 1
    p_lo, p_hi = softmax_interval(s_lo, s_hi)            # node 2
    a_lo, a_hi = matmul_interval(Wv, p_lo, p_hi)         # node 3
    r1_lo, r1_hi = ibp_add(x_lo, x_hi, a_lo, a_hi)       # node 4
    n1_lo, n1_hi = ibp_layernorm(r1_lo, r1_hi)           # node 5
    h_lo, h_hi = affine_interval(W1, b1, n1_lo, n1_hi)   # node 6
    h_lo, h_hi = relu_interval(h_lo, h_hi)               # node 7
    o_lo, o_hi = affine_interval(W2, b2, h_lo, h_hi)     # node 8
    r2_lo, r2_hi = ibp_add(n1_lo, n1_hi, o_lo, o_hi)     # node 9
    n2_lo, n2_hi = ibp_layernorm(r2_lo, r2_hi)           # node 10

    return {
        "graph": "transformer_encoder_workflow_v1",
        "input_box": {"id": 0, "dim": nModel, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 10, "dim": nModel, "lo": n2_lo, "hi": n2_hi}
    }


def main():
    """Write the transformer-like certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/transformer_encoder_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")

if __name__ == "__main__":
    main()
