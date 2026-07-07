#!/usr/bin/env python3
"""Export a compact deterministic PINN residual certificate."""
import json
import math
from typing import List, Dict, Any, Tuple

# Compact PINN-style workflow: 1D network u(x) approximates a solution on x in [0,1].
# Build a small tanh MLP (1 -> 16 -> 16 -> 1) with deterministic weights.
# Compute Interval Bound Propagation (IBP) bounds for u at x-h, x, x+h and produce
# a conservative bound on the finite-difference residual r ≈ (u(x+h) - 2u(x) + u(x-h)) / h^2.
# Use f(x) = 0 (homogeneous Poisson) so the exported certificate stays small and deterministic.

HIDDEN_WIDTH, MIDDLE_WIDTH = 16, 16


def seed_weights() -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    """Return deterministic tanh-MLP weights used by the bundled PINN fixture."""
    first_weight = [[(i + 1) * 0.1] for i in range(HIDDEN_WIDTH)]
    first_bias = [0.05 * (i - 8) for i in range(HIDDEN_WIDTH)]
    output_weight = [[0.1 + 0.01 * j for j in range(MIDDLE_WIDTH)]]
    output_bias = [0.0]
    # Middle layer 16x16 with mild coupling
    middle_weight = [[(1.0 if i == j else 0.05) for j in range(MIDDLE_WIDTH)]
                     for i in range(HIDDEN_WIDTH)]
    middle_bias = [0.0 for _ in range(HIDDEN_WIDTH)]
    return first_weight, first_bias, middle_weight, middle_bias, output_weight, output_bias


def ibp_linear(W: List[List[float]], b: List[float], lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate interval bounds through an affine layer."""
    m, n = len(W), len(W[0])
    out_lo, out_hi = [], []
    for i in range(m):
        li, ui = b[i], b[i]
        for j in range(n):
            a = W[i][j]
            p, q = a * lo[j], a * hi[j]
            li += min(p, q)
            ui += max(p, q)
        out_lo.append(li)
        out_hi.append(ui)
    return out_lo, out_hi


def ibp_tanh(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate interval bounds through elementwise tanh."""
    out_lo, out_hi = [], []
    for l, u in zip(lo, hi):
        tl, tu = math.tanh(l), math.tanh(u)
        out_lo.append(min(tl, tu))
        out_hi.append(max(tl, tu))
    return out_lo, out_hi


def fd_residual_bounds(
    u_minus: Tuple[float, float],
    u_center: Tuple[float, float],
    u_plus: Tuple[float, float],
    h: float,
) -> Tuple[float, float]:
    """Bound the centered finite-difference second derivative residual."""
    # u_minus, u_center, u_plus are (lo, hi)
    l_minus, h_minus = u_minus
    center_lo, center_hi = u_center
    l_plus, h_plus = u_plus
    # r_num = u+ - 2u0 + u-
    num_lo = l_plus - 2.0 * center_hi + l_minus
    num_hi = h_plus - 2.0 * center_lo + h_minus
    s = 1.0 / (h * h)
    return num_lo * s, num_hi * s


# Interval helpers
def mul_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Multiply two scalar intervals."""
    (al, ah), (bl, bh) = a, b
    products = [al * bl, al * bh, ah * bl, ah * bh]
    return min(products), max(products)


def add_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Add two scalar intervals."""
    return a[0] + b[0], a[1] + b[1]


def square_interval(a: Tuple[float, float]) -> Tuple[float, float]:
    """Square one scalar interval."""
    l, u = a
    lower_sq, upper_sq = l * l, u * u
    if l <= 0.0 <= u:
        return 0.0, max(lower_sq, upper_sq)
    return (min(lower_sq, upper_sq), max(lower_sq, upper_sq))


def lin_deriv(W: List[List[float]], d_lo: List[float], d_hi: List[float]) -> Tuple[List[float], List[float]]:
    """Propagate derivative interval bounds through a zero-bias linear layer."""
    # Like ibp_linear but with zero bias
    m, n = len(W), len(W[0])
    out_lo, out_hi = [], []
    for i in range(m):
        li, ui = 0.0, 0.0
        for j in range(n):
            a = W[i][j]
            p, q = a * d_lo[j], a * d_hi[j]
            li += min(p, q)
            ui += max(p, q)
        out_lo.append(li)
        out_hi.append(ui)
    return out_lo, out_hi


def tanh_prime_bounds(y_lo: float, y_hi: float) -> Tuple[float, float]:
    """Bound `d/dz tanh(z) = 1 - tanh(z)^2` from tanh-value bounds."""
    yl2, yh2 = y_lo * y_lo, y_hi * y_hi
    s_max = max(yl2, yh2)
    if y_lo <= 0.0 <= y_hi:
        s_min = 0.0
    else:
        s_min = min(yl2, yh2)
    return 1.0 - s_max, 1.0 - s_min


def tanh_second_bounds(y_lo: float, y_hi: float) -> Tuple[float, float]:
    """Bound the second tanh derivative expressed as a function of `y = tanh(z)`."""
    # f2(y) = -2y(1 - y^2) = -2y + 2y^3, extrema at y = ± 1/sqrt(3)
    def f2(y: float) -> float:
        """Evaluate the second-derivative polynomial in tanh-value space."""
        return -2.0 * y + 2.0 * (y * y * y)
    candidates = [f2(y_lo), f2(y_hi)]
    r = 1.0 / math.sqrt(3.0)
    for c in (-r, r):
        if y_lo < c < y_hi:
            candidates.append(f2(c))
    return min(candidates), max(candidates)


def deriv_and_second_for_mlp(
    first_weight: List[List[float]], first_bias: List[float],
    middle_weight: List[List[float]], middle_bias: List[float],
    output_weight: List[List[float]], output_bias: List[float],
    x_lo: List[float], x_hi: List[float]
) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
    """Propagate first- and second-derivative intervals through the tanh MLP."""
    # Value bounds through the network
    first_linear_lo, first_linear_hi = ibp_linear(first_weight, first_bias, x_lo, x_hi)
    first_activation_lo, first_activation_hi = ibp_tanh(first_linear_lo, first_linear_hi)
    middle_linear_lo, middle_linear_hi = ibp_linear(
        middle_weight,
        middle_bias,
        first_activation_lo,
        first_activation_hi,
    )
    middle_activation_lo, middle_activation_hi = ibp_tanh(middle_linear_lo, middle_linear_hi)
    _out_lo, _out_hi = ibp_linear(output_weight, output_bias, middle_activation_lo, middle_activation_hi)
    # First derivative intervals
    first_deriv_lo, first_deriv_hi = [1.0], [1.0]
    second_deriv_lo, second_deriv_hi = [0.0], [0.0]
    # Through first linear
    first_deriv_lo, first_deriv_hi = lin_deriv(first_weight, first_deriv_lo, first_deriv_hi)
    second_deriv_lo, second_deriv_hi = lin_deriv(first_weight, second_deriv_lo, second_deriv_hi)
    # Through tanh1
    first_tanh_deriv_lo: List[float] = []
    first_tanh_deriv_hi: List[float] = []
    first_tanh_second_lo: List[float] = []
    first_tanh_second_hi: List[float] = []
    for value_lo, value_hi, deriv_lo, deriv_hi, second_lo, second_hi in zip(
        first_activation_lo,
        first_activation_hi,
        first_deriv_lo,
        first_deriv_hi,
        second_deriv_lo,
        second_deriv_hi,
    ):
        tanh_prime_lo, tanh_prime_hi = tanh_prime_bounds(value_lo, value_hi)
        tanh_second_lo, tanh_second_hi = tanh_second_bounds(value_lo, value_hi)
        # first derivative
        deriv_out_lo, deriv_out_hi = mul_interval((tanh_prime_lo, tanh_prime_hi), (deriv_lo, deriv_hi))
        first_tanh_deriv_lo.append(deriv_out_lo); first_tanh_deriv_hi.append(deriv_out_hi)
        # second derivative: p2*(z')^2 + p1*z''
        deriv_sq = square_interval((deriv_lo, deriv_hi))
        curvature_term = mul_interval((tanh_second_lo, tanh_second_hi), deriv_sq)
        chain_term = mul_interval((tanh_prime_lo, tanh_prime_hi), (second_lo, second_hi))
        second_out_lo, second_out_hi = add_interval(curvature_term, chain_term)
        first_tanh_second_lo.append(second_out_lo); first_tanh_second_hi.append(second_out_hi)
    first_deriv_lo, first_deriv_hi = first_tanh_deriv_lo, first_tanh_deriv_hi
    second_deriv_lo, second_deriv_hi = first_tanh_second_lo, first_tanh_second_hi
    # Through middle linear
    first_deriv_lo, first_deriv_hi = lin_deriv(middle_weight, first_deriv_lo, first_deriv_hi)
    second_deriv_lo, second_deriv_hi = lin_deriv(middle_weight, second_deriv_lo, second_deriv_hi)
    # Through tanh2
    second_tanh_deriv_lo: List[float] = []
    second_tanh_deriv_hi: List[float] = []
    second_tanh_second_lo: List[float] = []
    second_tanh_second_hi: List[float] = []
    for value_lo, value_hi, deriv_lo, deriv_hi, second_lo, second_hi in zip(
        middle_activation_lo,
        middle_activation_hi,
        first_deriv_lo,
        first_deriv_hi,
        second_deriv_lo,
        second_deriv_hi,
    ):
        tanh_prime_lo, tanh_prime_hi = tanh_prime_bounds(value_lo, value_hi)
        tanh_second_lo, tanh_second_hi = tanh_second_bounds(value_lo, value_hi)
        deriv_out_lo, deriv_out_hi = mul_interval((tanh_prime_lo, tanh_prime_hi), (deriv_lo, deriv_hi))
        second_tanh_deriv_lo.append(deriv_out_lo); second_tanh_deriv_hi.append(deriv_out_hi)
        deriv_sq = square_interval((deriv_lo, deriv_hi))
        curvature_term = mul_interval((tanh_second_lo, tanh_second_hi), deriv_sq)
        chain_term = mul_interval((tanh_prime_lo, tanh_prime_hi), (second_lo, second_hi))
        second_out_lo, second_out_hi = add_interval(curvature_term, chain_term)
        second_tanh_second_lo.append(second_out_lo); second_tanh_second_hi.append(second_out_hi)
    first_deriv_lo, first_deriv_hi = second_tanh_deriv_lo, second_tanh_deriv_hi
    second_deriv_lo, second_deriv_hi = second_tanh_second_lo, second_tanh_second_hi
    # Final linear to scalar output
    first_deriv_lo, first_deriv_hi = lin_deriv(output_weight, first_deriv_lo, first_deriv_hi)
    second_deriv_lo, second_deriv_hi = lin_deriv(output_weight, second_deriv_lo, second_deriv_hi)
    return (first_deriv_lo, first_deriv_hi), (second_deriv_lo, second_deriv_hi)


def run_ibp() -> Dict[str, Any]:
    """Compute the bundled PINN certificate payload."""
    first_weight, first_bias, middle_weight, middle_bias, output_weight, output_bias = seed_weights()
    h = 1e-2
    eps = 0.01
    points = [0.25, 0.5, 0.75]
    resid_lo: List[float] = []
    resid_hi: List[float] = []
    resid_lo_deriv: List[float] = []
    resid_hi_deriv: List[float] = []
    u_bounds: List[Dict[str, List[float]]] = []

    for center in points:
        def box(x: float) -> Tuple[List[float], List[float]]:
            """Return the local interval around one scalar point."""
            return [x - eps], [x + eps]

        # Evaluate IBP through MLP at x-h, x, x+h
        res_u: List[Tuple[float, float]] = []
        for x in (center - h, center, center + h):
            x_lo, x_hi = box(x)
            hidden_lo, hidden_hi = ibp_linear(first_weight, first_bias, x_lo, x_hi)
            hidden_lo, hidden_hi = ibp_tanh(hidden_lo, hidden_hi)
            middle_lo, middle_hi = ibp_linear(middle_weight, middle_bias, hidden_lo, hidden_hi)
            middle_lo, middle_hi = ibp_tanh(middle_lo, middle_hi)
            y_lo, y_hi = ibp_linear(output_weight, output_bias, middle_lo, middle_hi)
            # output is scalar
            res_u.append((y_lo[0], y_hi[0]))

        # Finite-difference residual at the center point.
        lo_fd, hi_fd = fd_residual_bounds(res_u[0], res_u[1], res_u[2], h)
        resid_lo.append(lo_fd)
        resid_hi.append(hi_fd)

        # Derivative-based residual: compute u'' bounds at the center point.
        x_lo, x_hi = box(center)
        (_, _), (d2_lo, d2_hi) = deriv_and_second_for_mlp(
            first_weight,
            first_bias,
            middle_weight,
            middle_bias,
            output_weight,
            output_bias,
            x_lo,
            x_hi,
        )
        resid_lo_deriv.append(d2_lo[0])
        resid_hi_deriv.append(d2_hi[0])

        u_bounds.append({
            "x": center,
            "u_minus": {"lo": res_u[0][0], "hi": res_u[0][1]},
            "u": {"lo": res_u[1][0], "hi": res_u[1][1]},
            "u_plus": {"lo": res_u[2][0], "hi": res_u[2][1]},
        })

    cert = {
        "model": {
            "arch": [1, 16, 16, 1],
            "activations": ["tanh", "tanh"],
            "weights": {
                "W1": first_weight, "b1": first_bias,
                "Wm": middle_weight, "bm": middle_bias,
                "W2": output_weight, "b2": output_bias,
            }
        },
        "pinn": {
            "pde": "u''(x) = 0",
            "domain": [0.0, 1.0],
            "h": h,
            "eps": eps,
            "points": points,
        },
        "residual_bounds": {"lo": resid_lo, "hi": resid_hi},
        "residual_bounds_deriv": {"lo": resid_lo_deriv, "hi": resid_hi_deriv},
        "u_bounds": u_bounds,
    }
    return cert


def main():
    """Write the bundled PINN certificate JSON."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/PINN/pinn_cert.json"
    with open(out_path, "w") as f:
        json.dump(cert, f, indent=2)
    print(f"Wrote certificate to {out_path}")


if __name__ == "__main__":
    main()
