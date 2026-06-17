/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Group.MinMax
public import Mathlib.Analysis.Calculus.MeanValue
public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Data.List.FinRange
public import Mathlib.Analysis.Real.Sqrt
public import NN.Floats.NeuralFloat.NF
public import NN.Proofs.Gradients.Activation
public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox
public import NN.Proofs.RuntimeApprox.Rounding.RoundingApprox
public import NN.Proofs.Utils.List
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Layers.Activation


/-!
# NF Tensor Approximation Plumbing

Shape-generic lemmas for `approxT` and `linf_norm`.  The later NF proof files use these lemmas to
turn scalar absolute-error facts into tensor-level forward-error bounds.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

/-! ## Tensor Approximation Plumbing -/

/-- `linf_norm` is always nonnegative. -/
lemma linf_norm_nonneg : ∀ {s : Shape} (t : SpecTensor s), 0 ≤ linfNorm t := by
  intro s t
  induction s with
  | scalar =>
      cases t with
      | scalar x =>
          -- `tensor_linf_norm` uses `MathFunctions.abs`; for `ℝ` this is definitional `|·|`.
          dsimp [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm]
          -- Reduce to the usual `abs_nonneg`.
          have : (MathFunctions.abs x : ℝ) = |x| := by rfl
          rw [this]
          exact abs_nonneg x
  | dim n s ih =>
      cases t with
      | dim f =>
          have : (0 : ℝ) ≤
              (List.finRange n).foldl (fun acc i => max acc (tensorLinfNorm (α := ℝ) (f i))) 0 :=
                by
            simpa using (List.le_foldl_max_init (List.finRange n) (fun i => tensorLinfNorm (α :=
              ℝ) (f i)) 0)
          simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using this

/--
Componentwise bound for `linf_norm` on a dimensioned tensor.

The norm of any component `t[i]` is bounded by the norm of the whole tensor.
-/
lemma linf_norm_le_get_dim {n : Nat} {s : Shape} (t : SpecTensor (.dim n s)) (i : Fin n) :
    linfNorm (match t with | Tensor.dim f => f i) ≤ linfNorm t := by
  cases t with
  | dim f =>
      have hi : i ∈ List.finRange n := List.mem_finRange i
      have hle :=
        List.le_foldl_max_of_mem (List.finRange n) (fun j => linfNorm (f j)) (acc := (0 : ℝ)) hi
      simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using hle

/--
Scalar characterization of `approxT` on scalar tensors.

This rewrites `approxT (Tensor.scalar x) (Tensor.scalar xR) eps` into the usual absolute-error
inequality `|toSpec xR - x| ≤ eps`.
-/
lemma approxT_scalar_iff {α : Type} {toSpec : α → SpecScalar} {x : SpecScalar} {xR : α} {eps :
  SpecScalar} :
    approxT (α := α) (toSpec := toSpec) (Tensor.scalar x) (Tensor.scalar xR) eps ↔
      abs (toSpec xR - x) ≤ eps := by
  -- `tensor_distance linf_norm` is `|x - toSpec xR|` on scalar tensors.
  constructor
  · intro h
    have h' : abs (x - toSpec xR) ≤ eps := by
      simpa [approxT, approxWith, tensorToSpec, linfNorm, RuntimeApprox.linfNorm,
        tensorDistance, NN.MLTheory.Robustness.Spec.tensor_distance_tensor_sub_eq_sub_spec,
        tensorLinfNorm, Spec.mapTensor, Spec.Tensor.subSpec, map2Spec, MathFunctions.abs] using h
    simpa [abs_sub_comm] using h'
  · intro h
    have h' : abs (x - toSpec xR) ≤ eps := by
      simpa [abs_sub_comm] using h
    simpa [approxT, approxWith, tensorToSpec, linfNorm, RuntimeApprox.linfNorm,
      tensorDistance, NN.MLTheory.Robustness.Spec.tensor_distance_tensor_sub_eq_sub_spec,
      tensorLinfNorm, Spec.mapTensor, Spec.Tensor.subSpec, map2Spec, MathFunctions.abs] using h'

/--
Projection lemma for `approxT` on dimensioned tensors.

If `xS` approximates `xR` within `eps`, then each component `xS[i]` approximates `xR[i]` within
  `eps`.
-/
lemma approxT_dim_get {α : Type} {toSpec : α → SpecScalar} {n : Nat} {s : Shape}
    {xS : SpecTensor (.dim n s)} {xR : Tensor α (.dim n s)} {eps : SpecScalar}
    (h : approxT (α := α) (toSpec := toSpec) xS xR eps) (i : Fin n) :
    approxT (α := α) (toSpec := toSpec)
      (match xS with | Tensor.dim f => f i)
      (match xR with | Tensor.dim f => f i)
      eps := by
  cases xS with
  | dim xSf =>
      cases xR with
      | dim xRf =>
          -- Unfold `approxT` at dimension shape: it is a `foldl max` over component distances.
          have hi : i ∈ List.finRange n := List.mem_finRange i
          have hComp :
              tensorDistance (α := SpecScalar) linfNorm (xSf i)
                  (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))
                ≤ tensorDistance (α := SpecScalar) linfNorm (Tensor.dim xSf)
                  (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf)) := by
            -- Component distance is bounded by the max fold used in `linf_norm`.
            -- We unfold the RHS into the `foldl max` and use `le_foldl_max_of_mem`.
              have hle :=
                List.le_foldl_max_of_mem (ι := Fin n) (β := ℝ) (List.finRange n)
                    (fun j =>
                      tensorDistance (α := SpecScalar) linfNorm (xSf j)
                        (tensorToSpec (α := α) (toSpec := toSpec) (xRf j)))
                  (acc := (0 : ℝ)) hi
              -- Now rewrite the unfolded RHS back to `tensor_distance`.
              change
                tensorDistance (α := SpecScalar) linfNorm (xSf i)
                    (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))
                  ≤
                List.foldl
                  (fun a j =>
                    max a
                      (tensorDistance (α := SpecScalar) linfNorm (xSf j)
                        (tensorToSpec (α := α) (toSpec := toSpec) (xRf j))))
                  0 (List.finRange n)
              exact hle
          have := le_trans hComp h
          simpa [approxT, approxWith] using this

-- ---------------------------------------------------------------------------
-- Generic lifting lemmas for elementwise ops (`map_spec`, `map2_spec`)
-- ---------------------------------------------------------------------------

/--
Lift a scalar approximation bound to an elementwise `map_spec`.

Given a scalar bound of the form
`|toSpec (fR xR) - fS x| ≤ bnd (toSpec xR) eps`
and an input approximation `approxT xS xR eps`, this produces an approximation bound for
`map_spec fS xS` vs `map_spec fR xR`, with an output epsilon computed by taking the `linf_norm` of
the pointwise bound.
-/
theorem approxT_map_spec_of_scalar_bound {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    (fS : SpecScalar → SpecScalar) (fR : α → α) (bnd : SpecScalar → SpecScalar → SpecScalar) :
    ∀ {xS : SpecTensor s} {xR : Tensor α s} {eps : SpecScalar},
      approxT (α := α) (toSpec := toSpec) xS xR eps →
        (∀ {x : SpecScalar} {xR : α},
          abs (toSpec xR - x) ≤ eps →
            abs (toSpec (fR xR) - fS x) ≤ bnd (toSpec xR) eps) →
        approxT (α := α) (toSpec := toSpec)
          (mapSpec (s := s) fS xS)
          (mapSpec (s := s) fR xR)
          (linfNorm
            (mapSpec (s := s) (fun a => bnd a eps) (tensorToSpec (α := α) (toSpec := toSpec)
              xR))) := by
  intro xS xR eps hx hscalar
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              have hx' :=
                (approxT_scalar_iff (α := α) (toSpec := toSpec)
                  (x := x) (xR := xR) (eps := eps)).1 hx
              have herr :
                  abs (toSpec (fR xR) - fS x) ≤ bnd (toSpec xR) eps :=
                hscalar hx'
              have herr' : abs (toSpec (fR xR) - fS x) ≤ abs (bnd (toSpec xR) eps) :=
                le_trans herr (le_abs_self _)
              change approxT (α := α) (toSpec := toSpec)
                (Tensor.scalar (fS x)) (Tensor.scalar (fR xR))
                (linfNorm
                  (mapSpec (s := Shape.scalar) (fun a => bnd a eps)
                    (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar xR))))
              exact
                (approxT_scalar_iff (α := α) (toSpec := toSpec)
                  (x := fS x) (xR := fR xR)
                  (eps := linfNorm
                    (mapSpec (s := Shape.scalar) (fun a => bnd a eps)
                      (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar xR))))).2 (by
                        simpa [tensorToSpec, Spec.mapTensor, mapSpec,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
                          MathFunctions.abs] using herr')
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm
                  (mapSpec (s := Shape.dim n s) (fun a => bnd a eps)
                    (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf)))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using (linf_norm_nonneg
                  (t :=
                    mapSpec (s := Shape.dim n s) (fun a => bnd a eps)
                      (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf))))
              have hcomp :
                  ∀ i : Fin n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) fS (xSf i))
                        (tensorToSpec (α := α) (toSpec := toSpec) (mapSpec (s := s) fR (xRf i)))
                      ≤ B := by
                intro i
                have hx_i :=
                  approxT_dim_get (α := α) (toSpec := toSpec)
                    (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm
                        (mapSpec (s := s) (fun a => bnd a eps)
                          (tensorToSpec (α := α) (toSpec := toSpec) (xRf i)))
                      ≤ B := by
                  simpa [B, tensorToSpec, Spec.mapTensor, mapSpec] using
                    (linf_norm_le_get_dim
                      (t :=
                        mapSpec (s := Shape.dim n s) (fun a => bnd a eps)
                          (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf)))
                      i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) fS (xSf i))
                        (tensorToSpec (α := α) (toSpec := toSpec) (mapSpec (s := s) fR (xRf i)))
                      ≤
                      linfNorm
                        (mapSpec (s := s) (fun a => bnd a eps)
                          (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))) := by
                  simpa [approxT, approxWith] using hih
                exact le_trans hdist hB_ge
              have hf :
                  ∀ i ∈ List.finRange n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) fS (xSf i))
                        (tensorToSpec (α := α) (toSpec := toSpec) (mapSpec (s := s) fR (xRf i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := s) fS (xSf i))
                      (tensorToSpec (α := α) (toSpec := toSpec) (mapSpec (s := s) fR (xRf i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have :
                  tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := Shape.dim n s) fS (Tensor.dim xSf))
                      (tensorToSpec (α := α) (toSpec := toSpec)
                        (mapSpec (s := Shape.dim n s) fR (Tensor.dim xRf)))
                    ≤ B := by
                  change
                    List.foldl
                      (fun a i =>
                        max a
                          (tensorDistance (α := SpecScalar) linfNorm
                            (mapSpec (s := s) fS (xSf i))
                            (tensorToSpec (α := α) (toSpec := toSpec)
                              (mapSpec (s := s) fR (xRf i)))))
                      0 (List.finRange n) ≤ B
                  exact hfold
              simpa [approxT, approxWith, B] using this

/--
Lift a scalar approximation bound to an elementwise `map2_spec`.

This is the binary analogue of `approxT_map_spec_of_scalar_bound`, used for elementwise arithmetic
(`add`, `sub`, `mul_elem`, etc.).
-/
theorem approxT_map2_spec_of_scalar_bound {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    (fS : SpecScalar → SpecScalar → SpecScalar) (fR : α → α → α)
    (bnd : SpecScalar → SpecScalar → SpecScalar → SpecScalar → SpecScalar) :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor α s} {epsx epsy : SpecScalar},
      approxT (α := α) (toSpec := toSpec) xS xR epsx →
      approxT (α := α) (toSpec := toSpec) yS yR epsy →
        (∀ {x y : SpecScalar} {xR yR : α},
          abs (toSpec xR - x) ≤ epsx →
          abs (toSpec yR - y) ≤ epsy →
            abs (toSpec (fR xR yR) - fS x y) ≤ bnd (toSpec xR) (toSpec yR) epsx epsy) →
        approxT (α := α) (toSpec := toSpec)
          (map2Spec fS xS yS)
          (map2Spec fR xR yR)
          (linfNorm
            (map2Spec (fun a b => bnd a b epsx epsy)
              (tensorToSpec (α := α) (toSpec := toSpec) xR)
              (tensorToSpec (α := α) (toSpec := toSpec) yR))) := by
  intro xS yS xR yR epsx epsy hx hy hscalar
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases yS with
          | scalar y =>
              cases xR with
              | scalar xR =>
                  cases yR with
                    | scalar yR =>
                        have hx' :=
                          (approxT_scalar_iff (α := α) (toSpec := toSpec)
                            (x := x) (xR := xR) (eps := epsx)).1 hx
                        have hy' :=
                          (approxT_scalar_iff (α := α) (toSpec := toSpec)
                            (x := y) (xR := yR) (eps := epsy)).1 hy
                        have herr :
                            abs (toSpec (fR xR yR) - fS x y) ≤ bnd (toSpec xR) (toSpec yR) epsx epsy
                              :=
                          hscalar hx' hy'
                        have herr' :
                            abs (toSpec (fR xR yR) - fS x y) ≤ abs (bnd (toSpec xR) (toSpec yR) epsx
                              epsy) :=
                          le_trans herr (le_abs_self _)
                        change approxT (α := α) (toSpec := toSpec)
                          (Tensor.scalar (fS x y)) (Tensor.scalar (fR xR yR))
                          (linfNorm
                            (map2Spec (fun a b => bnd a b epsx epsy)
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar xR))
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar yR))))
                        exact
                          (approxT_scalar_iff (α := α) (toSpec := toSpec)
                            (x := fS x y) (xR := fR xR yR)
                          (eps := linfNorm
                            (map2Spec (fun a b => bnd a b epsx epsy)
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar xR))
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar yR))))).2
                                (by
                                simpa [tensorToSpec, Spec.mapTensor, map2Spec,
                                  linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
                                  MathFunctions.abs] using herr')
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases yS with
          | dim ySf =>
              cases xR with
              | dim xRf =>
                  cases yR with
                  | dim yRf =>
                      let B : ℝ :=
                        linfNorm
                          (map2Spec (fun a b => bnd a b epsx epsy)
                            (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf))
                            (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim yRf)))
                      have hB_nonneg : 0 ≤ B := by
                        simpa [B] using (linf_norm_nonneg
                          (t :=
                            map2Spec (fun a b => bnd a b epsx epsy)
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf))
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim yRf))))
                      have hcomp :
                          ∀ i : Fin n,
                            tensorDistance (α := SpecScalar) linfNorm
                                (map2Spec fS (xSf i) (ySf i))
                                (tensorToSpec (α := α) (toSpec := toSpec) (map2Spec fR (xRf i)
                                  (yRf i)))
                              ≤ B := by
                        intro i
                        have hx_i :=
                          approxT_dim_get (α := α) (toSpec := toSpec)
                            (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := epsx) hx i
                        have hy_i :=
                          approxT_dim_get (α := α) (toSpec := toSpec)
                            (xS := Tensor.dim ySf) (xR := Tensor.dim yRf) (eps := epsy) hy i
                        have hih := ih (xS := xSf i) (yS := ySf i) (xR := xRf i) (yR := yRf i) hx_i
                          hy_i
                        have hB_ge :
                            linfNorm
                                (map2Spec (fun a b => bnd a b epsx epsy)
                                  (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))
                                  (tensorToSpec (α := α) (toSpec := toSpec) (yRf i)))
                              ≤ B := by
                          simpa [B, tensorToSpec, Spec.mapTensor, map2Spec] using
                            (linf_norm_le_get_dim
                              (t :=
                                map2Spec (fun a b => bnd a b epsx epsy)
                                  (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf))
                                  (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim yRf)))
                              i)
                        have hdist :
                            tensorDistance (α := SpecScalar) linfNorm
                                (map2Spec fS (xSf i) (ySf i))
                                (tensorToSpec (α := α) (toSpec := toSpec) (map2Spec fR (xRf i)
                                  (yRf i)))
                              ≤
                              linfNorm
                                (map2Spec (fun a b => bnd a b epsx epsy)
                                  (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))
                                  (tensorToSpec (α := α) (toSpec := toSpec) (yRf i))) := by
                          simpa [approxT, approxWith] using hih
                        exact le_trans hdist hB_ge

                      have hf :
                          ∀ i ∈ List.finRange n,
                            tensorDistance (α := SpecScalar) linfNorm
                                (map2Spec fS (xSf i) (ySf i))
                                (tensorToSpec (α := α) (toSpec := toSpec) (map2Spec fR (xRf i)
                                  (yRf i)))
                              ≤ B := by
                          intro i _hi
                          exact hcomp i
                      have hfold :=
                        List.foldl_max_le_of_le (List.finRange n)
                          (fun i =>
                            tensorDistance (α := SpecScalar) linfNorm
                              (map2Spec fS (xSf i) (ySf i))
                              (tensorToSpec (α := α) (toSpec := toSpec) (map2Spec fR (xRf i) (yRf
                                i))))
                          (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
                      have :
                          tensorDistance (α := SpecScalar) linfNorm
                              (map2Spec fS (Tensor.dim xSf) (Tensor.dim ySf))
                              (tensorToSpec (α := α) (toSpec := toSpec)
                                (map2Spec fR (Tensor.dim xRf) (Tensor.dim yRf)))
                            ≤ B := by
                          change
                            List.foldl
                              (fun a i =>
                                max a
                                  (tensorDistance (α := SpecScalar) linfNorm
                                    (map2Spec fS (xSf i) (ySf i))
                                    (tensorToSpec (α := α) (toSpec := toSpec)
                                      (map2Spec fR (xRf i) (yRf i)))))
                              0 (List.finRange n) ≤ B
                          exact hfold
                      simpa [approxT, approxWith, B] using this

end

end RuntimeApprox
end Proofs
