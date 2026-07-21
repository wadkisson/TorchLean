/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.CommonHelpers

/-!
# PCA (spec model)

Principal Component Analysis is represented as a linear projection onto learned components,
plus an explicit mean for centering.

The exact model operations are the transform and inverse transform. A separate reference helper
below constructs a one-component approximation with power iteration; its name records that
numerical limitation explicitly.

PyTorch / ecosystem analogies:

- scikit-learn: `sklearn.decomposition.PCA` (fit + transform)
- PyTorch: `torch.pca_lowrank` or `torch.linalg.svd` (common building blocks)

References (background, not required to read the code):

- Pearson (1901), "On Lines and Planes of Closest Fit to Systems of Points in Space".
  https://doi.org/10.1080/14786440109462720
- Hotelling (1933), "Analysis of a complex of statistical variables into principal components".
  https://doi.org/10.2307/2333955
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-- Parameters for PCA as a linear map plus centering.

We store:

- `components : outDim × inDim` (rows are principal directions),
- `mean : inDim` (for centering),
- `explained_variance : outDim` (eigenvalues for the selected components).

This matches the typical PCA API: you can `transform` to `outDim` coordinates and `inverse` back
to `inDim`.
-/
structure PCASpec (α : Type) (inDim outDim : Nat) where
  /-- components. -/
  components : Tensor α (.dim outDim (.dim inDim .scalar))  -- Principal components (outDim × inDim)
  /-- mean. -/
  mean : Tensor α (.dim inDim .scalar)                     -- Data mean for centering
  /-- explained variance. -/
  explained_variance : Tensor α (.dim outDim .scalar)      -- Selected covariance eigenvalues

/-- Forward pass: center and project: `y = components · (x - mean)`. -/
def pcaForwardSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar)) :
  Tensor α (.dim outDim .scalar) :=
  -- Center the data: x_centered = x - mean
  let centered := subSpec input m.mean
  -- Project onto principal components: y = components * x_centered
  matVecMulSpec m.components centered

/-- Inverse transform: reconstruct `x ≈ componentsᵀ · y + mean`. -/
def pcaInverseSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (reduced : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim inDim .scalar) :=
  -- Reconstruct: x_reconstructed = components^T * reduced + mean
  let reconstructed := vecMatMulSpec reduced m.components
  addSpec reconstructed m.mean

/-- VJP contribution for `components`: outer product `dL/dy ⊗ (x - mean)`. -/
def pcaComponentsDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim outDim (.dim inDim .scalar)) :=
  let centered := subSpec input m.mean
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      match grad_output, centered with
      | Tensor.dim g_vals, Tensor.dim x_vals =>
        match g_vals i, x_vals j with
        | Tensor.scalar g, Tensor.scalar x => Tensor.scalar (g * x)
    ))

/-- VJP contribution for `mean`: `dL/dmean = -componentsᵀ · dL/dy`. -/
def pcaMeanDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (grad_output : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim inDim .scalar) :=
  negSpec (vecMatMulSpec grad_output m.components)

/-- VJP contribution for `input`: `dL/dx = componentsᵀ · dL/dy`. -/
def pcaInputDerivSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (grad_output : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim inDim .scalar) :=
  vecMatMulSpec grad_output m.components

/-- Full backward pass returning `(dComponents, dMean, dInput)`. -/
def pcaBackwardSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α (.dim outDim .scalar)) :
  (Tensor α (.dim outDim (.dim inDim .scalar)) ×
   Tensor α (.dim inDim .scalar) ×
   Tensor α (.dim inDim .scalar)) :=
  let d_components := pcaComponentsDerivSpec m input grad_output
  let d_mean := pcaMeanDerivSpec m grad_output
  let d_input := pcaInputDerivSpec m grad_output
  (d_components, d_mean, d_input)

/-- Approximate the leading PCA component using the scaled covariance matrix and power iteration.

Algorithm:

1. compute the mean and center the data,
2. form the covariance matrix `C = (1/(n-1)) Xᵀ X`,
3. run 20 power-iteration steps from the all-ones vector,
4. orient the resulting vector deterministically so results are reproducible.

The output has exactly one component. This is an executable approximation, not a theorem that the
returned vector is the dominant eigenvector. Such a theorem would require spectral hypotheses and
an error analysis. Numerical libraries generally use SVD or a convergent eigensolver for fitting.
-/
def pcaFitLeadingComponentApproxSpec {nSamples inDim : Nat}
  (data : Tensor α (.dim nSamples (.dim inDim .scalar)))
  (hSamples : 1 < nSamples) (hDim : 0 < inDim) :
  PCASpec α inDim 1 :=
  -- Compute mean
  have inst : Shape.valid_axis_inst 0 (Shape.dim nSamples (Shape.dim inDim Shape.scalar)) := by
    apply Shape.validAxisInstZeroAlt
    intro h
    subst nSamples
    simp at hSamples
  let mean := reduceMeanAuto 0 inst data

  -- Center the data
  let centered_data := Tensor.dim (fun i => subSpec (get data i) mean)

  -- Compute covariance matrix: C = (1/(n-1)) * X^T * X
  -- Using n-1 for unbiased estimator (Bessel's correction)
  let covariance := matMulSpec (matrixTransposeSpec centered_data) centered_data
  let n_minus_1 := nSamples - 1
  let covariance_scaled := scaleSpec covariance (1 / (n_minus_1 : α))

  let (eigenvalue, eigenvector) := leadingEigenpairPowerIterationApproxSpec covariance_scaled
  let first := toScalar (get eigenvector ⟨0, hDim⟩)
  let sign : α := if first < 0 then -1 else 1
  let oriented := scaleSpec eigenvector sign

  {
    components := Tensor.dim (fun _ => oriented),
    mean := mean,
    explained_variance := Tensor.dim (fun _ => Tensor.scalar eigenvalue)
  }

/-- Apply a fitted PCA transform to a batch of samples. -/
def pcaTransformSpec {nSamples inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (data : Tensor α (.dim nSamples (.dim inDim .scalar))) :
  Tensor α (.dim nSamples (.dim outDim .scalar)) :=
  match data with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => pcaForwardSpec m (batch_fn i))

/-- Reconstruction error: `||x - inverse(transform(x))||_2^2` (sum of squared coordinates).

PyTorch analogy: `torch.sum((x - x_hat) ** 2)`.
-/
def pcaReconstructionErrorSpec {inDim outDim : Nat}
  (m : PCASpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar)) (h : inDim ≠ 0) :
  α :=
  let reduced := pcaForwardSpec m input
  let reconstructed := pcaInverseSpec m reduced
  let error := subSpec input reconstructed
  let squared_error := squareSpec error
  have inst : Shape.valid_axis_inst 0 (Shape.dim inDim Shape.scalar) := by
    apply Shape.validAxisInstZeroAlt h
  toScalar (reduceSumAuto 0 squared_error)

/-- Cumulative explained variance (prefix sums of `explained_variance`). -/
def pcaCumulativeExplainedVarianceSpec {α : Type} [Add α] [Zero α]
    {inDim outDim : Nat} (m : PCASpec α inDim outDim) :
    Tensor α (.dim outDim .scalar) :=
  match m.explained_variance with
  | Tensor.dim f =>
    Tensor.dim (fun i =>
      -- For each position i, sum explained variances from 0 to i
      let rec sum_to_index (j : Nat) (acc : α) : α :=
        if j > i.val then acc
        else
          if h : j < outDim then
            match f ⟨j, h⟩ with
            | Tensor.scalar x => sum_to_index (j + 1) (acc + x)
          else acc
      Tensor.scalar (sum_to_index 0 0)
    )


end Spec
