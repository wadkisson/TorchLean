/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Core

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor

namespace Graph

open Runtime
open Runtime.Autograd

/-!
## Runtime link: `compileAux` + `Tape.backwardDenseFrom`

`compileAux` produces a runtime tape whose node ids correspond to positions in the proof context
`Γ ++ ss`, and bakes the proved `vjp` into each node’s runtime `backward` closure.

The theorem `backwardDenseFrom_compileAux_eq_backpropAllCtx` states that executing the runtime
reverse-mode loop on this compiled tape matches the proved `backpropAllCtx`.
-/

/--
All nodes produced by `compileAuxData` have `requires_grad = true`.

This is a simplifying invariant: the compiled tape is meant for correctness proofs, so we mark
every node as eligible for gradient accumulation (including leaves for inputs).
-/
theorem compileAuxData_all_requires_grad_true {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    ((compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.all (fun n =>
      n.requires_grad)) = true := by
  -- Helper: if the current tape has `.all requires_grad = true`, `addLeaves` preserves it.
  have addLeaves_all :
      ∀ (t : Tape α),
        t.nodes.all (fun n => n.requires_grad) = true →
          ∀ {Γ : List Shape} (xs : TList α Γ),
            (addLeaves (α := α) (t := t) (Γ := Γ) xs).nodes.all (fun n => n.requires_grad) = true :=
              by
    intro t ht Γ xs
    induction xs generalizing t with
    | nil =>
        simpa [addLeaves] using ht
    | cons x xs ih =>
        -- push one leaf (which has `requires_grad = true`) and recurse
        let t' : Tape α := (Runtime.Autograd.Tape.leaf (t := t) x).1
        have ht' : t'.nodes.all (fun n => n.requires_grad) = true := by
          simpa [t', Runtime.Autograd.Tape.leaf, Runtime.Autograd.Tape.addNode, Array.all_push]
            using ht
        simpa [addLeaves, t', Runtime.Autograd.Tape.leaf, Runtime.Autograd.Tape.addNode] using ih (t
          := t') ht'

  induction g with
  | nil =>
      have h0 : (Runtime.Autograd.Tape.empty (α := α)).nodes.all (fun n => n.requires_grad) = true
        := by
        simp [Runtime.Autograd.Tape.empty]
      simpa [compileAuxData] using addLeaves_all (t := Runtime.Autograd.Tape.empty (α := α)) h0 (Γ
        := Γ) x
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAuxData, Runtime.Autograd.Tape.addNode, ih]

/--
Pointwise form of `compileAuxData_all_requires_grad_true`: every node index is `requires_grad =
  true`.

This is often more convenient than the `.all` formulation when reasoning about array indexing.
-/
theorem compileAuxData_requires_grad_true {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    let t := (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1
    ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requires_grad = true := by
  intro t i hi
  have hall :
      t.nodes.all (fun n => n.requires_grad) = true := by
    simpa [t] using compileAuxData_all_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x
      d
  have := (Array.all_eq_true).1 hall i hi
  simpa using this

/--
Backward closure safety for `compileAuxData`: parent ids produced by any node are strictly smaller
  than the node id.

This is the “edges point backwards” invariant required by the runtime reverse loop: when processing
node `id`, every contribution targets an earlier node (`pid < id`), so accumulation is well-founded.
-/
theorem compileAuxData_backward_pids_lt_id {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d0 : Δ) :
    ∀ id (n : Runtime.Autograd.Node α),
      (Runtime.Autograd.Tape.getNode? (t := (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g
        x d0).1) id = some n) →
      ∀ (d : Runtime.AnyTensor α) (contribs : List (Nat × Runtime.AnyTensor α)),
        n.backward d = .ok contribs →
          ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < id := by
  induction g with
  | nil =>
      intro id n hn d contribs hback pid pg hmem
      -- `compileAuxData nil` produces only leaves with `backward = ok []`.
      have hn' :
          ((TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α)))[id]? = some n := by
        simpa [compileAuxData, Runtime.Autograd.Tape.getNode?, nodes_addLeaves,
          Runtime.Autograd.Tape.empty] using hn
      cases hx : (TList.toAnyArray (α := α) (ss := Γ) x)[id]? with
      | none =>
          simp [Array.getElem?_map, hx] at hn'
      | some v =>
          have hnEq : n = leafNodeOfAny (α := α) v := by
            symm
            simpa [Array.getElem?_map, hx] using hn'
          subst hnEq
          have hcontribs : contribs = [] := by
            have := congrArg (fun r => match r with | .ok l => l | .error _ => []) hback
            simpa [leafNodeOfAny] using this
          subst hcontribs
          cases hmem
  | snoc g node ih =>
      rename_i ssPrev τ
      intro id n hn d contribs hback pid pg hmem
      let prev := compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let tPrev := prev.1
      let ctxPrev := prev.2
      let y := node.forward ctxPrev d0
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d0 dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      have hnNodes :
          (tPrev.nodes.push runtimeNode)[id]? = some n := by
        simpa [compileAuxData, prev, tPrev, ctxPrev, y, runtimeNode, Runtime.Autograd.Tape.getNode?,
          Runtime.Autograd.Tape.addNode] using hn
      by_cases hlast : id = tPrev.nodes.size
      · subst hlast
        have hnEq : n = runtimeNode := by
          symm
          simpa [Array.getElem?_push] using hnNodes
        subst hnEq
        have hd : d.s = τ := by
          by_contra hne
          have : runtimeNode.backward d = .error "autograd: upstream gradient shape mismatch" := by
            have : d.s ≠ τ := hne
            simp [runtimeNode, this]
          simp [this]  at hback
        have hcontribs :
            contribs =
              TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev)
                (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0 := by
          let listExpr :=
            TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev)
              (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0
          have hret : runtimeNode.backward d = .ok listExpr := by
            simp [runtimeNode, hd, listExpr]
          have hok :
              (.ok listExpr : Result (List (Nat × Runtime.AnyTensor α))) = .ok contribs := by
            calc
              (.ok listExpr : Result (List (Nat × Runtime.AnyTensor α))) = runtimeNode.backward d :=
                by
                simpa using hret.symm
              _ = .ok contribs := hback
          have := congrArg (fun r => match r with | .ok l => l | .error _ => []) hok
          simpa [listExpr] using this.symm
        subst hcontribs
        have hpidlt :=
          TList.mem_toIndexedAnyList_lt (α := α) (ss := Γ ++ ssPrev)
            (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0 (pid := pid) (pg := pg) hmem
        -- `0 + (Γ ++ ssPrev).length = tPrev.nodes.size`
        have htPrev :
            tPrev.nodes.size = (Γ ++ ssPrev).length := by
          -- by the size lemma for compiled GraphData prefix
          simpa [prev] using
            compileAuxData_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
        simpa [htPrev] using hpidlt
      · have hidPrev : id < tPrev.nodes.size := by
          have hidPush : id < (tPrev.nodes.push runtimeNode).size := by
            rcases Array.getElem_of_getElem? hnNodes with ⟨hid, _⟩
            exact hid
          have hidLe : id ≤ tPrev.nodes.size := by
            have : id < tPrev.nodes.size + 1 := by
              simpa [Array.size_push] using hidPush
            exact Nat.le_of_lt_succ this
          exact Nat.lt_of_le_of_ne hidLe hlast
        have hnPrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some n := by
          have : tPrev.nodes[id]? = some n := by
            simpa [Array.getElem?_push, hlast] using hnNodes
          simpa [Runtime.Autograd.Tape.getNode?, tPrev] using this
        exact ih id n (by simpa [prev, tPrev] using hnPrev) d contribs hback hmem

/--
All nodes produced by `compileAux` have `requires_grad = true`.

This mirrors `compileAuxData_all_requires_grad_true` for the `Graph` interface.
-/
theorem compileAux_all_requires_grad_true {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d0 : Δ) :
    ((compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1.nodes.all (fun n =>
      n.requires_grad)) = true := by
  -- Helper: if the current tape has `.all requires_grad = true`, `addLeaves` preserves it.
  have addLeaves_all :
      ∀ (t : Tape α),
        t.nodes.all (fun n => n.requires_grad) = true →
          ∀ {Γ : List Shape} (xs : TList α Γ),
            (addLeaves (α := α) (t := t) (Γ := Γ) xs).nodes.all (fun n => n.requires_grad) = true :=
              by
    intro t ht Γ xs
    induction xs generalizing t with
    | nil =>
        simpa [addLeaves] using ht
    | cons x xs ih =>
        -- push one leaf (which has `requires_grad = true`) and recurse
        let t' : Tape α := (Runtime.Autograd.Tape.leaf (t := t) x).1
        have ht' : t'.nodes.all (fun n => n.requires_grad) = true := by
          -- `leaf` pushes a node with `requires_grad = true`, so `.all` is preserved
          simpa [t', Runtime.Autograd.Tape.leaf, Runtime.Autograd.Tape.addNode, Array.all_push]
            using ht
        simpa [addLeaves, t', Runtime.Autograd.Tape.leaf, Runtime.Autograd.Tape.addNode] using ih (t
          := t') ht'

  induction g with
  | nil =>
      -- Start from the empty tape where `.all _ = true`.
      have h0 : (Runtime.Autograd.Tape.empty (α := α)).nodes.all (fun n => n.requires_grad) = true
        := by
        simp [Runtime.Autograd.Tape.empty]
      simpa [compileAux] using addLeaves_all (t := Runtime.Autograd.Tape.empty (α := α)) h0 (Γ := Γ)
        x
  | snoc g node ih =>
      rename_i ssPrev τ
      -- `compileAux` appends a node with `requires_grad = true`.
      simp [compileAux, Runtime.Autograd.Tape.addNode, ih]

/-- Pointwise form of `compileAux_all_requires_grad_true`. -/
theorem compileAux_requires_grad_true {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d0 : Δ) :
    let t := (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1
    ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requires_grad = true := by
  intro t i hi
  have hall :
      t.nodes.all (fun n => n.requires_grad) = true := by
    simpa [t] using compileAux_all_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0
  -- `Array.all_eq_true` gives the pointwise result.
  have := (Array.all_eq_true).1 hall i hi
  simpa using this

/--
Backward closure safety for `compileAux`: parent ids produced by any node are strictly smaller than
  the node id.

This mirrors `compileAuxData_backward_pids_lt_id` for the `Graph` interface.
-/
theorem compileAux_backward_pids_lt_id {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d0 : Δ) :
    ∀ id (n : Runtime.Autograd.Node α),
      (Runtime.Autograd.Tape.getNode? (t := (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x
        d0).1) id = some n) →
      ∀ (d : Runtime.AnyTensor α) (contribs : List (Nat × Runtime.AnyTensor α)),
        n.backward d = .ok contribs →
          ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < id := by
  induction g with
  | nil =>
      intro id n hn d contribs hback pid pg hmem
      -- `compileAux nil` produces only leaves with `backward = ok []`.
      have hn' :
          ((TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α)))[id]? = some n := by
        simpa [compileAux, Runtime.Autograd.Tape.getNode?, nodes_addLeaves,
          Runtime.Autograd.Tape.empty] using hn
      cases hx : (TList.toAnyArray (α := α) (ss := Γ) x)[id]? with
      | none =>
          simp [Array.getElem?_map, hx] at hn'
      | some v =>
          have hnEq : n = leafNodeOfAny (α := α) v := by
            -- `getElem?_map` turns this into `some (leafNodeOfAny v) = some n`.
            symm
            simpa [Array.getElem?_map, hx] using hn'
          subst hnEq
          -- `leafNodeOfAny.backward = ok []`, so `contribs = []`.
          have hcontribs : contribs = [] := by
            have := congrArg (fun r => match r with | .ok l => l | .error _ => []) hback
            simpa [leafNodeOfAny] using this
          subst hcontribs
          cases hmem
  | snoc g node ih =>
      rename_i ssPrev τ
      intro id n hn d contribs hback pid pg hmem
      let prev := compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let tPrev := prev.1
      let ctxPrev := prev.2
      let y := node.forward ctxPrev d0
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d0 dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      have hnNodes :
          (tPrev.nodes.push runtimeNode)[id]? = some n := by
        simpa [compileAux, prev, tPrev, ctxPrev, y, runtimeNode, Runtime.Autograd.Tape.getNode?,
          Runtime.Autograd.Tape.addNode] using hn
      by_cases hlast : id = tPrev.nodes.size
      · subst hlast
        have hnEq : n = runtimeNode := by
          -- `getElem?_push` at `size` yields `some runtimeNode`.
          symm
          simpa [Array.getElem?_push] using hnNodes
        subst hnEq
        have hd : d.s = τ := by
          by_contra hne
          have : runtimeNode.backward d = .error "autograd: upstream gradient shape mismatch" := by
            have : d.s ≠ τ := hne
            simp [runtimeNode, this]
          simp [this]  at hback
        have hcontribs :
            contribs =
              TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev)
                (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0 := by
          let listExpr :=
            TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev)
              (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0
          have hret : runtimeNode.backward d = .ok listExpr := by
            simp [runtimeNode, hd, listExpr]
          have hok :
              (.ok listExpr : Result (List (Nat × Runtime.AnyTensor α))) = .ok contribs := by
            calc
              (.ok listExpr : Result (List (Nat × Runtime.AnyTensor α))) = runtimeNode.backward d :=
                by
                simpa using hret.symm
              _ = .ok contribs := hback
          have := congrArg (fun r => match r with | .ok l => l | .error _ => []) hok
          simpa [listExpr] using this.symm
        subst hcontribs
        have hpidlt :=
          TList.mem_toIndexedAnyList_lt (α := α) (ss := Γ ++ ssPrev)
            (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0 hmem
        have hlen : (Γ ++ ssPrev).length = tPrev.nodes.size := by
          have : tPrev.nodes.size = Γ.length + ssPrev.length := by
            simpa [tPrev, prev] using
              (compileAux_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
          simp [List.length_append, this]
        simpa [Nat.zero_add, hlen] using hpidlt
      · have hnPrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some n := by
          have : tPrev.nodes[id]? = some n := by
            simpa [Array.getElem?_push, hlast] using hnNodes
          simpa [Runtime.Autograd.Tape.getNode?, tPrev] using this
        exact ih id n (by simpa [prev, tPrev] using hnPrev) d contribs hback hmem

end Graph

end Algebra
end Autograd
end Proofs
