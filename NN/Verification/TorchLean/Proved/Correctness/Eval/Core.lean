/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.WellFormed

/-!
# Compiled Forward Evaluation: Shared Invariants
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

open NN.Verification.TorchLean

/-! ### Compiler correctness (forward fragment) -/

def evalFGraphVals
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (vals : Array (DVal α)) : Except String (Array (DVal α)) := do
  match g with
  | .ret _y =>
      pure vals
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext =>
      let vOut ←
        evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := mid)
          node params vals
      evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss ++ [mid])
        (out := out)
        gNext params (vals.push vOut)

  /--
  Compiling a let-chain does not change `ps.constVals` entries for keys `< c.graph.nodes.size`.
  Compilation only inserts payload at the fresh node id, so older keys are unchanged.
  -/
  theorem compileFGraph_ps_constVals_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.constVals.get? k = c.ps.constVals.get? k := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [compileFGraph]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      -- One compilation step pushes a fresh node at `id = c.graph.nodes.size` and only inserts
      -- payload at that id.
      let id := c.graph.nodes.size
      have hk' : k < id := by simpa [id] using hk
      have hk_succ : k < id + 1 := Nat.lt_succ_of_lt hk'
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hps' : ps'.constVals.get? k = c.ps.constVals.get? k := by
        -- `compileNode` only inserts into `constVals` at key `id`, and `k < id`.
        have hidk : id ≠ k := (ne_comm).1 hk'.ne
        cases node <;>
          simp [compileNode, res, ps', Std.HashMap.getElem?_insert,
            beq_eq_false_iff_ne.mpr hidk]
      -- Apply IH to the suffix compilation: keys < `c'.graph.nodes.size` are preserved.
      have hIH :=
        ih (c := c') (hk := by simpa [c', Array.size_push, id] using hk_succ)
      have : (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
        [mid₀]) (out := out₀)
          gNext params c').ps.constVals.get? k = c.ps.constVals.get? k := by
        calc
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
            [mid₀]) (out := out₀)
              gNext params c').ps.constVals.get? k
              =
            c'.ps.constVals.get? k := hIH
          _ = c.ps.constVals.get? k := by simpa [c'] using hps'
      simpa [compileFGraph, c', id, res] using this

  /--
  Compiling a let-chain does not change `ps.linearWB` entries for keys `< c.graph.nodes.size`.
  Compilation only inserts linear payload at the fresh node id, so older keys are unchanged.
  -/
  theorem compileFGraph_ps_linearWB_get?_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
      out)
        g params c).ps.linearWB.get? k = c.ps.linearWB.get? k := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [compileFGraph]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      have hk' : k < id := by simpa [id] using hk
      have hk_succ : k < id + 1 := Nat.lt_succ_of_lt hk'
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hps' : ps'.linearWB.get? k = c.ps.linearWB.get? k := by
        -- `compileNode` only inserts into `linearWB` at key `id`, and `k < id`.
        have hidk : id ≠ k := (ne_comm).1 hk'.ne
        cases node <;>
          simp [compileNode, res, ps', Std.HashMap.getElem?_insert,
            beq_eq_false_iff_ne.mpr hidk]
      have hIH :=
        ih (c := c') (hk := by simpa [c', Array.size_push, id] using hk_succ)
      have : (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
        [mid₀]) (out := out₀)
          gNext params c').ps.linearWB.get? k = c.ps.linearWB.get? k := by
        calc
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
            [mid₀]) (out := out₀)
              gNext params c').ps.linearWB.get? k
              =
            c'.ps.linearWB.get? k := hIH
          _ = c.ps.linearWB.get? k := by simpa [c'] using hps'
      simpa [compileFGraph, c', id, res] using this

  /--
  `compileFGraph` does not change existing nodes at indices `< c.graph.nodes.size`.
  Compilation only appends nodes, so `getNode` agrees on the prefix.
  -/
  theorem compileFGraph_getNode_lt
      {α : Type} [Context α]
      {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
      (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    {i : Nat} (hi : i < c.graph.nodes.size) :
    (NN.IR.Graph.getNode
        (g := (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := out)
          g params c).graph) i)
      =
    NN.IR.Graph.getNode (g := c.graph) i := by
  classical
  induction g generalizing c with
  | ret y =>
      simp [compileFGraph]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have hi' : i < c'.graph.nodes.size := by
        simpa [c', Array.size_push] using Nat.lt_succ_of_lt hi
      have hNext :
          NN.IR.Graph.getNode
              (g := (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
                ss₀ ++ [mid₀]) (out := out₀)
                gNext params c').graph) i
            =
          NN.IR.Graph.getNode (g := c'.graph) i :=
        ih (c := c') (hi := hi')
      have hPush :
          NN.IR.Graph.getNode (g := c'.graph) i = NN.IR.Graph.getNode (g := c.graph) i := by
        simpa [c', res, id] using getNode_push_lt (g := c.graph) (n := n) (hi := hi)
      simpa [compileFGraph, c', id, res] using Eq.trans hNext hPush

    /-- `compileFGraph` is monotone in `graph.nodes.size` (it only appends nodes). -/
    theorem compileFGraph_nodesSize_le
        {α : Type} [Context α]
        {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
        (g : FGraph α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (c : NN.Verification.TorchLean.CompiledIR α) :
      c.graph.nodes.size ≤
        (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
          out)
            g params c).graph.nodes.size := by
    classical
    induction g generalizing c with
    | ret y =>
        simp [compileFGraph]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      have h1 : c.graph.nodes.size ≤ c'.graph.nodes.size := by
        simp [c', Array.size_push]
      have h2 : c'.graph.nodes.size ≤
          (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
            [mid₀]) (out := out₀)
            gNext params c').graph.nodes.size := by
        exact ih (c := c')
      simpa [compileFGraph, c', id, res] using Nat.le_trans h1 h2

    /-- Shape lookup through `shapesOfVals` agrees with looking up the dynamic value first. -/
    lemma shapesOfVals_get?_eq
        {α : Type} [Context α] (vals : Array (DVal α)) (i : Nat) :
        (shapesOfVals (α := α) vals)[i]? = (vals[i]?).map (fun v => v.1) := by
    -- Avoid `simp` loops on `Array.getElem?_eq_toList_get?'`.
    have hToList : vals.toList[i]? = vals[i]? := by
      simp
    -- `List.getElem?_map` reduces the `map` and then we rewrite the list lookup to the array
    -- lookup.
    simp [shapesOfVals, List.getElem?_map, hToList]

  @[simp] lemma shapesOfVals_length {α : Type} [Context α] (vals : Array (DVal α)) :
      (shapesOfVals (α := α) vals).length = vals.size := by
    simp [shapesOfVals]

  @[simp] theorem shape_of_vals_of_hShapes
      {α : Type} [Context α]
      {inShape : Shape} {ss : List Shape} {s : Shape}
      (vals : Array (DVal α)) (idx : Idx (Ctx inShape ss) s)
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      (vals[idx.id]!).shape = s := by
    classical
    have hLen : vals.size = (Ctx inShape ss).length := by
      simpa [shapesOfVals_length] using congrArg List.length hShapes
    have hiΓ : idx.id < (Ctx inShape ss).length := idx_id_lt_length (x := idx)
    have hiVals : idx.id < vals.size := by simpa [hLen] using hiΓ
    have hFin : (⟨idx.id, hiΓ⟩ : Fin (Ctx inShape ss).length) = idx.i := by
      apply Fin.ext
      rfl
    have hGetElem : (Ctx inShape ss)[idx.id]'hiΓ = s := by
      -- `l[i]'h` is definitional `l.get ⟨i,h⟩`.
      simpa [Idx.id, List.get, hFin] using idx.h
    have hΓOpt : (Ctx inShape ss)[idx.id]? = some s := by
      have hSome : (Ctx inShape ss)[idx.id]? = some ((Ctx inShape ss)[idx.id]'hiΓ) := by
        simp
      simp [hSome, hGetElem]
    have hShapesAt :
        (shapesOfVals (α := α) vals)[idx.id]? = some s := by
      have hEq : (shapesOfVals (α := α) vals)[idx.id]? = (Ctx inShape ss)[idx.id]? :=
        congrArg (fun l => l[idx.id]?) hShapes
      exact Eq.trans hEq hΓOpt
    -- Convert to an Option statement about the Array lookup.
    have hAt :
        (vals[idx.id]?).map (fun v => v.1) = some s := by
      simpa [shapesOfVals_get?_eq] using hShapesAt
    -- `idx.id < vals.size`, so the Array lookup is `some (vals[idx.id]!)`.
    have hSome : vals[idx.id]? = some (vals[idx.id]!) := by
      simp [getElem?_pos, hiVals]
    -- Extract the shape from the mapped option.
    simpa [hSome] using hAt
end Correctness

end NN.Verification.TorchLean.Proved

