import VersoManual
import TorchLeanBlueprint.Guide

open Verso Doc
open Verso.Genre Manual

def main (args : List String) : IO UInt32 :=
  manualMain
    (%doc TorchLeanBlueprint.Guide)
    args
    (extensionImpls := by exact extension_impls%)
