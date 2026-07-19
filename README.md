# Fractal Mamba: Neural Renormalization-Group Flows for State-Space Models

A PyTorch implementation of `FractalMambaBlock` — a selective state-space
(Mamba-style) block that is applied **recursively across a hierarchy of
coarse-grained sequence resolutions**, with an explicit auxiliary loss
that pushes the model toward a **fixed point of a renormalization group
(RG) transformation**, and a spectral regularizer that encourages
scale-free (power-law) internal representations.

## Why RG flow?

In statistical/quantum field theory, an RG transformation `R` coarse-grains
microscopic degrees of freedom (e.g. block-spins on a lattice) and produces
a *renormalized* theory with rescaled coupling constants. A theory is
**scale-invariant** at a fixed point if applying the physics and then
coarse-graining gives the same result as coarse-graining and then applying
the physics: `R[F(x)] = F(R[x])`.

Standard deep sequence models (flat Transformer/Mamba stacks) have no
notion of scale at all — every layer sees the sequence at the same
resolution. This repo instead treats **sequence-length coarse-graining as
a literal RG step**, shares the SSM's parameters across scales (so the
"physics" has the same functional form at every resolution, exactly as
in a scale-invariant field theory), and lets a single learned per-level
gain (the RG eigenvalue `lambda^level`) decide whether the coarse
("infrared") mode is *relevant* (`lambda > 1`, amplified with depth) or
*irrelevant* (`lambda < 1`, suppressed with depth).

| RG concept | Code |
|---|---|
| Microscopic degrees of freedom | tokens at the finest resolution |
| Block-spin coarse-graining `R` | `CoarseGrain` (learned local pooling) |
| Renormalized Hamiltonian `H' = R[H]` | the same `SelectiveSSM` weights re-applied to the coarse sequence |
| Flow of coupling constants | `FractalMambaBlock.log_lambda` (per-level gain) |
| Relevant / irrelevant operators | sign & magnitude of the learned gain |
| Fixed point of the RG map | `RGFixedPointLoss` (commutation penalty) |
| Scale-free / critical spectrum | `PowerLawSpectrumLoss` (power ~ k^-alpha) |

## Repo layout

```
fractal_mamba/
  ssm.py      # SelectiveSSM: pure-PyTorch selective scan (Mamba-lite)
  block.py    # CoarseGrain + FractalMambaBlock (the recursive RG block)
  losses.py   # RGFixedPointLoss, PowerLawSpectrumLoss, ScaleInvariantLoss
  model.py    # FractalMambaNet: embedding + stacked blocks + head
examples/
  train_demo.py   # synthetic long-range "selective copy" recall task
tests/
  test_shapes.py  # shape/gradient sanity checks (all currently passing)
```

## Quickstart

```bash
pip install torch
python tests/test_shapes.py        # sanity checks
python examples/train_demo.py      # toy training FUNNNNNN
```

```python
from fractal_mamba import FractalMambaNet

model = FractalMambaNet(vocab_size=50257, d_model=256, d_state=16, n_blocks=6)
logits, aux = model(tokens)  # tokens: (B, L) long tensor

task_loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
loss = task_loss + 0.05 * aux["scale_invariant_loss"]
loss.backward()
```

`aux["components"]` exposes the unweighted `rg_fixed_point` and
`power_law_spectrum` sub-losses separately for logging/ablation.

## Honesty about scope

This is a **research scaffold**, not a paper's worth of validated
results:

- The selective-scan in `ssm.py` is a plain sequential `for`-loop for
  clarity/auditability, not the fused parallel-scan CUDA kernel from the
  official `mamba-ssm` — expect it to be slow on long sequences (O(L)
  Python-level steps). Swapping in a parallel associative scan is the
  main thing needed before scaling this up.
- `examples/train_demo.py` is a smoke test showing the whole pipeline
  trains without shape/gradient errors and that the RG fixed-point loss
  measurably decreases (i.e. the coarse-graining is learning toward
  self-consistency) — it is **not** a benchmark result, and 300 steps on
  a tiny model is nowhere near enough to claim the fractal hierarchy
  outperforms a flat baseline. A real ablation (fractal vs. flat Mamba,
  matched parameter count, on e.g. Long Range Arena or a real LM corpus)
  is the natural next step before writing this up.
- The "RG eigenvalue" `lambda` is a single learned scalar per level, a
  deliberately simple stand-in for the much richer operator-scaling
  structure of real RG flows (which has a whole spectrum of eigenvalues
  per coupling). Treat it as a suggestive inductive bias, not a
  first-principles derivation.

## credits
MEEE I MADE IT YIPEEEE
