# fractal-mamba

A PyTorch experiment with a Mamba-style selective state-space block. The block runs on the full sequence, then on a shortened version of it, then on an even shorter one, and so on. The same weights are used at every level. Two extra loss terms are meant to make the model behave consistently across those levels.

This is a research toy. I have not shown that it beats a normal flat model.

## The idea

The idea comes from physics. In the renormalization group (RG), you coarse-grain a system, for example by merging neighboring spins on a lattice, and then look at how the description changes. Models that behave the same way at every scale are called scale-invariant.

I borrowed that picture for sequences. Coarse-graining a sequence means pooling it down to fewer positions. Here is how the pieces map to code:

| Physics idea | In this repo |
| --- | --- |
| Microscopic degrees of freedom | tokens at the finest resolution |
| Coarse-graining step | `CoarseGrain`, a learned local pooling |
| Same "physics" at every scale | the same `SelectiveSSM` weights, reused on each coarser sequence |
| Couplings changing with scale | `FractalMambaBlock.log_lambda`, one learned gain per level |
| Fixed point of the RG map | `RGFixedPointLoss` |
| Scale-free spectrum | `PowerLawSpectrumLoss`, which pushes power toward k^-alpha |

The fixed-point loss checks a commutation condition: coarse-graining and then processing should give about the same result as processing and then coarse-graining, or `R[F(x)] = F(R[x])`. That is a consistency loss inspired by RG. It does not prove the model is at an RG fixed point.

## What is in the repo

```
fractal_mamba/
  __init__.py
  ssm.py      SelectiveSSM, a pure PyTorch selective scan
  block.py    CoarseGrain and FractalMambaBlock
  losses.py   RGFixedPointLoss, PowerLawSpectrumLoss, ScaleInvariantLoss
  model.py    FractalMambaNet: embedding, stacked blocks, output head
tests/
  test_shapes.py
```

## Usage

```
pip install torch
python tests/test_shapes.py
```

```python
import torch.nn.functional as F
from fractal_mamba import FractalMambaNet

model = FractalMambaNet(vocab_size=50257, d_model=256, d_state=16, n_blocks=6)
logits, aux = model(tokens)  # tokens: (B, L) long tensor

task_loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))
loss = task_loss + 0.05 * aux["scale_invariant_loss"]
loss.backward()
```

The 0.05 is just the weight in this example. `aux["components"]` gives the two sub-losses, `rg_fixed_point` and `power_law_spectrum`, separately and unweighted, so you can log them or turn one off.

## What is not shown

- **No results.** I have not compared this against a flat Mamba or a Transformer. A fair test would use matched parameter counts, several random seeds, and a task like Long Range Arena or a real text corpus. It would also include a third model: the same recursive model with the auxiliary loss weight set to 0. That separates the effect of the hierarchy from the effect of the loss.
- **The fixed-point loss going down proves little.** I'm minimizing it, so it should go down. A consistency loss like this can also be satisfied by collapsing the representations toward a constant. This repo doesn't check for that. Logging the feature variance or effective rank next to the loss would show it.
- **The scan is slow.** `ssm.py` uses a plain Python loop over the sequence, which is easy to read and check but slow on long sequences. It is not the fused CUDA kernel from the official `mamba-ssm` package. A parallel scan is the main thing to add before using this on long inputs.
- **The "RG eigenvalue" is one learned number per level.** Real RG flows have a whole spectrum of eigenvalues. Treat this as a simple stand-in.
- **The tests are shape and gradient checks.** They check that the code runs and gradients flow. They don't check that the model works well.
- **"Fractal" just means shared weights across scales.** I haven't shown self-similar behavior beyond that.

## Why the name

I got the idea from a short video I can't find anymore: a stock photo of a mamba in a suit and tie, with a Jujutsu Kaisen-style narrator explaining "fractal mamba." The snake and the architecture share a name, and I couldn't leave it alone. If you know where that video is, please open an issue.

## References

- Albert Gu and Tri Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023).
- Zihang Dai, Guokun Lai, Yiming Yang, and Quoc Le, "Funnel-Transformer" (2020), which also shortens sequences with pooling.
- Pankaj Mehta and David Schwab, "An exact mapping between the Variational Renormalization Group and Deep Learning" (2014).

## Credits

Made by kds1123001.
gng✈️📗🍏🥬🔵🟩🌳💙💦💦🏡🏡

## License

MIT. See `LICENSE`.
