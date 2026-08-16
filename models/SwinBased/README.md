# SwinBased variants

Three configurations of the same architecture: shifted-window local attention plus a
pooled coarse-global attention path, with `HAI` gated residual scaling on every block.

| Directory | Depth / width | HAI | Checkpoint |
|---|---|:--:|---|
| `base_hai` | base | on | `model_epoch_195.pth` |
| `deep_hai_P` | more groups | on | `model_epoch_220.pth` |
| `wide_hai_P` | wider `transformer_dim` | on | `model_epoch_260.pth` |

## On the ablation

`HAI` is now gated by a `use_hai` flag on `TransformerModel` (default `True`), so a true
control can be trained:

```bash
python train.py --model SwinBased/base_hai --use_hai false --data_dir data/DIV2K
```

With `use_hai=False` no `alpha` parameters are created and the wrapped block is applied
unchanged, so the control differs from the gated model by exactly the `alpha` tensors —
every other state-dict key is identical.

**All three committed checkpoints were trained with HAI enabled.** This directory was
previously named `non_hai`, which was wrong: it contained and instantiated the same `HAI`
blocks as the other two. Renamed to `base_hai` so the name matches the code. The
HAI-vs-no-HAI row in the top-level results table is therefore **not yet measured** — it
needs a training run with `--use_hai false`, which is the outstanding work on this repo.
