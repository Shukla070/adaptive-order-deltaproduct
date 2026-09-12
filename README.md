# Adaptive-Order State Transitions in Linear RNNs

**→ [`PLAN.md`](PLAN.md) is the master document**: the idea in plain language, the
status board, all results, and the reasoning behind every design decision.

---

## What this is

DeltaProduct's state-transition matrix is a product of `n_h` generalized Householder
factors. `n_h` is a **global hyperparameter** — set once, applied to every token.

But the expressivity a token *needs* varies. A permutation of transposition length ℓ
requires exactly ℓ Householder factors (verified — see `work/reference_deltaproduct.py`),
and ℓ ranges from 0 to 4 over S₅. A fixed-`n_h` model must provision for the worst case
and pay it everywhere.

This project makes that order **input-dependent and learned**, via a monotone halting
gate trained end-to-end with a budget penalty and no supervision on the allocation.

## Relationship to upstream

This repository is built on **[automl/DeltaProduct](https://github.com/automl/DeltaProduct)**
(Siems, Carstensen, Zela, Hutter, Pontil, Grazzi — NeurIPS 2025), whose history is
preserved here. Upstream's README is kept as [`README_upstream.md`](README_upstream.md).

Everything under `work/` is ours. Changes to upstream files are marked
`PATCH(adaptive-order)` in-line, with originals kept alongside as `*.orig`.

## Layout

| Path | |
|---|---|
| `PLAN.md` | Master plan, results, status |
| `work/` | Our tooling and verification (see below) |
| `state_tracking/` | Upstream benchmark + training entry point (one patch) |
| `flash-linear-attention/` | Vendored fla; the DeltaProduct layer we extend |

### `work/`

| File | Purpose |
|---|---|
| `reference_deltaproduct.py` | float64 CPU oracle + tests T1–T4. Correctness reference |
| `census.py` | Transposition-length census; derives the efficiency ceiling |
| `check_beta_identity.py` | Verifies β=0 ⇒ identity in the real Triton kernel |
| `check_data_histogram.py` | Confirms the generator samples uniformly over G |
| `bench_batch.py` | Finds the VRAM spill point for a given GPU |
| `bench_dataloader.py` | Separates dataloader / filesystem / compile bottlenecks |
| `install_deps.sh` | Dependencies beyond upstream's requirements |

## Reproducing

```bash
pip install --no-build-isolation -e ./flash-linear-attention
bash work/install_deps.sh
python work/reference_deltaproduct.py     # maths — all checks must pass
python work/check_beta_identity.py        # kernel — must PASS

cd state_tracking
PYTHONPATH=$PWD python src/generate_data.py --group=S3 --k=128 --samples=100000
PYTHONPATH=$PWD python src/main.py train --group=S3 --k=128 --k_test=128 \
  --n_layers=1 --epochs=100 --allow_neg_eigval=True --num_householder=2 \
  --batch_size=256 --seed=666 --lr=1e-3 --n_heads=8 --use_scheduler=True --logging=False
```

`n_h=2` reaches sequence accuracy 1.000 in one epoch; `n_h=1` stays at 0.000 — S₃'s
3-cycles need two transpositions. See `PLAN.md` Part 2 for the full status.

## License

Upstream code is MIT (see `LICENSE`). Our additions are released under the same terms.
