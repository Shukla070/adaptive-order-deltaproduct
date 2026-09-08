# Adaptive-Order State Transitions in Linear RNNs

## BTech Major Project — Master Plan

**Utkarsh** · Guide: **Anand Kumar M** · NITK Surathkal
September 2026 – April 2027

> **This document replaces every earlier version.** Delete `masterprojectdocument.md`,
> `major_project_v2.md` and `major_project_v3.md` — everything in them that is still
> true is here, and everything that was wrong is corrected and explained in Part 9.

---

# Part 1 — What we are doing, in plain language

*Read this part even if you read nothing else. No jargon.*

### 1.1 The background

A language model has to remember things as it reads. A Transformer remembers by keeping every past token around and looking back at all of them — accurate, but the cost grows with length. The newer "linear RNN" family (Mamba, GLA, DeltaNet, DeltaProduct) instead keeps **one fixed-size memory** and updates it as each token arrives. Much faster, constant memory — but now everything depends on **how good that update rule is**.

The update rule is a matrix. How expressive that matrix is decides what the model can and cannot keep track of:

| Update rule | Cost | What it can track |
|---|---|---|
| Diagonal (Mamba, GLA) | cheapest | only "commutative" bookkeeping — counters, parity |
| One Householder reflection (DeltaNet) | medium | can swap two things per step |
| `n_h` Householder reflections (**DeltaProduct**) | `n_h` × medium | can do `n_h` swaps per step |

**DeltaProduct is our starting point.** It has a dial, `n_h`, that buys expressive power with compute.

### 1.2 The gap we found

**That dial is set once, for the entire model, and never changes.**

But how much power a token needs is *not* constant. Think of tracking a shuffled deck:

- some steps do nothing at all (the identity) → need **0** work
- some steps swap one pair of cards → need **1** unit of work
- some steps scramble five cards → need **4** units of work

A fixed-`n_h` model has to be set to **4** so it can handle the hardest step — and then it pays 4 units on *every* token, including the ones that needed nothing. That waste is the gap.

### 1.3 What we are building

**Let the model decide, per token, how much work that token gets.**

A tiny extra component — a few thousand parameters — looks at each token and its memory so far and decides: this one needs 1 reflection, this one needs 3, this one needs none. Nobody tells it the right answer. It learns because using less work is rewarded and getting the task wrong is punished, so it has to figure out where the hard parts are on its own.

We call it **Adaptive-Order DeltaProduct**.

### 1.4 The three things we want to show

1. **It learns better.** (Our main claim.) Weirdly, the biggest problem in this field right now is not that models *can't* represent the answer — it's that gradient descent *can't find* it. A recent ICLR 2026 paper proved deep models are powerful enough in principle but fail to train in practice. Our bet: a model that starts easy and adds power only where needed is easier to train than one that starts at full power everywhere.
2. **It's interpretable.** After training, we check whether the model's decisions line up with where the genuinely hard steps actually were. It was never told — so if it matches, it discovered the structure by itself. That's a strong, citable result.
3. **It's faster.** Same accuracy, less compute. Real, but with a ceiling we already calculated (Part 4) and will publish ourselves.

### 1.5 Why this is a real project and not a toy

- **Nobody has done it.** We checked the literature carefully. The closest things route *depth* or route between *attention variants*; nobody has made DeltaProduct's order input-dependent.
- **The maths is proven, not hoped for.** We verified numerically that a permutation needing ℓ swaps requires exactly ℓ reflections (Part 5). The dial has a precise mathematical meaning.
- **The implementation is a small patch, not a rewrite.** We read the source. It's about 30 lines in one file (Part 6).
- **It has a safety net.** Phase 1 produces a publishable result on its own, before the new architecture is even built.

---

# Part 2 — Status board

### ✅ Done

| # | Item | Result |
|---|---|---|
| 1 | Literature review | Novelty confirmed; two 2026 papers reframed the project; premise corrected |
| 2 | **Efficiency ceiling computed** | Closed form `(Hₙ−1)/(n−1)`; **32.1% for S₅** — see Part 4 |
| 3 | **Maths verified** (CPU oracle) | All 4 tests pass; permutations reconstruct at 4.4e-16 — Part 5 |
| 4 | **Parity constraint discovered** | Gate must scale a *learnable* β, not switch {0,2} — Part 5.3 |
| 5 | Environment built | Python 3.12.7, torch 2.6.0+cu126, triton 3.2.0, CUDA 13.0, fla 0.2.2, all deps |
| 6 | **Path B confirmed** (source read) | fla already interleaves to `n_h·T` and passes `cu_seqlens` — Part 6 |
| 7 | **β=0 ⇒ identity in real fla** | signal 4.93e-03 vs control 1.09 → **222× ratio, PASS** |
| 8 | Sampling confirmed (source read) | `random.choices(range(|G|))` = uniform over the group |

**All Week-1 blocking questions are answered, and all answers are favourable.**

| 9 | **S₃ smoke test — positive control** | `n_h=2`: **seq acc 1.000**, token acc 1.000, loss 0.00195, converged in 1 epoch |
| 10 | **S₃ smoke test — negative control** | `n_h=1`: **seq acc 0.000**, token acc 0.355 after 5 epochs. 1/0 pattern reproduced |
| 11 | **Training-loop bottleneck found & patched** | metrics ran per batch; now every 50 steps. 22m34s → 4m47s (4.7×) |

**PHASE 0 IS COMPLETE.** Every blocking question answered; the stack reproduces the
paper's own documented outcomes.

### 🔜 Next

| # | Item | Note |
|---|---|---|
| 12 | Write `src/gating.py` — monotone halting head | Laptop-friendly; validate vs `work/reference_deltaproduct.py` |
| 13 | **Confirm cluster allocation in writing** | Blocks Phase 1's full grid. Highest-value non-terminal task |
| 14 | W&B project set up | Before the first Phase 1 run |
| 15 | Empirical ℓ histogram on S₅ | `python work/check_data_histogram.py` — confirms the 32.1% ceiling |
| 16 | Remaining 2 negative controls (`allow_neg_eigval=False`) | Confirmation only; queue overnight |
| 17 | Move repo off `/mnt/c` | Deferred — measured NOT to be the bottleneck |

### ⏳ Then

Phase 1 measurement study (Weeks 2–8) → Checkpoint A (Week 12) → Checkpoint B (Week 20) → efficiency + interpretability → language modelling → write-up.

---

### 2.1 Measured on this hardware (RTX 4050 Laptop, 6 GB)

| Fact | Value | Consequence |
|---|---|---|
| VRAM spill point | batch 2048 allocates 8.57 GB → 3.365 s/step vs 0.201 at 1024 | WSL spills to system RAM silently; never OOMs. Cap batch at ≤1024 |
| Throughput saturates | ~5,000 samples/s from batch 512 up | Large batches buy no speed, only cost optimizer steps |
| Chosen batch | **256** | Same epoch time as 512/1024, 2–8× the updates |
| Data loading | 0.0043 s/batch, 1.7 s/epoch | NOT a bottleneck. `num_workers` barely matters |
| Arrow cache | `/home/hp/.cache` (native ext4) | `/mnt/c` is NOT the data bottleneck |
| Per-batch train metrics | ~81% of epoch wall-clock | Patched: every 50 steps. 22m34s → 4m47s |
| Residual overhead | ~0.59 s/step vs 0.054 s compute | ~10× headroom remains (likely `loss.item()` syncs). Deliberately not chased |
| One S₃ run | **~5 min** | Phase 1's grid is an overnight job, not a fortnight |

**Optimization finding worth carrying into Phase 1:** the failed run used batch 2048 for
4,834 steps and reached 82% per-position accuracy; the successful run used batch 256 for
387 steps — 100× less data — and reached 100%. Large batch with few updates was actively
worse. Batch size is not a free knob in these sweeps.

**Caveat to record:** `n_h=2` has 399,008 parameters against `n_h=1`'s 332,448, so the
smoke test alone does not separate expressivity from capacity. Phase 1's matched-parameter
design is what settles that — it is the reason E1b exists.

# Part 3 — The method

### 3.1 The recurrence

DeltaProduct runs `n_h` sequential micro-steps per token. Each micro-step:

```
S  ←  S · (I − β k kᵀ)  +  β v kᵀ        ‖k‖ = 1,  β ∈ [0, 2]
```

so the transition applied by token *t* is `A_t = Π_i (I − β_t^i k_t^i k_t^iᵀ)`.

`β` controls what each factor does:

| β | factor is | eigenvalue along k |
|---|---|---|
| 0 | **the identity** — this micro-step does nothing | +1 |
| 1 | a projection | 0 |
| 2 | a **reflection** — a swap | −1 |

That `β = 0` row is the whole trick: *skipping a micro-step is already inside the model's parameter space.* No new operation is needed.

### 3.2 Adaptive-Order DeltaProduct

```
λ_t^i ∈ [0,1]        small MLP head on (token projection, state summary)
g_t^i = Π_{j≤i} λ_t^j          monotone: g_t^1 ≥ g_t^2 ≥ … ≥ g_t^K
β̃_t^i = g_t^i · β_t^i          gate SCALES beta (see 5.3 — this matters)
n_t   = Σ_i g_t^i               effective order of token t

L = L_task + γ · mean_t[n_t] + δ · KL(halting ‖ geometric prior)
```

The gate is **never supervised**. Budget penalty pushes `n_t` down; task loss pushes it up where needed.

Budget penalty is **asymmetric** — under-allocation is punished harder than over-allocation, because spending too much only wastes compute while spending too little breaks correctness.

### 3.3 Design decisions, locked

| Decision | Choice | Why |
|---|---|---|
| Base | DeltaProduct / Gated DeltaProduct in `fla` | The dial we are generalising lives here |
| Mechanism | Monotone halting gates scaling β | Verified: exact identity at β=0 |
| **Gate form** | Scale a **learnable** β — never switch {0, 2} | Parity wall, Part 5.3 |
| Granularity | **Per token** | Expansion is on the token axis, chunking happens after |
| Precision | **bfloat16** | Forced: `chunk_gated_delta_rule` rejects fp32 |
| Max order K | 4 for S₅ | Max transposition length in S₅ is 4 |
| Supervision | None on the gate | Required for the interpretability claim |
| Primary benchmark | Group word problems (S₃, S₄, S₅, A₅) | Heterogeneity inherent to the group |
| Realistic benchmark | Published code-REPL benchmark (Siems 2026) | Cite; do not rebuild |
| Baselines | DeltaNet, DeltaProduct n_h∈{2,3,4}, Mamba-2/GLA, Gated DeltaNet | Matched parameters |

---

# Part 4 — The efficiency ceiling *(result, computed Week 0)*

`work/census.py`. For uniform sampling from Sₙ, transposition length has a closed form:

> **E[ℓ] = n − Hₙ**,  max ℓ = n − 1,  so **ceiling = (Hₙ − 1)/(n − 1)**

| Group | max n_h | E[ℓ] | Ceiling |
|---|---|---|---|
| S₃ | 2 | 1.167 | 41.7% |
| S₄ | 3 | 1.917 | 36.1% |
| **S₅** | **4** | **2.717** | **32.1%** |
| A₅ | 4 | 2.767 | 30.8% |
| A₄ | 2 | 1.833 | **8.3%** |
| S₈ | 7 | 5.282 | 24.5% |

Decays like ln n / n. In S₅, 74 of 120 elements need ℓ ≥ 3 and the identity is 1 element in 120 — the distribution is top-heavy.

**What this means, and why it is good news:**

1. **Uniform group words are near the worst case for our method**, so the synthetic benchmark is where the *learnability* and *interpretability* claims live — not the efficiency claim.
2. **The efficiency claim moves to language/code modelling**, where most tokens need no state tracking and the gap between global-worst-case and local-demand is large. Affordable because DeltaProduct released LM checkpoints, configs and SLURM scripts.
3. **The ceiling is itself a contribution.** Stated in reverse: *no adaptive-expressivity method can gain more than (Hₙ−1)/(n−1) on uniformly-sampled group word problems, and it shrinks with group size.* A negative result about a whole class of methods on the benchmark family this subfield defaults to. Three lines to prove; unstated because nobody proposed adaptive order.

Publishing our own ceiling, in closed form, is what makes reviewers trust everything else.

---

# Part 5 — What has been verified

`work/reference_deltaproduct.py` — a slow, loop-based float64 CPU implementation written to be *obviously* correct, used as the oracle. All tests pass.

### 5.1 The maths

| Test | Claim | Result |
|---|---|---|
| **T1** | β=0 gives exactly the identity; gating micro-steps 2…K off reproduces a true `n_h=1` model | `max|ΔS| = 0.00e+00` — **bit-identical** |
| **T2** | Every permutation in S₅ of transposition length ℓ is exactly a product of ℓ Householder factors | **4.44e-16** across all 120 elements |
| **T2b** | A reflection's eigenvalues are {−1,1,1,1,1} | explains why `allow_neg_eigval=True` is mandatory |
| **T3** | Gates all open ≡ fixed-order DeltaProduct; order counts correctly; gates monotone | exact |
| **T4** | Fitting K free factors to a target permutation converges iff K ≥ ℓ | <0.02 when K ≥ ℓ, >0.4 otherwise |

**Proposition 2 is therefore verified, not conjectured.** It also matches the DeltaProduct repo's own note that S₃ needs `n_h ≥ 2` in one layer — max ℓ over S₃ is exactly 2.

### 5.2 The real implementation

`work/check_beta_identity.py`, on the RTX 4050 in bf16. Because bf16 carries ~3 decimal digits and the two paths compared run sequences of different lengths (4T vs T) with different chunk boundaries, bit-exactness is impossible; the test is signal-vs-noise:

| | max relative difference |
|---|---|
| **Signal** — betas 1–3 gated off vs true `n_h=1` | **4.93e-03** |
| **Control** — all betas live vs the same `n_h=1` | **1.09e+00** |
| **Ratio** | **222×** |

**PASS.** β=0 acts as the identity in the real Triton path; the residual is bf16 rounding.

### 5.3 The parity constraint *(discovered by T4 — a real design consequence)*

Fitting K factors to target permutations with **β pinned at 2** converges in a checkerboard pattern:

| target | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| identity (ℓ=0) | ✗ | **0.0000** | ✗ | **0.0000** |
| transposition (ℓ=1) | **0.0000** | ✗ | **0.0000** | ✗ |
| 3-cycle (ℓ=2) | ✗ | **0.0000** | ✗ | **0.0000** |
| 5-cycle (ℓ=4) | ✗ | ✗ | ✗ | **0.0000** |

Every β=2 factor is a reflection with determinant −1, so `det(A) = (−1)^K` must match `sgn(p) = (−1)^ℓ`: only `K ≡ ℓ (mod 2)` is reachable. **Half of all targets are unreachable at any K.** With β free, the clean rule returns (converges iff K ≥ ℓ).

> **Locked consequence:** the gate must **scale a learnable β**. A hard binary gate switching β between {0, 2} — the obvious first implementation, and what a Gumbel/straight-through formulation naturally produces — hits this wall and silently fails on half the targets. This would have appeared months later as an unexplained training failure with no signal in the loss curves. It is now Proposition 3 and an explicit ablation in Phase 3.

### 5.4 Precision constraint

`chunk_gated_delta_rule` asserts bfloat16; fp32 is rejected outright. Consequences:

- The β=0 identity is unaffected — zero is exact in bf16.
- Every experiment runs in bf16. All comparisons need tolerances; nothing is ever bit-exact against the float64 oracle. The oracle validates **logic** (does the index map select the right outputs), not numerics.
- **Confound to control for:** the plan's error-control thread says drift along state-distinguishing directions cannot be corrected in affine models. bf16 rounding is a *second* drift source. When Phase 2 tests long sequences, "low-order span accumulates error" and "bf16 accumulates error" must be separated — run a control through the non-chunked path at higher precision.

---

# Part 6 — Implementation

### 6.1 How fla already works *(read from source — this is Path B, confirmed)*

`fla/layers/gated_deltaproduct.py`:

```python
k    = interleave_multiple_sequences(ks)                          # (B, n_h*T, ...)
v    = interleave_multiple_sequences(vs)
beta = interleave_multiple_sequences(betas)
q    = interleave_multiple_sequences([zeros]*(n_h-1) + [q])       # query only on last micro-step
g    = interleave_multiple_sequences([g] + [zeros]*(n_h-1))       # decay only on first
o, state = chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=offsets, ...)
o = o[:, n_h - 1 :: n_h, :]                                       # fixed-stride gather
```

It **already expands the sequence to `n_h·T`** and **already passes `cu_seqlens`**. So adaptive order is a *ragged* interleave — this is the good case, and it means wall-clock savings are reachable, not just FLOPs.

### 6.2 The change

| Fixed order (now) | Adaptive order (ours) |
|---|---|
| uniform `n_h`-way interleave | ragged interleave with per-token counts `n_t` |
| expanded length `n_h·T` | expanded length `Σ n_t` ← **this ratio is the speedup** |
| fixed stride `o[:, n_h-1::n_h]` | index map to each token's last micro-step |
| `beta = b_projs[i](h).sigmoid()` (line 253) | `... .sigmoid() * g_t^i` ← **gate goes here** |

Roughly 30 lines in one file.

### 6.3 Two paths

- **Path A — safe, no kernel work.** Run at fixed K with betas multiplied by gates. Training cost stays `K·T`, but the model *is* the adaptive one and inference order is genuinely `n_t`. Establishes claims 1 and 2 plus inference FLOPs. **This is the fallback that makes the project safe.**
- **Path B — wall clock.** Ragged expansion. Confirmed available (6.1).

**Do Path A first**, validate against the oracle, then Path B. A is not an approximation of B — it is the same model, computed wastefully.

---

# Part 7 — Immediate next steps

### 7.1 S₃ smoke test *(do this next)*

Run from `state_tracking/`, **not** the repo root. `batch_size` dropped to 512 for a 6 GB card; the convergence *pattern* is what matters.

```bash
cd "/mnt/c/D/B.Tech/Major Project/DeltaProduct/state_tracking"
PYTHONPATH=$PWD python src/generate_data.py --group=S3 --k=128 --samples=100000
PYTHONPATH=$PWD python src/generate_data.py --group=S3 --k=512 --samples=100000

# MUST converge
PYTHONPATH=$PWD python src/main.py train --group=S3 --k=128 --k_test=512 \
  --n_layers=1 --epochs=100 --allow_neg_eigval=True --num_householder=2 \
  --batch_size=512 --seed=666 --lr=1e-3 --n_heads=8 --use_scheduler=True

# MUST NOT converge — the other three published configs
#   --num_householder=1 --allow_neg_eigval=True
#   --num_householder=1 --allow_neg_eigval=False
#   --num_householder=2 --allow_neg_eigval=False
```

**Gate: the 1 / 0 / 0 / 0 pattern must reproduce.** These are the paper's own documented outcomes, so this validates the whole stack end to end.

### 7.2 Confirm the sampling empirically

```bash
PYTHONPATH=$PWD python src/generate_data.py --group=S5 --k=128 --samples=100000
python ../work/check_data_histogram.py data/S5=128.csv
```

Expect 120 distinct elements, uniform, and the 32.1% ceiling printed back.

### 7.3 Move off `/mnt/c`

The repo sits on a Windows drive reached over WSL2's 9p protocol — roughly 10–50× slower than native ext4, and Python imports feel it. Before Phase 1's ~200 runs:

```bash
pip wheel causal-conv1d --no-build-isolation -w ~/wheels    # save the hard-won build first
# then copy the repo to ~/major-project/, fresh venv, pip install ~/wheels/causal_conv1d-*.whl
```

### 7.4 Housekeeping

Set up W&B before the first real run — every figure in the paper should be regenerable from a logged run ID. Confirm cluster allocation in writing, including whether jobs can run **uncontended** (shared-node wall-clock numbers are worthless).

---

# Part 8 — The full plan

| Phase | Weeks | Dates | Deliverable |
|---|---|---|---|
| **0. Feasibility gate** | 1 | Sep 1–7 | ✅ mostly done — only 7.1 remains |
| **1. Measurement study** | 2–8 | Sep 8–Oct 26 | **Standalone publishable** |
| **2. Oracle order** | 9–12 | Oct 27–Nov 23 | **Checkpoint A** |
| **3. Learned gates** | 13–20 | Nov 24–Jan 25 | **Checkpoint B** — core claim |
| **4. Efficiency + interpretability** | 21–25 | Jan 26–Feb 28 | Wall clock, allocation analysis |
| **5. Language modelling** | 26–30 | Mar 1–Apr 5 | The real efficiency claim |
| **6. Writing & defense** | 29–32 | Apr | Report, repo, presentation |

### Phase 1 — Measurement study *(the insurance policy)*

**Question: is the state-tracking gap between architectures a capacity limit or an optimization limit?**

- **E1a — Heterogeneity census.** ✅ done analytically (Part 4). Remaining: 7.2.
- **E1b — Capacity vs optimization.** Diagonal (Mamba-2/GLA) at depths 1–4 and DeltaProduct at `n_h` 1–4, on solvable (S₃, S₄) and non-solvable (A₅, S₅) groups, matched parameters, each trained from **both** random init and analytical-solution init. *Random-init vs analytic-init gap = the optimization component; residual failure at analytic init = the capacity component.* Not published for the DeltaProduct family. ~200 short runs — this is what the cluster is for.
- **E1c — Error control.** Distinguishability ratio across sequence length; locate the predicted collapse threshold per architecture. Include the bf16 control from 5.4.

### Phase 2 — Oracle order → **Checkpoint A**

Set `n_t = ℓ(g_t)` from the generator's known label — no learning yet. Compare to fixed `n_h = K`.

**Passes if** oracle-adaptive matches fixed-K accuracy at `Σn_t` within ~10% of the Part 4 ceiling, at two or more sequence lengths including one long enough to stress error control.

**If it fails:** diagnose (i) insufficient heterogeneity — 7.2 should have caught it; (ii) drift over low-order spans — test a minimum floor `n_t ≥ 1`; (iii) implementation defect against the oracle. **Do not proceed until the cause is known.**

### Phase 3 — Learned gates → **Checkpoint B**

Ablations, all at **matched budget** `Σn_t`:

| Ablation | Tests |
|---|---|
| Fixed `n_h` at matched *average* order | Does adaptivity help at all? |
| Random gates at matched budget | Does *learning where* help, or just varying? |
| Scheduled gates (every k-th token high) | Is a static schedule enough? |
| Oracle gates (Phase 2) | Upper bound — how much does learning recover? |
| Content-only gate (no state summary) | Does the gate need state, or is it token pattern-matching? |
| **Hard binary gate β ∈ {0,2}** | Confirms the parity wall (5.3) empirically |

**Learnability sub-experiment — the headline.** Compare *learning curves* against fixed `n_h = K` and deep diagonal, matched parameters and matched total compute. Hypothesis: adaptive order reaches a given loss in fewer steps because it starts in an effectively low-order regime. Also test whether it removes the need for the analytical initialization E1b shows is otherwise required — **that would be the single strongest result available to this project.**

**Passes if** learned gates beat random and scheduled at matched budget, ≥3 seeds, non-overlapping CIs.

**If it fails** (learned ≈ random): a genuine negative result about learned conditional expressivity. Paired with Phase 1, still an honest paper. Report it plainly.

### Phase 4 — Efficiency and interpretability

- **E4a** Path B ragged expansion; benchmark vs fixed `n_h=K` and DeltaNet on an **uncontended** node. Report FLOPs and wall clock separately — a FLOPs win with no wall-clock win is a reportable outcome, not a hidden failure.
- **E4b** Precision/recall of learned `n_t` against true ℓ(g_t). **Must include a non-locally-detectable condition** — the element at position *t* specified by reference to an earlier position, so local content does not reveal difficulty and the gate must use accumulated state. Without it the interpretability claim is vacuous: the gate could just be reading the token.
- **E4c** Allocation vs position and sequence length; does allocation rise near E1c's collapse threshold?

### Phase 5 — Language modelling

**Where the efficiency claim actually lives.** Use released DeltaProduct checkpoints and configs for `n_h ∈ {1,2,3}` baselines; train adaptive-order at matched parameters. Measure the **allocation histogram over natural text and code** — the central empirical question is whether real token streams show the heterogeneity that uniform group words lack. Evaluate with `lm-eval` and the released length-extrapolation scripts. Secondary: the published code-REPL benchmark.

---

# Part 9 — How this plan got here

*Recorded deliberately. The discarded reasoning is what makes the current version defensible, and reviewers ask these questions.*

### v1 — "Adaptive Mechanism Routing" (discarded)

A diagonal backbone plus a learned **chunk-level router** dispatching chunks to a DeltaNet branch, on a bespoke benchmark with a sparsity parameter `p_hard`. Discarded for four reasons:

1. **Arithmetic contradiction.** At chunk size 64 with i.i.d. sparsity, a chunk is cheap only if it contains no hard operation — so at `p_hard = 0.05`, `1 − 0.95⁶⁴ ≈ 96%` of chunks are expensive. Half-cheap needs `p_hard ≈ 1%`: ~22 hard ops in a 2048-token sequence. Chunk granularity and i.i.d. sparsity are mutually incompatible.
2. **Circular benchmark.** A benchmark whose tunable parameter controls exactly the property the method exploits gives a foregone conclusion. v1's own risk register conceded natural tasks aren't sparse.
3. **Load-bearing kernel risk.** The efficiency claim needed ragged dispatch between two *different* kernels with a state hand-off — the likeliest phase to fail, and nothing survived its failure.
4. **Novelty claim too broad.** "Routing which mechanism rather than how much compute is open" does not survive MoAS (Dec 2025).

### Two corrections to the premise

- **Not "non-Abelian."** A *single-layer* diagonal SSM tracks G iff G is Abelian, but a *k-layer* one tracks G iff G has a subnormal series of length k with Abelian factors — depth buys exactly the **solvable** groups. S₃ and S₄ are solvable. The real barrier is **non-solvable** groups; A₅ (order 60) is smallest. Always say "non-solvable."
- **Not parity.** Diagonal SSMs with negative eigenvalues solve parity, which is Abelian (Z₂).

### The finding that reframed everything

Two independent 2026 results say the bottleneck is **not** capacity:

- Shakerinava et al. (ICLR 2026): multi-layer diagonal models that provably *can* represent solvable-group tracking **fail to learn it by gradient descent**; analytic initialization restores trainability.
- *Error Control Dynamics* (2026) proves **affine neutrality**: an affine recurrence preserving symbolic state exactly acts as the identity on the symbolic subspace, so it cannot contract error along state-distinguishing directions. Exact preservation and error correction are incompatible.

So any project measuring "diagonal fails / DeltaNet succeeds" and calling it expressivity will be challenged. Hence learnability-first, and Phase 1's explicit decomposition.

### v2 → v3 → v4

v2 proposed adaptive order with group words as the primary efficiency benchmark. The Week-0 census showed that caps the win at 32% and shrinking → v3 moved efficiency to LM and made the ceiling a contribution. v4 adds everything verified in Week 1: the parity constraint, the bf16 constraint, Path B confirmed from source, and the β=0 identity confirmed on GPU.

---

# Part 10 — Novelty position

| Work | Relation | Distinction |
|---|---|---|
| **DeltaProduct** (NeurIPS 2025) | Direct parent | `n_h` fixed and global; we make it input-dependent and learned |
| **MoAS** (Dec 2025) | Nearest in *shape* | Per-token routing among MHA/GQA/MQA — cost tiers, no expressivity theorem |
| **suRNN** (May 2026) | Nearest in *mechanism* | Per-neuron gates on *whether* to update. Gates participation, not algebraic order |
| **Mixture-of-Depths / -Recursions** | Conditional compute | Routes depth; the operator never changes |
| **arXiv 2607.07953** | Name collision only | Static cross-layer information flow, not input-dependent dispatch |
| **AUSSM** (2507.05238) | Adjacent | Interleaves mechanisms at layer level, statically |

**The defensible claim:** routing along an axis carrying a *proven* expressivity guarantee, where the routed quantity has group-theoretic meaning (`n` factors ⇒ permutations of transposition length ≤ `n`) — verified in Part 5, not asserted.

---

# Part 11 — Theory (target: one page in the paper)

- **Prop 1 (strict generalization).** Gates open ⇒ exactly DeltaProduct(K); `g¹=1, g^{i>1}=0` ⇒ exactly DeltaNet; all zero ⇒ identity. ✅ *verified T1, T3*
- **Prop 2 (order requirement).** A permutation of transposition length ℓ is exactly a product of ℓ generalized Householder factors, and no fewer. ✅ *verified T2, T4*
- **Prop 3 (parity constraint).** With β pinned to 2, a K-factor product realizes only permutations with K ≡ ℓ (mod 2). Free β is necessary. ✅ *verified T4*
- **Prop 4 (fixed-order waste).** Fixed order needs `n_h ≥ maxₜ ℓ(gₜ)`, cost `T·max ℓ`; adaptive cost is `Σℓ(gₜ)`. Waste `T·(max ℓ − E[ℓ])` > 0 for any non-degenerate distribution.
- **Prop 5 (the ceiling).** For uniform Sₙ, `E[ℓ] = n − Hₙ`; relative saving bounded by `(Hₙ−1)/(n−1) → ln n / n`. ✅ *verified Part 4*. The motivating **and** limiting theorem.

**Framing caution.** *An Algebraic View of the Expressivity of Recurrent Language Models* (June 2026) gives a general wreath-product characterization. Check whether Props 2 and 4 are corollaries; if so, present them as instantiations, not new results.

---

# Part 12 — What a reviewer will attack

| Attack | Answer |
|---|---|
| "Mixture-of-Depths relabelled." | Those route depth or cost tiers heuristically. Our axis carries a proven expressivity guarantee (Prop 2, verified); Prop 4 bounds what fixed allocation must waste. |
| "You measured optimization and called it expressivity." | E1b decomposes it via analytic initialization. This is why Phase 1 exists. |
| "Heterogeneity is an artifact of your sampling." | Part 4 gives the closed-form distribution under uniform sampling and reports the ceiling *against our own interest*. Phase 5 shows heterogeneity in natural data. |
| "Only 32% savings." | Stated by us, in closed form, as a property of the benchmark family — and it is why efficiency is evaluated on LM. |
| "The gate just pattern-matches tokens." | E4b's non-locally-detectable condition + the content-only ablation. |
| "FLOPs win, no wall-clock win." | Reported as such; claims 1 and 2 don't depend on it. |
| "Long sequences: low-order spans drift." | E1c and E4c, with the bf16 confound controlled (5.4); minimum-order floor pre-registered. |
| "Synthetic only." | Phase 5. Honest scope: state-tracking-limited settings. |

---

# Part 13 — Risk register

| Risk | Severity | Status / mitigation |
|---|---|---|
| β=0 not identity in fla | ~~High~~ | ✅ **resolved** — 222× signal-to-noise, Part 5.2 |
| Path B unavailable | ~~Medium~~ | ✅ **resolved** — confirmed from source, Part 6.1 |
| Generator samples generators not uniform G | ~~High~~ | ✅ **resolved** — `random.choices(range(|G|))`; confirm empirically in 7.2 |
| Parity wall from a binary gate | — | ✅ **found and designed around**, Part 5.3 |
| bf16 confounds the drift analysis | Medium | Control run through the non-chunked path, 5.4 |
| Gates collapse to always-K or always-1 | Medium | Known MoE/MoD mode: strengthen budget term, tune geometric prior, check gate-logit init |
| Learned ≈ random at matched budget | Medium | Publishable negative result paired with Phase 1 |
| Drift over low-order spans | Medium | Predicted by affine neutrality; minimum-order floor pre-registered |
| Natural text shows no heterogeneity | Medium | Would cap efficiency at the measured ceiling; claims 1–2 unaffected |
| **Someone publishes adaptive order first** | **High** | arXiv Phases 1+2 in Dec–Jan without waiting. Being second by six weeks is being nowhere |
| Cluster / uncontended timing unavailable | Medium | 7.4. Phases 1–4 fit one GPU; only Phase 5 needs more |

---

# Part 14 — Publication

- **Venues:** ICLR / NeurIPS / ICML / COLM. Not ACL/EMNLP.
- **Timing:** ICLR 2027's deadline is ~September 2026 — out of reach. ICML 2027 (~late Jan) only if Checkpoint A clears by early December. **NeurIPS 2027 (~May 2027) is the realistic main-track target.** ICLR/ICML 2027 workshops in spring are the safe landing. *Verify in January; deadlines move.*
- **Sequencing:** arXiv Phases 1+2 as soon as solid. Priority is established on arXiv in this field.
- **Odds are raised most by:** Prop 5 (a result against our own method — reads as rigour), the E1b decomposition, and E4b under the non-locally-detectable condition. Reviewers here weight these above benchmark numbers.
- **Honest expectation.** A top-venue main-track acceptance from a solo undergraduate project in eight months is uncommon. The plan is built so a workshop paper plus a strong thesis is the **floor**.

---

# Part 15 — Files and environment

### What is in `work/` — every file, what it does

| File | Purpose | Status |
|---|---|---|
| `census.py` | Transposition-length census; derives the Part 4 ceiling. No GPU, 4 seconds | ✅ run |
| `reference_deltaproduct.py` | float64 CPU oracle + tests T1–T4. The correctness reference for everything | ✅ all pass |
| `check_beta_identity.py` | The blocking check: β=0 ⇒ identity in the real Triton path | ✅ PASS, 222× |
| `check_data_histogram.py` | Confirms the generator samples uniformly over G | ⏳ next |
| `install_deps.sh` | The ~10 packages `main.py`/`generate_data.py` need, incl. 2 git-only | ✅ run |

**One file was created outside `work/`:** `flash-linear-attention/README.md`. The vendored copy of fla was missing it, and `setup.py` reads it for `long_description`, so the install failed. It is a four-line placeholder. **No existing source file has been modified.**

### Environment

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 LTS on WSL2 |
| Python | 3.12.7 (pyenv) |
| PyTorch | 2.6.0+cu126 |
| Triton | 3.2.0 |
| CUDA toolkit | 13.0.88 (nvcc) |
| GPU | RTX 4050 Laptop, 6 GB |
| fla | 0.2.2 (editable, vendored) |
| transformers / datasets | 4.49.0 / 3.3.0 — **must stay <5 / <4**; fla 0.2.2 breaks on transformers 5 |
| causal-conv1d | 1.7.0 |

**Environment gotchas learned the hard way:**

- `TMPDIR=$HOME/tmp` is required — `/tmp` is a 3.8 GB tmpfs and isolated builds overflow it.
- Always `--no-build-isolation`, or pip downloads a second PyTorch (~500 MB plus CUDA wheels).
- transformers **must** stay on 4.x. fla 0.2.2 imports internals that moved in 5.x.
- The kernels are **bfloat16-only**.
- The repo lives on `/mnt/c` — see 7.3.

---

### Repository layout (target)

```
DeltaProduct/
├── work/                        # our code (see Part 15)
├── src/                         # adaptive-order implementation
│   ├── gating.py                #   monotone halting head
│   ├── model.py                 #   adaptive layer (Path A, then B)
│   └── data.py                  #   wrappers + oracle ℓ labels
├── experiments/                 # one config per E1a…E5, W&B run ids recorded
├── results/                     # figures regenerable from run ids
├── flash-linear-attention/      # vendored fla (editable install)
├── state_tracking/              # word-problem benchmark + training entry point
└── PLAN.md                      # this document
```

---

*v4 · 7 September 2026 · supersedes v1, v2, v3. Living document — record what changes and why.*
