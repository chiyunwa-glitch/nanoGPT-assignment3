# microgpt — Assignment 3 (MASC 515)

A single-file extension of Andrej Karpathy's
[microgpt](https://karpathy.github.io/2026/02/12/microgpt/), the dependency-free
~200-line GPT written in pure Python (its only imports are `os`, `math`, `random`).
This repo adds four modern transformer mechanisms to that base, implemented
directly against microgpt's `Value` scalar autograd, with no PyTorch or numpy
introduced.

## Run

```sh
python microgpt.py
```

The script downloads the names dataset on first run, trains a tiny GPT for 1000
steps, and then samples 20 hallucinated names. Architecture extensions are
toggled by the four flags near the top of the file.

## Commit history

| Commit | What it adds |
|---|---|
| `Initial commit` | Unmodified microgpt (Step 1 of the assignment) |
| `Add GELU activation in MLP` | Extension 1 (this repo) |
| `Add Rotary Position Embeddings` | Extension 2 |
| `Add sparse Mixture of Experts MLP` | Extension 3 |
| `Add LoRA low-rank adaptation of attention` | Extension 4 |

## Feature flags

```python
USE_GELU = True   # GELU activation in the MLP
USE_ROPE = True   # rotary position embeddings on Q, K
USE_MOE  = False  # sparse Mixture of Experts MLP
USE_LORA = False  # low-rank adapters on attention, base frozen
```

`USE_GELU` and `USE_ROPE` are both on by default — their addition is a strict
upgrade. `USE_MOE` and `USE_LORA` change the parameter count and training
objective so they are off by default; flip either to `True` to exercise it.

The four extensions are independent and compose cleanly: in particular `USE_LORA`
only touches attention projections, so it can be combined with `USE_MOE`.

## Algorithms

### 1. GELU — Gaussian Error Linear Units

Source: Hendrycks & Gimpel, [arXiv:1606.08415](https://arxiv.org/abs/1606.08415).

**Underlying idea.** ReLU multiplies an input by 0 or 1 depending on its sign —
a hard, non-differentiable gate. GELU instead multiplies the input by the
probability that a standard normal random variable falls below it, which gives
a smooth gate: small negatives leak through, large negatives still die, but the
transition is differentiable everywhere. This is the activation GPT-2 (and
modern transformers generally) actually use; the original microgpt simplifies
to ReLU.

The exact form is `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`. The paper also
gives a tanh approximation accurate to ~1e-3:

`GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`

**In this codebase.** microgpt's `Value` autograd exposes only `+ - * / ** log
exp relu` — there is no `erf` and no `tanh`. We use the tanh approximation and
build `tanh(u) = (eᵘ - e⁻ᵘ) / (eᵘ + e⁻ᵘ)` from the existing `exp`, so the chain
rule still flows through the autograd graph. The tanh approximation matches the
exact erf form to within ~4e-4 across the relevant range.

### 2. RoPE — Rotary Position Embeddings

Source: Su et al., [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

**Underlying idea.** Original Transformer-style position embeddings are *added*
to the input once; the model has to learn what those vectors mean. RoPE instead
encodes position by *rotating* the query and key vectors at position `m` by an
angle `m·θᵢ`, where `θᵢ = base^(-2i/d)` cycles slowly across the head
dimension. Because rotations are linear, the dot product `Q·Kᵀ` between a query
at position `m` and a key at position `n` reduces to a function of the
difference `m − n` only, not the absolute positions. So the attention scores
are intrinsically relative-position-aware, with no learned parameters needed
for position.

**In this codebase.** We drop the learned `wpe` matrix entirely when
`USE_ROPE=True`, saving `block_size × n_embd` parameters. The cos/sin tables
are precomputed as plain Python floats and enter the autograd graph through
scalar multiplication with `Value` objects, so RoPE adds no trainable params
and no extra autograd nodes for the rotation constants. We use the original
adjacent-pair convention `(vec[2i], vec[2i+1])` from eq. 15 of the paper.
Because microgpt processes one position per call and builds up a KV cache,
rotation is applied once at the time each `k` is appended to the cache; later
queries dot against the already-rotated keys.

### 3. Mixture of Experts (MoE)

Source: Hugging Face blog, [https://huggingface.co/blog/moe](https://huggingface.co/blog/moe).

**Underlying idea.** A dense MLP forces every token through the same set of
weights. MoE replaces it with `N` independent expert MLPs and a small *router*
that scores experts per token; each token visits only the top-`k` experts the
router likes best. Total parameters grow with `N`, but per-token compute stays
roughly constant (proportional to `k`, not `N`). Different experts can
specialise — punctuation, names, code, etc. — without any one being forced to
handle everything. The catch is *router collapse*: without intervention the
router happily ships every token to one or two favourite experts. The standard
fix is an auxiliary load-balancing loss added to the training objective.

**In this codebase.** We use 4 experts and top-2 routing (Mixtral-style), with
the top-2 router probabilities re-normalised to sum to 1 before combining
expert outputs. The aux loss follows the form referenced in the HF blog,
`aux = N · Σᵢ fᵢ · Pᵢ`, where `fᵢ` is the document-level fraction of tokens
that picked expert i (treated as constant) and `Pᵢ` is the document-level mean
of router softmax probability for expert i (carries gradient). Multiplying the
two pushes both selection counts and probabilities toward uniformity. Because
microgpt processes one token per `gpt()` call, the per-token routing stats
accumulate in a small module-level buffer that is reset at the top of each
document forward; the document-level aux loss is computed once after the
forward pass and added to the mean cross-entropy with weight `MOE_AUX_COEF`.

### 4. LoRA — Low-Rank Adaptation

Source: Hu et al., [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

**Underlying idea.** When you fine-tune a large pretrained model, the actual
update `ΔW` you make to any one weight matrix tends to live in a tiny
subspace — it is *low rank*. So instead of training `ΔW` directly (millions of
parameters), train a low-rank factorisation `ΔW ≈ B·A` where `A` is `r × in`,
`B` is `out × r`, and `r` is much smaller than `min(in, out)`. The base weight
`W` stays frozen, so `(forward) y = W·x + (α/r) · B·(A·x)`. `A` is randomly
initialised; `B` is initialised to **zero**, so at step 0 the adapter is a
no-op and the model output is bit-identical to the frozen base. Training
perturbs `B` away from zero only as the task demands.

**In this codebase.** We target the four attention projections
(`Wq, Wk, Wv, Wo`) per the paper's recommendation; the MLP / MoE / `wte` /
`lm_head` are not wrapped. The two-call form
`linear(linear(x, A), B)` keeps the intermediate at size `r`, so peak memory is
`O(r)` rather than `O(in·out)`. Freezing is implemented as a single
`trainable_mask: list[bool]` built once at startup; the Adam loop zeros the
gradient and skips the parameter update for masked-out entries, while
gradients still flow through the autograd graph correctly. With `n_embd=16`,
`rank=4`, four targets per layer: 512 trainable / 4192 total = **12.21%**. The
zero-init invariant is verified empirically — step-1 loss with `USE_LORA=True`
matches step-1 loss with `USE_LORA=False` to all printed digits.

## Acknowledgements

microgpt is © Andrej Karpathy and was originally published at
[karpathy.github.io/2026/02/12/microgpt/](https://karpathy.github.io/2026/02/12/microgpt/);
this assignment extends his single-file implementation. The LICENSE file
carried over from the upstream MIT license still applies.
