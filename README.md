
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)


---

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

---

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!

## Architecture Extensions

### GELU — Gaussian Error Linear Units (Hendrycks & Gimpel, [2016](https://arxiv.org/abs/1606.08415))

**Core idea:** GELU is "ReLU, but smooth." Where ReLU hard-cuts every negative input to zero, GELU multiplies each input by the probability that a standard normal random variable falls below it — so small negatives leak through, large negatives still get killed, but the transition is smooth instead of a sharp corner.

**What it does:** Replaces the activation function inside the MLP block. The smooth, differentiable curve gives gradients a cleaner signal to flow through during training, which is why most modern transformers (GPT-2 onward, BERT, T5) use GELU over ReLU. The exact formula is 0.5 · x · (1 + erf(x/√2)).

**Key design choices in this implementation:**
- The activation is pluggable via `config.activation` and applies uniformly to both the standard `MLP` and the `Expert` class inside the MoE layer — one setting controls all MLP-style blocks.
- Three options are supported: `nn.GELU` (default, uses PyTorch's optimised kernel), a `GELUFromScratch` class that implements the exact `0.5 * x * (1 + erf(x / sqrt(2)))` formula via `torch.erf` (not the tanh approximation used in some frameworks), and `nn.ReLU` for ablation comparisons.
- An unrecognised activation string raises `ValueError` with the valid options listed — silent fallback would hide config typos.

**Config flags:**
```python
activation: str = 'gelu'   # 'gelu' | 'relu' | 'gelu_scratch'
```

**Usage:**
```python
from model import GPTConfig, GPT

# Default — nn.GELU (fastest, recommended)
model = GPT(GPTConfig(activation='gelu'))

# Exact erf formula, pedagogical — numerically identical to nn.GELU
model = GPT(GPTConfig(activation='gelu_scratch'))

# ReLU — for ablation
model = GPT(GPTConfig(activation='relu'))
```

### RoPE — Rotary Position Embeddings (Su et al., [2021](https://arxiv.org/abs/2104.09864))

**Core idea:** Instead of *adding* a position vector to the input embedding once at the start, RoPE *rotates* each query and key vector inside attention by an angle that depends on its position. By construction, the dot product Q·Kᵀ between a query at position m and a key at position n depends only on the relative offset m − n, not the absolute positions. So the model gets relative-position awareness for free, with no learned position parameters.

**What it does:** Replaces nanoGPT's learned absolute position embeddings (`wpe`) with a precomputed rotation applied to Q and K inside `CausalSelfAttention`. The rotation is parameter-free — just fixed cos/sin tables computed from a base frequency θ. Modern LLMs (Llama, Mistral, Qwen, GPT-NeoX) all use RoPE for this reason.

**Key design choices in this implementation:**
- **Halved-pair convention:** the rotation splits each head vector into first-half and second-half and rotates them against each other — `[x1·cos − x2·sin, x1·sin + x2·cos]`. This matches the Llama/HuggingFace/GPT-NeoX convention. The original paper instead rotates adjacent pairs `(x₀,x₁), (x₂,x₃), ...`; the two are mathematically equivalent under a reordering of dimensions but the halved form is now the de-facto standard.
- **Base frequency hardcoded to 10000.0** in `CausalSelfAttention.__init__`, matching GPT-NeoX and the original paper. To use Llama-3's 500000.0 or any other value, edit the constant there directly — it is not exposed as a config field.
- **Registered buffers, not parameters:** `rope_cos` and `rope_sin` are stored via `register_buffer`, so they travel with the model on `.to(device)` but are excluded from `optimizer.param_groups` and saved state dicts without gradients.
- **`wpe` is removed entirely** when `use_rope=True` — the `nn.Embedding` for absolute positions is not added to the `transformer` ModuleDict, saving `block_size × n_embd` parameters.
- **Defensive guards:** `head_dim` must be even (asserted at init time); `crop_block_size` trims `rope_cos` and `rope_sin` alongside the causal mask so context-length surgery stays consistent; `from_pretrained` is safe by construction — it always builds a fresh `GPTConfig` with `use_rope=False` and constrains `override_args` to `dropout` only.
- **`use_rope=False` is bit-identical to the original** — the wpe path is unchanged and no code runs in the attention forward unless the flag is set.

**Config flags:**
```python
use_rope: bool = False   # True replaces wpe with RoPE; head_dim must be even
```

**Usage:**
```python
from model import GPTConfig, GPT

# Train from scratch with RoPE (no learned position embeddings)
model = GPT(GPTConfig(use_rope=True, n_embd=64, n_head=2))  # head_dim=32, even ✓

# Load pretrained GPT-2, then fine-tune with original absolute embeddings (use_rope=False)
model = GPT.from_pretrained('gpt2')   # always use_rope=False; safe by default
```

### MoE — Mixture of Experts (Switch Transformer / Mixtral; HuggingFace [blog](https://huggingface.co/blog/moe))

**Core idea:** Replace one MLP block with N independent expert MLPs and a small router that picks which experts each token visits. Each token only activates *k* of the *N* experts (e.g., 2 of 4), so total parameters grow with N but per-token compute stays roughly constant. Different experts can specialize in different patterns — punctuation, code, named entities — without any of them being forced to handle everything.

**What it does:** Replaces the `MLP` sub-layer in each transformer block with a `SparseMoE` module containing 4 experts and a top-2 softmax router. A load-balancing auxiliary loss (Switch Transformer style) is added to the training objective to keep the router from collapsing onto one or two favorite experts. The aux loss is plumbed up through `Block.last_aux_loss` and folded into the cross-entropy loss inside `GPT.forward`, so the training loop in `train.py` needs no changes.

**Key design choices in this implementation:**
- **4 experts, top-2 routing** — each token's output is the weighted sum of 2 expert outputs, with weights re-normalised from the top-2 softmax probabilities. This follows Mixtral rather than Switch Transformer's k=1; k=2 is more stable because the gradient path through the router is never a single hard choice.
- **Simple softmax router, no noisy gating** — router logits go straight to softmax with no added noise. Noisy top-k (GShard-style) adds training complexity; the aux loss alone is sufficient to prevent collapse at this scale.
- **Load-balancing aux loss:** `num_experts · Σ(fᵢ · Pᵢ)`, where `fᵢ` is the fraction of tokens hard-routed to expert i (computed from the discrete top-k mask — non-differentiable) and `Pᵢ` is the mean router softmax probability for expert i (differentiable, the gradient path). Multiplying the two pushes both the selection counts and the probabilities toward uniformity. Weighted by `moe_aux_loss_coef` (default 0.01) before being added to cross-entropy.
- **Stripped `Expert` class with no internal dropout** — each expert is `c_fc → activation → c_proj` only. A single `nn.Dropout` is applied once on the combined MoE output, avoiding double-dropout that would occur with k=2 if each expert had its own.
- **Expert dispatch is a Python loop** over `num_experts`, using `index_add_` to scatter weighted outputs back into a result tensor. `index_add_` gives well-defined gradient accumulation across PyTorch versions; the simpler boolean-index `+=` has inconsistent behaviour. Batched dispatch is a production optimisation not needed at this scale.
- **Aux loss surfaced via `Block.last_aux_loss`** — the attribute is written by `Block.forward` after each MoE call and read by `GPT.forward` after the block loop. This keeps `Block.forward`'s return type as just `x` and leaves `train.py` completely unchanged.
- **`config.activation` cascades into experts** — the same `'gelu'` / `'relu'` / `'gelu_scratch'` flag used for the standard MLP is respected inside each `Expert`, so activation choice is consistent across the whole model.
- **`from_pretrained` guard** — loading GPT-2 weights into a MoE model is blocked by an explicit `assert not config.use_moe` after the config is constructed, since the weight shapes are incompatible.

**Config flags:**
```python
use_moe: bool = False            # replace MLP with SparseMoE in every block
num_experts: int = 4             # total experts per layer
moe_top_k: int = 2              # experts activated per token
moe_aux_loss_coef: float = 0.01  # weight of load-balancing loss added to CE
```

**Usage:**
```python
from model import GPTConfig, GPT

# MoE model — 4 experts, top-2 routing
cfg = GPTConfig(use_moe=True, num_experts=4, moe_top_k=2, moe_aux_loss_coef=0.01)
model = GPT(cfg)

# Forward pass with targets: returned loss already includes aux contribution
logits, loss = model(x, targets=y)   # loss = CE + 0.01 * load_balance_aux
loss.backward()                       # train.py needs no changes
```

### LoRA — Low-Rank Adaptation (Hu et al., [2021](https://arxiv.org/abs/2106.09685))

**Core idea:** During fine-tuning, you don't need to update *every* weight in a giant matrix W. The actual update ΔW that adapts a pretrained model to a new task tends to live in a tiny subspace — it's *low rank*. So instead of training ΔW directly (millions of parameters), LoRA trains a low-rank factorization ΔW ≈ B·A where A and B are skinny matrices of rank r ≪ min(d_in, d_out). The base weight W stays frozen; only A and B are learned. For our default rank-8 attention LoRA on GPT-2 base, this works out to **442,368 trainable parameters out of 124,882,176 total — just 0.35%**.

**What it does:** Wraps targeted `nn.Linear` layers in a `LoRALinear` adapter that adds a small trainable update on the side: `y = W·x + (α/r) · B(A(x))`. The base weight is frozen on wrap; only A and B carry gradient. A is initialized with Kaiming uniform (standard linear init) but B is initialized to **zero** — so at step 0, B·A = 0 and the model's output is bit-identical to the pretrained base. Training perturbs B away from zero only as the task demands.

**Key design choices in this implementation:**
- **Default targets are `attn.c_attn` and `attn.c_proj`** — the fused QKV projection and attention output projection, matching the paper's recommendation. MLP layers and `lm_head` are not targeted by default; `lm_head` is intentionally excluded because it shares weights with `wte` via weight tying.
- **Path-suffix matching** in `apply_lora` disambiguates `attn.c_proj` from `mlp.c_proj` — both layers are named `c_proj`, so matching on the full dotted suffix rather than just the leaf name is necessary to target attention only.
- **Two-call forward** — `x → F.linear(x, lora_A) → F.linear(..., lora_B)` — rather than materialising `lora_B @ lora_A` first. Peak intermediate memory is O(r) not O(d_in · d_out); at rank 8 on GPT-2's 768-dim layers that is a 96× reduction in the intermediate allocation.
- **Zero-init of B verified at runtime:** running a forward pass immediately after `apply_lora` on a freshly constructed model produces max absolute diff of `0.00` versus the unwrapped model, confirming the adapter is a true no-op at initialisation.
- **`apply_lora` asserts every target was matched** — if a name in `lora_target_modules` matches nothing, an `AssertionError` is raised listing all available linear layer paths. A silent no-op from a typo would otherwise train zero LoRA parameters with no warning.
- **`freeze_non_lora` prints the trainable breakdown** in the format `LoRA: {trainable:,} trainable / {total:,} total ({pct:.2f}%)` immediately after freezing, so the parameter budget is always visible at the start of a fine-tuning run.
- **`save_lora_only` / `load_lora` for tiny adapter checkpoints** — `save_lora_only` filters the state dict to only `lora_A` and `lora_B` keys, producing a file at KB scale rather than hundreds of MB for a full GPT-2 checkpoint. `load_lora` restores these into a model that already has `LoRALinear` layers applied, using `strict=False` and asserting that no LoRA keys are missing or unexpected.
- **`apply_lora` is an explicit separate step**, not auto-triggered by `config.use_lora` inside `GPT.__init__`. This preserves the natural fine-tuning workflow and avoids entangling weight loading with adapter insertion.
- **Composes cleanly with MoE** — because the default targets only cover attention projections, `apply_lora` does not wrap `Expert` layers inside a `SparseMoE` block. Both features can be enabled together without conflict.

**Config flags:**
```python
use_lora: bool = False                                        # informational; apply_lora() does the actual wrapping
lora_rank: int = 8                                            # rank r of A and B matrices
lora_alpha: float = 16.0                                      # scale factor: delta weighted by alpha/rank
lora_target_modules: tuple = ('attn.c_attn', 'attn.c_proj')  # path-suffixes of linears to wrap
```

**Usage:**
```python
from model import GPTConfig, GPT, apply_lora, freeze_non_lora, save_lora_only, load_lora

# Full fine-tuning workflow
model = GPT.from_pretrained('gpt2')                 # load pretrained base
cfg = GPTConfig(lora_rank=8, lora_alpha=16.0)
apply_lora(model, cfg)                              # graft LoRA adapters in-place
freeze_non_lora(model)                              # prints: LoRA: 442,368 / 124,882,176 (0.35%)
# ... fine-tune with your training loop (train.py unchanged) ...

# Save only the tiny adapter (KB, not MB)
save_lora_only(model, 'lora_adapter.pt')

# Restore: load base fresh, apply structure, load adapter weights
model2 = GPT.from_pretrained('gpt2')
apply_lora(model2, cfg)
load_lora(model2, 'lora_adapter.pt')
```
