# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

**Quasar** is a small, self-contained **C + Metal inference engine purpose-built
for one model: Qwen3-30B-A3B**, optimized for Apple Silicon on a 24 GB Mac. It is
*not* a generic GGUF runner. It re-applies the concept of `antirez/ds4` (a
DeepSeek-V4 engine) to a model that fits a laptop: a 30 B MoE with **~3.3 B active
params/token**, with the routed experts quantized to ~2-bit while everything else
stays high-precision.

- **Design + roadmap:** `ARCHITECTURE.md` (authoritative — the in-tool task list
  does not persist across sessions).
- **What's been done + speed numbers:** `PROGRESS.md`.

No external dependencies. C11 + Objective-C (Metal). macOS only for the GPU path.

## Build & run

```sh
make                 # build ./quasar (links -framework Metal -framework Foundation on macOS)
make test-gguf       # generate a tiny synthetic qwen3moe GGUF and inspect it (no model download)
make clean

# Commands (run ./quasar --help for the list):
./quasar inspect       <model.gguf>                  # load + validate against the shape profile
./quasar tokenize      <model.gguf> "text"           # encode -> ids (+ round-trip check)
./quasar detokenize    <model.gguf> <id>...          # decode ids -> text
./quasar render        [model.gguf]                  # show the Qwen3 ChatML template
./quasar generate      <model.gguf> "prompt" [n] [metal]  # forward: top-5 + greedy n tokens; add 'metal' for GPU
./quasar metal-selftest [model.gguf]                 # validate Metal gemv kernels vs the CPU oracle
./quasar requant       <in.gguf> <out.gguf>          # crush routed experts to Q2_K (the asymmetric quant)
```

The binary always links Metal; CPU vs GPU is chosen per run (`generate` without
`metal` uses the scalar CPU path — slow but the reference). There is no separate
CPU-only build target.

## Getting the model

The Q4_K_M GGUF is the validation model; `requant` produces the 2-bit one we
actually run fast.

```sh
mkdir -p gguf
curl -L -C - -o gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
./quasar requant gguf/Qwen3-30B-A3B-Q4_K_M.gguf gguf/Qwen3-30B-A3B-quasar-q2k.gguf
```

Model files live in `gguf/` and are git-ignored.

## Code map

| File | Role |
|---|---|
| `gguf.{h,c}` | Self-contained GGUF v3 reader (mmap, metadata KV, tensor traits, array iterators) |
| `quant.{h,c}` | Dequant F32/F16/Q8_0/Q4_0/Q4_K/Q6_K/Q2_K + a Q2_K **encoder** |
| `quasar.{h,c}` | Model loader + shape-profile validation; CLI (`main`) for all commands |
| `tokenizer.{h,c}` | Qwen3 byte-level BPE (vocab/merges/special tokens, pre-tokenizer, decode) |
| `chat.{h,c}` | Qwen3 ChatML prompt rendering (thinking on/off) |
| `forward_cpu.{h,c}` | The forward pass + KV cache + incremental decode; `gemv` dispatches CPU or Metal |
| `metal.{h,m}` | Metal backend: chunked zero-copy weight buffers + dequant-matvec kernels (single + batched) |
| `requant.{h,c}` | GGUF writer: copy everything, requantize `*_exps` tensors to Q2_K |
| `tools/make_test_gguf.py` | Synthesizes a tiny but valid qwen3moe GGUF for testing without a download |

Data flow for decode: `tokenizer` → `forward_cpu.quasar_decode` (per token, per
layer: RMSNorm → q/k/v via `gemv` → QK-norm + RoPE → attention over KV cache →
o-proj → MoE router/top-8 → experts via `gemv` → residual) → final norm → logits.
The heavy matmuls go through `gemv`, which calls `metal.m` when `use_metal` is set.

## The Qwen3-30B-A3B shape profile (what the engine is built around)

48 layers · hidden 2048 · 32 attn heads / 4 KV heads · **head_dim 128** (decoupled
from hidden) · vocab 151936 · **128 experts, 8 used/token** · MoE intermediate 768 ·
ctx 40960 · RoPE θ=1e6 (**NeoX** style) · RMSNorm eps 1e-6 · bos 151643 / eos 151645.

Qwen3 specifics that bite if missed: **per-head QK RMSNorm** (`attn_q_norm`,
`attn_k_norm`, length head_dim, applied before RoPE), **no QKV bias**, **no shared
expert**, MoE gating = softmax over all experts → top-8 → renormalize.

## Conventions & gotchas

- **Correctness gate (do this for every backend/quant change):** the CPU f32
  forward is the oracle. `metal-selftest` checks kernels to ~1e-7 against it;
  end-to-end, `generate "The capital of France is"` must keep **" Paris"** as the
  top token. Don't trust a kernel that hasn't been diffed against the CPU path.
- **Metal buffer limit.** The model (18.6 GB) exceeds `maxBufferLength` (14.3 GB on
  M5 Pro). `metal.m` wraps the mmap as overlapping ≤12 GB chunks (1 GB overlap, far
  larger than any single tensor) and routes each gemv to the containing chunk. Keep
  the overlap ≥ the largest tensor.
- **Asymmetric quant** touches only tensors whose name contains `_exps.weight`.
- **GGUF specifics:** linear weights are stored `[in, out]` in `ne` order (ne0 =
  in-features, row-major with the row contiguous); MoE expert stacks are 3D
  `[in, out, n_expert]` with experts laid out consecutively. K-quant layouts match
  ggml exactly (block sizes in `gguf.c`'s trait table and the dequant routines).
- **Hashmap capacity must be a power of two** (open-addressing probe masks with
  `cap-1`) — see `tokenizer.c`.
- **Background-friendly:** a forward over the full model is slow on the CPU path
  and warms ~10–18 GB of mmap; run long generations in the background.

## When extending

Add a new milestone by implementing it behind the CPU oracle first, then porting to
Metal and gating on parity. The current next step (`ARCHITECTURE.md` #6) is the
fully GPU-resident forward: move RMSNorm/QK-norm/RoPE/attention/softmax/MoE into
Metal kernels and run a whole token in one command buffer (today the glue is on the
CPU and only the matmuls are on the GPU).
