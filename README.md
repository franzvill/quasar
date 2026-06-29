# Quasar

**A 30-billion-parameter language model running at ~46 tokens/sec on a 24 GB Mac.**

Quasar is a small, self-contained inference engine — ~3,000 lines of C, Objective-C,
and Metal with **no third-party dependencies** — purpose-built for a single model:
**Qwen3-30B-A3B**. It takes the ideas of [`antirez/ds4`](https://github.com/antirez/ds4)
(a DeepSeek-V4 engine for 96–128 GB machines) and applies them to a model that fits a
laptop.

```console
$ ./quasar generate model.gguf "Q: What is the capital of Japan? A:" 12 metal
Q: What is the capital of Japan? A: Tokyo
A: The capital of Japan is Tokyo.
```

## The idea

A 30 B Mixture-of-Experts model only activates **~3.3 B parameters per token**. Decode
speed on Apple Silicon is bounded by `memory_bandwidth / active_bytes_per_token`, so the
whole game is making those active bytes small — then keeping the GPU busy instead of
waiting on the CPU:

- **Asymmetric quantization** (the ds4 trick). The routed experts are ~95 % of the
  weights, so crush *only them* to ~2-bit (Q2_K) and keep attention, router, embeddings,
  and norms at higher precision. The model drops from **18.6 GB → 10.5 GB** — it now fits
  24 GB with room for the KV cache, so there's no SSD paging.
- **Fully GPU-resident forward.** The whole token — all 48 layers of attention + MoE plus
  the final projection — runs in **one Metal command buffer**, with activations and the
  KV cache never leaving the GPU. One CPU↔GPU sync per token instead of ~1,400.

The result is a 30 B-class model decoding at interactive speed on a laptop-class chip.

## Quickstart

Requires macOS with the Xcode command-line tools (for the Metal compiler) and an Apple
Silicon GPU. ~12 GB of free RAM for the 2-bit model; ~30 GB of disk for both GGUFs.

```sh
# 1. Build
make

# 2. Fetch the base model (Q4_K_M, ~17 GB) and make the 2-bit version (~10.5 GB)
mkdir -p gguf
curl -L -C - -o gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
./quasar requant gguf/Qwen3-30B-A3B-Q4_K_M.gguf gguf/quasar-q2k.gguf

# 3. Generate on the GPU
./quasar generate gguf/quasar-q2k.gguf "Once upon a time" 200 metal
```

No model on hand? `make test-gguf` synthesizes a tiny but valid `qwen3moe` file and
inspects it — no download required.

## Speed

Steady-state decode throughput, measured on an **Apple M5 Pro** with the 2-bit model.
Each row is a milestone that removed a specific bottleneck:

| Engine path | tok/s | what it fixed |
|---|---:|---|
| CPU reference (scalar f32) | ~0.1 | — (correctness oracle, not for speed) |
| Metal, model too big for 24 GB | — | 72 s/forward — constant SSD paging |
| 2-bit quant + KV cache | 9.7 | fits RAM; O(T) decode instead of O(T²) |
| + MoE block on GPU | 13.9 | router/top-8/experts in one command buffer |
| + attention block on GPU | 27.6 | rmsnorm/QK-norm/RoPE/attention on GPU |
| **+ whole token in one command buffer** | **46.1** | one sync/token instead of ~1,400 |

## How it works

Quasar is **not** a generic GGUF runner. It is hardcoded to the Qwen3-30B-A3B
architecture (48 layers, 2048 hidden, 32/4 GQA heads, head_dim 128, 128 experts / 8 used,
per-head QK-norm, NeoX RoPE) and validates a loaded model against that profile.

| Component | File |
|---|---|
| GGUF v3 reader (mmap, metadata, tensor traits) | `gguf.{h,c}` |
| Dequant Q4_K/Q6_K/Q8_0/Q2_K/F16 + a Q2_K encoder | `quant.{h,c}` |
| Qwen3 byte-level BPE tokenizer + ChatML template | `tokenizer.{h,c}`, `chat.{h,c}` |
| Model loader + profile validation + CLI | `quasar.{h,c}` |
| f32 CPU reference forward (the correctness oracle) | `forward_cpu.{h,c}` |
| Metal backend: kernels + GPU-resident forward | `metal.{h,m}` |
| Asymmetric 2-bit requantizer (GGUF writer) | `requant.{h,c}` |

**Correctness discipline.** The f32 CPU forward is the oracle. Every Metal kernel is
gated against it — `metal-selftest` checks the matmul kernels to ~1e-7, and end to end
`"The capital of France is"` must keep `" Paris"` as the top token. The Metal path
reproduces the CPU logits exactly. (The one real bug along the way — a `uint32` weight
offset overflowing past 4 GB into NaN — was caught precisely because of this gate.)

For the full design, the exact shape profile, the memory budget, and the roadmap, see
**[ARCHITECTURE.md](ARCHITECTURE.md)**; for the build log and the speed journey, see
**[PROGRESS.md](PROGRESS.md)**.

## Commands

```
quasar inspect       <model.gguf>                      load + validate a qwen3moe GGUF
quasar tokenize      <model.gguf> "text"               encode -> token ids (+ round-trip)
quasar detokenize    <model.gguf> <id>...              decode token ids -> text
quasar render        [model.gguf]                      show the Qwen3 ChatML template
quasar generate      <model.gguf> "text" [n] [metal]   forward: top-5 + greedy n tokens
quasar metal-selftest [model.gguf]                     validate Metal kernels vs CPU
quasar requant       <in.gguf> <out.gguf>              crush routed experts to 2-bit (Q2_K)
```

`generate` runs on the CPU reference path unless you add `metal`.

## Status

Beta. Runs Qwen3-30B-A3B end to end at ~46 tok/s with correct, coherent output. The
2-bit model is requantized from Q4_K_M (lossy-on-lossy) — coherent and reliable on
factual/short prompts; imatrix-calibrated 2-bit from FP16 is future work.

**Roadmap:** temperature / min-p sampling + an interactive CLI chat loop · an
OpenAI-compatible HTTP server with tool calling (to drop into coding agents) ·
imatrix-calibrated 2-bit / IQ2_XXS experts · KV-on-disk + SSD expert streaming.

## Acknowledgements

The concept and approach follow [`antirez/ds4`](https://github.com/antirez/ds4)
(DwarfStar). The GGUF format, the k-quant block layouts, and a great deal of hard-won
local-inference engineering come from [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and GGML. Quasar does not link against GGML, but it would not exist without it.

## License

MIT — see [LICENSE](LICENSE).
