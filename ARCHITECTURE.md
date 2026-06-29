# Quasar

A small, self-contained native inference engine, **purpose-built for one model:
Qwen3-30B-A3B**, optimized first for **Apple Silicon / Metal** on memory-constrained
Macs (target: 24 GB M5).

This project is a deliberate re-application of the ideas in
[`antirez/ds4`](https://github.com/antirez/ds4) ("DwarfStar", built for DeepSeek
V4 Flash on 96–128 GB machines) to a model and a memory class that fit a normal
laptop. We keep the *concept*; we change the *target*.

## The concept (inherited from ds4) and how it maps here

| ds4 pillar | Quasar |
|---|---|
| One model, hardcoded; **not** a generic GGUF runner | Hardcoded to `qwen3moe` (Qwen3-30B-A3B) |
| MoE: huge total params, **tiny active** params → fast decode | 30.5B total / **~3.3B active** per token |
| **Asymmetric quant**: crush only routed experts to low-bit, keep the rest high | Experts → Q2_K/IQ2_XXS (~29B params); attention/router/embed/norms → Q8_0/F16 |
| KV cache is a first-class **disk** citizen (RAM + SSD) | KV in RAM, persisted to disk, prompt-cache reuse |
| **SSD streaming** of experts → RAM becomes a speed spectrum | Resident non-routed weights + in-RAM expert cache, stream experts on miss |
| End-to-end: CLI + HTTP server + tool calling + agent | Same, scoped |
| **Logit-validated** correctness gate | Compare against HF/llama.cpp reference logits |
| Compressed KV attention (CSA/HCA) — *DeepSeek-specific* | N/A — Qwen3 uses plain GQA (KV already small) |

The single biggest lever for "crazy fast" is the same as ds4's: **only ~3.3B
parameters are touched per generated token**, even though the model is 30B. Decode
speed on Apple Silicon is bounded by `memory_bandwidth / active_bytes_per_token`,
so a 3B-active model at ~2-bit experts moves very little memory per token.

## Target model: Qwen3-30B-A3B (the "shape profile")

These are the fixed architecture constants the engine is built around. The loader
reads them from GGUF metadata and validates against this canonical profile.

| Field | Value | GGUF key |
|---|---:|---|
| Arch | `qwen3moe` | `general.architecture` |
| Layers | 48 | `qwen3moe.block_count` |
| Hidden size | 2048 | `qwen3moe.embedding_length` |
| Attention heads | 32 | `qwen3moe.attention.head_count` |
| KV heads (GQA) | 4 | `qwen3moe.attention.head_count_kv` |
| Head dim | 128 (decoupled from hidden) | `qwen3moe.attention.key_length` |
| Vocab | 151936 | `qwen3moe.vocab_size` |
| Experts | 128 | `qwen3moe.expert_count` |
| Experts used/token | 8 | `qwen3moe.expert_used_count` |
| MoE intermediate | 768 | `qwen3moe.expert_feed_forward_length` |
| Context (train) | 40960 | `qwen3moe.context_length` |
| RoPE theta | 1,000,000 | `qwen3moe.rope.freq_base` |
| RMSNorm eps | 1e-6 | `qwen3moe.attention.layer_norm_rms_epsilon` |
| Tie embeddings | false (separate `output.weight`) | — |
| norm_topk_prob | true (renormalize top-8 gate weights) | — |

Per-layer flow (the CPU reference, then the Metal port, implement exactly this):

```
x' = rmsnorm(x, attn_norm)
q  = Wq x'  -> [32,128];  per-head q_norm (rmsnorm over 128); RoPE
k  = Wk x'  -> [ 4,128];  per-head k_norm (rmsnorm over 128); RoPE
v  = Wv x'  -> [ 4,128]
a  = causal GQA attention (scale 1/sqrt(128), each KV head shared by 8 Q heads)
x  = x + Wo a
x' = rmsnorm(x, ffn_norm)
g  = softmax(Wrouter x') over 128 experts; pick top-8; renormalize to sum 1
y  = sum_e g_e * down_e( silu(gate_e x') * (up_e x') )
x  = x + y
```
Final: `logits = output * rmsnorm(x, output_norm)`.

Notable Qwen3 specifics vs Qwen2/Llama: **per-head QK RMSNorm** (`attn_q_norm`,
`attn_k_norm`, length = head_dim), **no QKV bias**, **head_dim 128 ≠ hidden/heads**,
**no shared expert**.

## Memory budget on a 24 GB M5

macOS + apps take ~6–8 GB; bump the GPU wired limit so weights+KV stay resident:
`sudo sysctl iogpu.wired_limit_mb=20480`.

Param split (why the asymmetric quant works): routed experts are
`128 × 3 × 2048 × 768 × 48 ≈ 29.0B` params — ~95% of the model. Everything else
(attention, router, embeddings, norms, output) is ~1.5B.

| Component | Quant | ~Size |
|---|---|---:|
| Routed experts (~29B) | IQ2_XXS gate/up + Q2_K down | ~8–9 GB |
| Attention + router + norms (~0.9B) | Q8_0 | ~1.0 GB |
| token_embd + output (~0.6B) | Q8_0 / F16 | ~0.7–1.2 GB |
| **Weights total** | | **~10–11 GB** |
| KV cache (GQA: 96 KB/token f16; ~48 KB q8) | f16/q8 | ~1.5–3 GB @ 32K ctx |
| Graph scratch / activations | | ~1 GB |

→ Comfortably under ~16 GB; room to push context. Bigger contexts or higher expert
precision spill gracefully to SSD streaming instead of OOM.

## File layout

```
gguf.{h,c}      self-contained GGUF v3 parser (mmap, metadata, tensor traits)   [done]
quasar.{h,c}    hparams + model loader/validator + `inspect` CLI                 [done]
tokenizer.{h,c} Qwen3 byte-level BPE (vocab/merges/specials, pretok, decode)     [done]
chat.{h,c}      Qwen3 ChatML prompt rendering (thinking on/off)                  [done]
forward_cpu.{h,c} f32 reference forward pass (correctness oracle)                [done]
quant.{h,c}     dequant Q4_K/Q6_K/Q8_0/Q4_0/F16/Q2_K + Q2_K encoder              [done]
requant.{h,c}   GGUF writer: crush routed experts to Q2_K (the ds4 trick)        [done]
metal.{h,m}     Metal backend: chunked unified buffers + dequant-matvec kernels  [done; resident fwd #6]
kv.c            RAM KV cache + on-disk persistence                              [todo #8,10]
server.c        OpenAI-compatible HTTP server + tool calling                    [todo #9]
tools/          GGUF generation + test helpers                                  [partial]
```

## Build & run

```sh
make                                  # builds ./quasar (CPU-side tools so far)
make test-gguf                        # synthesize a tiny qwen3moe GGUF + inspect it
./quasar inspect ./qwen3-30b-a3b.gguf # validate a real model against the profile
```

## Milestones

1. ✅ Scaffold + GGUF loader + model inspector
2. ✅ Qwen3 BPE tokenizer + chat template (pretokenizer Unicode classes to be validated vs real vocab)
3. ✅ CPU reference forward pass (f32) — validated: "The capital of France is" → " Paris"
4. ⬜ Logit validation harness (correctness gate)
5. ✅ Metal core kernels — dequant-matvec F32/Q4_K/Q6_K, validated exact vs CPU; full-forward parity (" Paris")
6. ◐ Fast Metal forward — batched dispatch groups (q/k/v, gate+up, down): ~1392→~240 syncs/token, 14-tok gen 9.7s→4.3s. Full on-GPU glue (norm/attn/rope/MoE in one command buffer) still todo for peak t/s
7. ✅ Asymmetric 2-bit expert quant — 18.6GB→10.5GB (fits 24GB); Metal fwd 72s→4.2s; still " Paris"
8. ✅ KV cache — incremental decode, ~7x faster generation (14 tok: 69s→9.7s), identical output; (temp/min_p sampling + CLI chat loop still todo)
9. ⬜ HTTP server + tool calling
10. ⬜ KV-on-disk + SSD expert streaming

## Acknowledgements

Concept and approach follow `antirez/ds4` (DwarfStar). The GGUF format, ggml quant
layouts, and the broader local-inference ecosystem come from
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) / GGML.
