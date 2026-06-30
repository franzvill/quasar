# Quasar — progress log

A from-scratch C + Metal inference engine for **Qwen3-30B-A3B**, applying the
ideas of [`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar, a
DeepSeek-V4 engine for 96–128 GB machines) to a model that fits a **24 GB Mac**.

## Why this exists

ds4 is hardcoded to DeepSeek V4 and its smallest model is ~81 GB on disk — it
cannot run on 24 GB. So instead of "applying ds4 to another model" (which would
mean rewriting its engine anyway), we rebuilt the *concept* from scratch around a
model that fits: a 30 B Mixture-of-Experts with only **~3.3 B active params per
token**, plus an **asymmetric quant** that crushes only the routed experts.

## What was built (session of 2026-06-29)

All verified on the real Qwen3-30B-A3B weights.

| # | Milestone | Status | Evidence |
|---|---|---|---|
| 1 | GGUF loader + `inspect` + shape-profile validation | ✅ | 579 tensors, canonical profile, 3.04 B/30.53 B active |
| 2 | Qwen3 byte-level BPE tokenizer + ChatML template | ✅ | `Hello world` → `[9707, 1879]` (exact Qwen ids); round-trips on ASCII/CJK/emoji |
| 3 | CPU reference forward (f32) — the correctness oracle | ✅ | `"The capital of France is"` → **" Paris"** |
| 5 | Metal dequant-matvec kernels (F32/Q4_K/Q6_K/Q2_K) | ✅ | match CPU to ~1e-7; full-forward parity |
| 7 | **Asymmetric 2-bit expert quant** (`requant`) | ✅ | 18.6 GB → **10.5 GB**, fits 24 GB, still coherent |
| 8 | KV cache → incremental decode | ✅ | identical output, O(T²)→O(T) |
| 6 | Batched GPU dispatch (q/k/v, gate+up, down grouped) | ◐ | ~1392→240 syncs/token; full on-GPU glue still todo |
| 9 | Sampling (temp / top-k / top-p / min-p / repeat penalty) | ✅ | greedy still exact; `seed` reproducible |
| 10 | **OpenAI-compatible HTTP server** (`serve`) + SSE | ✅ | `/v1/chat/completions`, `/v1/completions`, `/v1/models` |

## Speed journey (14-token generation, M5 Pro, 2-bit model)

| Stage | Time | What it unlocked |
|---|---:|---|
| Metal, Q4_K_M (18.6 GB) | ~72 s/fwd | — (SSD paging: model didn't fit in 24 GB) |
| + 2-bit quant → 10.5 GB (#7) | 69 s | fits RAM, paging eliminated (system time 31.9 s → 1.5 s) |
| + KV cache (#8) | 9.7 s | each token is one incremental step, not a full re-run |
| + batched dispatch (#6) | **4.3 s** | far fewer blocking CPU↔GPU round-trips |

Decode tokens/sec (steady-state, startup amortized), M5 Pro, 2-bit model:

| Stage | tok/s |
|---|---:|
| KV-cache hybrid | 9.7 |
| + GPU MoE block (router/top-8/experts/SwiGLU in one command buffer) | 13.9 |
| + GPU attention block (rmsnorm/QK-norm/RoPE/attention in one command buffer) | 27.6 |
| + full resident forward (whole token in ONE command buffer, 1 sync/token) | **46.1** |

The whole token — all 48 layers of attention + MoE + residuals + final norm/output
— now runs in a single GPU command buffer (1 sync/token, down from ~1392 dispatches
in the first naive hybrid), with cur and the KV cache GPU-resident; the CPU only
embeds the token and reads the logits. Output stays exact vs the CPU oracle ("Paris") and
coherent, e.g. `The meaning of life is a question that has long been pondered by
philosophers, theologians, and scientists`.

Gotcha that bit us: weight byte-offsets in kernel arg structs must be **uint64** —
a norm tensor >4 GB into a chunk overflowed a uint32 `w_off`, read garbage, and
produced NaN logits (output collapsed to "!!!!"). The matmul/MoE paths were already
uint64, so only the new attention-norm structs were affected.

## Serving it (session of 2026-06-30)

With decode fast and resident, the engine became usable as a drop-in local API.
Two new zero-dependency modules, then an HTTP layer:

- **`sample.{h,c}`** — greedy / temperature / top-k / top-p (nucleus) / min-p +
  repetition penalty, with a seedable xorshift RNG. To avoid sorting all ~152k
  logits per token, it prunes to a post-temperature 30-nat window first (anything
  below contributes <1e-13 to the softmax) and sorts only the survivors. `temp ≤ 0`
  is exact greedy, so the `" Paris"` parity check still holds.
- **`json.{h,c}`** — a small recursive-descent JSON parser (objects/arrays, all
  escapes incl. `\uXXXX` surrogate pairs) for request bodies, plus a growable
  string builder + escaper for responses.
- **`server.{h,c}`** — a minimal HTTP/1.1 server (`quasar serve`): `/v1/chat/
  completions` (ChatML render → tokenize → decode loop) with **SSE streaming**,
  `/v1/completions`, `/v1/models`, `/health`. Requests run **serially** (one model
  / KV cache / GPU pipeline). The model is loaded once and warmed up at startup so
  the first request is fast. Streaming holds back bytes on UTF-8 char boundaries
  (so multibyte never splits across chunks) and on the longest `stop` string (so a
  stop sequence is never partially emitted).

Verified end to end against the 2-bit model on Metal: non-streaming chat returns
`"Tokyo."` / `"…**Paris**."` with correct `usage`; streaming emits the role chunk,
content deltas, the `finish_reason` chunk, then `data: [DONE]`; `stop`, `seed`,
Japanese (multibyte) output, the legacy completion endpoint, and the 400/404/405
error paths all behave. Sustained ~33 tok/s for a 200-token response (the ~46 tok/s
figure is the pure marginal decode rate; per-request wall time also includes
prefill).

## Key technical decisions & gotchas

- **Not a generic GGUF runner.** Hardcoded to the `qwen3moe` arch / Qwen3-30B-A3B
  shape profile (see `ARCHITECTURE.md`). It validates the loaded model against it.
- **The CPU f32 forward is the oracle.** Every Metal kernel and quant change is
  gated by matching its logits (the `metal-selftest` command checks kernels to
  ~1e-7; end-to-end we check `"The capital of France is" → " Paris"`).
- **Metal buffer limit.** The M5 Pro reports `maxBufferLength` = 14.3 GB, below
  the 18.6 GB model. We wrap the mmap as several **overlapping** zero-copy
  (`newBufferWithBytesNoCopy`) chunks (≤12 GB each, 1 GB overlap ≫ any tensor) and
  route each gemv to the chunk that fully contains its tensor. The GPU reads
  mmap'd file pages fine once the buffer is valid — the earlier "all-zeros" output
  was a nil buffer (size > limit), not a paging/residency problem.
- **Asymmetric quant = the ds4 trick.** Only `*_exps.weight` tensors (the routed
  experts, ~29 B of 30.5 B params) are requantized to Q2_K; attention, router,
  embeddings, norms, and output keep their original precision. → 10.5 GB.
- **Q2_K encoder from scratch.** Per-16 sub-block min/scale, super-scaled to 4
  bits; layout matches ggml so dequant agrees. Quality is fine for the demo
  (requant is Q4_K→Q2_K, i.e. lossy²; from F16 would be better).
- **KV cache → incremental decode.** Per-layer persistent K/V cache; `quasar_decode`
  processes new tokens one at a time through all layers, appending K/V and
  attending over the cache.
- **Hashmap capacity must be a power of two** (the tokenizer's open-addressing map
  masks with `cap-1`); a non-power-of-two cap caused an infinite loop early on.

## Working commands

```sh
make                                              # build ./quasar (Metal on macOS)
make test-gguf                                    # tiny synthetic qwen3moe, no download

./quasar inspect      gguf/Qwen3-30B-A3B-Q4_K_M.gguf
./quasar tokenize     gguf/Qwen3-30B-A3B-Q4_K_M.gguf "Hello world"
./quasar metal-selftest gguf/Qwen3-30B-A3B-Q4_K_M.gguf
./quasar requant      gguf/Qwen3-30B-A3B-Q4_K_M.gguf gguf/q2k.gguf   # make the 2-bit model
./quasar generate     gguf/q2k.gguf "Once upon a time" 40 metal      # generate on the GPU
./quasar serve        gguf/q2k.gguf --port 8080 metal                # OpenAI-compatible API server
```

## Done since the initial roadmap

- ✅ **#6 fully GPU-resident forward** — the glue (RMSNorm, QK-norm, RoPE, attention,
  softmax, MoE) is on Metal; a whole token runs in one command buffer → **46 tok/s**.
- ✅ **Temperature / top-k / top-p / min-p sampling** (`sample.{h,c}`).
- ✅ **OpenAI-compatible HTTP server** with SSE streaming (`serve`).

## Remaining roadmap

- Qwen3 **tool calling** over the server (parse `<tool_call>` blocks → `tool_calls`).
- An interactive CLI chat loop (keeps the KV cache warm between turns).
- KV-on-disk + SSD expert streaming (ds4's RAM↔SSD spectrum).
- Quality: imatrix-calibrated 2-bit from F16 source; IQ2_XXS for gate/up.
