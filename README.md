# Quasar

**A 30-billion-parameter language model running at ~46 tokens/sec on a 24 GB Mac.**

Quasar is a small, self-contained inference engine — ~4,600 lines of C, Objective-C,
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

## Use it as an OpenAI API server

`quasar serve` exposes an OpenAI-compatible HTTP API, so existing OpenAI clients and
coding agents can talk to the local model unchanged:

```sh
./quasar serve gguf/quasar-q2k.gguf --port 8080 metal
```

```console
$ curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of Japan? Answer in one word."}]}'
{"id":"chatcmpl-...","object":"chat.completion","choices":[{"index":0,
 "message":{"role":"assistant","content":"Tokyo."},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":24,"completion_tokens":3,"total_tokens":27}}
```

- **Endpoints:** `POST /v1/chat/completions` (ChatML, with SSE streaming when
  `"stream": true`), `POST /v1/completions`, `GET /v1/models`, `GET /health`.
- **Sampling:** `temperature` (0 = greedy), `top_p`, plus `top_k` / `min_p` /
  `repeat_penalty` / `seed` extensions; `max_tokens` and `stop` sequences honored.
- Point any OpenAI client at `http://127.0.0.1:8080/v1` with any API key. Requests
  are served **serially** — one model, one KV cache, one GPU pipeline.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="quasar")
print(client.chat.completions.create(
    model="quasar", messages=[{"role": "user", "content": "Hello!"}]).choices[0].message.content)
```

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

### vs. a production engine (same machine, same model)

For reference, [LM Studio](https://lmstudio.ai)'s MLX backend running the *same model* on
the *same* 24 GB M5 Pro:

| Engine | Quant | On disk | Fits 24 GB? | Decode |
|---|---|---:|:--:|---:|
| **Quasar** | 2-bit (asymmetric) | 10.5 GB | ✅ | **~40–46 tok/s** |
| MLX · LM Studio | 3-bit | 12.5 GB | ✅ | ~18 tok/s |
| MLX · LM Studio | 4-bit | 17 GB | ❌ swaps | ~8 tok/s |

The 4-bit build overcommits 24 GB and thrashes (~12 GB pushed into the memory compressor
plus swap), so its throughput collapses — which is *precisely* the problem the asymmetric
2-bit quant exists to avoid. At a footprint that actually fits the machine, Quasar decodes
**~2× faster than the production framework**.

In fairness, the caveats: it isn't an identical recipe (Quasar's asymmetric scheme moves a
*comparable* number of bytes per token to MLX-3bit rather than simply using fewer bits),
MLX's 3-bit path is likely less tuned than its common 4-bit, and LM Studio adds some server
overhead. So this is a real result **on this hardware at this footprint** — not a claim that
a hand-rolled engine beats MLX in general.

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
| Sampler (temp / top-k / top-p / min-p) | `sample.{h,c}` |
| OpenAI-compatible HTTP server + JSON | `server.{h,c}`, `json.{h,c}` |

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
quasar serve         <model.gguf> [--port N] [metal]   OpenAI-compatible HTTP server (+ SSE)
```

`generate` runs on the CPU reference path unless you add `metal`.

## Status

Beta. Runs Qwen3-30B-A3B end to end at ~46 tok/s with correct, coherent output. The
2-bit model is requantized from Q4_K_M (lossy-on-lossy) — coherent and reliable on
factual/short prompts; imatrix-calibrated 2-bit from FP16 is future work.

Sampling (temperature / top-k / top-p / min-p) and an OpenAI-compatible HTTP server
with SSE streaming are built; point any OpenAI client at it.

**Roadmap:** Qwen3 tool calling over the server · an interactive CLI chat loop ·
imatrix-calibrated 2-bit / IQ2_XXS experts · KV-on-disk + SSD expert streaming.

## Acknowledgements

The concept and approach follow [`antirez/ds4`](https://github.com/antirez/ds4)
(DwarfStar). The GGUF format, the k-quant block layouts, and a great deal of hard-won
local-inference engineering come from [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and GGML. Quasar does not link against GGML, but it would not exist without it.

## License

MIT — see [LICENSE](LICENSE).
