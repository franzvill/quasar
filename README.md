# Quasar

A small native inference engine, **purpose-built for Qwen3-30B-A3B**, optimized for
**Apple Silicon / Metal** on a 24 GB Mac. It applies the ideas of
[`antirez/ds4`](https://github.com/antirez/ds4) (a DeepSeek-V4-specific engine for
96–128 GB machines) to a model that fits a laptop: a 30B Mixture-of-Experts model
with only **~3.3B active parameters per token**, quantized so the routed experts are
crushed to ~2-bit while everything else stays high precision.

Why this can be fast on 24 GB: decode speed on Apple Silicon ≈
`memory_bandwidth / active_bytes_per_token`. With ~3.3B active params at ~2-bit
experts, very little memory moves per token.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design, the Qwen3-30B-A3B shape
profile, the memory budget, and the milestone roadmap.

## Status

Runs Qwen3-30B-A3B end to end. Working today: GGUF loader, byte-level BPE
tokenizer + ChatML, a correct CPU reference forward, a Metal backend
(dequant-matvec kernels, batched dispatch), asymmetric **2-bit expert quant**
(`requant`: 18.6 GB → 10.5 GB, fits 24 GB), and a **KV cache** for incremental
decode. Measured on an M5 Pro with the 2-bit model: ~6–10 tokens/sec, coherent
output. Still to come: a fully GPU-resident forward (move the glue onto Metal)
for higher t/s, temperature/min-p sampling, a CLI chat loop, and an
OpenAI-compatible server.

```sh
make                                                  # build ./quasar
make test-gguf                                        # tiny synthetic model, no download

./quasar inspect   gguf/Qwen3-30B-A3B-Q4_K_M.gguf     # load + validate
./quasar requant   gguf/Qwen3-30B-A3B-Q4_K_M.gguf gguf/q2k.gguf   # make the 2-bit model
./quasar generate  gguf/q2k.gguf "Once upon a time" 40 metal      # generate on the GPU
```

## License

MIT. GGUF/ggml quant layouts and conventions derive from llama.cpp/GGML — see
ARCHITECTURE.md acknowledgements.
