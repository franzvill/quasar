/* forward_cpu.h - f32 reference forward pass (the correctness oracle).
 *
 * Slow and exact: dequantizes weights per row and runs plain f32 math, matching
 * the Qwen3-30B-A3B graph (GQA + per-head QK-norm + NeoX RoPE + softmax-top8 MoE).
 * Every later Metal/quant kernel is validated against this. */
#ifndef QUASAR_FORWARD_CPU_H
#define QUASAR_FORWARD_CPU_H

#include "quasar.h"

typedef struct quasar_ctx quasar_ctx;

quasar_ctx *quasar_ctx_new(quasar_model *m, int max_seq, int use_metal);
void        quasar_ctx_free(quasar_ctx *c);

/* Clear the KV cache (start a fresh sequence). */
void quasar_kv_reset(quasar_ctx *c);

/* Append n new tokens to the sequence (positions [n_past, n_past+n)), updating
 * the KV cache, and write logits for the LAST of them into logits_out
 * (size hp.n_vocab). Use n = prompt length for prefill, then n = 1 per token. */
void quasar_decode(quasar_ctx *c, const int32_t *ids, int n, float *logits_out);

/* Phase profiling: reset accumulators, then dump ms/token after a decode run. */
void quasar_profile_reset(void);
void quasar_profile_dump(int n_tokens);

#endif /* QUASAR_FORWARD_CPU_H */
