/* quasar.h - Qwen3-30B-A3B model loader on top of the GGUF reader. */
#ifndef QUASAR_H
#define QUASAR_H

#include "gguf.h"
#include <stdbool.h>

#define QUASAR_ARCH "qwen3moe"

/* Architecture hyper-parameters, read from GGUF metadata. */
typedef struct {
    char     arch[64];
    char     name[128];
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_vocab;
    uint32_t n_ff_exp;      /* MoE expert intermediate size */
    uint32_t n_ff_dense;    /* dense ffn size (unused in pure-MoE layers) */
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t ctx_train;
    float    rope_theta;
    float    rms_eps;
    bool     norm_topk_prob;
    int32_t  bos_id;
    int32_t  eos_id;
    int      missing_keys;
} quasar_hparams;

/* Resolved tensor handles for one transformer block. */
typedef struct {
    const gguf_tensor *attn_norm;
    const gguf_tensor *attn_q, *attn_k, *attn_v, *attn_o;
    const gguf_tensor *attn_q_norm, *attn_k_norm;   /* per-head QK RMSNorm */
    const gguf_tensor *ffn_norm;
    const gguf_tensor *ffn_gate_inp;                 /* router */
    const gguf_tensor *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;
} quasar_layer;

typedef struct {
    gguf_ctx          *gguf;
    quasar_hparams     hp;
    const gguf_tensor *tok_embd;
    const gguf_tensor *output_norm;
    const gguf_tensor *output;       /* may alias tok_embd if tied */
    quasar_layer      *layers;
    int                missing_tensors;
} quasar_model;

quasar_model *quasar_model_load(const char *path, char *err, size_t errlen);
void          quasar_model_free(quasar_model *m);

#endif /* QUASAR_H */
