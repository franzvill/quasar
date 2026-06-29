/* metal.h - Metal compute backend (C interface over Objective-C).
 *
 * The whole GGUF mmap is wrapped as a single zero-copy unified-memory buffer
 * (newBufferWithBytesNoCopy), so GPU kernels read weights straight from the
 * mapped file with no upload. gemv dequantizes per row on the GPU. */
#ifndef QUASAR_METAL_H
#define QUASAR_METAL_H

#include <stdint.h>
#include <stddef.h>

int  quasar_metal_init(void);                              /* 0 ok, <0 on failure */
int  quasar_metal_available(void);                         /* 1 if init succeeded  */
const char *quasar_metal_device_name(void);

/* Wrap a page-aligned region (the model mmap) as the weights buffer. 0 on ok. */
int  quasar_metal_set_weights(const void *base, size_t len);

/* y[n_out] = W @ x[n_in], where each row of W (ggml `type`) starts at byte
 * offset `row0` within the weights buffer. Blocking (waits for completion). */
void quasar_metal_gemv(uint32_t ggml_type, uint64_t row0,
                       const float *x, int n_in, int n_out, float *y);

/* Batched gemv: many independent matmuls sharing one input/output buffer,
 * submitted in a single command buffer with one sync. Each job reads
 * in[x_off..] and writes out[y_off..]. */
typedef struct {
    uint32_t type;
    uint64_t row0;
    uint32_t x_off, y_off;   /* float offsets into in / out */
    int      n_in, n_out;
} quasar_gemv_job;

void quasar_metal_gemv_batch(const quasar_gemv_job *jobs, int njobs,
                             const float *in, int in_floats, float *out, int out_floats);

/* GPU-resident MoE: router -> softmax/top-NU -> indexed expert gate/up/down ->
 * SwiGLU -> weighted combine, all in one command buffer. Routed experts are
 * Q2_K. `*_row0` are byte offsets of expert 0; `expert_bytes` is per-expert
 * stride. Reads xn[E], writes moe_y[E]. */
typedef struct {
    uint64_t router_row0;  uint32_t router_type;
    uint64_t gate_row0, up_row0, down_row0;  uint32_t expert_type;
    uint64_t expert_bytes;
    int E, FF, NE, NU, norm_topk;
} quasar_moe_desc;

void quasar_metal_moe(const quasar_moe_desc *d, const float *xn, float *moe_y);

/* GPU attention block: rmsnorm -> q/k/v (into the GPU KV cache) -> per-head
 * QK-norm + RoPE -> GQA attention -> o-proj -> residual, in one command buffer.
 * `*` are byte offsets of row 0; norms are F32. Updates cur[E] in place; the
 * KV cache persists across calls (one buffer per layer slot). */
typedef struct {
    uint64_t attn_norm, q, k, v, o, q_norm, k_norm;
    uint32_t q_type, k_type, v_type, o_type;
    int E, H, HK, HD, n_layer, max_seq, layer;
    float eps, theta;
} quasar_attn_desc;

void quasar_metal_attn(const quasar_attn_desc *d, float *cur, int pos);

/* Full GPU-resident forward: the whole token (all layers + final norm/output)
 * runs in one command buffer with cur/activations/KV all on the GPU. The CPU
 * only writes the token embedding into `embed` and reads `logits` (pass NULL to
 * skip the output projection for prefill tokens). All offsets are byte offsets
 * of row 0 in the weights mmap. */
typedef struct {
    uint64_t attn_norm, q, k, v, o, q_norm, k_norm;
    uint32_t q_type, k_type, v_type, o_type;
    uint64_t ffn_norm, router, gate, up, down;
    uint32_t router_type, expert_type;
    uint64_t expert_bytes;
} quasar_layer_desc;

typedef struct {
    uint64_t output_norm, output;
    uint32_t output_type;
    int E, H, HK, HD, FF, NE, NU, n_layer, max_seq, n_vocab;
    float eps, theta;
    int norm_topk;
} quasar_fwd_cfg;

void quasar_metal_forward_token(const quasar_fwd_cfg *cfg, const quasar_layer_desc *layers,
                                int pos, const float *embed, float *logits);

#endif /* QUASAR_METAL_H */
