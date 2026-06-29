/* forward_cpu.c - see forward_cpu.h.
 *
 * Incremental decode with a persistent per-layer KV cache: each call processes
 * the new tokens one at a time through all layers, appending K/V to the cache
 * and attending over it. This makes generation O(total tokens) of matmul work
 * instead of re-running the whole sequence every step. The heavy matmuls go
 * through gemv(), which uses Metal when enabled and the CPU oracle otherwise. */
#include "forward_cpu.h"
#include "quant.h"
#include "metal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

struct quasar_ctx {
    quasar_model *m;
    int   max_seq;
    int   n_past;              /* positions currently held in the KV cache */
    int   use_metal;
    const uint8_t *map_base;   /* mmap base, for absolute weight offsets */
    float *x;        /* [n_embd] current-token residual stream */
    float *xn;       /* [n_embd] normed input */
    float *q;        /* [n_head * head_dim] */
    float *att;      /* [n_head * head_dim] attention output */
    float *Kc, *Vc;  /* [n_layer * max_seq * n_head_kv * head_dim] persistent cache */
    float *scores;   /* [max_seq] */
    float *router;   /* [n_expert] */
    float *gate, *up, *hid;   /* [n_ff_exp] */
    float *moe_y, *tmp;       /* [n_embd] */
    float *scratch;  /* [n_embd] gemv dequant row */
    float *qkv_out;  /* batched q|k|v output (Metal) */
    float *moe_buf;  /* batched gate|up output: 2*n_expert_used*n_ff_exp */
    float *hid8;     /* silu(gate)*up for all selected experts */
    float *down_out; /* batched down output: n_expert_used*n_embd */
};

quasar_ctx *quasar_ctx_new(quasar_model *m, int max_seq, int use_metal) {
    quasar_hparams *h = &m->hp;
    quasar_ctx *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->m = m; c->max_seq = max_seq; c->n_past = 0;
    c->use_metal = use_metal;
    c->map_base = (const uint8_t *)m->gguf->map;
    int E = h->n_embd, HKHD = h->n_head_kv * h->head_dim, QHD = h->n_head * h->head_dim;
    size_t kv = (size_t)h->n_layer * max_seq * HKHD;
    c->x       = malloc((size_t)E * sizeof(float));
    c->xn      = malloc((size_t)E * sizeof(float));
    c->q       = malloc((size_t)QHD * sizeof(float));
    c->att     = malloc((size_t)QHD * sizeof(float));
    c->Kc      = malloc(kv * sizeof(float));
    c->Vc      = malloc(kv * sizeof(float));
    c->scores  = malloc((size_t)max_seq * sizeof(float));
    c->router  = malloc((size_t)h->n_expert * sizeof(float));
    c->gate    = malloc((size_t)h->n_ff_exp * sizeof(float));
    c->up      = malloc((size_t)h->n_ff_exp * sizeof(float));
    c->hid     = malloc((size_t)h->n_ff_exp * sizeof(float));
    c->moe_y   = malloc((size_t)E * sizeof(float));
    c->tmp     = malloc((size_t)E * sizeof(float));
    c->scratch = malloc((size_t)E * sizeof(float));
    c->qkv_out = malloc((size_t)(QHD + 2 * HKHD) * sizeof(float));
    c->moe_buf = malloc((size_t)2 * h->n_expert_used * h->n_ff_exp * sizeof(float));
    c->hid8    = malloc((size_t)h->n_expert_used * h->n_ff_exp * sizeof(float));
    c->down_out= malloc((size_t)h->n_expert_used * E * sizeof(float));
    return c;
}

void quasar_ctx_free(quasar_ctx *c) {
    if (!c) return;
    free(c->x); free(c->xn); free(c->q); free(c->att); free(c->Kc); free(c->Vc);
    free(c->scores); free(c->router); free(c->gate); free(c->up); free(c->hid);
    free(c->moe_y); free(c->tmp); free(c->scratch);
    free(c->qkv_out); free(c->moe_buf); free(c->hid8); free(c->down_out);
    free(c);
}

/* y[n_out] = W @ x[n_in], rows from element `elem_base` (MoE expert slice). */
static void gemv(const gguf_tensor *W, size_t elem_base, const float *x,
                 int n_in, int n_out, float *y, quasar_ctx *c) {
    uint32_t blck = ggml_blck_size(W->type);
    size_t   tsz  = ggml_type_size(W->type);
    if (c->use_metal) {
        uint64_t row0 = (uint64_t)((const uint8_t *)W->data - c->map_base)
                      + (uint64_t)(elem_base / blck) * tsz;
        quasar_metal_gemv(W->type, row0, x, n_in, n_out, y);
        return;
    }
    const uint8_t *data = W->data;
    for (int r = 0; r < n_out; r++) {
        size_t e = elem_base + (size_t)r * n_in;
        quasar_dequant_row(W->type, data + (e / blck) * tsz, c->scratch, n_in);
        float acc = 0.0f;
        for (int cc = 0; cc < n_in; cc++) acc += c->scratch[cc] * x[cc];
        y[r] = acc;
    }
}

/* Absolute byte offset of a tensor's row 0 (plus an expert slice) in the mmap. */
static uint64_t row0_of(quasar_ctx *c, const gguf_tensor *W, size_t elem_base) {
    uint32_t blck = ggml_blck_size(W->type);
    size_t   tsz  = ggml_type_size(W->type);
    return (uint64_t)((const uint8_t *)W->data - c->map_base) + (uint64_t)(elem_base / blck) * tsz;
}

static void rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void softmax(float *a, int n) {
    float mx = a[0];
    for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { a[i] = expf(a[i] - mx); sum += a[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) a[i] *= inv;
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* GPT-NeoX RoPE over the full head: rotate pairs (j, j + head_dim/2). */
static void rope_neox(float *v, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int j = 0; j < half; j++) {
        float freq = powf(theta, -2.0f * j / head_dim);
        float ang  = pos * freq;
        float cs = cosf(ang), sn = sinf(ang);
        float x0 = v[j], x1 = v[j + half];
        v[j]        = x0 * cs - x1 * sn;
        v[j + half] = x0 * sn + x1 * cs;
    }
}

static void embed_token(quasar_ctx *c, int32_t id, float *dst) {
    quasar_model *m = c->m;
    int E = (int)m->hp.n_embd;
    uint32_t et = m->tok_embd->type, eb = ggml_blck_size(et);
    size_t   es = ggml_type_size(et);
    quasar_dequant_row(et, (const uint8_t *)m->tok_embd->data + ((size_t)id * E / eb) * es, dst, E);
}

void quasar_kv_reset(quasar_ctx *c) { c->n_past = 0; }

void quasar_decode(quasar_ctx *c, const int32_t *ids, int n, float *logits) {
    quasar_model *m = c->m;
    quasar_hparams *h = &m->hp;
    const int E = h->n_embd, H = h->n_head, HK = h->n_head_kv, HD = h->head_dim;
    const int FF = h->n_ff_exp, NV = h->n_vocab, NE = h->n_expert, NU = h->n_expert_used;
    const float eps = h->rms_eps, scale = 1.0f / sqrtf((float)HD);
    const int gsize = H / HK;
    const size_t KVL = (size_t)c->max_seq * HK * HD;   /* per-layer stride */

    for (int t = 0; t < n; t++) {
        int pos = c->n_past + t;
        float *cur = c->x;
        embed_token(c, ids[t], cur);

        for (uint32_t il = 0; il < h->n_layer; il++) {
            quasar_layer *L = &m->layers[il];
            float *Kl = c->Kc + (size_t)il * KVL;
            float *Vl = c->Vc + (size_t)il * KVL;

            /* attention */
            rmsnorm(cur, (const float *)L->attn_norm->data, E, eps, c->xn);
            float *kt = Kl + (size_t)pos * HK * HD;
            float *vt = Vl + (size_t)pos * HK * HD;
            if (c->use_metal) {
                int QHD = H * HD, KVD = HK * HD;
                quasar_gemv_job jb[3] = {
                    { L->attn_q->type, row0_of(c, L->attn_q, 0), 0, 0,                       E, QHD },
                    { L->attn_k->type, row0_of(c, L->attn_k, 0), 0, (uint32_t)QHD,           E, KVD },
                    { L->attn_v->type, row0_of(c, L->attn_v, 0), 0, (uint32_t)(QHD + KVD),   E, KVD },
                };
                quasar_metal_gemv_batch(jb, 3, c->xn, E, c->qkv_out, QHD + 2 * KVD);
                memcpy(c->q, c->qkv_out,             (size_t)QHD * sizeof(float));
                memcpy(kt,   c->qkv_out + QHD,       (size_t)KVD * sizeof(float));
                memcpy(vt,   c->qkv_out + QHD + KVD, (size_t)KVD * sizeof(float));
            } else {
                gemv(L->attn_q, 0, c->xn, E, H  * HD, c->q, c);
                gemv(L->attn_k, 0, c->xn, E, HK * HD, kt,   c);
                gemv(L->attn_v, 0, c->xn, E, HK * HD, vt,   c);
            }
            for (int hh = 0; hh < H;  hh++) { rmsnorm(c->q + hh * HD, (const float *)L->attn_q_norm->data, HD, eps, c->q + hh * HD); rope_neox(c->q + hh * HD, HD, pos, h->rope_theta); }
            for (int hh = 0; hh < HK; hh++) { rmsnorm(kt   + hh * HD, (const float *)L->attn_k_norm->data, HD, eps, kt   + hh * HD); rope_neox(kt   + hh * HD, HD, pos, h->rope_theta); }

            for (int hh = 0; hh < H; hh++) {
                int kvh = hh / gsize;
                float *qh = c->q + hh * HD;
                for (int s = 0; s <= pos; s++) {
                    const float *ks = Kl + (size_t)s * HK * HD + kvh * HD;
                    float dot = 0.0f;
                    for (int d = 0; d < HD; d++) dot += qh[d] * ks[d];
                    c->scores[s] = dot * scale;
                }
                softmax(c->scores, pos + 1);
                float *ah = c->att + hh * HD;
                for (int d = 0; d < HD; d++) ah[d] = 0.0f;
                for (int s = 0; s <= pos; s++) {
                    const float *vs = Vl + (size_t)s * HK * HD + kvh * HD;
                    float w = c->scores[s];
                    for (int d = 0; d < HD; d++) ah[d] += w * vs[d];
                }
            }
            gemv(L->attn_o, 0, c->att, H * HD, E, c->tmp, c);
            for (int i = 0; i < E; i++) cur[i] += c->tmp[i];

            /* MoE */
            rmsnorm(cur, (const float *)L->ffn_norm->data, E, eps, c->xn);
            gemv(L->ffn_gate_inp, 0, c->xn, E, NE, c->router, c);
            softmax(c->router, NE);
            int idx[64]; float wt[64];
            unsigned char taken[256] = {0};
            for (int s = 0; s < NU; s++) {
                int bi = -1; float best = -1e30f;
                for (int e = 0; e < NE; e++) if (!taken[e] && c->router[e] > best) { best = c->router[e]; bi = e; }
                idx[s] = bi; wt[s] = c->router[bi]; taken[bi] = 1;
            }
            if (h->norm_topk_prob) { float sm = 0; for (int s = 0; s < NU; s++) sm += wt[s]; if (sm > 0) for (int s = 0; s < NU; s++) wt[s] /= sm; }
            size_t gpe = (size_t)E * FF, dpe = (size_t)FF * E;
            for (int i = 0; i < E; i++) c->moe_y[i] = 0.0f;
            if (c->use_metal) {
                quasar_gemv_job gj[16];
                for (int s = 0; s < NU; s++) {
                    gj[s]      = (quasar_gemv_job){ L->ffn_gate_exps->type, row0_of(c, L->ffn_gate_exps, (size_t)idx[s] * gpe), 0, (uint32_t)(s * FF),        E, FF };
                    gj[NU + s] = (quasar_gemv_job){ L->ffn_up_exps->type,   row0_of(c, L->ffn_up_exps,   (size_t)idx[s] * gpe), 0, (uint32_t)((NU + s) * FF), E, FF };
                }
                quasar_metal_gemv_batch(gj, 2 * NU, c->xn, E, c->moe_buf, 2 * NU * FF);
                for (int s = 0; s < NU; s++)
                    for (int i = 0; i < FF; i++)
                        c->hid8[s * FF + i] = silu(c->moe_buf[s * FF + i]) * c->moe_buf[(NU + s) * FF + i];
                quasar_gemv_job dj[8];
                for (int s = 0; s < NU; s++)
                    dj[s] = (quasar_gemv_job){ L->ffn_down_exps->type, row0_of(c, L->ffn_down_exps, (size_t)idx[s] * dpe), (uint32_t)(s * FF), (uint32_t)(s * E), FF, E };
                quasar_metal_gemv_batch(dj, NU, c->hid8, NU * FF, c->down_out, NU * E);
                for (int s = 0; s < NU; s++) { float w = wt[s]; for (int i = 0; i < E; i++) c->moe_y[i] += w * c->down_out[s * E + i]; }
            } else {
                for (int s = 0; s < NU; s++) {
                    int e = idx[s]; float w = wt[s];
                    gemv(L->ffn_gate_exps, (size_t)e * gpe, c->xn, E, FF, c->gate, c);
                    gemv(L->ffn_up_exps,   (size_t)e * gpe, c->xn, E, FF, c->up,   c);
                    for (int i = 0; i < FF; i++) c->hid[i] = silu(c->gate[i]) * c->up[i];
                    gemv(L->ffn_down_exps, (size_t)e * dpe, c->hid, FF, E, c->tmp, c);
                    for (int i = 0; i < E; i++) c->moe_y[i] += w * c->tmp[i];
                }
            }
            for (int i = 0; i < E; i++) cur[i] += c->moe_y[i];
        }

        if (t == n - 1 && logits) {
            rmsnorm(cur, (const float *)m->output_norm->data, E, eps, c->xn);
            gemv(m->output, 0, c->xn, E, NV, logits, c);
        }
    }
    c->n_past += n;
}
