/* quasar.c - model loader + `inspect` command.
 *
 * Quasar is a Qwen3-30B-A3B specific inference engine (see ARCHITECTURE.md).
 * Today this file loads and validates a qwen3moe GGUF against the canonical
 * shape profile; the forward pass / Metal backend land in later milestones. */
#include "quasar.h"
#include "tokenizer.h"
#include "chat.h"
#include "forward_cpu.h"
#include "quant.h"
#include "metal.h"
#include "requant.h"
#include "server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

/* ---- hparams ---------------------------------------------------------- */

static void copy_str_kv(gguf_ctx *g, const char *key, char *dst, size_t cap) {
    const char *s; uint64_t sl;
    if (gguf_str(g, key, &s, &sl)) {
        size_t n = sl < cap - 1 ? (size_t)sl : cap - 1;
        memcpy(dst, s, n); dst[n] = 0;
    } else {
        dst[0] = 0;
    }
}

static int read_hparams(gguf_ctx *g, quasar_hparams *hp) {
    int missing = 0;

    copy_str_kv(g, "general.architecture", hp->arch, sizeof hp->arch);
    copy_str_kv(g, "general.name",         hp->name, sizeof hp->name);
    if (hp->arch[0] == 0) missing++;

    #define RU(key, field) do { if (!gguf_u32(g, key, &hp->field)) { hp->field = 0; missing++; } } while (0)
    RU("qwen3moe.block_count",            n_layer);
    RU("qwen3moe.embedding_length",       n_embd);
    RU("qwen3moe.attention.head_count",   n_head);
    RU("qwen3moe.attention.head_count_kv",n_head_kv);
    RU("qwen3moe.expert_count",           n_expert);
    RU("qwen3moe.expert_used_count",      n_expert_used);
    #undef RU

    hp->head_dim = gguf_u32_or(g, "qwen3moe.attention.key_length", 0);
    if (hp->head_dim == 0 && hp->n_head) hp->head_dim = hp->n_embd / hp->n_head;
    hp->n_ff_exp   = gguf_u32_or(g, "qwen3moe.expert_feed_forward_length", 0);
    hp->n_ff_dense = gguf_u32_or(g, "qwen3moe.feed_forward_length", 0);
    hp->ctx_train  = gguf_u32_or(g, "qwen3moe.context_length", 0);
    hp->rope_theta = gguf_f32_or(g, "qwen3moe.rope.freq_base", 1000000.0f);
    hp->rms_eps    = gguf_f32_or(g, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6f);

    bool b;
    hp->norm_topk_prob = gguf_bool(g, "qwen3moe.expert_weights_norm", &b) ? b : true;

    hp->n_vocab = gguf_u32_or(g, "qwen3moe.vocab_size", 0);
    if (hp->n_vocab == 0) {
        const gguf_tensor *te = gguf_find_tensor(g, "token_embd.weight");
        if (te && te->n_dims >= 2) hp->n_vocab = (uint32_t)te->dims[1];
    }

    int32_t id;
    hp->bos_id = gguf_i32(g, "tokenizer.ggml.bos_token_id", &id) ? id : -1;
    hp->eos_id = gguf_i32(g, "tokenizer.ggml.eos_token_id", &id) ? id : -1;

    hp->missing_keys = missing;
    return missing;
}

/* ---- load ------------------------------------------------------------- */

static const gguf_tensor *req(gguf_ctx *g, int *missing, const char *name) {
    const gguf_tensor *t = gguf_find_tensor(g, name);
    if (!t) (*missing)++;
    return t;
}

quasar_model *quasar_model_load(const char *path, char *err, size_t errlen) {
    gguf_ctx *g = gguf_open(path);
    if (!g) { snprintf(err, errlen, "out of memory"); return NULL; }
    if (g->err[0]) { snprintf(err, errlen, "%s", g->err); gguf_close(g); return NULL; }

    quasar_model *m = calloc(1, sizeof *m);
    if (!m) { snprintf(err, errlen, "out of memory"); gguf_close(g); return NULL; }
    m->gguf = g;
    read_hparams(g, &m->hp);
    /* QUASAR_TOPK overrides experts-used-per-token (e.g. 4 vs the trained 8) for a
     * speed/quality A/B without re-encoding the model. */
    const char *topk_env = getenv("QUASAR_TOPK");
    if (topk_env) {
        int v = atoi(topk_env);
        if (v > 0 && v <= (int)m->hp.n_expert) m->hp.n_expert_used = (uint32_t)v;
    }

    uint32_t L = m->hp.n_layer;
    m->layers = calloc(L ? L : 1, sizeof(quasar_layer));
    if (!m->layers) { snprintf(err, errlen, "out of memory"); quasar_model_free(m); return NULL; }

    int missing = 0;
    char nb[160];

    m->tok_embd    = req(g, &missing, "token_embd.weight");
    m->output_norm = req(g, &missing, "output_norm.weight");
    m->output      = gguf_find_tensor(g, "output.weight");
    if (!m->output) m->output = m->tok_embd;   /* tied-embedding fallback */

    for (uint32_t il = 0; il < L; il++) {
        quasar_layer *ly = &m->layers[il];
        snprintf(nb, sizeof nb, "blk.%u.attn_norm.weight",   il); ly->attn_norm    = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.attn_q.weight",      il); ly->attn_q       = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.attn_k.weight",      il); ly->attn_k       = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.attn_v.weight",      il); ly->attn_v       = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.attn_output.weight", il); ly->attn_o       = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.attn_q_norm.weight", il); ly->attn_q_norm  = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.attn_k_norm.weight", il); ly->attn_k_norm  = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.ffn_norm.weight",    il); ly->ffn_norm     = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.ffn_gate_inp.weight",il); ly->ffn_gate_inp = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.ffn_gate_exps.weight",il);ly->ffn_gate_exps= req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.ffn_up_exps.weight", il); ly->ffn_up_exps  = req(g, &missing, nb);
        snprintf(nb, sizeof nb, "blk.%u.ffn_down_exps.weight",il);ly->ffn_down_exps= req(g, &missing, nb);
    }

    m->missing_tensors = missing;
    return m;
}

void quasar_model_free(quasar_model *m) {
    if (!m) return;
    if (m->gguf) gguf_close(m->gguf);
    free(m->layers);
    free(m);
}

/* ---- inspect ---------------------------------------------------------- */

static void human(double bytes, char *out, size_t n) {
    const char *u[] = { "B", "KB", "MB", "GB", "TB" };
    int i = 0;
    while (bytes >= 1024.0 && i < 4) { bytes /= 1024.0; i++; }
    snprintf(out, n, "%.2f %s", bytes, u[i]);
}

static int cmd_inspect(const char *path) {
    char err[256];
    quasar_model *m = quasar_model_load(path, err, sizeof err);
    if (!m) { fprintf(stderr, "quasar: load failed: %s\n", err); return 1; }

    gguf_ctx *g = m->gguf;
    quasar_hparams *hp = &m->hp;
    char hb[32];

    printf("== GGUF ==\n");
    printf("file:        %s\n", path);
    human((double)g->map_size, hb, sizeof hb);
    printf("size:        %s\n", hb);
    printf("version:     %u\n", g->version);
    printf("alignment:   %u\n", g->alignment);
    printf("metadata kv: %" PRIu64 "\n", g->n_kv);
    printf("tensors:     %" PRIu64 "\n", g->n_tensors);

    printf("\n== Model ==\n");
    printf("arch:        %s%s\n", hp->arch,
           strcmp(hp->arch, QUASAR_ARCH) == 0 ? "" : "   <-- expected qwen3moe");
    printf("name:        %s\n", hp->name);
    printf("layers:      %u\n", hp->n_layer);
    printf("n_embd:      %u\n", hp->n_embd);
    printf("heads:       %u  (kv %u, head_dim %u)\n", hp->n_head, hp->n_head_kv, hp->head_dim);
    printf("n_vocab:     %u\n", hp->n_vocab);
    printf("experts:     %u total, %u used/token\n", hp->n_expert, hp->n_expert_used);
    printf("moe ff:      %u\n", hp->n_ff_exp);
    printf("ctx_train:   %u\n", hp->ctx_train);
    printf("rope_theta:  %.1f\n", hp->rope_theta);
    printf("rms_eps:     %g\n", hp->rms_eps);
    printf("topk_norm:   %s\n", hp->norm_topk_prob ? "yes" : "no");
    printf("bos/eos:     %d / %d\n", hp->bos_id, hp->eos_id);

    /* parameter + byte accounting by dtype */
    double   total_params = 0, total_bytes = 0;
    uint64_t bytes_by_type[GGML_TYPE_COUNT] = {0};
    uint64_t elts_by_type[GGML_TYPE_COUNT]  = {0};
    for (uint64_t t = 0; t < g->n_tensors; t++) {
        gguf_tensor *ti = &g->tensors[t];
        total_params += (double)ti->n_elements;
        total_bytes  += (double)ti->nbytes;
        if (ti->type < GGML_TYPE_COUNT) {
            bytes_by_type[ti->type] += ti->nbytes;
            elts_by_type[ti->type]  += ti->n_elements;
        }
    }
    printf("\n== Tensors by dtype ==\n");
    for (uint32_t ty = 0; ty < GGML_TYPE_COUNT; ty++) {
        if (elts_by_type[ty] == 0) continue;
        human((double)bytes_by_type[ty], hb, sizeof hb);
        printf("  %-8s %16" PRIu64 " elts   %10s\n", ggml_type_name(ty), elts_by_type[ty], hb);
    }
    human(total_bytes, hb, sizeof hb);
    printf("  %-8s %16.0f params  %10s\n", "TOTAL", total_params, hb);

    /* active params per generated token (estimate) */
    double per_layer_attn = (double)hp->n_embd * hp->n_head    * hp->head_dim          /* q  */
                          + 2.0 * (double)hp->n_embd * hp->n_head_kv * hp->head_dim     /* k,v */
                          + (double)hp->n_head * hp->head_dim * hp->n_embd;             /* o  */
    double router        = (double)hp->n_embd * hp->n_expert;
    double active_expert = (double)hp->n_expert_used * 3.0 * hp->n_embd * hp->n_ff_exp;
    double out_proj      = (double)hp->n_embd * hp->n_vocab;
    double active        = hp->n_layer * (per_layer_attn + router + active_expert) + out_proj;
    printf("\n== Active params / token (est) ==\n");
    printf("  %.2f B active  of  %.2f B total\n", active / 1e9, total_params / 1e9);

    /* validation */
    printf("\n== Validation ==\n");
    if (m->missing_tensors == 0) printf("  tensors:  all expected present (%u layers) OK\n", hp->n_layer);
    else                         printf("  tensors:  WARNING %d expected tensor(s) missing\n", m->missing_tensors);
    if (hp->missing_keys)        printf("  metadata: WARNING %d expected key(s) missing\n", hp->missing_keys);
    else                         printf("  metadata: all expected keys present OK\n");

    int canon = strcmp(hp->arch, QUASAR_ARCH) == 0 &&
                hp->n_layer == 48 && hp->n_embd == 2048 && hp->n_head == 32 &&
                hp->n_head_kv == 4 && hp->head_dim == 128 &&
                hp->n_expert == 128 && hp->n_expert_used == 8 && hp->n_ff_exp == 768;
    printf("  profile:  %s\n", canon ? "Qwen3-30B-A3B canonical OK"
                                     : "non-canonical (engine targets Qwen3-30B-A3B)");

    quasar_model_free(m);
    return 0;
}

/* ---- tokenizer commands ----------------------------------------------- */

static quasar_tokenizer *open_tok(const char *path, gguf_ctx **out_g) {
    gguf_ctx *g = gguf_open(path);
    if (!g) { fprintf(stderr, "quasar: out of memory\n"); return NULL; }
    if (g->err[0]) { fprintf(stderr, "quasar: %s\n", g->err); gguf_close(g); return NULL; }
    char err[256];
    quasar_tokenizer *tk = quasar_tokenizer_load(g, err, sizeof err);
    if (!tk) { fprintf(stderr, "quasar: tokenizer: %s\n", err); gguf_close(g); return NULL; }
    *out_g = g;
    return tk;
}

static int cmd_tokenize(const char *path, const char *text) {
    gguf_ctx *g; quasar_tokenizer *tk = open_tok(path, &g);
    if (!tk) return 1;

    size_t n = 0;
    int32_t *ids = quasar_tokenize(tk, text, strlen(text), false, &n);
    printf("vocab:  %u\n", quasar_vocab_size(tk));
    printf("tokens: %zu\n", n);
    printf("ids:    ");
    for (size_t i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");
    if (n == 0) printf("\n");
    printf("pieces: ");
    for (size_t i = 0; i < n; i++) {
        char b[256]; int w = quasar_token_to_bytes(tk, ids[i], b, (int)sizeof b - 1); b[w] = 0;
        printf("|%s", b);
    }
    printf("|\n");

    size_t dl = 0; char *dec = quasar_detokenize(tk, ids, n, &dl);
    int ok = (dl == strlen(text) && memcmp(dec, text, dl) == 0);
    printf("roundtrip: %s\n", ok ? "OK" : "MISMATCH");
    if (!ok) printf("decoded:   %s\n", dec);

    free(dec); free(ids);
    quasar_tokenizer_free(tk); gguf_close(g);
    return ok ? 0 : 2;
}

static int cmd_detokenize(const char *path, int n_ids, char **id_args) {
    gguf_ctx *g; quasar_tokenizer *tk = open_tok(path, &g);
    if (!tk) return 1;
    int32_t *ids = malloc((size_t)n_ids * sizeof(int32_t));
    for (int i = 0; i < n_ids; i++) ids[i] = (int32_t)atoi(id_args[i]);
    size_t dl = 0; char *dec = quasar_detokenize(tk, ids, n_ids, &dl);
    printf("%s\n", dec);
    free(dec); free(ids);
    quasar_tokenizer_free(tk); gguf_close(g);
    return 0;
}

/* ---- generate (CPU reference forward) --------------------------------- */

static int argmax(const float *a, int n) {
    int bi = 0;
    for (int i = 1; i < n; i++) if (a[i] > a[bi]) bi = i;
    return bi;
}

static void print_topk(const float *logits, int nv, quasar_tokenizer *tk, int k) {
    unsigned char *taken = calloc((size_t)nv, 1);
    for (int i = 0; i < k; i++) {
        int bi = -1; float best = -1e30f;
        for (int v = 0; v < nv; v++) if (!taken[v] && logits[v] > best) { best = logits[v]; bi = v; }
        if (bi < 0) break;
        taken[bi] = 1;
        char b[256]; int w = quasar_token_to_bytes(tk, bi, b, (int)sizeof b - 1); b[w] = 0;
        printf("  %2d. id=%-6d  logit=%8.3f  |%s|\n", i + 1, bi, best, b);
    }
    free(taken);
}

static int cmd_generate(const char *path, const char *prompt, int n_new, int use_metal) {
    char err[256];
    quasar_model *m = quasar_model_load(path, err, sizeof err);
    if (!m) { fprintf(stderr, "quasar: load: %s\n", err); return 1; }
    if (m->missing_tensors) {
        fprintf(stderr, "quasar: model incomplete (%d tensors missing)\n", m->missing_tensors);
        quasar_model_free(m); return 1;
    }
    quasar_tokenizer *tk = quasar_tokenizer_load(m->gguf, err, sizeof err);
    if (!tk) { fprintf(stderr, "quasar: tokenizer: %s\n", err); quasar_model_free(m); return 1; }

    size_t np = 0;
    int32_t *pids = quasar_tokenize(tk, prompt, strlen(prompt), false, &np);
    if (np == 0) { fprintf(stderr, "quasar: empty prompt\n"); return 1; }

    int NV = (int)m->hp.n_vocab;
    int max_seq = (int)np + n_new + 2;
    int metal_ready = 0;
    if (use_metal) {
        if (quasar_metal_init() == 0 && quasar_metal_set_weights(m->gguf->map, m->gguf->map_size) == 0)
            metal_ready = 1;
        else
            fprintf(stderr, "quasar: Metal unavailable; using CPU\n");
    }
    quasar_ctx *ctx = quasar_ctx_new(m, max_seq, metal_ready);
    float   *logits = malloc((size_t)NV * sizeof(float));
    int32_t *seq    = malloc((size_t)max_seq * sizeof(int32_t));
    int n = (int)np;
    memcpy(seq, pids, np * sizeof(int32_t));

    printf("prompt: \"%s\"  (%d tokens)\n", prompt, n);
    printf("backend: %s\n", metal_ready ? "Metal (Apple GPU)" : "CPU reference (slow)");
    quasar_kv_reset(ctx);
    quasar_decode(ctx, seq, n, logits);          /* prefill the prompt */
    printf("top-5 next tokens:\n");
    print_topk(logits, NV, tk, 5);

    if (n_new > 0) {
        printf("\ngreedy continuation:\n%s", prompt);
        fflush(stdout);
        for (int s = 0; s < n_new; s++) {
            int nx = argmax(logits, NV);
            if (nx == quasar_token_eos(tk)) break;
            char b[256]; int w = quasar_token_to_bytes(tk, nx, b, (int)sizeof b - 1); b[w] = 0;
            printf("%s", b); fflush(stdout);
            int32_t one = nx;
            quasar_decode(ctx, &one, 1, logits);  /* one incremental step (KV-cached) */
        }
        printf("\n");
    }

    free(logits); free(seq); free(pids);
    quasar_ctx_free(ctx); quasar_tokenizer_free(tk); quasar_model_free(m);
    return 0;
}

/* ---- bench (steady-state decode tok/s) -------------------------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmd_bench(const char *path, int n_new, int use_metal, const char *prompt) {
    if (n_new <= 0)  n_new  = 64;
    if (!prompt)     prompt = "The capital of France is";
    char err[256];
    double t0 = now_s();
    quasar_model *m = quasar_model_load(path, err, sizeof err);
    if (!m) { fprintf(stderr, "quasar: load: %s\n", err); return 1; }
    if (m->missing_tensors) { fprintf(stderr, "quasar: model incomplete (%d tensors missing)\n", m->missing_tensors); quasar_model_free(m); return 1; }
    quasar_tokenizer *tk = quasar_tokenizer_load(m->gguf, err, sizeof err);
    if (!tk) { fprintf(stderr, "quasar: tokenizer: %s\n", err); quasar_model_free(m); return 1; }

    size_t np = 0;
    int32_t *pids = quasar_tokenize(tk, prompt, strlen(prompt), false, &np);
    if (np == 0) { fprintf(stderr, "quasar: empty prompt\n"); return 1; }
    int NV = (int)m->hp.n_vocab;
    int max_seq = (int)np + n_new + 16;

    int metal_ready = 0;
    if (use_metal) {
        if (quasar_metal_init() == 0 && quasar_metal_set_weights(m->gguf->map, m->gguf->map_size) == 0)
            metal_ready = 1;
        else fprintf(stderr, "quasar: Metal unavailable; using CPU\n");
    }
    quasar_ctx *ctx = quasar_ctx_new(m, max_seq, metal_ready);
    float *logits = malloc((size_t)NV * sizeof(float));
    double t_loaded = now_s();

    /* Warm-up: prefill + a few decode steps to page in the working set and warm GPU pipelines. */
    quasar_kv_reset(ctx);
    quasar_decode(ctx, pids, (int)np, logits);
    const int warm = 8;
    for (int s = 0; s < warm; s++) { int32_t one = argmax(logits, NV); quasar_decode(ctx, &one, 1, logits); }
    double t_warm = now_s();

    /* Measured steady-state decode: single-token steps. */
    quasar_profile_reset();
    double t_dec0 = now_s();
    for (int s = 0; s < n_new; s++) { int32_t one = argmax(logits, NV); quasar_decode(ctx, &one, 1, logits); }
    double t_dec1 = now_s();
    double dec_s = t_dec1 - t_dec0;

    /* Isolated prefill on a fresh KV (pages already resident). */
    quasar_kv_reset(ctx);
    double t_pf0 = now_s();
    quasar_decode(ctx, pids, (int)np, logits);
    double t_pf1 = now_s();

    printf("== bench ==\n");
    printf("model:    %s\n", path);
    printf("backend:  %s\n", metal_ready ? "Metal (Apple GPU)" : "CPU reference");
    printf("load:     %.2f s (mmap + set_weights, includes cold page-in)\n", t_loaded - t0);
    printf("warmup:   %.2f s (prefill %zu + %d decode)\n", t_warm - t_loaded, np, warm);
    printf("prefill:  %.1f tok/s  (%zu tok / %.3f s)\n", (double)np / (t_pf1 - t_pf0), np, t_pf1 - t_pf0);
    printf("DECODE:   %.2f tok/s  (%d tok / %.3f s = %.2f ms/tok)\n", (double)n_new / dec_s, n_new, dec_s, dec_s / n_new * 1e3);
    quasar_profile_dump(n_new);

    free(logits);
    quasar_ctx_free(ctx); quasar_tokenizer_free(tk); quasar_model_free(m);
    return 0;
}

static int cmd_render(const char *path) {
    quasar_msg msgs[2] = {
        { "system", "You are a helpful assistant." },
        { "user",   "Hello!" },
    };
    for (int think = 1; think >= 0; think--) {
        size_t len = 0;
        char *p = quasar_chat_render(msgs, 2, true, think != 0, &len);
        printf("--- thinking=%s (%zu bytes) ---\n%s\n", think ? "on" : "off", len, p);
        if (path) {
            gguf_ctx *g; quasar_tokenizer *tk = open_tok(path, &g);
            if (tk) {
                size_t n = 0; int32_t *ids = quasar_tokenize(tk, p, len, false, &n);
                printf("[%zu tokens; <|im_start|>=%d <|im_end|>=%d]\n\n", n,
                       quasar_token_id(tk, "<|im_start|>"), quasar_token_id(tk, "<|im_end|>"));
                free(ids); quasar_tokenizer_free(tk); gguf_close(g);
            }
        }
        free(p);
    }
    return 0;
}

/* ---- metal self-test -------------------------------------------------- */

static void metal_test_tensor(gguf_ctx *g, const gguf_tensor *W, size_t elem_base,
                              int n_in, int n_out, const char *label) {
    uint32_t blck = ggml_blck_size(W->type);
    size_t   tsz  = ggml_type_size(W->type);
    uint64_t row0 = (uint64_t)((const uint8_t *)W->data - (const uint8_t *)g->map)
                  + (uint64_t)(elem_base / blck) * tsz;
    float *x = malloc((size_t)n_in * sizeof(float));
    for (int c = 0; c < n_in; c++) x[c] = 0.5f * sinf(0.01f * c);

    /* CPU reference first (also faults the checked rows' pages into residency). */
    int chk = n_out < 8 ? n_out : 8;
    float *scratch = malloc((size_t)n_in * sizeof(float));
    double cpu[8] = {0};
    for (int r = 0; r < chk; r++) {
        size_t e = elem_base + (size_t)r * n_in;
        quasar_dequant_row(W->type, (const uint8_t *)W->data + (e / blck) * tsz, scratch, n_in);
        double a = 0; for (int c = 0; c < n_in; c++) a += (double)scratch[c] * x[c];
        cpu[r] = a;
    }

    float *ym = malloc((size_t)n_out * sizeof(float));
    quasar_metal_gemv(W->type, row0, x, n_in, n_out, ym);

    double maxerr = 0, maxmag = 0;
    for (int r = 0; r < chk; r++) {
        double err = fabs(cpu[r] - ym[r]); if (err > maxerr) maxerr = err;
        if (fabs(cpu[r]) > maxmag) maxmag = fabs(cpu[r]);
    }
    printf("  %-20s %-5s  max|cpu-gpu|=%.2e (mag~%.2f)  %s\n",
           label, ggml_type_name(W->type), maxerr, maxmag, maxerr < 1e-2 ? "OK" : "FAIL");
    free(x); free(ym); free(scratch);
}

static int cmd_metal_selftest(const char *model) {
    if (quasar_metal_init() != 0) { printf("Metal: unavailable on this system\n"); return 1; }
    printf("Metal device: %s\n", quasar_metal_device_name());

    size_t pg = (size_t)getpagesize();
    int n_in = 256, n_out = 32;
    size_t need = (size_t)n_in * n_out * sizeof(float);
    size_t rlen = (need + pg - 1) & ~(pg - 1);
    void *wmem = NULL;
    if (posix_memalign(&wmem, pg, rlen)) { printf("alloc failed\n"); return 1; }
    float *W = wmem;
    for (int r = 0; r < n_out; r++) for (int c = 0; c < n_in; c++) W[r * n_in + c] = sinf(0.1f * r + 0.013f * c);
    float *x  = malloc((size_t)n_in * sizeof(float));
    for (int c = 0; c < n_in; c++) x[c] = cosf(0.02f * c);
    float *yc = malloc((size_t)n_out * sizeof(float)), *ym = malloc((size_t)n_out * sizeof(float));
    for (int r = 0; r < n_out; r++) { double a = 0; for (int c = 0; c < n_in; c++) a += (double)W[r*n_in+c] * x[c]; yc[r] = (float)a; }
    quasar_metal_set_weights(wmem, need);
    quasar_metal_gemv(GGML_TYPE_F32, 0, x, n_in, n_out, ym);
    double me = 0; for (int r = 0; r < n_out; r++) { double ee = fabs(yc[r] - ym[r]); if (ee > me) me = ee; }
    printf("  %-20s %-5s  max|cpu-gpu|=%.2e  %s\n", "synthetic matvec", "F32", me, me < 1e-3 ? "OK" : "FAIL");
    free(x); free(yc); free(ym); free(wmem);

    if (model) {
        char err[256];
        quasar_model *m = quasar_model_load(model, err, sizeof err);
        if (!m) { printf("model load: %s\n", err); return 1; }
        if (m->missing_tensors) { printf("model incomplete\n"); quasar_model_free(m); return 1; }
        gguf_ctx *g = m->gguf;
        int sw = quasar_metal_set_weights(g->map, g->map_size);
        printf("set_weights(%.2f GB) -> %s\n", g->map_size / 1e9, sw == 0 ? "ok" : "FAILED");
        int E = (int)m->hp.n_embd, FF = (int)m->hp.n_ff_exp, NV = (int)m->hp.n_vocab;
        printf("validating quant kernels vs CPU dequant on real tensors:\n");
        metal_test_tensor(g, m->output,                  0, E, NV, "output.weight");
        metal_test_tensor(g, m->layers[0].ffn_gate_exps, 0, E, FF, "blk0 gate exp0");
        metal_test_tensor(g, m->layers[0].attn_q,        0, E, (int)(m->hp.n_head * m->hp.head_dim), "blk0 attn_q");
        quasar_model_free(m);
    }
    return 0;
}

/* ---- requant ---------------------------------------------------------- */

static int cmd_requant(const char *in, const char *out) {
    char err[256];
    printf("requantizing routed experts to Q2_K:\n  %s\n  -> %s\n", in, out);
    if (quasar_requant_experts_q2k(in, out, err, sizeof err) != 0) {
        fprintf(stderr, "quasar: requant failed: %s\n", err);
        return 1;
    }
    return 0;
}

/* ---- serve (OpenAI-compatible HTTP server) ---------------------------- */

static int cmd_serve(int argc, char **argv) {
    /* argv[0] == "serve"; remaining args are the model path and flags. */
    const char *model = NULL;
    quasar_server_opts o = { .host = "127.0.0.1", .port = 8080, .ctx = 8192, .use_metal = 0 };
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (strcmp(a, "metal") == 0)                   o.use_metal = 1;
        else if (strcmp(a, "--host") == 0 && i + 1 < argc)  o.host = argv[++i];
        else if (strcmp(a, "--port") == 0 && i + 1 < argc)  o.port = atoi(argv[++i]);
        else if (strcmp(a, "--ctx")  == 0 && i + 1 < argc)  o.ctx  = atoi(argv[++i]);
        else if (a[0] != '-' && !model)                     model = a;
        else { fprintf(stderr, "quasar: unknown serve argument: %s\n", a); return 1; }
    }
    if (!model) {
        fprintf(stderr, "usage: quasar serve <model.gguf> [--host H] [--port N] [--ctx N] [metal]\n");
        return 1;
    }
    return quasar_serve(model, &o);
}

/* ---- main ------------------------------------------------------------- */

static void usage(void) {
    printf(
        "Quasar - Qwen3-30B-A3B inference engine (early scaffold)\n"
        "\n"
        "usage:\n"
        "  quasar inspect <model.gguf>              load + validate a qwen3moe GGUF\n"
        "  quasar tokenize <model.gguf> \"text\"      encode text -> token ids (+roundtrip)\n"
        "  quasar detokenize <model.gguf> <id>...   decode token ids -> text\n"
        "  quasar render [model.gguf]               show the Qwen3 ChatML template\n"
        "  quasar generate <model.gguf> \"text\" [n] [metal]  forward: top-5 + greedy n (add 'metal' for GPU)\n"
        "  quasar bench <model.gguf> [n] [metal] [\"text\"]   steady-state decode tok/s (warmup + timed)\n"
        "  quasar metal-selftest [model.gguf]       validate Metal gemv kernels vs CPU\n"
        "  quasar requant <in.gguf> <out.gguf>      crush routed experts to 2-bit (Q2_K), keep the rest\n"
        "  quasar serve <model.gguf> [--host H] [--port N] [--ctx N] [metal]  OpenAI-compatible HTTP server\n"
        "  quasar <model.gguf>                      shorthand for inspect\n"
        "  quasar --help                            this help\n");
}

int main(int argc, char **argv) {
    const char *cmd = argc >= 2 ? argv[1] : NULL;
    if (!cmd || strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0) {
        usage();
        return cmd ? 0 : 1;
    }
    if (strcmp(cmd, "inspect") == 0) {
        if (argc < 3) { fprintf(stderr, "usage: quasar inspect <model.gguf>\n"); return 1; }
        return cmd_inspect(argv[2]);
    }
    if (strcmp(cmd, "tokenize") == 0) {
        if (argc < 4) { fprintf(stderr, "usage: quasar tokenize <model.gguf> \"text\"\n"); return 1; }
        return cmd_tokenize(argv[2], argv[3]);
    }
    if (strcmp(cmd, "detokenize") == 0) {
        if (argc < 4) { fprintf(stderr, "usage: quasar detokenize <model.gguf> <id>...\n"); return 1; }
        return cmd_detokenize(argv[2], argc - 3, &argv[3]);
    }
    if (strcmp(cmd, "render") == 0) {
        return cmd_render(argc >= 3 ? argv[2] : NULL);
    }
    if (strcmp(cmd, "generate") == 0) {
        if (argc < 4) { fprintf(stderr, "usage: quasar generate <model.gguf> \"prompt\" [n_tokens] [metal]\n"); return 1; }
        int nt = 0, um = 0;
        for (int i = 4; i < argc; i++) { if (strcmp(argv[i], "metal") == 0) um = 1; else nt = atoi(argv[i]); }
        return cmd_generate(argv[2], argv[3], nt, um);
    }
    if (strcmp(cmd, "bench") == 0) {
        if (argc < 3) { fprintf(stderr, "usage: quasar bench <model.gguf> [n_tokens] [metal] [\"prompt\"]\n"); return 1; }
        int nt = 0, um = 0; const char *bp = NULL;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "metal") == 0) um = 1;
            else if (atoi(argv[i]) > 0) nt = atoi(argv[i]);
            else bp = argv[i];
        }
        return cmd_bench(argv[2], nt, um, bp);
    }
    if (strcmp(cmd, "metal-selftest") == 0) {
        return cmd_metal_selftest(argc >= 3 ? argv[2] : NULL);
    }
    if (strcmp(cmd, "requant") == 0) {
        if (argc < 4) { fprintf(stderr, "usage: quasar requant <in.gguf> <out.gguf>\n"); return 1; }
        return cmd_requant(argv[2], argv[3]);
    }
    if (strcmp(cmd, "serve") == 0) {
        return cmd_serve(argc - 1, &argv[1]);
    }
    return cmd_inspect(argv[1]);     /* default: treat the arg as a model path */
}
