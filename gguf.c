/* gguf.c - see gguf.h. */
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ---- ggml type traits ------------------------------------------------- */

typedef struct { const char *name; uint32_t blck; uint32_t size; } type_trait;

static const type_trait TRAITS[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]     = { "F32",       1,   4 },
    [GGML_TYPE_F16]     = { "F16",       1,   2 },
    [GGML_TYPE_BF16]    = { "BF16",      1,   2 },
    [GGML_TYPE_Q4_0]    = { "Q4_0",     32,  18 },
    [GGML_TYPE_Q4_1]    = { "Q4_1",     32,  20 },
    [GGML_TYPE_Q5_0]    = { "Q5_0",     32,  22 },
    [GGML_TYPE_Q5_1]    = { "Q5_1",     32,  24 },
    [GGML_TYPE_Q8_0]    = { "Q8_0",     32,  34 },
    [GGML_TYPE_Q8_1]    = { "Q8_1",     32,  36 },
    [GGML_TYPE_Q2_K]    = { "Q2_K",    256,  84 },
    [GGML_TYPE_Q3_K]    = { "Q3_K",    256, 110 },
    [GGML_TYPE_Q4_K]    = { "Q4_K",    256, 144 },
    [GGML_TYPE_Q5_K]    = { "Q5_K",    256, 176 },
    [GGML_TYPE_Q6_K]    = { "Q6_K",    256, 210 },
    [GGML_TYPE_Q8_K]    = { "Q8_K",    256, 292 },
    [GGML_TYPE_IQ2_XXS] = { "IQ2_XXS", 256,  66 },
    [GGML_TYPE_IQ2_XS]  = { "IQ2_XS",  256,  74 },
    [GGML_TYPE_IQ4_XS]  = { "IQ4_XS",  256, 136 },
};

const char *ggml_type_name(uint32_t t) {
    if (t < GGML_TYPE_COUNT && TRAITS[t].name) return TRAITS[t].name;
    return "UNKNOWN";
}
uint32_t ggml_blck_size(uint32_t t) {
    return (t < GGML_TYPE_COUNT) ? TRAITS[t].blck : 0;
}
size_t ggml_type_size(uint32_t t) {
    return (t < GGML_TYPE_COUNT) ? TRAITS[t].size : 0;
}

/* ---- bounds-checked cursor over the mmap ------------------------------ */

typedef struct { const uint8_t *p, *end; int err; } cur;

static uint8_t c_u8(cur *c) {
    if (c->err || c->p + 1 > c->end) { c->err = 1; return 0; }
    return *c->p++;
}
static uint16_t c_u16(cur *c) {
    uint16_t v = 0;
    if (c->err || c->p + 2 > c->end) { c->err = 1; return 0; }
    memcpy(&v, c->p, 2); c->p += 2; return v;
}
static uint32_t c_u32(cur *c) {
    uint32_t v = 0;
    if (c->err || c->p + 4 > c->end) { c->err = 1; return 0; }
    memcpy(&v, c->p, 4); c->p += 4; return v;
}
static uint64_t c_u64(cur *c) {
    uint64_t v = 0;
    if (c->err || c->p + 8 > c->end) { c->err = 1; return 0; }
    memcpy(&v, c->p, 8); c->p += 8; return v;
}
static const uint8_t *c_bytes(cur *c, uint64_t n) {
    if (c->err || n > (uint64_t)(c->end - c->p)) { c->err = 1; return NULL; }
    const uint8_t *r = c->p; c->p += n; return r;
}
static const char *read_gstr(cur *c, uint64_t *len) {
    uint64_t n = c_u64(c);
    const uint8_t *b = c_bytes(c, n);
    if (!b) { *len = 0; return NULL; }
    *len = n; return (const char *)b;
}

static size_t gguf_scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL:   return 1;
        case GGUF_T_UINT16: case GGUF_T_INT16:                   return 2;
        case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
        case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
        default: return 0;
    }
}

static void read_scalar(cur *c, uint32_t type, gguf_kv *kv) {
    kv->type = type;
    switch (type) {
        case GGUF_T_UINT8:  { uint8_t  v = c_u8(c);            kv->u = v; kv->i = v; kv->d = v; } break;
        case GGUF_T_INT8:   { int8_t   v = (int8_t)c_u8(c);    kv->u = (uint64_t)v; kv->i = v; kv->d = v; } break;
        case GGUF_T_UINT16: { uint16_t v = c_u16(c);           kv->u = v; kv->i = v; kv->d = v; } break;
        case GGUF_T_INT16:  { int16_t  v = (int16_t)c_u16(c);  kv->u = (uint64_t)v; kv->i = v; kv->d = v; } break;
        case GGUF_T_UINT32: { uint32_t v = c_u32(c);           kv->u = v; kv->i = v; kv->d = v; } break;
        case GGUF_T_INT32:  { int32_t  v = (int32_t)c_u32(c);  kv->u = (uint64_t)v; kv->i = v; kv->d = v; } break;
        case GGUF_T_BOOL:   { uint8_t  v = c_u8(c);            kv->u = v ? 1 : 0; kv->i = kv->u; kv->d = (double)kv->u; } break;
        case GGUF_T_UINT64: { uint64_t v = c_u64(c);           kv->u = v; kv->i = (int64_t)v; kv->d = (double)v; } break;
        case GGUF_T_INT64:  { int64_t  v = (int64_t)c_u64(c);  kv->u = (uint64_t)v; kv->i = v; kv->d = (double)v; } break;
        case GGUF_T_FLOAT32:{ uint32_t b = c_u32(c); float  f; memcpy(&f, &b, 4); kv->d = f; kv->u = (uint64_t)f; kv->i = (int64_t)f; } break;
        case GGUF_T_FLOAT64:{ uint64_t b = c_u64(c); double f; memcpy(&f, &b, 8); kv->d = f; kv->u = (uint64_t)f; kv->i = (int64_t)f; } break;
        default: c->err = 1;
    }
}

static void read_value(cur *c, uint32_t type, gguf_kv *kv) {
    if (type == GGUF_T_STRING) {
        kv->type = type;
        kv->sptr = read_gstr(c, &kv->slen);
        return;
    }
    if (type == GGUF_T_ARRAY) {
        kv->type = type;
        uint32_t et = c_u32(c);
        uint64_t n  = c_u64(c);
        kv->arr_type = et;
        kv->arr_n    = n;
        kv->arr_data = c->p;
        if (et == GGUF_T_STRING) {
            const uint8_t *start = c->p;
            for (uint64_t k = 0; k < n && !c->err; k++) { uint64_t l; read_gstr(c, &l); }
            kv->arr_bytes = (uint64_t)(c->p - start);
        } else if (et == GGUF_T_ARRAY) {
            c->err = 1;                          /* nested arrays unsupported */
        } else {
            size_t sz = gguf_scalar_size(et);
            if (sz == 0) { c->err = 1; return; }
            uint64_t bytes = n * sz;
            if (!c_bytes(c, bytes)) return;
            kv->arr_bytes = bytes;
        }
        return;
    }
    read_scalar(c, type, kv);
}

/* ---- open / close ----------------------------------------------------- */

static uint64_t align_up(uint64_t x, uint64_t a) {
    return (a == 0) ? x : ((x + a - 1) & ~(a - 1));
}

gguf_ctx *gguf_open(const char *path) {
    gguf_ctx *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) { snprintf(g->err, sizeof g->err, "open(%s): %s", path, strerror(errno)); return g; }

    struct stat st;
    if (fstat(g->fd, &st) != 0) { snprintf(g->err, sizeof g->err, "fstat: %s", strerror(errno)); return g; }
    g->map_size = (size_t)st.st_size;
    if (g->map_size < 24) { snprintf(g->err, sizeof g->err, "file too small to be GGUF"); return g; }

    g->map = mmap(NULL, g->map_size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (g->map == MAP_FAILED) { g->map = NULL; snprintf(g->err, sizeof g->err, "mmap: %s", strerror(errno)); return g; }

    cur c = { .p = g->map, .end = (const uint8_t *)g->map + g->map_size, .err = 0 };

    if (c_u32(&c) != 0x46554747u) { snprintf(g->err, sizeof g->err, "not a GGUF file (bad magic)"); return g; }
    g->version   = c_u32(&c);
    g->n_tensors = c_u64(&c);
    g->n_kv      = c_u64(&c);
    if (c.err) { snprintf(g->err, sizeof g->err, "truncated header"); return g; }
    if (g->n_kv > 10000000ull || g->n_tensors > 10000000ull) {
        snprintf(g->err, sizeof g->err, "implausible counts (kv=%llu tensors=%llu)",
                 (unsigned long long)g->n_kv, (unsigned long long)g->n_tensors);
        return g;
    }

    g->kv = calloc(g->n_kv ? g->n_kv : 1, sizeof(gguf_kv));
    if (!g->kv) { snprintf(g->err, sizeof g->err, "oom kv"); return g; }
    for (uint64_t k = 0; k < g->n_kv; k++) {
        uint64_t klen; const char *kp = read_gstr(&c, &klen);
        if (c.err || !kp) { snprintf(g->err, sizeof g->err, "bad kv key at %llu", (unsigned long long)k); return g; }
        g->kv[k].key = strndup(kp, klen);
        uint32_t vt = c_u32(&c);
        read_value(&c, vt, &g->kv[k]);
        if (c.err) { snprintf(g->err, sizeof g->err, "bad kv value for '%s'", g->kv[k].key ? g->kv[k].key : "?"); return g; }
    }

    g->kv_end = (uint64_t)(c.p - (const uint8_t *)g->map);

    g->alignment = gguf_u32_or(g, "general.alignment", 32);
    if (g->alignment == 0) g->alignment = 32;

    g->tensors = calloc(g->n_tensors ? g->n_tensors : 1, sizeof(gguf_tensor));
    if (!g->tensors) { snprintf(g->err, sizeof g->err, "oom tensors"); return g; }
    for (uint64_t t = 0; t < g->n_tensors; t++) {
        gguf_tensor *ti = &g->tensors[t];
        uint64_t nlen; const char *np = read_gstr(&c, &nlen);
        if (c.err || !np) { snprintf(g->err, sizeof g->err, "bad tensor name at %llu", (unsigned long long)t); return g; }
        ti->name   = strndup(np, nlen);
        ti->n_dims = c_u32(&c);
        if (ti->n_dims > 4) { snprintf(g->err, sizeof g->err, "tensor '%s' has %u dims (>4)", ti->name, ti->n_dims); return g; }
        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) { ti->dims[d] = c_u64(&c); ti->n_elements *= ti->dims[d]; }
        ti->type   = c_u32(&c);
        ti->offset = c_u64(&c);
        if (c.err) { snprintf(g->err, sizeof g->err, "truncated tensor info for '%s'", ti->name); return g; }
    }

    /* data section begins at the first alignment boundary after the headers. */
    uint64_t header_end = (uint64_t)(c.p - (const uint8_t *)g->map);
    g->data_off = align_up(header_end, g->alignment);

    for (uint64_t t = 0; t < g->n_tensors; t++) {
        gguf_tensor *ti = &g->tensors[t];
        uint32_t blck = ggml_blck_size(ti->type);
        size_t   tsz  = ggml_type_size(ti->type);
        if (blck && tsz && (ti->n_elements % blck) == 0) {
            ti->nbytes = ti->n_elements / blck * tsz;
        } else {
            ti->nbytes = 0;             /* unknown type or unaligned element count */
        }
        uint64_t abs = g->data_off + ti->offset;
        if (ti->nbytes && abs + ti->nbytes <= g->map_size)
            ti->data = (const uint8_t *)g->map + abs;
        else
            ti->data = NULL;
    }

    g->err[0] = 0;                      /* success */
    return g;
}

void gguf_close(gguf_ctx *g) {
    if (!g) return;
    if (g->kv) {
        for (uint64_t k = 0; k < g->n_kv; k++) free(g->kv[k].key);
        free(g->kv);
    }
    if (g->tensors) {
        for (uint64_t t = 0; t < g->n_tensors; t++) free(g->tensors[t].name);
        free(g->tensors);
    }
    if (g->map && g->map != MAP_FAILED) munmap(g->map, g->map_size);
    if (g->fd >= 0) close(g->fd);
    free(g);
}

/* ---- lookups & typed accessors --------------------------------------- */

const gguf_kv *gguf_find_kv(const gguf_ctx *g, const char *key) {
    for (uint64_t k = 0; k < g->n_kv; k++)
        if (g->kv[k].key && strcmp(g->kv[k].key, key) == 0) return &g->kv[k];
    return NULL;
}
const gguf_tensor *gguf_find_tensor(const gguf_ctx *g, const char *name) {
    for (uint64_t t = 0; t < g->n_tensors; t++)
        if (g->tensors[t].name && strcmp(g->tensors[t].name, name) == 0) return &g->tensors[t];
    return NULL;
}

static bool numeric(const gguf_kv *kv) {
    return kv && kv->type != GGUF_T_STRING && kv->type != GGUF_T_ARRAY;
}

bool gguf_u32(const gguf_ctx *g, const char *key, uint32_t *out) {
    const gguf_kv *kv = gguf_find_kv(g, key); if (!numeric(kv)) return false;
    *out = (uint32_t)kv->u; return true;
}
bool gguf_u64(const gguf_ctx *g, const char *key, uint64_t *out) {
    const gguf_kv *kv = gguf_find_kv(g, key); if (!numeric(kv)) return false;
    *out = kv->u; return true;
}
bool gguf_i32(const gguf_ctx *g, const char *key, int32_t *out) {
    const gguf_kv *kv = gguf_find_kv(g, key); if (!numeric(kv)) return false;
    *out = (int32_t)kv->i; return true;
}
bool gguf_f32(const gguf_ctx *g, const char *key, float *out) {
    const gguf_kv *kv = gguf_find_kv(g, key); if (!numeric(kv)) return false;
    *out = (float)kv->d; return true;
}
bool gguf_bool(const gguf_ctx *g, const char *key, bool *out) {
    const gguf_kv *kv = gguf_find_kv(g, key); if (!numeric(kv)) return false;
    *out = kv->u != 0; return true;
}
bool gguf_str(const gguf_ctx *g, const char *key, const char **ptr, uint64_t *len) {
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv || kv->type != GGUF_T_STRING) return false;
    *ptr = kv->sptr; *len = kv->slen; return true;
}
uint32_t gguf_u32_or(const gguf_ctx *g, const char *key, uint32_t def) {
    uint32_t v; return gguf_u32(g, key, &v) ? v : def;
}
float gguf_f32_or(const gguf_ctx *g, const char *key, float def) {
    float v; return gguf_f32(g, key, &v) ? v : def;
}

/* ---- array helpers ---------------------------------------------------- */

uint64_t gguf_arr_n(const gguf_kv *kv) {
    return (kv && kv->type == GGUF_T_ARRAY) ? kv->arr_n : 0;
}

bool gguf_arr_i32(const gguf_kv *kv, uint64_t i, int32_t *out) {
    if (!kv || kv->type != GGUF_T_ARRAY || i >= kv->arr_n) return false;
    if (kv->arr_type != GGUF_T_INT32 && kv->arr_type != GGUF_T_UINT32) return false;
    const uint8_t *p = kv->arr_data + i * 4;
    if (p + 4 > kv->arr_data + kv->arr_bytes) return false;
    int32_t v; memcpy(&v, p, 4); *out = v; return true;
}

uint64_t gguf_arr_foreach_str(const gguf_kv *kv, gguf_str_iter_fn fn, void *ud) {
    if (!kv || kv->type != GGUF_T_ARRAY || kv->arr_type != GGUF_T_STRING) return 0;
    const uint8_t *p   = kv->arr_data;
    const uint8_t *end = kv->arr_data + kv->arr_bytes;
    uint64_t k = 0;
    for (; k < kv->arr_n; k++) {
        if (p + 8 > end) break;
        uint64_t l; memcpy(&l, p, 8); p += 8;
        if (p + l > end) break;
        if (fn && fn(ud, k, (const char *)p, l)) { k++; break; }
        p += l;
    }
    return k;
}
