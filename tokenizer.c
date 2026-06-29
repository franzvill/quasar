/* tokenizer.c - see tokenizer.h.
 *
 * Pipeline (matches Qwen2/Qwen3): split off special tokens -> pre-tokenize each
 * span with the Qwen regex -> map bytes to the GPT-2 byte-level alphabet ->
 * merge-rank BPE -> vocab lookup.
 *
 * NOTE: the pre-tokenizer's Unicode letter/number/space classification uses
 * pragmatic ranges (ASCII + common scripts), to be validated and hardened
 * against the real Qwen3 vocab once weights are available (see ARCHITECTURE.md
 * milestone #4). The BPE core, byte mapping, special-token handling, and decode
 * are exact. */
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* ---- tiny string->int32 hash map (open addressing, FNV-1a) ------------ */

typedef struct { char *key; uint32_t klen; int32_t val; int used; } hm_entry;
typedef struct { hm_entry *e; size_t cap, len; } hashmap;

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h;
}
static void hm_init(hashmap *m, size_t cap) {
    size_t p = 16;                 /* capacity must be a power of two: the */
    while (p < cap) p <<= 1;        /* probe step masks with (cap - 1).     */
    m->cap = p; m->len = 0;
    m->e = calloc(m->cap, sizeof(hm_entry));
}
static void hm_free(hashmap *m) {
    if (m->e) { for (size_t i = 0; i < m->cap; i++) free(m->e[i].key); free(m->e); }
    m->e = NULL; m->cap = m->len = 0;
}
static void hm_put_raw(hashmap *m, char *key, uint32_t klen, int32_t val);
static void hm_grow(hashmap *m) {
    size_t oc = m->cap; hm_entry *oe = m->e;
    m->cap *= 2; m->len = 0; m->e = calloc(m->cap, sizeof(hm_entry));
    for (size_t i = 0; i < oc; i++) if (oe[i].used) hm_put_raw(m, oe[i].key, oe[i].klen, oe[i].val);
    free(oe);
}
static void hm_put_raw(hashmap *m, char *key, uint32_t klen, int32_t val) {
    size_t i = fnv1a(key, klen) & (m->cap - 1);
    for (;;) {
        if (!m->e[i].used) {
            m->e[i].used = 1; m->e[i].key = key; m->e[i].klen = klen; m->e[i].val = val;
            m->len++;
            if (m->len * 10 >= m->cap * 7) hm_grow(m);
            return;
        }
        if (m->e[i].klen == klen && memcmp(m->e[i].key, key, klen) == 0) {
            m->e[i].val = val; free(key); return;   /* duplicate: overwrite */
        }
        i = (i + 1) & (m->cap - 1);
    }
}
static void hm_put(hashmap *m, const char *key, size_t klen, int32_t val) {
    char *k = malloc(klen ? klen : 1);
    memcpy(k, key, klen);
    hm_put_raw(m, k, (uint32_t)klen, val);
}
static bool hm_get(const hashmap *m, const char *key, size_t klen, int32_t *out) {
    if (!m->cap) return false;
    size_t i = fnv1a(key, klen) & (m->cap - 1);
    for (size_t probe = 0; probe < m->cap; probe++) {
        if (!m->e[i].used) return false;
        if (m->e[i].klen == klen && memcmp(m->e[i].key, key, klen) == 0) { *out = m->e[i].val; return true; }
        i = (i + 1) & (m->cap - 1);
    }
    return false;
}

/* ---- tokenizer state -------------------------------------------------- */

typedef struct { int32_t id; char *text; int len; } special_tok;

struct quasar_tokenizer {
    uint32_t  n_vocab;
    char    **id2tok;
    int      *id2len;
    hashmap   tok2id;
    hashmap   merges;        /* "left right" -> rank */
    special_tok *specials;   /* sorted by text length desc */
    int       n_special;
    int32_t   bos_id, eos_id;
    uint32_t  byte_to_cp[256];
    int16_t   cp_to_byte[512];
};

static void build_byte_map(quasar_tokenizer *t) {
    int taken[256] = {0};
    for (int b = 33;   b <= 126;  b++) taken[b] = 1;   /* '!'..'~' */
    for (int b = 0xA1; b <= 0xAC; b++) taken[b] = 1;
    for (int b = 0xAE; b <= 0xFF; b++) taken[b] = 1;
    int extra = 0;
    for (int b = 0; b < 256; b++)
        t->byte_to_cp[b] = taken[b] ? (uint32_t)b : (uint32_t)(256 + extra++);
    for (int i = 0; i < 512; i++) t->cp_to_byte[i] = -1;
    for (int b = 0; b < 256; b++) {
        uint32_t cp = t->byte_to_cp[b];
        if (cp < 512) t->cp_to_byte[cp] = (int16_t)b;
    }
}

/* ---- UTF-8 + Unicode class helpers ------------------------------------ */

static uint32_t utf8_next(const unsigned char *s, size_t len, size_t *i) {
    unsigned char c = s[*i];
    if (c < 0x80)                         { *i += 1; return c; }
    if ((c >> 5) == 0x6  && *i + 1 < len) { uint32_t cp = ((c & 0x1F) << 6) | (s[*i+1] & 0x3F); *i += 2; return cp; }
    if ((c >> 4) == 0xE  && *i + 2 < len) { uint32_t cp = ((c & 0x0F) << 12) | ((s[*i+1] & 0x3F) << 6) | (s[*i+2] & 0x3F); *i += 3; return cp; }
    if ((c >> 3) == 0x1E && *i + 3 < len) { uint32_t cp = ((c & 0x07) << 18) | ((s[*i+1] & 0x3F) << 12) | ((s[*i+2] & 0x3F) << 6) | (s[*i+3] & 0x3F); *i += 4; return cp; }
    *i += 1; return c;                    /* invalid: treat as raw byte */
}
static int cp_to_utf8(uint32_t cp, char *out) {
    if (cp < 0x80)    { out[0] = (char)cp; return 1; }
    if (cp < 0x800)   { out[0] = (char)(0xC0 | (cp >> 6));  out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

static bool cp_is_letter(uint32_t c) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return true;
    if (c < 0x80) return false;
    if (c >= 0x00C0 && c <= 0x024F) return true;   /* Latin-1 supp + Latin Ext A/B */
    if (c >= 0x0370 && c <= 0x03FF) return true;   /* Greek */
    if (c >= 0x0400 && c <= 0x04FF) return true;   /* Cyrillic */
    if (c >= 0x0590 && c <= 0x05FF) return true;   /* Hebrew */
    if (c >= 0x0600 && c <= 0x06FF) return true;   /* Arabic */
    if (c >= 0x0900 && c <= 0x097F) return true;   /* Devanagari */
    if (c >= 0x3040 && c <= 0x30FF) return true;   /* Hiragana + Katakana */
    if (c >= 0x3400 && c <= 0x9FFF) return true;   /* CJK */
    if (c >= 0xAC00 && c <= 0xD7A3) return true;   /* Hangul */
    if (c >= 0xF900 && c <= 0xFAFF) return true;   /* CJK compat */
    if (c >= 0x20000 && c <= 0x2FA1F) return true; /* CJK ext */
    return false;
}
static bool cp_is_number(uint32_t c) {
    if (c >= '0' && c <= '9') return true;
    if (c >= 0x0660 && c <= 0x0669) return true;   /* Arabic-Indic */
    if (c >= 0x06F0 && c <= 0x06F9) return true;   /* Ext Arabic-Indic */
    if (c >= 0xFF10 && c <= 0xFF19) return true;   /* fullwidth */
    return false;
}
static bool cp_is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C ||
           c == 0x85 || c == 0xA0 || c == 0x2028 || c == 0x2029 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x3000;
}

/* ---- loading ---------------------------------------------------------- */

static int on_token(void *ud, uint64_t idx, const char *s, uint64_t len) {
    quasar_tokenizer *t = ud;
    if (idx >= t->n_vocab) return 0;
    t->id2tok[idx] = malloc(len + 1);
    memcpy(t->id2tok[idx], s, len); t->id2tok[idx][len] = 0;
    t->id2len[idx] = (int)len;
    hm_put(&t->tok2id, s, len, (int32_t)idx);
    return 0;
}
static int on_merge(void *ud, uint64_t idx, const char *s, uint64_t len) {
    quasar_tokenizer *t = ud;
    hm_put(&t->merges, s, len, (int32_t)idx);   /* rank = position in list */
    return 0;
}
static int special_cmp(const void *a, const void *b) {
    return ((const special_tok *)b)->len - ((const special_tok *)a)->len;   /* desc */
}

quasar_tokenizer *quasar_tokenizer_load(const gguf_ctx *g, char *err, size_t errlen) {
    const gguf_kv *toks = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!toks || toks->type != GGUF_T_ARRAY || toks->arr_type != GGUF_T_STRING) {
        snprintf(err, errlen, "missing tokenizer.ggml.tokens"); return NULL;
    }
    quasar_tokenizer *t = calloc(1, sizeof *t);
    if (!t) { snprintf(err, errlen, "oom"); return NULL; }

    t->n_vocab = (uint32_t)gguf_arr_n(toks);
    t->id2tok  = calloc(t->n_vocab, sizeof(char *));
    t->id2len  = calloc(t->n_vocab, sizeof(int));
    hm_init(&t->tok2id, t->n_vocab * 2);
    gguf_arr_foreach_str(toks, on_token, t);

    const gguf_kv *merges = gguf_find_kv(g, "tokenizer.ggml.merges");
    hm_init(&t->merges, merges ? (size_t)gguf_arr_n(merges) * 2 : 16);
    if (merges) gguf_arr_foreach_str(merges, on_merge, t);

    build_byte_map(t);

    int32_t id;
    t->bos_id = gguf_i32(g, "tokenizer.ggml.bos_token_id", &id) ? id : -1;
    t->eos_id = gguf_i32(g, "tokenizer.ggml.eos_token_id", &id) ? id : -1;

    /* Special tokens: those flagged CONTROL(3)/USER_DEFINED(4) in token_type,
     * else fall back to the <|...|> convention. */
    const gguf_kv *ttype = gguf_find_kv(g, "tokenizer.ggml.token_type");
    t->specials = calloc(t->n_vocab ? t->n_vocab : 1, sizeof(special_tok));
    t->n_special = 0;
    for (uint32_t i = 0; i < t->n_vocab; i++) {
        if (!t->id2tok[i]) continue;
        bool special = false;
        int32_t tv;
        if (ttype && gguf_arr_i32(ttype, i, &tv)) special = (tv == 3 || tv == 4);
        else {
            int l = t->id2len[i];
            special = (l >= 4 && t->id2tok[i][0] == '<' && t->id2tok[i][1] == '|' &&
                       t->id2tok[i][l-2] == '|' && t->id2tok[i][l-1] == '>');
        }
        if (special && t->id2len[i] > 0) {
            t->specials[t->n_special].id   = (int32_t)i;
            t->specials[t->n_special].text = t->id2tok[i];
            t->specials[t->n_special].len  = t->id2len[i];
            t->n_special++;
        }
    }
    qsort(t->specials, t->n_special, sizeof(special_tok), special_cmp);

    (void)err;
    return t;
}

void quasar_tokenizer_free(quasar_tokenizer *t) {
    if (!t) return;
    if (t->id2tok) { for (uint32_t i = 0; i < t->n_vocab; i++) free(t->id2tok[i]); free(t->id2tok); }
    free(t->id2len);
    hm_free(&t->tok2id);
    hm_free(&t->merges);
    free(t->specials);
    free(t);
}

/* ---- output id vector ------------------------------------------------- */

typedef struct { int32_t *v; size_t n, cap; } idvec;
static void idv_push(idvec *a, int32_t x) {
    if (a->n == a->cap) { a->cap = a->cap ? a->cap * 2 : 64; a->v = realloc(a->v, a->cap * sizeof(int32_t)); }
    a->v[a->n++] = x;
}

/* ---- byte-level BPE over one pre-token span --------------------------- */

typedef struct { uint32_t off, len; } sym;

static void bpe_piece(quasar_tokenizer *t, const char *bytes, size_t len, idvec *out) {
    if (len == 0) return;

    /* map raw bytes to the byte-level alphabet (UTF-8 of byte_to_cp[b]) */
    char  *S   = malloc(len * 4 + 1);
    sym   *sy  = malloc(len * sizeof(sym));
    size_t soff = 0; int nsym = 0;
    for (size_t i = 0; i < len; i++) {
        int n = cp_to_utf8(t->byte_to_cp[(unsigned char)bytes[i]], S + soff);
        sy[nsym].off = (uint32_t)soff; sy[nsym].len = (uint32_t)n;
        soff += n; nsym++;
    }

    char *kbuf = malloc(soff * 2 + 2);
    for (;;) {
        int best_rank = INT_MAX, bi = -1;
        for (int i = 0; i + 1 < nsym; i++) {
            uint32_t la = sy[i].len, lb = sy[i+1].len;
            memcpy(kbuf, S + sy[i].off, la);
            kbuf[la] = ' ';
            memcpy(kbuf + la + 1, S + sy[i+1].off, lb);
            int32_t rank;
            if (hm_get(&t->merges, kbuf, la + 1 + lb, &rank) && rank < best_rank) { best_rank = rank; bi = i; }
        }
        if (bi < 0) break;
        sy[bi].len += sy[bi+1].len;
        memmove(&sy[bi+1], &sy[bi+2], (nsym - bi - 2) * sizeof(sym));
        nsym--;
    }

    for (int i = 0; i < nsym; i++) {
        int32_t id;
        if (hm_get(&t->tok2id, S + sy[i].off, sy[i].len, &id)) {
            idv_push(out, id);
        } else {
            /* fall back to single byte-level base tokens (always in vocab) */
            size_t p = 0;
            while (p < sy[i].len) {
                size_t step = p; (void)utf8_next((const unsigned char *)(S + sy[i].off), sy[i].len, &step);
                int32_t bid;
                if (hm_get(&t->tok2id, S + sy[i].off + p, step - p, &bid)) idv_push(out, bid);
                p = step;
            }
        }
    }
    free(kbuf); free(sy); free(S);
}

/* ---- Qwen pre-tokenizer ----------------------------------------------- */
/* Splits a special-token-free byte span into pre-token byte ranges, applying the
 * Qwen2/Qwen3 regex alternatives in priority order, then BPE-encodes each. */

static void encode_span(quasar_tokenizer *t, const char *bytes, size_t len, idvec *out) {
    if (len == 0) return;

    /* decode to codepoints with byte offsets */
    uint32_t *cp  = malloc((len + 1) * sizeof(uint32_t));
    size_t   *off = malloc((len + 1) * sizeof(size_t));
    size_t n = 0, bi = 0;
    while (bi < len) { off[n] = bi; cp[n] = utf8_next((const unsigned char *)bytes, len, &bi); n++; }
    off[n] = len;

    size_t k = 0;
    while (k < n) {
        size_t start = k, m;

        /* 1. contractions: 's 't 're 've 'm 'll 'd  (case-insensitive) */
        if (cp[k] == '\'' && k + 1 < n) {
            uint32_t c1 = cp[k+1] | 0x20;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { m = k + 2; goto emit; }
            if (k + 2 < n) {
                uint32_t c2 = cp[k+2] | 0x20;
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { m = k + 3; goto emit; }
            }
        }
        /* 2. [^\r\n\p{L}\p{N}]? \p{L}+ */
        {
            int prefix = (cp[k] != '\r' && cp[k] != '\n' && !cp_is_letter(cp[k]) && !cp_is_number(cp[k]));
            size_t q = prefix ? k + 1 : k;
            if (q < n && cp_is_letter(cp[q])) {
                m = q; while (m < n && cp_is_letter(cp[m])) m++;
                goto emit;
            }
        }
        /* 3. \p{N}  (single digit) */
        if (cp_is_number(cp[k])) { m = k + 1; goto emit; }
        /* 4.  ?[^\s\p{L}\p{N}]+[\r\n]* */
        {
            int sp = (cp[k] == ' ');
            size_t q = sp ? k + 1 : k;
            if (q < n && !cp_is_space(cp[q]) && !cp_is_letter(cp[q]) && !cp_is_number(cp[q])) {
                m = q; while (m < n && !cp_is_space(cp[m]) && !cp_is_letter(cp[m]) && !cp_is_number(cp[m])) m++;
                while (m < n && (cp[m] == '\r' || cp[m] == '\n')) m++;
                goto emit;
            }
        }
        /* 5/6/7. whitespace runs */
        if (cp_is_space(cp[k])) {
            size_t e = k; long last_nl = -1;
            while (e < n && cp_is_space(cp[e])) { if (cp[e] == '\r' || cp[e] == '\n') last_nl = (long)e; e++; }
            if (last_nl >= 0)        m = (size_t)last_nl + 1;   /* 5: \s*[\r\n]+ */
            else if (e == n)         m = e;                     /* 6: trailing run */
            else if (e - 1 > k)      m = e - 1;                 /* 6: leave 1 space for next token */
            else                     m = e;                     /* 7: lone space (e.g. before a digit) */
            goto emit;
        }
        m = k + 1;   /* safety: never stall */
    emit:
        bpe_piece(t, bytes + off[start], off[m] - off[start], out);
        k = m;
    }
    free(cp); free(off);
}

/* ---- public encode / decode ------------------------------------------ */

int32_t *quasar_tokenize(quasar_tokenizer *t, const char *text, size_t len,
                         bool add_bos, size_t *out_n) {
    idvec out = {0};
    if (add_bos && t->bos_id >= 0) idv_push(&out, t->bos_id);

    size_t i = 0;
    while (i < len) {
        /* longest special-token match at i */
        int matched = -1;
        for (int s = 0; s < t->n_special; s++) {
            int sl = t->specials[s].len;
            if ((size_t)sl <= len - i && memcmp(text + i, t->specials[s].text, sl) == 0) { matched = s; break; }
        }
        if (matched >= 0) { idv_push(&out, t->specials[matched].id); i += t->specials[matched].len; continue; }

        /* accumulate up to the next special-token start, then encode the span */
        size_t j = i;
        while (j < len) {
            int hit = 0;
            for (int s = 0; s < t->n_special; s++) {
                int sl = t->specials[s].len;
                if ((size_t)sl <= len - j && memcmp(text + j, t->specials[s].text, sl) == 0) { hit = 1; break; }
            }
            if (hit) break;
            j++;
        }
        encode_span(t, text + i, j - i, &out);
        i = j;
    }
    *out_n = out.n;
    return out.v;
}

int quasar_token_to_bytes(quasar_tokenizer *t, int32_t id, char *buf, int cap) {
    if (id < 0 || (uint32_t)id >= t->n_vocab || !t->id2tok[id]) return 0;
    const char *s = t->id2tok[id];
    int sl = t->id2len[id], w = 0;
    size_t p = 0;
    while (p < (size_t)sl) {
        uint32_t c = utf8_next((const unsigned char *)s, sl, &p);
        if (c < 512 && t->cp_to_byte[c] >= 0) {
            if (w < cap) buf[w] = (char)t->cp_to_byte[c];
            w++;
        } else {
            char tmp[4]; int n = cp_to_utf8(c, tmp);
            for (int b = 0; b < n; b++) { if (w < cap) buf[w] = tmp[b]; w++; }
        }
    }
    return w > cap ? cap : w;
}

char *quasar_detokenize(quasar_tokenizer *t, const int32_t *ids, size_t n, size_t *out_len) {
    size_t cap = 256, len = 0;
    char *s = malloc(cap);
    for (size_t i = 0; i < n; i++) {
        char tmp[1024];
        int w = quasar_token_to_bytes(t, ids[i], tmp, sizeof tmp);
        if (len + w + 1 > cap) { while (len + w + 1 > cap) cap *= 2; s = realloc(s, cap); }
        memcpy(s + len, tmp, w); len += w;
    }
    s[len] = 0;
    if (out_len) *out_len = len;
    return s;
}

uint32_t quasar_vocab_size(const quasar_tokenizer *t) { return t->n_vocab; }
int32_t  quasar_token_bos(const quasar_tokenizer *t)  { return t->bos_id; }
int32_t  quasar_token_eos(const quasar_tokenizer *t)  { return t->eos_id; }
int32_t  quasar_token_id(const quasar_tokenizer *t, const char *text) {
    int32_t id;
    return hm_get(&t->tok2id, text, strlen(text), &id) ? id : -1;
}
