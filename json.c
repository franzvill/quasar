/* json.c - see json.h. Recursive-descent parser + growable string builder. */
#include "json.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>

/* ---- parser ----------------------------------------------------------- */

typedef struct {
    const char *p, *end;
    char       *err;
    size_t      errlen;
    int         depth;
} jp;

#define JSON_MAX_DEPTH 128

static json_value *parse_value(jp *j);

static void jerr(jp *j, const char *msg) {
    if (j->err && j->errlen && !j->err[0])
        snprintf(j->err, j->errlen, "%s", msg);
}

static void skip_ws(jp *j) {
    while (j->p < j->end) {
        char c = *j->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') j->p++;
        else break;
    }
}

static json_value *new_val(json_type t) {
    json_value *v = calloc(1, sizeof *v);
    if (v) v->type = t;
    return v;
}

/* Append one Unicode code point as UTF-8 to buf[*len] (cap-checked by caller). */
static void utf8_encode(uint32_t cp, char *buf, size_t *len) {
    if (cp <= 0x7F) {
        buf[(*len)++] = (char)cp;
    } else if (cp <= 0x7FF) {
        buf[(*len)++] = (char)(0xC0 | (cp >> 6));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        buf[(*len)++] = (char)(0xE0 | (cp >> 12));
        buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else {
        buf[(*len)++] = (char)(0xF0 | (cp >> 18));
        buf[(*len)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[(*len)++] = (char)(0x80 | (cp & 0x3F));
    }
}

static int hex4(const char *p, uint32_t *out) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i]; v <<= 4;
        if      (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

/* Parse a JSON string (cursor just past the opening quote). Returns a malloc'd
 * NUL-terminated, unescaped UTF-8 string; writes byte length to *out_len. */
static char *parse_string_raw(jp *j, size_t *out_len) {
    const char *s = j->p;
    /* Worst case the decoded form is no longer than the source. */
    size_t cap = (size_t)(j->end - s) + 1;
    char  *out = malloc(cap);
    if (!out) { jerr(j, "out of memory"); return NULL; }
    size_t len = 0;

    while (j->p < j->end) {
        char c = *j->p++;
        if (c == '"') { out[len] = 0; if (out_len) *out_len = len; return out; }
        if (c == '\\') {
            if (j->p >= j->end) break;
            char e = *j->p++;
            switch (e) {
                case '"':  out[len++] = '"';  break;
                case '\\': out[len++] = '\\'; break;
                case '/':  out[len++] = '/';  break;
                case 'b':  out[len++] = '\b'; break;
                case 'f':  out[len++] = '\f'; break;
                case 'n':  out[len++] = '\n'; break;
                case 'r':  out[len++] = '\r'; break;
                case 't':  out[len++] = '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (j->end - j->p < 4 || hex4(j->p, &cp) != 0) { jerr(j, "bad \\u escape"); free(out); return NULL; }
                    j->p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {              /* high surrogate */
                        if (j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
                            uint32_t lo;
                            if (hex4(j->p + 2, &lo) == 0 && lo >= 0xDC00 && lo <= 0xDFFF) {
                                j->p += 6;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                        }
                    }
                    utf8_encode(cp, out, &len);
                    break;
                }
                default: jerr(j, "bad escape"); free(out); return NULL;
            }
        } else {
            out[len++] = c;
        }
    }
    jerr(j, "unterminated string");
    free(out);
    return NULL;
}

static json_value *parse_string(jp *j) {
    json_value *v = new_val(JSON_STR);
    if (!v) { jerr(j, "out of memory"); return NULL; }
    v->str.s = parse_string_raw(j, &v->str.len);
    if (!v->str.s) { free(v); return NULL; }
    return v;
}

static json_value *parse_number(jp *j) {
    char *endp = NULL;
    double d = strtod(j->p, &endp);
    if (endp == j->p) { jerr(j, "bad number"); return NULL; }
    j->p = endp;
    json_value *v = new_val(JSON_NUM);
    if (!v) { jerr(j, "out of memory"); return NULL; }
    v->num = d;
    return v;
}

static int match_lit(jp *j, const char *lit) {
    size_t n = strlen(lit);
    if ((size_t)(j->end - j->p) >= n && memcmp(j->p, lit, n) == 0) { j->p += n; return 0; }
    return -1;
}

static json_value *parse_array(jp *j) {
    j->p++;                       /* consume '[' */
    json_value *v = new_val(JSON_ARR);
    if (!v) { jerr(j, "out of memory"); return NULL; }
    skip_ws(j);
    if (j->p < j->end && *j->p == ']') { j->p++; return v; }
    size_t cap = 0;
    for (;;) {
        json_value *item = parse_value(j);
        if (!item) { json_free(v); return NULL; }
        if (v->arr.n == cap) {
            cap = cap ? cap * 2 : 4;
            json_value **ni = realloc(v->arr.items, cap * sizeof *ni);
            if (!ni) { json_free(item); json_free(v); jerr(j, "out of memory"); return NULL; }
            v->arr.items = ni;
        }
        v->arr.items[v->arr.n++] = item;
        skip_ws(j);
        if (j->p >= j->end) { jerr(j, "unterminated array"); json_free(v); return NULL; }
        if (*j->p == ',') { j->p++; skip_ws(j); continue; }
        if (*j->p == ']') { j->p++; return v; }
        jerr(j, "expected ',' or ']'"); json_free(v); return NULL;
    }
}

static json_value *parse_object(jp *j) {
    j->p++;                       /* consume '{' */
    json_value *v = new_val(JSON_OBJ);
    if (!v) { jerr(j, "out of memory"); return NULL; }
    skip_ws(j);
    if (j->p < j->end && *j->p == '}') { j->p++; return v; }
    size_t cap = 0;
    for (;;) {
        skip_ws(j);
        if (j->p >= j->end || *j->p != '"') { jerr(j, "expected object key"); json_free(v); return NULL; }
        j->p++;
        size_t klen = 0;
        char *key = parse_string_raw(j, &klen);
        if (!key) { json_free(v); return NULL; }
        skip_ws(j);
        if (j->p >= j->end || *j->p != ':') { jerr(j, "expected ':'"); free(key); json_free(v); return NULL; }
        j->p++;
        json_value *val = parse_value(j);
        if (!val) { free(key); json_free(v); return NULL; }
        if (v->obj.n == cap) {
            cap = cap ? cap * 2 : 4;
            char       **nk = realloc(v->obj.keys, cap * sizeof *nk);
            json_value **nv = realloc(v->obj.vals, cap * sizeof *nv);
            if (!nk || !nv) { free(nk); free(nv); free(key); json_free(val); json_free(v); jerr(j, "out of memory"); return NULL; }
            v->obj.keys = nk; v->obj.vals = nv;
        }
        v->obj.keys[v->obj.n] = key;
        v->obj.vals[v->obj.n] = val;
        v->obj.n++;
        skip_ws(j);
        if (j->p >= j->end) { jerr(j, "unterminated object"); json_free(v); return NULL; }
        if (*j->p == ',') { j->p++; continue; }
        if (*j->p == '}') { j->p++; return v; }
        jerr(j, "expected ',' or '}'"); json_free(v); return NULL;
    }
}

static json_value *parse_value(jp *j) {
    if (++j->depth > JSON_MAX_DEPTH) { jerr(j, "nesting too deep"); j->depth--; return NULL; }
    skip_ws(j);
    json_value *v = NULL;
    if (j->p >= j->end) { jerr(j, "unexpected end of input"); j->depth--; return NULL; }
    char c = *j->p;
    switch (c) {
        case '{': v = parse_object(j); break;
        case '[': v = parse_array(j);  break;
        case '"': j->p++; v = parse_string(j); break;
        case 't': if (match_lit(j, "true")  == 0) { v = new_val(JSON_BOOL); if (v) v->b = 1; } else jerr(j, "bad literal"); break;
        case 'f': if (match_lit(j, "false") == 0) { v = new_val(JSON_BOOL); if (v) v->b = 0; } else jerr(j, "bad literal"); break;
        case 'n': if (match_lit(j, "null")  == 0) { v = new_val(JSON_NULL); } else jerr(j, "bad literal"); break;
        default:
            if (c == '-' || (c >= '0' && c <= '9')) v = parse_number(j);
            else jerr(j, "unexpected character");
            break;
    }
    j->depth--;
    return v;
}

json_value *json_parse(const char *text, size_t len, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    jp j = { text, text + len, err, errlen, 0 };
    json_value *v = parse_value(&j);
    if (!v) return NULL;
    skip_ws(&j);
    if (j.p != j.end) { jerr(&j, "trailing data after JSON value"); json_free(v); return NULL; }
    return v;
}

void json_free(json_value *v) {
    if (!v) return;
    switch (v->type) {
        case JSON_STR: free(v->str.s); break;
        case JSON_ARR:
            for (size_t i = 0; i < v->arr.n; i++) json_free(v->arr.items[i]);
            free(v->arr.items);
            break;
        case JSON_OBJ:
            for (size_t i = 0; i < v->obj.n; i++) { free(v->obj.keys[i]); json_free(v->obj.vals[i]); }
            free(v->obj.keys); free(v->obj.vals);
            break;
        default: break;
    }
    free(v);
}

/* ---- accessors -------------------------------------------------------- */

const json_value *json_get(const json_value *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJ) return NULL;
    for (size_t i = 0; i < obj->obj.n; i++)
        if (strcmp(obj->obj.keys[i], key) == 0) return obj->obj.vals[i];
    return NULL;
}

const char *json_str(const json_value *v, const char *def) {
    return (v && v->type == JSON_STR) ? v->str.s : def;
}

double json_num(const json_value *v, double def) {
    if (!v) return def;
    if (v->type == JSON_NUM)  return v->num;
    if (v->type == JSON_BOOL) return v->b ? 1.0 : 0.0;
    return def;
}

int json_bool(const json_value *v, int def) {
    if (!v) return def;
    if (v->type == JSON_BOOL) return v->b;
    if (v->type == JSON_NUM)  return v->num != 0.0;
    return def;
}

/* ---- string builder --------------------------------------------------- */

void sb_init(strbuf *sb) { sb->buf = NULL; sb->len = 0; sb->cap = 0; }
void sb_free(strbuf *sb) { free(sb->buf); sb->buf = NULL; sb->len = sb->cap = 0; }

static void sb_reserve(strbuf *sb, size_t extra) {
    if (sb->len + extra + 1 <= sb->cap) return;
    size_t cap = sb->cap ? sb->cap : 256;
    while (cap < sb->len + extra + 1) cap *= 2;
    char *nb = realloc(sb->buf, cap);
    if (!nb) return;            /* best-effort: silently drop on OOM */
    sb->buf = nb; sb->cap = cap;
}

void sb_putc(strbuf *sb, char c) {
    sb_reserve(sb, 1);
    if (sb->cap) { sb->buf[sb->len++] = c; sb->buf[sb->len] = 0; }
}

void sb_append_len(strbuf *sb, const char *s, size_t n) {
    sb_reserve(sb, n);
    if (sb->cap) { memcpy(sb->buf + sb->len, s, n); sb->len += n; sb->buf[sb->len] = 0; }
}

void sb_append(strbuf *sb, const char *s) { sb_append_len(sb, s, strlen(s)); }

void sb_printf(strbuf *sb, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[512];
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof tmp) { sb_append_len(sb, tmp, (size_t)n); return; }
    /* rare: format longer than the stack buffer */
    char *big = malloc((size_t)n + 1);
    if (!big) return;
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    sb_append_len(sb, big, (size_t)n);
    free(big);
}

void sb_json_str(strbuf *sb, const char *s, size_t n) {
    sb_putc(sb, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  sb_append_len(sb, "\\\"", 2); break;
            case '\\': sb_append_len(sb, "\\\\", 2); break;
            case '\b': sb_append_len(sb, "\\b", 2);  break;
            case '\f': sb_append_len(sb, "\\f", 2);  break;
            case '\n': sb_append_len(sb, "\\n", 2);  break;
            case '\r': sb_append_len(sb, "\\r", 2);  break;
            case '\t': sb_append_len(sb, "\\t", 2);  break;
            default:
                if (c < 0x20) sb_printf(sb, "\\u%04x", c);   /* other control chars */
                else          sb_putc(sb, (char)c);          /* UTF-8 passes through */
        }
    }
    sb_putc(sb, '"');
}
