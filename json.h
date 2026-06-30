/* json.h - a tiny zero-dependency JSON parser + string builder.
 *
 * Just enough to read OpenAI-style request bodies (nested objects/arrays,
 * escaped strings incl. \uXXXX surrogate pairs) and to build response bodies.
 * The parser produces a small tagged-union DOM that the caller walks with the
 * json_get/json_str/json_num/json_bool accessors; json_free releases it. */
#ifndef QUASAR_JSON_H
#define QUASAR_JSON_H

#include <stddef.h>

typedef enum {
    JSON_NULL, JSON_BOOL, JSON_NUM, JSON_STR, JSON_ARR, JSON_OBJ
} json_type;

typedef struct json_value json_value;
struct json_value {
    json_type type;
    union {
        int    b;                                   /* JSON_BOOL */
        double num;                                 /* JSON_NUM  */
        struct { char *s; size_t len; } str;        /* JSON_STR (NUL-terminated) */
        struct { json_value **items; size_t n; } arr;
        struct { char **keys; json_value **vals; size_t n; } obj;
    };
};

/* Parse `len` bytes of JSON. Returns a malloc'd tree (free with json_free) or
 * NULL on error, writing a message into err. */
json_value *json_parse(const char *text, size_t len, char *err, size_t errlen);
void        json_free(json_value *v);

/* Accessors (all NULL/type-safe; return the default when absent or wrong type). */
const json_value *json_get(const json_value *obj, const char *key);  /* object field */
const char       *json_str(const json_value *v, const char *def);
double            json_num(const json_value *v, double def);
int               json_bool(const json_value *v, int def);

/* ---- growable string buffer (response building) ----------------------- */

typedef struct { char *buf; size_t len, cap; } strbuf;

void  sb_init(strbuf *sb);
void  sb_free(strbuf *sb);
void  sb_putc(strbuf *sb, char c);
void  sb_append(strbuf *sb, const char *s);
void  sb_append_len(strbuf *sb, const char *s, size_t n);
void  sb_printf(strbuf *sb, const char *fmt, ...);
/* Append a JSON string literal (with surrounding quotes), escaping as needed. */
void  sb_json_str(strbuf *sb, const char *s, size_t n);

#endif /* QUASAR_JSON_H */
