/* gguf.h - self-contained GGUF v3 reader for Quasar.
 *
 * Not a general ggml dependency: just enough to mmap a GGUF file, expose its
 * metadata key/values and tensor descriptors, and compute tensor byte sizes
 * from ggml type traits. No ownership of tensor data is taken; all pointers
 * returned point into the mmap and stay valid until gguf_close(). */
#ifndef QUASAR_GGUF_H
#define QUASAR_GGUF_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ggml tensor element types (subset relevant to Qwen3-MoE GGUFs). */
enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT   = 64
};

/* GGUF metadata value types. */
enum {
    GGUF_T_UINT8   = 0,  GGUF_T_INT8    = 1,
    GGUF_T_UINT16  = 2,  GGUF_T_INT16   = 3,
    GGUF_T_UINT32  = 4,  GGUF_T_INT32   = 5,
    GGUF_T_FLOAT32 = 6,  GGUF_T_BOOL    = 7,
    GGUF_T_STRING  = 8,  GGUF_T_ARRAY   = 9,
    GGUF_T_UINT64  = 10, GGUF_T_INT64   = 11,
    GGUF_T_FLOAT64 = 12
};

typedef struct {
    char    *key;          /* null-terminated, owned */
    uint32_t type;         /* GGUF_T_* */
    /* scalar interpretations (valid for numeric/bool types) */
    uint64_t u;            /* unsigned / raw / bool */
    int64_t  i;            /* signed */
    double   d;            /* floating */
    /* STRING */
    const char *sptr;      /* into mmap, not null-terminated */
    uint64_t    slen;
    /* ARRAY */
    uint32_t       arr_type;
    uint64_t       arr_n;
    const uint8_t *arr_data;   /* into mmap, first element */
    uint64_t       arr_bytes;
} gguf_kv;

typedef struct {
    char    *name;         /* null-terminated, owned */
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;         /* ggml type */
    uint64_t offset;       /* within the data section */
    uint64_t n_elements;
    uint64_t nbytes;       /* 0 if type traits unknown */
    const uint8_t *data;   /* into mmap; NULL if size unknown / out of range */
} gguf_tensor;

typedef struct {
    int      fd;
    void    *map;
    size_t   map_size;
    uint32_t version;
    uint32_t alignment;
    uint64_t n_kv;
    uint64_t n_tensors;
    gguf_kv     *kv;
    gguf_tensor *tensors;
    uint64_t data_off;     /* offset of tensor data section within the file */
    uint64_t kv_end;       /* byte offset where the metadata-KV section ends   */
    char     err[256];     /* empty on success */
} gguf_ctx;

/* Returns a context even on parse failure (check ->err[0]); NULL only on OOM. */
gguf_ctx *gguf_open(const char *path);
void      gguf_close(gguf_ctx *g);

const gguf_kv     *gguf_find_kv(const gguf_ctx *g, const char *key);
const gguf_tensor *gguf_find_tensor(const gguf_ctx *g, const char *name);

/* Typed metadata accessors. Return false if missing or wrong category. */
bool gguf_u32 (const gguf_ctx *g, const char *key, uint32_t *out);
bool gguf_u64 (const gguf_ctx *g, const char *key, uint64_t *out);
bool gguf_i32 (const gguf_ctx *g, const char *key, int32_t  *out);
bool gguf_f32 (const gguf_ctx *g, const char *key, float    *out);
bool gguf_bool(const gguf_ctx *g, const char *key, bool     *out);
bool gguf_str (const gguf_ctx *g, const char *key, const char **ptr, uint64_t *len);

uint32_t gguf_u32_or(const gguf_ctx *g, const char *key, uint32_t def);
float    gguf_f32_or(const gguf_ctx *g, const char *key, float    def);

/* ggml type traits. */
const char *ggml_type_name(uint32_t type);
uint32_t    ggml_blck_size(uint32_t type);  /* elements per block (0 = unknown) */
size_t      ggml_type_size(uint32_t type);  /* bytes per block (0 = unknown) */

/* Array helpers (tokenizer vocab / merges / token types). */
uint64_t gguf_arr_n  (const gguf_kv *kv);
bool     gguf_arr_i32(const gguf_kv *kv, uint64_t i, int32_t *out);

/* Single-pass iterator over a STRING array (random access would be O(n) per
 * element). fn returns non-zero to stop early; returns elements visited. */
typedef int (*gguf_str_iter_fn)(void *ud, uint64_t idx, const char *s, uint64_t len);
uint64_t gguf_arr_foreach_str(const gguf_kv *kv, gguf_str_iter_fn fn, void *ud);

#endif /* QUASAR_GGUF_H */
