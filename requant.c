/* requant.c - see requant.h. */
#include "requant.h"
#include "gguf.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_expert(const char *name) {
    return strstr(name, "_exps.weight") != NULL;   /* ffn_{gate,up,down}_exps.weight */
}
static uint64_t align32(uint64_t x) { return (x + 31) & ~(uint64_t)31; }

static int wzero(FILE *f, uint64_t n) {
    static const char z[4096] = {0};
    while (n) { size_t c = n > sizeof z ? sizeof z : (size_t)n; if (fwrite(z, 1, c, f) != c) return -1; n -= c; }
    return 0;
}

int quasar_requant_experts_q2k(const char *in_path, const char *out_path, char *err, size_t errlen) {
    gguf_ctx *g = gguf_open(in_path);
    if (!g) { snprintf(err, errlen, "out of memory"); return -1; }
    if (g->err[0]) { snprintf(err, errlen, "%s", g->err); gguf_close(g); return -1; }

    FILE *out = fopen(out_path, "wb");
    if (!out) { snprintf(err, errlen, "cannot open %s for writing", out_path); gguf_close(g); return -1; }

    uint64_t nt = g->n_tensors;
    uint32_t *ntype  = malloc(nt * sizeof(uint32_t));
    uint64_t *noff   = malloc(nt * sizeof(uint64_t));
    uint64_t *nbytes = malloc(nt * sizeof(uint64_t));
    uint64_t off = 0, n_req = 0;
    for (uint64_t t = 0; t < nt; t++) {
        gguf_tensor *ti = &g->tensors[t];
        if (is_expert(ti->name)) { ntype[t] = GGML_TYPE_Q2_K; nbytes[t] = ti->n_elements / 256 * 84; n_req++; }
        else                     { ntype[t] = ti->type;       nbytes[t] = ti->nbytes; }
        off = align32(off);
        noff[t] = off;
        off += nbytes[t];
    }

    /* header */
    uint32_t magic = 0x46554747u, ver = 3;
    fwrite(&magic, 4, 1, out); fwrite(&ver, 4, 1, out);
    fwrite(&g->n_tensors, 8, 1, out); fwrite(&g->n_kv, 8, 1, out);
    /* metadata KV blob: copied verbatim */
    fwrite((const uint8_t *)g->map + 24, 1, (size_t)(g->kv_end - 24), out);
    /* tensor infos (new types + offsets) */
    for (uint64_t t = 0; t < nt; t++) {
        gguf_tensor *ti = &g->tensors[t];
        uint64_t nlen = strlen(ti->name);
        fwrite(&nlen, 8, 1, out); fwrite(ti->name, 1, (size_t)nlen, out);
        fwrite(&ti->n_dims, 4, 1, out);
        for (uint32_t d = 0; d < ti->n_dims; d++) fwrite(&ti->dims[d], 8, 1, out);
        fwrite(&ntype[t], 4, 1, out);
        fwrite(&noff[t], 8, 1, out);
    }

    long hdr_end = ftell(out);
    uint64_t data_start = align32((uint64_t)hdr_end);
    wzero(out, data_start - (uint64_t)hdr_end);

    /* tensor data */
    float   *tmp = malloc(256 * sizeof(float));
    uint8_t  blk[84];
    uint64_t in_data = 0;
    for (uint64_t t = 0; t < nt; t++) {
        gguf_tensor *ti = &g->tensors[t];
        if (noff[t] > in_data) { wzero(out, noff[t] - in_data); in_data = noff[t]; }
        if (ntype[t] == GGML_TYPE_Q2_K && is_expert(ti->name)) {
            uint64_t nblk = ti->n_elements / 256;
            size_t   ssz  = ggml_type_size(ti->type);
            for (uint64_t bi = 0; bi < nblk; bi++) {
                quasar_dequant_row(ti->type, ti->data + bi * ssz, tmp, 256);
                quasar_quantize_row_q2_K(tmp, blk, 256);
                fwrite(blk, 1, 84, out);
            }
        } else {
            fwrite(ti->data, 1, (size_t)ti->nbytes, out);
        }
        in_data += nbytes[t];
    }
    free(tmp);

    uint64_t total = data_start + off;
    printf("requant: %llu tensors (%llu experts -> Q2_K), output %.2f GB\n",
           (unsigned long long)nt, (unsigned long long)n_req, total / 1e9);

    free(ntype); free(noff); free(nbytes);
    fclose(out);
    gguf_close(g);
    return 0;
}
