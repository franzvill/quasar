/* quant.c - see quant.h. K-quant layouts mirror ggml-quants.c. */
#include "quant.h"
#include "gguf.h"

#include <string.h>
#include <math.h>

float q_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {                                  /* subnormal */
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);  /* inf / nan */
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

static inline uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

static uint16_t fp32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {                                  /* subnormal / zero */
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        int shift = 14 - exp;
        uint32_t m = mant >> shift;
        if ((mant >> (shift - 1)) & 1) m++;          /* round to nearest */
        return (uint16_t)(sign | m);
    } else if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00);            /* inf */
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if ((mant >> 12) & 1) h++;                       /* round to nearest */
    return h;
}

/* Q4_K super-block scale/min unpacking (6-bit packed), per ggml. */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dequant_q8_0(const uint8_t *x, float *y, int n) {
    int nb = n / 32;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = x + (size_t)i * 34;
        float d = q_fp16_to_fp32(rd16(b));
        const int8_t *q = (const int8_t *)(b + 2);
        for (int j = 0; j < 32; j++) y[i * 32 + j] = d * q[j];
    }
}

static void dequant_q4_0(const uint8_t *x, float *y, int n) {
    int nb = n / 32;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = x + (size_t)i * 18;
        float d = q_fp16_to_fp32(rd16(b));
        const uint8_t *q = b + 2;
        for (int j = 0; j < 16; j++) {
            y[i * 32 + j]      = d * ((q[j] & 0xF) - 8);
            y[i * 32 + j + 16] = d * ((q[j] >>  4) - 8);
        }
    }
}

static void dequant_q4_K(const uint8_t *x, float *y, int n) {
    int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b      = x + (size_t)i * 144;
        float          d      = q_fp16_to_fp32(rd16(b));
        float          dmin   = q_fp16_to_fp32(rd16(b + 2));
        const uint8_t *scales = b + 4;
        const uint8_t *q      = b + 16;
        float *yy = y + (size_t)i * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m); float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; l++) *yy++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *yy++ = d2 * (q[l] >>  4) - m2;
            q += 32; is += 2;
        }
    }
}

static void dequant_q6_K(const uint8_t *x, float *y, int n) {
    int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b  = x + (size_t)i * 210;
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128;
        const int8_t  *sc = (const int8_t *)(b + 192);
        float          d  = q_fp16_to_fp32(rd16(b + 208));
        float *yy = y + (size_t)i * 256;
        for (int nn = 0; nn < 256; nn += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                yy[l]      = d * sc[is + 0] * q1;
                yy[l + 32] = d * sc[is + 2] * q2;
                yy[l + 64] = d * sc[is + 4] * q3;
                yy[l + 96] = d * sc[is + 6] * q4;
            }
            yy += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

/* Q2_K layout: scales[16] (4-bit scale | 4-bit min each), qs[64] (2-bit), d, dmin. */
static void dequant_q2_K(const uint8_t *x, float *y, int n) {
    int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b      = x + (size_t)i * 84;
        const uint8_t *scales = b;
        const uint8_t *q      = b + 16;
        float d    = q_fp16_to_fp32(rd16(b + 80));
        float dmin = q_fp16_to_fp32(rd16(b + 82));
        float *yy = y + (size_t)i * 256;
        int is = 0;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) *yy++ = dl * ((q[l]      >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) *yy++ = dl * ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

/* Simple Q2_K encoder: per 16-weight sub-block min/scale, super-scaled to 4 bits. */
void quasar_quantize_row_q2_K(const float *x, void *dst, int n) {
    int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const float *v  = x + (size_t)i * 256;
        uint8_t     *b  = (uint8_t *)dst + (size_t)i * 84;
        uint8_t     *scales = b;
        uint8_t     *qs = b + 16;
        memset(qs, 0, 64);

        float dls[16], mls[16], maxdl = 0.0f, maxml = 0.0f;
        for (int sb = 0; sb < 16; sb++) {
            float vmin = v[sb * 16], vmax = v[sb * 16];
            for (int l = 1; l < 16; l++) { float w = v[sb * 16 + l]; if (w < vmin) vmin = w; if (w > vmax) vmax = w; }
            float dl, ml;
            if (vmin < 0.0f) { ml = -vmin;     dl = (vmax - vmin) / 3.0f; }
            else             { ml = 0.0f;      dl = vmax / 3.0f; }
            if (dl < 0.0f) dl = 0.0f;
            dls[sb] = dl; mls[sb] = ml;
            if (dl > maxdl) maxdl = dl;
            if (ml > maxml) maxml = ml;
        }
        float d    = maxdl / 15.0f;
        float dmin = maxml / 15.0f;
        float id   = d    > 0.0f ? 1.0f / d    : 0.0f;
        float idm  = dmin > 0.0f ? 1.0f / dmin : 0.0f;

        for (int sb = 0; sb < 16; sb++) {
            int s4 = (int)floorf(dls[sb] * id  + 0.5f); if (s4 < 0) s4 = 0; if (s4 > 15) s4 = 15;
            int m4 = (int)floorf(mls[sb] * idm + 0.5f); if (m4 < 0) m4 = 0; if (m4 > 15) m4 = 15;
            scales[sb] = (uint8_t)(s4 | (m4 << 4));
            float dl_a = d * s4, ml_a = dmin * m4;
            float idl  = dl_a > 0.0f ? 1.0f / dl_a : 0.0f;
            int outer = sb / 8, g = sb % 8, shift = 2 * (g / 2), base = outer * 32 + (g % 2) * 16;
            for (int l = 0; l < 16; l++) {
                int q = (int)floorf((v[sb * 16 + l] + ml_a) * idl + 0.5f);
                if (q < 0) q = 0; if (q > 3) q = 3;
                qs[base + l] |= (uint8_t)(q << shift);
            }
        }
        uint16_t dh = fp32_to_fp16(d), dmh = fp32_to_fp16(dmin);
        memcpy(b + 80, &dh, 2);
        memcpy(b + 82, &dmh, 2);
    }
}

void quasar_dequant_row(uint32_t type, const void *src, float *dst, int n) {
    const uint8_t *p = src;
    switch (type) {
        case GGML_TYPE_F32:  memcpy(dst, src, (size_t)n * 4); break;
        case GGML_TYPE_F16: { const uint8_t *s = p; for (int i = 0; i < n; i++) dst[i] = q_fp16_to_fp32(rd16(s + (size_t)i * 2)); } break;
        case GGML_TYPE_Q8_0: dequant_q8_0(p, dst, n); break;
        case GGML_TYPE_Q4_0: dequant_q4_0(p, dst, n); break;
        case GGML_TYPE_Q2_K: dequant_q2_K(p, dst, n); break;
        case GGML_TYPE_Q4_K: dequant_q4_K(p, dst, n); break;
        case GGML_TYPE_Q6_K: dequant_q6_K(p, dst, n); break;
        default: for (int i = 0; i < n; i++) dst[i] = 0.0f;
    }
}
