/* quant.h - GGUF dequantization to f32 for the reference path.
 *
 * Block layouts and constants follow ggml/llama.cpp exactly (k-quants), so the
 * reference forward matches llama.cpp on the same GGUF. */
#ifndef QUASAR_QUANT_H
#define QUASAR_QUANT_H

#include <stdint.h>

float q_fp16_to_fp32(uint16_t h);

/* Dequantize n contiguous elements (n must be a multiple of the type's block
 * size) from src into dst. Supports F32, F16, Q8_0, Q4_0, Q4_K, Q6_K; unknown
 * types fill zeros. */
void quasar_dequant_row(uint32_t ggml_type, const void *src, float *dst, int n);

/* Quantize n f32 values (n % 256 == 0) into Q2_K blocks (84 bytes per 256). */
void quasar_quantize_row_q2_K(const float *src, void *dst, int n);

#endif /* QUASAR_QUANT_H */
