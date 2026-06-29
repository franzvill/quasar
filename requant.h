/* requant.h - produce the asymmetric 2-bit model (the ds4 trick).
 *
 * Reads a qwen3moe GGUF and writes a new one with every routed-expert tensor
 * (ffn_*_exps.weight) requantized to Q2_K, and all other tensors (attention,
 * router, embeddings, norms, output) copied byte-for-byte. */
#ifndef QUASAR_REQUANT_H
#define QUASAR_REQUANT_H

#include <stddef.h>

int quasar_requant_experts_q2k(const char *in_path, const char *out_path, char *err, size_t errlen);

#endif /* QUASAR_REQUANT_H */
