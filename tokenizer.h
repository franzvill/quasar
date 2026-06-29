/* tokenizer.h - Qwen3 byte-level BPE tokenizer (GPT-2 family) built from GGUF.
 *
 * Loads tokenizer.ggml.{tokens,merges,token_type,*_token_id} and implements the
 * Qwen2/Qwen3 pre-tokenizer split + byte-level BPE. See ARCHITECTURE.md. */
#ifndef QUASAR_TOKENIZER_H
#define QUASAR_TOKENIZER_H

#include "gguf.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct quasar_tokenizer quasar_tokenizer;

quasar_tokenizer *quasar_tokenizer_load(const gguf_ctx *g, char *err, size_t errlen);
void              quasar_tokenizer_free(quasar_tokenizer *t);

/* Encode UTF-8 text to token ids. Returns a malloc'd array (caller frees) and
 * writes the count to *out_n. Returns NULL on allocation failure. */
int32_t *quasar_tokenize(quasar_tokenizer *t, const char *text, size_t len,
                         bool add_bos, size_t *out_n);

/* Decode ids to a malloc'd UTF-8 string (NUL-terminated; caller frees). */
char *quasar_detokenize(quasar_tokenizer *t, const int32_t *ids, size_t n, size_t *out_len);

/* Bytes for a single token (streaming). Returns byte count written (<= cap). */
int quasar_token_to_bytes(quasar_tokenizer *t, int32_t id, char *buf, int cap);

uint32_t quasar_vocab_size(const quasar_tokenizer *t);
int32_t  quasar_token_bos(const quasar_tokenizer *t);
int32_t  quasar_token_eos(const quasar_tokenizer *t);
int32_t  quasar_token_id(const quasar_tokenizer *t, const char *text); /* exact, or -1 */

#endif /* QUASAR_TOKENIZER_H */
