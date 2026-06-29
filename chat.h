/* chat.h - Qwen3 ChatML prompt rendering.
 *
 * Qwen3 uses ChatML: each turn is <|im_start|>{role}\n{content}<|im_end|>\n.
 * A generation prompt opens an assistant turn; when thinking is disabled we
 * inject an empty <think></think> block (matches Qwen3's chat template). */
#ifndef QUASAR_CHAT_H
#define QUASAR_CHAT_H

#include <stddef.h>
#include <stdbool.h>

typedef struct { const char *role; const char *content; } quasar_msg;

/* Render messages to a malloc'd ChatML string (caller frees). */
char *quasar_chat_render(const quasar_msg *msgs, int n,
                         bool add_generation_prompt, bool enable_thinking,
                         size_t *out_len);

#endif /* QUASAR_CHAT_H */
