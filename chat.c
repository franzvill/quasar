/* chat.c - see chat.h. */
#include "chat.h"

#include <stdlib.h>
#include <string.h>

typedef struct { char *s; size_t n, cap; } sb;

static void sb_add(sb *b, const char *str) {
    size_t l = strlen(str);
    if (b->n + l + 1 > b->cap) {
        while (b->n + l + 1 > b->cap) b->cap = b->cap ? b->cap * 2 : 256;
        b->s = realloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, str, l);
    b->n += l;
    b->s[b->n] = 0;
}

char *quasar_chat_render(const quasar_msg *msgs, int n,
                         bool add_generation_prompt, bool enable_thinking,
                         size_t *out_len) {
    sb b = {0};
    sb_add(&b, "");                       /* ensure a valid empty buffer */
    for (int i = 0; i < n; i++) {
        sb_add(&b, "<|im_start|>");
        sb_add(&b, msgs[i].role);
        sb_add(&b, "\n");
        sb_add(&b, msgs[i].content);
        sb_add(&b, "<|im_end|>\n");
    }
    if (add_generation_prompt) {
        sb_add(&b, "<|im_start|>assistant\n");
        if (!enable_thinking) sb_add(&b, "<think>\n\n</think>\n\n");
    }
    if (out_len) *out_len = b.n;
    return b.s;
}
