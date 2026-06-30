/* sample.h - token sampling (greedy / temperature / top-k / top-p / min-p).
 *
 * A small stateful sampler: it owns the RNG state and a ring buffer of recent
 * tokens (for the repetition penalty), so the server and a future CLI chat loop
 * can share one configured sampler across a generation and reset it per
 * sequence. temperature <= 0 means deterministic greedy argmax. */
#ifndef QUASAR_SAMPLE_H
#define QUASAR_SAMPLE_H

#include <stdint.h>

typedef struct quasar_sampler quasar_sampler;

quasar_sampler *quasar_sampler_new(void);
void            quasar_sampler_free(quasar_sampler *s);

/* Configure sampling. Disable values: top_p=1, top_k<=0, min_p<=0,
 * repeat_penalty<=1. temperature<=0 forces greedy. seed=0 -> time-seeded. */
void quasar_sampler_config(quasar_sampler *s,
                           float temperature, float top_p, int top_k,
                           float min_p, float repeat_penalty, int repeat_last_n,
                           uint64_t seed);

/* Forget recent-token history (start of a new sequence). */
void quasar_sampler_reset(quasar_sampler *s);

/* Pick the next token from logits[n_vocab]. May reorder a scratch copy
 * internally; logits is not modified. Records the choice in history. */
int quasar_sampler_pick(quasar_sampler *s, const float *logits, int n_vocab);

#endif /* QUASAR_SAMPLE_H */
