/* sample.c - see sample.h.
 *
 * The full vocab is ~152k tokens, so we avoid sorting all of it every step: a
 * post-temperature logit window (anything >30 below the max contributes <1e-13
 * to the softmax and can never realistically be sampled) prunes the candidate
 * set to a few dozen entries before the sort. Greedy (temperature<=0) skips all
 * of it and just takes the argmax. */
#include "sample.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct { int id; float logit; float p; } cand_t;

struct quasar_sampler {
    float    temperature, top_p, min_p, repeat_penalty;
    int      top_k, repeat_last_n;
    uint64_t rng;

    int32_t *hist;                 /* ring buffer of recent token ids */
    int      hist_cap, hist_len, hist_head;

    float   *work;                 /* [n_vocab] penalized logits */
    uint8_t *penalize;             /* [n_vocab] history membership flags */
    cand_t  *cand;                 /* [n_vocab] candidate scratch */
    int      vocab_cap;            /* allocation size of the three arrays above */
};

quasar_sampler *quasar_sampler_new(void) {
    quasar_sampler *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->temperature = 0.0f;         /* greedy by default */
    s->top_p = 1.0f; s->top_k = 0; s->min_p = 0.0f;
    s->repeat_penalty = 1.0f; s->repeat_last_n = 0;
    s->rng = 0x9E3779B97F4A7C15ull;
    return s;
}

void quasar_sampler_free(quasar_sampler *s) {
    if (!s) return;
    free(s->hist); free(s->work); free(s->penalize); free(s->cand);
    free(s);
}

void quasar_sampler_config(quasar_sampler *s,
                           float temperature, float top_p, int top_k,
                           float min_p, float repeat_penalty, int repeat_last_n,
                           uint64_t seed) {
    s->temperature   = temperature;
    s->top_p         = (top_p > 0.0f && top_p < 1.0f) ? top_p : 1.0f;
    s->top_k         = top_k > 0 ? top_k : 0;
    s->min_p         = min_p > 0.0f ? min_p : 0.0f;
    s->repeat_penalty= repeat_penalty > 1.0f ? repeat_penalty : 1.0f;
    s->repeat_last_n = repeat_last_n > 0 ? repeat_last_n : 0;

    if (seed == 0) seed = (uint64_t)time(NULL) * 2654435761u + 0x1234567u;
    s->rng = seed ? seed : 0x9E3779B97F4A7C15ull;

    int cap = s->repeat_last_n;
    if (cap > 0 && cap != s->hist_cap) {
        free(s->hist);
        s->hist = calloc((size_t)cap, sizeof(int32_t));
        s->hist_cap = s->hist ? cap : 0;
    }
    s->hist_len = s->hist_head = 0;
}

void quasar_sampler_reset(quasar_sampler *s) { s->hist_len = s->hist_head = 0; }

static uint64_t xorshift64(uint64_t *st) {
    uint64_t x = *st;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *st = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static double rng_unit(quasar_sampler *s) {       /* uniform [0,1) */
    return (double)(xorshift64(&s->rng) >> 11) * (1.0 / 9007199254740992.0);
}

static void record(quasar_sampler *s, int id) {
    if (s->hist_cap <= 0) return;
    if (s->hist_len < s->hist_cap) {
        s->hist[(s->hist_head + s->hist_len) % s->hist_cap] = id;
        s->hist_len++;
    } else {
        s->hist[s->hist_head] = id;
        s->hist_head = (s->hist_head + 1) % s->hist_cap;
    }
}

static int ensure_scratch(quasar_sampler *s, int n) {
    if (s->vocab_cap >= n) return 0;
    float   *w = realloc(s->work,     (size_t)n * sizeof *w);
    uint8_t *p = realloc(s->penalize, (size_t)n * sizeof *p);
    cand_t  *c = realloc(s->cand,     (size_t)n * sizeof *c);
    if (!w || !p || !c) { s->work = w ? w : s->work; s->penalize = p ? p : s->penalize; s->cand = c ? c : s->cand; return -1; }
    s->work = w; s->penalize = p; s->cand = c;
    memset(s->penalize, 0, (size_t)n);
    s->vocab_cap = n;
    return 0;
}

static int cmp_desc(const void *a, const void *b) {
    float pa = ((const cand_t *)a)->p, pb = ((const cand_t *)b)->p;
    return (pa < pb) - (pa > pb);
}

int quasar_sampler_pick(quasar_sampler *s, const float *logits, int n_vocab) {
    if (n_vocab <= 0) return 0;
    if (ensure_scratch(s, n_vocab) != 0) {     /* OOM: fall back to plain argmax */
        int bi = 0; for (int i = 1; i < n_vocab; i++) if (logits[i] > logits[bi]) bi = i;
        record(s, bi); return bi;
    }

    /* Apply the repetition penalty into work[] and track the running max. */
    const float rp = s->repeat_penalty;
    if (rp > 1.0f) for (int i = 0; i < s->hist_len; i++) s->penalize[s->hist[(s->hist_head + i) % s->hist_cap]] = 1;
    float maxl = -INFINITY;
    for (int i = 0; i < n_vocab; i++) {
        float l = logits[i];
        if (s->penalize[i]) l = (l > 0.0f) ? l / rp : l * rp;
        s->work[i] = l;
        if (l > maxl) maxl = l;
    }
    if (rp > 1.0f) for (int i = 0; i < s->hist_len; i++) s->penalize[s->hist[(s->hist_head + i) % s->hist_cap]] = 0;

    /* Greedy: argmax of the penalized logits. */
    if (s->temperature <= 0.0f) {
        int bi = 0; for (int i = 1; i < n_vocab; i++) if (s->work[i] > s->work[bi]) bi = i;
        record(s, bi); return bi;
    }

    /* Build the candidate set: scaled logits within a 30-nat window of the max. */
    const float inv_t = 1.0f / s->temperature;
    const float msc   = maxl * inv_t;            /* max scaled logit */
    const float thr   = msc - 30.0f;
    int nc = 0;
    for (int i = 0; i < n_vocab; i++) {
        float sc = s->work[i] * inv_t;
        if (sc >= thr) { s->cand[nc].id = i; s->cand[nc].logit = sc; nc++; }
    }

    /* Softmax over the candidates, then sort by probability descending. */
    double sum = 0.0;
    for (int i = 0; i < nc; i++) { float e = expf(s->cand[i].logit - msc); s->cand[i].p = e; sum += e; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < nc; i++) s->cand[i].p *= inv;
    qsort(s->cand, (size_t)nc, sizeof(cand_t), cmp_desc);

    /* top-k */
    if (s->top_k > 0 && s->top_k < nc) nc = s->top_k;
    /* top-p (nucleus): smallest prefix whose cumulative mass reaches top_p */
    if (s->top_p < 1.0f) {
        double cum = 0.0; int keep = nc;
        for (int i = 0; i < nc; i++) { cum += s->cand[i].p; if (cum >= s->top_p) { keep = i + 1; break; } }
        nc = keep;
    }
    /* min-p: drop anything below min_p * p_max (candidates are sorted desc) */
    if (s->min_p > 0.0f) {
        float floor_p = s->min_p * s->cand[0].p;
        int keep = nc;
        for (int i = 1; i < nc; i++) if (s->cand[i].p < floor_p) { keep = i; break; }
        nc = keep;
    }
    if (nc < 1) nc = 1;

    /* Sample from the surviving candidates (renormalized). */
    double rsum = 0.0;
    for (int i = 0; i < nc; i++) rsum += s->cand[i].p;
    double r = rng_unit(s) * rsum, acc = 0.0;
    int chosen = s->cand[nc - 1].id;
    for (int i = 0; i < nc; i++) { acc += s->cand[i].p; if (r <= acc) { chosen = s->cand[i].id; break; } }

    record(s, chosen);
    return chosen;
}
