/* server.c - see server.h.
 *
 * A minimal HTTP/1.1 server (one connection at a time, Connection: close) that
 * speaks enough of the OpenAI API to drop into existing clients and coding
 * agents. The generation core (run_generation) is shared by the chat and
 * completion endpoints and streams tokens through a callback so the same code
 * serves both buffered JSON and Server-Sent-Events responses.
 *
 * Streaming responses are close-delimited (no Content-Length, Connection:
 * close): we write SSE events as tokens arrive and close the socket to mark the
 * end, which every SSE client handles. */
#include "server.h"
#include "quasar.h"
#include "tokenizer.h"
#include "chat.h"
#include "forward_cpu.h"
#include "metal.h"
#include "sample.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

/* ---- served state ----------------------------------------------------- */

typedef struct {
    quasar_model     *m;
    quasar_tokenizer *tk;
    quasar_ctx       *ctx;
    int               max_seq;
    int               n_vocab;
    int               eos;
    int               use_metal;
    char              model_id[128];
    float            *logits;       /* [n_vocab] reused */
    unsigned long     req_counter;
} server_st;

#define MAX_REQ_BYTES (64u << 20)   /* 64 MB request cap */

/* ---- socket helpers --------------------------------------------------- */

static int send_all(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t k = send(fd, p + off, len - off, 0);
        if (k <= 0) {
            if (k < 0 && errno == EINTR) continue;
            return -1;
        }
        off += (size_t)k;
    }
    return 0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- HTTP request reading --------------------------------------------- */

typedef struct {
    char   method[8];
    char  *raw;          /* the whole request buffer (owns path/body memory) */
    char  *path;         /* points into raw */
    char  *body;         /* points into raw */
    size_t body_len;
} http_req;

static void http_req_free(http_req *r) { free(r->raw); r->raw = NULL; }

/* Case-insensitive header lookup; returns value start (within [hdr,hdr_end)). */
static long header_content_length(const char *hdr, const char *hdr_end) {
    const char *p = hdr;
    while (p < hdr_end) {
        const char *eol = memchr(p, '\n', (size_t)(hdr_end - p));
        size_t linelen = eol ? (size_t)(eol - p) : (size_t)(hdr_end - p);
        if (linelen >= 15 && strncasecmp(p, "Content-Length:", 15) == 0) {
            const char *v = p + 15;
            while (v < p + linelen && (*v == ' ' || *v == '\t')) v++;
            return strtol(v, NULL, 10);
        }
        if (!eol) break;
        p = eol + 1;
    }
    return -1;
}

/* Read one full HTTP request (headers + Content-Length body). 0 ok, -1 fail. */
static int read_request(int fd, http_req *r) {
    memset(r, 0, sizeof *r);
    size_t cap = 8192, len = 0;
    char  *buf = malloc(cap);
    if (!buf) return -1;

    /* Read until the header terminator "\r\n\r\n". */
    char *hdr_end = NULL;
    for (;;) {
        if (len + 4096 > cap) {
            if (cap >= MAX_REQ_BYTES) { free(buf); return -1; }
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) { free(buf); return -1; }
            buf = nb;
        }
        ssize_t k = recv(fd, buf + len, cap - len - 1, 0);
        if (k < 0) { if (errno == EINTR) continue; free(buf); return -1; }
        if (k == 0) break;            /* peer closed */
        len += (size_t)k;
        buf[len] = 0;
        hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) break;
        if (len > 1u << 20) { free(buf); return -1; }   /* 1 MB header cap */
    }
    if (!hdr_end) { free(buf); return -1; }

    size_t header_bytes = (size_t)(hdr_end - buf) + 4;
    long clen = header_content_length(buf, hdr_end);
    size_t body_len = clen > 0 ? (size_t)clen : 0;
    if (body_len > MAX_REQ_BYTES) { free(buf); return -1; }

    /* Read the remaining body bytes. */
    size_t need = header_bytes + body_len;
    if (need + 1 > cap) {
        char *nb = realloc(buf, need + 1);
        if (!nb) { free(buf); return -1; }
        buf = nb; cap = need + 1;
    }
    while (len < need) {
        ssize_t k = recv(fd, buf + len, need - len, 0);
        if (k < 0) { if (errno == EINTR) continue; free(buf); return -1; }
        if (k == 0) break;
        len += (size_t)k;
    }
    buf[len] = 0;

    /* Parse the request line: METHOD SP PATH SP VERSION */
    char *sp1 = strchr(buf, ' ');
    if (!sp1) { free(buf); return -1; }
    size_t ml = (size_t)(sp1 - buf);
    if (ml == 0 || ml >= sizeof r->method) { free(buf); return -1; }
    memcpy(r->method, buf, ml); r->method[ml] = 0;
    char *path = sp1 + 1;
    char *sp2 = strchr(path, ' ');
    if (!sp2) { free(buf); return -1; }
    *sp2 = 0;
    char *q = strchr(path, '?');     /* drop any query string for routing */
    if (q) *q = 0;

    r->raw = buf;
    r->path = path;
    r->body = buf + header_bytes;
    r->body_len = (len >= header_bytes) ? len - header_bytes : 0;
    return 0;
}

/* ---- responses -------------------------------------------------------- */

static void send_json_response(int fd, int status, const char *status_text, const strbuf *body) {
    char hdr[256];
    int n = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n",
        status, status_text, body->len);
    if (n < 0) return;
    if (send_all(fd, hdr, (size_t)n) != 0) return;
    send_all(fd, body->buf ? body->buf : "", body->len);
}

static void send_error(int fd, int status, const char *status_text,
                       const char *type, const char *message) {
    strbuf sb; sb_init(&sb);
    sb_append(&sb, "{\"error\":{\"message\":");
    sb_json_str(&sb, message, strlen(message));
    sb_append(&sb, ",\"type\":");
    sb_json_str(&sb, type, strlen(type));
    sb_append(&sb, ",\"code\":null}}");
    send_json_response(fd, status, status_text, &sb);
    sb_free(&sb);
}

/* ---- generation core -------------------------------------------------- */

/* Streaming callback: emit `n` bytes of freshly decoded text. Return <0 to
 * abort generation (e.g. the client disconnected). */
typedef int (*delta_cb)(void *ud, const char *bytes, int n);

/* Largest index <= target (and >= low) that sits on a UTF-8 char boundary, so
 * we never stream a partial multi-byte character. */
static size_t utf8_floor(const char *s, size_t target, size_t low) {
    while (target > low && ((unsigned char)s[target] & 0xC0) == 0x80) target--;
    return target;
}

static const char *find_sub(const char *hay, size_t hlen, const char *needle, size_t nlen) {
    if (nlen == 0 || nlen > hlen) return NULL;
    for (size_t i = 0; i + nlen <= hlen; i++)
        if (memcmp(hay + i, needle, nlen) == 0) return hay + i;
    return NULL;
}

/* Prefill the prompt, then sample/stream up to max_tokens. Writes the final
 * (stop-truncated) completion text into out_text and the token count into
 * *out_n. Returns the finish reason ("stop" | "length"). */
static const char *run_generation(server_st *st, const int32_t *prompt, int n_prompt,
                                  int max_tokens, quasar_sampler *smp,
                                  char **stops, int n_stops,
                                  delta_cb cb, void *ud, strbuf *out_text, int *out_n) {
    quasar_kv_reset(st->ctx);
    quasar_decode(st->ctx, prompt, n_prompt, st->logits);

    size_t hmax = 0;                                   /* longest stop string */
    for (int i = 0; i < n_stops; i++) { size_t l = strlen(stops[i]); if (l > hmax) hmax = l; }

    strbuf text; sb_init(&text);
    size_t emitted = 0;
    int    pos = n_prompt;
    int    gen = 0, aborted = 0, stop_found = 0;
    size_t final_len = 0;
    const char *finish = "length";

    while (gen < max_tokens) {
        int id = quasar_sampler_pick(smp, st->logits, st->n_vocab);
        if (id == st->eos) { finish = "stop"; final_len = text.len; break; }

        char tb[256];
        int  w = quasar_token_to_bytes(st->tk, id, tb, (int)sizeof tb);
        size_t prev = text.len;
        sb_append_len(&text, tb, (size_t)w);
        gen++;

        /* Stop-string detection across the recent boundary. */
        if (n_stops) {
            size_t start = prev > hmax ? prev - hmax : 0;
            size_t best = (size_t)-1;
            for (int k = 0; k < n_stops; k++) {
                const char *h = find_sub(text.buf + start, text.len - start, stops[k], strlen(stops[k]));
                if (h) { size_t off = (size_t)(h - text.buf); if (off < best) best = off; }
            }
            if (best != (size_t)-1) { stop_found = 1; final_len = best; finish = "stop"; }
        }

        /* Emit the safe prefix: hold back hmax bytes (for stop matching) and
         * never split a UTF-8 character. On a stop hit, emit exactly up to it. */
        size_t target = stop_found ? final_len
                                   : (text.len > hmax ? text.len - hmax : 0);
        if (!stop_found) target = utf8_floor(text.buf, target, emitted);
        if (target > emitted && cb) {
            if (cb(ud, text.buf + emitted, (int)(target - emitted)) < 0) { aborted = 1; }
        }
        if (target > emitted) emitted = target;

        if (stop_found || aborted)  break;
        if (gen >= max_tokens)      { finish = "length"; final_len = text.len; break; }
        if (pos >= st->max_seq)     { finish = "length"; final_len = text.len; break; }

        int32_t one = id;       /* feed the token back to predict the next one */
        quasar_decode(st->ctx, &one, 1, st->logits);
        pos++;
    }
    if (!stop_found && final_len == 0) final_len = text.len;

    /* Flush any held-back tail (only when we didn't stop on a stop-string). */
    if (!aborted && !stop_found && cb && final_len > emitted)
        cb(ud, text.buf + emitted, (int)(final_len - emitted));

    sb_append_len(out_text, text.buf, final_len);
    if (out_n) *out_n = gen;
    sb_free(&text);
    return finish;
}

/* ---- sampling + request params ---------------------------------------- */

static void configure_sampler(quasar_sampler *smp, const json_value *req) {
    float temp = (float)json_num(json_get(req, "temperature"), 0.7);
    float top_p = (float)json_num(json_get(req, "top_p"), 0.95);
    int   top_k = (int)json_num(json_get(req, "top_k"), 40);          /* extension */
    float min_p = (float)json_num(json_get(req, "min_p"), 0.05);      /* extension */
    float rpen  = (float)json_num(json_get(req, "repeat_penalty"), 1.1);  /* extension */
    int   rlast = (int)json_num(json_get(req, "repeat_last_n"), 64);  /* extension */
    uint64_t seed = (uint64_t)(long long)json_num(json_get(req, "seed"), 0);
    quasar_sampler_config(smp, temp, top_p, top_k, min_p, rpen, rlast, seed);
}

static int read_max_tokens(const json_value *req, int room) {
    const json_value *mt = json_get(req, "max_tokens");
    if (!mt) mt = json_get(req, "max_completion_tokens");
    int n = (int)json_num(mt, 0);
    if (n <= 0) n = 512;
    if (n > room) n = room;
    return n;
}

/* Collect "stop" (string or array of strings) into a malloc'd char* array that
 * points into the JSON tree (valid as long as `req` lives). Returns count. */
static int collect_stops(const json_value *req, char ***out) {
    const json_value *s = json_get(req, "stop");
    *out = NULL;
    if (!s) return 0;
    if (s->type == JSON_STR) {
        char **a = malloc(sizeof *a);
        if (!a) return 0;
        a[0] = s->str.s; *out = a; return 1;
    }
    if (s->type == JSON_ARR) {
        char **a = malloc(s->arr.n * sizeof *a + 1);
        if (!a) return 0;
        int n = 0;
        for (size_t i = 0; i < s->arr.n; i++)
            if (s->arr.items[i]->type == JSON_STR) a[n++] = s->arr.items[i]->str.s;
        *out = a; return n;
    }
    return 0;
}

/* ---- chat message handling -------------------------------------------- */

/* Duplicate a message's "content" into a malloc'd string. Handles both a plain
 * string and the array-of-parts form ([{type:"text",text:"..."}, ...]). */
static char *content_dup(const json_value *content) {
    if (!content) return strdup("");
    if (content->type == JSON_STR) return strdup(content->str.s);
    if (content->type == JSON_ARR) {
        strbuf sb; sb_init(&sb);
        for (size_t i = 0; i < content->arr.n; i++) {
            const json_value *t = json_get(content->arr.items[i], "text");
            if (t && t->type == JSON_STR) sb_append_len(&sb, t->str.s, t->str.len);
        }
        if (!sb.buf) return strdup("");
        return sb.buf;            /* ownership transferred to caller */
    }
    return strdup("");
}

static void gen_id(server_st *st, const char *prefix, char *buf, size_t cap) {
    snprintf(buf, cap, "%s%lx%lx", prefix, (unsigned long)time(NULL), ++st->req_counter);
}

/* ---- streaming (SSE) -------------------------------------------------- */

typedef struct {
    int         fd;
    const char *id;
    long        created;
    const char *model;
    int         mode;     /* 0 = chat (delta.content), 1 = completion (choice.text) */
    int         ok;
} sse_ud;

static int sse_send_delta(void *ud_, const char *bytes, int n) {
    sse_ud *u = (sse_ud *)ud_;
    if (!u->ok) return -1;
    strbuf sb; sb_init(&sb);
    sb_append(&sb, "data: {\"id\":");
    sb_json_str(&sb, u->id, strlen(u->id));
    sb_printf(&sb, ",\"object\":\"%s\",\"created\":%ld,\"model\":",
              u->mode ? "text_completion" : "chat.completion.chunk", u->created);
    sb_json_str(&sb, u->model, strlen(u->model));
    if (u->mode) {
        sb_append(&sb, ",\"choices\":[{\"index\":0,\"text\":");
        sb_json_str(&sb, bytes, (size_t)n);
        sb_append(&sb, ",\"finish_reason\":null}]}\n\n");
    } else {
        sb_append(&sb, ",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
        sb_json_str(&sb, bytes, (size_t)n);
        sb_append(&sb, "},\"finish_reason\":null}]}\n\n");
    }
    int rc = send_all(u->fd, sb.buf, sb.len);
    sb_free(&sb);
    if (rc != 0) { u->ok = 0; return -1; }
    return 0;
}

static const char *SSE_HEADERS =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Connection: close\r\n\r\n";

/* ---- endpoint: /v1/chat/completions ----------------------------------- */

static void handle_chat(server_st *st, int fd, const json_value *req) {
    const json_value *messages = json_get(req, "messages");
    if (!messages || messages->type != JSON_ARR || messages->arr.n == 0) {
        send_error(fd, 400, "Bad Request", "invalid_request_error",
                   "'messages' must be a non-empty array");
        return;
    }

    /* Build the ChatML prompt from the messages. */
    int nmsg = (int)messages->arr.n;
    quasar_msg *msgs = calloc((size_t)nmsg, sizeof *msgs);
    char      **owned = calloc((size_t)nmsg, sizeof *owned);
    if (!msgs || !owned) { free(msgs); free(owned); send_error(fd, 500, "Internal Server Error", "server_error", "out of memory"); return; }
    for (int i = 0; i < nmsg; i++) {
        const json_value *mo = messages->arr.items[i];
        msgs[i].role    = json_str(json_get(mo, "role"), "user");
        owned[i]        = content_dup(json_get(mo, "content"));
        msgs[i].content = owned[i] ? owned[i] : "";
    }
    int enable_thinking = json_bool(json_get(req, "enable_thinking"), 0);
    size_t plen = 0;
    char *prompt = quasar_chat_render(msgs, nmsg, true, enable_thinking, &plen);
    for (int i = 0; i < nmsg; i++) free(owned[i]);
    free(owned); free(msgs);
    if (!prompt) { send_error(fd, 500, "Internal Server Error", "server_error", "prompt render failed"); return; }

    size_t np = 0;
    int32_t *ids = quasar_tokenize(st->tk, prompt, plen, false, &np);
    free(prompt);
    if (!ids || np == 0) { free(ids); send_error(fd, 400, "Bad Request", "invalid_request_error", "empty prompt after tokenization"); return; }
    if ((int)np >= st->max_seq) {
        free(ids);
        char msg[160];
        snprintf(msg, sizeof msg, "prompt is %zu tokens; context limit is %d", np, st->max_seq);
        send_error(fd, 400, "Bad Request", "context_length_exceeded", msg);
        return;
    }

    int stream = json_bool(json_get(req, "stream"), 0);
    int room   = st->max_seq - (int)np;
    int max_tokens = read_max_tokens(req, room);
    const char *model = json_str(json_get(req, "model"), st->model_id);

    quasar_sampler *smp = quasar_sampler_new();
    configure_sampler(smp, req);
    char **stops = NULL;
    int n_stops = collect_stops(req, &stops);

    char id[96]; gen_id(st, "chatcmpl-", id, sizeof id);
    long created = (long)time(NULL);
    double t0 = now_s();

    const char *finish; int n_comp = 0;

    if (stream) {
        if (send_all(fd, SSE_HEADERS, strlen(SSE_HEADERS)) != 0) { free(ids); free(stops); quasar_sampler_free(smp); return; }
        /* opening chunk: role only */
        {
            strbuf sb; sb_init(&sb);
            sb_append(&sb, "data: {\"id\":"); sb_json_str(&sb, id, strlen(id));
            sb_printf(&sb, ",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", created);
            sb_json_str(&sb, model, strlen(model));
            sb_append(&sb, ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n");
            send_all(fd, sb.buf, sb.len); sb_free(&sb);
        }
        sse_ud u = { fd, id, created, model, 0, 1 };
        strbuf text; sb_init(&text);
        finish = run_generation(st, ids, (int)np, max_tokens, smp, stops, n_stops, sse_send_delta, &u, &text, &n_comp);
        sb_free(&text);
        if (u.ok) {
            strbuf sb; sb_init(&sb);
            sb_append(&sb, "data: {\"id\":"); sb_json_str(&sb, id, strlen(id));
            sb_printf(&sb, ",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", created);
            sb_json_str(&sb, model, strlen(model));
            sb_printf(&sb, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n", finish);
            sb_append(&sb, "data: [DONE]\n\n");
            send_all(fd, sb.buf, sb.len); sb_free(&sb);
        }
    } else {
        strbuf text; sb_init(&text);
        finish = run_generation(st, ids, (int)np, max_tokens, smp, stops, n_stops, NULL, NULL, &text, &n_comp);
        strbuf sb; sb_init(&sb);
        sb_append(&sb, "{\"id\":"); sb_json_str(&sb, id, strlen(id));
        sb_printf(&sb, ",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", created);
        sb_json_str(&sb, model, strlen(model));
        sb_append(&sb, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        sb_json_str(&sb, text.buf ? text.buf : "", text.len);
        sb_printf(&sb, "},\"finish_reason\":\"%s\"}],", finish);
        sb_printf(&sb, "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                  (int)np, n_comp, (int)np + n_comp);
        send_json_response(fd, 200, "OK", &sb);
        sb_free(&sb); sb_free(&text);
    }

    double dt = now_s() - t0;
    fprintf(stderr, "  chat: %d prompt + %d gen tok in %.2fs (%.1f tok/s) finish=%s%s\n",
            (int)np, n_comp, dt, dt > 0 ? n_comp / dt : 0.0, finish, stream ? " [stream]" : "");

    free(ids); free(stops); quasar_sampler_free(smp);
}

/* ---- endpoint: /v1/completions (legacy) ------------------------------- */

static void handle_completions(server_st *st, int fd, const json_value *req) {
    const json_value *pv = json_get(req, "prompt");
    const char *prompt = json_str(pv, NULL);
    if (!prompt) { send_error(fd, 400, "Bad Request", "invalid_request_error", "'prompt' must be a string"); return; }

    size_t np = 0;
    int32_t *ids = quasar_tokenize(st->tk, prompt, strlen(prompt), false, &np);
    if (!ids || np == 0) { free(ids); send_error(fd, 400, "Bad Request", "invalid_request_error", "empty prompt"); return; }
    if ((int)np >= st->max_seq) { free(ids); send_error(fd, 400, "Bad Request", "context_length_exceeded", "prompt exceeds context limit"); return; }

    int stream = json_bool(json_get(req, "stream"), 0);
    int room = st->max_seq - (int)np;
    int max_tokens = read_max_tokens(req, room);
    const char *model = json_str(json_get(req, "model"), st->model_id);

    quasar_sampler *smp = quasar_sampler_new();
    configure_sampler(smp, req);
    char **stops = NULL; int n_stops = collect_stops(req, &stops);

    char id[96]; gen_id(st, "cmpl-", id, sizeof id);
    long created = (long)time(NULL);
    const char *finish; int n_comp = 0;

    if (stream) {
        if (send_all(fd, SSE_HEADERS, strlen(SSE_HEADERS)) != 0) { free(ids); free(stops); quasar_sampler_free(smp); return; }
        sse_ud u = { fd, id, created, model, 1, 1 };
        strbuf text; sb_init(&text);
        finish = run_generation(st, ids, (int)np, max_tokens, smp, stops, n_stops, sse_send_delta, &u, &text, &n_comp);
        sb_free(&text);
        if (u.ok) {
            strbuf sb; sb_init(&sb);
            sb_append(&sb, "data: {\"id\":"); sb_json_str(&sb, id, strlen(id));
            sb_printf(&sb, ",\"object\":\"text_completion\",\"created\":%ld,\"model\":", created);
            sb_json_str(&sb, model, strlen(model));
            sb_printf(&sb, ",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"%s\"}]}\n\n", finish);
            sb_append(&sb, "data: [DONE]\n\n");
            send_all(fd, sb.buf, sb.len); sb_free(&sb);
        }
    } else {
        strbuf text; sb_init(&text);
        finish = run_generation(st, ids, (int)np, max_tokens, smp, stops, n_stops, NULL, NULL, &text, &n_comp);
        strbuf sb; sb_init(&sb);
        sb_append(&sb, "{\"id\":"); sb_json_str(&sb, id, strlen(id));
        sb_printf(&sb, ",\"object\":\"text_completion\",\"created\":%ld,\"model\":", created);
        sb_json_str(&sb, model, strlen(model));
        sb_append(&sb, ",\"choices\":[{\"index\":0,\"text\":");
        sb_json_str(&sb, text.buf ? text.buf : "", text.len);
        sb_printf(&sb, ",\"finish_reason\":\"%s\"}],", finish);
        sb_printf(&sb, "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                  (int)np, n_comp, (int)np + n_comp);
        send_json_response(fd, 200, "OK", &sb);
        sb_free(&sb); sb_free(&text);
    }

    free(ids); free(stops); quasar_sampler_free(smp);
}

/* ---- endpoint: /v1/models --------------------------------------------- */

static void handle_models(server_st *st, int fd) {
    strbuf sb; sb_init(&sb);
    sb_append(&sb, "{\"object\":\"list\",\"data\":[{\"id\":");
    sb_json_str(&sb, st->model_id, strlen(st->model_id));
    sb_printf(&sb, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"quasar\"}]}", (long)time(NULL));
    send_json_response(fd, 200, "OK", &sb);
    sb_free(&sb);
}

/* ---- routing ---------------------------------------------------------- */

static int path_is(const char *path, const char *a, const char *b) {
    return strcmp(path, a) == 0 || strcmp(path, b) == 0;
}

static void handle_connection(server_st *st, int fd) {
    http_req r;
    if (read_request(fd, &r) != 0) return;

    int is_post = strcmp(r.method, "POST") == 0;
    int is_get  = strcmp(r.method, "GET") == 0;

    if (strcmp(r.method, "OPTIONS") == 0) {   /* CORS preflight */
        const char *resp =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            "Content-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(fd, resp, strlen(resp));
        http_req_free(&r);
        return;
    }

    if (is_get && (strcmp(r.path, "/") == 0 || strcmp(r.path, "/health") == 0)) {
        strbuf sb; sb_init(&sb);
        sb_append(&sb, "{\"status\":\"ok\",\"model\":");
        sb_json_str(&sb, st->model_id, strlen(st->model_id));
        sb_append(&sb, "}");
        send_json_response(fd, 200, "OK", &sb);
        sb_free(&sb);
        http_req_free(&r);
        return;
    }
    if (is_get && path_is(r.path, "/v1/models", "/models")) {
        handle_models(st, fd);
        http_req_free(&r);
        return;
    }

    if (path_is(r.path, "/v1/chat/completions", "/chat/completions") ||
        path_is(r.path, "/v1/completions", "/completions")) {
        if (!is_post) { send_error(fd, 405, "Method Not Allowed", "invalid_request_error", "use POST"); http_req_free(&r); return; }
        char errbuf[128];
        json_value *req = json_parse(r.body, r.body_len, errbuf, sizeof errbuf);
        if (!req) {
            char msg[200]; snprintf(msg, sizeof msg, "invalid JSON: %s", errbuf[0] ? errbuf : "parse error");
            send_error(fd, 400, "Bad Request", "invalid_request_error", msg);
            http_req_free(&r);
            return;
        }
        if (strstr(r.path, "chat")) handle_chat(st, fd, req);
        else                        handle_completions(st, fd, req);
        json_free(req);
        http_req_free(&r);
        return;
    }

    send_error(fd, 404, "Not Found", "invalid_request_error", "unknown route");
    http_req_free(&r);
}

/* ---- listen socket + accept loop -------------------------------------- */

static int tcp_listen(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (!host || strcmp(host, "0.0.0.0") == 0) addr.sin_addr.s_addr = htonl(INADDR_ANY);
    else if (strcmp(host, "localhost") == 0)   addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    else if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) { close(fd); return -1; }

    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) != 0) { close(fd); return -1; }
    if (listen(fd, 16) != 0) { close(fd); return -1; }
    return fd;
}

int quasar_serve(const char *model_path, const quasar_server_opts *opts) {
    const char *host = opts && opts->host ? opts->host : "127.0.0.1";
    int port = opts && opts->port ? opts->port : 8080;
    int max_seq = opts && opts->ctx > 0 ? opts->ctx : 8192;
    int use_metal = opts ? opts->use_metal : 0;

    signal(SIGPIPE, SIG_IGN);   /* a client vanishing mid-write must not kill us */

    char err[256];
    fprintf(stderr, "quasar: loading %s ...\n", model_path);
    quasar_model *m = quasar_model_load(model_path, err, sizeof err);
    if (!m) { fprintf(stderr, "quasar: load failed: %s\n", err); return 1; }
    if (m->missing_tensors) { fprintf(stderr, "quasar: model incomplete (%d tensors missing)\n", m->missing_tensors); quasar_model_free(m); return 1; }
    quasar_tokenizer *tk = quasar_tokenizer_load(m->gguf, err, sizeof err);
    if (!tk) { fprintf(stderr, "quasar: tokenizer: %s\n", err); quasar_model_free(m); return 1; }

    int metal_ready = 0;
    if (use_metal) {
        if (quasar_metal_init() == 0 && quasar_metal_set_weights(m->gguf->map, m->gguf->map_size) == 0)
            metal_ready = 1;
        else fprintf(stderr, "quasar: Metal unavailable; using CPU reference (slow)\n");
    }

    if (max_seq > (int)m->hp.ctx_train && m->hp.ctx_train > 0) max_seq = (int)m->hp.ctx_train;
    quasar_ctx *ctx = quasar_ctx_new(m, max_seq, metal_ready);
    if (!ctx) { fprintf(stderr, "quasar: failed to allocate context\n"); quasar_tokenizer_free(tk); quasar_model_free(m); return 1; }

    server_st st;
    memset(&st, 0, sizeof st);
    st.m = m; st.tk = tk; st.ctx = ctx;
    st.max_seq = max_seq;
    st.n_vocab = (int)m->hp.n_vocab;
    st.eos = quasar_token_eos(tk);
    st.use_metal = metal_ready;
    st.logits = malloc((size_t)st.n_vocab * sizeof(float));
    if (m->hp.name[0]) snprintf(st.model_id, sizeof st.model_id, "%s", m->hp.name);
    else               snprintf(st.model_id, sizeof st.model_id, "quasar");
    if (!st.logits) { fprintf(stderr, "quasar: out of memory\n"); quasar_ctx_free(ctx); quasar_tokenizer_free(tk); quasar_model_free(m); return 1; }

    /* Warm up: page in the working set and warm the GPU pipelines so the first
     * real request runs at full speed instead of paying cold-start. */
    {
        size_t wn = 0;
        int32_t *wids = quasar_tokenize(tk, "Hello", 5, false, &wn);
        if (wids && wn > 0) {
            quasar_kv_reset(ctx);
            quasar_decode(ctx, wids, (int)wn, st.logits);
            for (int i = 0; i < 4; i++) {
                int bi = 0;
                for (int v = 1; v < st.n_vocab; v++) if (st.logits[v] > st.logits[bi]) bi = v;
                int32_t one = bi;
                quasar_decode(ctx, &one, 1, st.logits);
            }
        }
        free(wids);
        quasar_kv_reset(ctx);
    }

    int lfd = tcp_listen(host, port);
    if (lfd < 0) { fprintf(stderr, "quasar: cannot bind %s:%d (%s)\n", host, port, strerror(errno)); free(st.logits); quasar_ctx_free(ctx); quasar_tokenizer_free(tk); quasar_model_free(m); return 1; }

    fprintf(stderr,
        "quasar: serving \"%s\"\n"
        "  backend:  %s\n"
        "  context:  %d tokens\n"
        "  endpoint: http://%s:%d/v1\n"
        "  try: curl http://%s:%d/v1/chat/completions -H 'content-type: application/json' \\\n"
        "         -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'\n",
        st.model_id, metal_ready ? "Metal (Apple GPU)" : "CPU reference (slow)",
        max_seq, host, port, host, port);

    for (;;) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) { if (errno == EINTR) continue; break; }
#ifdef SO_NOSIGPIPE
        int one = 1; setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        handle_connection(&st, cfd);
        close(cfd);
    }

    close(lfd);
    free(st.logits);
    quasar_ctx_free(ctx);
    quasar_tokenizer_free(tk);
    quasar_model_free(m);
    return 0;
}
