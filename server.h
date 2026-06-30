/* server.h - an OpenAI-compatible HTTP server for Quasar.
 *
 * Serves /v1/chat/completions (streaming via SSE + non-streaming),
 * /v1/completions, and /v1/models over a tiny zero-dependency HTTP/1.1 server.
 * Requests are handled serially: there is one model, one KV cache, and one GPU
 * pipeline, so generations cannot overlap. */
#ifndef QUASAR_SERVER_H
#define QUASAR_SERVER_H

typedef struct {
    const char *host;   /* bind address; default "127.0.0.1" ("0.0.0.0" = all) */
    int         port;   /* default 8080 */
    int         ctx;    /* max sequence length = KV budget; default 8192 */
    int         use_metal;
} quasar_server_opts;

/* Load the model and serve until killed. Returns nonzero on fatal setup error. */
int quasar_serve(const char *model_path, const quasar_server_opts *opts);

#endif /* QUASAR_SERVER_H */
