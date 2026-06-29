/* metal.h - Metal compute backend (C interface over Objective-C).
 *
 * The whole GGUF mmap is wrapped as a single zero-copy unified-memory buffer
 * (newBufferWithBytesNoCopy), so GPU kernels read weights straight from the
 * mapped file with no upload. gemv dequantizes per row on the GPU. */
#ifndef QUASAR_METAL_H
#define QUASAR_METAL_H

#include <stdint.h>
#include <stddef.h>

int  quasar_metal_init(void);                              /* 0 ok, <0 on failure */
int  quasar_metal_available(void);                         /* 1 if init succeeded  */
const char *quasar_metal_device_name(void);

/* Wrap a page-aligned region (the model mmap) as the weights buffer. 0 on ok. */
int  quasar_metal_set_weights(const void *base, size_t len);

/* y[n_out] = W @ x[n_in], where each row of W (ggml `type`) starts at byte
 * offset `row0` within the weights buffer. Blocking (waits for completion). */
void quasar_metal_gemv(uint32_t ggml_type, uint64_t row0,
                       const float *x, int n_in, int n_out, float *y);

/* Batched gemv: many independent matmuls sharing one input/output buffer,
 * submitted in a single command buffer with one sync. Each job reads
 * in[x_off..] and writes out[y_off..]. */
typedef struct {
    uint32_t type;
    uint64_t row0;
    uint32_t x_off, y_off;   /* float offsets into in / out */
    int      n_in, n_out;
} quasar_gemv_job;

void quasar_metal_gemv_batch(const quasar_gemv_job *jobs, int njobs,
                             const float *in, int in_floats, float *out, int out_floats);

#endif /* QUASAR_METAL_H */
