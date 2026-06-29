/* metal.m - see metal.h. Objective-C; build with -fobjc-arc. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal.h"
#include "gguf.h"
#include <unistd.h>

/* ---- Metal Shading Language kernels (compiled at runtime) ------------- */
/* One thread per output row: dequantize that row's blocks and dot with x.
 * x_off/y_off let many gemvs share one input/output buffer in one dispatch
 * batch (so a whole token's matmuls submit with a single sync). */
static const char *MSL =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct gemv_args { ulong row_base; uint n_in; uint n_out; uint row_bytes; uint x_off; uint y_off; };\n"
"\n"
"kernel void matvec_f32(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],\n"
"                       device float* y [[buffer(2)]], constant gemv_args& a [[buffer(3)]], uint r [[thread_position_in_grid]]) {\n"
"  if (r >= a.n_out) return;\n"
"  device const float* w = (device const float*)(W + a.row_base + (ulong)r * a.row_bytes);\n"
"  device const float* xx = x + a.x_off; float acc = 0.0f;\n"
"  for (uint c = 0; c < a.n_in; c++) acc += w[c] * xx[c];\n"
"  y[a.y_off + r] = acc;\n"
"}\n"
"\n"
"inline uchar2 gsmk4(uint j, device const uchar* q) {\n"
"  uchar d, m;\n"
"  if (j < 4) { d = q[j] & 63; m = q[j+4] & 63; }\n"
"  else { d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }\n"
"  return uchar2(d, m);\n"
"}\n"
"kernel void matvec_q4k(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],\n"
"                       device float* y [[buffer(2)]], constant gemv_args& a [[buffer(3)]], uint r [[thread_position_in_grid]]) {\n"
"  if (r >= a.n_out) return;\n"
"  device const uchar* row = W + a.row_base + (ulong)r * a.row_bytes;\n"
"  device const float* xx = x + a.x_off; float acc = 0.0f; uint nb = a.n_in / 256;\n"
"  for (uint i = 0; i < nb; i++) {\n"
"    device const uchar* b = row + (ulong)i * 144;\n"
"    float d = float(*(device const half*)(b)); float dmin = float(*(device const half*)(b + 2));\n"
"    device const uchar* scales = b + 4; device const uchar* qbase = b + 16; uint xo = i * 256;\n"
"    for (uint jj = 0; jj < 4; jj++) {\n"
"      uint j = jj * 64; uint is = jj * 2;\n"
"      uchar2 s0 = gsmk4(is + 0, scales); uchar2 s1 = gsmk4(is + 1, scales);\n"
"      float d1 = d*float(s0.x), m1 = dmin*float(s0.y); float d2 = d*float(s1.x), m2 = dmin*float(s1.y);\n"
"      device const uchar* q = qbase + jj * 32;\n"
"      for (uint l = 0; l < 32; l++) acc += (d1 * float(q[l] & 0xF) - m1) * xx[xo + j + l];\n"
"      for (uint l = 0; l < 32; l++) acc += (d2 * float(q[l] >> 4)  - m2) * xx[xo + j + 32 + l];\n"
"    }\n"
"  }\n"
"  y[a.y_off + r] = acc;\n"
"}\n"
"\n"
"kernel void matvec_q6k(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],\n"
"                       device float* y [[buffer(2)]], constant gemv_args& a [[buffer(3)]], uint r [[thread_position_in_grid]]) {\n"
"  if (r >= a.n_out) return;\n"
"  device const uchar* row = W + a.row_base + (ulong)r * a.row_bytes;\n"
"  device const float* xx = x + a.x_off; float acc = 0.0f; uint nb = a.n_in / 256;\n"
"  for (uint i = 0; i < nb; i++) {\n"
"    device const uchar* b  = row + (ulong)i * 210;\n"
"    device const uchar* ql = b; device const uchar* qh = b + 128;\n"
"    device const char*  sc = (device const char*)(b + 192); float d = float(*(device const half*)(b + 208)); uint xo = i * 256;\n"
"    for (uint nn = 0; nn < 256; nn += 128) {\n"
"      for (uint l = 0; l < 32; l++) {\n"
"        uint is = l / 16;\n"
"        int q1 = int((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;\n"
"        int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;\n"
"        int q3 = int((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;\n"
"        int q4 = int((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;\n"
"        acc += d * float(sc[is + 0]) * float(q1) * xx[xo + nn + l];\n"
"        acc += d * float(sc[is + 2]) * float(q2) * xx[xo + nn + l + 32];\n"
"        acc += d * float(sc[is + 4]) * float(q3) * xx[xo + nn + l + 64];\n"
"        acc += d * float(sc[is + 6]) * float(q4) * xx[xo + nn + l + 96];\n"
"      }\n"
"      ql += 64; qh += 32; sc += 8;\n"
"    }\n"
"  }\n"
"  y[a.y_off + r] = acc;\n"
"}\n"
"\n"
"kernel void matvec_q2k(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],\n"
"                       device float* y [[buffer(2)]], constant gemv_args& a [[buffer(3)]], uint r [[thread_position_in_grid]]) {\n"
"  if (r >= a.n_out) return;\n"
"  device const uchar* row = W + a.row_base + (ulong)r * a.row_bytes;\n"
"  device const float* xx = x + a.x_off; float acc = 0.0f; uint nb = a.n_in / 256;\n"
"  for (uint i = 0; i < nb; i++) {\n"
"    device const uchar* b = row + (ulong)i * 84;\n"
"    device const uchar* scales = b; device const uchar* q = b + 16;\n"
"    float d = float(*(device const half*)(b + 80)); float dmin = float(*(device const half*)(b + 82));\n"
"    uint xo = i * 256, oi = 0; int is = 0;\n"
"    for (uint nn = 0; nn < 256; nn += 128) {\n"
"      int shift = 0;\n"
"      for (int j = 0; j < 4; j++) {\n"
"        uchar sc = scales[is++]; float dl = d*float(sc & 0xF), ml = dmin*float(sc >> 4);\n"
"        for (uint l = 0; l < 16; l++) { acc += (dl*float((q[l]    >> shift) & 3) - ml) * xx[xo+oi]; oi++; }\n"
"        sc = scales[is++]; dl = d*float(sc & 0xF); ml = dmin*float(sc >> 4);\n"
"        for (uint l = 0; l < 16; l++) { acc += (dl*float((q[l+16] >> shift) & 3) - ml) * xx[xo+oi]; oi++; }\n"
"        shift += 2;\n"
"      }\n"
"      q += 32;\n"
"    }\n"
"  }\n"
"  y[a.y_off + r] = acc;\n"
"}\n";

/* ---- backend state ---------------------------------------------------- */

typedef struct { uint64_t row_base; uint32_t n_in, n_out, row_bytes, x_off, y_off; } gemv_args;

#define QM_MAX_CHUNKS 64

static id<MTLDevice>               g_dev;
static id<MTLCommandQueue>         g_queue;
static id<MTLComputePipelineState> g_f32, g_q4k, g_q6k, g_q2k;
static id<MTLBuffer>               g_chunk[QM_MAX_CHUNKS];
static uint64_t                    g_chunk_off[QM_MAX_CHUNKS];
static uint64_t                    g_chunk_len[QM_MAX_CHUNKS];
static int                         g_nchunks;
static id<MTLBuffer>               g_xbuf, g_ybuf;
static uint32_t                    g_xcap, g_ycap;
static char                        g_name[128];
static int                         g_ok;

static id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, const char *fn) {
    NSError *err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
    if (!f) { NSLog(@"quasar metal: missing kernel %s", fn); return nil; }
    id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) NSLog(@"quasar metal: pipeline %s: %@", fn, err);
    return p;
}

int quasar_metal_init(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return -1;
        g_queue = [g_dev newCommandQueue];
        snprintf(g_name, sizeof g_name, "%s", [[g_dev name] UTF8String]);
        NSError *err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:MSL] options:nil error:&err];
        if (!lib) { NSLog(@"quasar metal: library compile failed: %@", err); return -2; }
        g_f32 = make_pso(lib, "matvec_f32");
        g_q4k = make_pso(lib, "matvec_q4k");
        g_q6k = make_pso(lib, "matvec_q6k");
        g_q2k = make_pso(lib, "matvec_q2k");
        if (!g_f32 || !g_q4k || !g_q6k || !g_q2k) return -3;
        g_ok = 1;
        return 0;
    }
}

int quasar_metal_available(void) { return g_ok; }
const char *quasar_metal_device_name(void) { return g_ok ? g_name : "(none)"; }

int quasar_metal_set_weights(const void *base, size_t len) {
    if (!g_ok) return -1;
    @autoreleasepool {
        uint64_t pg = (uint64_t)getpagesize();
        uint64_t maxb = (uint64_t)g_dev.maxBufferLength;
        uint64_t chunk_len = maxb;
        const uint64_t CAP = 12ULL << 30;
        if (chunk_len > CAP) chunk_len = CAP;
        chunk_len &= ~(pg - 1);
        uint64_t overlap = (1ULL << 30) & ~(pg - 1);
        if (overlap >= chunk_len) overlap = chunk_len / 4;
        uint64_t stride = chunk_len - overlap;

        g_nchunks = 0;
        for (uint64_t start = 0; start < (uint64_t)len && g_nchunks < QM_MAX_CHUNKS; start += stride) {
            uint64_t clen = chunk_len;
            if (start + clen >= (uint64_t)len) clen = (uint64_t)len - start;
            uint64_t rlen = (clen + pg - 1) & ~(pg - 1);
            id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)((const uint8_t *)base + start)
                                                       length:(NSUInteger)rlen options:MTLResourceStorageModeShared deallocator:nil];
            if (!b) { fprintf(stderr, "quasar metal: chunk %d (%.2f GB) alloc failed\n", g_nchunks, rlen / 1e9); return -2; }
            g_chunk[g_nchunks] = b; g_chunk_off[g_nchunks] = start; g_chunk_len[g_nchunks] = clen;
            g_nchunks++;
            if (start + clen >= (uint64_t)len) break;
        }
        return g_nchunks > 0 ? 0 : -3;
    }
}

static void ensure_io(uint32_t in_floats, uint32_t out_floats) {
    if (g_xcap < in_floats)  { g_xbuf = [g_dev newBufferWithLength:(NSUInteger)in_floats  * 4 options:MTLResourceStorageModeShared]; g_xcap = in_floats; }
    if (g_ycap < out_floats) { g_ybuf = [g_dev newBufferWithLength:(NSUInteger)out_floats * 4 options:MTLResourceStorageModeShared]; g_ycap = out_floats; }
}

static id<MTLComputePipelineState> pso_for(uint32_t type) {
    return (type == GGML_TYPE_F32) ? g_f32 : (type == GGML_TYPE_Q4_K) ? g_q4k
         : (type == GGML_TYPE_Q6_K) ? g_q6k : (type == GGML_TYPE_Q2_K) ? g_q2k : nil;
}

/* find the chunk fully containing [row0, row0+extent); -1 if none */
static int chunk_for(uint64_t row0, uint64_t extent) {
    for (int i = 0; i < g_nchunks; i++)
        if (g_chunk_off[i] <= row0 && g_chunk_off[i] + g_chunk_len[i] >= row0 + extent) return i;
    return -1;
}

void quasar_metal_gemv(uint32_t type, uint64_t row0, const float *x, int n_in, int n_out, float *y) {
    if (!g_ok || g_nchunks == 0) return;
    id<MTLComputePipelineState> pso = pso_for(type);
    if (!pso) return;
    uint32_t blck = ggml_blck_size(type); size_t tsz = ggml_type_size(type);
    if (!blck || !tsz) return;
    uint32_t row_bytes = (uint32_t)((uint64_t)n_in / blck * tsz);
    int ci = chunk_for(row0, (uint64_t)n_out * row_bytes);
    if (ci < 0) { fprintf(stderr, "quasar metal: gemv span in no chunk\n"); return; }

    @autoreleasepool {
        ensure_io((uint32_t)n_in, (uint32_t)n_out);
        memcpy(g_xbuf.contents, x, (size_t)n_in * sizeof(float));
        gemv_args args = { row0 - g_chunk_off[ci], (uint32_t)n_in, (uint32_t)n_out, row_bytes, 0, 0 };
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:g_chunk[ci] offset:0 atIndex:0];
        [enc setBuffer:g_xbuf offset:0 atIndex:1];
        [enc setBuffer:g_ybuf offset:0 atIndex:2];
        [enc setBytes:&args length:sizeof args atIndex:3];
        NSUInteger tpt = pso.maxTotalThreadsPerThreadgroup; if (tpt > 256) tpt = 256;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)n_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(y, g_ybuf.contents, (size_t)n_out * sizeof(float));
    }
}

void quasar_metal_gemv_batch(const quasar_gemv_job *jobs, int nj,
                             const float *in, int in_floats, float *out, int out_floats) {
    if (!g_ok || g_nchunks == 0 || nj <= 0) return;
    @autoreleasepool {
        ensure_io((uint32_t)in_floats, (uint32_t)out_floats);
        memcpy(g_xbuf.contents, in, (size_t)in_floats * sizeof(float));
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBuffer:g_xbuf offset:0 atIndex:1];
        [enc setBuffer:g_ybuf offset:0 atIndex:2];
        for (int j = 0; j < nj; j++) {
            const quasar_gemv_job *J = &jobs[j];
            id<MTLComputePipelineState> pso = pso_for(J->type);
            if (!pso) continue;
            uint32_t blck = ggml_blck_size(J->type); size_t tsz = ggml_type_size(J->type);
            if (!blck || !tsz) continue;
            uint32_t row_bytes = (uint32_t)((uint64_t)J->n_in / blck * tsz);
            int ci = chunk_for(J->row0, (uint64_t)J->n_out * row_bytes);
            if (ci < 0) { fprintf(stderr, "quasar metal: batch job span in no chunk\n"); continue; }
            gemv_args args = { J->row0 - g_chunk_off[ci], (uint32_t)J->n_in, (uint32_t)J->n_out, row_bytes, J->x_off, J->y_off };
            [enc setComputePipelineState:pso];
            [enc setBuffer:g_chunk[ci] offset:0 atIndex:0];
            [enc setBytes:&args length:sizeof args atIndex:3];
            NSUInteger tpt = pso.maxTotalThreadsPerThreadgroup; if (tpt > 256) tpt = 256;
            [enc dispatchThreads:MTLSizeMake((NSUInteger)J->n_out, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
        }
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(out, g_ybuf.contents, (size_t)out_floats * sizeof(float));
    }
}
