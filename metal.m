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
"}\n"
"\n"
"struct expert_args { ulong base; ulong expert_bytes; uint n_in; uint n_out; uint row_bytes; uint x_stride; uint out_stride; uint nu; };\n"
"\n"
"kernel void topk_moe(device const float* router [[buffer(0)]], device int* sel_idx [[buffer(1)]],\n"
"                     device float* sel_w [[buffer(2)]], constant uint2& p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {\n"
"  if (tid != 0) return;\n"
"  uint NE = p.x, NU = p.y;\n"
"  float mx = router[0]; for (uint i = 1; i < NE; i++) mx = max(mx, router[i]);\n"
"  float sum = 0.0f; for (uint i = 0; i < NE; i++) sum += exp(router[i] - mx);\n"
"  bool taken[128]; for (uint i = 0; i < NE; i++) taken[i] = false;\n"
"  float wsum = 0.0f;\n"
"  for (uint s = 0; s < NU; s++) {\n"
"    int bi = -1; float best = -1e30f;\n"
"    for (uint i = 0; i < NE; i++) if (!taken[i] && router[i] > best) { best = router[i]; bi = i; }\n"
"    taken[bi] = true; sel_idx[s] = bi; float pr = exp(best - mx) / sum; sel_w[s] = pr; wsum += pr;\n"
"  }\n"
"  for (uint s = 0; s < NU; s++) sel_w[s] /= wsum;\n"
"}\n"
"\n"
"kernel void matvec_q2k_expert(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],\n"
"                              device float* y [[buffer(2)]], device const int* sel [[buffer(3)]],\n"
"                              constant expert_args& a [[buffer(4)]], uint g [[thread_position_in_grid]]) {\n"
"  uint slot = g / a.n_out, r = g % a.n_out;\n"
"  if (slot >= a.nu) return;\n"
"  int e = sel[slot];\n"
"  device const uchar* row = W + a.base + (ulong)e * a.expert_bytes + (ulong)r * a.row_bytes;\n"
"  device const float* xx = x + slot * a.x_stride;\n"
"  float acc = 0.0f; uint nb = a.n_in / 256;\n"
"  for (uint i = 0; i < nb; i++) {\n"
"    device const uchar* b = row + (ulong)i * 84;\n"
"    device const uchar* scales = b; device const uchar* q = b + 16;\n"
"    float d = float(*(device const half*)(b + 80)); float dmin = float(*(device const half*)(b + 82));\n"
"    uint xo = i * 256, oi = 0; int is = 0;\n"
"    for (uint nn = 0; nn < 256; nn += 128) { int shift = 0;\n"
"      for (int j = 0; j < 4; j++) {\n"
"        uchar sc = scales[is++]; float dl = d*float(sc & 0xF), ml = dmin*float(sc >> 4);\n"
"        for (uint l = 0; l < 16; l++) { acc += (dl*float((q[l]    >> shift) & 3) - ml) * xx[xo+oi]; oi++; }\n"
"        sc = scales[is++]; dl = d*float(sc & 0xF); ml = dmin*float(sc >> 4);\n"
"        for (uint l = 0; l < 16; l++) { acc += (dl*float((q[l+16] >> shift) & 3) - ml) * xx[xo+oi]; oi++; }\n"
"        shift += 2;\n"
"      } q += 32;\n"
"    }\n"
"  }\n"
"  y[slot * a.out_stride + r] = acc;\n"
"}\n"
"\n"
"kernel void silu_mul(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],\n"
"                     device float* hid [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
"  float g = gate[i]; hid[i] = (g / (1.0f + exp(-g))) * up[i];\n"
"}\n"
"\n"
"kernel void combine_moe(device const float* down [[buffer(0)]], device const float* sw [[buffer(1)]],\n"
"                        device float* moe_y [[buffer(2)]], constant uint2& p [[buffer(3)]], uint r [[thread_position_in_grid]]) {\n"
"  uint E = p.x, NU = p.y; float acc = 0.0f;\n"
"  for (uint s = 0; s < NU; s++) acc += sw[s] * down[s * E + r];\n"
"  moe_y[r] = acc;\n"
"}\n"
"\n"
"struct rms_args { ulong w_off; uint n; float eps; };\n"
"kernel void rmsnorm_k(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],\n"
"                      device const uchar* W [[buffer(2)]], constant rms_args& a [[buffer(3)]],\n"
"                      uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {\n"
"  threadgroup float sh[256];\n"
"  float ss = 0.0f; for (uint i = tid; i < a.n; i += tgs) { float v = in[i]; ss += v*v; }\n"
"  sh[tid] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  for (uint s = tgs/2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"  float scale = rsqrt(sh[0]/float(a.n) + a.eps);\n"
"  device const float* w = (device const float*)(W + a.w_off);\n"
"  for (uint i = tid; i < a.n; i += tgs) out[i] = in[i]*scale*w[i];\n"
"}\n"
"\n"
"struct qkr_args { ulong w_off; uint base; uint head_dim; uint pos; float eps; float theta; };\n"
"kernel void qknorm_rope_k(device float* qk [[buffer(0)]], device const uchar* W [[buffer(1)]],\n"
"                          constant qkr_args& a [[buffer(2)]], uint h [[threadgroup_position_in_grid]],\n"
"                          uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {\n"
"  device float* v = qk + a.base + h * a.head_dim;\n"
"  threadgroup float sh[128];\n"
"  float ss = 0.0f; for (uint i = tid; i < a.head_dim; i += tgs) { float x = v[i]; ss += x*x; }\n"
"  sh[tid] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  for (uint s = tgs/2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"  float scale = rsqrt(sh[0]/float(a.head_dim) + a.eps);\n"
"  device const float* w = (device const float*)(W + a.w_off);\n"
"  for (uint i = tid; i < a.head_dim; i += tgs) v[i] = v[i]*scale*w[i];\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  uint hf = a.head_dim/2;\n"
"  for (uint j = tid; j < hf; j += tgs) {\n"
"    float freq = pow(a.theta, -2.0f*float(j)/float(a.head_dim));\n"
"    float ang = float(a.pos)*freq; float c = cos(ang), sn = sin(ang);\n"
"    float x0 = v[j], x1 = v[j+hf];\n"
"    v[j] = x0*c - x1*sn; v[j+hf] = x0*sn + x1*c;\n"
"  }\n"
"}\n"
"\n"
"struct attn_args { uint n_head; uint n_head_kv; uint head_dim; uint pos; uint kv_off; float scale; };\n"
"kernel void attention_k(device const float* q [[buffer(0)]], device const float* Kc [[buffer(1)]],\n"
"                        device const float* Vc [[buffer(2)]], device float* att [[buffer(3)]],\n"
"                        constant attn_args& a [[buffer(4)]], uint h [[threadgroup_position_in_grid]],\n"
"                        uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {\n"
"  uint HD = a.head_dim, HK = a.n_head_kv, gsize = a.n_head / HK, kvh = h / gsize;\n"
"  device const float* qh = q + h*HD;\n"
"  threadgroup float sc[4096];\n"
"  uint P = a.pos + 1;\n"
"  for (uint s = tid; s < P; s += tgs) {\n"
"    device const float* ks = Kc + a.kv_off + s*HK*HD + kvh*HD;\n"
"    float dot = 0.0f; for (uint d = 0; d < HD; d++) dot += qh[d]*ks[d];\n"
"    sc[s] = dot * a.scale;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (tid == 0) {\n"
"    float mx = sc[0]; for (uint s = 1; s < P; s++) mx = max(mx, sc[s]);\n"
"    float sum = 0.0f; for (uint s = 0; s < P; s++) { sc[s] = exp(sc[s]-mx); sum += sc[s]; }\n"
"    float inv = 1.0f/sum; for (uint s = 0; s < P; s++) sc[s] *= inv;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  device float* ah = att + h*HD;\n"
"  for (uint d = tid; d < HD; d += tgs) {\n"
"    float acc = 0.0f; for (uint s = 0; s < P; s++) acc += sc[s]*Vc[a.kv_off + s*HK*HD + kvh*HD + d];\n"
"    ah[d] = acc;\n"
"  }\n"
"}\n"
"\n"
"kernel void add_k(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
"  dst[i] += src[i];\n"
"}\n";

/* ---- backend state ---------------------------------------------------- */

typedef struct { uint64_t row_base; uint32_t n_in, n_out, row_bytes, x_off, y_off; } gemv_args;

#define QM_MAX_CHUNKS 64

static id<MTLDevice>               g_dev;
static id<MTLCommandQueue>         g_queue;
static id<MTLComputePipelineState> g_f32, g_q4k, g_q6k, g_q2k;
static id<MTLComputePipelineState> g_topk, g_expert, g_silu, g_combine;
static id<MTLComputePipelineState> g_rms, g_qkr, g_attn, g_add;
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
        g_topk    = make_pso(lib, "topk_moe");
        g_expert  = make_pso(lib, "matvec_q2k_expert");
        g_silu    = make_pso(lib, "silu_mul");
        g_combine = make_pso(lib, "combine_moe");
        g_rms     = make_pso(lib, "rmsnorm_k");
        g_qkr     = make_pso(lib, "qknorm_rope_k");
        g_attn    = make_pso(lib, "attention_k");
        g_add     = make_pso(lib, "add_k");
        if (!g_f32 || !g_q4k || !g_q6k || !g_q2k || !g_topk || !g_expert || !g_silu || !g_combine ||
            !g_rms || !g_qkr || !g_attn || !g_add) return -3;
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

/* ---- GPU-resident MoE block (one command buffer, one sync) ------------ */

typedef struct { uint64_t base, expert_bytes; uint32_t n_in, n_out, row_bytes, x_stride, out_stride, nu; } expert_args;

static id<MTLBuffer> g_moe_xn, g_moe_router, g_moe_sel, g_moe_selw, g_moe_gate, g_moe_up, g_moe_hid, g_moe_down, g_moe_y;
static int g_moe_E, g_moe_FF, g_moe_NE, g_moe_NU;

static void ensure_moe(int E, int FF, int NE, int NU) {
    if (g_moe_xn && g_moe_E == E && g_moe_FF == FF && g_moe_NE == NE && g_moe_NU == NU) return;
    g_moe_E = E; g_moe_FF = FF; g_moe_NE = NE; g_moe_NU = NU;
    g_moe_xn     = [g_dev newBufferWithLength:(NSUInteger)E      * 4 options:MTLResourceStorageModeShared];
    g_moe_router = [g_dev newBufferWithLength:(NSUInteger)NE     * 4 options:MTLResourceStorageModeShared];
    g_moe_sel    = [g_dev newBufferWithLength:(NSUInteger)NU     * 4 options:MTLResourceStorageModeShared];
    g_moe_selw   = [g_dev newBufferWithLength:(NSUInteger)NU     * 4 options:MTLResourceStorageModeShared];
    g_moe_gate   = [g_dev newBufferWithLength:(NSUInteger)NU * FF * 4 options:MTLResourceStorageModeShared];
    g_moe_up     = [g_dev newBufferWithLength:(NSUInteger)NU * FF * 4 options:MTLResourceStorageModeShared];
    g_moe_hid    = [g_dev newBufferWithLength:(NSUInteger)NU * FF * 4 options:MTLResourceStorageModeShared];
    g_moe_down   = [g_dev newBufferWithLength:(NSUInteger)NU * E  * 4 options:MTLResourceStorageModeShared];
    g_moe_y      = [g_dev newBufferWithLength:(NSUInteger)E      * 4 options:MTLResourceStorageModeShared];
}

static void disp(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, NSUInteger n) {
    NSUInteger tpt = pso.maxTotalThreadsPerThreadgroup;
    if (tpt > 256) tpt = 256;
    if (tpt > n) tpt = n;
    if (tpt < 1) tpt = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpt, 1, 1)];
}

void quasar_metal_moe(const quasar_moe_desc *d, const float *xn, float *moe_y) {
    if (!g_ok || g_nchunks == 0) return;
    int E = d->E, FF = d->FF, NE = d->NE, NU = d->NU;
    ensure_moe(E, FF, NE, NU);
    memcpy(g_moe_xn.contents, xn, (size_t)E * 4);

    uint32_t fblck = ggml_blck_size(d->router_type); size_t ftsz = ggml_type_size(d->router_type);
    uint32_t router_rb = (uint32_t)((uint64_t)E / fblck * ftsz);
    uint32_t qblck = ggml_blck_size(d->expert_type); size_t qtsz = ggml_type_size(d->expert_type);
    uint32_t gate_rb = (uint32_t)((uint64_t)E  / qblck * qtsz);
    uint32_t down_rb = (uint32_t)((uint64_t)FF / qblck * qtsz);

    int rc = chunk_for(d->router_row0, (uint64_t)NE * router_rb);
    int gc = chunk_for(d->gate_row0,   (uint64_t)NE * d->expert_bytes);
    int uc = chunk_for(d->up_row0,     (uint64_t)NE * d->expert_bytes);
    int dc = chunk_for(d->down_row0,   (uint64_t)NE * d->expert_bytes);
    if (rc < 0 || gc < 0 || uc < 0 || dc < 0) { fprintf(stderr, "quasar metal: moe tensor in no chunk\n"); return; }

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        /* router (F32 matvec) -> g_moe_router */
        gemv_args ra = { d->router_row0 - g_chunk_off[rc], (uint32_t)E, (uint32_t)NE, router_rb, 0, 0 };
        [enc setComputePipelineState:g_f32];
        [enc setBuffer:g_chunk[rc] offset:0 atIndex:0];
        [enc setBuffer:g_moe_xn offset:0 atIndex:1];
        [enc setBuffer:g_moe_router offset:0 atIndex:2];
        [enc setBytes:&ra length:sizeof ra atIndex:3];
        disp(enc, g_f32, (NSUInteger)NE);

        /* softmax + top-NU selection -> g_moe_sel / g_moe_selw */
        uint32_t pk[2] = { (uint32_t)NE, (uint32_t)NU };
        [enc setComputePipelineState:g_topk];
        [enc setBuffer:g_moe_router offset:0 atIndex:0];
        [enc setBuffer:g_moe_sel offset:0 atIndex:1];
        [enc setBuffer:g_moe_selw offset:0 atIndex:2];
        [enc setBytes:pk length:sizeof pk atIndex:3];
        disp(enc, g_topk, 1);

        /* gate + up (indexed expert matmuls) */
        expert_args ga = { d->gate_row0 - g_chunk_off[gc], d->expert_bytes, (uint32_t)E, (uint32_t)FF, gate_rb, 0, (uint32_t)FF, (uint32_t)NU };
        [enc setComputePipelineState:g_expert];
        [enc setBuffer:g_chunk[gc] offset:0 atIndex:0];
        [enc setBuffer:g_moe_xn offset:0 atIndex:1];
        [enc setBuffer:g_moe_gate offset:0 atIndex:2];
        [enc setBuffer:g_moe_sel offset:0 atIndex:3];
        [enc setBytes:&ga length:sizeof ga atIndex:4];
        disp(enc, g_expert, (NSUInteger)NU * FF);

        expert_args ua = { d->up_row0 - g_chunk_off[uc], d->expert_bytes, (uint32_t)E, (uint32_t)FF, gate_rb, 0, (uint32_t)FF, (uint32_t)NU };
        [enc setComputePipelineState:g_expert];
        [enc setBuffer:g_chunk[uc] offset:0 atIndex:0];
        [enc setBuffer:g_moe_xn offset:0 atIndex:1];
        [enc setBuffer:g_moe_up offset:0 atIndex:2];
        [enc setBuffer:g_moe_sel offset:0 atIndex:3];
        [enc setBytes:&ua length:sizeof ua atIndex:4];
        disp(enc, g_expert, (NSUInteger)NU * FF);

        /* SwiGLU */
        [enc setComputePipelineState:g_silu];
        [enc setBuffer:g_moe_gate offset:0 atIndex:0];
        [enc setBuffer:g_moe_up offset:0 atIndex:1];
        [enc setBuffer:g_moe_hid offset:0 atIndex:2];
        disp(enc, g_silu, (NSUInteger)NU * FF);

        /* down (indexed) */
        expert_args da = { d->down_row0 - g_chunk_off[dc], d->expert_bytes, (uint32_t)FF, (uint32_t)E, down_rb, (uint32_t)FF, (uint32_t)E, (uint32_t)NU };
        [enc setComputePipelineState:g_expert];
        [enc setBuffer:g_chunk[dc] offset:0 atIndex:0];
        [enc setBuffer:g_moe_hid offset:0 atIndex:1];
        [enc setBuffer:g_moe_down offset:0 atIndex:2];
        [enc setBuffer:g_moe_sel offset:0 atIndex:3];
        [enc setBytes:&da length:sizeof da atIndex:4];
        disp(enc, g_expert, (NSUInteger)NU * E);

        /* weighted combine -> g_moe_y */
        uint32_t pc[2] = { (uint32_t)E, (uint32_t)NU };
        [enc setComputePipelineState:g_combine];
        [enc setBuffer:g_moe_down offset:0 atIndex:0];
        [enc setBuffer:g_moe_selw offset:0 atIndex:1];
        [enc setBuffer:g_moe_y offset:0 atIndex:2];
        [enc setBytes:pc length:sizeof pc atIndex:3];
        disp(enc, g_combine, (NSUInteger)E);

        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(moe_y, g_moe_y.contents, (size_t)E * 4);
    }
}

/* ---- GPU attention block (one command buffer; GPU-resident KV cache) -- */

typedef struct { uint64_t w_off; uint32_t n; float eps; } rms_args;
typedef struct { uint64_t w_off; uint32_t base, head_dim, pos; float eps, theta; } qkr_args;
typedef struct { uint32_t n_head, n_head_kv, head_dim, pos, kv_off; float scale; } attn_args;

static id<MTLBuffer> r_cur, r_xn, r_q, r_att, r_tmp, r_Kc, r_Vc;
static int r_E, r_H, r_HK, r_HD, r_nl, r_ms;

static void ensure_attn(int E, int H, int HK, int HD, int nl, int ms) {
    if (r_cur && r_E == E && r_H == H && r_HK == HK && r_HD == HD && r_nl == nl && r_ms == ms) return;
    r_E = E; r_H = H; r_HK = HK; r_HD = HD; r_nl = nl; r_ms = ms;
    int QHD = H * HD; size_t kv = (size_t)nl * ms * HK * HD;
    r_cur = [g_dev newBufferWithLength:(NSUInteger)E   * 4 options:MTLResourceStorageModeShared];
    r_xn  = [g_dev newBufferWithLength:(NSUInteger)E   * 4 options:MTLResourceStorageModeShared];
    r_q   = [g_dev newBufferWithLength:(NSUInteger)QHD * 4 options:MTLResourceStorageModeShared];
    r_att = [g_dev newBufferWithLength:(NSUInteger)QHD * 4 options:MTLResourceStorageModeShared];
    r_tmp = [g_dev newBufferWithLength:(NSUInteger)E   * 4 options:MTLResourceStorageModeShared];
    r_Kc  = [g_dev newBufferWithLength:(NSUInteger)kv  * 4 options:MTLResourceStorageModeShared];
    r_Vc  = [g_dev newBufferWithLength:(NSUInteger)kv  * 4 options:MTLResourceStorageModeShared];
}

void quasar_metal_attn(const quasar_attn_desc *d, float *cur, int pos) {
    if (!g_ok || g_nchunks == 0) return;
    int E = d->E, H = d->H, HK = d->HK, HD = d->HD;
    ensure_attn(E, H, HK, HD, d->n_layer, d->max_seq);
    memcpy(r_cur.contents, cur, (size_t)E * 4);

    uint32_t q_rb = (uint32_t)((uint64_t)E / ggml_blck_size(d->q_type) * ggml_type_size(d->q_type));
    uint32_t k_rb = (uint32_t)((uint64_t)E / ggml_blck_size(d->k_type) * ggml_type_size(d->k_type));
    uint32_t v_rb = (uint32_t)((uint64_t)E / ggml_blck_size(d->v_type) * ggml_type_size(d->v_type));
    uint32_t o_rb = (uint32_t)((uint64_t)(H*HD) / ggml_blck_size(d->o_type) * ggml_type_size(d->o_type));

    int anc = chunk_for(d->attn_norm, (uint64_t)E * 4);
    int qnc = chunk_for(d->q_norm,    (uint64_t)HD * 4);
    int knc = chunk_for(d->k_norm,    (uint64_t)HD * 4);
    int qc  = chunk_for(d->q, (uint64_t)(H*HD)  * q_rb);
    int kc  = chunk_for(d->k, (uint64_t)(HK*HD) * k_rb);
    int vc  = chunk_for(d->v, (uint64_t)(HK*HD) * v_rb);
    int oc  = chunk_for(d->o, (uint64_t)E * o_rb);
    if (anc<0||qnc<0||knc<0||qc<0||kc<0||vc<0||oc<0) { fprintf(stderr, "quasar metal: attn tensor in no chunk\n"); return; }

    uint32_t kv_off = (uint32_t)((uint64_t)d->layer * d->max_seq * HK * HD);
    uint32_t slot   = kv_off + (uint32_t)((uint64_t)pos * HK * HD);
    float esc = 1.0f / sqrtf((float)HD);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        rms_args rg = { d->attn_norm - g_chunk_off[anc], (uint32_t)E, d->eps };
        [enc setComputePipelineState:g_rms];
        [enc setBuffer:r_cur offset:0 atIndex:0]; [enc setBuffer:r_xn offset:0 atIndex:1];
        [enc setBuffer:g_chunk[anc] offset:0 atIndex:2]; [enc setBytes:&rg length:sizeof rg atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        gemv_args qa = { d->q - g_chunk_off[qc], (uint32_t)E, (uint32_t)(H*HD), q_rb, 0, 0 };
        [enc setComputePipelineState:pso_for(d->q_type)];
        [enc setBuffer:g_chunk[qc] offset:0 atIndex:0]; [enc setBuffer:r_xn offset:0 atIndex:1];
        [enc setBuffer:r_q offset:0 atIndex:2]; [enc setBytes:&qa length:sizeof qa atIndex:3];
        disp(enc, pso_for(d->q_type), (NSUInteger)(H*HD));

        gemv_args ka = { d->k - g_chunk_off[kc], (uint32_t)E, (uint32_t)(HK*HD), k_rb, 0, slot };
        [enc setComputePipelineState:pso_for(d->k_type)];
        [enc setBuffer:g_chunk[kc] offset:0 atIndex:0]; [enc setBuffer:r_xn offset:0 atIndex:1];
        [enc setBuffer:r_Kc offset:0 atIndex:2]; [enc setBytes:&ka length:sizeof ka atIndex:3];
        disp(enc, pso_for(d->k_type), (NSUInteger)(HK*HD));

        gemv_args va = { d->v - g_chunk_off[vc], (uint32_t)E, (uint32_t)(HK*HD), v_rb, 0, slot };
        [enc setComputePipelineState:pso_for(d->v_type)];
        [enc setBuffer:g_chunk[vc] offset:0 atIndex:0]; [enc setBuffer:r_xn offset:0 atIndex:1];
        [enc setBuffer:r_Vc offset:0 atIndex:2]; [enc setBytes:&va length:sizeof va atIndex:3];
        disp(enc, pso_for(d->v_type), (NSUInteger)(HK*HD));

        qkr_args qq = { d->q_norm - g_chunk_off[qnc], 0, (uint32_t)HD, (uint32_t)pos, d->eps, d->theta };
        [enc setComputePipelineState:g_qkr];
        [enc setBuffer:r_q offset:0 atIndex:0]; [enc setBuffer:g_chunk[qnc] offset:0 atIndex:1]; [enc setBytes:&qq length:sizeof qq atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)H,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)HD,1,1)];

        qkr_args kq = { d->k_norm - g_chunk_off[knc], slot, (uint32_t)HD, (uint32_t)pos, d->eps, d->theta };
        [enc setComputePipelineState:g_qkr];
        [enc setBuffer:r_Kc offset:0 atIndex:0]; [enc setBuffer:g_chunk[knc] offset:0 atIndex:1]; [enc setBytes:&kq length:sizeof kq atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)HK,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)HD,1,1)];

        attn_args aa = { (uint32_t)H, (uint32_t)HK, (uint32_t)HD, (uint32_t)pos, kv_off, esc };
        [enc setComputePipelineState:g_attn];
        [enc setBuffer:r_q offset:0 atIndex:0]; [enc setBuffer:r_Kc offset:0 atIndex:1];
        [enc setBuffer:r_Vc offset:0 atIndex:2]; [enc setBuffer:r_att offset:0 atIndex:3]; [enc setBytes:&aa length:sizeof aa atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)H,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)HD,1,1)];

        gemv_args oa = { d->o - g_chunk_off[oc], (uint32_t)(H*HD), (uint32_t)E, o_rb, 0, 0 };
        [enc setComputePipelineState:pso_for(d->o_type)];
        [enc setBuffer:g_chunk[oc] offset:0 atIndex:0]; [enc setBuffer:r_att offset:0 atIndex:1];
        [enc setBuffer:r_tmp offset:0 atIndex:2]; [enc setBytes:&oa length:sizeof oa atIndex:3];
        disp(enc, pso_for(d->o_type), (NSUInteger)E);

        [enc setComputePipelineState:g_add];
        [enc setBuffer:r_cur offset:0 atIndex:0]; [enc setBuffer:r_tmp offset:0 atIndex:1];
        disp(enc, g_add, (NSUInteger)E);

        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(cur, r_cur.contents, (size_t)E * 4);
    }
}

/* ---- full GPU-resident forward (one command buffer per token) --------- */

static id<MTLBuffer> r_logits;
static int r_logits_cap;

static void enc_matvec(id<MTLComputeCommandEncoder> enc, uint32_t type, uint64_t row0,
                       uint32_t x_off, uint32_t y_off, int n_in, int n_out,
                       id<MTLBuffer> xb, id<MTLBuffer> yb) {
    id<MTLComputePipelineState> pso = pso_for(type);
    if (!pso) { fprintf(stderr, "qf: bad matvec type %u\n", type); return; }
    uint32_t blck = ggml_blck_size(type); size_t tsz = ggml_type_size(type);
    uint32_t rb = (uint32_t)((uint64_t)n_in / blck * tsz);
    int ci = chunk_for(row0, (uint64_t)n_out * rb);
    if (ci < 0) { fprintf(stderr, "qf: matvec span no chunk\n"); return; }
    gemv_args a = { row0 - g_chunk_off[ci], (uint32_t)n_in, (uint32_t)n_out, rb, x_off, y_off };
    [enc setComputePipelineState:pso];
    [enc setBuffer:g_chunk[ci] offset:0 atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:yb offset:0 atIndex:2];
    [enc setBytes:&a length:sizeof a atIndex:3];
    disp(enc, pso, (NSUInteger)n_out);
}

static void enc_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> out,
                        uint64_t norm_row0, int n, float eps) {
    int ci = chunk_for(norm_row0, (uint64_t)n * 4);
    if (ci < 0) { fprintf(stderr, "qf: rmsnorm no chunk\n"); return; }
    rms_args a = { norm_row0 - g_chunk_off[ci], (uint32_t)n, eps };
    [enc setComputePipelineState:g_rms];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBuffer:g_chunk[ci] offset:0 atIndex:2];
    [enc setBytes:&a length:sizeof a atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}

static void enc_qknorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, uint32_t base,
                       uint64_t norm_row0, int head_dim, int pos, float eps, float theta, int nheads) {
    int ci = chunk_for(norm_row0, (uint64_t)head_dim * 4);
    if (ci < 0) { fprintf(stderr, "qf: qknorm no chunk\n"); return; }
    qkr_args a = { norm_row0 - g_chunk_off[ci], base, (uint32_t)head_dim, (uint32_t)pos, eps, theta };
    [enc setComputePipelineState:g_qkr];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:g_chunk[ci] offset:0 atIndex:1];
    [enc setBytes:&a length:sizeof a atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)nheads,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)head_dim,1,1)];
}

static void enc_add(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> dst, id<MTLBuffer> src, int n) {
    [enc setComputePipelineState:g_add];
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:src offset:0 atIndex:1];
    disp(enc, g_add, (NSUInteger)n);
}

static void enc_expert(id<MTLComputeCommandEncoder> enc, uint64_t base, uint64_t expert_bytes,
                       int n_in, int n_out, uint32_t x_stride, uint32_t out_stride, int nu, int n_expert,
                       id<MTLBuffer> xb, id<MTLBuffer> yb) {
    uint32_t blck = ggml_blck_size(GGML_TYPE_Q2_K); size_t tsz = ggml_type_size(GGML_TYPE_Q2_K);
    uint32_t rb = (uint32_t)((uint64_t)n_in / blck * tsz);
    int ci = chunk_for(base, (uint64_t)n_expert * expert_bytes);
    if (ci < 0) { fprintf(stderr, "qf: expert no chunk\n"); return; }
    expert_args a = { base - g_chunk_off[ci], expert_bytes, (uint32_t)n_in, (uint32_t)n_out, rb, x_stride, out_stride, (uint32_t)nu };
    [enc setComputePipelineState:g_expert];
    [enc setBuffer:g_chunk[ci] offset:0 atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:yb offset:0 atIndex:2];
    [enc setBuffer:g_moe_sel offset:0 atIndex:3];
    [enc setBytes:&a length:sizeof a atIndex:4];
    disp(enc, g_expert, (NSUInteger)nu * n_out);
}

void quasar_metal_forward_token(const quasar_fwd_cfg *cfg, const quasar_layer_desc *layers,
                                int pos, const float *embed, float *logits) {
    if (!g_ok || g_nchunks == 0) return;
    int E = cfg->E, H = cfg->H, HK = cfg->HK, HD = cfg->HD, FF = cfg->FF, NE = cfg->NE, NU = cfg->NU;
    ensure_attn(E, H, HK, HD, cfg->n_layer, cfg->max_seq);
    ensure_moe(E, FF, NE, NU);
    if (logits && r_logits_cap < cfg->n_vocab) {
        r_logits = [g_dev newBufferWithLength:(NSUInteger)cfg->n_vocab * 4 options:MTLResourceStorageModeShared];
        r_logits_cap = cfg->n_vocab;
    }
    memcpy(r_cur.contents, embed, (size_t)E * 4);
    float scale = 1.0f / sqrtf((float)HD);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        for (int il = 0; il < cfg->n_layer; il++) {
            const quasar_layer_desc *L = &layers[il];
            uint32_t kv_off = (uint32_t)((uint64_t)il * cfg->max_seq * HK * HD);
            uint32_t slot   = kv_off + (uint32_t)((uint64_t)pos * HK * HD);

            /* attention */
            enc_rmsnorm(enc, r_cur, r_xn, L->attn_norm, E, cfg->eps);
            enc_matvec(enc, L->q_type, L->q, 0, 0,    E, H  * HD, r_xn, r_q);
            enc_matvec(enc, L->k_type, L->k, 0, slot, E, HK * HD, r_xn, r_Kc);
            enc_matvec(enc, L->v_type, L->v, 0, slot, E, HK * HD, r_xn, r_Vc);
            enc_qknorm(enc, r_q,  0,    L->q_norm, HD, pos, cfg->eps, cfg->theta, H);
            enc_qknorm(enc, r_Kc, slot, L->k_norm, HD, pos, cfg->eps, cfg->theta, HK);
            { attn_args aa = { (uint32_t)H, (uint32_t)HK, (uint32_t)HD, (uint32_t)pos, kv_off, scale };
              [enc setComputePipelineState:g_attn];
              [enc setBuffer:r_q offset:0 atIndex:0]; [enc setBuffer:r_Kc offset:0 atIndex:1];
              [enc setBuffer:r_Vc offset:0 atIndex:2]; [enc setBuffer:r_att offset:0 atIndex:3]; [enc setBytes:&aa length:sizeof aa atIndex:4];
              [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)H,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)HD,1,1)]; }
            enc_matvec(enc, L->o_type, L->o, 0, 0, H * HD, E, r_att, r_tmp);
            enc_add(enc, r_cur, r_tmp, E);

            /* MoE */
            enc_rmsnorm(enc, r_cur, r_xn, L->ffn_norm, E, cfg->eps);
            enc_matvec(enc, L->router_type, L->router, 0, 0, E, NE, r_xn, g_moe_router);
            { uint32_t pk[2] = { (uint32_t)NE, (uint32_t)NU };
              [enc setComputePipelineState:g_topk];
              [enc setBuffer:g_moe_router offset:0 atIndex:0]; [enc setBuffer:g_moe_sel offset:0 atIndex:1];
              [enc setBuffer:g_moe_selw offset:0 atIndex:2]; [enc setBytes:pk length:sizeof pk atIndex:3];
              disp(enc, g_topk, 1); }
            enc_expert(enc, L->gate, L->expert_bytes, E, FF, 0, (uint32_t)FF, NU, NE, r_xn, g_moe_gate);
            enc_expert(enc, L->up,   L->expert_bytes, E, FF, 0, (uint32_t)FF, NU, NE, r_xn, g_moe_up);
            { [enc setComputePipelineState:g_silu];
              [enc setBuffer:g_moe_gate offset:0 atIndex:0]; [enc setBuffer:g_moe_up offset:0 atIndex:1]; [enc setBuffer:g_moe_hid offset:0 atIndex:2];
              disp(enc, g_silu, (NSUInteger)NU * FF); }
            enc_expert(enc, L->down, L->expert_bytes, FF, E, (uint32_t)FF, (uint32_t)E, NU, NE, g_moe_hid, g_moe_down);
            { uint32_t pc[2] = { (uint32_t)E, (uint32_t)NU };
              [enc setComputePipelineState:g_combine];
              [enc setBuffer:g_moe_down offset:0 atIndex:0]; [enc setBuffer:g_moe_selw offset:0 atIndex:1];
              [enc setBuffer:g_moe_y offset:0 atIndex:2]; [enc setBytes:pc length:sizeof pc atIndex:3];
              disp(enc, g_combine, (NSUInteger)E); }
            enc_add(enc, r_cur, g_moe_y, E);
        }

        if (logits) {
            enc_rmsnorm(enc, r_cur, r_xn, cfg->output_norm, E, cfg->eps);
            enc_matvec(enc, cfg->output_type, cfg->output, 0, 0, E, cfg->n_vocab, r_xn, r_logits);
        }
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    if (logits) memcpy(logits, r_logits.contents, (size_t)cfg->n_vocab * 4);
}
