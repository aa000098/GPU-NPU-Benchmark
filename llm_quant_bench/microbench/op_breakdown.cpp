// Per-op microbench for Llama-3.2-1B on Cortex-A76 NEON
// Measures peak throughput in isolation for each op type used per decode token.
// Compile: aarch64-linux-gnu-g++ -O3 -march=armv8.2-a+fp16 -fopenmp op_breakdown.cpp -o op_breakdown

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::high_resolution_clock;

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// ---------- fp16 GEMV (1 × M × K), weight fp16 row-major [M][K]
void gemv_fp16(const __fp16* w, const __fp16* x, __fp16* y, int M, int K) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        float16x8_t acc = vdupq_n_f16(0);
        const __fp16* wr = w + m * K;
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            float16x8_t wv = vld1q_f16(wr + k);
            float16x8_t xv = vld1q_f16(x + k);
            acc = vfmaq_f16(acc, wv, xv);
        }
        float16x4_t lo = vget_low_f16(acc), hi = vget_high_f16(acc);
        float16x4_t s = vadd_f16(lo, hi);
        __fp16 sum = vget_lane_f16(s, 0) + vget_lane_f16(s, 1) + vget_lane_f16(s, 2) + vget_lane_f16(s, 3);
        for (; k < K; k++) sum += wr[k] * x[k];
        y[m] = sum;
    }
}

// ---------- W8 (int8 weight, fp16 act) GEMV
void gemv_w8a16(const int8_t* w, const __fp16* w_scale, const __fp16* x, __fp16* y, int M, int K) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* wr = w + m * K;
        __fp16 s = w_scale[m];
        float16x8_t acc = vdupq_n_f16(0);
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            // Load 8 int8, sign-extend to fp16
            int8x8_t wb = vld1_s8(wr + k);
            int16x8_t w16 = vmovl_s8(wb);
            float16x8_t wv = vmulq_n_f16(vcvtq_f16_s16(w16), s);
            float16x8_t xv = vld1q_f16(x + k);
            acc = vfmaq_f16(acc, wv, xv);
        }
        float16x4_t lo = vget_low_f16(acc), hi = vget_high_f16(acc);
        float16x4_t r = vadd_f16(lo, hi);
        __fp16 sum = vget_lane_f16(r, 0) + vget_lane_f16(r, 1) + vget_lane_f16(r, 2) + vget_lane_f16(r, 3);
        for (; k < K; k++) sum += (__fp16)wr[k] * s * x[k];
        y[m] = sum;
    }
}

// ---------- W4 (int4 weight, fp16 act) GEMV with block-128 scales
// Layout: weight packed 2 nibbles/byte; one fp16 scale per 128-element block per row
void gemv_w4a16_b128(const uint8_t* w, const __fp16* w_scale, const __fp16* x, __fp16* y, int M, int K) {
    constexpr int BLK = 128;
    int nblk = K / BLK;
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const uint8_t* wr = w + m * (K / 2);
        const __fp16* sr = w_scale + m * nblk;
        float total = 0.0f;
        for (int b = 0; b < nblk; b++) {
            float scale = (float)sr[b];
            float16x8_t acc = vdupq_n_f16(0);
            const uint8_t* wb = wr + b * (BLK / 2);
            const __fp16* xb = x + b * BLK;
            for (int j = 0; j + 16 <= BLK; j += 16) {
                uint8x8_t bytes = vld1_u8(wb + j / 2);  // 8 bytes = 16 nibbles
                int8x8_t lo = vreinterpret_s8_u8(vand_u8(bytes, vdup_n_u8(0x0F)));
                int8x8_t hi = vreinterpret_s8_u8(vshr_n_u8(bytes, 4));
                // Center to [-8, 7]
                lo = vsub_s8(lo, vdup_n_s8(8));
                hi = vsub_s8(hi, vdup_n_s8(8));
                int16x8_t lo16 = vmovl_s8(lo);
                int16x8_t hi16 = vmovl_s8(hi);
                float16x8_t lof16 = vcvtq_f16_s16(lo16);
                float16x8_t hif16 = vcvtq_f16_s16(hi16);
                float16x8_t xv0 = vld1q_f16(xb + j);
                float16x8_t xv1 = vld1q_f16(xb + j + 8);
                acc = vfmaq_f16(acc, lof16, xv0);
                acc = vfmaq_f16(acc, hif16, xv1);
            }
            float16x4_t lo = vget_low_f16(acc), hi = vget_high_f16(acc);
            float16x4_t r = vadd_f16(lo, hi);
            float ssum = (float)(vget_lane_f16(r, 0) + vget_lane_f16(r, 1) + vget_lane_f16(r, 2) + vget_lane_f16(r, 3));
            total += scale * ssum;
        }
        y[m] = (__fp16)total;
    }
}

// ---------- RMSNorm (1 × D)
void rmsnorm(const __fp16* x, const __fp16* w, __fp16* y, int D, float eps = 1e-5f) {
    float sq = 0.0f;
    for (int i = 0; i < D; i++) sq += (float)x[i] * (float)x[i];
    float rms = 1.0f / sqrtf(sq / D + eps);
    int i = 0;
    float16x8_t r = vdupq_n_f16((__fp16)rms);
    for (; i + 8 <= D; i += 8) {
        float16x8_t xv = vld1q_f16(x + i);
        float16x8_t wv = vld1q_f16(w + i);
        vst1q_f16(y + i, vmulq_f16(vmulq_f16(xv, r), wv));
    }
    for (; i < D; i++) y[i] = x[i] * rms * w[i];
}

// ---------- RoPE (1 × D, paired complex rotation)
void rope(__fp16* x, const __fp16* cos_v, const __fp16* sin_v, int D) {
    int half = D / 2;
    for (int i = 0; i < half; i++) {
        __fp16 a = x[i], b = x[i + half];
        __fp16 c = cos_v[i], s = sin_v[i];
        x[i]        = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

// ---------- SiLU activation (in-place)
void silu(__fp16* x, int D) {
    for (int i = 0; i < D; i++) {
        float v = (float)x[i];
        x[i] = (__fp16)(v / (1.0f + expf(-v)));
    }
}

// ---------- Attention (decode, 1 query × N cached tokens), fp16 KV
// Q: [n_q × d], K_cache: [n_kv × N × d], V_cache: [n_kv × N × d]
void attention_fp16(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* out,
                    int n_q, int n_kv, int N, int d) {
    int group = n_q / n_kv;
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> scores(N);
    for (int qh = 0; qh < n_q; qh++) {
        int kvh = qh / group;
        const __fp16* q = Q + qh * d;
        const __fp16* Kh = K + kvh * N * d;
        const __fp16* Vh = V + kvh * N * d;
        // dot per token
        for (int t = 0; t < N; t++) {
            float16x8_t acc = vdupq_n_f16(0);
            int i = 0;
            for (; i + 8 <= d; i += 8) {
                acc = vfmaq_f16(acc, vld1q_f16(q + i), vld1q_f16(Kh + t * d + i));
            }
            float16x4_t lo = vget_low_f16(acc), hi = vget_high_f16(acc);
            float16x4_t r = vadd_f16(lo, hi);
            float s = (float)(vget_lane_f16(r, 0) + vget_lane_f16(r, 1) + vget_lane_f16(r, 2) + vget_lane_f16(r, 3));
            scores[t] = s * scale;
        }
        // softmax
        float maxv = -INFINITY;
        for (int t = 0; t < N; t++) if (scores[t] > maxv) maxv = scores[t];
        float sum = 0.0f;
        for (int t = 0; t < N; t++) { scores[t] = expf(scores[t] - maxv); sum += scores[t]; }
        float inv = 1.0f / sum;
        // weighted V sum
        float16x8_t out_acc[16];
        for (int i = 0; i < d / 8; i++) out_acc[i] = vdupq_n_f16(0);
        for (int t = 0; t < N; t++) {
            __fp16 w = (__fp16)(scores[t] * inv);
            for (int i = 0; i < d / 8; i++) {
                out_acc[i] = vfmaq_n_f16(out_acc[i], vld1q_f16(Vh + t * d + i * 8), w);
            }
        }
        for (int i = 0; i < d / 8; i++) vst1q_f16(out + qh * d + i * 8, out_acc[i]);
    }
}

template <typename Fn>
double bench(Fn fn, int iters) {
    fn();  // warm up
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) fn();
    return (now_ms() - t0) / iters;
}

void print_op(const char* name, double ms, double bytes_per_call, double flops_per_call) {
    double bw = bytes_per_call / (ms * 1e6);  // GB/s
    double gf = flops_per_call / (ms * 1e6);  // GFLOPS
    printf("  %-30s  %8.3f ms   %7.2f GB/s   %7.2f GFLOPS\n", name, ms, bw, gf);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 50;
    int ctx_for_attention = (argc > 2) ? atoi(argv[2]) : 1024;
    printf("Per-op microbench, iters=%d, attn ctx=%d\n", iters, ctx_for_attention);
    printf("Hardware: RK3588 4× Cortex-A76 + NEON fp16 (taskset -c 4-7 recommended)\n\n");

    // Llama-3.2-1B dims
    constexpr int H = 2048;
    constexpr int IM = 8192;
    constexpr int N_HEAD = 32;
    constexpr int N_KV = 8;
    constexpr int HEAD_DIM = 64;
    constexpr int VOCAB = 128256;
    constexpr int N_LAYER = 16;
    constexpr int QKV_OUT = N_HEAD * HEAD_DIM + 2 * N_KV * HEAD_DIM;  // 2048 + 1024 = 3072

    std::mt19937 rng(42);
    auto fill = [&](__fp16* p, int n) {
        std::uniform_real_distribution<float> d(-0.1f, 0.1f);
        for (int i = 0; i < n; i++) p[i] = (__fp16)d(rng);
    };

    // Allocate buffers (largest sized for FFN gate/up: 8192 × 2048)
    std::vector<__fp16> w_qkv(QKV_OUT * H), w_o(H * H), w_gate(IM * H), w_up(IM * H), w_down(H * IM);
    std::vector<__fp16> w_lmhead(VOCAB * H);
    std::vector<__fp16> w_norm(H);
    std::vector<__fp16> x(H), x_im(IM), y(VOCAB);
    std::vector<__fp16> q(N_HEAD * HEAD_DIM), k_cache(N_KV * ctx_for_attention * HEAD_DIM), v_cache(N_KV * ctx_for_attention * HEAD_DIM);
    std::vector<__fp16> attn_out(N_HEAD * HEAD_DIM);
    std::vector<__fp16> cos_v(HEAD_DIM / 2), sin_v(HEAD_DIM / 2);
    fill(w_qkv.data(), w_qkv.size()); fill(w_o.data(), w_o.size());
    fill(w_gate.data(), w_gate.size()); fill(w_up.data(), w_up.size());
    fill(w_down.data(), w_down.size()); fill(w_lmhead.data(), w_lmhead.size());
    fill(w_norm.data(), w_norm.size()); fill(x.data(), x.size()); fill(x_im.data(), x_im.size());
    fill(q.data(), q.size()); fill(k_cache.data(), k_cache.size()); fill(v_cache.data(), v_cache.size());
    fill(cos_v.data(), cos_v.size()); fill(sin_v.data(), sin_v.size());

    // === fp16 baseline ===
    printf("--- Per-op fp16 baseline ---\n");
    {
        double ms = bench([&] { gemv_fp16(w_qkv.data(), x.data(), x_im.data(), QKV_OUT, H); }, iters);
        print_op("QKV proj (3072×2048)", ms, (double)QKV_OUT * H * 2, (double)2 * QKV_OUT * H);
    }
    {
        double ms = bench([&] { gemv_fp16(w_o.data(), x.data(), x.data(), H, H); }, iters);
        print_op("O proj (2048×2048)", ms, (double)H * H * 2, (double)2 * H * H);
    }
    {
        double ms = bench([&] { gemv_fp16(w_gate.data(), x.data(), x_im.data(), IM, H); }, iters);
        print_op("FFN gate (8192×2048)", ms, (double)IM * H * 2, (double)2 * IM * H);
    }
    {
        double ms = bench([&] { gemv_fp16(w_up.data(), x.data(), x_im.data(), IM, H); }, iters);
        print_op("FFN up (8192×2048)", ms, (double)IM * H * 2, (double)2 * IM * H);
    }
    {
        double ms = bench([&] { gemv_fp16(w_down.data(), x_im.data(), x.data(), H, IM); }, iters);
        print_op("FFN down (2048×8192)", ms, (double)H * IM * 2, (double)2 * H * IM);
    }
    {
        double ms = bench([&] { gemv_fp16(w_lmhead.data(), x.data(), y.data(), VOCAB, H); }, iters);
        print_op("LM head (128256×2048)", ms, (double)VOCAB * H * 2, (double)2 * VOCAB * H);
    }
    {
        double ms = bench([&] { rmsnorm(x.data(), w_norm.data(), x.data(), H); }, iters * 10);
        print_op("RMSNorm (2048)", ms, (double)H * 2 * 3, (double)H * 3);
    }
    {
        double ms = bench([&] { rope(q.data(), cos_v.data(), sin_v.data(), HEAD_DIM); }, iters * 100);
        print_op("RoPE per head (64)", ms, (double)HEAD_DIM * 2 * 3, (double)HEAD_DIM * 4);
    }
    {
        double ms = bench([&] { silu(x_im.data(), IM); }, iters * 10);
        print_op("SiLU (8192)", ms, (double)IM * 2 * 2, (double)IM * 5);
    }
    {
        double ms = bench([&] {
            attention_fp16(q.data(), k_cache.data(), v_cache.data(), attn_out.data(),
                           N_HEAD, N_KV, ctx_for_attention, HEAD_DIM);
        }, iters);
        size_t kv_bytes = (size_t)2 * N_KV * ctx_for_attention * HEAD_DIM * 2;
        size_t flops = (size_t)2 * N_HEAD * ctx_for_attention * HEAD_DIM * 2;  // QK + softmax*V
        char name[64]; snprintf(name, 64, "Attention fp16 (ctx=%d)", ctx_for_attention);
        print_op(name, ms, kv_bytes, flops);
    }

    // === W8A16 weight ===
    printf("\n--- Per-op W8A16 (int8 weight + fp16 act) ---\n");
    std::vector<int8_t> w_qkv_i8(QKV_OUT * H), w_o_i8(H * H), w_gate_i8(IM * H), w_down_i8(H * IM), w_lmhead_i8(VOCAB * H);
    std::vector<__fp16> sc_qkv(QKV_OUT, (__fp16)0.01f), sc_o(H, (__fp16)0.01f), sc_gate(IM, (__fp16)0.01f), sc_down(H, (__fp16)0.01f), sc_lmhead(VOCAB, (__fp16)0.01f);
    for (size_t i = 0; i < w_qkv_i8.size(); i++) w_qkv_i8[i] = (int8_t)(rng() % 256 - 128);
    for (size_t i = 0; i < w_o_i8.size(); i++) w_o_i8[i] = (int8_t)(rng() % 256 - 128);
    for (size_t i = 0; i < w_gate_i8.size(); i++) w_gate_i8[i] = (int8_t)(rng() % 256 - 128);
    for (size_t i = 0; i < w_down_i8.size(); i++) w_down_i8[i] = (int8_t)(rng() % 256 - 128);
    for (size_t i = 0; i < w_lmhead_i8.size(); i++) w_lmhead_i8[i] = (int8_t)(rng() % 256 - 128);

    {
        double ms = bench([&] { gemv_w8a16(w_qkv_i8.data(), sc_qkv.data(), x.data(), x_im.data(), QKV_OUT, H); }, iters);
        print_op("QKV proj W8 (3072×2048)", ms, (double)QKV_OUT * H, (double)2 * QKV_OUT * H);
    }
    {
        double ms = bench([&] { gemv_w8a16(w_gate_i8.data(), sc_gate.data(), x.data(), x_im.data(), IM, H); }, iters);
        print_op("FFN gate W8 (8192×2048)", ms, (double)IM * H, (double)2 * IM * H);
    }
    {
        double ms = bench([&] { gemv_w8a16(w_down_i8.data(), sc_down.data(), x_im.data(), x.data(), H, IM); }, iters);
        print_op("FFN down W8 (2048×8192)", ms, (double)H * IM, (double)2 * H * IM);
    }
    {
        double ms = bench([&] { gemv_w8a16(w_lmhead_i8.data(), sc_lmhead.data(), x.data(), y.data(), VOCAB, H); }, iters);
        print_op("LM head W8 (128256×2048)", ms, (double)VOCAB * H, (double)2 * VOCAB * H);
    }

    // === W4A16 block-128 ===
    printf("\n--- Per-op W4A16 b128 (4-bit weight + fp16 act) ---\n");
    std::vector<uint8_t> w_qkv_q4(QKV_OUT * H / 2), w_gate_q4(IM * H / 2), w_down_q4(H * IM / 2), w_lmhead_q4(VOCAB * H / 2);
    int nblk_qkv = H / 128, nblk_gate = H / 128, nblk_down = IM / 128;
    std::vector<__fp16> sc_qkv_q4(QKV_OUT * nblk_qkv, (__fp16)0.01f);
    std::vector<__fp16> sc_gate_q4(IM * nblk_gate, (__fp16)0.01f);
    std::vector<__fp16> sc_down_q4(H * nblk_down, (__fp16)0.01f);
    std::vector<__fp16> sc_lmhead_q4(VOCAB * nblk_qkv, (__fp16)0.01f);
    for (size_t i = 0; i < w_qkv_q4.size(); i++) w_qkv_q4[i] = rng() & 0xFF;
    for (size_t i = 0; i < w_gate_q4.size(); i++) w_gate_q4[i] = rng() & 0xFF;
    for (size_t i = 0; i < w_down_q4.size(); i++) w_down_q4[i] = rng() & 0xFF;
    for (size_t i = 0; i < w_lmhead_q4.size(); i++) w_lmhead_q4[i] = rng() & 0xFF;

    {
        double ms = bench([&] { gemv_w4a16_b128(w_qkv_q4.data(), sc_qkv_q4.data(), x.data(), x_im.data(), QKV_OUT, H); }, iters);
        print_op("QKV proj W4 b128 (3072×2048)", ms, (double)QKV_OUT * H / 2, (double)2 * QKV_OUT * H);
    }
    {
        double ms = bench([&] { gemv_w4a16_b128(w_gate_q4.data(), sc_gate_q4.data(), x.data(), x_im.data(), IM, H); }, iters);
        print_op("FFN gate W4 b128 (8192×2048)", ms, (double)IM * H / 2, (double)2 * IM * H);
    }
    {
        double ms = bench([&] { gemv_w4a16_b128(w_down_q4.data(), sc_down_q4.data(), x_im.data(), x.data(), H, IM); }, iters);
        print_op("FFN down W4 b128 (2048×8192)", ms, (double)H * IM / 2, (double)2 * H * IM);
    }
    {
        double ms = bench([&] { gemv_w4a16_b128(w_lmhead_q4.data(), sc_lmhead_q4.data(), x.data(), y.data(), VOCAB, H); }, iters);
        print_op("LM head W4 b128 (128256×2048)", ms, (double)VOCAB * H / 2, (double)2 * VOCAB * H);
    }

    return 0;
}
