// q51_dequant_proof.cpp
//
// Numerical proof for the q5_1 support in ggml/src/ggml-cuda/fattn-q4p.cuh (L1). The kernel
// dequantizes a q5_1 value x*d + m (x: 0..31, d and m fp16) in two ways:
//
//   A: fp32 path (K side, V side fp32):  fmaf((float)d, (float)x, (float)m)
//      one correctly rounded operation.
//   B: fp16 H16PV path (V side, DPT 4/8): fl16(fmaf((float)d, (float)x, (float)m))
//      one exact fp32 fma, then a single round to nearest fp16, the same scheme as the f16
//      conversion kernels. (An all-fp16 scheme fl16(fl16(x*d) + m) was rejected: when x*d and
//      m partially cancel, the product's 2^-11 relative error survives - a run of the old scheme
//      measured max rel err 1024 vs the 4.9e-4 bound.)
//
// The CPU reference (dequantize_row_q5_1) computes fl32((float)x * (float)d) + m: a rounded
// product plus a rounded add. This program measures A and B against the reference and against
// the exact value (in double) over an exhaustive product of the relevant domains plus a large
// random sample. Host-only; no GPU.
//
// Build: g++ -O2 -std=c++17 -ffp-contract=off q51_dequant_proof.cpp -o q51_dequant_proof
// Run:   ./q51_dequant_proof [iterations for the random sample, default 300000000]
//
// Expected (recorded in the README):
//   A: always equal to the exactly rounded value (fma is correctly rounded); differs from the
//      2-rounding reference only where the reference's own product rounding shows (the x*d
//      product is always exact in fp32: 6 + 11 mantissa bits, so only the final add can differ).
//   B: for normal results (|exact| >= 2^-14, the smallest fp16 normal) the max rel err vs exact
//      is ~ 2^-11 (one fp16 RN); below that the relative error is unbounded but the absolute
//      error is at most 2^-25 (half an fp16 subnormal ulp) - inherent to fp16, same as the f16
//      cache conversion. B equals fl16(ref) except when the reference's product rounding shifts
//      the value across an fp16 rounding boundary (rare, counted); B = inf only when
//      |x*d + m| > 65504.
//   edges: d = 0 gives exactly m in both paths; x = 0 gives exactly m.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

static uint16_t f2h(float x) { // float -> half, round to nearest even
    uint32_t fx;
    std::memcpy(&fx, &x, 4);
    const uint32_t sign = (fx >> 16) & 0x8000u;
    const uint32_t exp = (fx >> 23) & 0xFFu, man0 = fx & 0x7FFFFFu;
    int e10 = (int) exp - 127 + 15;
    if (e10 <= 0) {
        if (e10 < -10) {
            return (uint16_t) sign; // underflow to zero
        }
        uint32_t man = man0 | 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e10);
        const uint32_t half = 1u << (shift - 1);
        const uint32_t rem = man & ((1u << shift) - 1);
        uint32_t m11 = man >> shift;
        if (rem > half || (rem == half && (m11 & 1))) {
            m11++;
        }
        return (uint16_t) (sign | m11);
    }
    if (e10 >= 31) {
        return (uint16_t) (sign | 0x7C00u); // overflow to infinity
    }
    uint32_t man = man0;
    uint32_t m11 = man >> 13;
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (m11 & 1))) {
        m11++;
        if (m11 == 0x400u) {
            m11 = 0;
            e10++;
        }
    }
    return (uint16_t) (sign | (e10 << 10) | m11);
}

static float h2f(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t e10 = (h >> 10) & 0x1Fu, m11 = h & 0x3FFu;
    uint32_t fx;
    if (e10 == 0) {
        // fp16 subnormal m11 * 2^-24: a normal fp32, exact
        if (m11 == 0) {
            return sign ? -0.0f : 0.0f;
        }
        const float v = (float) ((double) m11 * 0x1p-24);
        return sign ? -v : v;
    }
    if (e10 == 31) {
        fx = sign | 0x7F800000u | (m11 << 13);
    } else {
        fx = sign | (e10 + 127 - 15) << 23 | (m11 << 13);
    }
    float x;
    std::memcpy(&x, &fx, 4);
    return x;
}

static double ulp_rel(float x) { // relative size of one ulp of x
    const double a = std::abs((double) x);
    if (a == 0.0 || std::isinf(a) || std::isnan(a)) {
        return 0.0;
    }
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const double e = (double) ((u >> 23) & 0xFFu) - 127.0;
    return std::ldexp(1.0, (int) e - 23);
}

int main(int argc, char ** argv) {
    const long nsamp = argc > 1 ? atol(argv[1]) : 300000000L;

    // all finite fp16 bit patterns, as floats
    static float hv[65536];
    for (int i = 0; i < 65536; ++i) {
        hv[i] = h2f((uint16_t) i);
    }

    // ---- path A: fmaf(d, x, m) vs reference vs exact ----
    double a_max_ulp_ref = 0.0, a_max_rel_exact = 0.0, a_max_abs = 0.0;
    long a_ne_ref = 0, a_ne_exact = 0, a_ref_finite_a_inf = 0, nA = 0;
    {
        // exhaustive x * d over an m grid (m is a single addend: the rounding interaction is
        // the same for any m, the grid covers the scales and both signs)
        static const float mg[16] = {0.0f, 1e-7f, 6.103515625e-5f, 0.001f, 0.01f, 0.1f, 1.0f, 8.0f,
                                     1024.0f, 65504.0f, -1e-7f, -6.103515625e-5f, -0.1f, -1.0f,
                                     -1024.0f, -65504.0f};
        for (int x = 0; x < 32; ++x) {
            for (int di = 0; di < 65536; ++di) {
                const float d = hv[di];
                if (!std::isfinite(d)) {
                    continue;
                }
                const double dex = (double) x * (double) d;
                for (int mi = 0; mi < 16; ++mi) {
                    const float m = mg[mi];
                    const float A = fmaf(d, (float) x, m);
                    float prod = (float) x * d; // separate rounding, as the CPU does
                    const float ref = prod + m;
                    const double exact = dex + (double) m;
                    ++nA;
                    if (A != ref) {
                        ++a_ne_ref;
                        const double ulp = ulp_rel(ref);
                        const double rel = ulp > 0.0 ? std::abs((double) A - (double) ref) / ulp : 0.0;
                        if (rel > a_max_ulp_ref) {
                            a_max_ulp_ref = rel;
                        }
                    }
                    if (std::isfinite(exact) && !std::isfinite(A)) {
                        ++a_ref_finite_a_inf;
                    }
                    if (std::isfinite(A) && std::isfinite(exact) && A != (float) exact) {
                        ++a_ne_exact;
                        const double rel = std::abs((double) A - exact) / (std::abs(exact) + 1e-300);
                        if (rel > a_max_rel_exact) {
                            a_max_rel_exact = rel;
                        }
                    }
                    if (std::isfinite(A) && std::isfinite(ref)) {
                        const double diff = std::abs((double) A - (double) ref);
                        if (diff > a_max_abs) {
                            a_max_abs = diff;
                        }
                    }
                }
            }
        }
        // edges: d = 0 and x = 0 must give exactly m
        for (int mi = 0; mi < 65536; ++mi) {
            const float m = hv[mi];
            if (!std::isfinite(m)) {
                continue;
            }
            if (fmaf(0.0f, 31.0f, m) != m) {
                printf("FAIL: A d=0 x=31 m=%.8g -> %.8g\n", (double) m, (double) fmaf(0.0f, 31.0f, m));
                return 1;
            }
            if (fmaf(1234.5f, 0.0f, m) != m) {
                printf("FAIL: A x=0 m=%.8g -> %.8g\n", (double) m, (double) fmaf(1234.5f, 0.0f, m));
                return 1;
            }
        }
        // random sample
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> ux(0, 31), ui(0, 65535);
        for (long i = 0; i < nsamp; ++i) {
            const int x = ux(rng);
            const float d = hv[ui(rng)], m = hv[ui(rng)];
            const float A = fmaf(d, (float) x, m);
            float prod = (float) x * d;
            const float ref = prod + m;
            const double exact = (double) x * (double) d + (double) m;
            ++nA;
            if (A != ref) {
                ++a_ne_ref;
                const double ulp = ulp_rel(ref);
                const double rel = ulp > 0.0 ? std::abs((double) A - (double) ref) / ulp : 0.0;
                if (rel > a_max_ulp_ref) {
                    a_max_ulp_ref = rel;
                }
            }
            if (std::isfinite(exact) && !std::isfinite(A)) {
                ++a_ref_finite_a_inf;
            }
            if (std::isfinite(A) && std::isfinite(exact) && A != (float) exact) {
                ++a_ne_exact;
                const double rel = std::abs((double) A - exact) / (std::abs(exact) + 1e-300);
                if (rel > a_max_rel_exact) {
                    a_max_rel_exact = rel;
                }
            }
        }
    }
    printf("path A (fp32 fmaf): n = %ld\n", nA);
    printf("  A != ref: %ld (%.4g%%), max %f ulp of ref, max abs diff %g\n",
           a_ne_ref, 100.0 * (double) a_ne_ref / (double) nA, a_max_ulp_ref, (double) a_max_abs);
    printf("  A != exact-rounded: %ld, max rel err vs exact %g (1 ulp = %g), ref finite but A inf: %ld\n",
           a_ne_exact, a_max_rel_exact, 1e-7, a_ref_finite_a_inf);

    // ---- path B: fl16(fmaf(d, x, m)) vs reference vs exact ----
    const double HMIN = 0x1p-14; // smallest fp16 normal
    double b_max_rel_normal = 0.0, b_max_abs_sub = 0.0;
    long b_ne_fl16_ref = 0, b_ref_finite_b_inf = 0, nB = 0;
    {
        // exhaustive x * d over an m grid
        static const float mg[16] = {0.0f, 1e-7f, 6.103515625e-5f, 0.001f, 0.01f, 0.1f, 1.0f, 8.0f,
                                     1024.0f, 65504.0f, -1e-7f, -6.103515625e-5f, -0.1f, -1.0f,
                                     -1024.0f, -65504.0f};
        for (int x = 0; x < 32; ++x) {
            for (int di = 0; di < 65536; ++di) {
                const float d = hv[di];
                if (!std::isfinite(d)) {
                    continue;
                }
                for (int mi = 0; mi < 16; ++mi) {
                    const float m = mg[mi];
                    const float B = h2f(f2h(fmaf(d, (float) x, m)));
                    float prod = (float) x * d;
                    const float ref = prod + m;
                    const double exact = (double) x * (double) d + (double) m;
                    ++nB;
                    if (B != h2f(f2h(ref))) {
                        ++b_ne_fl16_ref;
                    }
                    if (std::isfinite(exact) && std::isinf(B)) {
                        ++b_ref_finite_b_inf;
                    }
                    if (std::isfinite(B) && std::isfinite(exact)) {
                        if (std::abs(exact) >= HMIN) {
                            const double rel = std::abs((double) B - exact) / std::abs(exact);
                            if (rel > b_max_rel_normal) {
                                b_max_rel_normal = rel;
                            }
                        } else {
                            const double abs = std::abs((double) B - exact);
                            if (abs > b_max_abs_sub) {
                                b_max_abs_sub = abs;
                            }
                        }
                    }
                }
            }
        }
        // edges: d = 0 and x = 0 must give exactly m (m is fp16, so fl16(m) = m)
        for (int mi = 0; mi < 65536; ++mi) {
            const float m = hv[mi];
            if (!std::isfinite(m)) {
                continue;
            }
            if (h2f(f2h(fmaf(0.0f, 31.0f, m))) != m) {
                printf("FAIL: B d=0 x=31 m=%.8g -> %g\n", (double) m, (double) h2f(f2h(fmaf(0.0f, 31.0f, m))));
                return 1;
            }
            if (h2f(f2h(fmaf(1234.5f, 0.0f, m))) != m) {
                printf("FAIL: B x=0 m=%.8g -> %g\n", (double) m, (double) h2f(f2h(fmaf(1234.5f, 0.0f, m))));
                return 1;
            }
        }
        // random sample of full (x, d, m) triples
        std::mt19937 rng(98765);
        std::uniform_int_distribution<int> ux(0, 31), ui(0, 65535);
        for (long i = 0; i < nsamp; ++i) {
            const int x = ux(rng);
            const float d = hv[ui(rng)], m = hv[ui(rng)];
            const float B = h2f(f2h(fmaf(d, (float) x, m)));
            float prod = (float) x * d;
            const float ref = prod + m;
            const double exact = (double) x * (double) d + (double) m;
            ++nB;
            if (B != h2f(f2h(ref))) {
                ++b_ne_fl16_ref;
            }
            if (std::isfinite(exact) && std::isinf(B)) {
                ++b_ref_finite_b_inf;
            }
            if (std::isfinite(B) && std::isfinite(exact)) {
                if (std::abs(exact) >= HMIN) {
                    const double rel = std::abs((double) B - exact) / std::abs(exact);
                    if (rel > b_max_rel_normal) {
                        b_max_rel_normal = rel;
                    }
                } else {
                    const double abs = std::abs((double) B - exact);
                    if (abs > b_max_abs_sub) {
                        b_max_abs_sub = abs;
                    }
                }
            }
        }
    }
    printf("path B (fp32 fma, then single round to fp16): n = %ld\n", nB);
    printf("  B != fl16(ref): %ld (%.4g%%) - ref product rounding across an fp16 boundary\n",
           b_ne_fl16_ref, 100.0 * (double) b_ne_fl16_ref / (double) nB);
    printf("  max rel err vs exact, |exact| >= 2^-14: %g (bound ~%g)\n", b_max_rel_normal, 1.0 / 2048.0);
    printf("  max abs err vs exact, |exact| < 2^-14: %g (bound %g = half a subnormal ulp)\n",
           b_max_abs_sub, 0.5 * 0x1p-24);
    printf("  ref finite but B inf: %ld (%.3g%% of n, full-range random d/m)\n",
           b_ref_finite_b_inf, 100.0 * (double) b_ref_finite_b_inf / (double) nB);

    printf("OK\n");
    return 0;
}
