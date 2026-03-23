/*
    ZArray is a C++ class (v1.3) that allows SIMD vector arrays to be indexed by
    integer equivalent in vector size vectors such as int8, int4, int2
    
    LICENSE: FREE for commercial and non-commercial use,
    it is an ENJOYWARE and hope it speeds up your project for you. Credit mention
    always welcome, as usual.
    
    NEW (v1.12: Can it be further optimized? YES it can be! and this is the living proof.
    
    Example usage: 

    ZArray<double4v> arr[1024], aVec = 0.0;
    for (int i=0; i<1024; i++) {
        arr[i] = double4{ i, -i,  M_PI, -M_PI };
    }
    int4v idx = { 0, 1, 100, 500 };
    
    aVec = arr[idx];  
    
    // aVec now contains: { 0.0, -1.0, 3.1415, -3.1415 }

    // lets change indexes to 10, 20, 30 and 40
    
    idx = int4{ 10, 20, 30, 40 };
    
    arr[idx] = aVec; 

    // now the arr contains:
    // arr[10][0] = 0.0; arr[20][1] = -1.0;  
    // arr[30][2] = 3.1415; arr[40][3] = -3.1415

    well you get the idea.
 */


#pragma once

#include <simd/simd.h>

typedef simd_float2 float2v;
typedef simd_float4 float4v;
typedef simd_float8 float8v;

typedef simd_double2 double2v;
typedef simd_double4 double4v;
typedef simd_double8 double8v;

typedef simd_long8 long8v;
typedef simd_long4 long4v;
typedef simd_long2 long2v;

typedef simd_int2 int2v;
typedef simd_int4 int4v;
typedef simd_int8 int8v;

template<typename T>
concept IsVector = requires { T{}[0]; };

static inline constexpr int alignFor(int typeSize, int nelem) {
   // if (typeSize > 2)
     //   return 16;
    return typeSize * nelem;
}

template<typename ZZ>
using SimdBase = decltype([] {
    using std::is_same_v;

    if constexpr(
        is_same_v<ZZ, int> ||
        is_same_v<ZZ, float> ||
        is_same_v<ZZ, double>
    )
        return ZZ{};
    else
        return ZZ{}[0] + 0;
}());

template<typename ZZ> struct SimdInfo {
    using Base = SimdBase<ZZ>;
    static constexpr int size = sizeof(ZZ) / sizeof(Base);
};

template <class ZZ>
inline constexpr int SimdSize = SimdInfo<ZZ>::size;

template<typename Z, int size>
using Simd = Z __attribute__((ext_vector_type(size),aligned(alignFor(sizeof(Z),size))));

template<typename ZZ, typename NewBase>
using SimdSame = Simd<NewBase, SimdInfo<ZZ>::size>;

template<typename ZZ, typename NewBase>
using SimdSameHalf = Simd<NewBase, SimdInfo<ZZ>::size/2>;

#define NOT_VECTOR(Z) (std::is_same_v<Z, float> || std::is_same_v<Z, double>)
#define IS_VECTOR(Z)  (IsVector<Z>)


#ifndef MALLOC_ALIGN
#define MALLOC_ALIGN 128
#endif

// ── ZArray ───────────────────────────────────────────────────────────
template<class T, size_t N, typename idx_t = int>
struct ZArray
{
    using IT   = SimdSame<T, idx_t>;
    using Base = SimdBase<T>;
    static constexpr int SIZE = SimdSize<T>;

    alignas(MALLOC_ALIGN) T dd[N];

    ZArray() { reset(); }
    void reset() { std::memset(dd, 0, sizeof(dd)); }

    // ─────────────────────────────────────────────────────────────
    //  Scalar access — arr[i] returns T& for plain integer index.
    //  Makes  arr[i] = double4{...}  work naturally.
    // ─────────────────────────────────────────────────────────────
    T&       operator[](size_t i)       { return dd[i]; }
    const T& operator[](size_t i) const { return dd[i]; }

    // ─────────────────────────────────────────────────────────────
    //  Core gather/scatter — generic, works for ALL SIMD types.
    //  Compiles to LDR (scalar) + INS (lane insert) per element.
    //  This is the optimal sequence on AArch64 NEON: LDR folds the
    //  lane byte-offset into its immediate, which LD1-lane cannot.
    // ─────────────────────────────────────────────────────────────
private:
    template<size_t... K>
    [[gnu::always_inline]]
    static T gather_impl(const T* dd, const IT& ii, std::index_sequence<K...>) {
        if constexpr (SIZE == 1)
            return dd[ii[0]];
        else
            return T{ dd[ii[K]][K]... };
    }

    template<size_t... K>
    [[gnu::always_inline]]
    static void scatter_impl(T* dd, const IT& ii, const T& v, std::index_sequence<K...>) {
        if constexpr (SIZE == 1)
            dd[ii[0]] = v;
        else
            ((dd[ii[K]][K] = v[K]), ...);
    }

    template<size_t... K>
    void prefetch_impl(const IT& ii, int rw, std::index_sequence<K...>) const {
        ((__builtin_prefetch(&dd[ii[K]], rw, 3)), ...);
    }

    using Seq = std::make_index_sequence<SIZE>;

public:
    // ── Prefetch NEXT iteration's addresses (where it helps) ─────
    void prefetch_read (const IT& ii) const { prefetch_impl(ii, 0, Seq{}); }
    void prefetch_write(const IT& ii) const { prefetch_impl(ii, 1, Seq{}); }

    // ── Explicit gather/scatter ──────────────────────────────────
    [[gnu::hot, gnu::always_inline]]
    T gather(const IT& ii) const { return gather_impl(dd, ii, Seq{}); }

    [[gnu::hot, gnu::always_inline]]
    void scatter(const IT& ii, const T& v) { scatter_impl(dd, ii, v, Seq{}); }

#if defined(__aarch64__) && defined(__ARM_NEON)
    // ─────────────────────────────────────────────────────────────
    //  TBL fast path for contiguous indices: {b, b+1, …, b+SIZE-1}
    //
    //  4× sequential LDR Q  +  1× TBL4 = 5 instructions
    //  vs generic:  4× LDR S  +  3× INS = 7 instructions
    //  Plus: sequential loads hit 1-2 cache lines, not 4 random.
    // ─────────────────────────────────────────────────────────────
    [[gnu::hot, gnu::always_inline]]
    T gather_contiguous(idx_t base) const {
        static_assert(SIZE >= 2);

        if constexpr (SIZE == 4 && sizeof(Base) == 4) {
            static const uint8x16_t diag_idx = {
                 0,  1,  2,  3,    // lane 0 from dd[base+0]
                20, 21, 22, 23,    // lane 1 from dd[base+1]
                40, 41, 42, 43,    // lane 2 from dd[base+2]
                60, 61, 62, 63     // lane 3 from dd[base+3]
            };
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&dd[base]);
            uint8x16x4_t table = {
                vld1q_u8(p), vld1q_u8(p + 16),
                vld1q_u8(p + 32), vld1q_u8(p + 48)
            };
            uint8x16_t result = vqtbl4q_u8(table, diag_idx);
            T out;
            __builtin_memcpy(&out, &result, sizeof(T));
            return out;
        }
        else if constexpr (SIZE == 2 && sizeof(Base) == 8) {
            static const uint8x16_t diag_idx = {
                 0,  1,  2,  3,  4,  5,  6,  7,    // lane 0 from dd[base+0]
                24, 25, 26, 27, 28, 29, 30, 31     // lane 1 from dd[base+1]
            };
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&dd[base]);
            uint8x16x2_t table = { vld1q_u8(p), vld1q_u8(p + 16) };
            uint8x16_t result = vqtbl2q_u8(table, diag_idx);
            T out;
            __builtin_memcpy(&out, &result, sizeof(T));
            return out;
        }
        else if constexpr (SIZE == 2 && sizeof(Base) == 4) {
            return T{ dd[base][0], dd[base + 1][1] };
        }
        else {
            IT ii;
            for (int k = 0; k < SIZE; k++)
                ii[k] = idx_t(base + k);
            return gather_impl(dd, ii, Seq{});
        }
    }

    [[gnu::hot, gnu::always_inline]]
    void scatter_contiguous(idx_t base, const T& v) {
        if constexpr (SIZE == 4 && sizeof(Base) == 4) {
            reinterpret_cast<Base*>(&dd[base    ])[0] = v[0];
            reinterpret_cast<Base*>(&dd[base + 1])[1] = v[1];
            reinterpret_cast<Base*>(&dd[base + 2])[2] = v[2];
            reinterpret_cast<Base*>(&dd[base + 3])[3] = v[3];
        }
        else {
            IT ii;
            for (int k = 0; k < SIZE; k++)
                ii[k] = idx_t(base + k);
            scatter_impl(dd, ii, v, Seq{});
        }
    }
#else
    [[gnu::hot, gnu::always_inline]]
    T gather_contiguous(idx_t base) const {
        IT ii;
        for (int k = 0; k < SIZE; k++)
            ii[k] = idx_t(base + k);
        return gather_impl(dd, ii, Seq{});
    }

    [[gnu::hot, gnu::always_inline]]
    void scatter_contiguous(idx_t base, const T& v) {
        IT ii;
        for (int k = 0; k < SIZE; k++)
            ii[k] = idx_t(base + k);
        scatter_impl(dd, ii, v, Seq{});
    }
#endif

    // ─────────────────────────────────────────────────────────────
    //  Vector-indexed proxy classes for arr[idx] syntax
    // ─────────────────────────────────────────────────────────────
    class ZProxy {
        ZArray& a;
        IT ii;
    public:
        ZProxy(ZArray& a, const IT& i) : a(a), ii(i) {}

        [[gnu::hot, gnu::always_inline]]
        operator T() const { return gather_impl(a.dd, ii, Seq{}); }

        [[gnu::hot, gnu::always_inline]]
        T operator=(const T& v) {
            scatter_impl(a.dd, ii, v, Seq{});
            return v;
        }
    };

    class ZProxyConst {
        const ZArray& a;
        IT ii;
    public:
        ZProxyConst(const ZArray& a, const IT& i) : a(a), ii(i) {}

        [[gnu::hot, gnu::always_inline]]
        operator T() const { return gather_impl(a.dd, ii, Seq{}); }
    };

    // ─────────────────────────────────────────────────────────────
    //  Vector-indexed operator[] — SFINAE ensures no ambiguity
    //  with the scalar operator[](size_t) above.
    //  int/size_t → scalar overload,  int4v/int8v → this one.
    // ─────────────────────────────────────────────────────────────
    template<typename I, std::enable_if_t<is_simd_vector<I>::value, int> = 0>
    ZProxyConst operator[](const I& i) const { return ZProxyConst(*this, IT(i)); }

    template<typename I, std::enable_if_t<is_simd_vector<I>::value, int> = 0>
    ZProxy operator[](const I& i) { return ZProxy(*this, IT(i)); }
};

