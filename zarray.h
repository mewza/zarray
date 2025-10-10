/*
    ZArray is a C++ class (v1.2) that allows SIMD vector arrays to be indexed by
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

template<class T, size_t N, typename idx_t = int> struct ZArray;
template<class T, size_t N, typename idx_t = int> struct ZArray
{
    using IT = SimdSame<T,idx_t>;
    static const int SIZE = SimdSize<T>;
public:
    alignas(MALLOC_ALIGN) T dd[N];
    
    ZArray() {
        reset();
    }
    
    void reset() {
        memset(dd, 0, sizeof(dd));
    }
    
    class ZProxy
    {
    public:
        ZArray &a;
        IT ii;
        
        ZProxy(ZArray &a, const IT& i) : a(a), ii(i) {}
       
#define ACCESS_T_NEON \
        [[gnu::hot, gnu::always_inline]] \
        inline operator T() const \
        { \
            if constexpr (SIZE == 8 && std::is_same_v<T, uint8x8_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                __builtin_prefetch(&a.dd[ii[4]][4], 0, 3); \
                __builtin_prefetch(&a.dd[ii[5]][5], 0, 3); \
                __builtin_prefetch(&a.dd[ii[6]][6], 0, 3); \
                __builtin_prefetch(&a.dd[ii[7]][7], 0, 3); \
                uint8x8_t result = vdup_n_u8(0); \
                result = vset_lane_u8(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_u8(a.dd[ii[1]][1], result, 1); \
                result = vset_lane_u8(a.dd[ii[2]][2], result, 2); \
                result = vset_lane_u8(a.dd[ii[3]][3], result, 3); \
                result = vset_lane_u8(a.dd[ii[4]][4], result, 4); \
                result = vset_lane_u8(a.dd[ii[5]][5], result, 5); \
                result = vset_lane_u8(a.dd[ii[6]][6], result, 6); \
                result = vset_lane_u8(a.dd[ii[7]][7], result, 7); \
                return result; \
            } \
            else if constexpr (SIZE == 8 && std::is_same_v<T, int8x8_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                __builtin_prefetch(&a.dd[ii[4]][4], 0, 3); \
                __builtin_prefetch(&a.dd[ii[5]][5], 0, 3); \
                __builtin_prefetch(&a.dd[ii[6]][6], 0, 3); \
                __builtin_prefetch(&a.dd[ii[7]][7], 0, 3); \
                int8x8_t result = vdup_n_s8(0); \
                result = vset_lane_s8(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_s8(a.dd[ii[1]][1], result, 1); \
                result = vset_lane_s8(a.dd[ii[2]][2], result, 2); \
                result = vset_lane_s8(a.dd[ii[3]][3], result, 3); \
                result = vset_lane_s8(a.dd[ii[4]][4], result, 4); \
                result = vset_lane_s8(a.dd[ii[5]][5], result, 5); \
                result = vset_lane_s8(a.dd[ii[6]][6], result, 6); \
                result = vset_lane_s8(a.dd[ii[7]][7], result, 7); \
                return result; \
            } \
            else if constexpr (SIZE == 4 && std::is_same_v<T, float32x4_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                float32x4_t result = vdupq_n_f32(0); \
                result = vsetq_lane_f32(a.dd[ii[0]][0], result, 0); \
                result = vsetq_lane_f32(a.dd[ii[1]][1], result, 1); \
                result = vsetq_lane_f32(a.dd[ii[2]][2], result, 2); \
                result = vsetq_lane_f32(a.dd[ii[3]][3], result, 3); \
                return result; \
            } \
            else if constexpr (SIZE == 4 && std::is_same_v<T, int32x4_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                int32x4_t result = vdupq_n_s32(0); \
                result = vsetq_lane_s32(a.dd[ii[0]][0], result, 0); \
                result = vsetq_lane_s32(a.dd[ii[1]][1], result, 1); \
                result = vsetq_lane_s32(a.dd[ii[2]][2], result, 2); \
                result = vsetq_lane_s32(a.dd[ii[3]][3], result, 3); \
                return result; \
            } \
            else if constexpr (SIZE == 4 && std::is_same_v<T, uint32x4_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                __builtin_prefetch(&a.dd[ii[2]][2], 0, 3); \
                __builtin_prefetch(&a.dd[ii[3]][3], 0, 3); \
                uint32x4_t result = vdupq_n_u32(0); \
                result = vsetq_lane_u32(a.dd[ii[0]][0], result, 0); \
                result = vsetq_lane_u32(a.dd[ii[1]][1], result, 1); \
                result = vsetq_lane_u32(a.dd[ii[2]][2], result, 2); \
                result = vsetq_lane_u32(a.dd[ii[3]][3], result, 3); \
                return result; \
            } \
            else if constexpr (SIZE == 2 && std::is_same_v<T, float32x2_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                float32x2_t result = vdup_n_f32(0); \
                result = vset_lane_f32(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_f32(a.dd[ii[1]][1], result, 1); \
                return result; \
            } \
            else if constexpr (SIZE == 2 && std::is_same_v<T, int32x2_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                int32x2_t result = vdup_n_s32(0); \
                result = vset_lane_s32(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_s32(a.dd[ii[1]][1], result, 1); \
                return result; \
            } \
            else if constexpr (SIZE == 2 && std::is_same_v<T, uint32x2_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                uint32x2_t result = vdup_n_u32(0); \
                result = vset_lane_u32(a.dd[ii[0]][0], result, 0); \
                result = vset_lane_u32(a.dd[ii[1]][1], result, 1); \
                return result; \
            } \
            else if constexpr (SIZE == 2 && std::is_same_v<T, float64x2_t>) { \
                __builtin_prefetch(&a.dd[ii[0]][0], 0, 3); \
                __builtin_prefetch(&a.dd[ii[1]][1], 0, 3); \
                float64x2_t result = vdupq_n_f64(0); \
                result = vsetq_lane_f64(a.dd[ii[0]][0], result, 0); \
                result = vsetq_lane_f64(a.dd[ii[1]][1], result, 1); \
                return result; \
            } \
            else { \
                __builtin_prefetch(&a.dd[ii[0]], 0, 3); \
                return a.dd[ii[0]]; \
            } \
        }
        
#define ACCESS_T \
        [[gnu::hot, gnu::always_inline]] \
        inline operator T() const \
        { \
            if constexpr( SIZE == 8 ) { \
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3], a.dd[ii[4]][4], a.dd[ii[5]][5], a.dd[ii[6]][6], a.dd[ii[7]][7] }; \
            } else if constexpr( SIZE == 4 ) { \
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3] }; \
            } else if constexpr( SIZE == 2 ) { \
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1] }; \
            } else if constexpr( SIZE == 1 ) { \
                return a.dd[ii[0]]; \
            } \
        }
        
#if defined(__aarch64__) && defined(__ARM_NEON)
        ACCESS_T_NEON
        
        [[gnu::hot, gnu::always_inline]]
        T operator = (const T& v) {
            // SIZE == 8
            if constexpr (SIZE == 8 && std::is_same_v<T, uint8x8_t>) {
                // Prefetch for write (1 = write intent, 3 = high temporal locality)
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                __builtin_prefetch(&a.dd[ii[4]][4], 1, 3);
                __builtin_prefetch(&a.dd[ii[5]][5], 1, 3);
                __builtin_prefetch(&a.dd[ii[6]][6], 1, 3);
                __builtin_prefetch(&a.dd[ii[7]][7], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_u8(v, 0);
                a.dd[ii[1]][1] = vget_lane_u8(v, 1);
                a.dd[ii[2]][2] = vget_lane_u8(v, 2);
                a.dd[ii[3]][3] = vget_lane_u8(v, 3);
                a.dd[ii[4]][4] = vget_lane_u8(v, 4);
                a.dd[ii[5]][5] = vget_lane_u8(v, 5);
                a.dd[ii[6]][6] = vget_lane_u8(v, 6);
                a.dd[ii[7]][7] = vget_lane_u8(v, 7);
                return v;
            }
            else if constexpr (SIZE == 8 && std::is_same_v<T, int8x8_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                __builtin_prefetch(&a.dd[ii[4]][4], 1, 3);
                __builtin_prefetch(&a.dd[ii[5]][5], 1, 3);
                __builtin_prefetch(&a.dd[ii[6]][6], 1, 3);
                __builtin_prefetch(&a.dd[ii[7]][7], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_s8(v, 0);
                a.dd[ii[1]][1] = vget_lane_s8(v, 1);
                a.dd[ii[2]][2] = vget_lane_s8(v, 2);
                a.dd[ii[3]][3] = vget_lane_s8(v, 3);
                a.dd[ii[4]][4] = vget_lane_s8(v, 4);
                a.dd[ii[5]][5] = vget_lane_s8(v, 5);
                a.dd[ii[6]][6] = vget_lane_s8(v, 6);
                a.dd[ii[7]][7] = vget_lane_s8(v, 7);
                return v;
            } // SIZE == 4
            else if constexpr (SIZE == 4 && std::is_same_v<T, float32x4_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                
                a.dd[ii[0]][0] = vgetq_lane_f32(v, 0);
                a.dd[ii[1]][1] = vgetq_lane_f32(v, 1);
                a.dd[ii[2]][2] = vgetq_lane_f32(v, 2);
                a.dd[ii[3]][3] = vgetq_lane_f32(v, 3);
                return v;
            }
            else if constexpr (SIZE == 4 && std::is_same_v<T, int32x4_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                
                a.dd[ii[0]][0] = vgetq_lane_s32(v, 0);
                a.dd[ii[1]][1] = vgetq_lane_s32(v, 1);
                a.dd[ii[2]][2] = vgetq_lane_s32(v, 2);
                a.dd[ii[3]][3] = vgetq_lane_s32(v, 3);
                return v;
            }
            else if constexpr (SIZE == 4 && std::is_same_v<T, uint32x4_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                __builtin_prefetch(&a.dd[ii[2]][2], 1, 3);
                __builtin_prefetch(&a.dd[ii[3]][3], 1, 3);
                
                a.dd[ii[0]][0] = vgetq_lane_u32(v, 0);
                a.dd[ii[1]][1] = vgetq_lane_u32(v, 1);
                a.dd[ii[2]][2] = vgetq_lane_u32(v, 2);
                a.dd[ii[3]][3] = vgetq_lane_u32(v, 3);
                return v;
            } // SIZE == 2
            else if constexpr (SIZE == 2 && std::is_same_v<T, float32x2_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_f32(v, 0);
                a.dd[ii[1]][1] = vget_lane_f32(v, 1);
                return v;
            }
            else if constexpr (SIZE == 2 && std::is_same_v<T, int32x2_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_s32(v, 0);
                a.dd[ii[1]][1] = vget_lane_s32(v, 1);
                return v;
            }
            else if constexpr (SIZE == 2 && std::is_same_v<T, uint32x2_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                
                a.dd[ii[0]][0] = vget_lane_u32(v, 0);
                a.dd[ii[1]][1] = vget_lane_u32(v, 1);
                return v;
            }
            else if constexpr (SIZE == 2 && std::is_same_v<T, float64x2_t>) {
                __builtin_prefetch(&a.dd[ii[0]][0], 1, 3);
                __builtin_prefetch(&a.dd[ii[1]][1], 1, 3);
                
                a.dd[ii[0]][0] = vgetq_lane_f64(v, 0);
                a.dd[ii[1]][1] = vgetq_lane_f64(v, 1);
                return v;
            } // SIZE == 1
            else {
                __builtin_prefetch(&a.dd[ii[0]], 1, 3);
                return a.dd[ii[0]] = v;
            }
        }
#else
        [[gnu::hot, gnu::always_inline]]
        inline T operator = (const T& v) {
            if constexpr( SIZE == 8 ) {
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1]; a.dd[ii[2]][2] = v[2]; a.dd[ii[3]][3] = v[3];
                a.dd[ii[4]][4] = v[4]; a.dd[ii[5]][5] = v[5]; a.dd[ii[6]][6] = v[6]; a.dd[ii[7]][7] = v[7];
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3], a.dd[ii[4]][4], a.dd[ii[5]][5], a.dd[ii[6]][6], a.dd[ii[7]][7] };
            }
            else if constexpr( SIZE == 4 ) {
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1]; a.dd[ii[2]][2] = v[2]; a.dd[ii[3]][3] = v[3];
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3] };
            }
            else if constexpr( SIZE == 2 ) {
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1];
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1] };
            }
            else {
                return a.dd[ii[0]] = v;
            }
        }
        ACCESS_T
#endif
    };
    class ZProxyConst
      {
      public:
          const ZArray &a;
          IT ii;
      
          ZProxyConst(const ZArray &a, const IT& i) : a(a), ii(i) {}
#if defined(__aarch64__) && defined(__ARM_NEON)
          ACCESS_T_NEON
#else
          ACCESS_T
#endif
      };
    ZProxyConst operator[] (const IT& i) const { return ZProxyConst(*this, i); }
    ZProxy operator[] (const IT& i) { return ZProxy(*this, i); }
};

