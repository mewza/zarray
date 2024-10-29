/*
    ZArray is a C++ class (v1.1) that allows SIMD vector arrays to be indexed by
    integer equivalent in vector size vectors such as int8, int4, int2
    
    LICENSE: FREE for commercial and non-commercial use,
    it is an ENJOYWARE and hope it speeds up your project for you. Credit mention
    always welcome, as usual.
    
    NEW (v1.1): Can it be further optimized? I am sure, if you have an idea write me
    to: subband@protonmail.com. In the meantime I added a nicer way identify T vector 
    size using special C++ macros (before included in const1.h in other projects, 
    but now specifically added to this project)
    
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

template<class T, size_t N> struct ZArray;
template<class T, size_t N> struct ZArray {
    
    using IT = SimdSame<T,int>;
    using T8 = Simd<SimdBase<T>,8>;
    using T4 = Simd<SimdBase<T>,4>;
    using T2 = Simd<SimdBase<T>,2>;
    using T1 = SimdBase<T>;

public:
    T dd[N];
    
    class ZProxy
    {
    public:
        ZArray &a;
        IT ii;
        
        ZProxy(ZArray &a, IT& i) : a(a), ii(i) {}
        T& operator = (const T& v) {
            if constexpr( SimdSize<T> == 8 ) {
                static T8 tmp;
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1]; a.dd[ii[2]][2] = v[2]; a.dd[ii[3]][3] = v[3];
                a.dd[ii[4]][4] = v[4]; a.dd[ii[5]][5] = v[5]; a.dd[ii[6]][6] = v[6]; a.dd[ii[7]][7] = v[7];
                tmp = T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3], a.dd[ii[4]][4], a.dd[ii[5]][5], a.dd[ii[6]][6], a.dd[ii[7]][7] };
                return tmp;
            } else if constexpr( SimdSize<T> == 4 ) {
                static T4 tmp;
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1]; a.dd[ii[2]][2] = v[2]; a.dd[ii[3]][3] = v[3];
                tmp = T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3] };
                return tmp;
            } else if constexpr( SimdSize<T> == 2 ) {
                static T2 tmp;
                a.dd[ii[0]][0] = v[0]; a.dd[ii[1]][1] = v[1];
                tmp = T{ a.dd[ii[0]][0], a.dd[ii[1]][1] };
                return tmp;
            } else if constexpr( SimdSize<T> == 1 ) {
                static T1 tmp;
                a.dd[ii[0]] = v;
                tmp = a.dd[ii[0]];
                return tmp;
            }
        }
        
        inline operator T() const
        {
            if constexpr( SimdSize<T> == 8 ) {
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3], a.dd[ii[4]][4], a.dd[ii[5]][5], a.dd[ii[6]][6], a.dd[ii[7]][7] };
            } else if constexpr( SimdSize<T> == 4 ) {
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3] };
            } else if constexpr( SimdSize<T> == 2 ) {
                return T{ a.dd[ii[0]][0], a.dd[ii[1]][1] };
            } else if constexpr( SimdSize<T> == 1 ) {
                return a.dd[ii[0]];
            }
        }
    };
    ZProxy operator[] (IT& i) { return ZProxy(*this, i); }
    ZArray() {}
};
