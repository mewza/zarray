
// zarray.h - ZArray C++ class allows vector arrays to be indexed 
// by an int-vectors such as int8, int4, int2, ..
// Copyright (C) 2024 Dmitry Boldyrev. All Rights Reserved.
// 
// License: FREE for commercial and non-commercial use,
// it is an ENJOYWARE and hope it speeds up your project for you.
// Here's an example:

/*
    ZArray<double4v> arr;
    arr.
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

template<class T, size_t N> struct ZArray;
template<class T, size_t N> struct ZArray {

#define DECL_TT(TT, X, FLOAT, DOUBLE) \
using TT =  std::conditional_t< std::is_same_v<X, float8v>, FLOAT##8v, \
std::conditional_t< std::is_same_v<X, float4v>, FLOAT##4v, \
std::conditional_t< std::is_same_v<X, float2v>, FLOAT##2v, \
std::conditional_t< std::is_same_v<X, float>,    FLOAT,  \
std::conditional_t< std::is_same_v<X, double8v>, DOUBLE##8v, \
std::conditional_t< std::is_same_v<X, double4v>, DOUBLE##4v, \
std::conditional_t< std::is_same_v<X, double2v>, DOUBLE##2v, T >>>>>>>;
    
    DECL_TT(T1, T, float, double);
    DECL_TT(T2, T, float, double);
    DECL_TT(T4, T, float, double);
    DECL_TT(T8, T, float, double);
#undef DECL_TT
    
    using IT =  std::conditional_t< std::is_same_v<T, float8v>, int8v,
                std::conditional_t< std::is_same_v<T, float4v>, int4v,
                std::conditional_t< std::is_same_v<T, float2v>, int2v,
                std::conditional_t< std::is_same_v<T, double8v>, int8v,
                std::conditional_t< std::is_same_v<T, double4v>, int4v,
                std::conditional_t< std::is_same_v<T, double2v>, int2v, int >>>>>>;
public:
    T dd[N];
    
    class ZProxy
    {
    public:
        ZArray &a;
        IT ii;
        
        ZProxy(ZArray &a, IT& i) : a(a), ii(i) {}
        T& operator = (T& v) {
            if constexpr(  std::is_same_v<T, float8v> || std::is_same_v<T, double8v> ) {
                static T8 tmp;
                a.dd[ii[0]][0]=v[0]; a.dd[ii[1]][1]=v[1]; a.dd[ii[2]][2]=v[2]; a.dd[ii[3]][3]=v[3];
                a.dd[ii[4]][4]=v[4]; a.dd[ii[5]][5]=v[5]; a.dd[ii[6]][6]=v[6]; a.dd[ii[7]][7]=v[7];
                tmp = T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3], a.dd[ii[4]][4], a.dd[ii[5]][5], a.dd[ii[6]][6], a.dd[ii[7]][7] };
                return tmp;
            } else if constexpr(  std::is_same_v<T, float4v> || std::is_same_v<T, double4v> ) {
                static T4 tmp;
                a.dd[ii[0]][0]=v[0]; a.dd[ii[1]][1]=v[1]; a.dd[ii[2]][2]=v[2]; a.dd[ii[3]][3]=v[3];
                tmp = T{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3] };
                return tmp;
            } else if constexpr(  std::is_same_v<T, float2v> || std::is_same_v<T, double2v> ) {
                static T2 tmp;
                a.dd[ii[0]][0]=v[0]; a.dd[ii[1]][1]=v[1];
                tmp = T2{ a.dd[ii[0]][0], a.dd[ii[1]][1] };
                return tmp;
            }
        }
        
        inline operator T() const 
        {
            if constexpr( std::is_same_v<T, float8v> || std::is_same_v<T, double8v> ) {
                return T8{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3], a.dd[ii[4]][4], a.dd[ii[5]][5], a.dd[ii[6]][6], a.dd[ii[7]][7] };
            } else if constexpr( std::is_same_v<T, float4v> || std::is_same_v<T, double4v> ) {
                return T4{ a.dd[ii[0]][0], a.dd[ii[1]][1], a.dd[ii[2]][2], a.dd[ii[3]][3] };
            } else if constexpr( std::is_same_v<T, float2v> || std::is_same_v<T, double2v> ) {
                return T2{ a.dd[ii[0]][0], a.dd[ii[1]][1] };
            }
        }
        
        
    };
    ZProxy operator[] (IT& i) { return ZProxy(*this, i); }
    ZArray() {}
    
};
