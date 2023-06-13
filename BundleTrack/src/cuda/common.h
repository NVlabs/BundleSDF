#pragma once

#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <string>
#include <functional>
#include <algorithm>
#include <fstream>
#include <memory>
#include <thread>
#include <mutex>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <numeric>
#include <type_traits>
#include <array>
#include <set>
#include <utility>
#include <limits>
#include <tuple>
#include <complex>
#include <queue>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda.h>
#include <bits/stdc++.h>


#define MAX_MATCHES_PER_IMAGE_PAIR_RAW 1024
#define MAX_MATCHES_PER_IMAGE_PAIR_FILTERED 1024
#define MINF __int_as_float(0xff800000)
#define FULL_MASK 0xFFFFFFFF
#define FLOAT_EPSILON 0.000001f
#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32


#if defined (LINUX)
#define __FUNCTION__ __func__
#ifndef __LINE__
#define __LINE__
#endif
#endif


inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
  if(err!=cudaSuccess) {
    printf("%s(%d) : cudaSafeCall() Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString( err ));
    exit(-1);
  }
};


inline void __cutilGetLastError( const char *errorMessage, const char *file, const int line )
{
  cudaError_t err = cudaGetLastError();
  if(err!=cudaSuccess) {
    printf("%s(%d) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString( err ));
    exit(-1);
  }
};


#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
#define cutilCheckMsg(msg)           __cutilGetLastError (msg, __FILE__, __LINE__)

#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=nullptr; } }
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=nullptr; } }


#ifndef UINT
typedef unsigned int UINT;
#endif

#ifndef UCHAR
typedef unsigned char UCHAR;
#endif

#ifndef INT64
#ifdef WIN32
typedef __int64 INT64;
#else
typedef int64_t INT64;
#endif
#endif

#ifndef UINT32
#ifdef WIN32
typedef unsigned __int32 UINT32;
#else
typedef uint32_t UINT32;
#endif
#endif

#ifndef UINT64
#ifdef WIN32
typedef unsigned __int64 UINT64;
#else
typedef uint64_t UINT64;
#endif
#endif

#ifndef FLOAT
typedef float FLOAT;
#endif

#ifndef DOUBLE
typedef double DOUBLE;
#endif

#ifndef BYTE
using BYTE = unsigned char;
#endif

#ifndef USHORT
typedef unsigned short USHORT;
#endif

#ifndef sint
typedef signed int sint;
#endif

#ifndef uint
typedef unsigned int uint;
#endif

#ifndef slong
typedef signed long slong;
#endif

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef schar
typedef signed char schar;
#endif


inline int divCeil(int a, int b)
{
  return (a+b-1)/b;
};


