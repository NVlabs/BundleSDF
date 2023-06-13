#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <bits/stdc++.h>
#include "common.h"


// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__inline__ __device__ float atomicMax(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
};


__inline__ __device__ float atomicMin(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
};



__inline__ __device__ float warpReduce(float val) {
	int offset = 32 >> 1;
	while (offset > 0) {
		val = val + __shfl_down_sync(FULL_MASK, val, offset, 32);
		offset = offset >> 1;
	}
	return val;
}

/**
 * @brief : From g2o
 * @e: squared error
 */
__inline__ __device__ void huberLoss(const float e, const float delta, float3 &rho)
{
	float dsqr = delta * delta;
	if (e <= dsqr)
	{												 // inlier
		rho.x = e;
		rho.y = 1.;
		rho.z = 0.;
	}
	else
	{																			// outlier
		double sqrte = sqrt(e);							// absolut value of the error
		rho.x = 2 * sqrte * delta - dsqr; // rho(e)   = 2 * delta * e^(1/2) - delta^2
		rho.y = delta / sqrte;						// rho'(e)  = delta / sqrt(e)
		rho.z = -0.5 * rho.y / e;					// rho''(e) = -1 / (2*e^(3/2)) = -1/2 * (delta/e) / e
	}
}

