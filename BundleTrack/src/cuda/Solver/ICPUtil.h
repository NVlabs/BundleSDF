#pragma once

#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "../cuda_SimpleMatrixUtil.h"


// color stuff
inline __device__ mat2x3 dCameraToScreen(const float3& p, float fx, float fy)
{
	mat2x3 res; res.setZero();
	const float wSquared = p.z*p.z;

	res(0, 0) = fx / p.z;
	res(1, 1) = fy / p.z;
	res(0, 2) = -fx * p.x / wSquared;
	res(1, 2) = -fy * p.y / wSquared;

	return res;
}

inline __device__ float2 bilinearInterpolationFloat2(float x, float y, const float2* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta = y - p00.y;

	float2 s0 = make_float2(0.0f); float w0 = 0.0f;
	if (p00.x < imageWidth && p00.y < imageHeight) { float2 v00 = d_input[p00.y*imageWidth + p00.x]; if(v00.x != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x < imageWidth && p10.y < imageHeight) { float2 v10 = d_input[p10.y*imageWidth + p10.x]; if(v10.x != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float2 s1 = make_float2(0.0f); float w1 = 0.0f;
	if (p01.x < imageWidth && p01.y < imageHeight) { float2 v01 = d_input[p01.y*imageWidth + p01.x]; if(v01.x != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x < imageWidth && p11.y < imageHeight) { float2 v11 = d_input[p11.y*imageWidth + p11.x]; if(v11.x != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float2 p0 = s0 / w0;
	const float2 p1 = s1 / w1;

	float2 ss = make_float2(0.0f); float ww = 0.0f;
	if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
	if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

	if (ww > 0.0f) return ss / ww;
	else		  return make_float2(MINF);
}

inline __device__ float bilinearInterpolationFloat(float x, float y, const float* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta = y - p00.y;

	float s0 = 0.0f; float w0 = 0.0f;
	if (p00.x < imageWidth && p00.y < imageHeight) { float v00 = d_input[p00.y*imageWidth + p00.x]; if(v00 != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x < imageWidth && p10.y < imageHeight) { float v10 = d_input[p10.y*imageWidth + p10.x]; if(v10 != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float s1 = 0.0f; float w1 = 0.0f;
	if (p01.x < imageWidth && p01.y < imageHeight) { float v01 = d_input[p01.y*imageWidth + p01.x]; if(v01 != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x < imageWidth && p11.y < imageHeight) { float v11 = d_input[p11.y*imageWidth + p11.x]; if(v11 != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float p0 = s0 / w0;
	const float p1 = s1 / w1;

	float ss = 0.0f; float ww = 0.0f;
	if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
	if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

	if (ww > 0.0f) return ss / ww;
	else		  return MINF;
}
inline __device__ float4 bilinearInterpolationFloat4(float x, float y, const float4* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta = y - p00.y;

	float4 s0 = make_float4(0.0f); float w0 = 0.0f;
	if (p00.x < imageWidth && p00.y < imageHeight) { float4 v00 = d_input[p00.y*imageWidth + p00.x]; if (v00.x != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x < imageWidth && p10.y < imageHeight) { float4 v10 = d_input[p10.y*imageWidth + p10.x]; if (v10.x != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float4 s1 = make_float4(0.0f); float w1 = 0.0f;
	if (p01.x < imageWidth && p01.y < imageHeight) { float4 v01 = d_input[p01.y*imageWidth + p01.x]; if(v01.x != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x < imageWidth && p11.y < imageHeight) { float4 v11 = d_input[p11.y*imageWidth + p11.x]; if(v11.x != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float4 p0 = s0 / w0;
	const float4 p1 = s1 / w1;

	float4 ss = make_float4(0.0f); float ww = 0.0f;
	if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
	if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

	if (ww > 0.0f) return ss / ww;
	else		  return make_float4(MINF);
}

