
#include "CUDAImageUtil.h"
#include "cudaUtil.h"
#include "common.h"

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)


namespace CUDAImageUtil
{
template<class T> void copy(T* d_output, T* d_input, unsigned int width, unsigned int height) {
	cutilSafeCall(cudaMemcpy(d_output, d_input, sizeof(T)*width*height, cudaMemcpyDeviceToDevice));
}

template<> void copy<float>(float* d_output, float* d_input, unsigned int width, unsigned int height){cutilSafeCall(cudaMemcpy(d_output, d_input, sizeof(float)*width*height, cudaMemcpyDeviceToDevice));}//change by guan
template<> void copy<uchar4>(uchar4* d_output, uchar4* d_input, unsigned int width, unsigned int height){cutilSafeCall(cudaMemcpy(d_output, d_input, sizeof(uchar4)*width*height, cudaMemcpyDeviceToDevice));}//change by guan



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resample Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float bilinearInterpolationFloat(float x, float y, const float* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta = y - p00.y;

	float s0 = 0.0f; float w0 = 0.0f;
	if (p00.x < imageWidth && p00.y < imageHeight) { float v00 = d_input[p00.y*imageWidth + p00.x]; if (v00 != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x < imageWidth && p10.y < imageHeight) { float v10 = d_input[p10.y*imageWidth + p10.x]; if (v10 != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float s1 = 0.0f; float w1 = 0.0f;
	if (p01.x < imageWidth && p01.y < imageHeight) { float v01 = d_input[p01.y*imageWidth + p01.x]; if (v01 != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x < imageWidth && p11.y < imageHeight) { float v11 = d_input[p11.y*imageWidth + p11.x]; if (v11 != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float p0 = s0 / w0;
	const float p1 = s1 / w1;

	float ss = 0.0f; float ww = 0.0f;
	if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
	if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

	if (ww > 0.0f) return ss / ww;
	else		  return MINF;
}

//template<class T>
//__global__ void resample_Kernel(T* d_output, T* d_input, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight)
//{
//	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
//
//	if (x < outputWidth && y < outputHeight)
//	{
//		const float scaleWidth = (float)(inputWidth - 1) / (float)(outputWidth - 1);
//		const float scaleHeight = (float)(inputHeight - 1) / (float)(outputHeight - 1);
//
//		const unsigned int xInput = (unsigned int)(x*scaleWidth + 0.5f);
//		const unsigned int yInput = (unsigned int)(y*scaleHeight + 0.5f);
//
//		if (xInput < inputWidth && yInput < inputHeight)
//		{
//			if (std::is_same<T, float>::value) {
//				d_output[y*outputWidth + x] = (T)bilinearInterpolationFloat(x*scaleWidth, y*scaleHeight, (float*)d_input, inputWidth, inputHeight);
//			}
//			else if (std::is_same<T, uchar4>::value) {
//				d_output[y*outputWidth + x] = d_input[yInput*inputWidth + xInput];
//			}
//			else {
//				//static_assert(false, "bla");
//			}
//		}
//	}
//}
//
//template<class T> void CUDAImageUtil::resample(T* d_output, unsigned int outputWidth, unsigned int outputHeight, T* d_input, unsigned int inputWidth, unsigned int inputHeight) {
//
//	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
//	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
//
//	resample_Kernel << <gridSize, blockSize >> >(d_output, d_input, inputWidth, inputHeight, outputWidth, outputHeight);
//
//#ifdef DEBUG
//	cutilSafeCall(cudaDeviceSynchronize());
//	cutilCheckMsg(__FUNCTION__);
//#endif
//}


__global__ void resampleFloat_Kernel(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const float* d_input, unsigned int inputWidth, unsigned int inputHeight)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < outputWidth && y < outputHeight)
	{
		const float scaleWidth = (float)(inputWidth-1) / (float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1) / (float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth + 0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight + 0.5f);

		if (xInput < inputWidth && yInput < inputHeight) {
			d_output[y*outputWidth + x] = d_input[yInput*inputWidth + xInput];
			//d_output[y*outputWidth + x] = bilinearInterpolationFloat(x*scaleWidth, y*scaleHeight, d_input, inputWidth, inputHeight);
		}
	}
}

void resampleFloat(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const float* d_input, unsigned int inputWidth, unsigned int inputHeight) {

	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	resampleFloat_Kernel << <gridSize, blockSize >> >(d_output, outputWidth, outputHeight, d_input, inputWidth, inputHeight);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void resampleFloat4_Kernel(float4* d_output, unsigned int outputWidth, unsigned int outputHeight, const float4* d_input, unsigned int inputWidth, unsigned int inputHeight)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < outputWidth && y < outputHeight)
	{
		const float scaleWidth = (float)(inputWidth-1) / (float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1) / (float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth + 0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight + 0.5f);

		if (xInput < inputWidth && yInput < inputHeight) {
			d_output[y*outputWidth + x] = d_input[yInput*inputWidth + xInput];
			//d_output[y*outputWidth + x] = bilinearInterpolationFloat(x*scaleWidth, y*scaleHeight, d_input, inputWidth, inputHeight);
		}
	}
}
void resampleFloat4(float4* d_output, unsigned int outputWidth, unsigned int outputHeight, const float4* d_input, unsigned int inputWidth, unsigned int inputHeight) {

	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	resampleFloat4_Kernel << <gridSize, blockSize >> >(d_output, outputWidth, outputHeight, d_input, inputWidth, inputHeight);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



__global__ void resampleUCHAR4_Kernel(uchar4* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < outputWidth && y < outputHeight)
	{
		const float scaleWidth = (float)(inputWidth-1) / (float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1) / (float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth + 0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight + 0.5f);

		if (xInput < inputWidth && yInput < inputHeight) {
			d_output[y*outputWidth + x] = d_input[yInput*inputWidth + xInput];
		}
	}
}

void resampleUCHAR4(uchar4* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight) {

	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	resampleUCHAR4_Kernel << <gridSize, blockSize >> >(d_output, outputWidth, outputHeight, d_input, inputWidth, inputHeight);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Color to Intensity
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__
float convertToIntensity(const uchar4& c) {
	return (0.299f*c.x + 0.587f*c.y + 0.114f*c.z) / 255.0f;
}

__global__ void convertUCHAR4ToIntensityFloat_Kernel(float* d_output, const uchar4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		d_output[y*width + x] = convertToIntensity(d_input[y*width + x]);
	}
}

void convertUCHAR4ToIntensityFloat(float* d_output, const uchar4* d_input, unsigned int width, unsigned int height) {

	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertUCHAR4ToIntensityFloat_Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void resampleToIntensity_Kernel(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < outputWidth && y < outputHeight)
	{
		const float scaleWidth = (float)(inputWidth-1) / (float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1) / (float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth + 0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight + 0.5f);

		if (xInput < inputWidth && yInput < inputHeight) {
			d_output[y*outputWidth + x] = convertToIntensity(d_input[yInput*inputWidth + xInput]);
		}
	}
}

void resampleToIntensity(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight) {

	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	resampleToIntensity_Kernel << <gridSize, blockSize >> >(d_output, outputWidth, outputHeight, d_input, inputWidth, inputHeight);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// derivatives
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeIntensityDerivatives_Kernel(float2* d_output, const float* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		d_output[y*width + x] = make_float2(MINF, MINF);

		//derivative
		if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
		{
			float pos00 = d_input[(y - 1)*width + (x - 1)]; if (pos00 == MINF) return;
			float pos01 = d_input[(y - 0)*width + (x - 1)];	if (pos01 == MINF) return;
			float pos02 = d_input[(y + 1)*width + (x - 1)];	if (pos02 == MINF) return;

			float pos10 = d_input[(y - 1)*width + (x - 0)]; if (pos10 == MINF) return;
			//float pos11 = d_input[(y-0)*width + (x-0)]; if (pos11 == MINF) return;
			float pos12 = d_input[(y + 1)*width + (x - 0)]; if (pos12 == MINF) return;

			float pos20 = d_input[(y - 1)*width + (x + 1)]; if (pos20 == MINF) return;
			float pos21 = d_input[(y - 0)*width + (x + 1)]; if (pos21 == MINF) return;
			float pos22 = d_input[(y + 1)*width + (x + 1)]; if (pos22 == MINF) return;

			float resU = (-1.0f)*pos00 + (1.0f)*pos20 +
				(-2.0f)*pos01 + (2.0f)*pos21 +
				(-1.0f)*pos02 + (1.0f)*pos22;
			resU /= 8.0f;

			float resV = (-1.0f)*pos00 + (-2.0f)*pos10 + (-1.0f)*pos20 +
				(1.0f)*pos02 + (2.0f)*pos12 + (1.0f)*pos22;
			resV /= 8.0f;

			d_output[y*width + x] = make_float2(resU, resV);
		}
	}
}

void computeIntensityDerivatives(float2* d_output, const float* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeIntensityDerivatives_Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void computeIntensityGradientMagnitude_Kernel(float* d_output, const float* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		d_output[y*width + x] = MINF;

		//derivative
		if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
		{
			float pos00 = d_input[(y - 1)*width + (x - 1)]; if (pos00 == MINF) return;
			float pos01 = d_input[(y - 0)*width + (x - 1)];	if (pos01 == MINF) return;
			float pos02 = d_input[(y + 1)*width + (x - 1)];	if (pos02 == MINF) return;

			float pos10 = d_input[(y - 1)*width + (x - 0)]; if (pos10 == MINF) return;
			//float pos11 = d_input[(y-0)*width + (x-0)]; if (pos11 == MINF) return;
			float pos12 = d_input[(y + 1)*width + (x - 0)]; if (pos12 == MINF) return;

			float pos20 = d_input[(y - 1)*width + (x + 1)]; if (pos20 == MINF) return;
			float pos21 = d_input[(y - 0)*width + (x + 1)]; if (pos21 == MINF) return;
			float pos22 = d_input[(y + 1)*width + (x + 1)]; if (pos22 == MINF) return;

			float resU = (-1.0f)*pos00 + (1.0f)*pos20 +
				(-2.0f)*pos01 + (2.0f)*pos21 +
				(-1.0f)*pos02 + (1.0f)*pos22;
			//resU /= 8.0f;

			float resV = (-1.0f)*pos00 + (-2.0f)*pos10 + (-1.0f)*pos20 +
				(1.0f)*pos02 + (2.0f)*pos12 + (1.0f)*pos22;
			//resV /= 8.0f;

			d_output[y*width + x] = sqrt(resU * resU + resV * resV);
		}
	}
}
void computeIntensityGradientMagnitude(float* d_output, const float* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeIntensityGradientMagnitude_Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Depth to Camera Space Positions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertDepthFloatToCameraSpaceFloat4_Kernel(float4* d_output, const float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		d_output[y*width + x] = make_float4(0,0,0,0);

		float depth = d_input[y*width + x];

		if (depth >= 0.1)
		{
			float4 cameraSpace(intrinsicsInv*make_float4((float)x*depth, (float)y*depth, depth, depth));
			d_output[y*width + x] = make_float4(cameraSpace.x, cameraSpace.y, cameraSpace.w, 1.0f);
			//d_output[y*width + x] = make_float4(depthCameraData.kinectDepthToSkeleton(x, y, depth), 1.0f);
		}
	}
}

void convertDepthFloatToCameraSpaceFloat4(float4* d_output, const float* d_input, const float4x4& intrinsicsInv, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertDepthFloatToCameraSpaceFloat4_Kernel << <gridSize, blockSize >> >(d_output, d_input, intrinsicsInv, width, height);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Normal Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief
 *
 * @param d_output
 * @param d_input : xyz map
 * @param width
 * @param height
 * @return __global__
 */

__global__ void computeNormals_Kernel(float4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	d_output[y*width + x] = make_float4(0,0,0,0);

	const float z_diff_thres = 0.02;

	if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
	{
		const float4 CC = d_input[(y + 0)*width + (x + 0)];
		const float4 PC = d_input[(y + 1)*width + (x + 0)];
		const float4 CP = d_input[(y + 0)*width + (x + 1)];
		const float4 MC = d_input[(y - 1)*width + (x + 0)];
		const float4 CM = d_input[(y + 0)*width + (x - 1)];

		if (CC.z<0.1) return;

		float3 x_dir = make_float3(0,0,0);
		float3 y_dir = make_float3(0,0,0);

		if (PC.z>=0.1 && MC.z>=0.1 && abs(PC.z-CC.z)<=z_diff_thres && abs(MC.z-CC.z)<=z_diff_thres)
		{
			x_dir = make_float3(PC)-make_float3(MC);
		}
		else if (PC.z>=0.1 && abs(PC.z-CC.z)<=z_diff_thres)
		{
			x_dir = make_float3(PC)-make_float3(CC);
		}
		else if (MC.z>=0.1 && abs(MC.z-CC.z)<=z_diff_thres)
		{
			x_dir = make_float3(MC)-make_float3(CC);
		}
		else
		{
			return;
		}

		if (CP.z>=0.1 && CM.z>=0.1 && abs(CP.z-CC.z)<=z_diff_thres && abs(CM.z-CC.z)<=z_diff_thres)
		{
			y_dir = make_float3(CP-CM);
		}
		else if (CP.z>=0.1 && abs(CP.z-CC.z)<=z_diff_thres)
		{
			y_dir = make_float3(CP-CC);
		}
		else if (CM.z>=0.1 && abs(CM.z-CC.z)<=z_diff_thres)
		{
			y_dir = make_float3(CM-CC);
		}
		else
		{
			return;
		}

		float3 n = cross(x_dir, y_dir);
		const float  l = length(n);
		n = n/l;
		if (dot(n, make_float3(-CC.x, -CC.y, -CC.z))<0)
		{
			n = -n;
		}

		if (l > 0.0f)
		{
			d_output[y*width + x] = make_float4(n, 0.0f);
		}
	}
}

void computeNormals(float4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeNormals_Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void computeNormalsSobel_Kernel(float4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	d_output[y*width + x] = make_float4(MINF, MINF, MINF, MINF);

	if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
	{
		float4 pos00 = d_input[(y - 1)*width + (x - 1)]; if (pos00.x == MINF) return;
		float4 pos01 = d_input[(y - 0)*width + (x - 1)]; if (pos01.x == MINF) return;
		float4 pos02 = d_input[(y + 1)*width + (x - 1)]; if (pos02.x == MINF) return;

		float4 pos10 = d_input[(y - 1)*width + (x - 0)]; if (pos10.x == MINF) return;
		//float4 pos11 = d_input[(y-0)*width + (x-0)]; if (pos11.x == MINF) return;
		float4 pos12 = d_input[(y + 1)*width + (x - 0)]; if (pos12.x == MINF) return;

		float4 pos20 = d_input[(y - 1)*width + (x + 1)]; if (pos20.x == MINF) return;
		float4 pos21 = d_input[(y - 0)*width + (x + 1)]; if (pos21.x == MINF) return;
		float4 pos22 = d_input[(y + 1)*width + (x + 1)]; if (pos22.x == MINF) return;

		float4 resU = (-1.0f)*pos00 + (1.0f)*pos20 +
			(-2.0f)*pos01 + (2.0f)*pos21 +
			(-1.0f)*pos02 + (1.0f)*pos22;

		float4 resV = (-1.0f)*pos00 + (-2.0f)*pos10 + (-1.0f)*pos20 +
			(1.0f)*pos02 + (2.0f)*pos12 + (1.0f)*pos22;

		const float3 n = cross(make_float3(resU.x, resU.y, resU.z), make_float3(resV.x, resV.y, resV.z));
		const float  l = length(n);

		if (l > 0.0f) d_output[y*width + x] = make_float4(n / l, 0.0f);
	}
}

void computeNormalsSobel(float4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeNormalsSobel_Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void convertNormalsFloat4ToUCHAR4_Kernel(uchar4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		d_output[y*width + x] = make_uchar4(0, 0, 0, 0);

		float4 p = d_input[y*width + x];

		if (p.x != MINF)
		{
			p = (p + 1.0f) / 2.0f; // -> [0, 1]
			d_output[y*width + x] = make_uchar4((uchar)round(p.x * 255), (uchar)round(p.y * 255), (uchar)round(p.z * 255), 0);
		}
	}
}

void convertNormalsFloat4ToUCHAR4(uchar4* d_output, const float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertNormalsFloat4ToUCHAR4_Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Joint Bilateral Filter
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float gaussD(float sigma, int x, int y)
{
	return exp(-((x*x + y*y) / (2.0f*sigma*sigma)));
}
inline __device__ float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist) / (2.0*sigma*sigma));
}

__global__ void bilateralFilterUCHAR4_Kernel(uchar4* d_output, uchar4* d_color, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width + x] = d_color[y*width + x];

	float3 sum = make_float3(0.0f, 0.0f, 0.0f);
	float sumWeight = 0.0f;

	const float depthCenter = d_depth[y*width + x];
	if (depthCenter != MINF)
	{
		for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
		{
			for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
			{
				if (m >= 0 && n >= 0 && m < width && n < height)
				{
					const uchar4 cur = d_color[n*width + m];
					const float currentDepth = d_depth[n*width + m];

					if (currentDepth != MINF) {
						const float weight = gaussD(sigmaD, m - x, n - y)*gaussR(sigmaR, currentDepth - depthCenter);

						sumWeight += weight;
						sum += weight*make_float3(cur.x, cur.y, cur.z);
					}
				}
			}
		}

		if (sumWeight > 0.0f) {
			float3 res = sum / sumWeight;
			d_output[y*width + x] = make_uchar4((uchar)res.x, (uchar)res.y, (uchar)res.z, 255);
		}
	}
}

void jointBilateralFilterColorUCHAR4(uchar4* d_output, uchar4* d_input, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	bilateralFilterUCHAR4_Kernel << <gridSize, blockSize >> >(d_output, d_input, d_depth, sigmaD, sigmaR, width, height);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void bilateralFilterFloat_Kernel(float* d_output, float* d_input, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width + x] = MINF;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float depthCenter = d_depth[y*width + x];
	if (depthCenter != MINF)
	{
		for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
		{
			for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
			{
				if (m >= 0 && n >= 0 && m < width && n < height)
				{
					const float cur = d_input[n*width + m];
					const float currentDepth = d_depth[n*width + m];

					if (currentDepth != MINF && fabs(depthCenter - currentDepth) < sigmaR)
					{ //const float weight = gaussD(sigmaD, m - x, n - y)*gaussR(sigmaR, currentDepth - depthCenter);
						const float weight = gaussD(sigmaD, m - x, n - y);
						sumWeight += weight;
						sum += weight*cur;
					}
				}
			}
		}

		if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;
	}
}
void jointBilateralFilterFloat(float* d_output, float* d_input, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	bilateralFilterFloat_Kernel << <gridSize, blockSize >> >(d_output, d_input, d_depth, sigmaD, sigmaR, width, height);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void adaptiveBilateralFilterIntensity_Kernel(float* d_output, const float* d_input, const float* d_depth, float sigmaD, float sigmaR, float adaptFactor, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	d_output[y*width + x] = MINF;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float depthCenter = d_depth[y*width + x];
	if (depthCenter != MINF)
	{
		const float curSigma = sigmaD * adaptFactor / depthCenter;
		const int kernelRadius = (int)ceil(2.0*curSigma);

		for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
		{
			for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
			{
				if (m >= 0 && n >= 0 && m < width && n < height)
				{
					const float cur = d_input[n*width + m];
					const float currentDepth = d_depth[n*width + m];

					if (currentDepth != MINF && fabs(depthCenter - currentDepth) < sigmaR)
					{ //const float weight = gaussD(curSigma, m - x, n - y)*gaussR(sigmaR, currentDepth - depthCenter);
						const float weight = gaussD(curSigma, m - x, n - y);
						sumWeight += weight;
						sum += weight*cur;
					}
				}
			}
		}

		if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;
	}
}
void adaptiveBilateralFilterIntensity(float* d_output, const float* d_input, const float* d_depth, float sigmaD, float sigmaR, float adaptFactor, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	adaptiveBilateralFilterIntensity_Kernel << <gridSize, blockSize >> >(d_output, d_input, d_depth, sigmaD, sigmaR, adaptFactor, width, height);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Erode Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void erodeDepthMapDevice(float* d_output, float* d_input, int structureSize, int width, int height, float dThresh, float fracReq, float zfar)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;


	if (x >= 0 && x < width && y >= 0 && y < height)
	{


		unsigned int count = 0;

		float oldDepth = d_input[y*width + x];
		if (oldDepth<=0.1f || oldDepth>zfar)
		{
			d_output[y*width + x] = 0;
			return;
		}
		for (int i = -structureSize; i <= structureSize; i++)
		{
			for (int j = -structureSize; j <= structureSize; j++)
			{
				if (x + j >= 0 && x + j < width && y + i >= 0 && y + i < height)
				{
					float depth = d_input[(y + i)*width + (x + j)];
					if (depth == MINF || depth < 0.1f || fabs(depth - oldDepth) > dThresh)
					{
						count++;
						//d_output[y*width+x] = MINF;
						//return;
					}
				}
			}
		}

		unsigned int sum = (2 * structureSize + 1)*(2 * structureSize + 1);
		if ((float)count / (float)sum >= fracReq) {
			d_output[y*width + x] = 0;
		}
		else {
			d_output[y*width + x] = d_input[y*width + x];
		}
		// printf("x=%d, y=%d, d_output=%f\n",x,y,d_output[y*width + x]);
	}
}

void erodeDepthMap(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height, float dThresh, float fracReq, float zfar)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	erodeDepthMapDevice << <gridSize, blockSize >> >(d_output, d_input, structureSize, width, height, dThresh, fracReq, zfar);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gauss Filter Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gaussFilterDepthMapDevice(float* d_output, const float* d_input, int radius, float sigmaD, float sigmaR, unsigned int width, unsigned int height, const float zfar)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int kernelRadius = radius;

	d_output[y*width + x] = 0;

	const float depthCenter = d_input[y*width + x];
	// if (depthCenter>0.1)
	// {
	// 	d_output[y*width + x] = depthCenter;
	// 	return;
	// }

	float mean_depth = 0;
	int num_valid = 0;
	for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
	{
		for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float currentDepth = d_input[n*width + m];

				if (currentDepth>=0.1f && currentDepth<=zfar)
				{
					num_valid++;
					mean_depth += currentDepth;
				}
			}
		}
	}
	if (num_valid==0) return;

	mean_depth /= num_valid;

	float sum = 0.0f;
	float sumWeight = 0.0f;
	for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
	{
		for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float currentDepth = d_input[n*width + m];

				if (currentDepth>=0.1f && currentDepth<=zfar && abs(currentDepth-mean_depth)<0.01)
				{
					const float weight = exp( -((m-x)*(m-x) + (y-n)*(y-n)) / (2.0f*sigmaD*sigmaD) - (depthCenter-currentDepth)*(depthCenter-currentDepth)/(2*sigmaR*sigmaR) );

					sumWeight += weight;
					sum += weight*currentDepth;
				}
			}
		}
	}

	float num_total = (2*kernelRadius+1)*(2*kernelRadius+1);
	if (sumWeight > 0.0f && num_valid/num_total>0)
	{
		d_output[y*width + x] = sum / sumWeight;
	}
}

void gaussFilterDepthMap(float* d_output, const float* d_input, int radius, float sigmaD, float sigmaR, unsigned int width, unsigned int height, const float zfar)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	gaussFilterDepthMapDevice <<<gridSize, blockSize >>>(d_output, d_input, radius, sigmaD, sigmaR, width, height, zfar);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


__global__ void gaussFilterIntensityDevice(float* d_output, const float* d_input, float sigmaD, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	//d_output[y*width + x] = MINF;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	//const float center = d_input[y*width + x];
	//if (center != MINF) {
	for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
	{
		for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float current = d_input[n*width + m];

				//if (current != MINF && fabs(center - current) < sigmaR) {
				const float weight = gaussD(sigmaD, m - x, n - y);

				sumWeight += weight;
				sum += weight*current;
				//}
			}
		}
	}
	//}

	if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;
}

void gaussFilterIntensity(float* d_output, const float* d_input, float sigmaD, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	gaussFilterIntensityDevice << <gridSize, blockSize >> >(d_output, d_input, sigmaD, width, height);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// adaptive gauss filter float map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void adaptiveGaussFilterDepthMap_Kernel(float* d_output, const float* d_input, float sigmaD, float sigmaR,
	unsigned int width, unsigned int height, float adaptFactor)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;


	d_output[y*width + x] = MINF;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float depthCenter = d_input[y*width + x];
	if (depthCenter != MINF)
	{
		const float curSigma = sigmaD / depthCenter * adaptFactor;
		const int kernelRadius = (int)ceil(2.0*curSigma);

		for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
		{
			for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
			{
				if (m >= 0 && n >= 0 && m < width && n < height)
				{
					const float currentDepth = d_input[n*width + m];

					if (currentDepth != MINF && fabs(depthCenter - currentDepth) < sigmaR)
					{
						const float weight = gaussD(curSigma, m - x, n - y);

						sumWeight += weight;
						sum += weight*currentDepth;
					}
				}
			}
		}
	}

	if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;
}
void adaptiveGaussFilterDepthMap(float* d_output, const float* d_input, float sigmaD, float sigmaR, float adaptFactor, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	adaptiveGaussFilterDepthMap_Kernel << <gridSize, blockSize >> >(d_output, d_input, sigmaD, sigmaR, width, height, adaptFactor);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void adaptiveGaussFilterIntensity_Kernel(float* d_output, const float* d_input, const float* d_depth, float sigmaD,
	unsigned int width, unsigned int height, float adaptFactor)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	d_output[y*width + x] = MINF; //(should not be used in the case of no valid depth)

	const float depthCenter = d_depth[y*width + x];
	if (depthCenter != MINF)
	{
		const float curSigma = sigmaD / depthCenter * adaptFactor;
		const int kernelRadius = (int)ceil(2.0*curSigma);

		for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
		{
			for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
			{
				if (m >= 0 && n >= 0 && m < width && n < height)
				{
					const float currentDepth = d_depth[n*width + m];
					if (currentDepth != MINF) // && fabs(depthCenter - currentDepth) < sigmaR)
					{
						const float current = d_input[n*width + m];
						const float weight = gaussD(curSigma, m - x, n - y);

						sumWeight += weight;
						sum += weight*current;
					}
				}
			}
		}
	}

	if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;
}

void adaptiveGaussFilterIntensity(float* d_output, const float* d_input, const float* d_depth, float sigmaD, float adaptFactor, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	adaptiveGaussFilterIntensity_Kernel << <gridSize, blockSize >> >(d_output, d_input, d_depth, sigmaD, width, height, adaptFactor);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


__global__ void filterDepthSmoothedEdgesDevice(float* d_output, const float* d_input, const float4* d_normal, unsigned int width, unsigned int height, const float angle_thres, const float fx, const float fy, const float cx, const float cy)
{
	const int u = blockIdx.x*blockDim.x + threadIdx.x;
	const int v = blockIdx.y*blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	const int pos = v*width+u;
	float Z = d_input[pos];
	if (Z<0.1) return;

	float X = (u-cx)*Z/fx;
	float Y = (v-cy)*Z/fy;
	float3 view_dir = make_float3(X,Y,Z);
	view_dir = normalize(view_dir);

	float3 normal_dir = make_float3(d_normal[pos].x,d_normal[pos].y,d_normal[pos].z);
	normal_dir = normalize(normal_dir);
	float dot = normal_dir.x*view_dir.x + normal_dir.y*view_dir.y + normal_dir.z*view_dir.z;
	float angle = acos(dot);    // [0,pi]
	if (abs(angle-M_PI/2)<angle_thres)
	{
		d_output[pos] = 0;
	}
	else
	{
		d_output[pos] = d_input[pos];
	}

}

void filterDepthSmoothedEdges(float* d_output, const float* d_input, const float4* d_normal, unsigned int width, unsigned int height, const float angle_thres, const float fx, const float fy, const float cx, const float cy)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	filterDepthSmoothedEdgesDevice << <gridSize, blockSize >> >(d_output, d_input, d_normal, width, height, angle_thres, fx,fy,cx,cy);
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


__global__ void countNumValidDepth_Kernel(int *d_cnt, const float* d_input, const int height, const int width)
{
	const int w = blockIdx.x*blockDim.x + threadIdx.x;
	const int h = blockIdx.y*blockDim.y + threadIdx.y;

	if (w >= width || h >= height) return;

	if (d_input[h*width+w]>=0.1)
	{
		atomicAdd(d_cnt, 1);
	}
}

void countNumValidDepth(int *d_cnt, const float *d_input, const int height, const int width)
{
	const dim3 grid((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 block(T_PER_BLOCK, T_PER_BLOCK);
	countNumValidDepth_Kernel<<<grid,block>>>(d_cnt, d_input, height, width);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


__global__ void computeCovisibilityKernel(const int H, const int W, const int stride, Eigen::Matrix4f *cur_in_kfcam, const float visible_angle_thres, const float4 *xyz_mapA, const float4 *normalA, int *n_visible, int *n_total_gpu)
{
	const int w = (blockIdx.x*blockDim.x + threadIdx.x) * stride;
	const int h = (blockIdx.y*blockDim.y + threadIdx.y) * stride;
	if (w >= W || h >= H) return;

	const int i_pix = h*W+w;

	float4 ptA = xyz_mapA[i_pix];
	if (ptA.z<0.1) return;
	float4 normalA_tmp = normalA[i_pix];
	if (normalA_tmp.x==0 && normalA_tmp.y==0 && normalA_tmp.z==0) return;

	Eigen::Vector3f ptA_ = (*cur_in_kfcam * Eigen::Vector4f(ptA.x, ptA.y, ptA.z, 1)).head(3);
	Eigen::Vector3f normalA_ = (*cur_in_kfcam).block(0,0,3,3) * Eigen::Vector3f(normalA_tmp.x, normalA_tmp.y, normalA_tmp.z);
	Eigen::Vector3f pt_to_eye = -ptA_;
	float dot_prod = pt_to_eye.normalized().dot(normalA_.normalized());

	atomicAdd(n_total_gpu, 1);

	if (dot_prod>visible_angle_thres)
	{
		atomicAdd(n_visible, 1);
	}

}



/**
 * @brief
 *
 * @param fA later than fB
 * @param fB
 * @return __inline__
 */
float computeCovisibility(const int H, const int W, int umin, int vmin, int umax, int vmax, const Eigen::Matrix3f &K, const Eigen::Matrix4f &cur_in_kfcam, const float visible_angle_thres, const float4 *normalA, const float *depthA)
{
  const int n_pixels = H*W;

  float4 *xyz_map_gpu;
  cudaMalloc(&xyz_map_gpu, n_pixels*sizeof(float4));
	cudaMemset(xyz_map_gpu, 0, n_pixels*sizeof(float4));
  float4x4 K_inv_data;
  K_inv_data.setIdentity();
  Eigen::Matrix3f K_inv = K.inverse();
  for (int row=0;row<3;row++)
  {
    for (int col=0;col<3;col++)
    {
      K_inv_data(row,col) = K_inv(row,col);
    }
  }
  CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(xyz_map_gpu, depthA, K_inv_data, W, H);

	//////////!DEBUG
	// int *n_valid;
	// int n_valid_cpu = 0;
  // cudaMalloc(&n_valid, sizeof(int));
	// cudaMemset(n_valid, 0, sizeof(int));
	// countNumValidDepth(n_valid, depthA, H, W);
  // cutilSafeCall(cudaMemcpy(&n_valid_cpu, n_valid, sizeof(int), cudaMemcpyDeviceToHost));
	// std::cout<<"n_valid_cpu: "<<n_valid_cpu<<std::endl;

	Eigen::Matrix4f *cur_in_kfcam_gpu;
	cudaMalloc(&cur_in_kfcam_gpu, sizeof(Eigen::Matrix4f));
	cudaMemcpy(cur_in_kfcam_gpu, &cur_in_kfcam, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);

  int *n_visible_gpu, *n_total_gpu;
  cudaMalloc(&n_visible_gpu, sizeof(int));
	cudaMemset(n_visible_gpu, 0, sizeof(int));
	cudaMalloc(&n_total_gpu, sizeof(int));
	cudaMemset(n_total_gpu, 0, sizeof(int));
	const int stride = 2;
  dim3 threads = {32, 32};
  dim3 blocks = {divCeil(int(W/stride), threads.x), divCeil(int(H/stride), threads.y)};
  CUDAImageUtil::computeCovisibilityKernel<<<blocks, threads>>>(H, W, stride, cur_in_kfcam_gpu, visible_angle_thres, xyz_map_gpu, normalA, n_visible_gpu, n_total_gpu);
  int n_visible = 0, n_total = 0;
  cutilSafeCall(cudaMemcpy(&n_visible, n_visible_gpu, sizeof(int), cudaMemcpyDeviceToHost));
  cutilSafeCall(cudaMemcpy(&n_total, n_total_gpu, sizeof(int), cudaMemcpyDeviceToHost));

  float visible = float(n_visible)/n_total;

	cutilSafeCall(cudaFree(xyz_map_gpu));
	cutilSafeCall(cudaFree(n_visible_gpu));
	cutilSafeCall(cudaFree(n_total_gpu));
	cutilSafeCall(cudaFree(cur_in_kfcam_gpu));

  return visible;
}


};