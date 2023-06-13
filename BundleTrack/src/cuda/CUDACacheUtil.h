#pragma once
#ifndef CUDA_CACHE_UTIL
#define CUDA_CACHE_UTIL

#include <cuda.h>
#include <cuda_runtime.h>


struct CUDACachedFrame {
	void alloc(unsigned int width, unsigned int height) {
		cudaMalloc(&d_depthDownsampled, sizeof(float) * width * height);
		//cudaMalloc(&d_colorDownsampled, sizeof(uchar4) * width * height));
		cudaMalloc(&d_cameraposDownsampled, sizeof(float4) * width * height);
		cudaMalloc(&d_num_valid_points, sizeof(int));
		cudaMemset(d_num_valid_points, 0, sizeof(int));

		cudaMalloc(&d_intensityDownsampled, sizeof(float) * width * height);
		cudaMalloc(&d_intensityDerivsDownsampled, sizeof(float2) * width * height);
		cudaMalloc(&d_normalsDownsampled, sizeof(float4) * width * height);
	}
	void free() {
		cudaFree(d_depthDownsampled);
		cudaFree(d_num_valid_points);
		//cudaFree(d_colorDownsampled);
		cudaFree(d_cameraposDownsampled);

		cudaFree(d_intensityDownsampled);
		cudaFree(d_intensityDerivsDownsampled);
		cudaFree(d_normalsDownsampled);
	}

	int* d_num_valid_points;
	float* d_depthDownsampled;
	//uchar4* d_colorDownsampled;
	float4* d_cameraposDownsampled;

	//for dense color term
	float* d_intensityDownsampled; //this could be packed with intensityDerivaties to a float4 dunno about the read there
	float2* d_intensityDerivsDownsampled; //TODO could have energy over intensity gradient instead of intensity
	float4* d_normalsDownsampled;
};

#endif //CUDA_CACHE_UTIL