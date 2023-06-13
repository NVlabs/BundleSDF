#include "common.h"
#include "CUDACache.h"


CUDACache::CUDACache(unsigned int widthDepthInput, unsigned int heightDepthInput, unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const Eigen::Matrix4f& inputIntrinsics)
{
	m_width = widthDownSampled;
	m_height = heightDownSampled;
	m_maxNumImages = maxNumImages;

	m_intrinsics = inputIntrinsics;
	// std::cout<<"m_intrinsics:\n"<<m_intrinsics<<std::endl;
	m_intrinsics(0,0) *= (float)widthDownSampled / (float)widthDepthInput;
	m_intrinsics(1,1) *= (float)heightDownSampled / (float)heightDepthInput;
	m_intrinsics(0,2) *= (float)(widthDownSampled -1)/ (float)(widthDepthInput-1);
	m_intrinsics(1,2) *= (float)(heightDownSampled-1) / (float)(heightDepthInput-1);
	m_intrinsicsInv = m_intrinsics.inverse();

	// std::cout<<"Accounting for downsample, m_intrinsics:\n"<<m_intrinsics<<std::endl;

	m_filterIntensitySigma = 2.5;
	m_filterDepthSigmaD = 1.0;
	m_filterDepthSigmaR = 0.05;

	m_inputDepthWidth = widthDepthInput;
	m_inputDepthHeight = heightDepthInput;
	m_inputIntrinsics = inputIntrinsics;
	m_inputIntrinsicsInv = m_inputIntrinsics.inverse();

	alloc();
	m_currentFrame = 0;
}

CUDACache::~CUDACache()
{
	free();
}


void CUDACache::storeFrame(unsigned int inputDepthWidth, unsigned int inputDepthHeight, const float* d_depth, const uchar4* d_color, const float4 *d_normals)
{
	CUDACachedFrame& frame = m_cache[m_currentFrame];

	// const float* d_inputDepth = d_depth;
	// if (m_filterDepthSigmaD > 0.0f) {
	// 	CUDAImageUtil::gaussFilterDepthMap(d_filterHelper, d_depth, m_filterDepthSigmaD, m_filterDepthSigmaR, inputDepthWidth, inputDepthHeight);
	// 	d_inputDepth = d_filterHelper;
	// }
	CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(d_helperCamPos, d_depth, *(float4x4*)&m_inputIntrinsicsInv, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat4(frame.d_cameraposDownsampled, m_width, m_height, d_helperCamPos, inputDepthWidth, inputDepthHeight);

	// CUDAImageUtil::computeNormals(d_helperNormals, d_helperCamPos, inputDepthWidth, inputDepthHeight);
	// cudaMemcpy(d_helperNormals, d_normals, sizeof(float4)*inputDepthWidth*inputDepthHeight, cudaMemcpyDeviceToDevice);
	// CUDAImageUtil::resampleFloat4(frame.d_normalsDownsampled, m_width, m_height, d_helperNormals, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat4(frame.d_normalsDownsampled, m_width, m_height, d_normals, inputDepthWidth, inputDepthHeight);
	//CUDAImageUtil::convertNormalsFloat4ToUCHAR4(frame.d_normalsDownsampledUCHAR4, frame.d_normalsDownsampled, m_width, m_height);

	CUDAImageUtil::resampleFloat(frame.d_depthDownsampled, m_width, m_height, d_depth, inputDepthWidth, inputDepthHeight);

	CUDAImageUtil::countNumValidDepth(frame.d_num_valid_points, frame.d_depthDownsampled, m_height, m_width);

	// std::vector<int> tmp(1);
	// cudaMemcpy(tmp.data(), frame.d_num_valid_points, sizeof(int), cudaMemcpyDeviceToHost);
	// std::cout<<"frame num valid points "<<tmp[0]<<std::endl;

	// std::vector<float> tmp(m_width*m_height);
	// cudaMemcpy(tmp.data(), frame.d_depthDownsampled, sizeof(float)*tmp.size(), cudaMemcpyDeviceToHost);
	// std::cout<<"tmp data: ";
	// for (int i=0;i<tmp.size();i++)
	// {
	// 	std::cout<<tmp[i]<<" ";
	// }
	// std::cout<<std::endl;

	//CUDAImageUtil::resampleUCHAR4(frame.d_colorDownsampled, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	//CUDAImageUtil::jointBilateralFilterFloatMap(frame.d_colorDownsampled)

	//color
	// CUDAImageUtil::resampleToIntensity(d_intensityHelper, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	// if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::gaussFilterIntensity(frame.d_intensityDownsampled, d_intensityHelper, m_filterIntensitySigma, m_width, m_height);
	//if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::jointBilateralFilterFloat(frame.d_intensityDownsampled, d_intensityHelper, frame.d_depthDownsampled, m_intensityFilterSigma, 0.01f, m_width, m_height);
	//if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::adaptiveBilateralFilterIntensity(frame.d_intensityDownsampled, d_intensityHelper, frame.d_depthDownsampled, m_filterIntensitySigma, 0.01f, 1.0f, m_width, m_height);
	// else std::swap(frame.d_intensityDownsampled, d_intensityHelper);
	// CUDAImageUtil::computeIntensityDerivatives(frame.d_intensityDerivsDownsampled, frame.d_intensityDownsampled, m_width, m_height);

	m_currentFrame++;
}
