#pragma once

#include "CUDACacheUtil.h"
#include "CUDAImageUtil.h"
#include <bits/stdc++.h>

class CUDACache {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	CUDACache(unsigned int widthDepthInput, unsigned int heightDepthInput, unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const Eigen::Matrix4f& inputIntrinsics);
	~CUDACache();
	void alloc();
	void free();

	void storeFrame(unsigned int inputDepthWidth, unsigned int inputDepthHeight, const float* d_depth, const uchar4* d_color, const float4 *d_normals);

	void reset() {
		m_currentFrame = 0;
	}

	const std::vector<CUDACachedFrame>& getCacheFrames() const { return m_cache; }
	const CUDACachedFrame* getCacheFramesGPU() const { return d_cache; }

	void copyCacheFrameFrom(CUDACache* other, unsigned int frameFrom);
	//! for invalid (global) frames don't need to copy
	void incrementCache() {
		m_currentFrame++;
	}

	unsigned int getWidth() const { return m_width; }
	unsigned int getHeight() const { return m_height; }

	const Eigen::Matrix4f& getIntrinsics() const { return m_intrinsics; }
	const Eigen::Matrix4f& getIntrinsicsInv() const { return m_intrinsicsInv; }

	unsigned int getNumFrames() const { return m_currentFrame; }

	//!debugging only
	std::vector<CUDACachedFrame>& getCachedFramesDEBUG() { return m_cache; }
	void setCurrentFrame(unsigned int c) { m_currentFrame = c; }
	void setIntrinsics(const Eigen::Matrix4f& inputIntrinsics, const Eigen::Matrix4f& intrinsics) {
		m_inputIntrinsics = inputIntrinsics; m_inputIntrinsicsInv = inputIntrinsics.inverse();
		m_intrinsics = intrinsics; m_intrinsicsInv = intrinsics.inverse();
	}

private:
	unsigned int m_width;
	unsigned int m_height;
	Eigen::Matrix4f		 m_intrinsics;
	Eigen::Matrix4f		 m_intrinsicsInv;

	unsigned int m_currentFrame;
	unsigned int m_maxNumImages;

	std::vector < CUDACachedFrame > m_cache;
	CUDACachedFrame*				d_cache;

	//for hi-res compute
	float* d_filterHelper;
	float4* d_helperCamPos, *d_helperNormals; //TODO ANGIE
	unsigned int m_inputDepthWidth;
	unsigned int m_inputDepthHeight;
	Eigen::Matrix4f		 m_inputIntrinsics;
	Eigen::Matrix4f		 m_inputIntrinsicsInv;

	float* d_intensityHelper;
	float m_filterIntensitySigma;
	float m_filterDepthSigmaD;
	float m_filterDepthSigmaR;
};
