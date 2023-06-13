
#include "SIFTImageManager.h"
#include "cudaUtil.h"
#include "CUDATimer.h"
#include "common.h"

#define SORT_NUM_BLOCK_THREADS_X (MAX_MATCHES_PER_IMAGE_PAIR_RAW / 2)


__device__ bool cmpAndSawp(
	volatile float* dist0,
	volatile uint2* idx0,
	volatile float* dist1,
	volatile uint2* idx1
	)
{
	if (dist0[0] > dist1[0]) {
		float tmpDist = dist0[0];
		dist0[0] = dist1[0];
		dist1[0] = tmpDist;

		const unsigned int tmpIdxX = idx0[0].x;
		idx0[0].x = idx1[0].x;
		idx1[0].x = tmpIdxX;

		const unsigned int tmpIdxY = idx0[0].y;
		idx0[0].y = idx1[0].y;
		idx1[0].y = tmpIdxY;
		return true;
	}
	else {
		return false;
	}
}

#define FILTER_NUM_BLOCK_THREADS_X MAX_MATCHES_PER_IMAGE_PAIR_RAW
#define FILTER_DENSE_VERIFY_THREAD_SPLIT 32

__device__ float3 computeProjError(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, float colorThresh, const float4x4& transform, const float4x4& intrinsics,
	const float* d_inputDepth, const float4* d_inputCamPos, const float4* d_inputNormal, const float* d_inputColor,
	const float* d_modelDepth, const float4* d_modelCamPos, const float4* d_modelNormal, const float* d_modelColor,
	float sensorDepthMin, float sensorDepthMax)
{
	float3 out = make_float3(0.0f);

	float4 pInput = d_inputCamPos[idx]; // point
	float4 nInput = d_inputNormal[idx]; nInput.w = 0.0f; // vector
	float dInput = d_inputDepth[idx];
	//float cInput = d_inputColor[idx];

	if (pInput.x != MINF && nInput.x != MINF && dInput >= sensorDepthMin && dInput <= sensorDepthMax) {
		const float4 pTransInput = transform * pInput;
		const float4 nTransInput = transform * nInput;

		float3 tmp = intrinsics * make_float3(pTransInput.x, pTransInput.y, pTransInput.z);
		const int2 screenPos = make_int2((int)roundf(tmp.x / tmp.z), (int)roundf(tmp.y / tmp.z)); // subsampled space

		if (screenPos.x >= 0 && screenPos.y >= 0 && screenPos.x < (int)imageWidth && screenPos.y < (int)imageHeight) {
			float4 pTarget = d_modelCamPos[screenPos.y * imageWidth + screenPos.x]; //getBestCorrespondence1x1
			//float cTarget = d_modelColor[screenPos.y * imageWidth + screenPos.x];
			float4 nTarget = d_modelNormal[screenPos.y * imageWidth + screenPos.x];
			if (pTarget.x != MINF && nTarget.x != MINF) {
				float d = length(pTransInput - pTarget);
				float dNormal = dot(make_float3(nTransInput.x, nTransInput.y, nTransInput.z), make_float3(nTarget.x, nTarget.y, nTarget.z)); // should be able to do dot(nTransInput, nTarget)
				//float c = length(cInput - cTarget);

				//float projInputDepth = (intrinsics * make_float3(pTransInput.x, pTransInput.y, pTransInput.z)).z;
				float projInputDepth = pTransInput.z;
				float tgtDepth = d_modelDepth[screenPos.y * imageWidth + screenPos.x];

				if (tgtDepth >= sensorDepthMin && tgtDepth <= sensorDepthMax) {
					bool b = ((tgtDepth != MINF && projInputDepth < tgtDepth) && d > distThresh); // bad matches that are known
					if ((dNormal >= normalThresh && d <= distThresh /*&& c <= colorThresh*/) || b) { // if normal/pos/color correspond or known bad match

						const float cameraToKinectProjZ = (pTransInput.z - sensorDepthMin) / (sensorDepthMax - sensorDepthMin);
						const float weight = max(0.0f, 0.5f*((1.0f - d / distThresh) + (1.0f - cameraToKinectProjZ))); // for weighted ICP;

						out.x = d;	//residual
						out.y = weight;							//corr weight
						out.z = 1.0f;
					}
				} // target depth within sensor min/max
			} // projected to valid depth
		} // inside image
	}

	return out;
}


void __global__ AddCurrToResidualsCU_Kernel(
	unsigned int curFrame,
	unsigned int startFrame,
	EntryJ* d_globMatches,
	uint2* d_globMatchesKeyPointIndices,
	int* d_globNumImagePairs,
	const int* d_currNumFilteredMatchesPerImagePair,
	const uint2* d_currFilteredMatchKeyPointIndices,
	const SIFTKeyPoint* d_keyPoints,
	const unsigned int maxKeyPointsPerImage,
	const float4x4 colorIntrinsicsInv
	)
{
	const unsigned int imagePairIdx = blockIdx.x + startFrame;
	if (imagePairIdx == curFrame) return;
	const unsigned int tidx = threadIdx.x;
	const unsigned int numMatches = d_currNumFilteredMatchesPerImagePair[imagePairIdx];
	__shared__ unsigned int basePtr;
	if (tidx == 0) {
		basePtr = atomicAdd(&d_globNumImagePairs[0], numMatches);
	}
	__syncthreads();

	//if (tidx == 0) {
	//	printf("[%d] baseAddr=%d\n", imagePairIdx, basePtr);
	//}

	if (tidx < numMatches) {
		const unsigned int srcAddr = imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + tidx;

		uint2 currFilteredMachtKeyPointIndices = d_currFilteredMatchKeyPointIndices[srcAddr];

		//printf("[%d] = [%d %d]\n", imagePairIdx, currFilteredMachtKeyPointIndices.x, currFilteredMachtKeyPointIndices.y);

		const SIFTKeyPoint& k_i = d_keyPoints[currFilteredMachtKeyPointIndices.x];
		const SIFTKeyPoint& k_j = d_keyPoints[currFilteredMachtKeyPointIndices.y];

		EntryJ e;
		const unsigned int imageIdx0 = imagePairIdx;
		const unsigned int imageIdx1 = curFrame;
		e.imgIdx_i = imageIdx0;
		e.imgIdx_j = imageIdx1;
		e.pos_i = colorIntrinsicsInv * (k_i.depth * make_float3(k_i.pos.x, k_i.pos.y, 1.0f));
		e.pos_j = colorIntrinsicsInv * (k_j.depth * make_float3(k_j.pos.x, k_j.pos.y, 1.0f));

		d_globMatches[basePtr + tidx] = e;
		d_globMatchesKeyPointIndices[basePtr + tidx] = currFilteredMachtKeyPointIndices;
	}
}

void SIFTImageManager::AddCurrToResidualsCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, const float4x4& colorIntrinsicsInv) {
	if (numFrames == 0) return;

	dim3 grid(numFrames - startFrame);
	const unsigned int threadsPerBlock = ((MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + 31) / 32) * 32;
	dim3 block(threadsPerBlock);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	AddCurrToResidualsCU_Kernel << <grid, block >> >(
		curFrame,
		startFrame,
		d_globMatches,
		d_globMatchesKeyPointIndices,
		d_globNumResiduals,
		d_currNumFilteredMatchesPerImagePair,
		d_currFilteredMatchKeyPointIndices,
		d_keyPoints,
		m_maxKeyPointsPerImage,
		colorIntrinsicsInv
		);

	cutilSafeCall(cudaMemcpy(&m_globNumResiduals, d_globNumResiduals, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	if (m_timer) m_timer->endEvent();
}


#define INVALIDATEIMAGE_TO_IMAGE_KERNEL_THREADS_X 128

void __global__ InvalidateImageToImageCU_Kernel(EntryJ* d_globMatches, unsigned int globNumResiduals, uint2 imageToImageIdx)
{
	const unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if (idx < globNumResiduals) {
		if (d_globMatches[idx].imgIdx_i == imageToImageIdx.x &&
			d_globMatches[idx].imgIdx_j == imageToImageIdx.y) {
			d_globMatches[idx].setInvalid();
		}

	}

}

void SIFTImageManager::InvalidateImageToImageCU(const uint2& imageToImageIdx) {

	const unsigned int threadsPerBlock = INVALIDATEIMAGE_TO_IMAGE_KERNEL_THREADS_X;
	dim3 grid((m_globNumResiduals + threadsPerBlock - 1) / threadsPerBlock);
	dim3 block(threadsPerBlock);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	InvalidateImageToImageCU_Kernel << <grid, block >> >(d_globMatches, m_globNumResiduals, imageToImageIdx);

	if (m_timer) m_timer->endEvent();
}


#define CHECK_FOR_INVALID_FRAMES_X 128
#define CHECK_FOR_INVALID_FRAMES_THREADS_X 16

void __global__ CheckForInvalidFramesCU_Kernel(const int* d_varToCorrNumEntriesPerRow, int* d_validImages, unsigned int numVars,
	EntryJ* d_globMatches, unsigned int numGlobResiduals)
{
	const unsigned int resIdx = blockDim.x*blockIdx.x + blockIdx.y;
	const unsigned int varIdx = gridDim.x*threadIdx.x + threadIdx.y;

	if (varIdx < numVars && resIdx < numGlobResiduals) {
		if (d_varToCorrNumEntriesPerRow[varIdx] == 0) { // no connections!
			if (d_globMatches[resIdx].isValid() && (d_globMatches[resIdx].imgIdx_i == varIdx || d_globMatches[resIdx].imgIdx_j == varIdx)) { // invalidate residuals
				d_globMatches[resIdx].setInvalid();
			}
			if (d_validImages[varIdx] != 0) {
				if (varIdx == 0) printf("ERROR ERROR INVALIDATING THE FIRST FRAME\n");
				//printf("[CheckForInvalidFramesCU] invalidating frame %d\n", varIdx);
				d_validImages[varIdx] = 0;
			}
		}
	}

}
//TODO CHECK grid/block dim (too many threads?)
void SIFTImageManager::CheckForInvalidFramesCU(const int* d_varToCorrNumEntriesPerRow, unsigned int numVars)
{
	dim3 block((m_globNumResiduals + CHECK_FOR_INVALID_FRAMES_X - 1) / CHECK_FOR_INVALID_FRAMES_X, CHECK_FOR_INVALID_FRAMES_X);
	dim3 threadsPerBlock((numVars + CHECK_FOR_INVALID_FRAMES_THREADS_X - 1) / CHECK_FOR_INVALID_FRAMES_THREADS_X, CHECK_FOR_INVALID_FRAMES_THREADS_X);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	cutilSafeCall(cudaMemcpy(d_validImages, m_validImages.data(), sizeof(int) * numVars, cudaMemcpyHostToDevice));

	CheckForInvalidFramesCU_Kernel << <block, threadsPerBlock >> >(d_varToCorrNumEntriesPerRow, d_validImages, numVars, d_globMatches, m_globNumResiduals);

	cutilSafeCall(cudaMemcpy(m_validImages.data(), d_validImages, sizeof(int) * numVars, cudaMemcpyDeviceToHost));

	if (m_timer) m_timer->endEvent();
}

void __global__ CheckForInvalidFramesSimpleCU_Kernel(const int* d_varToCorrNumEntriesPerRow, int* d_validImages, unsigned int numVars)
{
	const unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if (idx < numVars) {
		if (d_varToCorrNumEntriesPerRow[idx] == 0) { // no connections!
			//printf("[CheckForInvalidFramesCU] invalidating frame %d\n", idx);
			d_validImages[idx] = 0;
		}
	}
}
//TODO CHECK grid/block dim (too many threads?)
void SIFTImageManager::CheckForInvalidFramesSimpleCU(const int* d_varToCorrNumEntriesPerRow, unsigned int numVars)
{
	const unsigned int threadsPerBlock = CHECK_FOR_INVALID_FRAMES_THREADS_X;
	dim3 grid((numVars + threadsPerBlock - 1) / threadsPerBlock);
	dim3 block(threadsPerBlock);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	cutilSafeCall(cudaMemcpy(d_validImages, m_validImages.data(), sizeof(int) * numVars, cudaMemcpyHostToDevice));

	CheckForInvalidFramesSimpleCU_Kernel << <grid, block >> >(d_varToCorrNumEntriesPerRow, d_validImages, numVars);

	cutilSafeCall(cudaMemcpy(m_validImages.data(), d_validImages, sizeof(int) * numVars, cudaMemcpyDeviceToHost));

	if (m_timer) m_timer->endEvent();
}



void SIFTImageManager::alloc()
{
	m_numKeyPoints = 0;

	cutilSafeCall(cudaMalloc(&d_keyPoints, sizeof(SIFTKeyPoint)*m_maxNumImages*m_maxKeyPointsPerImage));
	cutilSafeCall(cudaMalloc(&d_keyPointDescs, sizeof(SIFTKeyPointDesc)*m_maxNumImages*m_maxKeyPointsPerImage));
	//cutilSafeCall(cudaMalloc(&d_keyPointCounters, sizeof(int)*m_maxNumImages));
	//cutilSafeCall(cudaMemset(d_keyPointCounters, 0, sizeof(int)*m_maxNumImages));

	// matching
	m_currImagePairMatches.resize(m_maxNumImages);

	const unsigned maxImageMatches = m_maxNumImages;
	cutilSafeCall(cudaMalloc(&d_currNumMatchesPerImagePair, sizeof(int)*maxImageMatches));
	cutilSafeCall(cudaMalloc(&d_currMatchDistances, sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));
	cutilSafeCall(cudaMalloc(&d_currMatchKeyPointIndices, sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));

	cutilSafeCall(cudaMalloc(&d_currNumFilteredMatchesPerImagePair, sizeof(int)*maxImageMatches));
	cutilSafeCall(cudaMalloc(&d_currFilteredMatchDistances, sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	cutilSafeCall(cudaMalloc(&d_currFilteredMatchKeyPointIndices, sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	cutilSafeCall(cudaMalloc(&d_currFilteredTransforms, sizeof(float4x4)*maxImageMatches));
	cutilSafeCall(cudaMalloc(&d_currFilteredTransformsInv, sizeof(float4x4)*maxImageMatches));

	m_validImages.resize(m_maxNumImages, 0);
	m_validImages[0] = 1; // first is valid
	cutilSafeCall(cudaMalloc(&d_validImages, sizeof(int) *  m_maxNumImages));
	cutilSafeCall(cudaMemcpy(d_validImages, m_validImages.data(), sizeof(int), cudaMemcpyHostToDevice)); // first is valid

	const unsigned int maxResiduals = MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * (m_maxNumImages*(m_maxNumImages - 1)) / 2;
	m_globNumResiduals = 0;
	cutilSafeCall(cudaMalloc(&d_globNumResiduals, sizeof(int)));
	cutilSafeCall(cudaMemset(d_globNumResiduals, 0, sizeof(int)));

	cutilSafeCall(cudaMalloc(&d_globMatches, sizeof(EntryJ)*maxResiduals));
	cutilSafeCall(cudaMalloc(&d_globMatchesKeyPointIndices, sizeof(uint2)*maxResiduals));

	cutilSafeCall(cudaMalloc(&d_validOpt, sizeof(int)));

	initializeMatching();
}


void SIFTImageManager::free()
{
	m_SIFTImagesGPU.clear();

	m_numKeyPoints = 0;
	m_numKeyPointsPerImage.clear();
	m_numKeyPointsPerImagePrefixSum.clear();

	cutilSafeCall(cudaFree(d_keyPoints));
	cutilSafeCall(cudaFree(d_keyPointDescs));
	//cutilSafeCall(cudaFree(cudaFree(d_keyPointCounters));

	m_currImagePairMatches.clear();

	cutilSafeCall(cudaFree(d_currNumMatchesPerImagePair));
	cutilSafeCall(cudaFree(d_currMatchDistances));
	cutilSafeCall(cudaFree(d_currMatchKeyPointIndices));

	cutilSafeCall(cudaFree(d_currNumFilteredMatchesPerImagePair));
	cutilSafeCall(cudaFree(d_currFilteredMatchDistances));
	cutilSafeCall(cudaFree(d_currFilteredMatchKeyPointIndices));
	cutilSafeCall(cudaFree(d_currFilteredTransforms));
	cutilSafeCall(cudaFree(d_currFilteredTransformsInv));

	m_validImages.clear();
	cutilSafeCall(cudaFree(d_validImages));

	m_globNumResiduals = 0;
	cutilSafeCall(cudaFree(d_globNumResiduals));
	cutilSafeCall(cudaFree(d_globMatches));
	cutilSafeCall(cudaFree(d_globMatchesKeyPointIndices));

	cutilSafeCall(cudaFree(d_validOpt));

	//cutilSafeCall(cudaFree(d_fuseGlobalKeyCount);
	//cutilSafeCall(cudaFree(d_fuseGlobalKeyMarker);

	m_imagesToRetry.clear();

	SAFE_DELETE(m_timer);
}

