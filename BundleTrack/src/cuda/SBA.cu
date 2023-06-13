#include "common.h"
#include <math_constants.h>
#include "SBA.h"
#include "cudaUtil.h"
#include "Solver/LieDerivUtil.h"

#define THREADS_PER_BLOCK 512

__global__ void convertMatricesToPosesCU_Kernel(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans, const int* d_validImages)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numTransforms && d_validImages[idx]) {
		matrixToPose(d_transforms[idx], d_rot[idx], d_trans[idx]);
	}
}


extern "C" void convertMatricesToPosesCU(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans, const int* d_validImages)
{
	const unsigned int N = numTransforms;

	convertMatricesToPosesCU_Kernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_transforms, numTransforms, d_rot, d_trans, d_validImages);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



__global__ void convertPosesToMatricesCU_Kernel(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms, const int* d_validImages)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numImages && d_validImages[idx]) {
		poseToMatrix(d_rot[idx], d_trans[idx], d_transforms[idx]);
	}
}

extern "C" void convertPosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms, const int* d_validImages)
{
	const unsigned int N = numImages;

	convertPosesToMatricesCU_Kernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_rot, d_trans, numImages, d_transforms, d_validImages);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


SBA::~SBA()
{
	SAFE_DELETE(m_solver);
	cutilSafeCall(cudaFree(d_xRot));
	cutilSafeCall(cudaFree(d_xTrans));

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



void SBA::init(unsigned int maxImages, unsigned int maxNumResiduals, unsigned int max_corr_per_image, const std::vector<int> &update_pose_flags)
{

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	unsigned int maxNumImages = maxImages;
	cutilSafeCall(cudaMalloc(&d_xRot, sizeof(float3)*maxNumImages));
	cutilSafeCall(cudaMalloc(&d_xTrans, sizeof(float3)*maxNumImages));

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	m_solver = new CUDASolverBundling(maxImages, maxNumResiduals,max_corr_per_image,update_pose_flags,yml);
	m_bVerify = false;

	m_bUseComprehensiveFrameInvalidation = false;
	m_bUseLocalDense = true;
}



bool SBA::align(const std::vector<EntryJ> &global_corres, const std::vector<int> &n_match_per_pair, int n_images, const CUDACache* cudaCache, float4x4* d_transforms, bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt, unsigned int revalidateIdx)
{
	if (recordConvergence) m_recordedConvergence.push_back(std::vector<float>());
	_global_corres = global_corres;
	_n_images = n_images;
	m_bVerify = false;
	m_maxResidual = -1.0f;

	//dense opt params
	bool usePairwise = true;
	const CUDACache* cache = cudaCache;
	std::vector<float> weightsDenseDepth, weightsDenseColor, weightsSparse;

	weightsSparse = m_localWeightsSparse;

	weightsDenseDepth = m_localWeightsDenseDepth; //turn on
	weightsDenseColor = m_localWeightsDenseColor;

	unsigned int numImages = n_images;
	std::vector<int> valid_images_cpu(n_images,1);
	cutilSafeCall(cudaMalloc(&d_validImages, n_images*sizeof(int)));
	cutilSafeCall(cudaMemcpy(d_validImages, valid_images_cpu.data(), sizeof(int)*n_images, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMalloc(&d_correspondences, sizeof(EntryJ)*_global_corres.size()));
	cudaMemcpy(d_correspondences, _global_corres.data(), sizeof(EntryJ)*_global_corres.size(), cudaMemcpyHostToDevice);

	//if (isStart) siftManager->updateGPUValidImages(); //should be in sync already
	// const int* d_validImages = siftManager->getValidImagesGPU();

	convertMatricesToPosesCU(d_transforms, numImages, d_xRot, d_xTrans, d_validImages);

	bool removed = alignCUDA(cache, usePairwise, weightsSparse, weightsDenseDepth, weightsDenseColor, isStart, isEnd, revalidateIdx);
	if (recordConvergence) {
		const std::vector<float>& conv = m_solver->getConvergenceAnalysis();
		m_recordedConvergence.back().insert(m_recordedConvergence.back().end(), conv.begin(), conv.end());
	}

	// if (useVerify) {
	// 	if (weightsSparse.front() > 0) m_bVerify = m_solver->useVerification(siftManager->getGlobalCorrespondencesGPU(), siftManager->getNumGlobalCorrespondences());
	// 	else m_bVerify = true; //TODO this should not happen except for debugging
	// }

	convertPosesToMatricesCU(d_xRot, d_xTrans, numImages, d_transforms, d_validImages);

	cutilSafeCall(cudaFree(d_validImages));
	cutilSafeCall(cudaFree(d_correspondences));

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	return removed;
}


SBA::SBA(int maxNumImages, int maxNumResiduals, int max_corr_per_image, const std::vector<int> &update_pose_flags, std::shared_ptr<YAML::Node> yml1)
{
	yml = yml1;
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	m_bUseComprehensiveFrameInvalidation = false;

	const unsigned int maxNumIts = (*yml)["bundle"]["num_iter_outter"].as<int>();
	m_localWeightsSparse.resize(maxNumIts, 1.0f);

	m_localWeightsDenseDepth.resize(maxNumIts,1);
	for (unsigned int i = 0; i < maxNumIts; i++)
	{
		// m_localWeightsDenseDepth[i] = (i + 10.0/maxNumIts);
	}
	m_localWeightsDenseColor.resize(maxNumIts, 0.0f);

	// m_globalWeightsSparse.resize(maxNumIts, 1.0f);
	// m_globalWeightsDenseDepth.resize(maxNumIts, 1.0f);
	// for (unsigned int i = 2; i < maxNumIts; i++)
	// 	m_globalWeightsDenseDepth[i] = (float)i;
	// m_globalWeightsDenseColor.resize(maxNumIts, 0.1f);

	m_maxResidual = -1.0f;

	m_bUseGlobalDenseOpt = true;
	m_bUseLocalDense = true;



	init(maxNumImages, maxNumResiduals, max_corr_per_image, update_pose_flags);
}


bool SBA::alignCUDA(const CUDACache* cudaCache, bool useDensePairwise, const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool isStart, bool isEnd, unsigned int revalidateIdx)
{
	// m_numCorrespondences = siftManager->getNumGlobalCorrespondences();
	m_numCorrespondences = _global_corres.size();
	// transforms
	unsigned int numImages = _n_images;
	auto begin = std::chrono::steady_clock::now();
	m_solver->solve(d_correspondences, m_numCorrespondences, d_validImages, numImages, cudaCache, weightsSparse, weightsDenseDepth, weightsDenseColor, useDensePairwise, d_xRot, d_xTrans, isStart, isEnd, revalidateIdx); //isStart -> rebuild jt, isEnd -> remove max residual
	auto end = std::chrono::steady_clock::now();
	std::cout << "m_solver->solve Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0 << "[ms]" << std::endl;
	bool removed = false;
	return removed;
}