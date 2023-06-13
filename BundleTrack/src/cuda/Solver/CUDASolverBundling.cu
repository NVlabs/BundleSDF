
// #include "stdafx.h"
#include "common.h"
#include "CUDASolverBundling.h"
#include "cudaUtil.h"
#include "../CUDACache.h"
#define THREADS_PER_BLOCK 512  // keep consistent with the GPU

extern "C" void evalMaxResidual(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer);
// extern "C" void buildVariablesToCorrespondencesTableCUDA(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow, CUDATimer* timer);
extern "C" void buildVariablesToCorrespondencesTableCUDA(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow, CUDATimer* timer, int *d_n_match_per_pair, const int n_images);
extern "C" void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, float* convergenceAnalysis, CUDATimer* timer);

extern "C" int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);

extern "C" void convertLiePosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numTransforms, float4x4* d_transforms, float4x4* d_transformInvs);
extern "C" void collectHighResiduals(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer);
extern "C" void VisualizeCorrespondences(const uint2& imageIndices, const SolverInput& input, SolverState& state, SolverParameters& parameters, float3* d_corrImage);

#ifdef DEBUG_PRINT_SPARSE_RESIDUALS
extern "C" float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);
#endif

CUDASolverBundling::CUDASolverBundling(unsigned int maxNumberOfImages, unsigned int maxNumResiduals, const int max_corr_per_image, const std::vector<int> &update_pose_flags, std::shared_ptr<YAML::Node> yml1)
	: m_maxNumberOfImages(maxNumberOfImages)
{
	yml = yml1;
	m_timer = NULL;
	//m_timer = new CUDATimer();
	//if (GlobalBundlingState::get().s_enableDetailedTimings) m_timer = new CUDATimer();
	m_bRecordConvergence = false;

	//TODO PARAMS
	// const unsigned int submapSize = GlobalBundlingState::get().s_submapSize;
	const unsigned int submapSize = 20;

	m_verifyOptDistThresh = 0.02f;//GlobalAppState::get().s_verifyOptDistThresh;
	m_verifyOptPercentThresh = 0.05f;//GlobalAppState::get().s_verifyOptPercentThresh;

	const unsigned int numberOfVariables = maxNumberOfImages;
	m_maxCorrPerImage = max_corr_per_image;

	printf("maxNumResiduals / maxNumberOfImages = %d / %d = %d\n", maxNumResiduals, maxNumberOfImages, int(maxNumResiduals/maxNumberOfImages));
	printf("m_maxNumberOfImages*m_maxCorrPerImage = %d x %d = %d\n", m_maxNumberOfImages, m_maxCorrPerImage, m_maxNumberOfImages*m_maxCorrPerImage);

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	// State
	cudaMalloc(&m_solverState.d_deltaRot, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_deltaTrans, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_rRot, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_rTrans, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_zRot, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_zTrans, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_pRot, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_pTrans, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_Jp, sizeof(float3)*maxNumResiduals);
	cudaMalloc(&m_solverState.d_Ap_XRot, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_Ap_XTrans, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_scanAlpha, sizeof(float) * 2);
	cudaMalloc(&m_solverState.d_rDotzOld, sizeof(float) *numberOfVariables);
	cudaMalloc(&m_solverState.d_precondionerRot, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_precondionerTrans, sizeof(float3)*numberOfVariables);
	cudaMalloc(&m_solverState.d_sumResidual, sizeof(float));
	unsigned int n = (maxNumResiduals + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	cudaMalloc(&m_solverExtra.d_maxResidual, sizeof(float) * n);
	cudaMalloc(&m_solverExtra.d_maxResidualIndex, sizeof(int) * n);
	m_solverExtra.h_maxResidual = new float[n];
	m_solverExtra.h_maxResidualIndex = new int[n];

	cudaMalloc(&d_variablesToCorrespondences, sizeof(int)*m_maxNumberOfImages*m_maxCorrPerImage);
	cudaMalloc(&d_numEntriesPerRow, sizeof(int)*m_maxNumberOfImages);
	cudaMalloc(&d_n_match_per_pair, sizeof(int)*m_maxNumberOfImages*m_maxNumberOfImages);
	cudaMalloc(&d_update_pose_flags, sizeof(int)*update_pose_flags.size());
	cudaMemcpy(d_update_pose_flags, update_pose_flags.data(), sizeof(int)*update_pose_flags.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&m_solverState.d_countHighResidual, sizeof(int));

	cudaMalloc(&m_solverState.d_denseJtJ, sizeof(float) * 36 * numberOfVariables * numberOfVariables);
	cudaMalloc(&m_solverState.d_denseJtr, sizeof(float) * 6 * numberOfVariables);
	m_maxNumDenseImPairs = m_maxNumberOfImages * (m_maxNumberOfImages - 1) / 2;
	cudaMalloc(&m_solverState.d_denseCorrCounts, sizeof(float) * m_maxNumDenseImPairs);
	cudaMalloc(&m_solverState.d_denseOverlappingImages, sizeof(uint2) * m_maxNumDenseImPairs);
	cudaMalloc(&m_solverState.d_numDenseOverlappingImages, sizeof(int));

	cudaMalloc(&m_solverState.d_corrCount, sizeof(int));
	cudaMalloc(&m_solverState.d_corrCountColor, sizeof(int));
	cudaMalloc(&m_solverState.d_sumResidualColor, sizeof(float));
	cudaMalloc(&m_solverState.d_xTransforms, sizeof(float4x4)*m_maxNumberOfImages);
	cudaMalloc(&m_solverState.d_xTransformInverses, sizeof(float4x4)*m_maxNumberOfImages);

#ifdef NEW_GUIDED_REMOVE
	cudaMalloc(&d_transforms, sizeof(float4x4)*m_maxNumberOfImages);
#endif

	//solve params
	m_maxResidualThresh = 100;
	m_defaultParams.denseDistThresh = (*yml)["p2p"]["max_dist"].as<float>();
	m_defaultParams.denseNormalThresh = std::cos((*yml)["p2p"]["max_normal_angle"].as<float>()/180.0*M_PI);
	m_defaultParams.sparse_dist_thres = (*yml)["bundle"]["feature_edge_dist_thres"].as<float>();
	m_defaultParams.sparse_normal_thres = std::cos((*yml)["bundle"]["feature_edge_normal_thres"].as<float>()/180.0*M_PI);
	m_defaultParams.icp_pose_rot_thres = (*yml)["bundle"]["icp_pose_rot_thres"].as<float>()/180.0*M_PI;
	m_defaultParams.denseColorThresh = 0.1;
	m_defaultParams.denseColorGradientMin = 0.005;
	m_defaultParams.denseDepthMin = 0.1;
	m_defaultParams.denseDepthMax = 9999;
	m_defaultParams.denseOverlapCheckSubsampleFactor = 1;

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	//!!!DEBUGGING
	cudaMemset(m_solverState.d_deltaRot, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_deltaTrans, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_rRot, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_rTrans, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_zRot, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_zTrans, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_pRot, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_pTrans, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_Jp, 0, sizeof(float3)*maxNumResiduals);
	cudaMemset(m_solverState.d_Ap_XRot, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_Ap_XTrans, 0, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_scanAlpha, 0, sizeof(float) * 2);
	cudaMemset(m_solverState.d_rDotzOld, -1, sizeof(float) *numberOfVariables);
	cudaMemset(m_solverState.d_precondionerRot, 1, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_precondionerTrans, 1, sizeof(float3)*numberOfVariables);
	cudaMemset(m_solverState.d_sumResidual, 0, sizeof(float));
	cudaMemset(m_solverExtra.d_maxResidual, 0, sizeof(float) * n);
	cudaMemset(m_solverExtra.d_maxResidualIndex, 0, sizeof(int) * n);
	cudaMemset(d_variablesToCorrespondences, 0, sizeof(int)*m_maxNumberOfImages*m_maxCorrPerImage);
	cudaMemset(d_numEntriesPerRow, 0, sizeof(int)*m_maxNumberOfImages);
	cudaMemset(d_n_match_per_pair, 0, sizeof(int)*m_maxNumberOfImages*m_maxNumberOfImages);
	cudaMemset(m_solverState.d_countHighResidual, -1, sizeof(int));
	cudaMemset(m_solverState.d_denseJtJ, 0, sizeof(float) * 36 * numberOfVariables * numberOfVariables);
	cudaMemset(m_solverState.d_denseJtr, 0, sizeof(float) * 6 * numberOfVariables);
	cudaMemset(m_solverState.d_denseCorrCounts, -1, sizeof(float) * m_maxNumDenseImPairs);
	cudaMemset(m_solverState.d_denseOverlappingImages, -1, sizeof(uint2) * m_maxNumDenseImPairs);
	cudaMemset(m_solverState.d_numDenseOverlappingImages, -1, sizeof(int));

	cudaMemset(m_solverState.d_corrCount, -1, sizeof(int));
	cudaMemset(m_solverState.d_corrCountColor, -1, sizeof(int));
	cudaMemset(m_solverState.d_sumResidualColor, -1, sizeof(float));

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	//!!!DEBUGGING
}

CUDASolverBundling::~CUDASolverBundling()
{
	if (m_timer) delete m_timer;

	// State
	cutilSafeCall(cudaFree(m_solverState.d_deltaRot));
	cutilSafeCall(cudaFree(m_solverState.d_deltaTrans));
	cutilSafeCall(cudaFree(m_solverState.d_rRot));
	cutilSafeCall(cudaFree(m_solverState.d_rTrans));
	cutilSafeCall(cudaFree(m_solverState.d_zRot));
	cutilSafeCall(cudaFree(m_solverState.d_zTrans));
	cutilSafeCall(cudaFree(m_solverState.d_pRot));
	cutilSafeCall(cudaFree(m_solverState.d_pTrans));
	cutilSafeCall(cudaFree(m_solverState.d_Jp));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_XRot));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_XTrans));
	cutilSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cutilSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cutilSafeCall(cudaFree(m_solverState.d_precondionerRot));
	cutilSafeCall(cudaFree(m_solverState.d_precondionerTrans));
	cutilSafeCall(cudaFree(m_solverState.d_sumResidual));
	cutilSafeCall(cudaFree(m_solverExtra.d_maxResidual));
	cutilSafeCall(cudaFree(m_solverExtra.d_maxResidualIndex));
	SAFE_DELETE_ARRAY(m_solverExtra.h_maxResidual);
	SAFE_DELETE_ARRAY(m_solverExtra.h_maxResidualIndex);

	cutilSafeCall(cudaFree(d_variablesToCorrespondences));
	cutilSafeCall(cudaFree(d_numEntriesPerRow));
	cutilSafeCall(cudaFree(d_n_match_per_pair));
	cutilSafeCall(cudaFree(d_update_pose_flags));

	cutilSafeCall(cudaFree(m_solverState.d_countHighResidual));
	cutilSafeCall(cudaFree(m_solverState.d_denseCorrCounts));
	cutilSafeCall(cudaFree(m_solverState.d_denseJtJ));
	cutilSafeCall(cudaFree(m_solverState.d_denseJtr));

	cutilSafeCall(cudaFree(m_solverState.d_xTransforms));
	cutilSafeCall(cudaFree(m_solverState.d_xTransformInverses));
	cutilSafeCall(cudaFree(m_solverState.d_denseOverlappingImages));
	cutilSafeCall(cudaFree(m_solverState.d_numDenseOverlappingImages));

	cutilSafeCall(cudaFree(m_solverState.d_corrCount));
	cutilSafeCall(cudaFree(m_solverState.d_sumResidualColor));
	cutilSafeCall(cudaFree(m_solverState.d_corrCountColor));

#ifdef NEW_GUIDED_REMOVE
	cutilSafeCall(cudaFree(d_transforms));
#endif
}

void CUDASolverBundling::solve(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, const int* d_validImages, unsigned int numberOfImages,const CUDACache* cudaCache, const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool usePairwiseDense, float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns, bool rebuildJT, bool findMaxResidual, unsigned int revalidateIdx)
{
	unsigned int nNonLinearIterations = (*yml)["bundle"]["num_iter_outter"].as<int>();
	assert(numberOfImages > 1 && nNonLinearIterations > 0);
	if (numberOfCorrespondences > m_maxCorrPerImage*m_maxNumberOfImages) {
		//warning: correspondences will be invalidated AT RANDOM!
		std::cerr << "WARNING: #corr (" << numberOfCorrespondences << ") exceeded limit (" << m_maxCorrPerImage << "*" << m_maxNumberOfImages << "), please increase max #corr per image in the GAS" << std::endl;
	}

	float* convergence = NULL;
	if (m_bRecordConvergence) {
		m_convergence.resize(nNonLinearIterations + 1, -1.0f);
		convergence = m_convergence.data();
	}

	m_solverState.d_xRot = d_rotationAnglesUnknowns;
	m_solverState.d_xTrans = d_translationUnknowns;

	SolverParameters parameters = m_defaultParams;
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinIterations = (*yml)["bundle"]["num_iter_inner"].as<int>();;
	parameters.verifyOptDistThresh = m_verifyOptDistThresh;
	parameters.verifyOptPercentThresh = m_verifyOptPercentThresh;
	parameters.highResidualThresh = std::numeric_limits<float>::infinity();
	parameters.robust_delta = (*yml)["bundle"]["robust_delta"].as<float>();
	parameters.depth_association_radius = (*yml)["bundle"]["depth_association_radius"].as<int>();

	parameters.weightSparse = weightsSparse.front();
	parameters.weightDenseDepth = weightsDenseDepth.front();
	parameters.weightDenseColor = weightsDenseColor.front();
	parameters.useDense = (parameters.weightDenseDepth > 0 || parameters.weightDenseColor > 0);
	parameters.useDenseDepthAllPairwise = usePairwiseDense;

	SolverInput solverInput;
	solverInput.d_correspondences = d_correspondences;
	solverInput.d_variablesToCorrespondences = d_variablesToCorrespondences;
	solverInput.d_numEntriesPerRow = d_numEntriesPerRow;
	solverInput.numberOfImages = numberOfImages;
	solverInput.numberOfCorrespondences = numberOfCorrespondences;
	solverInput.d_n_match_per_pair = d_n_match_per_pair;
	solverInput.d_update_pose_flags = d_update_pose_flags;
	solverInput.maxNumberOfImages = m_maxNumberOfImages;
	solverInput.maxCorrPerImage = m_maxCorrPerImage;
	solverInput.maxNumDenseImPairs = m_maxNumDenseImPairs;

	solverInput.weightsSparse = weightsSparse.data();
	solverInput.weightsDenseDepth = weightsDenseDepth.data();
	solverInput.weightsDenseColor = weightsDenseColor.data();
	solverInput.d_validImages = d_validImages;
	solverInput.d_cacheFrames = cudaCache->getCacheFramesGPU();
	solverInput.denseDepthWidth = cudaCache->getWidth(); //TODO constant buffer for this?
	solverInput.denseDepthHeight = cudaCache->getHeight();
	Eigen::Matrix4f intrinsics = cudaCache->getIntrinsics();
	solverInput.intrinsics = make_float4(intrinsics(0, 0), intrinsics(1, 1), intrinsics(0, 2), intrinsics(1, 2));
	assert(solverInput.denseDepthWidth / parameters.denseOverlapCheckSubsampleFactor > 8); //need enough samples


#ifdef NEW_GUIDED_REMOVE
	convertLiePosesToMatricesCU(m_solverState.d_xRot, m_solverState.d_xTrans, solverInput.numberOfImages, d_transforms, m_solverState.d_xTransformInverses); //debugging only (store transforms before opt)
#endif
#ifdef DEBUG_PRINT_SPARSE_RESIDUALS
	if (findMaxResidual) {
		float residualBefore = EvalResidual(solverInput, m_solverState, parameters, NULL);
		computeMaxResidual(solverInput, parameters, (unsigned int)-1);
		Eigen::Vector<uint, 2> beforeMaxImageIndices;
    float beforeMaxRes;
    unsigned int curFrame = (revalidateIdx == (unsigned int)-1) ? solverInput.numberOfImages - 1 : revalidateIdx;
		getMaxResidual(curFrame, d_correspondences, beforeMaxImageIndices, beforeMaxRes);
		std::cout << "\tbefore: (" << solverInput.numberOfImages << ") sumres = " << residualBefore << " / " << solverInput.numberOfCorrespondences << " = " << residualBefore / (float)solverInput.numberOfCorrespondences << " | maxres = " << beforeMaxRes << " images (" << beforeMaxImageIndices << ")" << std::endl;
	}
#endif


	if (rebuildJT)
	{
		buildVariablesToCorrespondencesTable(d_correspondences, numberOfCorrespondences);
	}

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	solveBundlingStub(solverInput, m_solverState, parameters, m_solverExtra, convergence, m_timer);

	if (findMaxResidual) {
		computeMaxResidual(solverInput, parameters, revalidateIdx);
#ifdef DEBUG_PRINT_SPARSE_RESIDUALS
		float residualAfter = EvalResidual(solverInput, m_solverState, parameters, NULL);
		Eigen::Vector<uint, 2> afterMaxImageIndices;
    float afterMaxRes;
    unsigned int curFrame = (revalidateIdx == (unsigned int)-1) ? solverInput.numberOfImages - 1 : revalidateIdx;
		getMaxResidual(curFrame, d_correspondences, afterMaxImageIndices, afterMaxRes);
		std::cout << "\tafter: (" << solverInput.numberOfImages << ") sumres = " << residualAfter << " / " << solverInput.numberOfCorrespondences << " = " << residualAfter / (float)solverInput.numberOfCorrespondences << " | maxres = " << afterMaxRes << " images (" << afterMaxImageIndices << ")" << std::endl;
#endif
	}
}

void CUDASolverBundling::buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences)
{
	cutilSafeCall(cudaMemset(d_numEntriesPerRow, 0, sizeof(int)*m_maxNumberOfImages));

	if (numberOfCorrespondences > 0)
		buildVariablesToCorrespondencesTableCUDA(d_correspondences, numberOfCorrespondences, m_maxCorrPerImage, d_variablesToCorrespondences, d_numEntriesPerRow, m_timer, d_n_match_per_pair, m_maxNumberOfImages);
}

void CUDASolverBundling::computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters, unsigned int revalidateIdx)
{
	if (m_timer) m_timer->startEvent(__FUNCTION__);
	if (parameters.weightSparse > 0.0f) {
		evalMaxResidual(solverInput, m_solverState, m_solverExtra, parameters, NULL);//m_timer);
		// copy to cpu
		unsigned int n = (solverInput.numberOfCorrespondences + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		cutilSafeCall(cudaMemcpy(m_solverExtra.h_maxResidual, m_solverExtra.d_maxResidual, sizeof(float) * n, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(m_solverExtra.h_maxResidualIndex, m_solverExtra.d_maxResidualIndex, sizeof(int) * n, cudaMemcpyDeviceToHost));
		// compute max
		float maxResidual = 0.0f; int maxResidualIndex = 0;
		for (unsigned int i = 0; i < n; i++) {
			if (maxResidual < m_solverExtra.h_maxResidual[i]) {
				maxResidual = m_solverExtra.h_maxResidual[i];
				maxResidualIndex = m_solverExtra.h_maxResidualIndex[i];
			}
		}
#ifdef NEW_GUIDED_REMOVE

		//if (solverInput.numberOfImages == 51) {
		//	SensorData sd; sd.loadFromFile("../data/iclnuim/aliv2.sens");
		//	std::vector<mat4f> trajectory(solverInput.numberOfImages);
		//	cudaMemcpy(trajectory.data(), d_transforms, sizeof(mat4f)*trajectory.size(), cudaMemcpyDeviceToHost));
		//	sd.saveToPointCloud("debug/tmp.ply", trajectory, 0, solverInput.numberOfImages*10, 10, true);
		//	int a = 5;
		//}

		m_maxResImPairs.clear();
		if (maxResidual > GUIDED_SEARCH_MAX_RES_THRESH) {
			parameters.highResidualThresh = std::min(std::max(0.2f * maxResidual, 0.1f), 0.4f);
			collectHighResiduals(solverInput, m_solverState, m_solverExtra, parameters, m_timer);
			unsigned int highResCount;
			cutilSafeCall(cudaMemcpy(&highResCount, m_solverState.d_countHighResidual, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			n = std::min(highResCount, (m_maxCorrPerImage*m_maxNumberOfImages + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
			cutilSafeCall(cudaMemcpy(m_solverExtra.h_maxResidual, m_solverExtra.d_maxResidual, sizeof(float) * n, cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(m_solverExtra.h_maxResidualIndex, m_solverExtra.d_maxResidualIndex, sizeof(int) * n, cudaMemcpyDeviceToHost));

			if (n > 1) {
				// check high residuals with previous trajectory as reference //TODO MAKE EFFICIENT
				std::vector<float4x4> transforms(solverInput.numberOfImages);
				cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*solverInput.numberOfImages, cudaMemcpyDeviceToHost);
				std::unordered_map<vec2ui, float> residualMap; //TODO should be something better than this...
				std::unordered_map<vec2ui, float> allCollectedResidualMap; //debugging
				std::vector<EntryJ> corrs(n);
				for (unsigned int i = 0; i < n; i++) {
					cudaMemcpy(corrs.data() + i, solverInput.d_correspondences + m_solverExtra.h_maxResidualIndex[i], sizeof(EntryJ), cudaMemcpyDeviceToHost);
					const EntryJ& h_corr = corrs[i];
					Eigen::Vector<uint, 2> imageIndices(h_corr.imgIdx_i, h_corr.imgIdx_j);
					//compute res at previous
					if (h_corr.imgIdx_j == solverInput.numberOfImages - 1 && std::abs((int)h_corr.imgIdx_i - (int)h_corr.imgIdx_j) > 10) { //introduced by latest image
						float3 prevRes = fabs(transforms[h_corr.imgIdx_i] * h_corr.pos_i - transforms[h_corr.imgIdx_j] * h_corr.pos_j); //eval new corrs with previous trajectory
						float prevMaxRes = fmaxf(prevRes.z, fmaxf(prevRes.x, prevRes.y));
						if (prevMaxRes > 1.5f*m_solverExtra.h_maxResidual[i]) {
							auto it = residualMap.find(imageIndices);
							if (it == residualMap.end()) residualMap[imageIndices] = m_solverExtra.h_maxResidual[i];
							else it->second = std::max(m_solverExtra.h_maxResidual[i], it->second);
						}
					}
					else if (h_corr.imgIdx_j == revalidateIdx && std::abs((int)h_corr.imgIdx_i - (int)h_corr.imgIdx_j) > 10) { //introduced by latest revalidate
						auto it = residualMap.find(imageIndices);
						if (it == residualMap.end()) residualMap[imageIndices] = m_solverExtra.h_maxResidual[i];
						else it->second = std::max(m_solverExtra.h_maxResidual[i], it->second);
					}
					auto it = allCollectedResidualMap.find(imageIndices);
					if (it == allCollectedResidualMap.end()) allCollectedResidualMap[imageIndices] = m_solverExtra.h_maxResidual[i];
					else it->second = std::max(m_solverExtra.h_maxResidual[i], it->second);
				}
				if (!residualMap.empty()) { //debug print
					unsigned int rep = residualMap.begin()->first.x;
					std::cout << "rep: (" << rep << ", " << solverInput.numberOfImages - 1 << ")" << std::endl;
					for (const auto& r : residualMap) m_maxResImPairs.push_back(r.first);

					////one extra solve
					//parameters.nNonLinearIterations = 1;
					//solveBundlingStub(solverInput, m_solverState, parameters, m_solverExtra, NULL, m_timer);

					////!!!debugging
					//{
					//	static SensorData sd;
					//	if (sd.m_frames.empty()) sd.loadFromFile("../data/iclnuim/aliv2.sens");
					//	std::vector<mat4f> trajectory(solverInput.numberOfImages);
					//	cudaMemcpy(trajectory.data(), d_transforms, sizeof(mat4f)*trajectory.size(), cudaMemcpyDeviceToHost));
					//	sd.saveToPointCloud("debug/tmp/" + std::to_string(solverInput.numberOfImages) + "-init.ply", trajectory, 0, solverInput.numberOfImages*10, 10, true);
					//	convertLiePosesToMatricesCU(m_solverState.d_xRot, m_solverState.d_xTrans, solverInput.numberOfImages, d_transforms, m_solverState.d_xTransformInverses);
					//	cudaMemcpy(trajectory.data(), d_transforms, sizeof(mat4f)*trajectory.size(), cudaMemcpyDeviceToHost));
					//	sd.saveToPointCloud("debug/tmp/" + std::to_string(solverInput.numberOfImages) + "-opt.ply", trajectory, 0, solverInput.numberOfImages*10, 10, true);
					//	int a = 5;
					//}
					////!!!debugging
				}
			}
		}
#endif
		m_solverExtra.h_maxResidual[0] = maxResidual;
		m_solverExtra.h_maxResidualIndex[0] = maxResidualIndex;
	}
	else {
		m_solverExtra.h_maxResidual[0] = 0.0f;
		m_solverExtra.h_maxResidualIndex[0] = 0;
	}
	if (m_timer) m_timer->endEvent();
}

bool CUDASolverBundling::getMaxResidual(unsigned int curFrame, EntryJ* d_correspondences, Eigen::Vector<uint, 2>& imageIndices, float& maxRes)
{
	maxRes = m_solverExtra.h_maxResidual[0];

	// for debugging get image indices regardless
	EntryJ h_corr;
	unsigned int imIdx = m_solverExtra.h_maxResidualIndex[0];
	cutilSafeCall(cudaMemcpy(&h_corr, d_correspondences + imIdx, sizeof(EntryJ), cudaMemcpyDeviceToHost));
	imageIndices = Eigen::Vector<uint, 2>(h_corr.imgIdx_i, h_corr.imgIdx_j);

	bool remove = false;
	//const float curThresh = (imageIndices.y == curFrame) ? m_maxResidualThresh : m_maxResidualThresh * 2.0f; //TODO try this out
	const float curThresh = m_maxResidualThresh;
	if (!(imageIndices(0) == 0 && imageIndices(1) < 10) && m_solverExtra.h_maxResidual[0] > curThresh) remove = true; //don't remove the first frame

	//!!!debugging //TODO REMOVE THIS
	if (m_solverExtra.h_maxResidual[0] > curThresh && imageIndices(0) == 0 && imageIndices(1) < 10) {
		std::cout << "warning! max residual would invalidate images " << imageIndices.transpose() << " (" << m_solverExtra.h_maxResidual[0] << ")" << std::endl;
		//getchar();
	}
	//!!!debugging

	return remove;
}

bool CUDASolverBundling::useVerification(EntryJ* d_correspondences, unsigned int numberOfCorrespondences)
{
	SolverParameters parameters;
	parameters.nNonLinearIterations = 0;
	parameters.nLinIterations = 0;
	parameters.verifyOptDistThresh = m_verifyOptDistThresh;
	parameters.verifyOptPercentThresh = m_verifyOptPercentThresh;

	SolverInput solverInput;
	solverInput.d_correspondences = d_correspondences;
	solverInput.d_variablesToCorrespondences = NULL;
	solverInput.d_numEntriesPerRow = NULL;
	solverInput.numberOfImages = 0;
	solverInput.numberOfCorrespondences = numberOfCorrespondences;

	solverInput.maxNumberOfImages = m_maxNumberOfImages;
	solverInput.maxCorrPerImage = m_maxCorrPerImage;

	unsigned int numHighResiduals = countHighResiduals(solverInput, m_solverState, parameters, m_timer);
	//std::cout << "\t[ useVerification ] " << numHighResiduals << " / " << solverInput.numberOfCorrespondences << " = " << (float)numHighResiduals / solverInput.numberOfCorrespondences << " vs " << parameters.verifyOptPercentThresh << std::endl;
	if ((float)numHighResiduals / solverInput.numberOfCorrespondences >= parameters.verifyOptPercentThresh) return true;
	return false;
}
