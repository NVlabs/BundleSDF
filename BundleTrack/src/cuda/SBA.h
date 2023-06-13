#pragma once


#include "SIFTImageManager.h"
#include "Solver/CUDASolverBundling.h"
#include "yaml-cpp/yaml.h"

struct JacobianBlock {
	Eigen::Vector3f data[6];
};

class SBA
{
public:
	SBA(int maxNumImages, int maxNumResiduals, int max_corr_per_image, const std::vector<int> &update_pose_flags, std::shared_ptr<YAML::Node> yml1);
	~SBA();
	void init(unsigned int maxImages, unsigned int maxNumResiduals, unsigned int max_corr_per_image, const std::vector<int> &update_pose_flags);

	//return if removed res
	bool align(const std::vector<EntryJ> &global_corres, const std::vector<int> &n_match_per_pair, int n_images, const CUDACache* cudaCache, float4x4* d_transforms, bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt, unsigned int revalidateIdx);

	float getMaxResidual() const { return m_maxResidual; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_solver->getLinearConvergenceAnalysis(); }
	bool useVerification() const { return m_bVerify; }

	void evaluateSolverTimings() {
		m_solver->evaluateTimings();
	}
	//void setLocalWeights(const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor) {
	//	m_localWeightsSparse = weightsSparse;
	//	m_localWeightsDenseDepth = weightsDenseDepth;
	//	m_localWeightsDenseColor = weightsDenseColor;
	//}
	void setGlobalWeights(const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool useGlobalDenseOpt) {
		m_globalWeightsMutex.lock();
		m_globalWeightsSparse = weightsSparse;
		m_globalWeightsDenseDepth = weightsDenseDepth;
		m_globalWeightsDenseColor = weightsDenseColor;
		m_bUseGlobalDenseOpt = useGlobalDenseOpt;
		m_globalWeightsMutex.unlock();
	}


public:

	bool alignCUDA(const CUDACache* cudaCache, bool useDensePairwise,
		const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool isStart, bool isEnd, unsigned int revalidateIdx);

	float3*			d_xRot;
	float3*			d_xTrans;
	unsigned int	m_numCorrespondences;
	EntryJ*			d_correspondences;
	int* d_validImages;


	std::vector<EntryJ> _global_corres;
	int _n_images;

	//dense opt params
	bool m_bUseLocalDense;
	bool m_bUseGlobalDenseOpt;
	std::vector<float> m_localWeightsSparse;
	std::vector<float> m_localWeightsDenseDepth;
	std::vector<float> m_localWeightsDenseColor;
	std::vector<float> m_globalWeightsSparse;
	std::vector<float> m_globalWeightsDenseDepth;
	std::vector<float> m_globalWeightsDenseColor;
	std::mutex m_globalWeightsMutex;

	CUDASolverBundling* m_solver;

	bool m_bUseComprehensiveFrameInvalidation;

	//record residual removal
	float m_maxResidual;
	//for gpu solver
	bool m_bVerify;

	std::vector< std::vector<float> > m_recordedConvergence;
	std::shared_ptr<YAML::Node> yml;

};

