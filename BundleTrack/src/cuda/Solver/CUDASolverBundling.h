#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "SolverBundlingParameters.h"
#include "SolverBundlingState.h"
#include "common.h"
#include "../cuda_SimpleMatrixUtil.h"
#include "../CUDATimer.h"
#include "yaml-cpp/yaml.h"

class CUDACache;


class CUDASolverBundling
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	CUDASolverBundling(unsigned int maxNumberOfImages, unsigned int maxNumResiduals, const int max_corr_per_image, const std::vector<int> &update_pose_flags, std::shared_ptr<YAML::Node> yml1);
	~CUDASolverBundling();

	//weightSparse*Esparse + (#iters*weightDenseLinFactor + weightDense)*Edense
	void solve(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, const int* d_validImages, unsigned int numberOfImages,const CUDACache* cudaCache, const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool usePairwiseDense, float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns, bool rebuildJT, bool findMaxResidual, unsigned int revalidateIdx);
	const std::vector<float>& getConvergenceAnalysis() const { return m_convergence; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_linConvergence; }

	void getMaxResidual(float& max, int& index) const {
		max = m_solverExtra.h_maxResidual[0];
		index = m_solverExtra.h_maxResidualIndex[0];
	};
	bool getMaxResidual(unsigned int curFrame, EntryJ* d_correspondences, Eigen::Vector<uint, 2>& imageIndices, float& maxRes);
	bool useVerification(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);

	const int* getVariablesToCorrespondences() const { return d_variablesToCorrespondences; }
	const int* getVarToCorrNumEntriesPerRow() const { return d_numEntriesPerRow; }

	void evaluateTimings() {
		if (m_timer) {
			//std::cout << "********* SOLVER TIMINGS *********" << std::endl;
			m_timer->evaluate(true);
			std::cout << std::endl << std::endl;
		}
	}

	void resetTimer() {
		if (m_timer) m_timer->reset();
	}

#ifdef NEW_GUIDED_REMOVE
	const std::vector<Eigen::Vector<uint, 2>>& getGuidedMaxResImagesToRemove() const { return m_maxResImPairs; }
#endif
private:

	//!helper
	static bool isSimilarImagePair(const Eigen::Vector<uint, 2>& pair0, const Eigen::Vector<uint, 2>& pair1) {
		if ((std::abs((int)pair0(0) - (int)pair1(0)) < 10 && std::abs((int)pair0(1) - (int)pair1(1)) < 10) ||
			(std::abs((int)pair0(0) - (int)pair1(1)) < 10 && std::abs((int)pair0(1) - (int)pair1(0)) < 10))
			return true;
		return false;
	}

	void buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);
	void computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters, unsigned int revalidateIdx);

	SolverState	m_solverState;
	SolverStateAnalysis m_solverExtra;

	unsigned int m_maxNumberOfImages;
	unsigned int m_maxCorrPerImage;

	unsigned int m_maxNumDenseImPairs;

	int* d_variablesToCorrespondences;
	int* d_numEntriesPerRow;
	int *d_n_match_per_pair;
	int *d_update_pose_flags;

	std::vector<float> m_convergence; // convergence analysis (energy per non-linear iteration)
	std::vector<float> m_linConvergence; // linear residual per linear iteration, concatenates for nonlinear its

	float m_verifyOptDistThresh;
	float m_verifyOptPercentThresh;

	bool		m_bRecordConvergence;
	CUDATimer *m_timer;

	SolverParameters m_defaultParams;
	float			 m_maxResidualThresh;
	std::shared_ptr<YAML::Node> yml;

#ifdef NEW_GUIDED_REMOVE
	//for more than one im-pair removal
	std::vector<Eigen::Vector<uint, 2>> m_maxResImPairs;

	//!!!debugging
	float4x4*	d_transforms;
	//!!!debugging
#endif
};


