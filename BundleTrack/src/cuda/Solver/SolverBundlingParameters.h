#pragma once

#include <cmath>

struct SolverParameters
{
	int depth_association_radius;
	unsigned int nNonLinearIterations;		// Steps of the non-linear solver
	unsigned int nLinIterations;			// Steps of the linear solver

	float verifyOptDistThresh; // for verifying local
	float verifyOptPercentThresh;

	float highResidualThresh;
	float robust_delta;
	float sparse_dist_thres = 99999;
	float sparse_normal_thres = -1;  // cosine of normal dot product
	float icp_pose_rot_thres = M_PI*2;

	// dense depth corr
	float denseDistThresh;
	float denseNormalThresh;
	float denseColorThresh;
	float denseColorGradientMin;
	float denseDepthMin;
	float denseDepthMax;

	bool useDenseDepthAllPairwise; // instead of frame-to-frame
	unsigned int denseOverlapCheckSubsampleFactor;

	float weightSparse;
	float weightDenseDepth;
	float weightDenseColor;
	bool useDense;
};

