#pragma once


#define THREADS_PER_BLOCK_JT_DENSE 128
#define THREADS_PER_BLOCK_JT 128

#include "cuda.h"

#include "../cuda_SimpleMatrixUtil.h"

#include "common.h"
#include "SolverBundlingState.h"
#include "SolverBundlingParameters.h"
#include "cudaUtil.h"
#include "ICPUtil.h"
#include "LieDerivUtil.h"

// residual functions only for sparse!

// not squared!
__inline__ __device__ float evalAbsMaxResidualDevice(unsigned int corrIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float3 r = make_float3(0.0f, 0.0f, 0.0f);

	const EntryJ& corr = input.d_correspondences[corrIdx];
	// if (!corr.active) return 0;
	if (corr.isValid()) {
		float4x4 TI = poseToMatrix(state.d_xRot[corr.imgIdx_i], state.d_xTrans[corr.imgIdx_i]);
		float4x4 TJ = poseToMatrix(state.d_xRot[corr.imgIdx_j], state.d_xTrans[corr.imgIdx_j]);
		r = parameters.weightSparse * fabs((TI*corr.pos_i) - (TJ*corr.pos_j));

		return max(r.z, max(r.x, r.y));
	}
	return 0.0f;
}

__inline__ __device__ float evalFDevice(unsigned int corrIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float weight)
{
	float3 r = make_float3(0.0f, 0.0f, 0.0f);

	EntryJ& corr = input.d_correspondences[corrIdx];
	if (corr.isValid()) {
		float4x4 TI = poseToMatrix(state.d_xRot[corr.imgIdx_i], state.d_xTrans[corr.imgIdx_i]);
		float4x4 TJ = poseToMatrix(state.d_xRot[corr.imgIdx_j], state.d_xTrans[corr.imgIdx_j]);

		r = (TI*corr.pos_i) - (TJ*corr.pos_j);
		// float normal_dot = dot(TI.getFloat3x3()*corr.normal_i, TJ.getFloat3x3()*corr.normal_j);
		// if (length(r)>parameters.sparse_dist_thres || normal_dot<parameters.sparse_normal_thres)
		// {
		// 	corr.active = false;
		// 	return 0;
		// }
		// else
		// {
		// 	corr.active = true;
		// }

		// const float weight = 1000.0/input.d_n_match_per_pair[corr.imgIdx_i*input.maxNumberOfImages+corr.imgIdx_j];
		// if (!isfinite(weight))
		// {
		// 	printf("weight=%f\n",weight);
		// }
		// float res = parameters.weightSparse*weight * dot(r, r);
		float res = parameters.weightSparse * dot(r, r);
		return res;
	}
	return 0.0f;
}

////////////////////////////////////////
// applyJT (compute J^T * r) : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

//@variableIdx: image id
template<bool useDense>
__inline__ __device__ void evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float3& resRot, float3& resTrans)
{
	float3 rRot = make_float3(0.0f, 0.0f, 0.0f);
	float3 rTrans = make_float3(0.0f, 0.0f, 0.0f);

	float3 pRot = make_float3(0.0f, 0.0f, 0.0f);
	float3 pTrans = make_float3(0.0f, 0.0f, 0.0f);

	// Reset linearized update vector
	state.d_deltaRot[variableIdx] = make_float3(0.0f, 0.0f, 0.0f);
	state.d_deltaTrans[variableIdx] = make_float3(0.0f, 0.0f, 0.0f);

	// Compute -JTF here
	int N = min(input.d_numEntriesPerRow[variableIdx], input.maxCorrPerImage);

	for (int i = 0; i < N; i++)
	{
		int corrIdx = input.d_variablesToCorrespondences[variableIdx*input.maxCorrPerImage + i];
		EntryJ &corr = input.d_correspondences[corrIdx];
		if (corr.isValid())
		{
			const float4x4 TI = state.d_xTransforms[corr.imgIdx_i];
			const float4x4 TJ = state.d_xTransforms[corr.imgIdx_j];

			float3 worldP;
			float variableSign = 1;
			if (variableIdx != corr.imgIdx_i)
			{
				variableSign = -1;
				worldP = TJ * corr.pos_j;
			}
			else
			{
				worldP = TI * corr.pos_i;
			}
			const float3 da = evalLie_dAlpha(worldP); //d(e) * T * p  Transformed point wrt. rotation alpha
			const float3 db = evalLie_dBeta(worldP);
			const float3 dc = evalLie_dGamma(worldP);

			const float3 r = (TI * corr.pos_i) - (TJ * corr.pos_j);
			float3 rho;
			const float e = dot(r,r);
			// float normal_dot = dot(TI.getFloat3x3()*corr.normal_i, TJ.getFloat3x3()*corr.normal_j);
			// if (e>parameters.sparse_dist_thres*parameters.sparse_dist_thres || normal_dot<parameters.sparse_normal_thres)
			// {
			// 	corr.active = false;
			// 	continue;
			// }
			// else
			// {
			// 	corr.active = true;
			// }
			huberLoss(e, parameters.robust_delta, rho);
			// printf("i=%d, rho=(%f,%f,%f), r=(%f,%f,%f), da=(%f,%f,%f), db=(%f,%f,%f), dc=(%f,%f,%f)\n",i,rho.x,rho.y,rho.z,r.x,r.y,r.z,da.x,da.y,da.z,db.x,db.y,db.z,dc.x,dc.y,dc.z);

			// printf("worldP=%f, %f, %f\n", worldP.x, worldP.y, worldP.z);
			// printf("|r|=%f, r=%f, %f, %f\n", length(r), r.x, r.y, r.z);
			// printf("TI\n %f, %f, %f, %f\n %f, %f, %f, %f\n %f, %f, %f, %f\n	%f, %f, %f, %f\n", TI.entries[0],TI.entries[1],TI.entries[2],TI.entries[3],TI.entries[4],TI.entries[5],TI.entries[6],TI.entries[7],TI.entries[8],TI.entries[9],TI.entries[10],TI.entries[11],TI.entries[12],TI.entries[13],TI.entries[14],TI.entries[15]);
			// printf("TJ\n %f, %f, %f, %f\n %f, %f, %f, %f\n %f, %f, %f, %f\n	%f, %f, %f, %f\n", TJ.entries[0],TJ.entries[1],TJ.entries[2],TJ.entries[3],TJ.entries[4],TJ.entries[5],TJ.entries[6],TJ.entries[7],TJ.entries[8],TJ.entries[9],TJ.entries[10],TJ.entries[11],TJ.entries[12],TJ.entries[13],TJ.entries[14],TJ.entries[15]);

			// const float weight = 1000.0/input.d_n_match_per_pair[corr.imgIdx_i*input.maxNumberOfImages+corr.imgIdx_j];
			// if (!isfinite(weight))
			// {
			// 	printf("weight=%f\n",weight);
			// }
			// rRot += weight* rho.y * variableSign * make_float3(dot(da, r), dot(db, r), dot(dc, r));
			// rTrans += weight*rho.y * variableSign * r;

			// pRot += weight*rho.y * make_float3(dot(da, da), dot(db, db), dot(dc, dc));
			// pTrans += weight*rho.y * make_float3(1.0f, 1.0f, 1.0f);

			rRot += rho.y * variableSign * make_float3(dot(da, r), dot(db, r), dot(dc, r));
			rTrans += rho.y * variableSign * r;

			pRot += rho.y * make_float3(dot(da, da), dot(db, db), dot(dc, dc));
			pTrans += rho.y * make_float3(1.0f, 1.0f, 1.0f);

			// rRot += variableSign * make_float3(dot(da, r), dot(db, r), dot(dc, r));
			// rTrans += variableSign * r;

			// pRot += make_float3(dot(da, da), dot(db, db), dot(dc, dc));
			// pTrans += make_float3(1.0f, 1.0f, 1.0f);
		}
	}
	if (N>0)
	{
		resRot = -parameters.weightSparse * rRot;
		resTrans = -parameters.weightSparse * rTrans;
		// resRot = -parameters.weightSparse/float(N) * rRot;
		// resTrans = -parameters.weightSparse/float(N) * rTrans;
		// pRot *= parameters.weightSparse/float(N);
		// pTrans *= parameters.weightSparse/float(N);
	}

	if (useDense) { // add dense term
		uint3 transIndices = make_uint3(variableIdx * 6 + 0, variableIdx * 6 + 1, variableIdx * 6 + 2);
		uint3 rotIndices = make_uint3(variableIdx * 6 + 3, variableIdx * 6 + 4, variableIdx * 6 + 5);
		resRot -= make_float3(state.d_denseJtr[rotIndices.x], state.d_denseJtr[rotIndices.y], state.d_denseJtr[rotIndices.z]); //minus since -Jtf, weight already built in
		resTrans -= make_float3(state.d_denseJtr[transIndices.x], state.d_denseJtr[transIndices.y], state.d_denseJtr[transIndices.z]); //minus since -Jtf, weight already built in
		//// preconditioner
		//pRot += make_float3(
		//	state.d_denseJtJ[rotIndices.x * input.numberOfImages * 6 + rotIndices.x],
		//	state.d_denseJtJ[rotIndices.y * input.numberOfImages * 6 + rotIndices.y],
		//	state.d_denseJtJ[rotIndices.z * input.numberOfImages * 6 + rotIndices.z]);
		//pTrans += make_float3(
		//	state.d_denseJtJ[transIndices.x * input.numberOfImages * 6 + transIndices.x],
		//	state.d_denseJtJ[transIndices.y * input.numberOfImages * 6 + transIndices.y],
		//	state.d_denseJtJ[transIndices.z * input.numberOfImages * 6 + transIndices.z]);
	}

	// Preconditioner depends on last solution P(input.d_x)
	if (pRot.x > FLOAT_EPSILON)   state.d_precondionerRot[variableIdx].x = 1.0f / pRot.x;
	else					      state.d_precondionerRot[variableIdx].x = 1.0f;

	if (pRot.y > FLOAT_EPSILON)   state.d_precondionerRot[variableIdx].y = 1.0f / pRot.y;
	else					      state.d_precondionerRot[variableIdx].y = 1.0f;

	if (pRot.z > FLOAT_EPSILON)   state.d_precondionerRot[variableIdx].z = 1.0f / pRot.z;
	else						  state.d_precondionerRot[variableIdx].z = 1.0f;

	if (pTrans.x > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx].x = 1.0f / pTrans.x;
	else					      state.d_precondionerTrans[variableIdx].x = 1.0f;

	if (pTrans.y > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx].y = 1.0f / pTrans.y;
	else					      state.d_precondionerTrans[variableIdx].y = 1.0f;

	if (pTrans.z > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx].z = 1.0f / pTrans.z;
	else					      state.d_precondionerTrans[variableIdx].z = 1.0f;
}

////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ void applyJTDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, const SolverParameters& parameters,
	float3& outRot, float3& outTrans, unsigned int threadIdx, unsigned int lane)
{
	// Compute J^T*d_Jp here
	outRot = make_float3(0.0f, 0.0f, 0.0f);
	outTrans = make_float3(0.0f, 0.0f, 0.0f);

	int N = min(input.d_numEntriesPerRow[variableIdx], input.maxCorrPerImage);

	for (int i = threadIdx; i < N; i += THREADS_PER_BLOCK_JT)
	{
		int corrIdx = input.d_variablesToCorrespondences[variableIdx*input.maxCorrPerImage + i];
		const EntryJ& corr = input.d_correspondences[corrIdx];
		// if (corr.isValid() && corr.active) {
		if (corr.isValid()) {
			const float4x4 TI = state.d_xTransforms[corr.imgIdx_i];
			const float4x4 TJ = state.d_xTransforms[corr.imgIdx_j];

			float3 worldP;
			float  variableSign = 1;
			if (variableIdx != corr.imgIdx_i)
			{
				variableSign = -1;
				worldP = TJ * corr.pos_j;
			}
			else {
				worldP = TI * corr.pos_i;
			}
			const float3 da = evalLie_dAlpha(worldP);
			const float3 db = evalLie_dBeta(worldP);
			const float3 dc = evalLie_dGamma(worldP);

			outRot += variableSign * make_float3(dot(da, state.d_Jp[corrIdx]), dot(db, state.d_Jp[corrIdx]), dot(dc, state.d_Jp[corrIdx]));
			outTrans += variableSign * state.d_Jp[corrIdx];
		}
	}
	//apply j already applied the weight

	outRot.x = warpReduce(outRot.x);	 outRot.y = warpReduce(outRot.y);	  outRot.z = warpReduce(outRot.z);
	outTrans.x = warpReduce(outTrans.x); outTrans.y = warpReduce(outTrans.y); outTrans.z = warpReduce(outTrans.z);
}

__inline__ __device__ float3 applyJDevice(unsigned int corrIdx, SolverInput& input, SolverState& state, const SolverParameters& parameters)
{
	// Compute Jp here
	float3 b = make_float3(0.0f, 0.0f, 0.0f);
	const EntryJ& corr = input.d_correspondences[corrIdx];

	// if (corr.isValid() && corr.active) {
	if (corr.isValid()) {
		// const float weight = 1000.0/input.d_n_match_per_pair[corr.imgIdx_i*input.maxNumberOfImages+corr.imgIdx_j];
		// if (!isfinite(weight))
		// {
		// 	printf("weight=%f\n",weight);
		// }
		if (corr.imgIdx_i > 0)	// get transform 0
		{
			const float4x4 TI = state.d_xTransforms[corr.imgIdx_i];
			const float3 worldP = TI * corr.pos_i;
			const float3 da = evalLie_dAlpha(worldP);
			const float3 db = evalLie_dBeta(worldP);
			const float3 dc = evalLie_dGamma(worldP);

			const float3  pp0 = state.d_pRot[corr.imgIdx_i];
			b += da*pp0.x + db*pp0.y + dc*pp0.z + state.d_pTrans[corr.imgIdx_i];
		}

		if (corr.imgIdx_j > 0)	// get transform 1
		{
			const float4x4 TJ = state.d_xTransforms[corr.imgIdx_j];
			const float3 worldP = TJ * corr.pos_j;
			const float3 da = evalLie_dAlpha(worldP);
			const float3 db = evalLie_dBeta(worldP);
			const float3 dc = evalLie_dGamma(worldP);

			const float3  pp1 = state.d_pRot[corr.imgIdx_j];
			b -= da*pp1.x + db*pp1.y + dc*pp1.z + state.d_pTrans[corr.imgIdx_j];
		}
		// b *= parameters.weightSparse*weight;
		b *= parameters.weightSparse;
	}
	return b;
}

////////////////////////////////////////
// dense depth term
////////////////////////////////////////

/**
 * @brief
 *
 * @param jacBlockRow
 * @param transform_i
 * @param invTransform_j
 * @param camPosSrc
 * @param normalTgt
 * @param use_normal
 * @param rho from robust estimator (error, deriv, second-order deriv)
 * @return __inline__
 */
__inline__ __device__ void computeJacobianBlockRow_i(matNxM<1, 6>& jacBlockRow, const float4x4& transform_i, const float4x4& invTransform_j, const float3& camPosSrc, const float3& normalTgt)
{
	matNxM<3, 6> jac = evalLie_derivI(invTransform_j, transform_i, camPosSrc);
	for (unsigned int i = 0; i < 6; i++) {
		jacBlockRow(i) = -dot(make_float3(jac(0, i), jac(1, i), jac(2, i)), normalTgt);
	}
}

__inline__ __device__ void computeJacobianBlockRow_j(matNxM<1, 6>& jacBlockRow, const float4x4& invTransform_i,
	const float4x4& transform_j, const float3& camPosSrc, const float3& normalTgt)
{
	matNxM<3, 6> jac = evalLie_derivJ(invTransform_i, transform_j, camPosSrc);
	for (unsigned int i = 0; i < 6; i++) {
		jacBlockRow(i) = -dot(make_float3(jac(0, i), jac(1, i), jac(2, i)), normalTgt);
	}
}
////////////////////////////////////////
// dense color term
////////////////////////////////////////
__inline__ __device__ float computeColorDProjLookup(const float4& dx, const float3& camPosSrcToTgt, const float2& intensityDerivTgt, const float2& colorFocalLength)
{
	mat3x1 dcdx; dcdx(0) = dx.x; dcdx(1) = dx.y; dcdx(2) = dx.z;
	mat2x3 dProjectionC = dCameraToScreen(camPosSrcToTgt, colorFocalLength.x, colorFocalLength.y);
	mat1x2 dColorB(intensityDerivTgt);
	mat1x1 dadx = dColorB * dProjectionC * dcdx;

	return dadx(0);
}
__inline__ __device__ void computeJacobianBlockIntensityRow_i(matNxM<1, 6>& jacBlockRow, const float2& colorFocal, const float4x4& transform_i,
	const float4x4& invTransform_j, const float3& camPosSrc, const float3& camPosSrcToTgt, const float2& intensityDerivTgt)
{
	matNxM<3, 6> jac = evalLie_derivI(invTransform_j, transform_i, camPosSrc);					//TODO shared compute here with depth and j
	mat2x3 dProj = dCameraToScreen(camPosSrcToTgt, colorFocal.x, colorFocal.y);
	mat1x2 dColorB(intensityDerivTgt);
	jacBlockRow = dColorB * (dProj * jac);
}
__inline__ __device__ void computeJacobianBlockIntensityRow_j(matNxM<1, 6>& jacBlockRow, const float2& colorFocal, const float4x4& invTransform_i,
	const float4x4& transform_j, const float3& camPosSrc, const float3& camPosSrcToTgt, const float2& intensityDerivTgt)
{
	matNxM<3, 6> jac = evalLie_derivJ(invTransform_i, transform_j, camPosSrc);			//TODO shared compute here with depth and j
	mat2x3 dProj = dCameraToScreen(camPosSrcToTgt, colorFocal.x, colorFocal.y);
	mat1x2 dColorB(intensityDerivTgt);
	jacBlockRow = dColorB * (dProj * jac);

	////one transform only!
	//float gz2 = camPosSrcToTgt.z * camPosSrcToTgt.z;
	//matNxM<2, 6> j; j.setZero();
	//j(0, 0) = colorFocal.x / camPosSrcToTgt.z;
	//j(1, 1) = colorFocal.y / camPosSrcToTgt.z;
	//j(0, 2) = -colorFocal.x * camPosSrcToTgt.x / gz2;
	//j(1, 2) = -colorFocal.y * camPosSrcToTgt.y / gz2;

	//j(0, 3) = -colorFocal.x * camPosSrcToTgt.x * camPosSrcToTgt.y / gz2;
	//j(1, 3) = -colorFocal.y * (1.0f + (camPosSrcToTgt.y * camPosSrcToTgt.y / gz2));
	//j(0, 4) = colorFocal.x * (1.0f + (camPosSrcToTgt.x * camPosSrcToTgt.x / gz2));
	//j(1, 4) = colorFocal.y * camPosSrcToTgt.x * camPosSrcToTgt.y / gz2;
	//j(0, 5) = -colorFocal.x * camPosSrcToTgt.y / camPosSrcToTgt.z;
	//j(1, 5) = colorFocal.y * camPosSrcToTgt.x / camPosSrcToTgt.z;
	//mat1x2 iDeriv; iDeriv(0) = intensityDerivTgt.x; iDeriv(1) = intensityDerivTgt.y;
	////matNxM<1, 6> tmp = iDeriv *  j;
	////for (unsigned int i = 0; i < 3; i++) {
	////	jacBlockRow(i) = tmp(i + 3);
	////	jacBlockRow(i + 3) = tmp(i);
	////}
	//jacBlockRow = iDeriv *  j;
}

