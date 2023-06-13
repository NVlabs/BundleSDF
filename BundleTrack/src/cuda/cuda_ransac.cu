#include "cuda_ransac.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cudaUtil.h"
#include "common.h"


#define gone					1065353216
#define gsine_pi_over_eight		1053028117
#define gcosine_pi_over_eight   1064076127
#define gone_half				0.5f
#define gsmall_number			1.e-12f
#define gtiny_number			1.e-20f
#define gfour_gamma_squared		5.8284273147583007813f



union un { float f; unsigned int ui; };


//From https://github.com/kuiwuchn/3x3_SVD_CUDA
__device__ void svd(
	float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33,			// input A
		float &u11, float &u12, float &u13, float &u21, float &u22, float &u23, float &u31, float &u32, float &u33,	// output U
	float &s11,
	//float &s12, float &s13, float &s21,
	float &s22,
	//float &s23, float &s31, float &s32,
	float &s33,	// output S
	float &v11, float &v12, float &v13, float &v21, float &v22, float &v23, float &v31, float &v32, float &v33	// output V
)
{
	un Sa11, Sa21, Sa31, Sa12, Sa22, Sa32, Sa13, Sa23, Sa33;
	un Su11, Su21, Su31, Su12, Su22, Su32, Su13, Su23, Su33;
	un Sv11, Sv21, Sv31, Sv12, Sv22, Sv32, Sv13, Sv23, Sv33;
	un Sc, Ss, Sch, Ssh;
	un Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;
	un Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;
	un Sqvs, Sqvvx, Sqvvy, Sqvvz;

	Sa11.f = a11; Sa12.f = a12; Sa13.f = a13;
	Sa21.f = a21; Sa22.f = a22; Sa23.f = a23;
	Sa31.f = a31; Sa32.f = a32; Sa33.f = a33;

	//###########################################################
	// Compute normal equations matrix
	//###########################################################

	Ss11.f = Sa11.f*Sa11.f;
	Stmp1.f = Sa21.f*Sa21.f;
	Ss11.f = __fadd_rn(Stmp1.f, Ss11.f);
	Stmp1.f = Sa31.f*Sa31.f;
	Ss11.f = __fadd_rn(Stmp1.f, Ss11.f);

	Ss21.f = Sa12.f*Sa11.f;
	Stmp1.f = Sa22.f*Sa21.f;
	Ss21.f = __fadd_rn(Stmp1.f, Ss21.f);
	Stmp1.f = Sa32.f*Sa31.f;
	Ss21.f = __fadd_rn(Stmp1.f, Ss21.f);

	Ss31.f = Sa13.f*Sa11.f;
	Stmp1.f = Sa23.f*Sa21.f;
	Ss31.f = __fadd_rn(Stmp1.f, Ss31.f);
	Stmp1.f = Sa33.f*Sa31.f;
	Ss31.f = __fadd_rn(Stmp1.f, Ss31.f);

	Ss22.f = Sa12.f*Sa12.f;
	Stmp1.f = Sa22.f*Sa22.f;
	Ss22.f = __fadd_rn(Stmp1.f, Ss22.f);
	Stmp1.f = Sa32.f*Sa32.f;
	Ss22.f = __fadd_rn(Stmp1.f, Ss22.f);

	Ss32.f = Sa13.f*Sa12.f;
	Stmp1.f = Sa23.f*Sa22.f;
	Ss32.f = __fadd_rn(Stmp1.f, Ss32.f);
	Stmp1.f = Sa33.f*Sa32.f;
	Ss32.f = __fadd_rn(Stmp1.f, Ss32.f);

	Ss33.f = Sa13.f*Sa13.f;
	Stmp1.f = Sa23.f*Sa23.f;
	Ss33.f = __fadd_rn(Stmp1.f, Ss33.f);
	Stmp1.f = Sa33.f*Sa33.f;
	Ss33.f = __fadd_rn(Stmp1.f, Ss33.f);

	Sqvs.f = 1.f; Sqvvx.f = 0.f; Sqvvy.f = 0.f; Sqvvz.f = 0.f;

	//###########################################################
	// Solve symmetric eigenproblem using Jacobi iteration
	//###########################################################
	for (int i = 0; i < 4; i++)
	{
		Ssh.f = Ss21.f * 0.5f;
		Stmp5.f = __fsub_rn(Ss11.f, Ss22.f);

		Stmp2.f = Ssh.f*Ssh.f;
		Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
		Ssh.ui = Stmp1.ui&Ssh.ui;
		Sch.ui = Stmp1.ui&Stmp5.ui;
		Stmp2.ui = ~Stmp1.ui&gone;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f*Ssh.f;
		Stmp2.f = Sch.f*Sch.f;
		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Stmp4.f = __frsqrt_rn(Stmp3.f);

		Ssh.f = Stmp4.f*Ssh.f;
		Sch.f = Stmp4.f*Sch.f;
		Stmp1.f = gfour_gamma_squared*Stmp1.f;
		Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

		Stmp2.ui = gsine_pi_over_eight&Stmp1.ui;
		Ssh.ui = ~Stmp1.ui&Ssh.ui;
		Ssh.ui = Ssh.ui | Stmp2.ui;
		Stmp2.ui = gcosine_pi_over_eight&Stmp1.ui;
		Sch.ui = ~Stmp1.ui&Sch.ui;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f * Ssh.f;
		Stmp2.f = Sch.f * Sch.f;
		Sc.f = __fsub_rn(Stmp2.f, Stmp1.f);
		Ss.f = Sch.f * Ssh.f;
		Ss.f = __fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f, Sch.f);
#endif
		//###########################################################
		// Perform the actual Givens conjugation
		//###########################################################

		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Ss33.f = Ss33.f * Stmp3.f;
		Ss31.f = Ss31.f * Stmp3.f;
		Ss32.f = Ss32.f * Stmp3.f;
		Ss33.f = Ss33.f * Stmp3.f;

		Stmp1.f = Ss.f * Ss31.f;
		Stmp2.f = Ss.f * Ss32.f;
		Ss31.f = Sc.f * Ss31.f;
		Ss32.f = Sc.f * Ss32.f;
		Ss31.f = __fadd_rn(Stmp2.f, Ss31.f);
		Ss32.f = __fsub_rn(Ss32.f, Stmp1.f);

		Stmp2.f = Ss.f*Ss.f;
		Stmp1.f = Ss22.f*Stmp2.f;
		Stmp3.f = Ss11.f*Stmp2.f;
		Stmp4.f = Sc.f*Sc.f;
		Ss11.f = Ss11.f*Stmp4.f;
		Ss22.f = Ss22.f*Stmp4.f;
		Ss11.f = __fadd_rn(Ss11.f, Stmp1.f);
		Ss22.f = __fadd_rn(Ss22.f, Stmp3.f);
		Stmp4.f = __fsub_rn(Stmp4.f, Stmp2.f);
		Stmp2.f = __fadd_rn(Ss21.f, Ss21.f);
		Ss21.f = Ss21.f*Stmp4.f;
		Stmp4.f = Sc.f*Ss.f;
		Stmp2.f = Stmp2.f*Stmp4.f;
		Stmp5.f = Stmp5.f*Stmp4.f;
		Ss11.f = __fadd_rn(Ss11.f, Stmp2.f);
		Ss21.f = __fsub_rn(Ss21.f, Stmp5.f);
		Ss22.f = __fsub_rn(Ss22.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("%.20g\n", Ss11.f);
		printf("%.20g %.20g\n", Ss21.f, Ss22.f);
		printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

		//###########################################################
		// Compute the cumulative rotation, in quaternion form
		//###########################################################

		Stmp1.f = Ssh.f*Sqvvx.f;
		Stmp2.f = Ssh.f*Sqvvy.f;
		Stmp3.f = Ssh.f*Sqvvz.f;
		Ssh.f = Ssh.f*Sqvs.f;

		Sqvs.f = Sch.f*Sqvs.f;
		Sqvvx.f = Sch.f*Sqvvx.f;
		Sqvvy.f = Sch.f*Sqvvy.f;
		Sqvvz.f = Sch.f*Sqvvz.f;

		Sqvvz.f = __fadd_rn(Sqvvz.f, Ssh.f);
		Sqvs.f = __fsub_rn(Sqvs.f, Stmp3.f);
		Sqvvx.f = __fadd_rn(Sqvvx.f, Stmp2.f);
		Sqvvy.f = __fsub_rn(Sqvvy.f, Stmp1.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU q %.20g %.20g %.20g %.20g\n", Sqvvx.f, Sqvvy.f, Sqvvz.f, Sqvs.f);
#endif

		//////////////////////////////////////////////////////////////////////////
		// (1->3)
		//////////////////////////////////////////////////////////////////////////
		Ssh.f = Ss32.f * 0.5f;
		Stmp5.f = __fsub_rn(Ss22.f, Ss33.f);

		Stmp2.f = Ssh.f * Ssh.f;
		Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
		Ssh.ui = Stmp1.ui&Ssh.ui;
		Sch.ui = Stmp1.ui&Stmp5.ui;
		Stmp2.ui = ~Stmp1.ui&gone;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f * Ssh.f;
		Stmp2.f = Sch.f * Sch.f;
		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Stmp4.f = __frsqrt_rn(Stmp3.f);

		Ssh.f = Stmp4.f * Ssh.f;
		Sch.f = Stmp4.f * Sch.f;
		Stmp1.f = gfour_gamma_squared * Stmp1.f;
		Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

		Stmp2.ui = gsine_pi_over_eight&Stmp1.ui;
		Ssh.ui = ~Stmp1.ui&Ssh.ui;
		Ssh.ui = Ssh.ui | Stmp2.ui;
		Stmp2.ui = gcosine_pi_over_eight&Stmp1.ui;
		Sch.ui = ~Stmp1.ui&Sch.ui;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f * Ssh.f;
		Stmp2.f = Sch.f * Sch.f;
		Sc.f = __fsub_rn(Stmp2.f, Stmp1.f);
		Ss.f = Sch.f*Ssh.f;
		Ss.f = __fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f, Sch.f);
#endif

		//###########################################################
		// Perform the actual Givens conjugation
		//###########################################################

		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Ss11.f = Ss11.f * Stmp3.f;
		Ss21.f = Ss21.f * Stmp3.f;
		Ss31.f = Ss31.f * Stmp3.f;
		Ss11.f = Ss11.f * Stmp3.f;

		Stmp1.f = Ss.f*Ss21.f;
		Stmp2.f = Ss.f*Ss31.f;
		Ss21.f = Sc.f*Ss21.f;
		Ss31.f = Sc.f*Ss31.f;
		Ss21.f = __fadd_rn(Stmp2.f, Ss21.f);
		Ss31.f = __fsub_rn(Ss31.f, Stmp1.f);

		Stmp2.f = Ss.f*Ss.f;
		Stmp1.f = Ss33.f*Stmp2.f;
		Stmp3.f = Ss22.f*Stmp2.f;
		Stmp4.f = Sc.f * Sc.f;
		Ss22.f = Ss22.f * Stmp4.f;
		Ss33.f = Ss33.f * Stmp4.f;
		Ss22.f = __fadd_rn(Ss22.f, Stmp1.f);
		Ss33.f = __fadd_rn(Ss33.f, Stmp3.f);
		Stmp4.f = __fsub_rn(Stmp4.f, Stmp2.f);
		Stmp2.f = __fadd_rn(Ss32.f, Ss32.f);
		Ss32.f = Ss32.f*Stmp4.f;
		Stmp4.f = Sc.f*Ss.f;
		Stmp2.f = Stmp2.f*Stmp4.f;
		Stmp5.f = Stmp5.f*Stmp4.f;
		Ss22.f = __fadd_rn(Ss22.f, Stmp2.f);
		Ss32.f = __fsub_rn(Ss32.f, Stmp5.f);
		Ss33.f = __fsub_rn(Ss33.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("%.20g\n", Ss11.f);
		printf("%.20g %.20g\n", Ss21.f, Ss22.f);
		printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

		//###########################################################
		// Compute the cumulative rotation, in quaternion form
		//###########################################################

		Stmp1.f = Ssh.f*Sqvvx.f;
		Stmp2.f = Ssh.f*Sqvvy.f;
		Stmp3.f = Ssh.f*Sqvvz.f;
		Ssh.f = Ssh.f*Sqvs.f;

		Sqvs.f = Sch.f*Sqvs.f;
		Sqvvx.f = Sch.f*Sqvvx.f;
		Sqvvy.f = Sch.f*Sqvvy.f;
		Sqvvz.f = Sch.f*Sqvvz.f;

		Sqvvx.f = __fadd_rn(Sqvvx.f, Ssh.f);
		Sqvs.f = __fsub_rn(Sqvs.f, Stmp1.f);
		Sqvvy.f = __fadd_rn(Sqvvy.f, Stmp3.f);
		Sqvvz.f = __fsub_rn(Sqvvz.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU q %.20g %.20g %.20g %.20g\n", Sqvvx.f, Sqvvy.f, Sqvvz.f, Sqvs.f);
#endif
#if 1
		//////////////////////////////////////////////////////////////////////////
		// 1 -> 2
		//////////////////////////////////////////////////////////////////////////

		Ssh.f = Ss31.f * 0.5f;
		Stmp5.f = __fsub_rn(Ss33.f, Ss11.f);

		Stmp2.f = Ssh.f*Ssh.f;
		Stmp1.ui = (Stmp2.f >= gtiny_number) ? 0xffffffff : 0;
		Ssh.ui = Stmp1.ui&Ssh.ui;
		Sch.ui = Stmp1.ui&Stmp5.ui;
		Stmp2.ui = ~Stmp1.ui&gone;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f*Ssh.f;
		Stmp2.f = Sch.f*Sch.f;
		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Stmp4.f = __frsqrt_rn(Stmp3.f);

		Ssh.f = Stmp4.f*Ssh.f;
		Sch.f = Stmp4.f*Sch.f;
		Stmp1.f = gfour_gamma_squared*Stmp1.f;
		Stmp1.ui = (Stmp2.f <= Stmp1.f) ? 0xffffffff : 0;

		Stmp2.ui = gsine_pi_over_eight&Stmp1.ui;
		Ssh.ui = ~Stmp1.ui&Ssh.ui;
		Ssh.ui = Ssh.ui | Stmp2.ui;
		Stmp2.ui = gcosine_pi_over_eight&Stmp1.ui;
		Sch.ui = ~Stmp1.ui&Sch.ui;
		Sch.ui = Sch.ui | Stmp2.ui;

		Stmp1.f = Ssh.f*Ssh.f;
		Stmp2.f = Sch.f*Sch.f;
		Sc.f = __fsub_rn(Stmp2.f, Stmp1.f);
		Ss.f = Sch.f*Ssh.f;
		Ss.f = __fadd_rn(Ss.f, Ss.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("GPU s %.20g, c %.20g, sh %.20g, ch %.20g\n", Ss.f, Sc.f, Ssh.f, Sch.f);
#endif

		//###########################################################
		// Perform the actual Givens conjugation
		//###########################################################

		Stmp3.f = __fadd_rn(Stmp1.f, Stmp2.f);
		Ss22.f = Ss22.f * Stmp3.f;
		Ss32.f = Ss32.f * Stmp3.f;
		Ss21.f = Ss21.f * Stmp3.f;
		Ss22.f = Ss22.f * Stmp3.f;

		Stmp1.f = Ss.f*Ss32.f;
		Stmp2.f = Ss.f*Ss21.f;
		Ss32.f = Sc.f*Ss32.f;
		Ss21.f = Sc.f*Ss21.f;
		Ss32.f = __fadd_rn(Stmp2.f, Ss32.f);
		Ss21.f = __fsub_rn(Ss21.f, Stmp1.f);

		Stmp2.f = Ss.f*Ss.f;
		Stmp1.f = Ss11.f*Stmp2.f;
		Stmp3.f = Ss33.f*Stmp2.f;
		Stmp4.f = Sc.f*Sc.f;
		Ss33.f = Ss33.f*Stmp4.f;
		Ss11.f = Ss11.f*Stmp4.f;
		Ss33.f = __fadd_rn(Ss33.f, Stmp1.f);
		Ss11.f = __fadd_rn(Ss11.f, Stmp3.f);
		Stmp4.f = __fsub_rn(Stmp4.f, Stmp2.f);
		Stmp2.f = __fadd_rn(Ss31.f, Ss31.f);
		Ss31.f = Ss31.f*Stmp4.f;
		Stmp4.f = Sc.f*Ss.f;
		Stmp2.f = Stmp2.f*Stmp4.f;
		Stmp5.f = Stmp5.f*Stmp4.f;
		Ss33.f = __fadd_rn(Ss33.f, Stmp2.f);
		Ss31.f = __fsub_rn(Ss31.f, Stmp5.f);
		Ss11.f = __fsub_rn(Ss11.f, Stmp2.f);

#ifdef DEBUG_JACOBI_CONJUGATE
		printf("%.20g\n", Ss11.f);
		printf("%.20g %.20g\n", Ss21.f, Ss22.f);
		printf("%.20g %.20g %.20g\n", Ss31.f, Ss32.f, Ss33.f);
#endif

		//###########################################################
		// Compute the cumulative rotation, in quaternion form
		//###########################################################

		Stmp1.f = Ssh.f*Sqvvx.f;
		Stmp2.f = Ssh.f*Sqvvy.f;
		Stmp3.f = Ssh.f*Sqvvz.f;
		Ssh.f = Ssh.f*Sqvs.f;

		Sqvs.f = Sch.f*Sqvs.f;
		Sqvvx.f = Sch.f*Sqvvx.f;
		Sqvvy.f = Sch.f*Sqvvy.f;
		Sqvvz.f = Sch.f*Sqvvz.f;

		Sqvvy.f = __fadd_rn(Sqvvy.f, Ssh.f);
		Sqvs.f = __fsub_rn(Sqvs.f, Stmp2.f);
		Sqvvz.f = __fadd_rn(Sqvvz.f, Stmp1.f);
		Sqvvx.f = __fsub_rn(Sqvvx.f, Stmp3.f);
#endif
	}

	//###########################################################
	// Normalize quaternion for matrix V
	//###########################################################

	Stmp2.f = Sqvs.f*Sqvs.f;
	Stmp1.f = Sqvvx.f*Sqvvx.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = Sqvvy.f*Sqvvy.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = Sqvvz.f*Sqvvz.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);

	Stmp1.f = __frsqrt_rn(Stmp2.f);
	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);

	Sqvs.f = Sqvs.f*Stmp1.f;
	Sqvvx.f = Sqvvx.f*Stmp1.f;
	Sqvvy.f = Sqvvy.f*Stmp1.f;
	Sqvvz.f = Sqvvz.f*Stmp1.f;

	//###########################################################
	// Transform quaternion to matrix V
	//###########################################################

	Stmp1.f = Sqvvx.f*Sqvvx.f;
	Stmp2.f = Sqvvy.f*Sqvvy.f;
	Stmp3.f = Sqvvz.f*Sqvvz.f;
	Sv11.f = Sqvs.f*Sqvs.f;
	Sv22.f = __fsub_rn(Sv11.f, Stmp1.f);
	Sv33.f = __fsub_rn(Sv22.f, Stmp2.f);
	Sv33.f = __fadd_rn(Sv33.f, Stmp3.f);
	Sv22.f = __fadd_rn(Sv22.f, Stmp2.f);
	Sv22.f = __fsub_rn(Sv22.f, Stmp3.f);
	Sv11.f = __fadd_rn(Sv11.f, Stmp1.f);
	Sv11.f = __fsub_rn(Sv11.f, Stmp2.f);
	Sv11.f = __fsub_rn(Sv11.f, Stmp3.f);
	Stmp1.f = __fadd_rn(Sqvvx.f, Sqvvx.f);
	Stmp2.f = __fadd_rn(Sqvvy.f, Sqvvy.f);
	Stmp3.f = __fadd_rn(Sqvvz.f, Sqvvz.f);
	Sv32.f = Sqvs.f*Stmp1.f;
	Sv13.f = Sqvs.f*Stmp2.f;
	Sv21.f = Sqvs.f*Stmp3.f;
	Stmp1.f = Sqvvy.f*Stmp1.f;
	Stmp2.f = Sqvvz.f*Stmp2.f;
	Stmp3.f = Sqvvx.f*Stmp3.f;
	Sv12.f = __fsub_rn(Stmp1.f, Sv21.f);
	Sv23.f = __fsub_rn(Stmp2.f, Sv32.f);
	Sv31.f = __fsub_rn(Stmp3.f, Sv13.f);
	Sv21.f = __fadd_rn(Stmp1.f, Sv21.f);
	Sv32.f = __fadd_rn(Stmp2.f, Sv32.f);
	Sv13.f = __fadd_rn(Stmp3.f, Sv13.f);

	///###########################################################
	// Multiply (from the right) with V
	//###########################################################

	Stmp2.f = Sa12.f;
	Stmp3.f = Sa13.f;
	Sa12.f = Sv12.f*Sa11.f;
	Sa13.f = Sv13.f*Sa11.f;
	Sa11.f = Sv11.f*Sa11.f;
	Stmp1.f = Sv21.f*Stmp2.f;
	Sa11.f = __fadd_rn(Sa11.f, Stmp1.f);
	Stmp1.f = Sv31.f*Stmp3.f;
	Sa11.f = __fadd_rn(Sa11.f, Stmp1.f);
	Stmp1.f = Sv22.f*Stmp2.f;
	Sa12.f = __fadd_rn(Sa12.f, Stmp1.f);
	Stmp1.f = Sv32.f*Stmp3.f;
	Sa12.f = __fadd_rn(Sa12.f, Stmp1.f);
	Stmp1.f = Sv23.f*Stmp2.f;
	Sa13.f = __fadd_rn(Sa13.f, Stmp1.f);
	Stmp1.f = Sv33.f*Stmp3.f;
	Sa13.f = __fadd_rn(Sa13.f, Stmp1.f);

	Stmp2.f = Sa22.f;
	Stmp3.f = Sa23.f;
	Sa22.f = Sv12.f*Sa21.f;
	Sa23.f = Sv13.f*Sa21.f;
	Sa21.f = Sv11.f*Sa21.f;
	Stmp1.f = Sv21.f*Stmp2.f;
	Sa21.f = __fadd_rn(Sa21.f, Stmp1.f);
	Stmp1.f = Sv31.f*Stmp3.f;
	Sa21.f = __fadd_rn(Sa21.f, Stmp1.f);
	Stmp1.f = Sv22.f*Stmp2.f;
	Sa22.f = __fadd_rn(Sa22.f, Stmp1.f);
	Stmp1.f = Sv32.f*Stmp3.f;
	Sa22.f = __fadd_rn(Sa22.f, Stmp1.f);
	Stmp1.f = Sv23.f*Stmp2.f;
	Sa23.f = __fadd_rn(Sa23.f, Stmp1.f);
	Stmp1.f = Sv33.f*Stmp3.f;
	Sa23.f = __fadd_rn(Sa23.f, Stmp1.f);

	Stmp2.f = Sa32.f;
	Stmp3.f = Sa33.f;
	Sa32.f = Sv12.f*Sa31.f;
	Sa33.f = Sv13.f*Sa31.f;
	Sa31.f = Sv11.f*Sa31.f;
	Stmp1.f = Sv21.f*Stmp2.f;
	Sa31.f = __fadd_rn(Sa31.f, Stmp1.f);
	Stmp1.f = Sv31.f*Stmp3.f;
	Sa31.f = __fadd_rn(Sa31.f, Stmp1.f);
	Stmp1.f = Sv22.f*Stmp2.f;
	Sa32.f = __fadd_rn(Sa32.f, Stmp1.f);
	Stmp1.f = Sv32.f*Stmp3.f;
	Sa32.f = __fadd_rn(Sa32.f, Stmp1.f);
	Stmp1.f = Sv23.f*Stmp2.f;
	Sa33.f = __fadd_rn(Sa33.f, Stmp1.f);
	Stmp1.f = Sv33.f*Stmp3.f;
	Sa33.f = __fadd_rn(Sa33.f, Stmp1.f);

	//###########################################################
	// Permute columns such that the singular values are sorted
	//###########################################################

	Stmp1.f = Sa11.f*Sa11.f;
	Stmp4.f = Sa21.f*Sa21.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp4.f = Sa31.f*Sa31.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);

	Stmp2.f = Sa12.f*Sa12.f;
	Stmp4.f = Sa22.f*Sa22.f;
	Stmp2.f = __fadd_rn(Stmp2.f, Stmp4.f);
	Stmp4.f = Sa32.f*Sa32.f;
	Stmp2.f = __fadd_rn(Stmp2.f, Stmp4.f);

	Stmp3.f = Sa13.f*Sa13.f;
	Stmp4.f = Sa23.f*Sa23.f;
	Stmp3.f = __fadd_rn(Stmp3.f, Stmp4.f);
	Stmp4.f = Sa33.f*Sa33.f;
	Stmp3.f = __fadd_rn(Stmp3.f, Stmp4.f);

	// Swap columns 1-2 if necessary

	Stmp4.ui = (Stmp1.f < Stmp2.f) ? 0xffffffff : 0;
	Stmp5.ui = Sa11.ui^Sa12.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa11.ui = Sa11.ui^Stmp5.ui;
	Sa12.ui = Sa12.ui^Stmp5.ui;

	Stmp5.ui = Sa21.ui^Sa22.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa21.ui = Sa21.ui^Stmp5.ui;
	Sa22.ui = Sa22.ui^Stmp5.ui;

	Stmp5.ui = Sa31.ui^Sa32.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa31.ui = Sa31.ui^Stmp5.ui;
	Sa32.ui = Sa32.ui^Stmp5.ui;

	Stmp5.ui = Sv11.ui^Sv12.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv11.ui = Sv11.ui^Stmp5.ui;
	Sv12.ui = Sv12.ui^Stmp5.ui;

	Stmp5.ui = Sv21.ui^Sv22.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv21.ui = Sv21.ui^Stmp5.ui;
	Sv22.ui = Sv22.ui^Stmp5.ui;

	Stmp5.ui = Sv31.ui^Sv32.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv31.ui = Sv31.ui^Stmp5.ui;
	Sv32.ui = Sv32.ui^Stmp5.ui;

	Stmp5.ui = Stmp1.ui^Stmp2.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp1.ui = Stmp1.ui^Stmp5.ui;
	Stmp2.ui = Stmp2.ui^Stmp5.ui;

	// If columns 1-2 have been swapped, negate 2nd column of A and V so that V is still a rotation

	Stmp5.f = -2.f;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp4.f = 1.f;
	Stmp4.f = __fadd_rn(Stmp4.f, Stmp5.f);

	Sa12.f = Sa12.f*Stmp4.f;
	Sa22.f = Sa22.f*Stmp4.f;
	Sa32.f = Sa32.f*Stmp4.f;

	Sv12.f = Sv12.f*Stmp4.f;
	Sv22.f = Sv22.f*Stmp4.f;
	Sv32.f = Sv32.f*Stmp4.f;

	// Swap columns 1-3 if necessary

	Stmp4.ui = (Stmp1.f < Stmp3.f) ? 0xffffffff : 0;
	Stmp5.ui = Sa11.ui^Sa13.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa11.ui = Sa11.ui^Stmp5.ui;
	Sa13.ui = Sa13.ui^Stmp5.ui;

	Stmp5.ui = Sa21.ui^Sa23.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa21.ui = Sa21.ui^Stmp5.ui;
	Sa23.ui = Sa23.ui^Stmp5.ui;

	Stmp5.ui = Sa31.ui^Sa33.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa31.ui = Sa31.ui^Stmp5.ui;
	Sa33.ui = Sa33.ui^Stmp5.ui;

	Stmp5.ui = Sv11.ui^Sv13.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv11.ui = Sv11.ui^Stmp5.ui;
	Sv13.ui = Sv13.ui^Stmp5.ui;

	Stmp5.ui = Sv21.ui^Sv23.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv21.ui = Sv21.ui^Stmp5.ui;
	Sv23.ui = Sv23.ui^Stmp5.ui;

	Stmp5.ui = Sv31.ui^Sv33.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv31.ui = Sv31.ui^Stmp5.ui;
	Sv33.ui = Sv33.ui^Stmp5.ui;

	Stmp5.ui = Stmp1.ui^Stmp3.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp1.ui = Stmp1.ui^Stmp5.ui;
	Stmp3.ui = Stmp3.ui^Stmp5.ui;

	// If columns 1-3 have been swapped, negate 1st column of A and V so that V is still a rotation

	Stmp5.f = -2.f;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp4.f = 1.f;
	Stmp4.f = __fadd_rn(Stmp4.f, Stmp5.f);

	Sa11.f = Sa11.f*Stmp4.f;
	Sa21.f = Sa21.f*Stmp4.f;
	Sa31.f = Sa31.f*Stmp4.f;

	Sv11.f = Sv11.f*Stmp4.f;
	Sv21.f = Sv21.f*Stmp4.f;
	Sv31.f = Sv31.f*Stmp4.f;

	// Swap columns 2-3 if necessary

	Stmp4.ui = (Stmp2.f < Stmp3.f) ? 0xffffffff : 0;
	Stmp5.ui = Sa12.ui^Sa13.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa12.ui = Sa12.ui^Stmp5.ui;
	Sa13.ui = Sa13.ui^Stmp5.ui;

	Stmp5.ui = Sa22.ui^Sa23.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa22.ui = Sa22.ui^Stmp5.ui;
	Sa23.ui = Sa23.ui^Stmp5.ui;

	Stmp5.ui = Sa32.ui^Sa33.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sa32.ui = Sa32.ui^Stmp5.ui;
	Sa33.ui = Sa33.ui^Stmp5.ui;

	Stmp5.ui = Sv12.ui^Sv13.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv12.ui = Sv12.ui^Stmp5.ui;
	Sv13.ui = Sv13.ui^Stmp5.ui;

	Stmp5.ui = Sv22.ui^Sv23.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv22.ui = Sv22.ui^Stmp5.ui;
	Sv23.ui = Sv23.ui^Stmp5.ui;

	Stmp5.ui = Sv32.ui^Sv33.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Sv32.ui = Sv32.ui^Stmp5.ui;
	Sv33.ui = Sv33.ui^Stmp5.ui;

	Stmp5.ui = Stmp2.ui^Stmp3.ui;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp2.ui = Stmp2.ui^Stmp5.ui;
	Stmp3.ui = Stmp3.ui^Stmp5.ui;

	// If columns 2-3 have been swapped, negate 3rd column of A and V so that V is still a rotation

	Stmp5.f = -2.f;
	Stmp5.ui = Stmp5.ui&Stmp4.ui;
	Stmp4.f = 1.f;
	Stmp4.f = __fadd_rn(Stmp4.f, Stmp5.f);

	Sa13.f = Sa13.f*Stmp4.f;
	Sa23.f = Sa23.f*Stmp4.f;
	Sa33.f = Sa33.f*Stmp4.f;

	Sv13.f = Sv13.f*Stmp4.f;
	Sv23.f = Sv23.f*Stmp4.f;
	Sv33.f = Sv33.f*Stmp4.f;

	//###########################################################
	// Construct QR factorization of A*V (=U*D) using Givens rotations
	//###########################################################

	Su11.f = 1.f; Su12.f = 0.f; Su13.f = 0.f;
	Su21.f = 0.f; Su22.f = 1.f; Su23.f = 0.f;
	Su31.f = 0.f; Su32.f = 0.f; Su33.f = 1.f;

	Ssh.f = Sa21.f*Sa21.f;
	Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
	Ssh.ui = Ssh.ui&Sa21.ui;

	Stmp5.f = 0.f;
	Sch.f = __fsub_rn(Stmp5.f, Sa11.f);
	Sch.f = max(Sch.f, Sa11.f);
	Sch.f = max(Sch.f, gsmall_number);
	Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);
	Stmp1.f = Stmp1.f*Stmp2.f;

	Sch.f = __fadd_rn(Sch.f, Stmp1.f);

	Stmp1.ui = ~Stmp5.ui&Ssh.ui;
	Stmp2.ui = ~Stmp5.ui&Sch.ui;
	Sch.ui = Stmp5.ui&Sch.ui;
	Ssh.ui = Stmp5.ui&Ssh.ui;
	Sch.ui = Sch.ui | Stmp1.ui;
	Ssh.ui = Ssh.ui | Stmp2.ui;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);

	Sch.f = Sch.f*Stmp1.f;
	Ssh.f = Ssh.f*Stmp1.f;

	Sc.f = Sch.f*Sch.f;
	Ss.f = Ssh.f*Ssh.f;
	Sc.f = __fsub_rn(Sc.f, Ss.f);
	Ss.f = Ssh.f*Sch.f;
	Ss.f = __fadd_rn(Ss.f, Ss.f);

	//###########################################################
	// Rotate matrix A
	//###########################################################

	Stmp1.f = Ss.f*Sa11.f;
	Stmp2.f = Ss.f*Sa21.f;
	Sa11.f = Sc.f*Sa11.f;
	Sa21.f = Sc.f*Sa21.f;
	Sa11.f = __fadd_rn(Sa11.f, Stmp2.f);
	Sa21.f = __fsub_rn(Sa21.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa12.f;
	Stmp2.f = Ss.f*Sa22.f;
	Sa12.f = Sc.f*Sa12.f;
	Sa22.f = Sc.f*Sa22.f;
	Sa12.f = __fadd_rn(Sa12.f, Stmp2.f);
	Sa22.f = __fsub_rn(Sa22.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa13.f;
	Stmp2.f = Ss.f*Sa23.f;
	Sa13.f = Sc.f*Sa13.f;
	Sa23.f = Sc.f*Sa23.f;
	Sa13.f = __fadd_rn(Sa13.f, Stmp2.f);
	Sa23.f = __fsub_rn(Sa23.f, Stmp1.f);

	//###########################################################
	// Update matrix U
	//###########################################################

	Stmp1.f = Ss.f*Su11.f;
	Stmp2.f = Ss.f*Su12.f;
	Su11.f = Sc.f*Su11.f;
	Su12.f = Sc.f*Su12.f;
	Su11.f = __fadd_rn(Su11.f, Stmp2.f);
	Su12.f = __fsub_rn(Su12.f, Stmp1.f);

	Stmp1.f = Ss.f*Su21.f;
	Stmp2.f = Ss.f*Su22.f;
	Su21.f = Sc.f*Su21.f;
	Su22.f = Sc.f*Su22.f;
	Su21.f = __fadd_rn(Su21.f, Stmp2.f);
	Su22.f = __fsub_rn(Su22.f, Stmp1.f);

	Stmp1.f = Ss.f*Su31.f;
	Stmp2.f = Ss.f*Su32.f;
	Su31.f = Sc.f*Su31.f;
	Su32.f = Sc.f*Su32.f;
	Su31.f = __fadd_rn(Su31.f, Stmp2.f);
	Su32.f = __fsub_rn(Su32.f, Stmp1.f);

	// Second Givens rotation

	Ssh.f = Sa31.f*Sa31.f;
	Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
	Ssh.ui = Ssh.ui&Sa31.ui;

	Stmp5.f = 0.f;
	Sch.f = __fsub_rn(Stmp5.f, Sa11.f);
	Sch.f = max(Sch.f, Sa11.f);
	Sch.f = max(Sch.f, gsmall_number);
	Stmp5.ui = (Sa11.f >= Stmp5.f) ? 0xffffffff : 0;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);
	Stmp1.f = Stmp1.f*Stmp2.f;

	Sch.f = __fadd_rn(Sch.f, Stmp1.f);

	Stmp1.ui = ~Stmp5.ui&Ssh.ui;
	Stmp2.ui = ~Stmp5.ui&Sch.ui;
	Sch.ui = Stmp5.ui&Sch.ui;
	Ssh.ui = Stmp5.ui&Ssh.ui;
	Sch.ui = Sch.ui | Stmp1.ui;
	Ssh.ui = Ssh.ui | Stmp2.ui;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);

	Sch.f = Sch.f*Stmp1.f;
	Ssh.f = Ssh.f*Stmp1.f;

	Sc.f = Sch.f*Sch.f;
	Ss.f = Ssh.f*Ssh.f;
	Sc.f = __fsub_rn(Sc.f, Ss.f);
	Ss.f = Ssh.f*Sch.f;
	Ss.f = __fadd_rn(Ss.f, Ss.f);

	//###########################################################
	// Rotate matrix A
	//###########################################################

	Stmp1.f = Ss.f*Sa11.f;
	Stmp2.f = Ss.f*Sa31.f;
	Sa11.f = Sc.f*Sa11.f;
	Sa31.f = Sc.f*Sa31.f;
	Sa11.f = __fadd_rn(Sa11.f, Stmp2.f);
	Sa31.f = __fsub_rn(Sa31.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa12.f;
	Stmp2.f = Ss.f*Sa32.f;
	Sa12.f = Sc.f*Sa12.f;
	Sa32.f = Sc.f*Sa32.f;
	Sa12.f = __fadd_rn(Sa12.f, Stmp2.f);
	Sa32.f = __fsub_rn(Sa32.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa13.f;
	Stmp2.f = Ss.f*Sa33.f;
	Sa13.f = Sc.f*Sa13.f;
	Sa33.f = Sc.f*Sa33.f;
	Sa13.f = __fadd_rn(Sa13.f, Stmp2.f);
	Sa33.f = __fsub_rn(Sa33.f, Stmp1.f);

	//###########################################################
	// Update matrix U
	//###########################################################

	Stmp1.f = Ss.f*Su11.f;
	Stmp2.f = Ss.f*Su13.f;
	Su11.f = Sc.f*Su11.f;
	Su13.f = Sc.f*Su13.f;
	Su11.f = __fadd_rn(Su11.f, Stmp2.f);
	Su13.f = __fsub_rn(Su13.f, Stmp1.f);

	Stmp1.f = Ss.f*Su21.f;
	Stmp2.f = Ss.f*Su23.f;
	Su21.f = Sc.f*Su21.f;
	Su23.f = Sc.f*Su23.f;
	Su21.f = __fadd_rn(Su21.f, Stmp2.f);
	Su23.f = __fsub_rn(Su23.f, Stmp1.f);

	Stmp1.f = Ss.f*Su31.f;
	Stmp2.f = Ss.f*Su33.f;
	Su31.f = Sc.f*Su31.f;
	Su33.f = Sc.f*Su33.f;
	Su31.f = __fadd_rn(Su31.f, Stmp2.f);
	Su33.f = __fsub_rn(Su33.f, Stmp1.f);

	// Third Givens Rotation

	Ssh.f = Sa32.f*Sa32.f;
	Ssh.ui = (Ssh.f >= gsmall_number) ? 0xffffffff : 0;
	Ssh.ui = Ssh.ui&Sa32.ui;

	Stmp5.f = 0.f;
	Sch.f = __fsub_rn(Stmp5.f, Sa22.f);
	Sch.f = max(Sch.f, Sa22.f);
	Sch.f = max(Sch.f, gsmall_number);
	Stmp5.ui = (Sa22.f >= Stmp5.f) ? 0xffffffff : 0;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);
	Stmp1.f = Stmp1.f*Stmp2.f;

	Sch.f = __fadd_rn(Sch.f, Stmp1.f);

	Stmp1.ui = ~Stmp5.ui&Ssh.ui;
	Stmp2.ui = ~Stmp5.ui&Sch.ui;
	Sch.ui = Stmp5.ui&Sch.ui;
	Ssh.ui = Stmp5.ui&Ssh.ui;
	Sch.ui = Sch.ui | Stmp1.ui;
	Ssh.ui = Ssh.ui | Stmp2.ui;

	Stmp1.f = Sch.f*Sch.f;
	Stmp2.f = Ssh.f*Ssh.f;
	Stmp2.f = __fadd_rn(Stmp1.f, Stmp2.f);
	Stmp1.f = __frsqrt_rn(Stmp2.f);

	Stmp4.f = Stmp1.f*0.5f;
	Stmp3.f = Stmp1.f*Stmp4.f;
	Stmp3.f = Stmp1.f*Stmp3.f;
	Stmp3.f = Stmp2.f*Stmp3.f;
	Stmp1.f = __fadd_rn(Stmp1.f, Stmp4.f);
	Stmp1.f = __fsub_rn(Stmp1.f, Stmp3.f);

	Sch.f = Sch.f*Stmp1.f;
	Ssh.f = Ssh.f*Stmp1.f;

	Sc.f = Sch.f*Sch.f;
	Ss.f = Ssh.f*Ssh.f;
	Sc.f = __fsub_rn(Sc.f, Ss.f);
	Ss.f = Ssh.f*Sch.f;
	Ss.f = __fadd_rn(Ss.f, Ss.f);

	//###########################################################
	// Rotate matrix A
	//###########################################################

	Stmp1.f = Ss.f*Sa21.f;
	Stmp2.f = Ss.f*Sa31.f;
	Sa21.f = Sc.f*Sa21.f;
	Sa31.f = Sc.f*Sa31.f;
	Sa21.f = __fadd_rn(Sa21.f, Stmp2.f);
	Sa31.f = __fsub_rn(Sa31.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa22.f;
	Stmp2.f = Ss.f*Sa32.f;
	Sa22.f = Sc.f*Sa22.f;
	Sa32.f = Sc.f*Sa32.f;
	Sa22.f = __fadd_rn(Sa22.f, Stmp2.f);
	Sa32.f = __fsub_rn(Sa32.f, Stmp1.f);

	Stmp1.f = Ss.f*Sa23.f;
	Stmp2.f = Ss.f*Sa33.f;
	Sa23.f = Sc.f*Sa23.f;
	Sa33.f = Sc.f*Sa33.f;
	Sa23.f = __fadd_rn(Sa23.f, Stmp2.f);
	Sa33.f = __fsub_rn(Sa33.f, Stmp1.f);

	//###########################################################
	// Update matrix U
	//###########################################################

	Stmp1.f = Ss.f*Su12.f;
	Stmp2.f = Ss.f*Su13.f;
	Su12.f = Sc.f*Su12.f;
	Su13.f = Sc.f*Su13.f;
	Su12.f = __fadd_rn(Su12.f, Stmp2.f);
	Su13.f = __fsub_rn(Su13.f, Stmp1.f);

	Stmp1.f = Ss.f*Su22.f;
	Stmp2.f = Ss.f*Su23.f;
	Su22.f = Sc.f*Su22.f;
	Su23.f = Sc.f*Su23.f;
	Su22.f = __fadd_rn(Su22.f, Stmp2.f);
	Su23.f = __fsub_rn(Su23.f, Stmp1.f);

	Stmp1.f = Ss.f*Su32.f;
	Stmp2.f = Ss.f*Su33.f;
	Su32.f = Sc.f*Su32.f;
	Su33.f = Sc.f*Su33.f;
	Su32.f = __fadd_rn(Su32.f, Stmp2.f);
	Su33.f = __fsub_rn(Su33.f, Stmp1.f);

	v11 = Sv11.f; v12 = Sv12.f; v13 = Sv13.f;
	v21 = Sv21.f; v22 = Sv22.f; v23 = Sv23.f;
	v31 = Sv31.f; v32 = Sv32.f; v33 = Sv33.f;

	u11 = Su11.f; u12 = Su12.f; u13 = Su13.f;
	u21 = Su21.f; u22 = Su22.f; u23 = Su23.f;
	u31 = Su31.f; u32 = Su32.f; u33 = Su33.f;

	s11 = Sa11.f;
	//s12 = Sa12.f; s13 = Sa13.f; s21 = Sa21.f;
	s22 = Sa22.f;
	//s23 = Sa23.f; s31 = Sa31.f; s32 = Sa32.f;
	s33 = Sa33.f;
}


__device__ int evalPoseKernel(const float4 *ptsA, const float4 *ptsB, const int n_pts, const float4x4 &pose, const float dist_thres, int *inlier_ids)
{
	int inliers = 0;

	for (int i = 0; i < n_pts; i++)
	{
		float4 ptA_transformed = pose*ptsA[i];
		float dist = length(ptsB[i]-ptA_transformed);
		if (dist>dist_thres)
		{
			continue;
		}

		inlier_ids[inliers] = i;
		inliers++;
	}

	return inliers;
}

__device__ bool procrustesKernel(const float4 *src_samples, const float4 *dst_samples, const int n_pts, float4x4 &pose)
{
	pose.setIdentity();

	float3 src_mean = make_float3(0.0f, 0.0f, 0.0f);
	float3 dst_mean = make_float3(0.0f, 0.0f, 0.0f);

	for (int i=0;i<n_pts;i++)
	{
		src_mean.x += src_samples[i].x;
		src_mean.y += src_samples[i].y;
		src_mean.z += src_samples[i].z;

		dst_mean.x += dst_samples[i].x;
		dst_mean.y += dst_samples[i].y;
		dst_mean.z += dst_samples[i].z;
	}
	src_mean.x /= n_pts;
	src_mean.y /= n_pts;
	src_mean.z /= n_pts;

	dst_mean.x /= n_pts;
	dst_mean.y /= n_pts;


	dst_mean.z /= n_pts;

	float3x3 S;
	S.setZero();
	for (int i=0;i<n_pts;i++)
	{
		float sx = src_samples[i].x - src_mean.x;
		float sy = src_samples[i].y - src_mean.y;
		float sz = src_samples[i].z - src_mean.z;

		float dx = dst_samples[i].x - dst_mean.x;
		float dy = dst_samples[i].y - dst_mean.y;
		float dz = dst_samples[i].z - dst_mean.z;

		S(0,0) += sx * dx;
		S(0,1) += sx * dy;
		S(0,2) += sx * dz;
		S(1,0) += sy * dx;
		S(1,1) += sy * dy;
		S(1,2) += sy * dz;
		S(2,0) += sz * dx;
		S(2,1) += sz * dy;
		S(2,2) += sz * dz;
	}


	float3x3 U, V;
	float3 ss;
	svd(S(0,0),S(0,1),S(0,2),S(1,0),S(1,1),S(1,2),S(2,0),S(2,1),S(2,2),
		U(0,0),U(0,1),U(0,2),U(1,0),U(1,1),U(1,2),U(2,0),U(2,1),U(2,2),
		ss.x, ss.y, ss.z,
		V(0,0),V(0,1),V(0,2),V(1,0),V(1,1),V(1,2),V(2,0),V(2,1),V(2,2)
		);

	float3x3 R = V * U.getTranspose();
	float3x3 identity;
	identity.setIdentity();

	// printf("R:\n");
	// R.print();

	//////////////////// Check R
	{
		float3x3 tmp = R.getTranspose()*R - identity;
		float diff = 0;
		for (int h=0;h<3;h++)
		{
			for (int w=0;w<3;w++)
			{
				diff += tmp(h,w)*tmp(h,w);
			}
		}
		diff = sqrt(diff);
		if (diff>=1e-3)
		{
			printf("R is not valid\n");
			R.print();
			return false;
		}

		if (R.det()<0)
		{
			for (int i=0;i<3;i++)
			{
				V(i,2) = -V(i,2);
			}
			R = V * U.getTranspose();
		}
	}

	for (int h=0;h<3;h++)
	{
		for (int w=0;w<3;w++)
		{
			pose(h,w) = R(h,w);
		}
	}

	float3 t = dst_mean - R*src_mean;
	pose(0,3) = t.x;
	pose(1,3) = t.y;
	pose(2,3) = t.z;

	return true;
}




__global__ void ransacMultiPairKernel(const float4 *ptsA, const float4 *ptsB, const int max_n_pts, const int n_pairs, const int *n_pts, const int4 *rand_list, const float dist_thres, const int n_trials, int *inlier_ids, int *n_inliers, float4x4 *poses)
{
	const int pair_id = blockIdx.y*blockDim.y + threadIdx.y;
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;

	if (pair_id>=n_pairs) return;

	if(trial_id >= n_trials) return;

	const int global_trial_id = pair_id*n_trials + trial_id;
	const int global_pts_id = pair_id*max_n_pts;

	int rand_idx[3];
	rand_idx[0] = rand_list[global_trial_id].x;
	rand_idx[1] = rand_list[global_trial_id].y;
	rand_idx[2] = rand_list[global_trial_id].z;

	if (rand_idx[0]==rand_idx[1] || rand_idx[1]==rand_idx[2] || rand_idx[0]==rand_idx[2]) return;
	if (rand_idx[0]<0 || rand_idx[1]<0 || rand_idx[2]<0) return;

	float4 src_samples[3];
	float4 dst_samples[3];

	for (int i = 0; i < 3; i++)
	{
		src_samples[i] = ptsA[rand_idx[i]];
		dst_samples[i] = ptsB[rand_idx[i]];
	}

	bool res = procrustesKernel(src_samples, dst_samples, 3, poses[global_trial_id]);

	if (!res)
	{
		return;
	}
	n_inliers[global_trial_id] = evalPoseKernel(ptsA+global_pts_id, ptsB+global_pts_id, n_pts[pair_id], poses[global_trial_id], dist_thres, inlier_ids+pair_id*max_n_pts*n_trials);
}



__global__ void ransacEstimateModelFundamentalMatrixKernel1(const float4* ptsA, const float4* ptsB, const float2* uvA, const float2* uvB, const int n_pts, const int n_sample_per_model, const int n_trials, float *Fs, int *isgood)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= n_trials)
	{
		return;
	}

	curandState state;
	curand_init(0, idx, 0, &state);

	// int rand_idx[n_sample_per_model];
	int *rand_idx = new int[n_sample_per_model];

	for (int i=0;i<n_sample_per_model;i++)
	{
		rand_idx[i] = round(curand_uniform(&state) * (n_pts-1));
	}

	for (int i = 0; i < n_sample_per_model; i++)
	{
		int start = idx*n_sample_per_model*9 + i;
		float uA = uvA[rand_idx[i]].x;
		float vA = uvA[rand_idx[i]].y;
		float uB = uvB[rand_idx[i]].x;
		float vB = uvB[rand_idx[i]].y;
		Fs[start										 ] = uA*uB;
		Fs[start+n_sample_per_model*1] = uA*vB;
		Fs[start+n_sample_per_model*2] = uA;
		Fs[start+n_sample_per_model*3] = vA*uB;
		Fs[start+n_sample_per_model*4] = vA*vB;
		Fs[start+n_sample_per_model*5] = vA;
		Fs[start+n_sample_per_model*6] = uB;
		Fs[start+n_sample_per_model*7] = vB;
		Fs[start+n_sample_per_model*8] = 1;
	}

	///////////!DEBUG
	// if (idx==0)
	// {
	// 	printf("Fs=\n");
	// 	for (int i = 0; i < n_sample_per_model; i++)
	// 	{
	// 		int start = idx*n_sample_per_model*9 + i;
	// 		printf("%f,%f,%f,%f,%f,%f,%f,%f,%f,\n",Fs[start],Fs[start+n_sample_per_model],Fs[start+n_sample_per_model*2],Fs[start+n_sample_per_model*3],Fs[start+n_sample_per_model*4],Fs[start+n_sample_per_model*5],Fs[start+n_sample_per_model*6],Fs[start+n_sample_per_model*7],Fs[start+n_sample_per_model*8]);
	// 	}
	// }

	delete rand_idx;
}



__global__ void ransacEstimateModelFundamentalMatrixKernel2(const float *V, const int n_trials, const int n_sample_per_model, float *Fs, int *isgood)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= n_trials)
	{
		return;
	}

	int start = idx*9*9;
	int start1 = idx*9;
	for (int i=0;i<9;i++)
	{
		Fs[start1+i] = V[start+i+9*8];
	}

	// //////////!DEBUG
	// if (idx==0)
	// {
	// 	printf("V=\n");
	// 	for (int i=0;i<9;i++)
	// 	{
	// 		int start = idx*9*9+i;
	// 		printf("%f, %f, %f, %f, %f, %f, %f, %f, %f,\n",V[start],V[start+9],V[start+9*2],V[start+9*3],V[start+9*4],V[start+9*5],V[start+9*6],V[start+9*7],V[start+9*8]);
	// 	}
	// 	printf("F=\n");
	// 	for (int i=0;i<3;i++)
	// 	{
	// 		printf("%f, %f, %f\n",Fs[start1+i],Fs[start1+3+i],Fs[start1+6+i]);
	// 	}
	// }
}

/**
 * @brief Set S(2,2) to zero and reconstruct F
 *
 * @param U
 * @param S
 * @param V
 * @param n_trials
 * @param n_sample_per_model
 * @param Fs
 * @param isgood
 * @return __global__
 */
__global__ void ransacEstimateModelFundamentalMatrixKernel3(float *U, float *S, float *V, const int n_trials, const int n_sample_per_model, float *Fs, int *isgood)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= n_trials)
	{
		return;
	}

	S[idx*3+2] = 0;
	int start = idx*9;

	Eigen::Matrix3f U_, S_, V_;
	S_.setIdentity();
	for (int h=0;h<3;h++)
	{
		for (int w=0;w<3;w++)
		{
			U_(h,w) = U[start+h+w*3];
			V_(h,w) = V[start+h+w*3];
			if (h==w)
			{
				S_(h,w) = S[start+h];
			}
		}
	}

	Eigen::Matrix3f F = U_*S_*V_.transpose();
	for (int i=0;i<9;i++)
	{
		Fs[start+i] = F(i%3,i/3);
	}

	/////////!DEBUG
	// if (idx==0)
	// {
	// 	printf("S_=\n");
	// 	for (int i=0;i<3;i++)
	// 	{
	// 		printf("%f,%f,%f\n",S_(i,0),S_(i,1),S_(i,2));
	// 	}
	// 	printf("U_=\n");
	// 	for (int i=0;i<3;i++)
	// 	{
	// 		printf("%f,%f,%f\n",U_(i,0),U_(i,1),U_(i,2));
	// 	}
	// 	printf("V_=\n");
	// 	for (int i=0;i<3;i++)
	// 	{
	// 		printf("%f,%f,%f\n",V_(i,0),V_(i,1),V_(i,2));
	// 	}
	// 	printf("F=U_*S_*V_.T\n");
	// 	for (int i=0;i<3;i++)
	// 	{
	// 		printf("%f,%f,%f\n",F(i,0),F(i,1),F(i,2));
	// 	}
	// }

	isgood[idx] = 1;
}


__global__ void ransacEstimateModelKernel(const float4* ptsA, const float4* ptsB, const float2* uvA, const float2* uvB, const int n_pts, const int n_trials, float4x4 *poses, int *isgood)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= n_trials)
	{
		return;
	}

	curandState state;
	curand_init(0, idx, 0, &state);

	int rand_idx[3];

	rand_idx[0] = round(curand_uniform(&state) * (n_pts-1));
	rand_idx[1] = round(curand_uniform(&state) * (n_pts-1));
	rand_idx[2] = round(curand_uniform(&state) * (n_pts-1));

	if (rand_idx[0]==rand_idx[1] || rand_idx[1]==rand_idx[2] || rand_idx[0]==rand_idx[2]) return;
	if (rand_idx[0]<0 || rand_idx[1]<0 || rand_idx[2]<0) return;

	float4 src_samples[3];
	float4 dst_samples[3];

	#pragma unroll
	for (int i = 0; i < 3; i++)
	{
		src_samples[i] = ptsA[rand_idx[i]];
		dst_samples[i] = ptsB[rand_idx[i]];
	}

	bool res = procrustesKernel(src_samples, dst_samples, 3, poses[idx]);
	if (res)
	{
		isgood[idx] = 1;
	}

}

__global__ void printDebug(int *n_inliers, const int n_trials)
{
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (trial_id>=n_trials) return;
	if (n_inliers[trial_id]==0) return;
	printf("trial_id: %d, n_inliers: %d\n", trial_id, n_inliers[trial_id]);
	// n_inliers[trial_id] = 0;

}

__global__ void ransacEvalModelKernel(const float4* ptsA, const float4* ptsB, const float4* normalsA, const float4* normalsB, const float2* uvA, const float2* uvB, const float *confs, const int n_pts, const float4x4 *poses, const int *isgood, const float dist_thres, const float cos_normal_angle, const int n_trials, int *inlier_flags, float *n_inliers)
{
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;
	const int pt_id = blockIdx.y*blockDim.y + threadIdx.y;
	if (trial_id>=n_trials || pt_id>=n_pts) return;
	if (isgood[trial_id]==0) return;

	auto pose = poses[trial_id];
	float4 ptA_transformed = pose*ptsA[pt_id];
	float dist = length(ptsB[pt_id]-ptA_transformed);
	if (dist>dist_thres)
	{
		return;
	}

	float4 normalA_tf = pose*normalsA[pt_id];
	auto normalB = normalsB[pt_id];
	float dot = normalA_tf.x*normalB.x + normalA_tf.y*normalB.y + normalA_tf.z*normalB.z;
	if (dot<cos_normal_angle)
	{
		return;
	}

	atomicAdd(n_inliers+trial_id, confs[pt_id]);
	inlier_flags[trial_id*n_pts+pt_id] = 1;

}


__global__ void ransacEvalModelFundamentalKernel(const float2* uvA, const float2* uvB, const int n_pts, const float *Fs, const int *isgood, const float dist_thres, const int n_trials, int *inlier_flags, int *n_inliers)
{
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;
	const int pt_id = blockIdx.y*blockDim.y + threadIdx.y;
	if (trial_id>=n_trials || pt_id>=n_pts) return;
	if (isgood[trial_id]==0) return;

	int start = trial_id*9;
	Eigen::Matrix3f F;
	F<<Fs[start],Fs[start+3],Fs[start+6],
		Fs[start+1],Fs[start+4],Fs[start+7],
		Fs[start+2],Fs[start+5],Fs[start+8];

	Eigen::Vector3f pA(uvA[pt_id].x, uvA[pt_id].y, 1);
	Eigen::Vector3f pB(uvB[pt_id].x, uvB[pt_id].y, 1);

	Eigen::Vector3f tmp = F*pA;   //transpose() is not supported for vector mult on CUDA
	float dist = abs(tmp.dot(pB));

	// if (trial_id==0)
	// {
	// 	printf("dist=%f, pA=(%f,%f), pB=(%f,%f)\n",dist,pA(0),pA(1),pB(0),pB(1));
	// }

	if (dist>dist_thres)
	{
		return;
	}

	atomicAdd(n_inliers+trial_id, 1);
	inlier_flags[trial_id*n_pts+pt_id] = 1;

}



__global__ void findBestInlier(float *n_inliers, const int n_trials, const float4x4 *poses, const float max_trans, const float max_rot, float *best_trial_num_inliers)
{
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (trial_id>=n_trials) return;

	auto pose = poses[trial_id];
	if (length(pose.getTranslation())>max_trans)
	{
		n_inliers[trial_id] = 0;
		return;
	}
	float3x3 eye;
	eye.setIdentity();
	float rot_diff = rotationGeodesicDistance(pose.getFloat3x3(), eye);
	if (rot_diff>max_rot)
	{
		n_inliers[trial_id] = 0;
		return;
	}

	const float cur_n_inliers = n_inliers[trial_id];
	atomicMax(best_trial_num_inliers, cur_n_inliers);

}

__global__ void getBestTrial(const float *n_inliers, const int n_trials, const float *best_trial_num_inliers, int *best_trial_id)
{
	const int trial_id = blockIdx.x*blockDim.x + threadIdx.x;
	if (trial_id>=n_trials) return;

	if (n_inliers[trial_id]==*best_trial_num_inliers)
	{
		*best_trial_id = trial_id;
	}
}


void ransacMultiPairGPU(const std::vector<float4*> &ptsA, const std::vector<float4*> &ptsB, const std::vector<float4*> &normalsA, const std::vector<float4*> &normalsB, std::vector<float2*> uvA, std::vector<float2*> uvB, const std::vector<float*> confs, const std::vector<int> &n_pts, const int n_trials, const float dist_thres, const float cos_normal_angle, const std::vector<float> &max_transs, const std::vector<float> &max_rots, std::vector<std::vector<int>> &inlier_ids)
{
	const int n_frame_pairs = ptsA.size();
	inlier_ids.resize(n_frame_pairs);
	const int n_thread = 512;
	const int n_block = divCeil(n_trials, n_thread);

	std::vector<float*> n_inliers_gpu(n_frame_pairs);
	std::vector<float4x4*> poses_gpu(n_frame_pairs);
	std::vector<int*> isgood_gpu(n_frame_pairs);
	std::vector<int*> inlier_flags_gpu(n_frame_pairs);
	std::vector<float*> best_trial_num_inliers_gpu(n_frame_pairs);
	std::vector<int*> best_trial_id_gpu(n_frame_pairs);

	cudaStream_t streams[n_frame_pairs];

	for (int i=0;i<n_frame_pairs;i++)
	{
		cudaStreamCreate(&streams[i]);
		const int n_pts_cur_pair = n_pts[i];
		cudaMalloc(&n_inliers_gpu[i], sizeof(float)*n_trials);
		cudaMalloc(&poses_gpu[i], sizeof(float4x4)*n_trials);
		cudaMalloc(&isgood_gpu[i], sizeof(int)*n_trials);
		cudaMalloc(&inlier_flags_gpu[i], sizeof(int)*n_trials*n_pts_cur_pair);
		cudaMalloc(&best_trial_num_inliers_gpu[i], sizeof(float));
		cudaMalloc(&best_trial_id_gpu[i], sizeof(int));


		std::vector<float4x4> poses(n_trials);
		for (int j=0;j<poses.size();j++)
		{
			poses[j].setIdentity();
		}
		cutilSafeCall(cudaMemcpy(poses_gpu[i], poses.data(), sizeof(float4x4)*n_trials, cudaMemcpyHostToDevice));
		cudaMemset(n_inliers_gpu[i], 0, sizeof(float)*n_trials);
		cudaMemset(isgood_gpu[i], 0, sizeof(int)*n_trials);
		cudaMemset(inlier_flags_gpu[i], 0, sizeof(int)*n_trials*n_pts_cur_pair);
		cudaMemset(best_trial_num_inliers_gpu[i], 0, sizeof(float));
		cudaMemset(best_trial_id_gpu[i], 0, sizeof(int));

	}

	cutilSafeCall(cudaDeviceSynchronize());   //!NOTE We need this to sycn before running each stream!

	for (int i=0;i<n_frame_pairs;i++)
	{
		const int n_pts_cur_pair = n_pts[i];
		int threads = 512;
		int blocks = divCeil(n_trials, threads);
		ransacEstimateModelKernel<<<blocks, threads, 0, streams[i]>>>(ptsA[i], ptsB[i], uvA[i], uvB[i], n_pts_cur_pair, n_trials, poses_gpu[i], isgood_gpu[i]);

		int threadsx = 32;
		int blocksx = divCeil(n_trials,threadsx);
		int threadsy = 32;
		int blocksy = divCeil(n_pts_cur_pair,threadsy);

		ransacEvalModelKernel<<<dim3(blocksx,blocksy), dim3(threadsx,threadsy), 0, streams[i]>>>(ptsA[i], ptsB[i], normalsA[i], normalsB[i], uvA[i], uvB[i], confs[i], n_pts_cur_pair, poses_gpu[i], isgood_gpu[i], dist_thres, cos_normal_angle, n_trials, inlier_flags_gpu[i], n_inliers_gpu[i]);

		findBestInlier<<<n_block, n_thread, 0, streams[i]>>>(n_inliers_gpu[i], n_trials, poses_gpu[i], max_transs[i], max_rots[i], best_trial_num_inliers_gpu[i]);
		getBestTrial<<<n_block, n_thread, 0, streams[i]>>>(n_inliers_gpu[i], n_trials, best_trial_num_inliers_gpu[i], best_trial_id_gpu[i]);
	}


	for (int i=0;i<n_frame_pairs;i++)
	{
		cudaStreamSynchronize(streams[i]);
		cutilSafeCall(cudaStreamDestroy(streams[i]));
	}

	for (int i=0;i<n_frame_pairs;i++)
	{
		const int n_pts_cur_pair = n_pts[i];
		int best_trial_id = -1;
		cudaMemcpy(&best_trial_id, best_trial_id_gpu[i], sizeof(int), cudaMemcpyDeviceToHost);

		std::vector<int> inlier_flags(n_pts_cur_pair, 0);
		cudaMemcpy(inlier_flags.data(), inlier_flags_gpu[i]+best_trial_id*n_pts_cur_pair, sizeof(int)*n_pts_cur_pair, cudaMemcpyDeviceToHost);
		inlier_ids[i].clear();
		inlier_ids[i].reserve(n_pts_cur_pair);
		for (int ii=0;ii<inlier_flags.size();ii++)
		{
			if (inlier_flags[ii]>0)
			{
				inlier_ids[i].push_back(ii);
			}
		}
	}

	for (int i=0;i<n_frame_pairs;i++)
	{
		cutilSafeCall(cudaFree(n_inliers_gpu[i]));
		cutilSafeCall(cudaFree(poses_gpu[i]));
		cutilSafeCall(cudaFree(isgood_gpu[i]));
		cutilSafeCall(cudaFree(inlier_flags_gpu[i]));
		cutilSafeCall(cudaFree(best_trial_num_inliers_gpu[i]));
		cutilSafeCall(cudaFree(best_trial_id_gpu[i]));

	}
}