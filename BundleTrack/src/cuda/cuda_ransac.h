/*
Copyright 2011 Nghia Ho. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY NGHIA HO ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BY NGHIA HO OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Nghia Ho.
*/

#pragma once

#include <vector>
#include <cuda.h>
#include "common.h"
#include "cuda_SimpleMatrixUtil.h"
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include <Eigen/Dense>
#include <curand.h>
#include <curand_kernel.h>
#include <Eigen/Core>


void ransacMultiPairGPU(const std::vector<float4*> &ptsA, const std::vector<float4*> &ptsB, const std::vector<float4*> &normalsA, const std::vector<float4*> &normalsB, std::vector<float2*> uvA, std::vector<float2*> uvB, const std::vector<float*> confs, const std::vector<int> &n_pts, const int n_trials, const float dist_thres, const float cos_normal_angle, const std::vector<float> &max_transs, const std::vector<float> &max_rots, std::vector<std::vector<int>> &inlier_ids);
