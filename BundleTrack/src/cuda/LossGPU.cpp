#include "LossGPU.h"
#include <cuda.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "SBA.h"
#include "CUDACache.h"
#include "common.h"

OptimizerGpu::OptimizerGpu(std::shared_ptr<YAML::Node> yml1)
{
  yml = yml1;
}


OptimizerGpu::~OptimizerGpu()
{

}



void OptimizerGpu::optimizeFrames(const std::vector<EntryJ> &global_corres, const std::vector<int> &n_match_per_pair, int n_frames, int H, int W, const std::vector<float*> &depths_gpu, const std::vector<uchar4*> &colors_gpu, const std::vector<float4*> &normals_gpu, const std::vector<int> &update_pose_flags, std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > &poses, const Eigen::Matrix3f &K)
{
  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();

  auto savePoses = [&](std::string out_file)
  {
    if ((*yml)["SPDLOG"].as<int>()<2) return;
    std::ofstream ff(out_file);
    for (int i=0;i<n_frames;i++)
    {
      const auto &pose = poses[i];
      for (int h=0;h<4;h++)
      {
        for (int w=0;w<4;w++)
        {
          ff<<std::setprecision(16)<<pose(h,w)<<" ";
        }
        ff<<std::endl;
      }
      ff<<std::endl;
    }
    ff.close();
  };

  savePoses(fmt::format("{}/{}/opt_before_poses.txt",debug_dir,_id_str));

  float4x4* d_transforms;
  cudaMalloc(&d_transforms, sizeof(float4x4)*n_frames);
  std::vector<float4x4> transforms_cpu(n_frames);
  for (int i=0;i<n_frames;i++)
  {
    for (int h=0;h<4;h++)
    {
      for (int w=0;w<4;w++)
      {
        transforms_cpu[i].entries2[h][w] = poses[i](h,w);
      }
    }
    // std::cout<<"Before local frame "+std::to_string(i)+" pose\n";
    // transforms_cpu[i].print();
  }
  cudaMemcpy(d_transforms, transforms_cpu.data(), sizeof(float4x4)*n_frames, cudaMemcpyHostToDevice);

  std::cout<<"global_corres="<<global_corres.size()<<std::endl;
  std::map<int,int> n_corres_per_frame;
  for (int i=0;i<global_corres.size();i++)
  {
    const auto &corr = global_corres[i];
    n_corres_per_frame[corr.imgIdx_i]++;
    n_corres_per_frame[corr.imgIdx_j]++;
  }
  int max_corr_per_image = 0;
  for (const auto &h:n_corres_per_frame)
  {
    max_corr_per_image = std::max(max_corr_per_image, h.second);
  }

  const auto image_downscale = (*yml)["bundle"]["image_downscale"].as<std::vector<float>>();

  int iter = -1;
  for (const auto &scale:image_downscale)
  {
    iter++;
    const int W_down = W/scale;
    const int H_down = H/scale;
    Eigen::Matrix4f inputIntrinsics(Eigen::Matrix4f::Identity());
    inputIntrinsics.block(0,0,3,3) = K;

  #ifdef DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
  #endif

    CUDACache cuda_cache(W, H, W_down, H_down, n_frames, inputIntrinsics);
    for (int i=0;i<n_frames;i++)
    {
      cuda_cache.storeFrame(W, H, depths_gpu[i], colors_gpu[i], normals_gpu[i]);
    }

  #ifdef DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
  #endif

    const uint max_n_residuals = n_frames*(n_frames-1)/2*H_down*W_down/4 + global_corres.size();

    SBA sba(n_frames, max_n_residuals, max_corr_per_image, update_pose_flags, yml);
    if (iter>0)
    {
      sba.m_localWeightsSparse.resize(sba.m_localWeightsSparse.size(),0);
    }
    sba.align(global_corres, n_match_per_pair, n_frames, &cuda_cache, d_transforms, false, true, false, true, false, false, -1);

    transforms_cpu.clear();
    cudaMemcpy(transforms_cpu.data(), d_transforms, sizeof(float4x4)*n_frames, cudaMemcpyDeviceToHost);
    for (int i=0;i<n_frames;i++)
    {
      for (int h=0;h<4;h++)
      {
        for (int w=0;w<4;w++)
        {
          poses[i](h,w) = transforms_cpu[i].entries2[h][w];
        }
      }
      // std::cout<<"After local frame "+std::to_string(i)+" pose\n"<<poses[i]<<std::endl;
    }

    savePoses(fmt::format("{}/{}/opt_after_downscale_{:.2f}_poses.txt",debug_dir,_id_str,scale));

  }

  cutilSafeCall(cudaFree(d_transforms));

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}
