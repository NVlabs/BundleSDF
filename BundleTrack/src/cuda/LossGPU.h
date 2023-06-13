#pragma once

#include "Utils.h"
// #include "CUDACache.h"
#include "SIFTImageManager.h"


class OptimizerGpu
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  float _w_fm, _w_p2p, _w_rpi, _w_sdf, _w_pm;
  std::shared_ptr<YAML::Node> yml;
  std::string _id_str;

public:
  OptimizerGpu(std::shared_ptr<YAML::Node> yml1);
  ~OptimizerGpu();
  void optimizeFrames(const std::vector<EntryJ> &global_corres, const std::vector<int> &n_match_per_pair, int n_frames, int H, int W, const std::vector<float*> &depths_gpu, const std::vector<uchar4*> &colors_gpu, const std::vector<float4*> &normals_gpu, const std::vector<int> &update_pose_flags, std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > &poses, const Eigen::Matrix3f &K);

};
