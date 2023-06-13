/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef FRAME_HH_
#define FRAME_HH_
#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDAImageUtil.h"
#include "Utils.h"
#include "FeatureManager.h"
#include <opencv2/core/cuda.hpp>


class MapPoint;

class Frame
{
public:
  enum Status
  {
    FAIL,
    NO_BA,
    OTHER,
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  friend class boost::serialization::access;
  cv::Mat _color, _color_raw, _depth, _depth_raw, _depth_sim, _gray, _fg_mask, _gt_fg_mask, _occ_mask, _normal_map;
  PointCloudRGBNormal::Ptr _cloud, _cloud_down, _real_model;
  Eigen::Matrix4f _pose_in_model = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f _gt_pose_in_model;
  Eigen::Matrix3f _K;
  Eigen::Vector4f _roi;  //umin,umax,vmin,vmax
  int _id;
  double _stamp = -1;
  std::string _id_str;
  std::string _color_file;
  int _H, _W;
  std::shared_ptr<YAML::Node> yml;
  std::vector<cv::KeyPoint> _keypts;  // (u,v)
  cv::Mat _feat_des;
  cv::cuda::GpuMat _feat_des_gpu;
  Status _status;
  static Eigen::Vector3f model_dimensions;
  std::map<std::pair<float,float>, std::shared_ptr<MapPoint>> _map_points;
  int _ref_frame_id = -1;     //!NOTE dont use shared_ptr, it will lead to a reference chain and never memory release
  bool _nerfed = false;

  float *_depth_gpu;
  uchar4 *_color_gpu;
  float4 *_normal_gpu;

  static zmq::context_t context;
  static zmq::socket_t socket;

  Frame();
  Frame(const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depth_raw, const cv::Mat &depth_sim, const Eigen::Vector4f &roi, const Eigen::Matrix4f &pose_in_model, int id, std::string id_str, const Eigen::Matrix3f &K, std::shared_ptr<YAML::Node> yml1, PointCloudRGBNormal::Ptr cloud=NULL, const Eigen::Matrix4f &gt_pose_in_model=Eigen::Matrix4f::Identity(), const cv::Mat &gt_fg_mask=cv::Mat(), PointCloudRGBNormal::Ptr real_model=NULL);
  Frame(const py::array_t<uchar> &color, const py::array_t<float> &depth, const Eigen::Vector4f &roi, const Eigen::Matrix4f &pose_in_model, int id, std::string id_str, const Eigen::Matrix3f &K, std::shared_ptr<YAML::Node> yml1);
  void init();
  ~Frame();
  void setNewInitCoordinate();
  void updateDepthCPU();
  void updateDepthGPU();
  void updateColorGPU();
  void updateNormalGPU();
  void updateRoi();
  void processDepth();
  void pointCloudDenoise();
  void depthToCloudAndNormals();
  void invalidatePixel(const int h, const int w);
  void invalidatePixelsByMask(const cv::Mat &fg_mask);
  void segmentationByGtPose();
  void segmentationByGt();
  int countValidPoints();
  bool operator < (const Frame &other);
};

class FramePtrComparator
{
public:
  bool operator () (const std::shared_ptr<Frame> &f1, const std::shared_ptr<Frame> &f2)
  {
    if (f1->_id < f2->_id) return true;
    return false;
  }
};

class FramePairComparator
{
public:
  typedef std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>> FramePair;

  bool operator ()(const FramePair &p1, const FramePair &p2)
  {
    const int &id11 = p1.first->_id;
    const int &id12 = p1.second->_id;
    const int &id21 = p2.first->_id;
    const int &id22 = p2.second->_id;
    if (id11<id21) return true;
    if (id11>id21) return false;
    if (id12<id22) return true;
    return false;
  }
};


/**
 * @brief
 *
 * @param fA later than fB
 * @param fB
 * @return __inline__
 */
__inline__ float computeCovisibility(const std::shared_ptr<Frame> &fA, const std::shared_ptr<Frame> &fB)
{
  Eigen::Matrix4f cur_in_kfcam = fB->_pose_in_model.inverse()*fA->_pose_in_model;
  // PointCloudRGBNormal::Ptr cloud(new PointCloudRGBNormal);
  // Utils::downsamplePointCloud(fA->_cloud, cloud, 0.005);
  // Utils::passFilterPointCloud(cloud,cloud,"z",0.1, (*fA->yml)["depth_processing"]["zfar"].as<float>());
  // pcl::transformPointCloudWithNormals(*cloud, *cloud, cur_in_kfcam);

  float visible = 0;
  const float thres = std::cos((*fA->yml)["visible_angle"].as<float>()/180.0*M_PI);
  // for (int i=0;i<cloud->points.size();i++)
  // {
  //   const auto &pt = cloud->points[i];
  //   if (!Utils::isPclPointNormalValid(pt)) continue;
  //   Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
  //   Eigen::Vector3f pt_to_eye(-pt.x, -pt.y, -pt.z);
  //   float dot_prod = pt_to_eye.dot(normal);
  //   visible += dot_prod>thres? 1:0;
  // }

  const int stride = 2;
  int total = 0;
  for (int h=fA->_roi(2);h<fA->_roi(3);h+=stride)
  {
    for (int w=fA->_roi(0); w<fA->_roi(1); w+=stride)
    {
      if (fA->_depth.at<float>(h,w)<0.1) continue;
      auto pt = (*fA->_cloud)(w,h);
      if (!Utils::isPclPointNormalValid(pt)) continue;
      total++;
      pt = pcl::transformPointWithNormal(pt, cur_in_kfcam);
      // Eigen::Vector3f projected = fA->_K * Eigen::Vector3f(pt.x, pt.y, pt.z);  //!NOTE this assumes the corase pose is quite good
      // int u = std::round(projected(0)/projected(2));
      // int v = std::round(projected(1)/projected(2));
      // if (u<0 || u>=fA->_W || v<0 || v>=fA->_H) continue;
      // if (fB->_fg_mask.at<uchar>(v,u)==0) continue;
      Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
      Eigen::Vector3f pt_to_eye(-pt.x, -pt.y, -pt.z);
      float dot_prod = pt_to_eye.normalized().dot(normal.normalized());
      visible += dot_prod>thres? 1:0;
    }
  }

    visible /= float(total)+1e-7;

  /////////////!DEBUG
  // if (fA->_id_str>="0028" && fB->_id==0)
  // {
  //   fmt::print("visible: {}\n",visible);
  //   for (int i=0;i<cloud->points.size();i++)
  //   {
  //     auto &pt = cloud->points[i];
  //     if (!Utils::isPclPointNormalValid(pt)) continue;
  //     Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
  //     Eigen::Vector3f pt_to_eye(-pt.x, -pt.y, -pt.z);
  //     float dot_prod = pt_to_eye.dot(normal);
  //     // std::cout<<pt_to_eye.transpose()<<" dot "<< normal.transpose()<<"\n";
  //     if (dot_prod<thres)
  //     {
  //       pt.r = 0;
  //       pt.g = 0;
  //       pt.b = 0;
  //     }
  //   }
  //   pcl::io::savePLYFile("/home/bowen/debug/cloud_in_B.ply", *cloud);
  //   exit(1);
  // }
  return visible;
}



/**
 * @brief
 *
 * @param fA later than fB
 * @param fB
 * @return __inline__
 */
__inline__ float computeCovisibilityCuda(const std::shared_ptr<Frame> &fA, const std::shared_ptr<Frame> &fB)
{
  Eigen::Matrix4f cur_in_kfcam = fB->_pose_in_model.inverse()*fA->_pose_in_model;
  const float thres = std::cos((*fA->yml)["visible_angle"].as<float>()/180.0*M_PI);
  const auto &roi = fA->_roi;
  float visible = CUDAImageUtil::computeCovisibility(fA->_H, fA->_W, roi(0), roi(2), roi(1), roi(3), fA->_K, cur_in_kfcam, thres, fA->_normal_gpu, fA->_depth_gpu);
  return visible;
}



#endif