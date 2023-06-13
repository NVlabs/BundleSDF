/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "Frame.h"

Eigen::Vector3f Frame::model_dimensions = Eigen::Vector3f::Zero();
zmq::context_t Frame::context;
zmq::socket_t Frame::socket;

Frame::Frame()
{

}


Frame::Frame(const py::array_t<uchar> &color, const py::array_t<float> &depth, const Eigen::Vector4f &roi, const Eigen::Matrix4f &pose_in_model, int id, std::string id_str, const Eigen::Matrix3f &K, std::shared_ptr<YAML::Node> yml1)
{
  _status = OTHER;
  py::buffer_info buf = color.request();
  _color = cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, (uchar*)buf.ptr).clone();
  buf = depth.request();
  _depth = cv::Mat(buf.shape[0], buf.shape[1], CV_32F, (uchar*)buf.ptr).clone();
  _roi = roi;
  _pose_in_model = pose_in_model;
  _id = id;
  _id_str = id_str;
  _K = K;
  yml = yml1;

  _depth_raw = _depth.clone();
  _depth_sim = _depth.clone();

  init();
}


Frame::Frame(const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depth_raw, const cv::Mat &depth_sim, const Eigen::Vector4f &roi, const Eigen::Matrix4f &pose_in_model, int id, std::string id_str, const Eigen::Matrix3f &K, std::shared_ptr<YAML::Node> yml1, PointCloudRGBNormal::Ptr cloud, const Eigen::Matrix4f &gt_pose_in_model, const cv::Mat &gt_fg_mask, PointCloudRGBNormal::Ptr real_model)
{
  _status = OTHER;
  yml = yml1;

  _color = color;

  _depth = depth;
  _depth_raw = depth_raw;
  _depth_sim = depth_sim;
  _id = id;
  _id_str = id_str;
  _pose_in_model = pose_in_model;
  _gt_pose_in_model = gt_pose_in_model;
  _K = K;

  _real_model = real_model;
  _roi = roi;
  _gt_fg_mask = gt_fg_mask;

  if (!_cloud)
  {

  }
  else
  {
    _cloud_down = _cloud;
    SPDLOG("id {}, cloud has been assigned, #points {}", id, _cloud_down->points.size());
  }


  init();
}


void Frame::init()
{
  _H = _color.rows;
  _W = _color.cols;

  if (_color.channels()==1)
  {
    cv::Mat tmp = cv::Mat::zeros(_H,_W,CV_8UC3);
    for (int h=0;h<_H;h++)
    {
      for (int w=0;w<_W;w++)
      {
        uchar gray = _color.at<uchar>(h,w);
        tmp.at<cv::Vec3b>(h,w) = {gray,gray,gray};
      }
    }
    _color = tmp;
    printf("Converted gray to RGB, new color size (%dx%dx%d)\n",_color.rows,_color.cols,_color.channels());
  }
  _color_raw = _color.clone();

  Utils::normalizeRotationMatrix(_pose_in_model);

  _cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
  _cloud_down = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();

  if (_roi(0)<0 || _roi(0)>=_W || _roi(1)<0 || _roi(1)>=_W || _roi(2)<0 || _roi(2)>=_H || _roi(3)<0 || _roi(3)>=_H)
  {
    updateRoi();
  }

  const int n_pixels = _H*_W;
  cudaMalloc(&_depth_gpu, n_pixels*sizeof(float));
  cudaMalloc(&_normal_gpu, n_pixels*sizeof(float4));
  cudaMalloc(&_color_gpu, n_pixels*sizeof(uchar4));

  cudaMemset(_depth_gpu, 0, n_pixels*sizeof(float));
  cudaMemset(_normal_gpu, 0, n_pixels*sizeof(float4));
  cudaMemset(_color_gpu, 0, n_pixels*sizeof(uchar4));

  cv::cvtColor(_color, _gray, cv::COLOR_RGB2GRAY);

  updateDepthGPU();
  processDepth();

  depthToCloudAndNormals();

  if (Frame::model_dimensions==Eigen::Vector3f::Zero())  //Measure model dimensions
  {
    PointCloudRGBNormal::Ptr cloud_world(new PointCloudRGBNormal);
    Utils::passFilterPointCloud(_cloud, cloud_world, "z", 0.1, (*yml)["depth_processing"]["zfar"].as<float>());
    pcl::transformPointCloudWithNormals(*cloud_world, *cloud_world, _pose_in_model);
    pcl::PointXYZRGBNormal min_pt, max_pt;
    pcl::getMinMax3D(*cloud_world, min_pt, max_pt);
    Frame::model_dimensions(0) = max_pt.x - min_pt.x;
    Frame::model_dimensions(1) = max_pt.y - min_pt.y;
    Frame::model_dimensions(2) = max_pt.z - min_pt.z;
  }
}

Frame::~Frame()
{
  cudaFree(_depth_gpu);
  cudaFree(_normal_gpu);
  cudaFree(_color_gpu);
}

void Frame::setNewInitCoordinate()
{
  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();
  PointCloudRGBNormal::Ptr cloud(new PointCloudRGBNormal);
  for (int w=0;w<_W;w++)
  {
    for (int h=0;h<_H;h++)
    {
      const auto &pt = (*_cloud)(w,h);
      if (pt.z>0.1 && _fg_mask.at<uchar>(h,w)>0)
      {
        cloud->points.push_back(pt);
      }
    }
  }
  pcl::io::savePLYFile(fmt::format("{}/cloud_init.ply", debug_dir), *cloud);
  Utils::outlierRemovalStatistic(cloud,cloud,3,30);
  pcl::io::savePLYFile(fmt::format("{}/cloud_for_init_coord.ply", debug_dir), *cloud);
  Eigen::MatrixXf mat = cloud->getMatrixXfMap();  // (D,N)
  Eigen::MatrixXf pts = mat.block(0,0,3,cloud->points.size());
  Eigen::Vector3f max_xyz = pts.rowwise().maxCoeff();
  Eigen::Vector3f min_xyz = pts.rowwise().minCoeff();
  _pose_in_model.block(0,3,3,1) << -(max_xyz+min_xyz)/2;
}

void Frame::updateDepthCPU()
{
  const int n_pixels = _H*_W;
  _depth = cv::Mat::zeros(1, n_pixels, CV_32F);
  cudaMemcpy(_depth.data, _depth_gpu, n_pixels*sizeof(float), cudaMemcpyDeviceToHost);
  _depth = _depth.reshape(1,_H);
}

void Frame::updateDepthGPU()
{
  const int n_pixels = _H*_W;
  cv::Mat depth_flat = _depth.reshape(1,1);
  cudaMemcpy(_depth_gpu, depth_flat.data, n_pixels*sizeof(float), cudaMemcpyHostToDevice);
}

void Frame::updateColorGPU()
{
  const int n_pixels = _H*_W;
  std::vector<uchar4> color_array(n_pixels);
  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      const auto &bgr = _color.at<cv::Vec3b>(h,w);
      color_array[h*_W+w] = make_uchar4(bgr[0],bgr[1],bgr[2],0);
    }
  }
  cudaMemcpy(_color_gpu, color_array.data(), sizeof(uchar4)*color_array.size(), cudaMemcpyHostToDevice);
}

void Frame::updateNormalGPU()
{
  const int n_pixels = _H*_W;
  std::vector<float4> normal_array(n_pixels);
  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      const auto &pt = (*_cloud)(w,h);
      if (pt.z>0.1)
      {
        normal_array[h*_W+w] = make_float4(pt.normal_x, pt.normal_y, pt.normal_z, 0);
      }
      else
      {
        normal_array[h*_W+w] = make_float4(0,0,0,0);
      }
    }
  }
  cudaMemcpy(_normal_gpu, normal_array.data(), sizeof(float4)*normal_array.size(), cudaMemcpyHostToDevice);
}


void Frame::processDepth()
{
  const int n_pixels = _H*_W;

  float *depth_tmp_gpu;
  cudaMalloc(&depth_tmp_gpu, n_pixels*sizeof(float));

  const float sigma_D = (*yml)["depth_processing"]["bilateral_filter"]["sigma_D"].as<float>();
  const float sigma_R = (*yml)["depth_processing"]["bilateral_filter"]["sigma_R"].as<float>();
  const int bf_radius = (*yml)["depth_processing"]["bilateral_filter"]["radius"].as<int>();
  const float erode_ratio = (*yml)["depth_processing"]["erode"]["ratio"].as<float>();
  const float erode_radius = (*yml)["depth_processing"]["erode"]["radius"].as<float>();
  const float erode_diff = (*yml)["depth_processing"]["erode"]["diff"].as<float>();
  const float zfar = (*yml)["depth_processing"]["zfar"].as<float>();

  CUDAImageUtil::erodeDepthMap(depth_tmp_gpu, _depth_gpu, erode_radius, _W,_H, erode_diff, erode_ratio, zfar);
  CUDAImageUtil::gaussFilterDepthMap(_depth_gpu, depth_tmp_gpu, bf_radius, sigma_D, sigma_R, _W, _H, zfar);
  CUDAImageUtil::gaussFilterDepthMap(depth_tmp_gpu, _depth_gpu, bf_radius, sigma_D, sigma_R, _W, _H, zfar);
  std::swap(depth_tmp_gpu,_depth_gpu);

  // CUDAImageUtil::gaussFilterDepthMap(depth_tmp_gpu, _depth_gpu, bf_radius, sigma_D, sigma_R, _W, _H, zfar);
  // CUDAImageUtil::gaussFilterDepthMap(_depth_gpu, depth_tmp_gpu, bf_radius, sigma_D, sigma_R, _W, _H, zfar);

  updateDepthCPU();

  cudaFree(depth_tmp_gpu);

}

void Frame::depthToCloudAndNormals()
{
  const int n_pixels = _H*_W;
  float4 *xyz_map_gpu;
  cudaMalloc(&xyz_map_gpu, n_pixels*sizeof(float4));
  float4x4 K_inv_data;
  K_inv_data.setIdentity();
  Eigen::Matrix3f K_inv = _K.inverse();
  for (int row=0;row<3;row++)
  {
    for (int col=0;col<3;col++)
    {
      K_inv_data(row,col) = K_inv(row,col);
    }
  }
  CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(xyz_map_gpu, _depth_gpu, K_inv_data, _W, _H);

  CUDAImageUtil::computeNormals(_normal_gpu, xyz_map_gpu, _W, _H);

  float *depth_tmp_gpu;
  cudaMalloc(&depth_tmp_gpu, n_pixels*sizeof(float));
  const float angle_thres = (*yml)["depth_processing"]["edge_normal_thres"].as<float>()/180.0*M_PI;
  CUDAImageUtil::filterDepthSmoothedEdges(depth_tmp_gpu,_depth_gpu,_normal_gpu,_W,_H,angle_thres,_K(0,0),_K(1,1),_K(0,2),_K(1,2));
  std::swap(depth_tmp_gpu, _depth_gpu);
  updateDepthCPU();
  cudaFree(depth_tmp_gpu);
  CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(xyz_map_gpu, _depth_gpu, K_inv_data, _W, _H);  //!NOTE first time compute normal to filter edge area's depth, then recompute point cloud

  ///////////// Copy to pcl cloud
  std::vector<float4> xyz_map(n_pixels);
  cudaMemcpy(xyz_map.data(), xyz_map_gpu, sizeof(float4)*n_pixels, cudaMemcpyDeviceToHost);
  std::vector<float4> normals(n_pixels);
  cudaMemcpy(normals.data(), _normal_gpu, sizeof(float4)*n_pixels, cudaMemcpyDeviceToHost);
  _normal_map = cv::Mat::zeros(_H, _W, CV_32FC3);

  _cloud->height = _H;
  _cloud->width = _W;
  _cloud->is_dense = false;
  _cloud->points.resize(_cloud->width * _cloud->height);

  #pragma omp parallel for schedule(dynamic)
  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      const int id = h*_W + w;
      const auto &xyz = xyz_map[id];
      auto &pt = (*_cloud)(w,h);
      if (xyz.z<0.1)
      {
        pt.x = 0;
        pt.y = 0;
        pt.z = 0;
        pt.r = 0;
        pt.g = 0;
        pt.b = 0;
        pt.normal_x = 0;
        pt.normal_y = 0;
        pt.normal_z = 0;
        continue;
      }

      pt.x = xyz.x;
      pt.y = xyz.y;
      pt.z = xyz.z;

      const auto &color = _color.at<cv::Vec3b>(h,w);
      pt.b = color[0];
      pt.g = color[1];
      pt.r = color[2];

      const auto &normal = normals[id];
      pt.normal_x = normal.x;
      pt.normal_y = normal.y;
      pt.normal_z = normal.z;
      _normal_map.at<cv::Vec3f>(h,w) = {normal.x, normal.y, normal.z};
    }
  }

  cudaFree(xyz_map_gpu);
}


void Frame::pointCloudDenoise()
{
  const auto &debug_dir = (*yml)["debug_dir"].as<std::string>();
  Utils::downsamplePointCloud(_cloud, _cloud_down, 0.005);
  Utils::passFilterPointCloud(_cloud_down,_cloud_down,"z",0.1, (*yml)["depth_processing"]["zfar"].as<float>());
  SPDLOG("_cloud_down# {}",_cloud_down->points.size());
  if ((*yml)["SPDLOG"].as<int>()>=3)
  {
    pcl::io::savePLYFile(fmt::format("{}/{}/cloud_before_denoise_down.ply",debug_dir,_id_str), *_cloud_down);
  }

  const auto &num = (*yml)["depth_processing"]["outlier_removal"]["num"].as<int>();
  const auto &std_mul = (*yml)["depth_processing"]["outlier_removal"]["std_mul"].as<float>();
  Utils::outlierRemovalStatistic(_cloud_down, _cloud_down, std_mul, num);

  if ((*yml)["SPDLOG"].as<int>()>=3)
  {
    pcl::io::savePLYFile(fmt::format("{}/{}/cloud_before_euclidean cluster.ply",debug_dir,_id_str), *_cloud_down);
  }

  pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
  tree->setInputCloud(_cloud_down);

  tree->setInputCloud(_cloud_down);
  for (int i=0;i<_cloud->points.size();i++)
  {
    const auto &pt = _cloud->points[i];
    if (pt.z<0.1) continue;
    std::vector<int> ids;
    std::vector<float> sq_dists;
    tree->nearestKSearch(pt, 1, ids, sq_dists);
    if (sq_dists.size()==0) continue;
    if (sq_dists[0]>0.005*0.005)
    {
      Eigen::Vector3f P(pt.x,pt.y,pt.z);
      Eigen::Vector3f projected = _K*P;
      int u = std::round(projected(0)/projected(2));
      int v = std::round(projected(1)/projected(2));
      invalidatePixel(v,u);
      _fg_mask.at<uchar>(v,u) = 0;
    }
  }

  updateDepthGPU();
  // updateColorGPU();
  updateNormalGPU();
  SPDLOG("after denoise cloud #pts: {}",_cloud->points.size());
}

void Frame::updateRoi()
{
  float umin = 9999;
  float vmin = 9999;
  float umax = 0;
  float vmax = 0;
  if (_fg_mask.empty())
  {
    _roi<<0,_W-1,0,_H-1;
    return;
  }

  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      if (_fg_mask.at<uchar>(h,w)>0)
      {
        umin = std::min(umin, float(w));
        umax = std::max(umax, float(w));
        vmin = std::min(vmin, float(h));
        vmax = std::max(vmax, float(h));
      }
    }
  }
  _roi<<umin,umax,vmin,vmax;
}



void Frame::invalidatePixel(const int h, const int w)
{
  _color.at<cv::Vec3b>(h,w) = {0,0,0};
  _depth.at<float>(h,w) = 0;
  _gray.at<uchar>(h,w) = 0;
  {
    auto &pt = (*_cloud)(w,h);
    pt.x = 0;
    pt.y = 0;
    pt.z = 0;
    pt.normal_x = 0;
    pt.normal_y = 0;
    pt.normal_z = 0;
  }
}

void Frame::invalidatePixelsByMask(const cv::Mat &fg_mask)
{
  _fg_mask = fg_mask.clone();
  updateRoi();

  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      if (fg_mask.at<uchar>(h,w)==0)
      {
        invalidatePixel(h,w);
      }
    }
  }

  // updateColorGPU();
  updateDepthGPU();
  updateNormalGPU();
}

int Frame::countValidPoints()
{
  int cnt = 0;
  for (int h=_roi(2);h<_roi(3);h++)
  {
    for (int w=_roi(0);w<_roi(1);w++)
    {
      if (_depth.at<float>(h,w)>=0.1) cnt++;
    }
  }
  return cnt;
}



bool Frame::operator < (const Frame &other)
{
  if (_id<other._id) return true;
  return false;
}

