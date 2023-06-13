/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef COMMON_IO__HH
#define COMMON_IO__HH

#include <boost/serialization/array.hpp>
#define EIGEN_DENSEBASE_PLUGIN "EigenDenseBaseAddons.h"

// STL
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <time.h>
#include <queue>
#include <climits>
#include <boost/assign.hpp>
#include <boost/algorithm/string.hpp>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <string>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <math.h>
#include <boost/format.hpp>
#include <numeric>
#include <thread>
#include <omp.h>
#include <exception>
#include <deque>
#include <random>
#include <bits/stdc++.h>

// For IO
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>

// For OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include "opencv2/calib3d/calib3d.hpp"

// For Visualization
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/cloud_viewer.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>
#include <pcl/common/distances.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
#include <pcl/recognition/ransac_based/trimmed_icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/features/ppf.h>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/xpressive/xpressive.hpp>
#include <regex>
#include <pcl/features/integral_image_normal.h>
#include <pcl/tracking/normal_coherence.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/features/principal_curvatures.h>
#include "yaml-cpp/yaml.h"
#include <unordered_map>
#include <unsupported/Eigen/NonLinearOptimization>
#include <boost/filesystem.hpp>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/bundled/ostream.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

#ifdef SPDLOG
#undef SPDLOG
#endif
#define SPDLOG SPDLOG_WARN
#define SPDLOG_USE_MY_FORMAT spdlog::set_pattern("%^[%D %H:%M:%S.%F][%@] %v%$");

namespace py = pybind11;

// definitions
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudRGBNormal;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudNormal;
typedef pcl::PointCloud<pcl::PointSurfel> PointCloudSurfel;
typedef pcl::PointCloud<pcl::PrincipalCurvatures> PointCloudCurvatures;
typedef std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > PoseVector;
using uchar = unsigned char;
using VectorMatrix4x4 = std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> >;
using VectorEigenVector4f = std::vector<Eigen::Vector4f,Eigen::aligned_allocator<Eigen::Vector4f> >;
using ll = long long;
using ull = unsigned long long;

namespace Utils
{
class Timer
{
public:
  Timer(std::string func_name="") : _func_name(func_name)
  {
    reset();
  }
  void reset()
  {
#if TIMER
    cudaDeviceSynchronize();
    beg_ = clock_::now();
#endif
  }
  double elapsed()
  {
#if TIMER
    cudaDeviceSynchronize();
    return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
#endif
  }
  void print(std::string message = "")
  {
#if TIMER
    double t = elapsed();
    fmt::print("[TIMER][{}][{}] elasped time: {: f}\n", _func_name, message, t);
    reset();
#endif
  }

public:
  std::string _func_name;

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1>> second_;
  std::chrono::time_point<clock_> beg_;

};

std::string type2str(int type);
int str2type(std::string s);
float rotationGeodesicDistance(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2);
float rotationGeodesicDistanceIgnoreRotationAroundCamZ(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2);
template<class PointT>
void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<PointT>> objCloud);
template<class PointT>
float pointToPlaneICP(boost::shared_ptr<pcl::PointCloud<PointT> > pclSegment,
                     boost::shared_ptr<pcl::PointCloud<PointT> > pclModel,
                     Eigen::Matrix4f &offsetTransform, int max_iter=30, float rejection_angle=60, float max_corres_dist=0.01, float score_thres=1e-4);

float pointToPointICP(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr src,pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dst,                Eigen::Matrix4f &offsetTransform, int max_iter, float rejection_angle, float max_corres_dist, float score_thres);


template<class PointT>
void outlierRemovalRadius(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, float radius, int min_num);

template<class PointT>
void outlierRemovalStatistic(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, float std_mul, int num);

template<class PointType>
void downsamplePointCloud(boost::shared_ptr<pcl::PointCloud<PointType> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointType> > cloud_out, float vox_size);

template<class PointT>
void passFilterPointCloud(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, const std::string &axis, float min, float max);

template <class PointT>
void calNormalIntegralImage(boost::shared_ptr<pcl::PointCloud<PointT>> cloud, int method, float max_depth_change_factor, float smooth_size,bool depth_dependent_smooth);


template <typename T>
std::vector<int> vectorArgsort(const std::vector<T> &v, bool min_to_max);


void normalizeRotationMatrix(Eigen::Matrix3f &R);
void normalizeRotationMatrix(Eigen::Matrix4f &pose);

template<class PointT>
void cloudAMinusCloudB(boost::shared_ptr<pcl::PointCloud<PointT>> cloudA, boost::shared_ptr<pcl::PointCloud<PointT>> cloudB, boost::shared_ptr<pcl::PointCloud<PointT>> cloud_out, const float dist_thres);

bool isPixelInsideImage(const int H, const int W, float u, float v);

template<class PointT>
bool isPclPointNormalValid(PointT pt);

float trilinearInterpolation(float x, float y, float z, const Eigen::Matrix<float,8,3> &corners, const Eigen::Matrix<float,8,1> &values);

void solveRigidTransformBetweenPoints(const Eigen::MatrixXf &points1, const Eigen::MatrixXf &points2, Eigen::Matrix4f &pose);
void computeMatrixMeanAndCov(const Eigen::MatrixXf &X, Eigen::VectorXf &mean, Eigen::MatrixXf &cov);

template<class T>
bool isImageChangedAbsDiff(const cv::Mat &imgA, const cv::Mat &imgB, cv::Mat mask, float &change_value, float thres);

bool isImageChangedOpticalFlow(const cv::Mat &imgA, const cv::Mat &imgB, const cv::Mat &maskB, float &change_value, float thres);
void getRotateImageTransform(int H, int W, float rot, Eigen::Matrix3f &forward_transform);
void getNearestValidPixelOnDepth(const cv::Mat &depth, int &u, int &v);

/**
 * @brief
 *
 * @tparam T1
 * @tparam T2
 * @param points : (N,D)
 * @param tf : (D+1, D+1) homogeneous transform
 * @return Eigen::MatrixBase<T1>
 */
template<class T1, class T2, int D>
Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> transformPoints(const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> &points, const Eigen::Matrix<T2, D, D> &tf)
{
  const int n_dim = points.cols();
  return ((tf.block(0,0,n_dim,n_dim)*points.transpose()).colwise() + Eigen::Matrix<T2, Eigen::Dynamic, 1>(tf.block(0,n_dim,n_dim,1))).transpose();
}


template<typename Derived>
inline bool isMatrixFinite(const Eigen::MatrixBase<Derived>& x)
{
	return (x.array().isFinite()).all();
};

} // namespace Utils



namespace pcl
{
 template <typename PointT, typename Scalar>
 inline PointT transformPointWithNormal(const PointT &point, const Eigen::Matrix<Scalar,4,4> &transform)
 {
   PointT ret = point;
   pcl::detail::Transformer<Scalar> tf (transform);
   tf.se3 (point.data, ret.data);
   tf.so3 (point.data_n, ret.data_n);
   return (ret);
 }

template <typename PointT, typename Scalar>
 inline PointT transformPoint(const PointT &point, const Eigen::Matrix<Scalar,4,4> &transform)
 {
   PointT ret = point;
   pcl::detail::Transformer<Scalar> tf (transform);
   tf.se3 (point.data, ret.data);
   return (ret);
 }

};



#endif