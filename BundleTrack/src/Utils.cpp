/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "Utils.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>


namespace Utils
{


std::string type2str(int type)
{
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int str2type(std::string s)
{
  std::vector<int> enum_ints = {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                                CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                                CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                                CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                                CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                                CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                                CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

  std::vector<std::string> enum_strings = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                                          "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                                          "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                                          "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                                          "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                                          "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                                          "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

  if (enum_ints.size()!=enum_strings.size())
  {
    std::cout<<"Lookup table size is wrong\n";
    exit(1);
  }

  auto it = std::find(enum_strings.begin(),enum_strings.end(),s);
  if (it==enum_strings.end())
  {
    std::cout<<"unknown image type"<<std::endl;
    exit(1);
  }
  int pos = std::distance(enum_strings.begin(),it);
  return enum_ints[pos];
}


// Difference angle in radian
float rotationGeodesicDistance(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2)
{
  float tmp = ((R1 * R2.transpose()).trace()-1) / 2.0;
  tmp = std::max(std::min(1.0f, tmp), -1.0f);
  return std::acos(tmp);
}


float rotationGeodesicDistanceIgnoreRotationAroundCamZ(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2)
{
  Eigen::Matrix3f R_AB_in_cam = R2*R1.inverse();
  Eigen::AngleAxis<float> axis_angle;
  axis_angle.fromRotationMatrix(R_AB_in_cam);
  axis_angle.axis()(2) = 0;  // Make rotation around camera z axis zero
  axis_angle.axis().normalize();
  Eigen::Matrix3f out = axis_angle.toRotationMatrix();
  float diff = rotationGeodesicDistance(out,Eigen::Matrix3f::Identity());
  return diff;
}


/********************************* function: convert3dOrganizedRGB<pcl::PointXYZRGB> *************************************
	Description: Convert Depth image to point cloud. TODO: Could it be faster?
	Reference: https://gist.github.com/jacyzon/fa868d0bcb13abe5ade0df084618cf9c
colImage: 8UC3
objDepth: 16UC1
	*******************************************************************************************************/
template<class PointT>
void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<PointT>> objCloud)
{
  const int imgWidth = objDepth.cols;
  const int imgHeight = objDepth.rows;

  objCloud->height = (uint32_t)imgHeight;
  objCloud->width = (uint32_t)imgWidth;
  objCloud->is_dense = false;
  objCloud->points.resize(objCloud->width * objCloud->height);

  const float bad_point = 0;  // this can cause the point cloud visualization problem !!

  for (int u = 0; u < imgHeight; u++)
    for (int v = 0; v < imgWidth; v++)
    {
      float depth = objDepth.at<float>(u, v);
      cv::Vec3b colour = colImage.at<cv::Vec3b>(u, v); // 3*8 bits
      if (depth > 0.1 && depth < 2.0)
      {
        (*objCloud)(v, u).x = (float)((v - camIntrinsic(0, 2)) * depth / camIntrinsic(0, 0));
        (*objCloud)(v, u).y = (float)((u - camIntrinsic(1, 2)) * depth / camIntrinsic(1, 1));
        (*objCloud)(v, u).z = depth;
        (*objCloud)(v, u).b = colour[0];
        (*objCloud)(v, u).g = colour[1];
        (*objCloud)(v, u).r = colour[2];
      }
      else
      {
        (*objCloud)(v, u).x = bad_point;
        (*objCloud)(v, u).y = bad_point;
        (*objCloud)(v, u).z = bad_point;
        (*objCloud)(v, u).b = 0;
        (*objCloud)(v, u).g = 0;
        (*objCloud)(v, u).r = 0;
      }
    }
}
template void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> objCloud);
template void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal>> objCloud);




/********************************* function: pointToPlaneICP *******************************************
	*******************************************************************************************************/

//@src: source
//@dst: target
//@score_thres: squared dist thres to account for final per point dist
template<class PointT>
float pointToPlaneICP(boost::shared_ptr<pcl::PointCloud<PointT> > src, boost::shared_ptr<pcl::PointCloud<PointT> > dst, Eigen::Matrix4f &offsetTransform, int max_iter, float rejection_angle, float max_corres_dist, float score_thres)
{

  PointCloudNormal::Ptr modelCloud(new PointCloudNormal);
  PointCloudNormal::Ptr segmentCloud(new PointCloudNormal);
  PointCloudNormal::Ptr segCloudTrans(new PointCloudNormal);
  copyPointCloud(*dst, *modelCloud);
  copyPointCloud(*src, *segmentCloud);

  std::vector<int> indices;
  pcl::removeNaNNormalsFromPointCloud(*modelCloud, *modelCloud, indices);
  pcl::removeNaNNormalsFromPointCloud(*segmentCloud, *segmentCloud, indices);

  pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> reg;

  // pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>::Ptr trans_lls (
  //     new pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>);
  pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>::Ptr trans_lls (
      new pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>);
  // pcl::registration::TransformationEstimationPointToPlaneWeighted<pcl::PointNormal, pcl::PointNormal>::Ptr trans_lls (
  //     new pcl::registration::TransformationEstimationPointToPlaneWeighted<pcl::PointNormal, pcl::PointNormal>);
  // trans_lls->setUseCorrespondenceWeights(true);

  // pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointNormal, pcl::PointNormal, pcl::PointNormal>::Ptr cens (
  //     new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointNormal, pcl::PointNormal, pcl::PointNormal>);
  // pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointNormal, pcl::PointNormal, pcl::PointNormal>::Ptr cens (
  //     new pcl::registration::CorrespondenceEstimationBackProjection<pcl::PointNormal, pcl::PointNormal, pcl::PointNormal>);
  pcl::registration::CorrespondenceEstimation<pcl::PointNormal, pcl::PointNormal>::Ptr cens (
    new pcl::registration::CorrespondenceEstimation<pcl::PointNormal, pcl::PointNormal>);

  // cens->setInputSource (segmentCloud);
  // cens->setInputTarget (modelCloud);
  // cens->setSourceNormals (segmentCloud);

  // pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointNormal>::Ptr rej (
  //     new pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointNormal> ());

  // rej->setInputSource (segmentCloud);
  // rej->setInputTarget (modelCloud);
  // rej->setMaximumIterations (100);
  // rej->setInlierThreshold (0.01);

  pcl::registration::CorrespondenceRejectorSurfaceNormal::Ptr rej1 (new pcl::registration::CorrespondenceRejectorSurfaceNormal);
  rej1->setThreshold (std::cos(rejection_angle/180.0*M_PI));   // we dont need to add src and tgt here!

  reg.convergence_criteria_->setRelativeMSE(1e-10);
  reg.convergence_criteria_->setAbsoluteMSE(1e-6);

  reg.setInputSource (segmentCloud);
  reg.setInputTarget (modelCloud);

  reg.setCorrespondenceEstimation (cens);
  reg.setTransformationEstimation (trans_lls);
  // reg.addCorrespondenceRejector (rej);
  reg.addCorrespondenceRejector (rej1);
  reg.setMaximumIterations (max_iter);
  // reg.setUseReciprocalCorrespondences(true);
  reg.setMaxCorrespondenceDistance (max_corres_dist);
  // reg.setTransformationEpsilon (1e-8);
  // reg.setTransformationRotationEpsilon(1e-8);
  reg.align (*segCloudTrans);

  // float avg_dist = computeAverageDistance<pcl::PointNormal>(segCloudTrans,modelCloud);
  // std::cout<<"avg_dist = "<<avg_dist<<std::endl;



  if (reg.hasConverged())
  {
    offsetTransform = reg.getFinalTransformation();
  }
  else
  {
    std::cout << "ICP did not converge." << std::endl;
    offsetTransform.setIdentity();
  }

  return reg.getFitnessScore(score_thres);

}
template float pointToPlaneICP<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > pclSegment, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > pclModel, Eigen::Matrix4f &offsetTransform, int max_iter, float rejection_angle, float max_corres_dist, float score_thres);
template float pointToPlaneICP<pcl::PointSurfel>(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > pclSegment, boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > pclModel, Eigen::Matrix4f &offsetTransform, int max_iter, float rejection_angle, float max_corres_dist, float score_thres);




template<class PointT>
bool isPclPointNormalValid(PointT pt)
{
  if (!std::isfinite(pt.normal_x) || !std::isfinite(pt.normal_y) || !std::isfinite(pt.normal_z)) return false;
  if (pt.normal_x==0 && pt.normal_y==0 && pt.normal_z==0) return false;
  return true;
}
template bool isPclPointNormalValid(pcl::PointXYZRGBNormal pt);




template<class PointT>
void outlierRemovalRadius(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, float radius, int min_num)
{
  pcl::RadiusOutlierRemoval<PointT> outrem;
  outrem.setInputCloud(cloud_in);
  outrem.setRadiusSearch(radius);
  outrem.setMinNeighborsInRadius (min_num);
  outrem.filter(*cloud_out);
}
template void outlierRemovalRadius(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_out, float radius, int min_num);


template<class PointT>
void outlierRemovalStatistic(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, float std_mul, int num)
{
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(cloud_in);
  sor.setMeanK(num);
  sor.setStddevMulThresh(std_mul);
  sor.filter(*cloud_out);
}
template void outlierRemovalStatistic(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_out, float std_mul, int num);

template<class PointT>
void downsamplePointCloud(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, float vox_size)
{
  pcl::VoxelGrid<PointT> vox;
  vox.setInputCloud(cloud_in);
  vox.setLeafSize(vox_size, vox_size, vox_size);
  vox.filter(*cloud_out);
}
template void downsamplePointCloud<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointXYZRGB>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointXYZ>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > cloud_out, float vox_size);
template void downsamplePointCloud<pcl::PointSurfel>(boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointSurfel> > cloud_out, float vox_size);

template<class PointT>
void passFilterPointCloud(boost::shared_ptr<pcl::PointCloud<PointT> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointT> > cloud_out, const std::string &axis, float min, float max)
{
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud (cloud_in);
  pass.setFilterFieldName (axis);
  pass.setFilterLimits (min, max);
  pass.filter (*cloud_out);
}
template void passFilterPointCloud<pcl::PointXYZRGBNormal>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> > cloud_out, const std::string &axis, float min, float max);
template void passFilterPointCloud<pcl::PointXYZRGB>(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_in, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > cloud_out, const std::string &axis, float min, float max);

//Return argsort indices
template <typename T>
std::vector<int> vectorArgsort(const std::vector<T> &v, bool min_to_max)
{

  // initialize original index locations
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  if (min_to_max)
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] < v[i2]; });
  else
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] > v[i2]; });

  return idx;
}
template std::vector<int> vectorArgsort(const std::vector<float> &v, bool min_to_max);
template std::vector<int> vectorArgsort(const std::vector<int> &v, bool min_to_max);



void normalizeRotationMatrix(Eigen::Matrix3f &R)
{
  for (int col=0;col<3;col++)
  {
    R.col(col).normalize();
  }
}

void normalizeRotationMatrix(Eigen::Matrix4f &pose)
{
  for (int col=0;col<3;col++)
  {
    pose.block(0,col,3,1).normalize();
  }
}


bool isPixelInsideImage(const int H, const int W, float u, float v)
{
  u = std::round(u);
  v = std::round(v);
  if (u<0 || u>=W || v<0 || v>=H) return false;
  return true;
}

/**
 * @brief solve for pose 1 to 2 http://nghiaho.com/?page_id=671
 *
 * @param points1 : Nx3
 * @param points2
 * @param pose
 */
void solveRigidTransformBetweenPoints(const Eigen::MatrixXf &points1, const Eigen::MatrixXf &points2, Eigen::Matrix4f &pose)
{
  assert(points1.cols()==3 && points1.rows()>=3 && points2.cols()==3 && points2.rows()>=3);
  pose.setIdentity();

  Eigen::Vector3f mean1 = points1.colwise().mean();
  Eigen::Vector3f mean2 = points2.colwise().mean();
  // std::cout<<"mean1\n"<<mean1<<"\n\n";
  // std::cout<<"mean2\n"<<mean2<<"\n\n";

  Eigen::MatrixXf P = points1.rowwise() - mean1.transpose();
  Eigen::MatrixXf Q = points2.rowwise() - mean2.transpose();
  // std::cout<<"P\n"<<P<<"\n\n";
  // std::cout<<"Q\n"<<Q<<"\n\n";
  Eigen::MatrixXf S = P.transpose() * Q;     //3x3
  assert(S.rows()==3 && S.cols()==3);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
  // std::cout<<"singularValues: "<<svd.singularValues().transpose()<<std::endl;
  // std::cout<<"S\n"<<S<<"\n\n";
  // std::cout<<"svd.matrixU()\n"<<svd.matrixU()<<"\n\n";
  // std::cout<<"svd.matrixV()\n"<<svd.matrixV()<<"\n\n";
  Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
  if ( !((R.transpose()*R).isApprox(Eigen::Matrix3f::Identity())) )
  {
    std::cout<<"solveRigidTransformBetweenPoints failed, R matrix is not orthonormal\n";
    pose.setIdentity();
    return;
  }

  if (R.determinant()<0)
  {
    auto V_new = svd.matrixV();
    V_new.col(2) = (-V_new.col(2)).eval();
    R = V_new * svd.matrixU().transpose();
  }
  pose.block(0,0,3,3) = R;
  pose.block(0,3,3,1) = mean2 - R * mean1;
  if (!isMatrixFinite(pose))
  {
    std::cout<<"solveRigidTransformBetweenPoints failed, got infinite number in pose, set to I\n";
    pose.setIdentity();
    return;
  }

}




/**
 * @brief Find the transformation that rotates the image while ensuring the rotated image pixels are all within bound
 *
 * @param H
 * @param W
 * @param rot
 * @param forward_transform
 */
void getRotateImageTransform(int H, int W, float rot, Eigen::Matrix3f &forward_transform)
{
  Eigen::Matrix3f tf(Eigen::Matrix3f::Identity());
  tf(0,2) -= W/2.0;
  tf(1,2) -= H/2.0;
  forward_transform = tf*forward_transform;
  tf.setIdentity();
  tf.block(0,0,2,2) << std::cos(rot), -std::sin(rot), std::sin(rot), std::cos(rot);
  forward_transform = tf*forward_transform;

  Eigen::Matrix<float,4,3> corners;
  corners<< 0,0,1,
            W,0,1,
            0,H,1,
            W,H,1;
  Eigen::MatrixXf transformed_corners = (forward_transform*corners.transpose()).transpose();
  float umin = transformed_corners.col(0).minCoeff();
  float umax = transformed_corners.col(0).maxCoeff();
  float vmin = transformed_corners.col(1).minCoeff();
  float vmax = transformed_corners.col(1).maxCoeff();
  int side = std::max(umax-umin, vmax-vmin);

  tf.setIdentity();
  tf(0,2) = side/2.0;
  tf(1,2) = side/2.0;
  forward_transform = tf*forward_transform;
}


} // namespace Utils

