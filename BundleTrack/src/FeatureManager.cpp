/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "FeatureManager.h"
#include "cuda_ransac.h"
#include <opencv2/cudafeatures2d.hpp>

using namespace std;



/**
 * @brief
 *
 * @param colorA
 * @param colorB
 * @param corres (N,4)
 * @param out
 */
void cvDrawMatches(const cv::Mat &colorA, const cv::Mat &colorB, const std::vector<Correspondence> &corres, cv::Mat &out)
{
  const int H_out = 900;
  const float scaleA = float(H_out)/colorA.rows;
  const float scaleB = float(H_out)/colorB.rows;

  cv::Mat A,B;

  cv::resize(colorA, A,{int(colorA.cols*scaleA), H_out});
  cv::resize(colorB, B,{int(colorB.cols*scaleA), H_out});

  cv::hconcat(A,B,out);
  cv::Mat conf_mat = cv::Mat::zeros(corres.size(), 1, CV_8UC1);

  for (int i=0;i<corres.size();i++)
  {
    const auto &corr = corres[i];
    conf_mat.at<uchar>(i,0) = corr._confidence*255;
  }
  double min, max;
  cv::minMaxLoc(conf_mat, &min, &max);
  if (min==max)   // No valid confidence, use random color
  {
    conf_mat = cv::Mat::zeros(corres.size(), 1, CV_8UC3);
    for (int i=0;i<corres.size();i++)
    {
      for (int j=0;j<3;j++)
      {
        conf_mat.at<cv::Vec3b>(i,0)[j] = uchar(rand()/float(RAND_MAX)*255.0);
      }
    }
  }
  else
  {
    cv::applyColorMap(conf_mat,conf_mat,cv::COLORMAP_JET);
  }


  for (int i=0;i<corres.size();i++)
  {
    const auto &corr = corres[i];
    const auto &color = conf_mat.at<cv::Vec3b>(i,0);
    int uA = std::round(corr._uA*float(A.cols)/colorA.cols);
    int vA = std::round(corr._vA*float(A.rows)/colorA.rows);
    int uB = std::round(corr._uB*float(B.cols)/colorB.cols+A.cols);
    int vB = std::round(corr._vB*float(B.rows)/colorB.rows);
    cv::circle(out, {uA, vA}, 3, color, 1);
    cv::circle(out, {uB, vB}, 3, color, 1);
    cv::line(out, {uA,vA}, {uB,vB}, color, 1);
  }
}



void processImage(const std::shared_ptr<Frame> &frame, const int out_side, cv::Mat &window, Eigen::Matrix3f &tf)
{
  tf.setIdentity();
  Eigen::Matrix3f tf1(Eigen::Matrix3f::Identity());

  const int H = frame->_H;
  const int W = frame->_W;
  const auto &roi = frame->_roi;
  int side = std::max(roi(1)-roi(0)+1,roi(3)-roi(2)+1);
  window = cv::Mat::zeros(side,side,CV_8UC3);
  for (int h=roi(2);h<roi(3)+1;h++)
  {
    for (int w=roi(0);w<roi(1)+1;w++)
    {
      window.at<cv::Vec3b>(h-roi(2),w-roi(0)) = frame->_color.at<cv::Vec3b>(h,w);
    }
  }
  tf1.setIdentity();
  tf1(0,2) = -roi(0);
  tf1(1,2) = -roi(2);
  tf = tf1 * tf;
  Eigen::AngleAxisf axis_angle;
  Eigen::Matrix3f R = frame->_pose_in_model.block(0,0,3,3).transpose();
  axis_angle.fromRotationMatrix(R);
  float rot_deg = axis_angle.axis()(2)*axis_angle.angle();
  Utils::getRotateImageTransform(H, W, rot_deg, tf1);
  tf = tf1 * tf;
  int H0 = window.rows;
  int W0 = window.cols;
  cv::resize(window, window, {out_side, out_side});
  tf1.setIdentity();
  tf1(0,0) = out_side/float(W0);
  tf1(1,1) = out_side/float(H0);
  tf = tf1 * tf;

  // cv::imshow("1",window);
  // cv::Mat tmp,M;
  // cv::eigen2cv(tf,M);
  // cv::warpPerspective(frame->_color,tmp,M,{out_side,out_side});
  // cv::imshow("2",tmp);
  // cv::waitKey(0);
};


void processImagePair(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, cv::Mat &outA, cv::Mat &outB, const int out_size, const bool use_gray, Eigen::Matrix3f &tfA, Eigen::Matrix3f &tfB)
{
  tfA.setIdentity();
  tfB.setIdentity();

  const auto &roiA = frameA->_roi;
  const auto &roiB = frameB->_roi;

  const int H = frameB->_H;
  const int W = frameB->_W;

  Eigen::Matrix3f tf(Eigen::Matrix3f::Identity());

  ////////// Rotate B to A
  Eigen::Matrix3f RA = frameA->_pose_in_model.block(0,0,3,3).transpose();
  Eigen::Matrix3f RB = frameB->_pose_in_model.block(0,0,3,3).transpose();
  Eigen::Matrix3f R_BA_in_cam = RA*RB.inverse();
  Eigen::AngleAxis<float> axis_angle;
  axis_angle.fromRotationMatrix(R_BA_in_cam);
  tf.setIdentity();
  Utils::getRotateImageTransform(H, W, axis_angle.angle()*axis_angle.axis()(2), tf);
  tfB = tf*tfB;

  Eigen::Matrix<float,4,3> corners;
  corners<< roiB(0),roiB(2),1,
            roiB(0),roiB(3),1,
            roiB(1),roiB(2),1,
            roiB(1),roiB(3),1;
  Eigen::MatrixXf transformed_corners = (tfB*corners.transpose()).transpose();
  float umin = transformed_corners.col(0).minCoeff();
  float umax = transformed_corners.col(0).maxCoeff();
  float vmin = transformed_corners.col(1).minCoeff();
  float vmax = transformed_corners.col(1).maxCoeff();

  Eigen::Vector2f uvB_min(umin,vmin);
  Eigen::Vector2f uvB_max(umax,vmax);

  ////////!DEBUG
  // if (frameA->_id>=60)
  // {
  //   cv::imshow("rotated B",outB);

  //   cv::Mat vis = frameB->_color.clone();
  //   cv::rectangle(vis,{roiB(0),roiB(2)}, {roiB(1),roiB(3)}, {0,255,0});
  //   cv::imshow("B",vis);
  //   cv::Mat M;
  //   cv::eigen2cv(tfB,M);
  //   int H = outB.rows;
  //   int W = outB.cols;
  //   cv::warpPerspective(frameB->_color, outB, M, {W,H});
  //   cv::rectangle(outB,{uvB_min(0),uvB_min(1)}, {uvB_max(0),uvB_max(1)}, {0,255,0});
  //   cv::imshow("0",outB);
  // }


  const int margin = 10;
  tf.setIdentity();
  tf(0,2) = -roiA(0)+margin;
  tf(1,2) = -roiA(2)+margin;
  tfA = tf*tfA;
  tf.setIdentity();
  tf(0,2) = -uvB_min(0)+margin;
  tf(1,2) = -uvB_min(1)+margin;
  tfB = tf*tfB;


  //////!DEBUG
  // if (frameA->_id>=60)
  // {
  //   SPDLOG("uvB={},{}  {},{}",uvB_min(0),uvB_min(1),uvB_max(0),uvB_max(1));
  //   cv::Mat M;
  //   cv::eigen2cv(tfB,M);
  //   cv::warpPerspective(frameB->_color, outB, M, {out_size,out_size});
  //   cv::imshow("1",outB);
  // }

  int WA = roiA(1)-roiA(0)+margin*2;   // Pad margin around 4 sides
  int HA = roiA(3)-roiA(2)+margin*2;
  int HB = uvB_max(1)-uvB_min(1)+margin*2;
  int WB = uvB_max(0)-uvB_min(0)+margin*2;
  int max_dim = std::max({WA,HA,WB,HB});
  tf.setIdentity();
  tf.block(0,0,2,2) *= float(max_dim)/std::max(WA,HA);
  tfA = tf*tfA;
  tf.setIdentity();
  tf.block(0,0,2,2) *= float(max_dim)/std::max(WB,HB);
  tfB = tf*tfB;

  ////////!DEBUG
  // if (frameA->_id>=60)
  // {
  //   cv::Mat M;
  //   cv::eigen2cv(tfB,M);
  //   cv::warpPerspective(frameB->_color, outB, M, {out_size,out_size});
  //   cv::imshow("2",outB);
  // }

  tf.setIdentity();
  tf.block(0,0,2,2) *= float(out_size)/max_dim;
  tfA = tf*tfA;
  tfB = tf*tfB;

  ////////!DEBUG
  // if (frameA->_id>=60)
  // {
  //   cv::Mat M;
  //   cv::eigen2cv(tfB,M);
  //   cv::warpPerspective(frameB->_color, outB, M, {out_size,out_size});
  //   cv::imshow("3",outB);
  //   cv::waitKey(0);
  // }

  cv::Mat M;
  cv::eigen2cv(tfA,M);
  if (use_gray)
  {
    cv::warpPerspective(frameA->_gray, outA, M, {out_size,out_size});
  }
  else
  {
    cv::warpPerspective(frameA->_color, outA, M, {out_size,out_size});
  }
  cv::eigen2cv(tfB,M);
  if (use_gray)
  {
    cv::warpPerspective(frameB->_gray, outB, M, {out_size,out_size});
  }
  else
  {
    cv::warpPerspective(frameB->_color, outB, M, {out_size,out_size});
  }
}


MapPoint::MapPoint()
{

}

MapPoint::MapPoint(std::shared_ptr<Frame> frame, float u, float v)
{
  _img_pt[frame] = {u,v};
}

MapPoint::~MapPoint()
{

}

Correspondence::Correspondence()
{
  _uA = -1;
  _vA = -1;
  _uB = -1;
  _vB = -1;
  _isinlier = false;
  _ispropogated = false;
}


Correspondence::Correspondence(float uA, float vA, float uB, float vB, pcl::PointXYZRGBNormal ptA_cam, pcl::PointXYZRGBNormal ptB_cam, bool isinlier) : _uA(uA), _uB(uB), _vA(vA), _vB(vB), _ptA_cam(ptA_cam), _ptB_cam(ptB_cam), _isinlier(isinlier), _ispropogated(false)
{

}

Correspondence::~Correspondence()
{

}

bool Correspondence::operator == (const Correspondence &other) const
{
  if (_uA==other._uA && _uB==other._uB && _vA==other._vA && _vB==other._vB) return true;
  return false;
}


SiftManager::SiftManager(std::shared_ptr<YAML::Node> yml1, Bundler *bundler) : _rng(std::random_device{}())
{
  _bundler = bundler;
  yml = yml1;
  srand(0);
  _rng.seed(0);

  // https://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
  // _detector = cv::xfeatures2d::SIFT::create(0,(*yml)["sift"]["nOctaveLayers"].as<int>(),(*yml)["sift"]["contrastThreshold"].as<float>(),(*yml)["sift"]["edgeThreshold"].as<float>(),(*yml)["sift"]["sigma"].as<float>());

}

SiftManager::~SiftManager()
{

}


/**
 * @brief https://stackoverflow.com/questions/5461148/sift-implementation-with-opencv-2-2
 *
 */
void SiftManager::detectFeature(std::shared_ptr<Frame> frame)
{
  if (frame->_keypts.size()>0) return;
  std::vector<float> scales = (*yml)["sift"]["scales"].as<std::vector<float>>();
  for (int i=0;i<scales.size();i++)
  {
    const auto &scale = scales[i];
    cv::Mat cur;
    cv::resize(frame->_gray, cur, {0,0}, scale,scale);
    std::vector<cv::KeyPoint> keypts;
    cv::Mat des;
    _detector->detectAndCompute(cur, cv::noArray(), keypts, des);
    for (int ii=0;ii<keypts.size();ii++)
    {
      keypts[ii].pt.x = keypts[ii].pt.x/scale;
      keypts[ii].pt.y = keypts[ii].pt.y/scale;
    }
    frame->_keypts.insert(frame->_keypts.end(),keypts.begin(),keypts.end());
    if (frame->_feat_des.rows>0)
      cv::vconcat(frame->_feat_des,des,frame->_feat_des);
    else
      frame->_feat_des = des;
  }

  SPDLOG("frame->_feat_des rows={}, cols={}, dims={}",frame->_feat_des.rows, frame->_feat_des.cols, frame->_feat_des.dims);

}

void SiftManager::rejectFeatures(std::shared_ptr<Frame> frame)
{
  const float max_view_normal_angle = (*yml)["feature_corres"]["max_view_normal_angle"].as<float>()*M_PI/180.0;
  const float cos_thres = std::cos(max_view_normal_angle);

  const int H = frame->_H;
  const int W = frame->_W;
  auto kpts_tmp = frame->_keypts;
  auto kfeat_tmp = frame->_feat_des.clone();
  frame->_keypts.clear();
  frame->_feat_des.release();
  for (int i=0;i<kpts_tmp.size();i++)
  {
    const auto &uv = kpts_tmp[i].pt;
    int u = std::round(uv.x);
    int v = std::round(uv.y);
    if (!Utils::isPixelInsideImage(H, W,u,v)) continue;
    if (frame->_depth.at<float>(v,u)<0.1) continue;
    // if (Utils::isPclPointNormalValid(pt))
    // {
    //   Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
    //   Eigen::Vector3f view(-pt.x, -pt.y, -pt.z);
    //   normal.normalize();
    //   view.normalize();
    //   if(std::cos(normal.dot(view)) >= cos_thres)
    //   {
    //     frame->_keypts.push_back(kpts_tmp[i]);
    //     frame->_feat_des.push_back(kfeat_tmp.row(i));
    //   }
    // }
    // else
    // {
    //   frame->_keypts.push_back(kpts_tmp[i]);
    //   frame->_feat_des.push_back(kfeat_tmp.row(i));
    // }
    frame->_keypts.push_back(kpts_tmp[i]);
    frame->_feat_des.push_back(kfeat_tmp.row(i));
  }

}

void SiftManager::correspondenceFilterWinnerTakeAll(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  const auto &matches = _matches[{frameA,frameB}];

  // Eigen::Matrix4f pose = procrustesByCorrespondence(frameA,frameB);
  // Eigen::Matrix4f A_in_model = pose*frameA->_pose_in_model;

  const int patch_size = (*yml)["feature_corres"]["suppression_patch_size"].as<int>();

  std::map<std::pair<int,int>, std::vector<Correspondence>> grid;
  for (int i=0;i<matches.size();i++)
  {
    const auto &match = matches[i];
    if (match._isinlier==false) continue;
    int wid = std::floor(match._uA/patch_size);
    int hid = std::floor(match._vA/patch_size);
    grid[{wid, hid}].push_back(match);
  }

  _matches[{frameA,frameB}].clear();
  for (const auto &g:grid)
  {
    const auto &matches = g.second;
    // float min_dist = std::numeric_limits<float>::max();
    float umin = std::numeric_limits<float>::max();
    float vmin = std::numeric_limits<float>::max();
    auto best_match = matches[0];
    for (int i=0;i<matches.size();i++)
    {
      const auto &match = matches[i];
      // auto ptA = pcl::transformPointWithNormal(match._ptA_cam,A_in_model);
      // auto ptB = pcl::transformPointWithNormal(match._ptB_cam, frameB->_pose_in_model);
      // float dist = (ptA.x-ptB.x)*(ptA.x-ptB.x) + (ptA.y-ptB.y)*(ptA.y-ptB.y) + (ptA.z-ptB.z)*(ptA.z-ptB.z);
      // if (dist<min_dist)
      // {
      //   min_dist = dist;
      //   best_match = match;
      // }
      if (match._uA<umin && match._vA<vmin)
      {
        umin = match._uA;
        vmin = match._vA;
        best_match = match;
      }
    }
    _matches[{frameA,frameB}].push_back(best_match);
  }

}

void SiftManager::vizKeyPoints(std::shared_ptr<Frame> frame)
{
  if ((*yml)["SPDLOG"].as<int>()<3) return;
  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+"/"+frame->_id_str+"/";
  if (!boost::filesystem::exists(out_dir))
  {
    system(std::string("mkdir -p "+out_dir).c_str());
  }
  cv::Mat out = frame->_color.clone();
  cv::drawKeypoints(out, frame->_keypts, out);
  cv::imwrite(out_dir+"keypoints.jpg", out);

  const auto &kpts = frame->_keypts;
  std::ofstream ff(out_dir+"keypoints.txt");
  for (int i=0;i<kpts.size();i++)
  {
    ff<<kpts[i].pt.x<<" "<<kpts[i].pt.y<<std::endl;
  }
  ff.close();
}


void SiftManager::forgetFrame(std::shared_ptr<Frame> frame)
{
  SPDLOG("forgetting frame {}",frame->_id_str);
  auto _matches_tmp = _matches;
  for (const auto& h:_matches_tmp)
  {
    const auto& f_pair = h.first;
    if (f_pair.first->_id==frame->_id || f_pair.second->_id==frame->_id)
    {
      _matches.erase(f_pair);
      _gt_matches.erase(f_pair);
    }
  }
  auto _raw_matches_tmp = _raw_matches;
  for (const auto& h:_raw_matches_tmp)
  {
    const auto& f_pair = h.first;
    if (f_pair.first->_id==frame->_id || f_pair.second->_id==frame->_id)
    {
      _raw_matches.erase(f_pair);
    }
  }
  auto _covisible_mappoints_tmp = _covisible_mappoints;
  for (const auto &h:_covisible_mappoints_tmp)
  {
    const auto& f_pair = h.first;
    if (f_pair.first->_id==frame->_id || f_pair.second->_id==frame->_id)
    {
      _covisible_mappoints.erase(f_pair);
    }
  }
  auto _gt_matches_tmp = _gt_matches;
  for (const auto &h:_gt_matches_tmp)
  {
    const auto& f_pair = h.first;
    if (f_pair.first->_id==frame->_id || f_pair.second->_id==frame->_id)
    {
      _gt_matches_tmp.erase(f_pair);
    }
  }
  for (const auto &mpt:_map_points_global)
  {
    mpt->_img_pt.erase(frame);
  }
}


void SiftManager::findCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  if (_matches.find({frameA,frameB})!=_matches.end()) return;

  const auto &map_points = (*yml)["feature_corres"]["map_points"].as<bool>();

  bool match_ref = frameA->_ref_frame_id==frameB->_id;
  SPDLOG("finding corres between {}(id={}) and {}(id={})", frameA->_id_str, frameA->_id, frameB->_id_str, frameB->_id);

  if (match_ref)
  {
    findCorresbyNN(frameA,frameB);
    vizCorresBetween(frameA, frameB, "before_ransac");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    // vizPositiveMatches(frameA, frameB);

    runRansacBetween(frameA, frameB);
    vizCorresBetween(frameA, frameB, "after_ransac");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    if (map_points)
    {
      updateFramePairMapPoints(frameA,frameB);
      vizCorresBetween(frameA, frameB, "after_mappoints");
    }
  }
  else
  {
    if (frameA->_pose_in_model!=Eigen::Matrix4f::Identity())
    {
      // float rot_diff = Utils::rotationGeodesicDistanceIgnoreRotationAroundCamZ(frameA->_pose_in_model.block(0,0,3,3).transpose(), frameB->_pose_in_model.block(0,0,3,3).transpose());
      float visible = computeCovisibility(frameA, frameB);
      if (visible<(*yml)["bundle"]["non_neighbor_min_visible"].as<float>())
      {
        SPDLOG("frame {} and {} visible={} skip matching",frameA->_id_str,frameB->_id_str,visible);
        _matches[{frameA,frameB}].clear();
        return;
      }
    }
    findCorresbyNN(frameA,frameB);
    vizCorresBetween(frameA, frameB, "before_ransac");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    if (map_points)
    {
      int num_matches = _matches[{frameA,frameB}].size();
      findCorresByMapPoints(frameA,frameB);
      SPDLOG("frame {} and {}, findCorresByMapPoints changed num matches from {} to {}",frameA->_id_str, frameB->_id_str, num_matches, _matches[{frameA,frameB}].size());
      vizCorresBetween(frameA, frameB, "after_mappoints");
    }

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    // vizPositiveMatches(frameA, frameB);

    runRansacBetween(frameA, frameB);
    vizCorresBetween(frameA, frameB, "after_ransac");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    if (map_points)
    {
      updateFramePairMapPoints(frameA,frameB);
    }

  }

}



void SiftManager::findCorresMultiPairGPU(std::vector<FramePair> &pairs)
{
  std::vector<FramePair> pairs_tmp;
  findCorresbyNNMultiPair(pairs);
  for (const auto &pair:pairs)
  {
    const auto &fB = pair.second;
    const auto &fA = pair.first;
    vizCorresBetween(fA, fB, "before_ransac");
    int num_matches = _matches[{fA,fB}].size();
    if ((*yml)["feature_corres"]["map_points"].as<bool>())
    {
      findCorresByMapPoints(fA,fB);
      SPDLOG("frame {} and {}, findCorresByMapPoints changed num matches from {} to {}",fA->_id_str, fB->_id_str, num_matches, _matches[{fA,fB}].size());
      vizCorresBetween(fA, fB, "after_mappoints");
    }
    if (_matches[{fA,fB}].size()<5)
    {
      SPDLOG("frame {} and {} corres too few {} to do ransac, set to 0", fA->_id_str, fB->_id_str,_matches[{fA,fB}].size());
      _matches[{fA,fB}].clear();
      continue;
    }
    pairs_tmp.push_back(pair);
  }
  pairs = pairs_tmp;
  if (_bundler->_newframe->_status==Frame::FAIL) return;

  runRansacMultiPairGPU(pairs);

  if (_bundler->_newframe->_status==Frame::FAIL)
  {
    return;
  }


  for (const auto &pair:pairs)
  {
    const auto &fA = pair.first;
    const auto &fB = pair.second;
    vizCorresBetween(fA, fB, "after_ransac");
    if ((*yml)["feature_corres"]["map_points"].as<bool>())
    {
      updateFramePairMapPoints(fA,fB);
    }
    if (_matches[{fA,fB}].size()<5)
    {
      _matches[{fA,fB}].clear();
    }
  }

}


/**
 * @brief https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
 * frameA stamp is later than frameB
 */
void SiftManager::findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  if (frameA->_keypts.size()==0 || frameB->_keypts.size()==0) return;

  const int H = frameA->_H;
  const int W = frameA->_W;

  bool match_ref = frameA->_ref_frame_id==frameB->_id;

  std::vector<cv::DMatch> matches_AB, matches_BA;
  std::vector< std::vector<cv::DMatch> > knn_matchesAB, knn_matchesBA;
  const int k_near = 5;

#if CUDA_MATCHING==0
  // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
  matcher->knnMatch( frameA->_feat_des, frameB->_feat_des, knn_matchesAB, k_near);
  matcher->knnMatch( frameB->_feat_des, frameA->_feat_des, knn_matchesBA, k_near);
#else
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
  matcher->knnMatch( frameA->_feat_des_gpu, frameB->_feat_des_gpu, knn_matchesAB, k_near);
  matcher->knnMatch( frameB->_feat_des_gpu, frameA->_feat_des_gpu, knn_matchesBA, k_near);
#endif
  SPDLOG("#knn_matchesAB={}, #knn_matchesBA={}",knn_matchesAB.size(),knn_matchesBA.size());
  pruneMatches(frameA,frameB,knn_matchesAB,matches_AB);

  pruneMatches(frameB,frameA,knn_matchesBA,matches_BA);

  SPDLOG("#matches_AB={}, #matches_BA={}",matches_AB.size(),matches_BA.size());
  collectMutualMatches(frameA,frameB,matches_AB,matches_BA);
  SPDLOG("collected mutual #matches={}",_matches[{frameA,frameB}].size());

  if (_matches[{frameA,frameB}].size()<5 && match_ref)
  {
    _matches[{frameA,frameB}].clear();
  }

}

void SiftManager::pruneMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector< std::vector<cv::DMatch> > &knn_matchesAB, std::vector<cv::DMatch> &matches_AB)
{
  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float max_dist_neighbor = (*yml)["feature_corres"]["max_dist_neighbor"].as<float>();
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);

  const int H = frameA->_H;
  const int W = frameA->_W;

  float dist_thres = std::numeric_limits<float>::max();
  float normal_thres = -1;
  if (frameA->_ref_frame_id==frameB->_id && frameA->_id==frameB->_id+1)
  {
    dist_thres = max_dist_neighbor;
    normal_thres = cos_max_normal_neighbor;
  }
  if (frameA->_ref_frame_id!=frameB->_id)
  {
    dist_thres = max_dist_no_neighbor;
    normal_thres = cos_max_normal_no_neighbor;
  }

  matches_AB.clear();
  matches_AB.reserve(knn_matchesAB.size());
  for (int i=0;i<knn_matchesAB.size();i++)
  {
    for (int k=0;k<knn_matchesAB[i].size();k++)
    {
      const auto &match = knn_matchesAB[i][k];
      auto pA = frameA->_keypts[match.queryIdx].pt;
      auto pB = frameB->_keypts[match.trainIdx].pt;
      int uA = std::round(pA.x);
      int vA = std::round(pA.y);
      int uB = std::round(pB.x);
      int vB = std::round(pB.y);
      if (!Utils::isPixelInsideImage(H, W, uA, vA) || !Utils::isPixelInsideImage(H, W, uB, vB)) continue;
      const auto &ptA = (*frameA->_cloud)(uA, vA);
      const auto &ptB = (*frameB->_cloud)(uB, vB);
      if (ptA.z<0.1 || ptB.z<0.1) continue;
      auto PA_world = pcl::transformPointWithNormal(ptA, frameA->_pose_in_model);
      auto PB_world = pcl::transformPointWithNormal(ptB, frameB->_pose_in_model);
      float dist = pcl::geometry::distance(PA_world, PB_world);
      Eigen::Vector3f n1(PA_world.normal_x,PA_world.normal_y,PA_world.normal_z);
      Eigen::Vector3f n2(PB_world.normal_x,PB_world.normal_y,PB_world.normal_z);
      if (dist>dist_thres || n1.normalized().dot(n2.normalized())<normal_thres) continue;
      matches_AB.push_back(match);
      break;
    }
  }
}

void SiftManager::collectMutualMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector<cv::DMatch> &matches_AB, const std::vector<cv::DMatch> &matches_BA)
{
  auto &matches = _matches[{frameA, frameB}];

  for (int i=0;i<matches_AB.size();i++)
  {
    int Aid = matches_AB[i].queryIdx;
    int Bid = matches_AB[i].trainIdx;
    float uA = frameA->_keypts[Aid].pt.x;
    float vA = frameA->_keypts[Aid].pt.y;
    float uB = frameB->_keypts[Bid].pt.x;
    float vB = frameB->_keypts[Bid].pt.y;
    const auto &ptA = frameA->_cloud->at(std::round(uA), std::round(vA));
    const auto &ptB = frameB->_cloud->at(std::round(uB), std::round(vB));
    Correspondence corr(uA,vA,uB,vB, ptA, ptB, true);
    matches.push_back(corr);
  }
  for (int i=0;i<matches_BA.size();i++)
  {
    int Aid = matches_BA[i].trainIdx;
    int Bid = matches_BA[i].queryIdx;
    float uA = frameA->_keypts[Aid].pt.x;
    float vA = frameA->_keypts[Aid].pt.y;
    float uB = frameB->_keypts[Bid].pt.x;
    float vB = frameB->_keypts[Bid].pt.y;
    const auto &ptA = frameA->_cloud->at(std::round(uA), std::round(vA));
    const auto &ptB = frameB->_cloud->at(std::round(uB), std::round(vB));
    Correspondence corr(uA,vA,uB,vB, ptA, ptB, true);
    matches.push_back(corr);
  }
}

void SiftManager::findCorresbyNNMultiPair(std::vector<FramePair> &pairs)
{
  const bool mutual = (*yml)["feature_corres"]["mutual"].as<bool>();
  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);
  const int k_near = 5;

  std::vector<cv::cuda::Stream> streams(pairs.size()*2);
  std::vector<cv::cuda::GpuMat> matchesAB_gpus(pairs.size()), matchesBA_gpus(pairs.size());
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
  for (int i=0;i<pairs.size();i++)
  {
    auto &fA = pairs[i].first;
    auto &fB = pairs[i].second;

    float visible = computeCovisibility(fA, fB);
    SPDLOG("frame {} and {} visible={}",fA->_id_str,fB->_id_str,visible);
    if (visible<(*yml)["bundle"]["non_neighbor_min_visible"].as<float>())
    {
      SPDLOG("frame {} and {} visible={} skip matching",fA->_id_str,fB->_id_str,visible);
      _matches[{fA,fB}].clear();
      continue;
    }

    // matcher->knnMatch(fA->_feat_des_gpu, fB->_feat_des_gpu, knn_matchesAB, k_near);
    // matcher->knnMatch(fB->_feat_des_gpu, fA->_feat_des_gpu, knn_matchesBA, k_near);
    matcher->knnMatchAsync(fA->_feat_des_gpu, fB->_feat_des_gpu, matchesAB_gpus[i], k_near, cv::noArray(), streams[2*i]);
    if (mutual)
    {
      matcher->knnMatchAsync(fB->_feat_des_gpu, fA->_feat_des_gpu, matchesBA_gpus[i], k_near, cv::noArray(), streams[2*i+1]);
    }
  }

  for (int i=0;i<pairs.size();i++)
  {
    streams[2*i].waitForCompletion();
    streams[2*i+1].waitForCompletion();
  }

  for (int i=0;i<pairs.size();i++)
  {
    auto &fA = pairs[i].first;
    auto &fB = pairs[i].second;
    std::vector< std::vector<cv::DMatch> > knn_matchesAB, knn_matchesBA;
    matcher->knnMatchConvert(matchesAB_gpus[i], knn_matchesAB);
    if (mutual)
    {
      matcher->knnMatchConvert(matchesBA_gpus[i], knn_matchesBA);
    }
    std::vector<cv::DMatch> matchesAB, matchesBA;
    pruneMatches(fA,fB,knn_matchesAB,matchesAB);
    if (mutual)
    {
      pruneMatches(fB,fA,knn_matchesBA,matchesBA);
    }
    collectMutualMatches(fA,fB,matchesAB,matchesBA);
  }
}



/**
 * @brief propogate feature points from B to A
 *
 * @param frameA
 * @param frameB
 */
void SiftManager::updateFramePairMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  const auto &matches = _matches[{frameA,frameB}];
  for (int i=0;i<matches.size();i++)
  {
    const auto &match = matches[i];
    if (match._isinlier==false) continue;
    const float uA = match._uA;
    const float vA = match._vA;
    const float uB = match._uB;
    const float vB = match._vB;

    if (frameA->_map_points.find({uA,vA})!=frameA->_map_points.end() && frameB->_map_points.find({uB,vB})!=frameB->_map_points.end()) continue;

    std::shared_ptr<MapPoint> mpt;
    bool existed = false;
    if (frameB->_map_points.find({uB,vB})==frameB->_map_points.end())
    {
      mpt = std::make_shared<MapPoint>(frameB,uB,vB);
      frameB->_map_points[{uB,vB}] = mpt;
      mpt->_img_pt[frameB] = {uB,vB};
      existed = false;
    }
    else
    {
      mpt = frameB->_map_points[{uB,vB}];
      existed = true;
    }
    mpt->_img_pt[frameA] = {uA,vA};
    frameA->_map_points[{uA,vA}] = mpt;

    if (!existed)
    {
      _map_points_global.push_back(mpt);
    }
  }
}


void SiftManager::debugMapPoints()
{
  if ((*yml)["SPDLOG"].as<int>()<2) return;

  std::map<std::shared_ptr<Frame>, cv::Mat> frames;
  const int H_viz = 1000;
  const int W_viz = 1000;
  for (int i=0;i<_map_points_global.size();i++)
  {
    const auto &mpt = _map_points_global[i];
    int r = rand()%255;
    int g = rand()%255;
    int b = rand()%255;
    for (const auto &h:mpt->_img_pt)
    {
      const auto &f = h.first;
      const auto &roi = f->_roi;
      if (frames.find(f)==frames.end())
      {
        frames[f] = f->_color.clone();
        frames[f] = frames[f](cv::Rect(roi(0),roi(2),roi(1)-roi(0),roi(3)-roi(2)));
        cv::resize(frames[f], frames[f], {W_viz,H_viz});
      }
      const auto &uv = h.second;
      const float u = (uv.first-roi(0))/(roi(1)-roi(0)) * W_viz;
      const float v = (uv.second-roi(2))/(roi(3)-roi(2)) * H_viz;
      // cv::circle(frames[f], {u,v}, 2, {b,g,r}, -1);
      cv::putText(frames[f], std::to_string(i), {u,v}, cv::FONT_HERSHEY_SIMPLEX, 0.3, {b,g,r}, 1);
    }
  }

  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+"debug_mappoints/";
  if (!boost::filesystem::exists(out_dir))
  {
    system(std::string("mkdir -p "+out_dir).c_str());
  }

  for (const auto &h:frames)
  {
    const auto &f = h.first;
    cv::imwrite(out_dir+f->_id_str+"_mappoints.jpg", h.second, {cv::IMWRITE_JPEG_QUALITY,100});
  }

}

void SiftManager::findCorresByMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  auto &matches = _matches[{frameA,frameB}];
  for (const auto &h:frameA->_map_points)
  {
    const auto &uvA = h.first;
    const auto &mpt = h.second;
    if (mpt->_img_pt.find(frameB)==mpt->_img_pt.end()) continue;
    const auto &uvB = mpt->_img_pt[frameB];
    const auto &pA = frameA->_cloud->at(std::round(uvA.first), std::round(uvA.second));
    const auto &pB = frameB->_cloud->at(std::round(uvB.first), std::round(uvB.second));
    Correspondence match(uvA.first, uvA.second, uvB.first, uvB.second, pA, pB, true);
    bool existed = false;
    for (int i=0;i<matches.size();i++)
    {
      if (matches[i]._uA==match._uA && matches[i]._vA==match._vA)
      {
        existed = true;
        break;
      }
      if (matches[i]._uB==match._uB && matches[i]._vB==match._vB)
      {
        existed = true;
        break;
      }
    }
    if (existed) continue;
    match._ispropogated = true;
    matches.push_back(match);
  }
}

std::vector<std::shared_ptr<MapPoint>> SiftManager::getCovisibleMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  if (_covisible_mappoints.find({frameA,frameB})!=_covisible_mappoints.end()) return _covisible_mappoints[{frameA,frameB}];
  if (_covisible_mappoints.find({frameB,frameA})!=_covisible_mappoints.end()) return _covisible_mappoints[{frameB,frameA}];

  auto &cov_mpts = _covisible_mappoints[{frameA,frameB}];
  for (const auto &h:frameA->_map_points)
  {
    const auto &mpt = h.second;
    if (mpt->_img_pt.find(frameB)!=mpt->_img_pt.end())
    {
      cov_mpts.push_back(mpt);
    }
  }
  _covisible_mappoints[{frameB,frameA}] = cov_mpts;
  return cov_mpts;
}

void SiftManager::findCorresbyGroundtruth(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  if (_matches.find({frameA, frameB})!=_matches.end()) return;
  if (_matches.find({frameB, frameA})!=_matches.end()) return;

  if (frameA->_keypts.size()==0 || frameB->_keypts.size()==0) return;
  assert(frameA->_id > frameB->_id);

  const float max_dist = (*yml)["feature_corres"]["max_dist"].as<float>();

  const Eigen::Matrix3f &K = frameA->_K;
  std::vector<Correspondence> matches;
  for (int i=0;i<frameA->_keypts.size();i++)
  {
    const auto &kpt = frameA->_keypts[i];
    float uA = frameA->_keypts[i].pt.x;
    float vA = frameA->_keypts[i].pt.y;
    auto ptA = frameA->_cloud->at(std::round(uA), std::round(vA));
    ptA = pcl::transformPointWithNormal(ptA, frameA->_gt_pose_in_model);

    float min_dist = 999;
    int nearest_j = -1;
    for (int j=0;j<frameB->_keypts.size();j++)
    {
      float uB = frameB->_keypts[j].pt.x;
      float vB = frameB->_keypts[j].pt.y;
      auto ptB = frameB->_cloud->at(std::round(uB), std::round(vB));
      ptB = pcl::transformPointWithNormal(ptB, frameB->_gt_pose_in_model);
      float dist = (ptA.x-ptB.x)*(ptA.x-ptB.x) + (ptA.y-ptB.y)*(ptA.y-ptB.y) + (ptA.z-ptB.z)*(ptA.z-ptB.z);
      if (dist<min_dist)
      {
        min_dist = dist;
        nearest_j = j;
      }
    }

    if (min_dist<=0.002*0.002)
    {
      float uB = frameB->_keypts[nearest_j].pt.x;
      float vB = frameB->_keypts[nearest_j].pt.y;
      auto ptA = frameA->_cloud->at(std::round(uA), std::round(vA));
      auto ptB = frameB->_cloud->at(std::round(uB), std::round(vB));
      matches.push_back(Correspondence(uA,vA,uB,vB,ptA,ptB,true));
    }
  }

  _matches[{frameA,frameB}] = matches;
  vizCorresBetween(frameA, frameB, "gt");

}


/**
 * @brief
 *
 * @param frameA
 * @param frameB
 * @param matches
 * @return Eigen::Matrix4f : cam pose residual A-->B in world frame
 */
Eigen::Matrix4f SiftManager::procrustesByCorrespondence(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
  const auto &matches = _matches[{frameA, frameB}];

  int n_inliers = countInlierCorres(frameA,frameB);
  if (n_inliers<5)
  {
    SPDLOG("frame {} and {} inlier corres {}<5",frameA->_id_str,frameB->_id_str,n_inliers);
    return pose;
  }

  Eigen::MatrixXf src(Eigen::MatrixXf::Constant(matches.size(),3,1));
  Eigen::MatrixXf dst(Eigen::MatrixXf::Constant(matches.size(),3,1));
  for (int i=0;i<matches.size();i++)
  {
    const auto &match = matches[i];
    if (!match._isinlier) continue;
    auto pcl_pt1 = match._ptA_cam;
    auto pcl_pt2 = match._ptB_cam;
    // std::cout<<"pcl_pt1 before "<<pcl_pt1<<std::endl;
    // std::cout<<"pcl_pt2 before "<<pcl_pt2<<std::endl;
    pcl_pt1 = pcl::transformPointWithNormal<pcl::PointXYZRGBNormal>(pcl_pt1,frameA->_pose_in_model);
    pcl_pt2 = pcl::transformPointWithNormal<pcl::PointXYZRGBNormal>(pcl_pt2,frameB->_pose_in_model);
    src.row(i) << pcl_pt1.x, pcl_pt1.y, pcl_pt1.z;
    dst.row(i) << pcl_pt2.x, pcl_pt2.y, pcl_pt2.z;
    // std::cout<<"pcl_pt1 "<<pcl_pt1<<std::endl;
    // std::cout<<"pcl_pt2 "<<pcl_pt2<<std::endl;
  }

  Utils::solveRigidTransformBetweenPoints(src, dst, pose);
  if (pose==Eigen::Matrix4f::Identity()) return Eigen::Matrix4f::Identity();
  // std::cout<<"ransac computed pose\n"<<pose<<"\n\n";
  Eigen::MatrixXf src_est = src;
  for (int i=0;i<src_est.rows();i++)
  {
    src_est.row(i) = pose.block(0,0,3,3)*src_est.row(i).transpose() + pose.block(0,3,3,1);
  }
  // Eigen::Vector3f trans = pose.block(0,3,3,1);
  // src_est = (src_est.colwise() + trans).transpose();
  float err = (src_est - dst).norm()/src_est.rows();
  // std::cout<<"src\n"<<src<<"\n\n";
  // std::cout<<"src_est\n"<<src_est<<"\n\n";
  // std::cout<<"dst\n"<<dst<<"\n\n";
  SPDLOG("procrustesByCorrespondence err per point between {} and {}: {}", frameA->_id_str, frameB->_id_str, err);
  if (frameB->_id-frameA->_id==1 && err>1e-3)
  {
    PointCloudRGBNormal::Ptr cloud1(new PointCloudRGBNormal);
    PointCloudRGBNormal::Ptr cloud2(new PointCloudRGBNormal);
    PointCloudRGBNormal::Ptr cloud3(new PointCloudRGBNormal);

    for (int i=0;i<src.rows();i++)
    {
      pcl::PointXYZRGBNormal pt1;
      pt1.x = src(i,0);
      pt1.y = src(i,1);
      pt1.z = src(i,2);
      cloud1->points.push_back(pt1);

      pcl::PointXYZRGBNormal pt2;
      pt2.x = dst(i,0);
      pt2.y = dst(i,1);
      pt2.z = dst(i,2);
      cloud2->points.push_back(pt2);

      pcl::PointXYZRGBNormal pt3;
      pt3.x = src_est(i,0);
      pt3.y = src_est(i,1);
      pt3.z = src_est(i,2);
      cloud3->points.push_back(pt3);
    }
    pcl::io::savePLYFile("/home/bowen/debug/src.ply", *cloud1);
    pcl::io::savePLYFile("/home/bowen/debug/dst.ply", *cloud2);
    pcl::io::savePLYFile("/home/bowen/debug/src_est.ply", *cloud3);
    SPDLOG("error too big, pause");
    cv::waitKey(0);
  }
  return pose;
}


void SiftManager::filterCorrespondenceByEssential(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);

  const int num_sample = (*yml)["ransac"]["num_sample"].as<int>();
  const int max_iter = (*yml)["ransac"]["max_iter"].as<int>();
  const float inlier_dist = (*yml)["ransac"]["inlier_dist"].as<float>();
  const float cos_normal_angle = std::cos((*yml)["ransac"]["inlier_normal_angle"].as<float>());
  const float max_rot_neighbor = (*yml)["ransac"]["max_rot_deg"].as<float>()/180.0*M_PI;
  const float max_trans = (*yml)["ransac"]["max_trans"].as<float>();
  const float max_trans_no_neighbor = (*yml)["ransac"]["max_trans_no_neighbor"].as<float>();
  const float max_rot_no_neighbor = (*yml)["ransac"]["max_rot_no_neighbor"].as<float>()/180.0*M_PI;
  const float epipolar_thres = (*yml)["ransac"]["epipolar_thres"].as<float>();

  while (1)
  {
    const auto matches_backup = _matches[{frameA, frameB}];
    auto &matches = _matches[{frameA, frameB}];
    if (matches_backup.size()<5)
    {
      _matches[{frameA, frameB}].clear();
      break;
    }
    std::vector<cv::Point2f> pointsA, pointsB;
    for (int i=0;i<matches_backup.size();i++)
    {
      const auto &match = matches_backup[i];
      pointsA.push_back(cv::Point2f(match._uA, match._vA));
      pointsB.push_back(cv::Point2f(match._uB, match._vB));
    }
    cv::Mat mask;   //8UC1, Nx1
    cv::Mat K_cv;
    cv::eigen2cv(frameA->_K, K_cv);
    cv::findEssentialMat(pointsA, pointsB, K_cv, cv::RANSAC, 0.999, epipolar_thres, mask);

    matches.clear();
    for (int i=0;i<matches_backup.size();i++)
    {
      if (mask.at<unsigned char>(i))
      {
        matches.push_back(matches_backup[i]);
      }
    }

    if (matches.size()<5)
    {
      matches.clear();
      break;
    }

    auto offset = procrustesByCorrespondence(frameA,frameB);
    Eigen::Matrix4f B_in_cam = frameB->_pose_in_model.inverse();
    Eigen::Matrix4f A_in_cam = (offset * frameA->_pose_in_model).inverse();
    Eigen::Matrix4f poseAB_cam = B_in_cam * A_in_cam.inverse();
    float rot_offset, trans_offset;
    bool isgood = true;
    // if (is_neighbor)
    // {
    //   rot_offset = Utils::rotationGeodesicDistance(poseAB_cam.block(0,0,3,3), Eigen::Matrix3f::Identity());
    //   trans_offset = (B_in_cam.block(0,3,3,1)-A_in_cam.block(0,3,3,1)).norm();
    //   if (rot_offset>max_rot_neighbor || trans_offset>=max_trans)
    //   {
    //     // SPDLOG("solved pose rot_offset {}rad or dist {}m bad", rot_offset, trans_offset);
    //     // SPDLOG("A_in_cam\n{}\nB_in_cam\n{}",A_in_cam, B_in_cam);
    //     // debugSampledMatch(frameA, frameB, chosen_match_ids, offset, {});
    //     isgood = false;
    //   }
    // }
    // else //A has been estimated before, this pair of non-neighbor is for BA
    // {
    //   rot_offset = Utils::rotationGeodesicDistance(offset.block(0,0,3,3), Eigen::Matrix3f::Identity());
    //   trans_offset = offset.block(0,3,3,1).norm();
    //   if (rot_offset>max_rot_no_neighbor || trans_offset>=max_trans_no_neighbor)
    //   {
    //     // SPDLOG("solved pose rot_offset {}rad, or dist {}m bad", rot_offset, trans_offset);
    //     // SPDLOG("A_in_cam\n{}\nB_in_cam\n{}",A_in_cam, B_in_cam);
    //     // debugSampledMatch(frameA, frameB, chosen_match_ids, offset, {});
    //     isgood = false;
    //   }
    // }

    if (isgood)
    {
      break;
    }
    else   //This set is actually the bad one
    {
      matches.clear();
      for (int i=0;i<matches.size();i++)
      {
        if (mask.at<unsigned char>(i)==0)
        {
          matches.push_back(matches[i]);
        }
      }
    }
  }

}

void SiftManager::runRansacBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  SPDLOG("start ransac");

  int n_inlier = countInlierCorres(frameA, frameB);
  if (n_inlier<=5)
  {
    SPDLOG("{} and {}, matches={} inlier={}<5",frameA->_id_str, frameB->_id_str,_matches[{frameA, frameB}].size(),n_inlier);
    _matches[{frameA, frameB}].clear();
    return;
  }

#if CUDA_RANSAC>0
  runRansacMultiPairGPU({{frameA, frameB}});

#else
  const int num_sample = (*yml)["ransac"]["num_sample"].as<int>();
  const int max_iter = (*yml)["ransac"]["max_iter"].as<int>();
  const float inlier_dist = (*yml)["ransac"]["inlier_dist"].as<float>();
  const float cos_normal_angle = std::cos((*yml)["ransac"]["inlier_normal_angle"].as<float>()/180.0*M_PI);
  const float max_rot_deg_neighbor = (*yml)["ransac"]["max_rot_deg_neighbor"].as<float>()/180.0*M_PI;
  const float max_trans_neighbor = (*yml)["ransac"]["max_trans_neighbor"].as<float>();
  const float max_trans_no_neighbor = (*yml)["ransac"]["max_trans_no_neighbor"].as<float>();
  const float max_rot_no_neighbor = (*yml)["ransac"]["max_rot_no_neighbor"].as<float>()/180.0*M_PI;

  float rot_thres = std::numeric_limits<float>::max();
  float trans_thres = std::numeric_limits<float>::max();
  if (frameA->_ref_frame_id==frameB->_id && frameA->_id==frameB->_id+1)
  {
    trans_thres = max_trans_neighbor;
    rot_thres = max_rot_deg_neighbor;
  }
  if (frameA->_ref_frame_id!=frameB->_id)
  {
    trans_thres = max_trans_no_neighbor;
    rot_thres = max_rot_no_neighbor;
  }

  const auto matches = _matches[{frameA, frameB}];
  std::vector<int> propogated_indices;
  propogated_indices.reserve(matches.size());
  for (int i=0;i<matches.size();i++)
  {
    const auto &m = matches[i];
    if (m._ispropogated)
    {
      propogated_indices.push_back(i);
    }
  }
  std::vector<std::vector<int>> propogated_samples;
  for (int i=0;i<propogated_indices.size() && propogated_samples.size()<max_iter;i++)
  {
    for (int j=i+1;j<propogated_indices.size() && propogated_samples.size()<max_iter;j++)
    {
      for (int k=j+1;k<propogated_indices.size() && propogated_samples.size()<max_iter;k++)
      {
        propogated_samples.push_back({propogated_indices[i],propogated_indices[j],propogated_indices[k]});
      }
    }
  }

  std::map<std::vector<int>, bool> tried_sample;
  int max_num_comb = 1;
  for (int i=0;i<num_sample;i++)
  {
    max_num_comb *= matches.size()-i;
  }
  for (int i=0;i<num_sample-1;i++)
  {
    max_num_comb /= i+1;
  }

  std::vector<std::vector<int>> samples_to_try;
  samples_to_try.reserve(max_iter);
  for (int i=0;i<max_iter;i++)
  {
    if (tried_sample.size()==max_num_comb) break;
    std::vector<int> chosen_match_ids;
    if (i<propogated_samples.size())   //Priotize to propogated features, propogated_samples is already sorted
    {
      chosen_match_ids = propogated_samples[i];
    }
    else
    {
      // std::shuffle(indices.begin(), indices.end(), _rng);
      // chosen_match_ids.insert(chosen_match_ids.begin(), indices.begin(), indices.begin()+num_sample);

      while (chosen_match_ids.size()<num_sample)
      {
        int id = rand()%matches.size();
        if (std::find(chosen_match_ids.begin(),chosen_match_ids.end(),id)==chosen_match_ids.end())
          chosen_match_ids.push_back(id);
      }

      std::sort(chosen_match_ids.begin(),chosen_match_ids.end());
    }

    if (tried_sample.find(chosen_match_ids)!=tried_sample.end())
    {
      continue;
    }
    tried_sample[chosen_match_ids] = true;
    samples_to_try.push_back(chosen_match_ids);
  }

  std::vector<Correspondence> inliers;

  Eigen::Vector3f xyz_min = Eigen::Vector3f(1,1,1)*std::numeric_limits<float>::max();
  Eigen::Vector3f xyz_max = Eigen::Vector3f(1,1,1)*(-std::numeric_limits<float>::max());
  for (int i=0;i<frameA->_cloud->points.size();i++)
  {
    const auto &pt = frameA->_cloud->points[i];
    if (pt.z>=0.1)
    {
      Eigen::Vector3f pt_eigen(pt.x, pt.y, pt.z);
      for (int ii=0;ii<3;ii++)
      {
        xyz_min(ii) = std::min(xyz_min(ii), pt_eigen(ii));
        xyz_max(ii) = std::max(xyz_max(ii), pt_eigen(ii));
      }
    }
  }

  const float vox_size = 0.005;
  class Comp
  {
  public:
    bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b)
    {
      for (int i=0;i<3;i++)
      {
        if (a(i)<b(i)) return true;
        else if (a(i)>b(i)) return false;
      }
      return false;
    };
  };
  std::map<Eigen::Vector3i, std::vector<Correspondence>, Comp> best_hash_corres; //Ues voxel coverage to evlauate

// #pragma omp parallel for schedule(dynamic), firstprivate(matches)
  for (int i=0;i<samples_to_try.size();i++)
  {
    Eigen::MatrixXf points1(num_sample,3), points2(num_sample,3);
    std::vector<int> chosen_match_ids = samples_to_try[i];

    //!NOTE Check sampled points quality
    bool isbad = false;
    for (int i=0;i<chosen_match_ids.size();i++)
    {
      for (int j=i+1;j<chosen_match_ids.size();j++)
      {
        const auto &match1 = matches[chosen_match_ids[i]];
        const auto &match2 = matches[chosen_match_ids[j]];

        if (i==0 && j==1)  //Sampled 3 points need to span enough
        {
          const auto &match3 = matches[2];
          Eigen::Vector3f p1(match1._ptA_cam.x,match1._ptA_cam.y,match1._ptA_cam.z);
          Eigen::Vector3f p2(match2._ptA_cam.x,match2._ptA_cam.y,match2._ptA_cam.z);
          Eigen::Vector3f p3(match3._ptA_cam.x,match3._ptA_cam.y,match3._ptA_cam.z);
          const float cos_span_angle = (p2-p1).normalized().dot((p3-p1).normalized());
          if (std::abs(cos_span_angle)>=std::cos(10/180.0*M_PI))
          {
            isbad = true;
            break;
          }
        }

        float distA = pcl::geometry::distance(match1._ptA_cam, match2._ptA_cam);
        float distB = pcl::geometry::distance(match1._ptB_cam, match2._ptB_cam);
        if (std::abs(distA-distB)>=0.005)
        {
          // SPDLOG("distA and distB bad {}", std::abs(distA-distB));
          isbad = true;
          break;
        }
        // Eigen::Vector3f normalA1(match1._ptA_cam.normal_x, match1._ptA_cam.normal_y, match1._ptA_cam.normal_z);
        // Eigen::Vector3f normalA2(match2._ptA_cam.normal_x, match2._ptA_cam.normal_y, match2._ptA_cam.normal_z);
        // Eigen::Vector3f normalB1(match1._ptB_cam.normal_x, match1._ptB_cam.normal_y, match1._ptB_cam.normal_z);
        // Eigen::Vector3f normalB2(match2._ptB_cam.normal_x, match2._ptB_cam.normal_y, match2._ptB_cam.normal_z);
        // const float normal_angleA = std::acos(normalA1.normalized().dot(normalA2.normalized()));
        // const float normal_angleB = std::acos(normalB1.normalized().dot(normalB2.normalized()));
        // if (std::abs(normal_angleA-normal_angleB)>=30/M_PI*180)
        // {
        //   isbad = true;
        //   break;
        // }
      }
      if (isbad) break;
    }
    if (isbad)
    {
      // debugSampledMatch(frameA, frameB, chosen_match_ids, Eigen::Matrix4f::Identity(),{});
      continue;
    }

    for (int i=0;i<num_sample;i++)
    {
      const auto &match = matches[chosen_match_ids[i]];
      if (!match._isinlier) continue;
      auto pcl_pt1 = match._ptA_cam;
      auto pcl_pt2 = match._ptB_cam;
      pcl_pt1 = pcl::transformPointWithNormal<pcl::PointXYZRGBNormal>(pcl_pt1,frameA->_pose_in_model);
      pcl_pt2 = pcl::transformPointWithNormal<pcl::PointXYZRGBNormal>(pcl_pt2,frameB->_pose_in_model);
      points1.row(i) << pcl_pt1.x, pcl_pt1.y, pcl_pt1.z;
      points2.row(i) << pcl_pt2.x, pcl_pt2.y, pcl_pt2.z;
    }

    Eigen::Matrix4f offset(Eigen::Matrix4f::Identity());   // A in world offset
    Utils::solveRigidTransformBetweenPoints(points1, points2, offset);
    if (offset==Eigen::Matrix4f::Identity()) continue;

    Eigen::Matrix4f A_in_cam = (offset * frameA->_pose_in_model).inverse();

    //Eval
    std::vector<Correspondence> cur_inliers;
    cur_inliers.reserve(matches.size());
    std::map<Eigen::Vector3i, std::vector<Correspondence>, Comp> hash_corres; //Ues voxel coverage to evlauate
    for (int i=0;i<matches.size();i++)
    {
      const auto& match = matches[i];
      if (std::find(chosen_match_ids.begin(), chosen_match_ids.end(), i)!=chosen_match_ids.end())
      {
        cur_inliers.push_back(match);
        continue;
      }

      auto pcl_pt1 = match._ptA_cam;
      auto pcl_pt2 = match._ptB_cam;
      pcl_pt1 = pcl::transformPointWithNormal<pcl::PointXYZRGBNormal>(pcl_pt1,A_in_cam.inverse().eval());
      pcl_pt2 = pcl::transformPointWithNormal<pcl::PointXYZRGBNormal>(pcl_pt2,frameB->_pose_in_model);
      float dist = pcl::geometry::distance(pcl_pt1, pcl_pt2);
      if (dist>inlier_dist)
      {
        // if (frameA->_id_str=="0008" && frameB->_id_str=="0000")
        // {
        //   SPDLOG("bad");
        //   debugSampledMatch(frameA, frameB, chosen_match_ids, offset, {});
        // }
        continue;
      }
      Eigen::Vector3f n1(pcl_pt1.normal_x, pcl_pt1.normal_y, pcl_pt1.normal_z);
      Eigen::Vector3f n2(pcl_pt2.normal_x, pcl_pt2.normal_y, pcl_pt2.normal_z);
      if (n1.normalized().dot(n2.normalized())<cos_normal_angle) continue;

      //!NOTE check angles formed with two basis sampled points should be similar in A,B
      const auto &base1 = matches[chosen_match_ids[0]];
      const auto &base2 = matches[chosen_match_ids[1]];
      Eigen::Vector3f dir1A = match._ptA_cam.getVector3fMap() - base1._ptA_cam.getVector3fMap();
      Eigen::Vector3f dir2A = match._ptA_cam.getVector3fMap() - base2._ptA_cam.getVector3fMap();
      Eigen::Vector3f dir1B = match._ptB_cam.getVector3fMap() - base1._ptB_cam.getVector3fMap();
      Eigen::Vector3f dir2B = match._ptB_cam.getVector3fMap() - base2._ptB_cam.getVector3fMap();
      float angleA = std::acos(std::abs(dir1A.dot(dir2A)));
      float angleB = std::acos(std::abs(dir1B.dot(dir2B)));
      if (std::abs(angleA-angleB)>=5/180.0*M_PI)
      {
        continue;
      }

      Eigen::Vector3i key = ((Eigen::Vector3f(match._ptA_cam.x, match._ptA_cam.y, match._ptA_cam.z) - xyz_min)/vox_size).cast<int>();
      hash_corres[key].push_back(match);

      cur_inliers.push_back(match);
    }

    //////////////// Drop unreasonable pose
    offset = procrustesByCorrespondence(frameA, frameB);
    // Eigen::Matrix4f B_in_cam = frameB->_pose_in_model.inverse();
    // A_in_cam = (offset * frameA->_pose_in_model).inverse();
    // Eigen::Matrix4f poseAB_cam = B_in_cam * A_in_cam.inverse();
    // float rot_offset = std::numeric_limits<float>::max(), trans_offset = std::numeric_limits<float>::max();
    rot_offset = Utils::rotationGeodesicDistance(offset.block(0,0,3,3), Eigen::Matrix3f::Identity());
    trans_offset = offset.block(0,3,3,1).norm();
    if (rot_offset>rot_thres || trans_offset>trans_thres)
    {
      continue;
    }

    // #pragma omp critical
    if (hash_corres.size()>best_hash_corres.size())
    {
      best_hash_corres = hash_corres;
    }

  } //iter finsish

  inliers.clear();
  for (const auto &h:best_hash_corres)
  {
    for (const auto &c:h.second)
    {
      inliers.push_back(c);
      break;
    }
  }
  SPDLOG("best_hash_corres#={}, #inliers={}",best_hash_corres.size(),inliers.size());

  SPDLOG("ransac makes match betwee frame {} {} size from {} to {}", frameA->_id_str, frameB->_id_str, matches.size(), inliers.size());
  if (inliers.size()<5)
  {
    inliers.clear();
  }
  _matches[{frameA, frameB}] = inliers;

  SPDLOG("finish ransac, inliers#={}", inliers.size());

#endif  //CUDA_RANSAC


}


Correspondence SiftManager::makeCorrespondence(float uA_, float vA_, float uB_, float vB_, const std::shared_ptr<Frame> &fA, const std::shared_ptr<Frame> &fB, const float dist_thres, const float dot_thres, const float confidence)
{
  if (!Utils::isPixelInsideImage(fA->_H, fA->_W,uA_,vA_) || !Utils::isPixelInsideImage(fB->_H, fB->_W,uB_,vB_))
  {
    Correspondence corres;
    corres._isinlier = false;
    return corres;
  }

  int uA = std::round(uA_);
  int vA = std::round(vA_);
  int uB = std::round(uB_);
  int vB = std::round(vB_);
  const auto &ptA = (*fA->_cloud)(uA,vA);
  const auto &ptB = (*fB->_cloud)(uB,vB);
  if (ptA.z<=0.1 || ptB.z<=0.1)
  {
    Correspondence corres;
    corres._isinlier = false;
    return corres;
  }

  auto PA_world = pcl::transformPointWithNormal(ptA, fA->_pose_in_model);
  auto PB_world = pcl::transformPointWithNormal(ptB, fB->_pose_in_model);
  float dist = pcl::geometry::distance(PA_world, PB_world);
  Eigen::Vector3f n1(PA_world.normal_x,PA_world.normal_y,PA_world.normal_z);
  Eigen::Vector3f n2(PB_world.normal_x,PB_world.normal_y,PB_world.normal_z);
  if (dist>dist_thres || n1.normalized().dot(n2.normalized())<dot_thres)
  {
    Correspondence corres;
    corres._isinlier = false;
    return corres;
  }

  // cv::circle(visA,{int(uvA(0)),int(uvA(1))},1,{0,255,0},-1);

  Correspondence corres(uA,vA,uB,vB,ptA,ptB,true);
  corres._confidence = confidence;
  return corres;
}


void SiftManager::runRansacMultiPairGPU(const std::vector<std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>>> &pairs)
{
  SPDLOG("start multi pair ransac GPU, pairs#={}", pairs.size());

  const int num_sample = (*yml)["ransac"]["num_sample"].as<int>();
  const int max_iter = (*yml)["ransac"]["max_iter"].as<int>();
  const float inlier_dist = (*yml)["ransac"]["inlier_dist"].as<float>();
  const int min_match_after_ransac = (*yml)["ransac"]["min_match_after_ransac"].as<int>();
  const float max_trans_neighbor = (*yml)["ransac"]["max_trans_neighbor"].as<float>();
  const float max_rot_neighbor = (*yml)["ransac"]["max_rot_deg_neighbor"].as<float>()/180.0*M_PI;
  const float max_trans_no_neighbor = (*yml)["ransac"]["max_trans_no_neighbor"].as<float>();
  const float max_rot_no_neighbor = (*yml)["ransac"]["max_rot_no_neighbor"].as<float>()/180.0*M_PI;
  const float cos_normal_angle = std::cos((*yml)["ransac"]["inlier_normal_angle"].as<float>()/180.0*M_PI);

  std::vector<float4*> ptsA_gpu(pairs.size()), ptsB_gpu(pairs.size());
  std::vector<float4*> normalsA_gpu(pairs.size()), normalsB_gpu(pairs.size());
  std::vector<float2*> uvA_gpu(pairs.size()), uvB_gpu(pairs.size());
  std::vector<float*> confs_gpu(pairs.size());
  std::vector<float> max_transs(pairs.size(), std::numeric_limits<float>::max());
  std::vector<float> max_rots(pairs.size(), std::numeric_limits<float>::max());
  std::vector<std::vector<int>> ids(pairs.size());
  std::vector<int> n_pts(pairs.size());
  std::vector<cudaStream_t> streams(pairs.size());

  for (int i_pair=0;i_pair<pairs.size();i_pair++)
  {
    // cudaStreamCreate(&streams[i]);
    const auto pair = pairs[i_pair];
    std::vector<float4> ptsA_cur_pair, ptsB_cur_pair;
    std::vector<float4> normalsA_cur_pair, normalsB_cur_pair;
    std::vector<float2> uvA_cur_pair, uvB_cur_pair;
    std::vector<float> confs_cur_pair;

    const auto &frameA = pair.first;
    const auto &frameB = pair.second;
    const auto &matches = _matches[{frameA, frameB}];

    if (frameA->_ref_frame_id==frameB->_id && frameA->_id==frameB->_id+1)
    {
      max_transs[i_pair] = max_trans_neighbor;
      max_rots[i_pair] = max_rot_neighbor;
    }
    if (frameA->_ref_frame_id!=frameB->_id)
    {
      max_transs[i_pair] = max_trans_no_neighbor;
      max_rots[i_pair] = max_rot_no_neighbor;
    }

    std::vector<int> ids_cur_pair;
    for (int i=0;i<matches.size();i++)
    {
      const auto &match = matches[i];
      if (match._isinlier==false) continue;
      const auto pA = pcl::transformPointWithNormal(match._ptA_cam, frameA->_pose_in_model);
      const auto pB = pcl::transformPointWithNormal(match._ptB_cam, frameB->_pose_in_model);
      ptsA_cur_pair.push_back(make_float4(pA.x, pA.y, pA.z, 1));
      ptsB_cur_pair.push_back(make_float4(pB.x, pB.y, pB.z, 1));
      normalsA_cur_pair.push_back(make_float4(pA.normal[0], pA.normal[1], pA.normal[2], 1));
      normalsB_cur_pair.push_back(make_float4(pB.normal[0], pB.normal[1], pB.normal[2], 1));
      uvA_cur_pair.push_back(make_float2(match._uA,match._vA));
      uvB_cur_pair.push_back(make_float2(match._uB,match._vB));
      ids_cur_pair.push_back(i);
      confs_cur_pair.push_back(1);
    }

    n_pts[i_pair] = int(ptsA_cur_pair.size());
    ids[i_pair] = ids_cur_pair;

    cudaMalloc(&ptsA_gpu[i_pair], sizeof(float4)*ptsA_cur_pair.size());
    cudaMalloc(&ptsB_gpu[i_pair], sizeof(float4)*ptsB_cur_pair.size());
    cudaMalloc(&normalsA_gpu[i_pair], sizeof(float4)*normalsA_cur_pair.size());
    cudaMalloc(&normalsB_gpu[i_pair], sizeof(float4)*normalsB_cur_pair.size());
    cudaMalloc(&uvA_gpu[i_pair], sizeof(float2)*uvA_cur_pair.size());
    cudaMalloc(&uvB_gpu[i_pair], sizeof(float2)*uvB_cur_pair.size());
    cudaMalloc(&confs_gpu[i_pair], sizeof(float)*confs_cur_pair.size());

    cutilSafeCall(cudaMemcpy(ptsA_gpu[i_pair], ptsA_cur_pair.data(), sizeof(float4)*ptsA_cur_pair.size(), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(ptsB_gpu[i_pair], ptsB_cur_pair.data(), sizeof(float4)*ptsB_cur_pair.size(), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(normalsA_gpu[i_pair], normalsA_cur_pair.data(), sizeof(float4)*normalsA_cur_pair.size(), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(normalsB_gpu[i_pair], normalsB_cur_pair.data(), sizeof(float4)*normalsB_cur_pair.size(), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(uvA_gpu[i_pair], uvA_cur_pair.data(), sizeof(float2)*uvA_cur_pair.size(), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(uvB_gpu[i_pair], uvB_cur_pair.data(), sizeof(float2)*uvB_cur_pair.size(), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(confs_gpu[i_pair], confs_cur_pair.data(), sizeof(float)*confs_cur_pair.size(), cudaMemcpyHostToDevice));
  }

  Eigen::Matrix4f best_pose(Eigen::Matrix4f::Identity());
  std::vector<std::vector<int>> inlier_ids;


  ransacMultiPairGPU(ptsA_gpu, ptsB_gpu, normalsA_gpu, normalsB_gpu, uvA_gpu, uvB_gpu, confs_gpu, n_pts, max_iter, inlier_dist, cos_normal_angle, max_transs, max_rots, inlier_ids);

  for (int i=0;i<pairs.size();i++)
  {
    const auto &frameA = pairs[i].first;
    const auto &frameB = pairs[i].second;
    const auto &inlier_ids_cur_pair = inlier_ids[i];
    const auto &ids_cur_pair = ids[i];

    auto &matches_cur_pair = _matches[{frameA, frameB}];
    const int n_prev = matches_cur_pair.size();
    std::vector<Correspondence> matches_new;
    for (int i_inlier=0;i_inlier<inlier_ids_cur_pair.size();i_inlier++)
    {
      matches_new.push_back(matches_cur_pair[ids_cur_pair[inlier_ids_cur_pair[i_inlier]]]);
    }
    matches_cur_pair = matches_new;
    if (matches_cur_pair.size()<min_match_after_ransac)
    {
      SPDLOG("after ransac, frame {} and {} has too few matches #{}, ignore",frameA->_id_str,frameB->_id_str,matches_cur_pair.size());
      matches_cur_pair.clear();
      continue;
    }
    SPDLOG("ransac makes match betwee frame {} {} #inliers={}, #prev {}", frameA->_id_str, frameB->_id_str, matches_cur_pair.size(), n_prev);

  }

  for (int i=0;i<pairs.size();i++)
  {
    cutilSafeCall(cudaFree(ptsA_gpu[i]));
    cutilSafeCall(cudaFree(ptsB_gpu[i]));
    cutilSafeCall(cudaFree(normalsA_gpu[i]));
    cutilSafeCall(cudaFree(normalsB_gpu[i]));
    cutilSafeCall(cudaFree(uvA_gpu[i]));
    cutilSafeCall(cudaFree(uvB_gpu[i]));
  }


}


void SiftManager::debugSampledMatch(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector<int> &chosen_match_ids, const Eigen::Matrix4f &offset, const std::vector<Correspondence> &inliers, bool compare_with_gt)
{
  bool is_gt_samples = true;
  const auto &gt_matches = _gt_matches[{frameA,frameB}];
  const auto &matches = _matches[{frameA,frameB}];
  for (int i=0;i<chosen_match_ids.size();i++)
  {
    const auto &match = matches[chosen_match_ids[i]];
    if (std::find(gt_matches.begin(),gt_matches.end(), match)==gt_matches.end())
    {
      is_gt_samples = false;
      return;
    }
  }
  if (is_gt_samples || compare_with_gt==false)
  {
    // SPDLOG("between frame {} and {}, chosen indices {} is gt match but rejected, solved offset in world\n{}", frameA->_id_str, frameB->_id_str, chosen_match_ids, offset);
    Eigen::Matrix4f B_in_cam = frameB->_pose_in_model.inverse();
    Eigen::Matrix4f A_in_cam = (offset * frameA->_pose_in_model).inverse();
    Eigen::Matrix4f poseAB_cam = B_in_cam * A_in_cam.inverse();

    float rot_offset = std::numeric_limits<float>::max(), trans_offset = std::numeric_limits<float>::max();
    if (frameA->_id==frameB->_id+1 && frameA->_ref_frame_id==frameB->_id)
    {
      rot_offset = Utils::rotationGeodesicDistance(poseAB_cam.block(0,0,3,3), Eigen::Matrix3f::Identity());
      trans_offset = (B_in_cam.block(0,3,3,1)-A_in_cam.block(0,3,3,1)).norm();
    }
    if (frameA->_ref_frame_id!=frameB->_id) //A has been estimated before, this pair of non-neighbor is for BA
    {
      rot_offset = Utils::rotationGeodesicDistance(offset.block(0,0,3,3), Eigen::Matrix3f::Identity());
      trans_offset = offset.block(0,3,3,1).norm();
    }
    SPDLOG("debugSampledMatch, frame {} and {}, trans_offset={}, rot_offset={}",frameA->_id_str, frameB->_id_str, trans_offset, rot_offset);

    PointCloudRGBNormal::Ptr ptsA_before(new PointCloudRGBNormal);
    PointCloudRGBNormal::Ptr ptsA_after(new PointCloudRGBNormal);
    PointCloudRGBNormal::Ptr ptsB(new PointCloudRGBNormal);
    cv::Mat vizA = frameA->_color.clone();
    cv::Mat vizB = frameB->_color.clone();
    if (inliers.size()==0)
    {
      for (int i=0;i<chosen_match_ids.size();i++)
      {
        const auto &match = matches[chosen_match_ids[i]];
        int r = rand()%255;
        int g = rand()%255;
        int b = rand()%255;
        auto ptA_cam = match._ptA_cam;
        ptA_cam.r = r;
        ptA_cam.g = g;
        ptA_cam.b = b;
        ptsA_before->points.push_back(ptA_cam);
        ptA_cam = pcl::transformPointWithNormal(ptA_cam, poseAB_cam);
        ptsA_after->points.push_back(ptA_cam);
        auto ptB_world = match._ptB_cam;
        ptB_world.r = r;
        ptB_world.g = g;
        ptB_world.b = b;
        ptsB->points.push_back(ptB_world);
        cv::circle(vizA, {match._uA, match._vA}, 2, {b,g,r}, 1, 1);
        cv::circle(vizB, {match._uB, match._vB}, 2, {b,g,r}, 1, 1);
      }
    }
    else
    {
      for (int i=0;i<inliers.size();i++)
      {
        const auto &match = inliers[i];
        int r = rand()%255;
        int g = rand()%255;
        int b = rand()%255;
        auto ptA_cam = match._ptA_cam;
        ptA_cam.r = r;
        ptA_cam.g = g;
        ptA_cam.b = b;
        ptsA_before->points.push_back(ptA_cam);
        ptA_cam = pcl::transformPointWithNormal(ptA_cam, poseAB_cam);
        ptsA_after->points.push_back(ptA_cam);
        auto ptB_world = match._ptB_cam;
        ptB_world.r = r;
        ptB_world.g = g;
        ptB_world.b = b;
        ptsB->points.push_back(ptB_world);
        cv::circle(vizA, {match._uA, match._vA}, 2, {b,g,r}, 1, 1);
        cv::circle(vizB, {match._uB, match._vB}, 2, {b,g,r}, 1, 1);
      }
    }

    pcl::io::savePLYFile("/home/bowen/debug/ptsA_before.ply", *ptsA_before);
    pcl::io::savePLYFile("/home/bowen/debug/ptsA_after.ply", *ptsA_after);
    pcl::io::savePLYFile("/home/bowen/debug/ptsB.ply", *ptsB);

    PointCloudRGBNormal::Ptr cloudA_in_B(new PointCloudRGBNormal);
    pcl::transformPointCloudWithNormals(*(frameA->_cloud),*cloudA_in_B, poseAB_cam);
    pcl::io::savePLYFile("/home/bowen/debug/cloudA_after.ply", *cloudA_in_B);
    pcl::io::savePLYFile("/home/bowen/debug/cloudB.ply", *(frameB->_cloud));
    pcl::io::savePLYFile("/home/bowen/debug/cloudA.ply", *(frameA->_cloud));

    cv::imshow("A",vizA);
    cv::imshow("B",vizB);
    cv::waitKey(0);
  }
}

void SiftManager::runMagsacBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  // if (countInlierCorres(frameA, frameB)<=5)
  // {
  //   _matches[{frameA, frameB}].clear();
  //   return;
  // }
  // const auto &matches = _matches[{frameA, frameB}];
  // std::vector<int> indices(matches.size());
  // std::iota(indices.begin(), indices.end(), 0);

  // // std::cout<<"matches: "<<matches<<std::endl;

  // std::vector<Correspondence> inliers;
  // const int num_sample = (*yml)["ransac"]["num_sample"].as<int>();
  // const int max_iter = (*yml)["ransac"]["max_iter"].as<int>();
  // const float inlier_dist = (*yml)["ransac"]["inlier_dist"].as<float>();
  // const float cos_normal_angle = std::cos((*yml)["ransac"]["inlier_normal_angle"].as<float>());
  // const float inlier_prob = (*yml)["ransac"]["inlier_prob"].as<float>();
  // const float desired_succ_rate = (*yml)["ransac"]["desired_succ_rate"].as<float>();
  // const float max_rot_deg = (*yml)["ransac"]["max_rot_deg"].as<float>();
  // const float max_trans = (*yml)["ransac"]["max_trans"].as<float>();
  // // int desired_num_iter = int(std::log(1-desired_succ_rate)/std::log(1-std::pow(inlier_prob,3)));
  // // int num_iter = std::min(max_iter, desired_num_iter);
  // // SPDLOG("ransac num_iter={}",num_iter);

  // cv::Mat points(matches.size(),4,CV_64F); // The point correspondences, each is of format x1 y1 x2 y2
  // for (int i=0;i<matches.size();i++)
  // {
  //   const auto &match = matches[i];
  //   points.at<double>(i,0) = match._uA;
  //   points.at<double>(i,1) = match._vA;
  //   points.at<double>(i,2) = match._uB;
  //   points.at<double>(i,3) = match._vB;
  // }

  // // Initialize the sampler used for selecting minimal samples
  // gcransac::sampler::UniformSampler main_sampler(&points);

  // MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator> magsac;
  // magsac.setMaximumThreshold(5.0); // The maximum noise scale sigma allowed
  // magsac.setIterationLimit(1e4); // Iteration limit to interrupt the cases when the algorithm run too long.

  // int iteration_number = 0; // Number of iterations required

  // magsac::utils::DefaultFundamentalMatrixEstimator estimator(0.1); // The robust homography estimator class containing the function for the fitting and residual calculation
  // gcransac::Pose6D model; // The estimated model
  // const bool success = magsac.run(points, // The data points
  //   desired_succ_rate, // The required confidence in the results
  //   estimator, // The used estimator
  //   main_sampler, // The sampler used for selecting minimal samples in each iteration
  //   model, // The estimated model
  //   iteration_number); // The number of iterations

  // _matches[{frameA, frameB}] = inliers;
}

void SiftManager::getMatch3DPointInCam(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const IndexPair &match, pcl::PointXYZRGBNormal &pcl_pt1, pcl::PointXYZRGBNormal &pcl_pt2)
{
  const auto &pts1 = frameA->_keypts;
  const auto &pts2 = frameB->_keypts;
  const auto &cloud1 = frameA->_cloud;
  const auto &cloud2 = frameB->_cloud;
  const cv::Point2f &pt1 = pts1[match.first].pt;
  const cv::Point2f &pt2 = pts2[match.second].pt;
  pcl_pt1 = cloud1->at(std::round(pt1.x), std::round(pt1.y));
  pcl_pt2 = cloud2->at(std::round(pt2.x), std::round(pt2.y));
}


/**
 * @brief
 *
 * @param frameA : earlier frame
 * @param frameB : later frame
 */
void SiftManager::pruneBadFeatureCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  // auto matches_tmp = _matches[{frameA, frameB}];
  // _matches[{frameA, frameB}].clear();
  // _matches[{frameA, frameB}].reserve(matches_tmp.size());
  // for (int i=0;i<matches_tmp.size();i++)
  // {
  //   const auto &match = matches_tmp[i];
  //   pcl::PointXYZRGBNormal pt1, pt2;
  //   getMatch3DPointInCam(frameA, frameB, match, pt1, pt2);
  //   pt1 = pcl::transformPointWithNormal(pt1,frameA->_pose_in_model);
  //   pt2 = pcl::transformPointWithNormal(pt2,frameB->_pose_in_model);
  //   Eigen::Vector3f P1(pt1.x,pt1.y,pt1.z);
  //   Eigen::Vector3f P2(pt2.x,pt2.y,pt2.z);
  //   if ((P1-P2).norm()>=0.01) continue;
  //   _matches[{frameA, frameB}].push_back(match);
  // }

}

int SiftManager::countInlierCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  int cnt = 0;
  const auto &corres = _matches[{frameA, frameB}];
  for (int i=0;i<corres.size();i++)
  {
    if (corres[i]._isinlier)
    {
      cnt++;
    }
  }
  return cnt;
}

void SiftManager::vizCorresBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::string &name)
{
  if ((*yml)["SPDLOG"].as<int>()<2) return;
  const auto &corres = _matches[{frameA,frameB}];
  if (corres.size()==0) return;
  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+_bundler->_newframe->_id_str+"/";
  if (!boost::filesystem::exists(out_dir))
  {
    system(("mkdir -p "+out_dir).c_str());
  }
  const std::string out_match_file = out_dir+frameA->_id_str+"_match_"+frameB->_id_str+"_"+name+".jpg";
  const auto& cornerA = frameA->_roi;
  const auto& cornerB = frameB->_roi; //umin,umax,vmin,vmax

  cv::Mat colorA = frameA->_color(cv::Rect(cornerA(0),cornerA(2),cornerA(1)-cornerA(0),cornerA(3)-cornerA(2))).clone();
  cv::Mat colorB = frameB->_color(cv::Rect(cornerB(0),cornerB(2),cornerB(1)-cornerB(0),cornerB(3)-cornerB(2))).clone();

  std::vector<Correspondence> matches_transformed;

  std::ofstream ff(fmt::format("{}/{}_match_{}_{}_uvs.txt",out_dir,frameA->_id_str,frameB->_id_str,name));
  for (int i=0;i<corres.size();i++)
  {
    if (!corres[i]._isinlier) continue;
    ff<<corres[i]._uA<<" "<<corres[i]._vA<<" "<<corres[i]._uB<<" "<<corres[i]._vB<<std::endl;

    float uA = corres[i]._uA-cornerA(0);
    float vA = corres[i]._vA-cornerA(2);
    float uB = corres[i]._uB-cornerB(0);
    float vB = corres[i]._vB-cornerB(2);
    Correspondence corr(uA,vA,uB,vB,corres[i]._ptA_cam,corres[i]._ptB_cam,true);
    corr._confidence = corres[i]._confidence;
    matches_transformed.push_back(corr);
  }
  ff.close();

  if ((*yml)["SPDLOG"].as<int>()<3) return;

  cv::Mat out;
  cvDrawMatches(colorA, colorB, matches_transformed, out);

  cv::imwrite(out_match_file, out, {cv::IMWRITE_JPEG_QUALITY,80});
}

void SiftManager::vizPositiveMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  if ((*yml)["SPDLOG"].as<int>()<1) return;

  const auto &K = frameA->_K;
  std::vector<Correspondence> positive_corres;
  auto matches = _matches[{frameA, frameB}];
  const float dist_thres = 0.005;
  for (int i=0;i<matches.size();i++)
  {
    const auto &match = matches[i];
    pcl::PointXYZRGBNormal ptA, ptB;
    float uA = match._uA;
    float vA = match._vA;
    float uB = match._uB;
    float vB = match._vB;
    float depthA = frameA->_depth_sim.at<float>(std::round(vA),std::round(uA));
    float depthB = frameB->_depth_sim.at<float>(std::round(vB),std::round(uB));
    ptA.x = (uA-K(0,2))*depthA/K(0,0);
    ptA.y = (vA-K(1,2))*depthA/K(1,1);
    ptA.z = depthA;
    ptB.x = (uB-K(0,2))*depthB/K(0,0);
    ptB.y = (vB-K(1,2))*depthB/K(1,1);
    ptB.z = depthB;
    ptA = pcl::transformPointWithNormal(ptA, frameA->_gt_pose_in_model);
    ptB = pcl::transformPointWithNormal(ptB, frameB->_gt_pose_in_model);
    float dist = pcl::geometry::distance(ptA, ptB);
    if (dist<=dist_thres)
    {
      positive_corres.push_back(match);
    }
  }
  _matches[{frameA, frameB}] = positive_corres;
  vizCorresBetween(frameA, frameB, "positive_match");

  _gt_matches[{frameA, frameB}] = positive_corres;

}


void SiftManager::vizCorres3DBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  if ((*yml)["SPDLOG"].as<int>()<2) return;

  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+_bundler->_newframe->_id_str+"/";
  if (!boost::filesystem::exists(out_dir))
  {
    system(std::string("mkdir -p "+out_dir).c_str());
  }
  const std::string out_match_file1 = out_dir+frameA->_id_str+"_match_"+frameB->_id_str+"_3D_"+frameA->_id_str+".ply";
  const std::string out_match_file2 = out_dir+frameA->_id_str+"_match_"+frameB->_id_str+"_3D_"+frameB->_id_str+".ply";

  const auto &pts1 = frameA->_keypts;
  const auto &pts2 = frameB->_keypts;
  const auto &cloud1 = frameA->_cloud;
  const auto &cloud2 = frameB->_cloud;

  PointCloudRGBNormal::Ptr cloudi(new PointCloudRGBNormal);
  PointCloudRGBNormal::Ptr cloudj(new PointCloudRGBNormal);
  pcl::copyPointCloud(*cloud1, *cloudi);
  pcl::copyPointCloud(*cloud2, *cloudj);

  auto corres = _matches[{frameA, frameB}];
  for (int i=0;i<corres.size();i++)
  {
    if (corres[i]._isinlier==false) continue;
    float uA = corres[i]._uA;
    float vA = corres[i]._vA;
    float uB = corres[i]._uB;
    float vB = corres[i]._vB;
    pcl::PointXYZRGBNormal &pcl_pt1 = cloudi->at(std::round(uA), std::round(vA));
    pcl::PointXYZRGBNormal &pcl_pt2 = cloudj->at(std::round(uB), std::round(vB));
    unsigned char r = rand()%255;
    unsigned char g = rand()%255;
    unsigned char b = rand()%255;
    pcl_pt1.r = r;
    pcl_pt1.g = g;
    pcl_pt1.b = b;
    pcl_pt2.r = r;
    pcl_pt2.g = g;
    pcl_pt2.b = b;
  }

  {
    pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
    pass.setInputCloud (cloudi);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.1, 2.0);
    pass.filter (*cloudi);
  }
  {
    pcl::PassThrough<pcl::PointXYZRGBNormal> pass;
    pass.setInputCloud (cloudj);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.1, 2.0);
    pass.filter (*cloudj);
  }

  pcl::transformPointCloudWithNormals(*cloudi,*cloudi,frameA->_pose_in_model);
  pcl::transformPointCloudWithNormals(*cloudj,*cloudj,frameB->_pose_in_model);

  pcl::io::savePLYFile(out_match_file1, *cloudi,true);
  pcl::io::savePLYFile(out_match_file2, *cloudj,true);

}

Lfnet::Lfnet(std::shared_ptr<YAML::Node> yml1, Bundler *bundler) : SiftManager(yml1, bundler), _context(1), _socket(_context, ZMQ_REQ)
{
  const std::string port = (*yml)["port"].as<std::string>();
  _socket.connect("tcp://0.0.0.0:"+port);
  SPDLOG("Connected to port {}", port);
}

Lfnet::~Lfnet()
{

}

void Lfnet::warpAndDetectFeature(std::shared_ptr<Frame> frame, const Eigen::Matrix4f &cur_in_init)
{
  const int H = frame->_H;
  const int W = frame->_W;
  const Eigen::Matrix3f K = frame->_K;
  const auto &depth = frame->_depth;
  std::vector<std::vector<Eigen::Matrix3f>> forward_transform(H, std::vector<Eigen::Matrix3f>(W, Eigen::Matrix3f::Identity()));
  Eigen::Matrix3f new_transform(Eigen::Matrix3f::Identity());

  cv::Mat map1 = cv::Mat::zeros(H,W,CV_32FC1);
  cv::Mat map2 = cv::Mat::zeros(H,W,CV_32FC1);

  for (int h=0;h<H;h++)
  {
    for (int w=0;w<W;w++)
    {
      int pos = h*W + w;
      new_transform.setIdentity();
      Eigen::Vector3f p(w,h,1);
      const float z = depth.at<float>(h,w);
      if (z<=0.1) continue;
      new_transform = K * cur_in_init.block(0,0,3,3) * z * K.inverse();
      p = new_transform * p;
      for (int row=0;row<3;row++)
      {
        for (int col=0;col<3;col++)
        {
          new_transform(row,col) /= p(2);
        }
      }
      forward_transform[h][w] = new_transform * forward_transform[h][w];
      int u = std::round(p(0)/p(2));
      int v = std::round(p(1)/p(2));
      if (u>=0 && u<W && v>=0 && v<H)
      {
        map1.at<float>(v,u) = w;
        map2.at<float>(v,u) = h;
      }
    }
  }

  cv::Mat img = cv::Mat::zeros(H,W,CV_8UC3);
  cv::remap(frame->_color, img, map1, map2, cv::INTER_LINEAR);
  cv::imshow("1",img);
  cv::waitKey(0);
}


void Lfnet::detectFeature(std::shared_ptr<Frame> frame, float rot_deg)
{
  const auto debug_dir = (*yml)["debug_dir"].as<std::string>();
  const auto &roi = frame->_roi;
  const int W = roi(1)-roi(0);
  const int H = roi(3)-roi(2);

  Eigen::Matrix3f forward_transform(Eigen::Matrix3f::Identity());
  Eigen::Matrix3f new_transform(Eigen::Matrix3f::Identity());

  int side = std::max(H,W);
  cv::Mat img = cv::Mat::zeros(side,side,CV_8UC3);
  for (int h=0;h<H;h++)
  {
    for (int w=0;w<W;w++)
    {
      img.at<cv::Vec3b>(h,w) = frame->_color.at<cv::Vec3b>(h+roi(2), w+roi(0));
    }
  }
  new_transform.setIdentity();
  new_transform(0,2) = -roi(0);
  new_transform(1,2) = -roi(2);
  forward_transform = new_transform * forward_transform;

  Eigen::AngleAxisf axis_angle;
  Eigen::Matrix3f R = frame->_pose_in_model.block(0,0,3,3).transpose();
  axis_angle.fromRotationMatrix(R);
  rot_deg = axis_angle.axis()(2) * axis_angle.angle();
  if (rot_deg!=0)
  {
    Eigen::Matrix3f tf1(Eigen::Matrix3f::Identity());
    Utils::getRotateImageTransform(side, side, rot_deg, tf1);
    forward_transform = tf1*forward_transform;
    // cv::imshow("after rot",img);
    // cv::waitKey(0);
  }

  const int H_input = (*yml)["feature_corres"]["resize"].as<int>();
  const int W_input = (*yml)["feature_corres"]["resize"].as<int>();

  int H0 = img.rows;
  int W0 = img.cols;

  cv::resize(img, img, {W_input, H_input});
  new_transform.setIdentity();
  new_transform(0,0) = W_input/float(W0);
  new_transform(1,1) = H_input/float(H0);
  forward_transform = new_transform * forward_transform;

  if ((*yml)["SPDLOG"].as<int>()>=2)
  {
    cv::imwrite(fmt::format("{}/{}/feature_det_image.png",debug_dir,frame->_id_str), img);
  }

  SPDLOG("prepare zmq msgs");

  {
    zmq::message_t msg(2*sizeof(int));
    std::vector<int> wh = {img.cols, img.rows};
    std::memcpy(msg.data(), wh.data(), 2*sizeof(int));
    _socket.send(msg, ZMQ_SNDMORE);
  }

  SPDLOG("sent zmq HW");

  {
    cv::Mat flat = img.reshape(1, img.total()*img.channels());
    std::vector<unsigned char> vec = img.isContinuous()? flat : flat.clone();
    zmq::message_t msg(vec.size()*sizeof(unsigned char));
    std::memcpy(msg.data(), vec.data(), vec.size()*sizeof(unsigned char));
    _socket.send(msg, 0);
  }

  SPDLOG("zmq start waiting for reply");
  std::vector<zmq::message_t> recv_msgs;
  zmq::recv_multipart(_socket, std::back_inserter(recv_msgs));
  SPDLOG("zmq got reply");

  std::vector<int> info(2);
  std::memcpy(info.data(), recv_msgs[0].data(), info.size()*sizeof(int));
  const int num_feat = info[0];
  const int feat_dim = info[1];
  SPDLOG("#feat={}, dim={}",num_feat,feat_dim);

  std::vector<float> kpts_array(num_feat*2);
  std::memcpy(kpts_array.data(), recv_msgs[1].data(), kpts_array.size()*sizeof(float));

  std::vector<float> feat_array(num_feat*feat_dim);
  std::memcpy(feat_array.data(), recv_msgs[2].data(), feat_array.size()*sizeof(float));

  frame->_keypts.resize(num_feat);
  frame->_feat_des = cv::Mat::zeros(num_feat, feat_dim, CV_32F);
  Eigen::Matrix3f backward_transform = forward_transform.inverse();
  for (int i=0;i<num_feat;i++)
  {
    Eigen::Vector3f p(kpts_array[2*i], kpts_array[2*i+1], 1);
    p = backward_transform * p;
    cv::KeyPoint kpt({p(0), p(1)}, 1);
    frame->_keypts[i] = kpt;

    for (int j=0;j<feat_dim;j++)
    {
      frame->_feat_des.at<float>(i,j) = feat_array[i*feat_dim+j];
    }
  }
  frame->_feat_des_gpu.upload(frame->_feat_des);
}



DeepOpticalFlow::DeepOpticalFlow()
{

}

DeepOpticalFlow::DeepOpticalFlow(std::shared_ptr<YAML::Node> yml1, Bundler *bundler) : Lfnet(yml1, bundler)
{

}


DeepOpticalFlow::~DeepOpticalFlow()
{

}


void DeepOpticalFlow::detectFeature(std::shared_ptr<Frame> frame, const float rot_deg)
{
  //Set to dummy values not used
  frame->_keypts.resize(100);
  frame->_feat_des = cv::Mat::zeros(100,128,CV_32F);
}

void DeepOpticalFlow::findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  if (_matches.find({frameA, frameB})!=_matches.end()) return;

  const int H = frameA->_H;
  const int W = frameA->_W;

  //Pad to the same size and then resize to the same size
  const auto &roiA = frameA->_roi;
  const auto &roiB = frameB->_roi;
  cv::Mat imgA, imgB;
  int H_max, W_max;
  // cropImagePair(frameA,frameB, imgA, imgB, H_max, W_max);
  imgA = frameA->_color.clone();
  imgB = frameB->_color.clone();
  // const int H_out = 480;
  // const int W_out = 480;
  // cv::resize(imgA, imgA, {W_out, H_out});
  // cv::resize(imgB, imgB, {W_out, H_out});

  Eigen::MatrixXf corres_array;
  exchangeZmqMsg(imgA, imgB, {W,H,W,H}, corres_array);

  _matches[{frameA,frameB}].clear();
  int n_pairs = corres_array.rows();
  for (int i=0;i<n_pairs;i++)
  {
    // float uA = corres_array[4*i  ];
    // float vA = corres_array[4*i+1];
    // float uB = corres_array[4*i+2];
    // float vB = corres_array[4*i+3];
    float uA = corres_array(i,0);
    float vA = corres_array(i,1);
    float uB = corres_array(i,2);
    float vB = corres_array(i,3);
    // SPDLOG("uA={},vA={},uB={},vB={}",uA,vA,uB,vB);
    if (!Utils::isPixelInsideImage(H, W,uA,vA) || !Utils::isPixelInsideImage(H, W,uB,vB)) continue;
    const auto &ptA = frameA->_cloud->at(std::round(uA),std::round(vA));
    const auto &ptB = frameB->_cloud->at(std::round(uB),std::round(vB));
    if (ptA.z<=0.1 || ptB.z<=0.1) continue;
    _matches[{frameA,frameB}].push_back(Correspondence(uA,vA,uB,vB,ptA,ptB,true));
  }

  // correspondenceFilterWinnerTakeAll(frameA, frameB);
  vizCorresBetween(frameA, frameB, "before_ransac");

  // vizPositiveMatches(frameA, frameB);

  runRansacBetween(frameA, frameB);
  vizCorresBetween(frameA, frameB, "after_ransac");


}


void DeepOpticalFlow::exchangeZmqMsg(const cv::Mat &img1, const cv::Mat &img2, const std::vector<int> &sizes, Eigen::MatrixXf &corres_array)
{
  using namespace zmq;

  SPDLOG("zmq start sending msgs");
  {
    zmq::message_t msg(sizes.size()*sizeof(int));
    std::memcpy(msg.data(), sizes.data(), sizes.size()*sizeof(int));
    _socket.send(msg, ZMQ_SNDMORE);
  }

  std::vector<cv::Mat> imgs = {img1,img2};
  for (int i=0;i<2;i++)
  {
    SPDLOG("zmq start copy data to msg {}",i);
    const auto &img = imgs[i];
    cv::Mat flat = img.reshape(1, img.total()*img.channels());
    std::vector<unsigned char> vec = img.isContinuous()? flat : flat.clone();

    zmq::message_t msg(vec.size()*sizeof(unsigned char));
    std::memcpy(msg.data(), vec.data(), vec.size()*sizeof(unsigned char));
    if (i==1)
    {
      _socket.send(msg, 0);
    }
    else
    {
      _socket.send(msg, ZMQ_SNDMORE);
    }
  }

  SPDLOG("zmq start waiting for reply");
  std::vector<zmq::message_t> recv_msgs;
  zmq::recv_multipart(_socket, std::back_inserter(recv_msgs));
  SPDLOG("zmq got reply");

  std::vector<int> dims(2);
  std::memcpy(dims.data(), recv_msgs[0].data(), sizeof(int)*dims.size());
  SPDLOG("dims={}x{}",dims[0],dims[1]);

  corres_array.resize(dims[0],dims[1]);
  // std::vector<float> corres_data(dims[0]*dims[1]);
  // std::memcpy(corres_data.data(), recv_msgs[1].data(), corres_data.size()*sizeof(float));
  // corres_array = Eigen::Map<Eigen::MatrixXf>(corres_data.data(),corres_array.rows(),corres_array.cols());
  corres_array = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((float*)recv_msgs[1].data(),corres_array.rows(),corres_array.cols());

}


void DeepOpticalFlow::exchangeZmqMsg(const std::vector<cv::Mat> &imgs, const std::vector<int> &sizes, std::vector<Eigen::MatrixXf> &corres_array)
{
  using namespace zmq;

  SPDLOG("zmq start sending msgs");
  {
    zmq::message_t msg(sizes.size()*sizeof(int));
    std::memcpy(msg.data(), sizes.data(), sizes.size()*sizeof(int));
    _socket.send(msg, ZMQ_SNDMORE);
  }

  for (int i=0;i<imgs.size();i++)
  {
    // SPDLOG("zmq start copy data to msg {}",i);
    const auto &img = imgs[i];
    cv::Mat flat = img.reshape(1, img.total()*img.channels());
    std::vector<unsigned char> vec = img.isContinuous()? flat : flat.clone();

    zmq::message_t msg(vec.size()*sizeof(unsigned char));
    std::memcpy(msg.data(), vec.data(), vec.size()*sizeof(unsigned char));
    if (i==imgs.size()-1)
    {
      _socket.send(msg, 0);
    }
    else
    {
      _socket.send(msg, ZMQ_SNDMORE);
    }
  }

  SPDLOG("zmq start waiting for reply");
  std::vector<zmq::message_t> recv_msgs;
  zmq::recv_multipart(_socket, std::back_inserter(recv_msgs));
  SPDLOG("zmq got reply recv_msgs#: {}",recv_msgs.size());

  int pos = 0;
  corres_array.clear();
  for (int i=0;i<imgs.size()/2;i++)
  {
    std::vector<int> dims(2);
    std::memcpy(dims.data(), recv_msgs[pos].data(), sizeof(int)*dims.size());
    pos++;
    SPDLOG("dims={}x{}",dims[0],dims[1]);

    Eigen::MatrixXf cur_corres_array;
    cur_corres_array.resize(dims[0],dims[1]);
    cur_corres_array = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((float*)recv_msgs[pos].data(),cur_corres_array.rows(),cur_corres_array.cols());
    pos++;
    corres_array.push_back(cur_corres_array);
  }

}


GluNet::GluNet()
{

}


GluNet::GluNet(std::shared_ptr<YAML::Node> yml1, Bundler *bundler) : DeepOpticalFlow(yml1,bundler)
{

}

GluNet::~GluNet()
{

}


void GluNet::findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  if (_matches.find({frameA, frameB})!=_matches.end()) return;

  const int H = frameA->_H;
  const int W = frameA->_W;
  const auto debug_dir = (*yml)["debug_dir"].as<std::string>();


  cv::Mat imgA, imgB;
  Eigen::Matrix3f tfA(Eigen::Matrix3f::Identity()), tfB(Eigen::Matrix3f::Identity());
  const int out_side = (*yml)["feature_corres"]["resize"].as<int>();

  // imgA = frameA->_color.clone();
  // imgB = frameB->_color.clone();
  // processImage(frameA, out_side, imgA, tfA);
  // processImage(frameB, out_side, imgB, tfB);

  processImagePair(frameA, frameB, imgA, imgB, out_side, true, tfA, tfB);

  if ((*yml)["SPDLOG"].as<int>()>=3)
  {
    cv::Mat canvas;
    cv::hconcat(imgA, imgB, canvas);
    cv::imwrite(fmt::format("{}/{}/feature_match_{}_{}.png",debug_dir,frameA->_id_str,frameA->_id_str,frameB->_id_str), canvas);
  }

  std::vector<Eigen::MatrixXf> corres_array_arr;
  // exchangeZmqMsg(imgA, imgB, {out_side,out_side,out_side,out_side}, corres_array);
  exchangeZmqMsg({imgA,imgB}, {2,out_side,out_side}, corres_array_arr);
  Eigen::MatrixXf corres_array = corres_array_arr[0];
  const int n_pairs = corres_array.rows();
  Eigen::Matrix3f tfA_inv = tfA.inverse();
  Eigen::Matrix3f tfB_inv = tfB.inverse();
  Eigen::MatrixXf corres = corres_array.block(0, 0, n_pairs, 4);
  corres.block(0, 0, n_pairs, 2) = Utils::transformPoints<float,float,3>(corres.block(0, 0, n_pairs, 2), tfA_inv);
  corres.block(0, 2, n_pairs, 2) = Utils::transformPoints<float,float,3>(corres.block(0, 2, n_pairs, 2), tfB_inv);
  _raw_matches[{frameA, frameB}] = corres.array().round().cast<uint16_t>();

  _matches[{frameA,frameB}].clear();
  _matches[{frameA,frameB}].reserve(n_pairs);

  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float max_dist_neighbor = (*yml)["feature_corres"]["max_dist_neighbor"].as<float>();
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);
  float dist_thres = std::numeric_limits<float>::max();
  float dot_thres = -1;
  // if (frameA->_ref_frame_id==frameB->_id && frameA->_id==frameB->_id+1)
  // {
  //   dist_thres = max_dist_neighbor;
  //   dot_thres = cos_max_normal_neighbor;
  // }
  // if (frameA->_ref_frame_id!=frameB->_id)
  // {
  //   dist_thres = max_dist_no_neighbor;
  //   dot_thres = cos_max_normal_no_neighbor;
  // }
  // SPDLOG("dist_thres: {}, dot_thres: {}",dist_thres,dot_thres);

  for (int i=0;i<n_pairs;i++)
  {
    const int grid_step = 1;
    std::map<std::array<int,2>, Eigen::Vector2f, std::less<std::array<int,2>>, Eigen::aligned_allocator<std::pair<const std::array<int,2>, Eigen::Vector2f> > > feature_gridA;
    std::map<std::array<int,2>, Eigen::Vector2f, std::less<std::array<int,2>>, Eigen::aligned_allocator<std::pair<const std::array<int,2>, Eigen::Vector2f> > > feature_gridB;
    const auto &corres_array = _raw_matches[{frameA, frameB}].cast<float>();
    float conf = corres_array.cols()==5? corres_array(i,4) : 1;
    auto corres = makeCorrespondence(corres_array(i,0), corres_array(i,1), corres_array(i,2), corres_array(i,3), frameA,frameB,dist_thres,dot_thres,conf);
    if (corres._isinlier==false) continue;

    // std::array<int,2> keyA = {uvA(0)/grid_step, uvA(1)/grid_step};
    // std::array<int,2> keyB = {uvB(0)/grid_step, uvB(1)/grid_step};
    // if (feature_gridA.find(keyA)!=feature_gridA.end()) continue;
    // if (feature_gridB.find(keyB)!=feature_gridB.end()) continue;
    // feature_gridA[keyA] = Eigen::Vector2f(uvA(0),uvA(1));
    // feature_gridB[keyB] = Eigen::Vector2f(uvB(0),uvB(1));

    _matches[{frameA,frameB}].push_back(corres);
  }

  ///////////!DEBUG
  // if (frameA->_id>=30)
  // {
  //   cv::Mat map = cv::Mat::zeros(H,W,CV_32FC2);
  //   const auto &matches = _matches[{frameA,frameB}];
  //   for (const auto &m:matches)
  //   {
  //     map.at<cv::Vec2f>(int(m._vA),int(m._uA)) = {m._uB, m._vB};
  //   }
  //   cv::Mat mapped;
  //   cv::remap(frameB->_color,mapped,map,cv::Mat(),cv::INTER_LINEAR);
  //   cv::imshow("A",frameA->_color);
  //   cv::imshow("B",frameB->_color);
  //   cv::imshow("B->A",mapped);
  //   cv::waitKey(0);
  // }

  vizCorresBetween(frameA, frameB, "before_ransac");

  // cv::imshow("1",visA);
  // cv::waitKey(0);
}



void GluNet::findCorresbyNNBatch(const std::vector<FramePair> &pairs)
{
  if (pairs.size()==0) return;

  const auto debug_dir = (*yml)["debug_dir"].as<std::string>();
  const int out_side = (*yml)["feature_corres"]["resize"].as<int>();

  std::vector<cv::Mat> imgs;
  std::vector<Eigen::Matrix3f> tfs;
  std::vector<FramePair> query_pairs;
  imgs.reserve(pairs.size());
  tfs.reserve(pairs.size());
  query_pairs.reserve(pairs.size());

  SPDLOG("process images before zmq");

  #pragma omp parallel for schedule (dynamic)
  for (int i=0;i<pairs.size();i++)
  {
    const auto &frameA = pairs[i].first;
    const auto &frameB = pairs[i].second;

    if (_raw_matches.find(pairs[i])!=_raw_matches.end())
    {
      SPDLOG("_raw_matches found exsting pair ({}, {})", frameA->_id_str, frameB->_id_str);
      continue;
    }

    cv::Mat imgA, imgB;
    Eigen::Matrix3f tfA(Eigen::Matrix3f::Identity()), tfB(Eigen::Matrix3f::Identity());
    processImagePair(frameA, frameB, imgA, imgB, out_side, true, tfA, tfB);

    #pragma omp critical
    {
      imgs.push_back(imgA);
      imgs.push_back(imgB);
      tfs.push_back(tfA);
      tfs.push_back(tfB);
      query_pairs.push_back(pairs[i]);
    }

    if ((*yml)["SPDLOG"].as<int>()>=3)
    {
      cv::Mat canvas;
      cv::hconcat(imgA, imgB, canvas);
      cv::imwrite(fmt::format("{}/{}/feature_match_{}_{}.png",debug_dir,frameA->_id_str,frameA->_id_str,frameB->_id_str), canvas);
    }
  }

  SPDLOG("query_pairs#:{}\n", query_pairs.size());
  std::vector<Eigen::MatrixXf> corres_array_arr;
  exchangeZmqMsg(imgs, {query_pairs.size()*2, out_side, out_side}, corres_array_arr);

  for (int i=0;i<query_pairs.size();i++)
  {
    Eigen::Matrix3f tfA_inv = tfs[i*2].inverse();
    Eigen::Matrix3f tfB_inv = tfs[i*2+1].inverse();
    const int n_corres = corres_array_arr[i].rows();
    Eigen::MatrixXf corres = corres_array_arr[i].block(0, 0, n_corres, 4);
    corres.block(0, 0, n_corres, 2) = Utils::transformPoints<float,float,3>(corres.block(0, 0, n_corres, 2), tfA_inv);
    corres.block(0, 2, n_corres, 2) = Utils::transformPoints<float,float,3>(corres.block(0, 2, n_corres, 2), tfB_inv);
    _raw_matches[query_pairs[i]] = corres.array().round().cast<uint16_t>();
  }

  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float max_dist_neighbor = (*yml)["feature_corres"]["max_dist_neighbor"].as<float>();
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);

  for (int i=0;i<pairs.size();i++)
  {
    const auto &frameA = pairs[i].first;
    const auto &frameB = pairs[i].second;
    _matches[{frameA,frameB}].clear();
    float dist_thres = std::numeric_limits<float>::max();
    float dot_thres = -1;
    // if (frameA->_ref_frame_id==frameB->_id && frameA->_id==frameB->_id+1)
    // {
    //   dist_thres = max_dist_neighbor;
    //   dot_thres = cos_max_normal_neighbor;
    // }
    // if (frameA->_ref_frame_id!=frameB->_id)
    // {
    //   dist_thres = max_dist_no_neighbor;
    //   dot_thres = cos_max_normal_no_neighbor;
    // }
    // SPDLOG("dist_thres: {}, dot_thres: {}",dist_thres,dot_thres);

    const int grid_step = 1;
    std::map<std::array<int,2>, Eigen::Vector2f, std::less<std::array<int,2>>, Eigen::aligned_allocator<std::pair<const std::array<int,2>, Eigen::Vector2f> > > feature_gridA;
    std::map<std::array<int,2>, Eigen::Vector2f, std::less<std::array<int,2>>, Eigen::aligned_allocator<std::pair<const std::array<int,2>, Eigen::Vector2f> > > feature_gridB;
    const auto &corres_array = _raw_matches[pairs[i]].cast<float>();
    for (int i=0;i<corres_array.rows();i++)
    {
      float conf = 1;
      Correspondence corres = makeCorrespondence(corres_array(i,0), corres_array(i,1), corres_array(i,2), corres_array(i,3), frameA,frameB,dist_thres,dot_thres,conf);
      if (corres._isinlier==false) continue;

      // cv::circle(visA,{int(uvA(0)),int(uvA(1))},1,{0,255,0},-1);

      // std::array<int,2> keyA = {uvA(0)/grid_step, uvA(1)/grid_step};
      // std::array<int,2> keyB = {uvB(0)/grid_step, uvB(1)/grid_step};
      // if (feature_gridA.find(keyA)!=feature_gridA.end()) continue;
      // if (feature_gridB.find(keyB)!=feature_gridB.end()) continue;
      // feature_gridA[keyA] = uvA.head(2);
      // feature_gridB[keyB] = uvB.head(2);
      _matches[{frameA,frameB}].push_back(corres);
    }
  }

}



std::tuple<std::vector<cv::Mat>, std::vector<Eigen::Matrix3f>, std::vector<FramePair>> GluNet::getProcessedImagePairs(const std::vector<FramePair> &pairs)
{
  const auto debug_dir = (*yml)["debug_dir"].as<std::string>();
  const int out_side = (*yml)["feature_corres"]["resize"].as<int>();

  std::vector<cv::Mat> imgs;
  std::vector<Eigen::Matrix3f> tfs;
  std::vector<FramePair> query_pairs;
  imgs.reserve(pairs.size());
  tfs.reserve(pairs.size());
  query_pairs.reserve(pairs.size());

  #pragma omp parallel for schedule (dynamic)
  for (int i=0;i<pairs.size();i++)
  {
    const auto &frameA = pairs[i].first;
    const auto &frameB = pairs[i].second;

    if (_raw_matches.find(pairs[i])!=_raw_matches.end())
    {
      SPDLOG("_raw_matches found exsting pair ({}, {})", frameA->_id_str, frameB->_id_str);
      continue;
    }

    cv::Mat imgA, imgB;
    Eigen::Matrix3f tfA(Eigen::Matrix3f::Identity()), tfB(Eigen::Matrix3f::Identity());
    processImagePair(frameA, frameB, imgA, imgB, out_side, true, tfA, tfB);

    #pragma omp critical
    {
      imgs.push_back(imgA);
      imgs.push_back(imgB);
      tfs.push_back(tfA);
      tfs.push_back(tfB);
      query_pairs.push_back(pairs[i]);
    }

    if ((*yml)["SPDLOG"].as<int>()>=3)
    {
      cv::Mat canvas;
      cv::hconcat(imgA, imgB, canvas);
      cv::imwrite(fmt::format("{}/{}/feature_match_{}_{}.png",debug_dir,frameA->_id_str,frameA->_id_str,frameB->_id_str), canvas);
    }
  }

  return {imgs, tfs, query_pairs};
}



void GluNet::rawMatchesToCorres(const std::vector<FramePair> &pairs)
{
  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float max_dist_neighbor = (*yml)["feature_corres"]["max_dist_neighbor"].as<float>();
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);

  for (int i=0;i<pairs.size();i++)
  {
    const auto &frameA = pairs[i].first;
    const auto &frameB = pairs[i].second;
    _matches[{frameA,frameB}].clear();
    float dist_thres = std::numeric_limits<float>::max();
    float dot_thres = -1;
    // if (frameA->_ref_frame_id==frameB->_id && frameA->_id==frameB->_id+1)
    // {
    //   dist_thres = max_dist_neighbor;
    //   dot_thres = cos_max_normal_neighbor;
    // }
    // if (frameA->_ref_frame_id!=frameB->_id)
    // {
    //   dist_thres = max_dist_no_neighbor;
    //   dot_thres = cos_max_normal_no_neighbor;
    // }
    // SPDLOG("dist_thres: {}, dot_thres: {}",dist_thres,dot_thres);

    const int grid_step = 1;
    std::map<std::array<int,2>, Eigen::Vector2f, std::less<std::array<int,2>>, Eigen::aligned_allocator<std::pair<const std::array<int,2>, Eigen::Vector2f> > > feature_gridA;
    std::map<std::array<int,2>, Eigen::Vector2f, std::less<std::array<int,2>>, Eigen::aligned_allocator<std::pair<const std::array<int,2>, Eigen::Vector2f> > > feature_gridB;
    const auto &corres_array = _raw_matches[pairs[i]].cast<float>();
    _matches[{frameA,frameB}].reserve(corres_array.rows());
    for (int i=0;i<corres_array.rows();i++)
    {
      float conf = 1;
      Correspondence corres = makeCorrespondence(corres_array(i,0), corres_array(i,1), corres_array(i,2), corres_array(i,3), frameA,frameB,dist_thres,dot_thres,conf);
      if (corres._isinlier==false) continue;

      // cv::circle(visA,{int(uvA(0)),int(uvA(1))},1,{0,255,0},-1);

      // std::array<int,2> keyA = {uvA(0)/grid_step, uvA(1)/grid_step};
      // std::array<int,2> keyB = {uvB(0)/grid_step, uvB(1)/grid_step};
      // if (feature_gridA.find(keyA)!=feature_gridA.end()) continue;
      // if (feature_gridB.find(keyB)!=feature_gridB.end()) continue;
      // feature_gridA[keyA] = uvA.head(2);
      // feature_gridB[keyB] = uvB.head(2);
      _matches[{frameA,frameB}].push_back(corres);
    }
  }

}



void GluNet::findCorresbyNNMultiPair(std::vector<FramePair> &pairs)
{
  const bool mutual = (*yml)["feature_corres"]["mutual"].as<bool>();
  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);

  std::vector<FramePair> pairs_to_run;
  for (const auto &pair:pairs)
  {
    const auto &fA = pair.first;
    const auto &fB = pair.second;
    if (_matches.find({fA, fB})!=_matches.end())
    {
      SPDLOG("Found _matches between ({}, {}), skip", fA->_id_str, fB->_id_str);
      continue;
    }
    // float rot_diff = Utils::rotationGeodesicDistanceIgnoreRotationAroundCamZ(fA->_pose_in_model.block(0,0,3,3).transpose(), fB->_pose_in_model.block(0,0,3,3).transpose());
    // if (rot_diff>(*yml)["bundle"]["non_neighbor_max_rot"].as<float>()/180.0*M_PI)
    // {
    //   SPDLOG("frame {} and {} rot_diff={} skip matching",fA->_id_str,fB->_id_str,rot_diff);
    //   continue;
    // }
    float visible = computeCovisibility(fA, fB);
    SPDLOG("frame {} and {} visible={}",fA->_id_str,fB->_id_str,visible);
    if (visible<(*yml)["bundle"]["non_neighbor_min_visible"].as<float>())
    {
      SPDLOG("frame {} and {} visible={} skip matching",fA->_id_str,fB->_id_str,visible);
      _matches[{fA,fB}].clear();
      continue;
    }

    // findCorresbyNN(fA,fB);
    pairs_to_run.push_back(pair);
  }

  findCorresbyNNBatch(pairs_to_run);

}



SceneFlow::SceneFlow(std::shared_ptr<YAML::Node> yml1, Bundler *bundler) : DeepOpticalFlow(yml1,bundler)
{

}

SceneFlow::~SceneFlow()
{

}

void SceneFlow::findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  using namespace zmq;
  assert(frameA->_id > frameB->_id);
  if (_matches.find({frameA, frameB})!=_matches.end()) return;

  const int H = frameA->_H;
  const int W = frameA->_W;

  //Pad to the same size and then resize to the same size
  const auto &K = frameA->_K;
  const auto &roiA = frameA->_roi;
  const auto &roiB = frameB->_roi;

  PointCloudRGBNormal::Ptr cloudB_pcl(new PointCloudRGBNormal);
  Eigen::MatrixXf cloudA, cloudB;
  cloudA.resize(H*W,3);
  cloudB.resize(H*W,3);
  int cntA = 0;
  int cntB = 0;
  for (int h=0;h<H;h++)
  {
    for (int w=0;w<W;w++)
    {
      if (frameA->_fg_mask.at<uchar>(h,w)>0 && frameA->_depth.at<float>(h,w)>=0.1)
      {
        const auto &P = (*frameA->_cloud)(w,h);
        cloudA.block(cntA,0,1,3)<<P.x, P.y, P.z;
        cntA++;
      }
      if (frameB->_fg_mask.at<uchar>(h,w)>0 && frameB->_depth.at<float>(h,w)>=0.1)
      {
        const auto &P = (*frameB->_cloud)(w,h);
        cloudB_pcl->points.push_back(P);
        cloudB.block(cntB,0,1,3)<<P.x, P.y, P.z;
        cntB++;
      }
    }
  }
  pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> kdtree;
  kdtree.setInputCloud(cloudB_pcl);

  const int n_sample = 8012;
  Eigen::MatrixXf cloudA_, cloudB_;
  cloudA_.resize(n_sample,3);
  cloudB_.resize(n_sample,3);
  for (int i=0;i<n_sample;i++)
  {
    int idA = rand()%cntA;
    cloudA_.row(i) = cloudA.row(idA);
    int idB = rand()%cntB;
    cloudB_.row(i) = cloudB.row(idB);
  }
  cloudA.swap(cloudA_);
  cloudB.swap(cloudB_);

  SPDLOG("zmq start sending msgs");
  int i_msg = 0;
  for (const auto &cloud:{cloudA,cloudB})
  {
    zmq::message_t msg(n_sample*3*sizeof(float));
    SPDLOG("msg.size()={}",msg.size());
    std::memcpy(msg.data(), cloud.data(), sizeof(float)*n_sample*3);  //Eigen is col major
    SPDLOG("zmq start sending one of msg {}, length={} bytes",i_msg,n_sample*3*sizeof(float));
    if (i_msg!=1)
    {
      _socket.send(msg, ZMQ_SNDMORE);
    }
    else
    {
      _socket.send(msg, 0);
    }
    SPDLOG("zmq finish sending one msg");
    i_msg++;
  }

  SPDLOG("zmq start waiting for reply");
  std::vector<zmq::message_t> recv_msgs;
  zmq::recv_multipart(_socket, std::back_inserter(recv_msgs));
  SPDLOG("zmq got reply");

  std::vector<float> flow_data(n_sample*3);
  std::memcpy(flow_data.data(), recv_msgs[0].data(), flow_data.size()*sizeof(float));
  for (int i=0;i<n_sample;i++)
  {
    pcl::PointXYZRGBNormal PA_shifted;
    PA_shifted.x = cloudA(i,0)+flow_data[i*3];
    PA_shifted.y = cloudA(i,1)+flow_data[i*3+1];
    PA_shifted.z = cloudA(i,2)+flow_data[i*3+2];
    std::vector<float> sq_dists;
    std::vector<int> indices;
    if (kdtree.nearestKSearch(PA_shifted, 1, indices, sq_dists) > 0)
    {
      const auto &neiB = cloudB_pcl->points[indices[0]];
      pcl::PointXYZRGBNormal ptA;
      ptA.x = cloudA(i,0);
      ptA.y = cloudA(i,1);
      ptA.z = cloudA(i,2);
      float uA = K(0,0)*cloudA(i,0)/cloudA(i,2) + K(0,2);
      float vA = K(1,1)*cloudA(i,1)/cloudA(i,2) + K(1,2);
      float uB = K(0,0)*neiB.x/neiB.z + K(0,2);
      float vB = K(1,1)*neiB.y/neiB.z + K(1,2);
      _matches[{frameA,frameB}].push_back(Correspondence(uA,vA,uB,vB,ptA,neiB,true));
    }
  }
}
