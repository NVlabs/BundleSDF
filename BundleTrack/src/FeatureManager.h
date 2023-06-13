/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef FEATURE_MANAGER_H_
#define FEATURE_MANAGER_H_
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/types.hpp>

// #include "magsac_utils.h"
// #include "utils.h"
// #include "magsac.h"
// #include "uniform_sampler.h"
// #include "flann_neighborhood_graph.h"
// #include "fundamental_estimator.h"
// #include "homography_estimator.h"
// #include "types.h"
// #include "model.h"
// #include "estimators.h"

#include "Utils.h"
#include "Frame.h"
#include "Bundler.h"

class Bundler;
class Frame;
class FramePairComparator;
class Correspondence;

typedef std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>> FramePair;
typedef std::pair<int,int> IndexPair;  // first < second !!
typedef std::map<FramePair, std::vector<IndexPair>, FramePairComparator, Eigen::aligned_allocator<std::pair<const FramePair, std::vector<IndexPair> > > > MatchMap;
typedef std::map<FramePair, Eigen::Matrix4f, FramePairComparator, Eigen::aligned_allocator<std::pair<const FramePair, Eigen::Matrix4f > > > PoseMap;

PYBIND11_MAKE_OPAQUE(std::map<int, std::shared_ptr<Frame>>);
PYBIND11_MAKE_OPAQUE(std::map<FramePair, std::vector<Correspondence>>);
PYBIND11_MAKE_OPAQUE(std::map<FramePair, Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>>);


class MapPoint
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  friend class boost::serialization::access;
  std::map<std::shared_ptr<Frame>, std::pair<float,float>> _img_pt;

public:
  MapPoint();
  MapPoint(std::shared_ptr<Frame> frame, float u, float v);
  ~MapPoint();

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar &_img_pt;
  };
};


class Correspondence
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  friend class boost::serialization::access;
  // int _feat_idA, _feat_idB;
  float _uA,_vA,_uB,_vB;   //A: later, B: earlier
  pcl::PointXYZRGBNormal _ptA_cam, _ptB_cam;
  float _confidence = 1;
  bool _isinlier = true;
  bool _ispropogated = false;

public:
  Correspondence();
  Correspondence(float uA, float vA, float uB, float vB, pcl::PointXYZRGBNormal ptA_cam, pcl::PointXYZRGBNormal ptB_cam, bool isinlier);
  ~Correspondence();
  bool operator == (const Correspondence &other) const;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar &_uA & _vA & _uB & _vB;
    ar &_ptA_cam & _ptB_cam;
    ar &_isinlier;
    ar &_ispropogated;
  };
};


class SiftManager
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  friend class boost::serialization::access;
  std::shared_ptr<YAML::Node> yml;
  cv::Ptr<cv::Feature2D> _detector;
  std::mt19937 _rng;
  Bundler *_bundler;

  std::map<FramePair, std::vector<Correspondence>> _matches;   //match between frame pair, first(later) --> second(earlier)
  std::map<FramePair, std::vector<Correspondence>> _gt_matches;
  std::map<FramePair, std::vector<std::shared_ptr<MapPoint>> > _covisible_mappoints;  //Undirected
  std::map<FramePair, Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>> _raw_matches;  // Not yet filtered ransac. Useful if _matches are deleted after nerf, so we dont need to query the network again for the raw matches.
  std::vector<std::shared_ptr<MapPoint>> _map_points_global;


public:
  SiftManager(){};
  SiftManager(std::shared_ptr<YAML::Node> yml1, Bundler *bundler);
  ~SiftManager();
  void detectFeature(std::shared_ptr<Frame> frame);
  void rejectFeatures(std::shared_ptr<Frame> frame);
  int countInlierCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void vizKeyPoints(std::shared_ptr<Frame> frame);
  void getMatch3DPointInCam(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const IndexPair &match, pcl::PointXYZRGBNormal &pcl_pt1, pcl::PointXYZRGBNormal &pcl_pt2);
  void updateFramePairMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void findCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  Correspondence makeCorrespondence(float uA, float vA, float uB, float vB, const std::shared_ptr<Frame> &fA, const std::shared_ptr<Frame> &fB, const float dist_thres, const float dot_thres, const float confidence);
  void findCorresMultiPairGPU(std::vector<FramePair> &pairs);
  void findCorresByMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  std::vector<std::shared_ptr<MapPoint>> getCovisibleMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  virtual void findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void pruneMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector< std::vector<cv::DMatch> > &knn_matchesAB, std::vector<cv::DMatch> &matches_AB);
  void collectMutualMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector<cv::DMatch> &matches_AB, const std::vector<cv::DMatch> &matches_BA);
  virtual void findCorresbyNNMultiPair(std::vector<FramePair> &pairs);
  void vizPositiveMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void findCorresbyGroundtruth(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void pruneBadFeatureCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void correspondenceFilterWinnerTakeAll(std::shared_ptr<Frame> framei, std::shared_ptr<Frame> framej);
  Eigen::Matrix4f procrustesByCorrespondence(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void vizCorresBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::string &name);
  void vizCorres3DBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void filterCorrespondenceByEssential(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void runRansacBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void runRansacMultiPairGPU(const std::vector<std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>>> &pairs);
  void runMagsacBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void debugSampledMatch(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector<int> &chosen_match_ids, const Eigen::Matrix4f &offset, const std::vector<Correspondence> &inliers, bool debugSampledMatch=false);
  void debugMapPoints();
  void forgetFrame(std::shared_ptr<Frame> frame);
};


class Lfnet : public SiftManager
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  zmq::context_t _context;
  zmq::socket_t _socket;

public:
  Lfnet(){};
  Lfnet(std::shared_ptr<YAML::Node> yml1, Bundler *bundler);
  ~Lfnet();
  void detectFeature(std::shared_ptr<Frame> frame, float rot_deg=0);
  void warpAndDetectFeature(std::shared_ptr<Frame> frame, const Eigen::Matrix4f &cur_in_init);
};



class DeepOpticalFlow : public Lfnet
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
  DeepOpticalFlow();
  DeepOpticalFlow(std::shared_ptr<YAML::Node> yml1, Bundler *bundler);
  ~DeepOpticalFlow();
  void detectFeature(std::shared_ptr<Frame> frame, const float rot_deg=0);
  void findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB) override;
  void exchangeZmqMsg(const cv::Mat &img1, const cv::Mat &img2, const std::vector<int> &sizes, Eigen::MatrixXf &corres_array);
  void exchangeZmqMsg(const std::vector<cv::Mat> &imgs, const std::vector<int> &sizes, std::vector<Eigen::MatrixXf> &corres_array);
};



class GluNet : public DeepOpticalFlow
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  GluNet();
  GluNet(std::shared_ptr<YAML::Node> yml1, Bundler *bundler);
  ~GluNet();
  void findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB) override;
  void findCorresbyNNMultiPair(std::vector<FramePair> &pairs) override;
  void findCorresbyNNBatch(const std::vector<FramePair> &pairs);
  void rawMatchesToCorres(const std::vector<FramePair> &pairs);
  std::tuple<std::vector<cv::Mat>, std::vector<Eigen::Matrix3f>, std::vector<FramePair>> getProcessedImagePairs(const std::vector<FramePair> &pairs);
};



class SceneFlow : public DeepOpticalFlow
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
  SceneFlow(std::shared_ptr<YAML::Node> yml1, Bundler *bundler);
  ~SceneFlow();
  void findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB) override;
};


#endif