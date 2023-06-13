/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef BUNDLER_HH__
#define BUNDLER_HH__


#include "Utils.h"
#include "Frame.h"
#include "FeatureManager.h"

class Loss;
class Frame;
class FramePtrComparator;
class SiftManager;
class FeatureMatchLoss;
class Lfnet;
class GluNet;
class DeepMatching;
class DeepOpticalFlow;
class SuperGlue;
class SceneFlow;
class OptimizerG2o;

class Bundler
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  friend class boost::serialization::access;
  typedef std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>> FramePair;

  std::shared_ptr<Frame> _newframe, _firstframe;
  std::deque<std::shared_ptr<Frame>> _keyframes;
  std::map<int,std::shared_ptr<Frame>> _frames;    //!NOTE store past few frames and keyframes. To make easier get frame by id, needed by frame->_ref_frame_id

  std::shared_ptr<GluNet> _fm;
  // std::shared_ptr<Lfnet> _fm;

  float _max_dist, _max_normal_angle;
  int _max_iter;
  bool _need_reinit;
  std::shared_ptr<YAML::Node> yml;
  std::vector<std::shared_ptr<Frame>> _local_frames;

  std::thread _global_optimzation_worker;
  std::thread::id _global_optimzation_id;
  std::atomic<bool> _need_global_optimization;

  zmq::context_t _context;
  zmq::socket_t _socket;


public:
  Bundler();
  Bundler(std::shared_ptr<YAML::Node> yml1);
  Bundler(const Bundler &other){};
  ~Bundler();
  bool forgetFrame(const std::shared_ptr<Frame> &f);
  void processNewFrame(std::shared_ptr<Frame> frame);
  bool checkAndAddKeyframe(std::shared_ptr<Frame> frame);
  void optimizeToPrev(std::shared_ptr<Frame> frame);
  void optimizeG2o(std::vector<std::shared_ptr<Frame>> frames);
  void optimizeGPU(std::vector<std::shared_ptr<Frame>> &frames, bool find_matches);
  void optimizationGlobal();
  void bruteForceCombination(std::set<std::shared_ptr<Frame>, FramePtrComparator> frames, std::set<std::shared_ptr<Frame>, FramePtrComparator> &best_frames, int pos, float &min_rot_dist);
  void selectKeyFramesForBA();
  std::vector<FramePair> getFeatureMatchPairs(std::vector<std::shared_ptr<Frame>> &frames);
  void maxNumEdgePathDfs(std::shared_ptr<Frame> cur, std::shared_ptr<Frame> goal, const std::vector<std::shared_ptr<Frame>> &frames_pool,  std::set<std::shared_ptr<Frame>, FramePtrComparator> &path, std::set<std::shared_ptr<Frame>, FramePtrComparator> &best_path, std::map<std::set<std::shared_ptr<Frame>, FramePtrComparator>, bool> &visited, int &best_n_edges);
  void nearEnoughRotSearch(std::shared_ptr<Frame> cur, std::shared_ptr<Frame> goal, const std::vector<std::shared_ptr<Frame>> &frames_pool,  std::set<std::shared_ptr<Frame>, FramePtrComparator> &path, std::set<std::shared_ptr<Frame>, FramePtrComparator> &best_path, std::map<std::set<std::shared_ptr<Frame>, FramePtrComparator>, bool> &visited);
  void saveNewframeResult();
  void saveFramesCloud(std::vector<std::shared_ptr<Frame>> frames, std::string prefix);
  void saveKeyframesPose();
  void saveFramesData(std::vector<std::shared_ptr<Frame>> frames, std::string foldername);
  void runNerf(std::vector<std::shared_ptr<Frame>> &frames);
};

#endif