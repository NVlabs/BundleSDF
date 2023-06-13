/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <random>
#include <iostream>
#include <stdint.h>
#include "Bundler.h"
#include "LossGPU.h"
#if GUROBI
#include "gurobi_c++.h"
#endif


typedef std::pair<int,int> IndexPair;
using namespace std;
using namespace Eigen;
// using namespace g2o;

Bundler::Bundler()
{

}


Bundler::Bundler(std::shared_ptr<YAML::Node> yml1): _context(1), _socket(_context, ZMQ_REQ)
{
  yml = yml1;

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  if ((*yml)["SPDLOG"].as<int>()<1)
  {
    spdlog::set_level(spdlog::level::off);
  }
  else
  {
    spdlog::set_level(spdlog::level::trace);
  }

  const std::string port = (*yml)["nerf_port"].as<std::string>();
  _socket.connect("tcp://0.0.0.0:"+port);
  SPDLOG("Connected to nerf_port {}", port);

  _fm = std::make_shared<GluNet>(yml, this);

  _need_global_optimization = true;
}


Bundler::~Bundler()
{
  SPDLOG("Destructor");
}

bool Bundler::forgetFrame(const std::shared_ptr<Frame> &f)
{
  if (f==NULL) return false;
  if (std::find(_keyframes.begin(),_keyframes.end(),f)==_keyframes.end())
  {
    SPDLOG("forgetting frame {}",f->_id_str);
    _fm->forgetFrame(f);
    _frames.erase(f->_id);
    return true;
  }
  return false;
};


void Bundler::processNewFrame(std::shared_ptr<Frame> frame)
{
  std::cout<<"\n\n";
  SPDLOG("New frame {}",frame->_id_str);
  _newframe = frame;

  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+frame->_id_str+"/";
  if ((*yml)["SPDLOG"].as<int>())
  {
    if (!boost::filesystem::exists(out_dir))
    {
      system(std::string("mkdir -p "+out_dir).c_str());
    }
  }

  std::shared_ptr<Frame> ref_frame = NULL;

  if (frame->_id>0)
  {
    ref_frame = _frames.rbegin()->second;
    frame->_ref_frame_id = ref_frame->_id;
    frame->_pose_in_model = ref_frame->_pose_in_model;
  }
  else  //The first frame
  {
    _firstframe = frame;
  }
  frame->invalidatePixelsByMask(frame->_fg_mask);
  if (frame->_id==0 && (frame->_pose_in_model-Eigen::Matrix4f::Identity()).norm()<1e-3)
  {
    frame->setNewInitCoordinate();  // Move first frame object cloud origin to center
    SPDLOG("Set new coordinate frame");
  }

  int n_fg = 0;
  for (int h=0;h<frame->_H;h++)
  {
    for (int w=0;w<frame->_W;w++)
    {
      if (frame->_fg_mask.at<uchar>(h,w)>0)
      {
        n_fg++;
      }
    }
  }
  SPDLOG("n_fg: {}",n_fg);
  if (n_fg<100)
  {
    frame->_status = Frame::FAIL;
    SPDLOG("Frame {} cloud is empty, marked FAIL, roi={}", frame->_id_str,n_fg);
    forgetFrame(frame);
    return;
  }

  const int min_match_with_ref = (*yml)["feature_corres"]["min_match_with_ref"].as<int>();

  if ((*yml)["depth_processing"]["denoise_cloud"].as<bool>())
  {
    frame->pointCloudDenoise();
  }

  int n_valid = frame->countValidPoints();
  int n_valid_first = _firstframe->countValidPoints();
  SPDLOG("n_valid/n_valid_first: {}/{}={}", n_valid, n_valid_first, float(n_valid)/n_valid_first);
  if (n_valid < n_valid_first/40.0)
  {
    SPDLOG("frame _cloud_down points#: {} too small compared to first frame points# {}, mark as FAIL",n_valid,n_valid_first);
    frame->_status = Frame::FAIL;
    forgetFrame(frame);
    return;
  }

  if (frame->_status==Frame::FAIL)
  {
    forgetFrame(frame);
    return;
  }

  try
  {
    float rot_deg = 0;
    Eigen::Matrix4f prev_in_init(Eigen::Matrix4f::Identity());
    _fm->detectFeature(frame, rot_deg);
  }
  catch (const std::exception &e)
  {
    SPDLOG("frame {} marked as FAIL since feature detection failed, ERROR: {}", frame->_id_str, e.what());
    frame->_status = Frame::FAIL;
    forgetFrame(frame);
    return;
  }

  _fm->vizKeyPoints(frame);

  if (_frames.size()>0)
  {
    _fm->findCorres(frame, ref_frame);
    if (_fm->_matches[{frame,ref_frame}].size()<min_match_with_ref)  // Find new reference frame from keyframes
    {
      SPDLOG("frame {} with last frame failed, re-choose new reference from keyframes",frame->_id_str);
      std::vector<float> visibles;
      for (const auto &kf:_keyframes)
      {
        float visible = computeCovisibility(frame, kf);
        visibles.push_back(visible);
      }
      std::vector<int> ids = Utils::vectorArgsort(visibles,false);
      bool found = false;
      for (const auto &id:ids)
      {
        const auto &kf = _keyframes[id];
        ref_frame = kf;
        frame->_ref_frame_id = kf->_id;
        frame->_pose_in_model = kf->_pose_in_model;
        _fm->findCorres(frame, kf);
        if (_fm->_matches[{frame,kf}].size()>=min_match_with_ref)
        {
          SPDLOG("re-choose new ref frame to {}",kf->_id_str);
          found = true;
          break;
        }
      }
      if (!found)
      {
        frame->_status = Frame::FAIL;
        SPDLOG("frame {} has not enough corres with ref_frame or any keyframe, mark as FAIL",frame->_id_str);
        forgetFrame(frame);
        return;
      }
    }

    if (frame->_status==Frame::FAIL)
    {
      forgetFrame(frame);
      return;
    }

    SPDLOG("frame {} pose update before\n{}", frame->_id_str, frame->_pose_in_model);
    Eigen::Matrix4f offset = _fm->procrustesByCorrespondence(frame, ref_frame);
    frame->_pose_in_model = offset * frame->_pose_in_model;
    SPDLOG("frame {} pose update after\n{}", frame->_id_str, frame->_pose_in_model);
  }

  if (frame->_status==Frame::FAIL)
  {
    forgetFrame(frame);
    return;
  }

  const int window_size = (*yml)["bundle"]["window_size"].as<int>();
  if (_frames.size()-_keyframes.size()>window_size)
  {
    for (auto &h:_frames)
    {
      bool isforget = forgetFrame(h.second);
      if (isforget)
      {
        SPDLOG("exceed window size, forget frame {}",h.second->_id_str);
        break;
      }
    }
  }

  _frames[frame->_id] = frame;

  if (frame->_id==0)  //First frame is always keyframe
  {
    checkAndAddKeyframe(frame);
    return;
  }

  if (frame->_id>=1)
  {
    selectKeyFramesForBA();
    optimizeGPU(_local_frames, true);
  }

  if (frame->_status==Frame::FAIL)
  {
    forgetFrame(frame);
    return;
  }

  bool added = checkAndAddKeyframe(frame);
}


bool Bundler::checkAndAddKeyframe(std::shared_ptr<Frame> frame)
{
  if (frame->_id==0)
  {
    _keyframes.push_back(frame);
    SPDLOG("Added frame {} as keyframe, current #keyframe: {}", frame->_id_str, _keyframes.size());
    return true;
  }
  if (frame->_status!=Frame::OTHER) return false;

  const int min_interval = (*yml)["keyframe"]["min_interval"].as<int>();
  const int min_feat_num = (*yml)["keyframe"]["min_feat_num"].as<int>();
  const float min_trans = (*yml)["keyframe"]["min_trans"].as<float>();
  const float min_rot = (*yml)["keyframe"]["min_rot"].as<float>()/180.0*M_PI;

  if (frame->_keypts.size()<min_feat_num)
  {
    SPDLOG("frame {} not selected as keyframe since its kpts size is {}", frame->_id_str, frame->_keypts.size());
    return false;
  }

  int n_valid = frame->countValidPoints();
  int n_first_valid = _firstframe->countValidPoints();
  if (n_valid<n_first_valid/10.0)
  {
    SPDLOG("frame {} not selected as keyframe, valid pts# {} too small compared to {}", n_valid, n_first_valid);
    return false;
  }

  //Check trans and rot diversity
  for (int i=0;i<_keyframes.size();i++)
  {
    const auto &kf = _keyframes[i];
    const auto &k_pose = kf->_pose_in_model;
    const auto &cur_pose = frame->_pose_in_model;
    float rot_diff = Utils::rotationGeodesicDistanceIgnoreRotationAroundCamZ(cur_pose.block(0,0,3,3).transpose(), k_pose.block(0,0,3,3).transpose());
    float trans_diff = (cur_pose.inverse().block(0,3,3,1)-k_pose.inverse().block(0,3,3,1)).norm();
    if (rot_diff<min_rot)
    {
      SPDLOG("frame {} not selected as keyframe since its rot diff with frame {} is {} deg", frame->_id_str, kf->_id_str, rot_diff/M_PI*180);
      return false;
    }
  }

  const float &min_visible = (*yml)["keyframe"]["min_visible"].as<float>();
  for (int i=0;i<_keyframes.size();i++)
  {
    const auto &kf = _keyframes[i];
    float visible = computeCovisibility(_newframe, kf);
    if (visible>min_visible)
    {
      SPDLOG("frame {} not selected as keyframe since share visible {} with frame {}", _newframe->_id_str, visible, kf->_id_str);
      return false;
    }
  }

  _keyframes.push_back(frame);
  SPDLOG("Added frame {} as keyframe, current #keyframe: {}", frame->_id_str, _keyframes.size());
  return true;

}


void Bundler::optimizeG2o(std::vector<std::shared_ptr<Frame>> frames)
{
#if G2O
  const int num_iter_outter = (*yml)["bundle"]["num_iter_outter"].as<int>();
  const int num_iter_inner = (*yml)["bundle"]["num_iter_inner"].as<int>();
  const int min_fm_edges_newframe = (*yml)["bundle"]["min_fm_edges_newframe"].as<int>();

  std::sort(frames.begin(), frames.end(), FramePtrComparator());
  printf("#optimizeGPU frames=%d, #keyframes=%d, #_frames=%d\n",frames.size(),_keyframes.size(),_frames.size());
  for (int i=0;i<frames.size();i++)
  {
    std::cout<<frames[i]->_id_str<<" ";
  }
  std::cout<<std::endl;

  int n_edges_newframe = 0;

#if CUDA_RANSAC==0
  for (int i=0;i<frames.size();i++)
  {
    for (int j=i+1;j<frames.size();j++)
    {
      const auto &frameA = frames[j];
      const auto &frameB = frames[i];
      _fm->findCorres(frameA, frameB);
    }
  }

#else
  std::vector<FramePair> pairs;

  const auto &non_neighbor_max_rot = (*yml)["non_neighbor_max_rot"].as<float>()/180.0*M_PI;
  for (int i=0;i<frames.size();i++)
  {
    const auto &f = frames[i];
    if (f==_newframe) continue;
    float rot_diff = Utils::rotationGeodesicDistanceIgnoreRotationAroundCamZ(f->_pose_in_model.block(0,0,3,3), _newframe->_pose_in_model.block(0,0,3,3));
    if (rot_diff<non_neighbor_max_rot)
    {
      pairs.push_back({_newframe, f});
    }
  }

  _fm->findCorresMultiPairGPU(pairs);
  if (_newframe->_status==Frame::FAIL) return;
#endif

  for (int i=0;i<frames.size();i++)
  {
    for (int j=i+1;j<frames.size();j++)
    {
      const auto &frameA = frames[j];
      const auto &frameB = frames[i];
      _fm->vizCorresBetween(frameA,frameB,"BA");
    }
  }


  saveFramesCloud(frames, "optCUDA_before");

  OptimizerG2o opt(yml, this);
  opt.optimizeFrames(_local_frames);
#endif
}


void Bundler::bruteForceCombination(std::set<std::shared_ptr<Frame>, FramePtrComparator> frames, std::set<std::shared_ptr<Frame>, FramePtrComparator> &best_frames, int pos, float &min_rot_dist)
{

  float sum = 0;
  std::vector<std::shared_ptr<Frame>> frames_vec(frames.begin(),frames.end());
  for (int i=0;i<frames_vec.size();i++)
  {
    for (int j=i+1;j<frames_vec.size();j++)
    {
      const auto &fA = frames_vec[i];
      const auto &fB = frames_vec[j];
      sum += Utils::rotationGeodesicDistance(fA->_pose_in_model.block(0,0,3,3), fB->_pose_in_model.block(0,0,3,3));
    }
  }

  const int max_BA_frames = (*yml)["bundle"]["max_BA_frames"].as<int>();
  if (frames.size()>=max_BA_frames)
  {
    if (sum<min_rot_dist)
    {
      min_rot_dist = sum;
      best_frames = frames;
    }
    return;
  }

  if (sum>=min_rot_dist) return;

  for (int i=pos+1;i<_keyframes.size();i++)
  {
    const auto &kf = _keyframes[i];
    if (frames.find(kf)!=frames.end()) continue;
    std::set<std::shared_ptr<Frame>, FramePtrComparator> cur_frames = frames;
    cur_frames.insert(kf);
    bruteForceCombination(cur_frames, best_frames, i, min_rot_dist);
  }
}

void Bundler::selectKeyFramesForBA()
{
  std::set<std::shared_ptr<Frame>, FramePtrComparator> frames = {_newframe};
  const auto &debug_dir = (*yml)["debug_dir"].as<std::string>();
  const int max_BA_frames = (*yml)["bundle"]["max_BA_frames"].as<int>();
  SPDLOG("total keyframes={}, want to select {}", _keyframes.size(), max_BA_frames);
  if (_keyframes.size()+frames.size()<=max_BA_frames)
  {
    for (const auto &kf:_keyframes)
    {
      frames.insert(kf);
    }
    _local_frames = std::vector<std::shared_ptr<Frame>>(frames.begin(),frames.end());
    return;
  }

  frames.insert(_keyframes[0]);

  const std::string method = (*yml)["bundle"]["subset_selection_method"].as<std::string>();
  if (method=="greedy_rot")
  {
    while (frames.size()<max_BA_frames)
    {
      float best_dist = std::numeric_limits<float>::max();
      std::shared_ptr<Frame> best_kf;
      for (int i=0;i<_keyframes.size();i++)
      {
        const auto &kf = _keyframes[i];
        if (frames.find(kf)!=frames.end()) continue;
        float cum_dist = 0;
        for (const auto &f:frames)
        {
          float rot_diff = Utils::rotationGeodesicDistanceIgnoreRotationAroundCamZ(kf->_pose_in_model.block(0,0,3,3).transpose(), f->_pose_in_model.block(0,0,3,3).transpose());
          cum_dist += rot_diff;
        }
        if (cum_dist<best_dist)
        {
          best_dist = cum_dist;
          best_kf = kf;
        }
      }
      frames.insert(best_kf);
    }
  }
  else if (method=="nearest_rotations")
  {
    frames = {_newframe};
    std::vector<float> rot_dists;
    std::vector<std::shared_ptr<Frame>> tmp_frames;

    for (const auto &kf:_keyframes)
    {
      if (frames.find(kf)!=frames.end()) continue;
      float rot_dist = Utils::rotationGeodesicDistanceIgnoreRotationAroundCamZ(_newframe->_pose_in_model.block(0,0,3,3).transpose(), kf->_pose_in_model.block(0,0,3,3).transpose());
      // SPDLOG("{} and {} rot_dist: {}[deg]", _newframe->_id_str, kf->_id_str, rot_dist/M_PI*180);
      rot_dists.push_back(rot_dist);
      tmp_frames.push_back(kf);
    }

    std::vector<int> ids = Utils::vectorArgsort(rot_dists, true);
    SPDLOG("ids#={}, max_BA_frames-frames.size()={}",ids.size(), max_BA_frames-frames.size());
    for (int i=0;i<ids.size();i++)
    {
      frames.insert(tmp_frames[ids[i]]);
      if (frames.size()==max_BA_frames)
      {
        break;
      }
    }
    SPDLOG("frames#={}",frames.size());
  }
  else if (method=="normal_orientation_nearest")
  {
    frames = {_newframe};
    std::vector<float> visibles(_keyframes.size());

    #pragma omp parallel for schedule (dynamic)
    for (int i=0;i<_keyframes.size();i++)
    {
      const auto &kf = _keyframes[i];
      // float visible = computeCovisibilityCuda(_newframe, kf);
      float visible = computeCovisibility(_newframe, kf);
      visibles[i] = visible;
      // SPDLOG("{} and {} visible: {}", _newframe->_id_str, kf->_id_str, visible);
    }
    std::vector<int> ids = Utils::vectorArgsort(visibles, false);
    SPDLOG("ids#={}, max_BA_frames-frames.size()={}",ids.size(), max_BA_frames-frames.size());
    for (int i=0;i<ids.size();i++)
    {
      frames.insert(_keyframes[ids[i]]);
      if (frames.size()==max_BA_frames)
      {
        break;
      }
    }
    SPDLOG("frames#={}",frames.size());
  }
  else if (method=="normal_orientation_greedy")
  {
    frames = {_newframe, _keyframes[0]};
    std::vector<float> visibles;
    while (frames.size()<max_BA_frames)
    {
      float best_visible = 0;
      std::shared_ptr<Frame> best_kf = NULL;
      for (int i=0;i<_keyframes.size();i++)
      {
        const auto &kf = _keyframes[i];
        if (frames.find(kf)!=frames.end()) continue;
        float visible_sum = 0;
        for (const auto &f:frames)
        {
          float visible = computeCovisibility(kf,f);
          visible_sum += visible;
        }
        if (visible_sum>best_visible)
        {
          best_visible = visible_sum;
          best_kf = kf;
        }
      }
      frames.insert(best_kf);
    }
  }
  else if (method=="greedy_covisible_points")
  {
    std::vector<std::shared_ptr<Frame>> ref_frames = {_keyframes[0], _newframe};
    while (frames.size()<max_BA_frames)   //!TODO: try real covisible pt, i.e. same point visible by multiple frames
    {
      int best_num = 0;
      std::shared_ptr<Frame> best_kf;
      for (int i=0;i<_keyframes.size();i++)
      {
        const auto &kf = _keyframes[i];
        if (frames.find(kf)!=frames.end()) continue;
        int num_cov_pts = 0;
        for (const auto &f:ref_frames)
        {
          auto cur_cov_pts = _fm->getCovisibleMapPoints(f,kf);
          num_cov_pts += cur_cov_pts.size();
        }
        if (num_cov_pts>best_num)
        {
          best_num = num_cov_pts;
          best_kf = kf;
        }
      }
      frames.insert(best_kf);
    }
  }
  else if (method=="max_edge")   //Super slow
  {
    std::set<std::shared_ptr<Frame>, FramePtrComparator> path, best_path;
    path.insert(_keyframes[0]);
    std::vector<std::shared_ptr<Frame>> frames_pool(_keyframes.begin()+1,_keyframes.end());
    frames_pool.push_back(_newframe);
    std::map<std::set<std::shared_ptr<Frame>, FramePtrComparator>, bool> visited;
    int best_n_edges = 0;
    maxNumEdgePathDfs(_keyframes[0], _newframe, frames_pool,path, best_path, visited, best_n_edges);
    frames = best_path;
  }
  else if (method=="near_enough_rot")
  {
    std::set<std::shared_ptr<Frame>, FramePtrComparator> path, best_path;
    path.insert(_keyframes[0]);
    std::vector<std::shared_ptr<Frame>> frames_pool(_keyframes.begin()+1,_keyframes.end());
    frames_pool.push_back(_newframe);
    std::map<std::set<std::shared_ptr<Frame>, FramePtrComparator>, bool> visited;
    nearEnoughRotSearch(_keyframes[0],_newframe, frames_pool,path, best_path, visited);
    frames = best_path;
  }
  else
  {
    std::cout<<"method not exist\n";
    exit(1);
  }

  _local_frames = std::vector<std::shared_ptr<Frame>>(frames.begin(),frames.end());

}


void Bundler::maxNumEdgePathDfs(std::shared_ptr<Frame> cur, std::shared_ptr<Frame> goal, const std::vector<std::shared_ptr<Frame>> &frames_pool,  std::set<std::shared_ptr<Frame>, FramePtrComparator> &path, std::set<std::shared_ptr<Frame>, FramePtrComparator> &best_path, std::map<std::set<std::shared_ptr<Frame>, FramePtrComparator>, bool> &visited, int &best_n_edges)
{
  // //!DEBUG
  // for (const auto &f:path)
  // {
  //   std::cout<<f->_id_str<<" ";
  // }
  // std::cout<<std::endl;


  if (visited.find(path)!=visited.end()) return;
  visited[path] = true;
  const int max_BA_frames = (*yml)["bundle"]["max_BA_frames"].as<int>();

  if (path.size()==max_BA_frames)
  {
    if (path.find(goal)!=path.end())
    {
      int num_cur=0;
      std::vector<std::shared_ptr<Frame>> frames(path.begin(),path.end());
      for (int i=0;i<frames.size();i++)
      {
        for (int j=i+1;j<frames.size();j++)
        {
          if (frames[i]->_id > frames[j]->_id)
          {
            num_cur += _fm->_matches[{frames[i],frames[j]}].size();
          }
          else
          {
            num_cur += _fm->_matches[{frames[j],frames[i]}].size();
          }
        }
      }

      if (best_path.size()==0)
      {
        best_path = path;
        best_n_edges = num_cur;
        return;
      }

      if (num_cur>best_n_edges)
      {
        best_path = path;
        best_n_edges = num_cur;
        return;
      }
    }
    return;
  }

  for (int i=0;i<frames_pool.size();i++)
  {
    const auto &kf = frames_pool[i];
    auto cur_path = path;
    cur_path.insert(kf);
    std::shared_ptr<Frame> fA,fB;
    if (kf->_id > cur->_id)
    {
      fA = kf;
      fB = cur;
    }
    else
    {
      fA = cur;
      fB = kf;
    }
    _fm->findCorres(fA,fB);
    if (_fm->_matches[{fA,fB}].size()>0)
    {
      maxNumEdgePathDfs(kf, goal, frames_pool, cur_path, best_path, visited, best_n_edges);
    }
  }
}



void Bundler::nearEnoughRotSearch(std::shared_ptr<Frame> cur, std::shared_ptr<Frame> goal, const std::vector<std::shared_ptr<Frame>> &frames_pool,  std::set<std::shared_ptr<Frame>, FramePtrComparator> &path, std::set<std::shared_ptr<Frame>, FramePtrComparator> &best_path, std::map<std::set<std::shared_ptr<Frame>, FramePtrComparator>, bool> &visited)
{
  // //!DEBUG
  // std::cout<<"path: ";
  // for (const auto &f:path)
  // {
  //   std::cout<<f->_id_str<<" ";
  // }
  // std::cout<<std::endl;


  if (visited.find(path)!=visited.end()) return;
  visited[path] = true;
  const int max_BA_frames = (*yml)["bundle"]["max_BA_frames"].as<int>();

  if (best_path.size()>0)
  {
    if (path.size()>best_path.size()) return; //No hope to be better
  }

  if (path.find(goal)!=path.end())
  {
    if (best_path.size()==0)
    {
      best_path = path;
      return;
    }
    if (path.size()<best_path.size())
    {
      best_path = path;
      return;
    }

    return;
  }

  for (int i=0;i<frames_pool.size();i++)
  {
    const auto &kf = frames_pool[i];
    float min_rot_diff = std::numeric_limits<float>::max();
    bool is_near = false;
    for (const auto &f:path)
    {
      float rot_diff = Utils::rotationGeodesicDistance(kf->_pose_in_model.block(0,0,3,3), f->_pose_in_model.block(0,0,3,3));
      if (rot_diff<30/180.0*M_PI)
      {
        is_near = true;
        break;
      }
    }
    if (!is_near) continue;
    auto cur_path = path;
    cur_path.insert(kf);
    nearEnoughRotSearch(kf, goal, frames_pool, cur_path, best_path, visited);
  }
}


void Bundler::optimizationGlobal()
{
  SPDLOG("start global optimization");
  _global_optimzation_id = std::this_thread::get_id();

  while (1)
  {
    if (!_need_global_optimization || _keyframes.size()<10)
    {
      sleep(0.1);
      continue;
    }
    _need_global_optimization = false;

    std::vector<std::shared_ptr<Frame>> frames(_keyframes.begin(),_keyframes.end());
    std::string s = "optimizationGlobal frames: ";
    for (const auto &f:frames)
    {
      s += f->_id_str+" ";
    }
    // SPDLOG(s);
    std::cout<<s<<std::endl;

    optimizeGPU(frames, true);
    std::cout<<"optimizationGlobal finished"<<std::endl;


    // SPDLOG("optimizationGlobal");
    // sleep(0.1);
  }
}


std::vector<FramePair> Bundler::getFeatureMatchPairs(std::vector<std::shared_ptr<Frame>> &frames)
{
  std::vector<FramePair> pairs;
  for (int i=0;i<frames.size();i++)
  {
    for (int j=i+1;j<frames.size();j++)
    {
      const auto &fA = frames[j];
      const auto &fB = frames[i];
      if (_fm->_matches.find({fA, fB})==_fm->_matches.end() && fA->_pose_in_model!=Eigen::Matrix4f::Identity())
      {
        float visible = computeCovisibility(fA, fB);
        SPDLOG("frame {} and {} visible={}",fA->_id_str,fB->_id_str,visible);
        if (visible<(*yml)["bundle"]["non_neighbor_min_visible"].as<float>())
        {
          SPDLOG("frame {} and {} visible={} skip matching",fA->_id_str,fB->_id_str,visible);
          _fm->_matches[{fA,fB}].clear();
          continue;
        }

        pairs.push_back({fA, fB});
        SPDLOG("add frame ({}, {}) into pairs", fA->_id_str, fB->_id_str);
      }
    }
  }
  return pairs;
}


void Bundler::optimizeGPU(std::vector<std::shared_ptr<Frame>> &frames, bool find_matches)
{
  const int num_iter_outter = (*yml)["bundle"]["num_iter_outter"].as<int>();
  const int num_iter_inner = (*yml)["bundle"]["num_iter_inner"].as<int>();
  const int min_fm_edges_newframe = (*yml)["bundle"]["min_fm_edges_newframe"].as<int>();

  std::sort(frames.begin(), frames.end(), FramePtrComparator());
  printf("#optimizeGPU frames=%d, #keyframes=%d, #_frames=%d\n",frames.size(),_keyframes.size(),_frames.size());
  for (int i=0;i<frames.size();i++)
  {
    std::cout<<frames[i]->_id_str<<" ";
  }
  std::cout<<std::endl;

  std::vector<EntryJ> global_corres;
  std::vector<int> n_match_per_pair;
  int n_edges_newframe = 0;

  if (find_matches)
  {
#if CUDA_RANSAC==0
    for (int i=0;i<frames.size();i++)
    {
      for (int j=i+1;j<frames.size();j++)
      {
        const auto &frameA = frames[j];
        const auto &frameB = frames[i];
        _fm->findCorres(frameA, frameB);
      }
    }

#else
    std::vector<FramePair> pairs;
    for (int i=0;i<frames.size();i++)
    {
      for (int j=i+1;j<frames.size();j++)
      {
        const auto &frameA = frames[j];
        const auto &frameB = frames[i];
        if (_fm->_matches.find({frameA, frameB})==_fm->_matches.end())
        {
          pairs.push_back({frameA, frameB});
          SPDLOG("add frame ({}, {}) into pairs", frameA->_id_str, frameB->_id_str);
        }
      }
    }

    _fm->findCorresMultiPairGPU(pairs);
    if (_newframe->_status==Frame::FAIL) return;

#endif
  }


  for (int i=0;i<frames.size();i++)
  {
    for (int j=i+1;j<frames.size();j++)
    {
      const auto &frameA = frames[j];
      const auto &frameB = frames[i];
      _fm->vizCorresBetween(frameA,frameB,"BA");
      const auto &matches = _fm->_matches[{frameA,frameB}];
      for (int k=0;k<matches.size();k++)
      {
        const auto &match = matches[k];
        EntryJ corres;
        corres.imgIdx_j = j;
        corres.imgIdx_i = i;
        corres.normal_i = make_float3(match._ptA_cam.normal_x,match._ptA_cam.normal_y,match._ptA_cam.normal_z);
        corres.normal_j = make_float3(match._ptB_cam.normal_x,match._ptB_cam.normal_y,match._ptB_cam.normal_z);
        corres.pos_j = make_float3(match._ptA_cam.x,match._ptA_cam.y,match._ptA_cam.z);
        corres.pos_i = make_float3(match._ptB_cam.x,match._ptB_cam.y,match._ptB_cam.z);
        global_corres.push_back(corres);
        // if (frameA==_newframe || frameB==_newframe)
        // {
        //   n_edges_newframe++;
        // }
      }
      n_match_per_pair.push_back(matches.size());
    }
  }

  if (global_corres.size()==0)
  {
    SPDLOG("frame {} few global_corres, mark as FAIL",_newframe->_id_str);
    _newframe->_status = Frame::FAIL;
  }

  const int H = frames[0]->_H;
  const int W = frames[0]->_W;
  const int n_pixels = H*W;

  std::vector<float*> depths_gpu;
  std::vector<uchar4*> colors_gpu;
  std::vector<float4*> normals_gpu;
  std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
  std::vector<int> update_pose_flags(frames.size(),1);
  for (int i=0;i<frames.size();i++)
  {
    const auto &f = frames[i];
    depths_gpu.push_back(f->_depth_gpu);
    colors_gpu.push_back(f->_color_gpu);
    normals_gpu.push_back(f->_normal_gpu);
    poses.push_back(f->_pose_in_model);
    if (i==0 || frames[i]->_nerfed) update_pose_flags[i] = 0;
    // if (i==0) update_pose_flags[i] = 0;
  }

  saveFramesCloud(frames, "optCUDA_before");

  SPDLOG("OptimizerGPU begin, global_corres#={}",global_corres.size());
  OptimizerGpu opt(yml);
  opt._id_str = _newframe->_id_str;
  opt.optimizeFrames(global_corres, n_match_per_pair, frames.size(), H, W, depths_gpu, colors_gpu, normals_gpu, update_pose_flags, poses, frames[0]->_K);
  SPDLOG("OptimizerGPU finish");

  /////////If the newframe has abnormal pose change
  if (_newframe->_ref_frame_id==_newframe->_id-1 && _frames.find(_newframe->_ref_frame_id)!=_frames.end())
  {
    const float &max_trans = (*yml)["ransac"]["max_trans_neighbor"].as<float>();
    const float &max_rot = (*yml)["ransac"]["max_rot_deg_neighbor"].as<float>()/180.0*M_PI;
    const auto &ref_frame = _frames[_newframe->_ref_frame_id];
    float trans_diff = (_newframe->_pose_in_model.inverse().block(0,3,3,1)-ref_frame->_pose_in_model.inverse().block(0,3,3,1)).norm();
    if (trans_diff>max_trans)
    {
      _newframe->_status = Frame::FAIL;
      fmt::print("frame {} trans_diff to neighbor: {} too big, FAIL", _newframe->_id_str, trans_diff);
      return;
    }
    float rot_diff = Utils::rotationGeodesicDistance(_newframe->_pose_in_model.inverse().block(0,0,3,3), ref_frame->_pose_in_model.inverse().block(0,0,3,3));
    if (rot_diff>max_rot)
    {
      _newframe->_status = Frame::FAIL;
      fmt::print("frame {} rot_diff to neighbor: {} too big, FAIL", _newframe->_id_str, rot_diff);
      return;
    }
  }

  for (int i=0;i<frames.size();i++)
  {
    const auto &f = frames[i];
    f->_pose_in_model = poses[i];
  }

  saveFramesCloud(frames, "optCUDA_after");

}


void Bundler::saveNewframeResult()
{
  SPDLOG("Welcome saveNewframeResult");
  std::string K_file = fmt::format("{}/cam_K.txt",(*yml)["debug_dir"].as<std::string>());
  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();
  const std::string out_dir = debug_dir+_newframe->_id_str+"/";
  const std::string pose_out_dir = debug_dir+"ob_in_cam/";

  if (!boost::filesystem::exists(K_file))
  {
    std::ofstream ff(K_file);
    ff<<std::setprecision(10)<<_newframe->_K<<std::endl;
    ff.close();
  }

  if (!boost::filesystem::exists(pose_out_dir))
  {
    system(std::string("mkdir -p "+pose_out_dir).c_str());
    system(std::string("mkdir -p "+debug_dir+"/color").c_str());
    system(std::string("mkdir -p "+debug_dir+"/color_viz").c_str());
    system(std::string("mkdir -p "+debug_dir+"/color_keyframes").c_str());
    system(std::string("mkdir -p "+debug_dir+"/depth").c_str());
    system(std::string("mkdir -p "+debug_dir+"/depth_filtered").c_str());
    system(std::string("mkdir -p "+debug_dir+"/depth_vis").c_str());
    system(std::string("mkdir -p "+debug_dir+"/normal").c_str());
    system(std::string("mkdir -p "+debug_dir+"/mask").c_str());
  }

  Eigen::Matrix4f cur_in_model = _newframe->_pose_in_model;
  Eigen::Matrix4f ob_in_cam = cur_in_model.inverse();

  if ((*yml)["SPDLOG"].as<int>()>=1)
  {
    std::ofstream ff(pose_out_dir+_newframe->_id_str+".txt");
    ff<<std::setprecision(10)<<ob_in_cam<<std::endl;
    ff.close();

    cv::imwrite(fmt::format("{}/color/{}.png",debug_dir,_newframe->_id_str),_newframe->_color_raw);
    cv::Mat depth_u16;
    _newframe->_depth_raw.convertTo(depth_u16,CV_16UC1,1000);
    cv::imwrite(fmt::format("{}/depth/{}.png",debug_dir,_newframe->_id_str),depth_u16);
    _newframe->_depth.convertTo(depth_u16,CV_16UC1,1000);
    cv::imwrite(fmt::format("{}/depth_filtered/{}.png",debug_dir,_newframe->_id_str),depth_u16);
    cv::Mat mask = cv::Mat::zeros(_newframe->_H,_newframe->_W,CV_8U);
    for (int h=0;h<_newframe->_H;h++)
    {
      for (int w=0;w<_newframe->_W;w++)
      {
        if (_newframe->_fg_mask.at<uchar>(h,w)>0)
        {
          mask.at<uchar>(h,w) = 255;
        }
      }
    }
    cv::imwrite(fmt::format("{}/mask/{}.png",debug_dir,_newframe->_id_str),mask);

    ///////// Normal
    cv::Mat normal_img = cv::Mat::zeros(_newframe->_H,_newframe->_W,CV_8UC3);
    for (int h=0;h<_newframe->_H;h++)
    {
      for (int w=0;w<_newframe->_W;w++)
      {
        Eigen::Vector3f n(Eigen::Vector3f::Zero());
        if (_newframe->_depth.at<float>(h,w)>=0.1)
        {
          const auto &pt = (*_newframe->_cloud)(w,h);
          n << pt.normal_x, pt.normal_y, pt.normal_z;
          n.normalize();
        }
        n.array() = (n.array()+1)/2.0 * 255;
        normal_img.at<cv::Vec3b>(h,w) = {n(2),n(1),n(0)};    // BGR
      }
    }
    cv::imwrite(fmt::format("{}/normal/{}.png",debug_dir,_newframe->_id_str),normal_img);

    const std::string raw_dir = debug_dir+"color_segmented/";
    if (!boost::filesystem::exists(raw_dir))
    {
      system(std::string("mkdir -p "+raw_dir).c_str());
    }
    cv::imwrite(raw_dir+_newframe->_id_str+".png",_newframe->_color);

    const int H = _newframe->_H;
    const int W = _newframe->_W;
    {
      cv::Mat depth_vis = cv::Mat::zeros(H,W,CV_8U);
      for (int h=0;h<H;h++)
      {
        for (int w=0;w<W;w++)
        {
          if (_newframe->_depth.at<float>(h,w)<0.1) continue;
          depth_vis.at<uchar>(h,w) = 1.0/_newframe->_depth.at<float>(h,w) / 10 * 255;
        }
      }
      cv::imwrite(fmt::format("{}/depth_vis/{}.png",debug_dir,_newframe->_id_str),depth_vis);
    }

    {
      std::ofstream ff(fmt::format("{}/opt_frames.txt",out_dir));
      for (const auto &f:_local_frames)
      {
        ff<<f->_id_str<<" ";
      }
      ff.close();
    }
  }

  if ((*yml)["SPDLOG"].as<int>()>=1)
  {
    //////////// Save keyframe poses
    YAML::Node node;
    for (const auto &kf:_keyframes)
    {
      std::vector<float> data;
      for (int h=0;h<4;h++)
      {
        for (int w=0;w<4;w++)
        {
          data.push_back(kf->_pose_in_model(h,w));
        }
      }
      node[fmt::format("keyframe_{}",kf->_id_str)]["cam_in_ob"] = data;   //!NOTE Just saving id_str causes ambiguity for yaml to load
    }
    std::ofstream ff(fmt::format("{}/{}/keyframes.yml",debug_dir,_newframe->_id_str));
    ff<<node<<std::endl;
    ff.close();

    ff.open(fmt::format("{}/{}/frame.txt",debug_dir,_newframe->_id_str));
    ff<<"status: "<<_newframe->_status<<std::endl;
    if (_newframe->_ref_frame_id>=0)
    {
      ff<<"ref_frame_id: "<<_newframe->_ref_frame_id<<std::endl;
      ff<<"ref_frame_id_str: "<<_frames[_newframe->_ref_frame_id]->_id_str<<std::endl;
    }
    ff.close();
  }


  if ((*yml)["SPDLOG"].as<int>()>=4)
  {
    PointCloudRGBNormal::Ptr cloud_raw_depth(new PointCloudRGBNormal);
    Utils::convert3dOrganizedRGB(_newframe->_depth_raw, _newframe->_color_raw, _newframe->_K, cloud_raw_depth);
    Utils::downsamplePointCloud(cloud_raw_depth,cloud_raw_depth,0.003);
    Utils::passFilterPointCloud(cloud_raw_depth,cloud_raw_depth,"z",0.1,(*yml)["depth_processing"]["zfar"].as<float>());
    pcl::io::savePLYFile(out_dir+"cloud_raw_depth.ply",*cloud_raw_depth);

    PointCloudRGBNormal::Ptr cloud_world_gt(new PointCloudRGBNormal);
    pcl::transformPointCloudWithNormals(*(_newframe->_cloud_down), *cloud_world_gt, _newframe->_gt_pose_in_model);
    pcl::io::savePLYFile(out_dir+"cloud_world_gt.ply",*cloud_world_gt);
  }

  SPDLOG("saveNewframeResult done");
}



void Bundler::saveFramesCloud(std::vector<std::shared_ptr<Frame>> frames, std::string prefix)
{
  if ((*yml)["SPDLOG"].as<int>()<4) return;

  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+_newframe->_id_str+"/";
  if (!boost::filesystem::exists(out_dir))
  {
    system(std::string("mkdir -p "+out_dir).c_str());
  }

  for (int i=0;i<frames.size();i++)
  {
    const auto &frame = frames[i];
    PointCloudRGBNormal::Ptr cloud(new PointCloudRGBNormal);
    Utils::downsamplePointCloud(frame->_cloud, cloud, 0.001);
    pcl::transformPointCloudWithNormals(*cloud, *cloud, frame->_pose_in_model);
    pcl::io::savePLYFile(out_dir+prefix+"_"+frame->_id_str+".ply", *cloud);
    SPDLOG("Save to {}",out_dir+prefix+"_"+frame->_id_str+".ply");
  }
};



void Bundler::saveKeyframesPose()
{
  if ((*yml)["SPDLOG"].as<int>()<2) return;
  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+"/keyframes_pose/";
  if (!boost::filesystem::exists(out_dir))
  {
    system(std::string("mkdir -p "+out_dir).c_str());
  }

  for (int i=0;i<_keyframes.size();i++)
  {
    const auto &kf = _keyframes[i];
    std::ofstream ff(out_dir+"pose_in_model"+kf->_id_str+".txt");
    ff<<kf->_pose_in_model<<std::endl;
    ff.close();
  }
}

void Bundler::saveFramesData(std::vector<std::shared_ptr<Frame>> frames, std::string foldername)
{
  std::cout << std::setprecision(7);
  std::string s = "saveFramesData frames: ";
  for (const auto &f:frames)
  {
    s += f->_id_str+" ";
  }
  SPDLOG(s);

  const std::string out_dir = fmt::format("{}/{}",(*yml)["debug_dir"].as<std::string>(),foldername);
  system(fmt::format("rm -rf {} && mkdir -p {}",out_dir,out_dir).c_str());

  {
    std::ofstream ff(fmt::format("{}/K.txt",out_dir));
    ff<<frames[0]->_K<<std::endl;
    ff.close();
  }

  ///////// Save data to compare with other optimization
  for (const auto &f:frames)
  {
    std::string tmp_dir = fmt::format("{}/{}",out_dir,f->_id_str);
    system(fmt::format("mkdir -p {}",tmp_dir).c_str());
    cv::imwrite(fmt::format("{}/color.png",tmp_dir), f->_color_raw);
    cv::Mat depth;
    f->_depth.convertTo(depth,CV_16U,1000);
    cv::imwrite(fmt::format("{}/depth.png",tmp_dir), depth);
    cv::imwrite(fmt::format("{}/mask.png",tmp_dir), f->_fg_mask);
    std::ofstream ff(fmt::format("{}/cam_in_ob.txt",tmp_dir));
    ff<<f->_pose_in_model<<std::endl;
    ff.close();
  }
  for (int i=0;i<frames.size();i++)
  {
    for (int j=i+1;j<frames.size();j++)
    {
      auto fi = frames[i];
      auto fj = frames[j];
      if (fi->_id > fj->_id)
      {
        std::swap(fi,fj);
      }
      if (_fm->_matches.find({fj,fi})==_fm->_matches.end())
      {
        SPDLOG("{} and {} no feature matches found",fj->_id_str,fi->_id_str);
        continue;
      }
      const auto &matches = _fm->_matches[{fj,fi}];
      std::ofstream ff(fmt::format("{}/matches_{}_{}.txt",out_dir,fj->_id_str,fi->_id_str));
      for (int i_match=0;i_match<matches.size();i_match++)
      {
        const auto &m = matches[i_match];
        ff<<fmt::format("{} {} {} {}",m._uA,m._vA,m._uB,m._vB)<<std::endl;
      }
      ff.close();
    }
  }
  SPDLOG("saved global frames results");
}


void Bundler::runNerf(std::vector<std::shared_ptr<Frame>> &frames)
{
  SPDLOG("NERF start");
  const std::string debug_dir = (*yml)["debug_dir"].as<std::string>();

  /////////// Write images to load by NERF
  const auto &last_frame = frames.back();
  cv::imwrite(fmt::format("{}/color/{}.png",debug_dir,last_frame->_id_str),last_frame->_color_raw);
  cv::Mat depth_u16;
  last_frame->_depth.convertTo(depth_u16,CV_16UC1,1000);
  cv::imwrite(fmt::format("{}/depth_filtered/{}.png",debug_dir,last_frame->_id_str),depth_u16);
  cv::Mat mask = cv::Mat::zeros(last_frame->_H,last_frame->_W,CV_8U);
  for (int h=0;h<last_frame->_H;h++)
  {
    for (int w=0;w<last_frame->_W;w++)
    {
      if (last_frame->_fg_mask.at<uchar>(h,w)>0)
      {
        mask.at<uchar>(h,w) = 255;
      }
    }
  }
  cv::imwrite(fmt::format("{}/mask/{}.png",debug_dir,last_frame->_id_str),mask);

  ///////// Normal
  cv::Mat normal_img = cv::Mat::zeros(last_frame->_H,last_frame->_W,CV_8UC3);
  for (int h=0;h<last_frame->_H;h++)
  {
    for (int w=0;w<last_frame->_W;w++)
    {
      Eigen::Vector3f n(Eigen::Vector3f::Zero());
      if (last_frame->_depth.at<float>(h,w)>=0.1)
      {
        const auto &pt = (*last_frame->_cloud)(w,h);
        n << pt.normal_x, pt.normal_y, pt.normal_z;
        n.normalize();
      }
      n.array() = (n.array()+1)/2.0 * 255;
      normal_img.at<cv::Vec3b>(h,w) = {n(2),n(1),n(0)};    // BGR
    }
  }
  cv::imwrite(fmt::format("{}/normal/{}.png",debug_dir,last_frame->_id_str),normal_img);
  /////////// Write images to load by NERF


  std::vector<float> pose_data;
  std::string s;
  for (const auto &f:frames)
  {
    s += f->_id_str+" ";
    const auto &cur_pose = f->_pose_in_model;
    for (int h=0;h<4;h++)
    {
      for (int w=0;w<4;w++)
      {
        pose_data.push_back(cur_pose(h,w));
      }
    }
  }
  zmq::message_t msg(s.size());
  std::memcpy(msg.data(), s.data(), s.size());
  _socket.send(msg, ZMQ_SNDMORE);

  msg.rebuild(pose_data.size()*sizeof(float));
  std::memcpy(msg.data(), pose_data.data(), pose_data.size()*sizeof(float));
  _socket.send(msg, 0);

  if ((*yml)["SPDLOG"].as<int>()>=2)
  {
    std::ofstream ff(fmt::format("{}/{}/poses_before_nerf.txt",debug_dir,last_frame->_id_str));
    for (int i=0;i<pose_data.size();i++)
    {
      ff<<pose_data[i]<<" ";
    }
    ff.close();

    ff.open(fmt::format("{}/{}/nerf_frames.txt",debug_dir,last_frame->_id_str));
    for (const auto &f:frames)
    {
      ff<<f->_id_str<<"\n";
    }
    ff.close();
  }

  SPDLOG("zmq start waiting for reply");
  std::vector<zmq::message_t> recv_msgs;
  zmq::recv_multipart(_socket, std::back_inserter(recv_msgs));
  SPDLOG("zmq got reply");
  std::vector<float> optimized_pose_data(frames.size()*16);
  std::memcpy(optimized_pose_data.data(), recv_msgs[0].data(), optimized_pose_data.size()*sizeof(float));

  for (int i=0;i<frames.size();i++)
  {
    const auto &f = frames[i];
    for (int h=0;h<4;h++)
    {
      for (int w=0;w<4;w++)
      {
        f->_pose_in_model(h,w) = optimized_pose_data[i*16+h*4+w];
      }
    }
    f->_nerfed = true;
  }

  if ((*yml)["SPDLOG"].as<int>()>=2)
  {
    std::ofstream ff(fmt::format("{}/{}/poses_after_nerf.txt",debug_dir,last_frame->_id_str));
    for (int i=0;i<optimized_pose_data.size();i++)
    {
      ff<<optimized_pose_data[i]<<" ";
    }
    ff.close();
  }
  SPDLOG("NERF done");

}
