/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "FeatureManager.h"
#include "Frame.h"
#include "Utils.h"
#include "Bundler.h"



PYBIND11_MODULE(my_cpp, m)
{
  py::class_<YAML::Node, std::shared_ptr<YAML::Node>>(m, "YamlNode")
    .def(py::init<>())
    .def("size", &YAML::Node::size)
    .def("__getitem__", [](const YAML::Node &node, const std::string& key)
      {
        return node[key];
      })
    .def("Scalar", &YAML::Node::Scalar)
    ;

  m.def("computeCovisibility", &computeCovisibility);
  m.def("YamlLoadFile", &YAML::LoadFile);
  m.def("YamlDump", &YAML::Dump);
  m.def("numpy", [](const std::vector<Correspondence> &matches)
    {
      std::vector<std::vector<float>> out(matches.size());
      for (int i=0;i<matches.size();i++)
      {
        const auto &m = matches[i];
        out[i] = {m._uA, m._vA, m._uB, m._vB};
      }
      return out;
    });

  py::class_<cv::Mat>(m, "cvMat", py::buffer_protocol())
    .def_buffer([](cv::Mat &in)->py::buffer_info
      {
        if (!in.isContinuous())
        {
          in = in.clone();
        }
        const int H = in.rows;
        const int W = in.cols;
        int n_channel = 1;
        std::string type = Utils::type2str(in.type());
        if (type.find("C")!=-1)
        {
          if (type.find("C3")!=-1)
          {
            n_channel = 3;
          }
        }

        if (type.find("U")!=-1)
        {
          return py::buffer_info(in.data, sizeof(uchar), py::format_descriptor<uchar>::format(), 3, {H,W,n_channel}, {sizeof(uchar)*W*n_channel, sizeof(uchar)*n_channel, sizeof(uchar)});
        }
        if (type.find("F")!=-1)
        {
          return py::buffer_info(in.data, sizeof(float), py::format_descriptor<float>::format(), 3, {H,W,n_channel}, {sizeof(float)*W*n_channel, sizeof(float)*n_channel, sizeof(float)});
        }
      })
    .def(py::init([](py::buffer b)->cv::Mat
      {
        py::buffer_info info = b.request();
        std::string type;
        if (info.format == py::format_descriptor<float>::format())
        {
          type = "CV_32FC1";
          if (info.ndim==3 && info.shape[2]==3)
          {
            type = "CV_32FC3";
          }
        }
        else if (info.format == py::format_descriptor<uchar>::format())
        {
          type = "CV_8UC1";
          if (info.ndim==3 && info.shape[2]==3)
          {
            type = "CV_8UC3";
          }
        }

        cv::Mat out = cv::Mat(info.shape[0], info.shape[1], Utils::str2type(type), info.ptr).clone();

        return out;
      }))
    ;

  py::class_<Correspondence>(m, "Correspondence", py::buffer_protocol())
    .def("numpy", [](Correspondence &in)
      {
        std::vector<float> tmp = {in._uA, in._vA, in._uB, in._vB};
        return tmp;
      })
    .def_readwrite("_uA", &Correspondence::_uA)
    .def_readwrite("_vA", &Correspondence::_vA)
    .def_readwrite("_uB", &Correspondence::_uB)
    .def_readwrite("_vB", &Correspondence::_vB)
    .def_readwrite("_confidence", &Correspondence::_confidence)
    ;

  py::class_<SiftManager, std::shared_ptr<SiftManager>>(m, "SiftManager")
    .def_readwrite("_matches", &SiftManager::_matches)
    ;

  py::class_<GluNet, std::shared_ptr<GluNet>>(m, "GluNet")
    .def("findCorresbyNNBatch", &GluNet::findCorresbyNNBatch)
    .def("findCorres", &GluNet::findCorres)
    .def("vizKeyPoints", &GluNet::vizKeyPoints)
    .def("detectFeature", &GluNet::detectFeature)
    .def("getProcessedImagePairs", &GluNet::getProcessedImagePairs)
    .def("rawMatchesToCorres", &GluNet::rawMatchesToCorres)
    .def("runRansacMultiPairGPU", &SiftManager::runRansacMultiPairGPU)
    .def("procrustesByCorrespondence", &SiftManager::procrustesByCorrespondence)
    .def("vizCorresBetween", &SiftManager::vizCorresBetween)
    .def_readwrite("_matches", &GluNet::_matches)
    .def_readwrite("_raw_matches", &GluNet::_raw_matches)
    ;

  py::class_<Frame, std::shared_ptr<Frame>> frame(m, "Frame");
  frame.def(py::init<const py::array_t<uchar> &, const py::array_t<float> &, const Eigen::Vector4f &, const Eigen::Matrix4f &, int, std::string, const Eigen::Matrix3f &, std::shared_ptr<YAML::Node> >(), py::call_guard<py::gil_scoped_release>())
    .def("invalidatePixelsByMask", &Frame::invalidatePixelsByMask, py::call_guard<py::gil_scoped_release>())
    .def("setNewInitCoordinate", &Frame::setNewInitCoordinate, py::call_guard<py::gil_scoped_release>())
    .def("depthToCloudAndNormals", &Frame::depthToCloudAndNormals, py::call_guard<py::gil_scoped_release>())
    .def("updateDepthGPU", &Frame::updateDepthGPU, py::call_guard<py::gil_scoped_release>())
    .def("pointCloudDenoise", &Frame::pointCloudDenoise, py::call_guard<py::gil_scoped_release>())
    .def("countValidPoints", &Frame::countValidPoints, py::call_guard<py::gil_scoped_release>())
    .def_readwrite("_depth", &Frame::_depth)
    .def_readwrite("_color", &Frame::_color)
    .def_readwrite("_H", &Frame::_H)
    .def_readwrite("_W", &Frame::_W)
    .def_readwrite("_id", &Frame::_id)
    .def_readwrite("_id_str", &Frame::_id_str)
    .def_readwrite("_roi", &Frame::_roi)
    .def_readwrite("_fg_mask", &Frame::_fg_mask)
    .def_readwrite("_occ_mask", &Frame::_occ_mask)
    .def_readwrite("_normal_map", &Frame::_normal_map)
    .def_readwrite("_status", &Frame::_status)
    .def_readwrite("_ref_frame_id", &Frame::_ref_frame_id)
    .def_readwrite("_pose_in_model", &Frame::_pose_in_model)
    .def_readwrite("_nerfed", &Frame::_nerfed)
    .def("pointcloud", [](Frame &in)
      {
        Eigen::MatrixXf out = in._cloud->getMatrixXfMap().transpose();  //(N,D)
        return out;
      })
    ;

  py::enum_<Frame::Status>(frame, "Status")
    .value("FAIL", Frame::Status::FAIL)
    .value("NO_BA", Frame::Status::NO_BA)
    .value("OTHER", Frame::Status::OTHER)
    .export_values();

  py::class_<Bundler, std::shared_ptr<Bundler>>(m, "Bundler")
    .def(py::init<>())
    .def(py::init<std::shared_ptr<YAML::Node>>())
    .def(py::pickle(
      [](const Bundler &p) { // __getstate__
        return true;
      },
      [](bool a) { // __setstate__
        Bundler p;
        return p;
      }))
    .def("forgetFrame", &Bundler::forgetFrame, py::call_guard<py::gil_scoped_release>())
    .def("checkAndAddKeyframe", &Bundler::checkAndAddKeyframe, py::call_guard<py::gil_scoped_release>())
    .def("optimizeGPU", &Bundler::optimizeGPU, py::call_guard<py::gil_scoped_release>())
    .def("selectKeyFramesForBA", &Bundler::selectKeyFramesForBA, py::call_guard<py::gil_scoped_release>())
    .def("saveNewframeResult", &Bundler::saveNewframeResult, py::call_guard<py::gil_scoped_release>())
    .def("saveFramesData", &Bundler::saveFramesData, py::call_guard<py::gil_scoped_release>())
    .def("processNewFrame", &Bundler::processNewFrame, py::call_guard<py::gil_scoped_release>())
    .def("getFeatureMatchPairs", &Bundler::getFeatureMatchPairs, py::call_guard<py::gil_scoped_release>())
    .def_readwrite("yml", &Bundler::yml)
    .def_readwrite("_frames", &Bundler::_frames)
    .def_readwrite("_newframe", &Bundler::_newframe)
    .def_readwrite("_keyframes", &Bundler::_keyframes)
    .def_readwrite("_local_frames", &Bundler::_local_frames)
    .def_readwrite("_fm", &Bundler::_fm)
    .def_readwrite("_firstframe", &Bundler::_firstframe)
    ;

  py::bind_map<std::map<int, std::shared_ptr<Frame>>>(m, "MapFrame");
  py::bind_map<std::map<FramePair, std::vector<Correspondence>>>(m, "MapCorrespondences");
  py::bind_map<std::map<FramePair, Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>>>(m, "MapMatrixUint16");


}

