# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import pickle,glob,cv2,imageio,os,sys,pdb,re,json,trimesh,copy,pdb,logging,multiprocessing,subprocess,joblib
import numpy as np
import ruamel.yaml
yaml = ruamel.yaml.YAML()
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
from Utils import *

HO3D_ROOT = '/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3'


class YcbineoatReader:
  def __init__(self,video_dir, downscale=1, shorter_side=None):
    self.video_dir = video_dir
    self.downscale = downscale
    self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
    self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3,3)
    self.id_strs = []
    for color_file in self.color_files:
      id_str = os.path.basename(color_file).replace('.png','')
      self.id_strs.append(id_str)
    self.H,self.W = cv2.imread(self.color_files[0]).shape[:2]

    if shorter_side is not None:
      self.downscale = shorter_side/min(self.H, self.W)

    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale

    self.gt_pose_files = sorted(glob.glob(f'{self.video_dir}/annotated_poses/*'))

    self.videoname_to_object = {
      'bleach0': "021_bleach_cleanser",
      'bleach_hard_00_03_chaitanya': "021_bleach_cleanser",
      'cracker_box_reorient': '003_cracker_box',
      'cracker_box_yalehand0': '003_cracker_box',
      'mustard0': '006_mustard_bottle',
      'mustard_easy_00_02': '006_mustard_bottle',
      'sugar_box1': '004_sugar_box',
      'sugar_box_yalehand0': '004_sugar_box',
      'tomato_soup_can_yalehand0': '005_tomato_soup_can',
    }


  def get_video_name(self):
    return self.video_dir.split('/')[-1]

  def __len__(self):
    return len(self.color_files)

  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self,i):
    mask = cv2.imread(self.color_files[i].replace('rgb','masks'),-1)
    if len(mask.shape)==3:
      mask = (mask.sum(axis=-1)>0).astype(np.uint8)
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return mask

  def get_depth(self,i):
    depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
    depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return depth


  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.K)
    return xyz_map

  def get_occ_mask(self,i):
    hand_mask_file = self.color_files[i].replace('rgb','masks_hand')
    occ_mask = np.zeros((self.H,self.W), dtype=bool)
    if os.path.exists(hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(hand_mask_file,-1)>0)

    right_hand_mask_file = self.color_files[i].replace('rgb','masks_hand_right')
    if os.path.exists(right_hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(right_hand_mask_file,-1)>0)

    occ_mask = cv2.resize(occ_mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST)

    return occ_mask.astype(np.uint8)

  def get_gt_mesh(self):
    ob_name = self.videoname_to_object[self.get_video_name()]
    mesh = trimesh.load(f'/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/YCB_Video_Models/models/{ob_name}/textured_simple.obj')
    return mesh


class Ho3dReader:
  def __init__(self,video_dir):
    self.video_dir = video_dir
    self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.jpg"))
    meta_file = self.color_files[0].replace('.jpg','.pkl').replace('rgb','meta')
    self.K = pickle.load(open(meta_file,'rb'))['camMat']

    self.id_strs = []
    for i in range(len(self.color_files)):
      id = os.path.basename(self.color_files[i]).split('.')[0]
      self.id_strs.append(id)


  def __len__(self):
    return len(self.color_files)

  def get_video_name(self):
    return os.path.dirname(os.path.abspath(self.color_files[0])).split('/')[-2]

  def get_mask(self,i):
    video_name = self.get_video_name()
    index = int(os.path.basename(self.color_files[i]).split('.')[0])
    mask = cv2.imread(f'{HO3D_ROOT}/masks_XMem/{video_name}/{index:05d}.png',-1)
    return mask


  def get_occ_mask(self,i):
    video_name = self.get_video_name()
    index = int(os.path.basename(self.color_files[i]).split('.')[0])
    mask = cv2.imread(f'{HO3D_ROOT}/masks_XMem/{video_name}_hand/{index:04d}.png',-1)
    return mask


  def get_gt_mesh(self):
    video2name = {
      'AP': '019_pitcher_base',
      'MPM': '010_potted_meat_can',
      'SB': '021_bleach_cleanser',
      'SM': '006_mustard_bottle',
    }
    video_name = self.get_video_name()
    for k in video2name:
      if video_name.startswith(k):
        ob_name = video2name[k]
        break
    mesh = trimesh.load(f'{HO3D_ROOT}/models/{ob_name}/textured_simple.obj')
    return mesh


  def get_depth(self,i):
    color = imageio.imread(self.color_files[i])
    depth_scale = 0.00012498664727900177
    depth = cv2.imread(self.color_files[i].replace('.jpg','.png').replace('rgb','depth'), -1)
    depth = (depth[...,2]+depth[...,1]*256)*depth_scale
    return depth

  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth,self.K)
    return xyz_map


  def get_gt_pose(self,i):
    meta_file = self.color_files[i].replace('.jpg','.pkl').replace('rgb','meta')
    meta = pickle.load(open(meta_file,'rb'))
    ob_in_cam_gt = np.eye(4)
    if meta['objTrans'] is None:
      return None
    else:
      ob_in_cam_gt[:3,3] = meta['objTrans']
      ob_in_cam_gt[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
      ob_in_cam_gt = glcam_in_cvcam@ob_in_cam_gt
    return ob_in_cam_gt