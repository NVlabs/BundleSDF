# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys, time,torch,pickle,trimesh,itertools,pdb,zipfile,datetime,imageio,gzip,logging,importlib
import open3d as o3d
from uuid import uuid4
import cv2
from PIL import Image
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import math,glob,re,copy
from transformations import *
from scipy.spatial import cKDTree
from collections import OrderedDict
import ruamel.yaml
yaml = ruamel.yaml.YAML()
try:
  import kaolin
except Exception as e:
  print(f"Import kaolin failed, {e}")
try:
  from mycuda import common
except:
  pass


BAD_DEPTH = 99
BAD_COLOR = 128

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]])

COLOR_MAP=np.array([[0, 0, 0], #Ignore
                    [128,0,0], #Background
                    [0,128,0], #Wall
                    [128,128,0], #Floor
                    [0,0,128], #Ceiling
                    [128,0,128], #Table
                    [0,128,128], #Chair
                    [128,128,128], #Window
                    [64,0,0], #Door
                    [192,0,0], #Monitor
                    [64, 128, 0],     # 11th
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0], # defined for 18 classes currently
                    ])


def set_logging_format():
  importlib.reload(logging)
  FORMAT = '[%(filename)s] %(message)s'
  logging.basicConfig(level=logging.INFO, format=FORMAT)

set_logging_format()


def set_seed(random_seed):
  import torch,random
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False



def add_err(pred,gt,model_pts):
  """
  Average Distance of Model Points for objects with no indistinguishable views
  - by Hinterstoisser et al. (ACCV 2012).
  """
  pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
  gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
  e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
  return e

def adi_err(pred,gt,model_pts):
  """
  @pred: 4x4 mat
  @gt:
  @model: (N,3)
  """
  pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
  gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
  nn_index = cKDTree(pred_pts)
  nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
  e = nn_dists.mean()
  return e


class Iou3d:
  def __init__(self,model_pts):
    self.max_xyz = model_pts.max(axis=0)
    self.min_xyz = model_pts.min(axis=0)
    self.bbox = np.array([self.min_xyz,self.max_xyz]).reshape(2,3)
    resolution = np.linalg.norm(self.max_xyz-self.min_xyz)/100
    xx,yy,zz = np.meshgrid(np.arange(self.min_xyz[0],self.max_xyz[0],resolution), np.arange(self.min_xyz[1],self.max_xyz[1],resolution), np.arange(self.min_xyz[2],self.max_xyz[2],resolution))
    self.pts = np.stack((xx,yy,zz), axis=-1).reshape(-1,3)

  def compute(self,pred,gt):
    pred_in_gt = np.linalg.inv(gt)@pred
    pts = (pred_in_gt@to_homo(self.pts).T).T[:,:3]
    inside_mask = (pts[:,0]<=self.max_xyz[0]) & (pts[:,0]>=self.min_xyz[0]) & (pts[:,1]<=self.max_xyz[1]) & (pts[:,1]>=self.min_xyz[1]) & (pts[:,2]<=self.max_xyz[2]) & (pts[:,2]>=self.min_xyz[2])
    return inside_mask.sum()/len(pts)


def compute_3d_iou_new(RT_1, RT_2, noc_cube_1, noc_cube_2):
  '''Computes IoU overlaps between two 3d bboxes.
      bbox_3d_1, bbox_3d_1: [3, 8]
  '''
  def transform_coordinates_3d(coordinates, RT):
    '''
    @coordinates: (3,N)
    '''
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

  def asymmetric_3d_iou(RT_1, RT_2, noc_cube_1, noc_cube_2):
    bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)
    bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

    bbox_1_max = np.amax(bbox_3d_1, axis=0)
    bbox_1_min = np.amin(bbox_3d_1, axis=0)
    bbox_2_max = np.amax(bbox_3d_2, axis=0)
    bbox_2_min = np.amin(bbox_3d_2, axis=0)

    overlap_min = np.maximum(bbox_1_min, bbox_2_min)
    overlap_max = np.minimum(bbox_1_max, bbox_2_max)

    if np.amin(overlap_max - overlap_min) <0:
      intersections = 0
    else:
      intersections = np.prod(overlap_max - overlap_min)
    union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
    overlaps = intersections / union
    return overlaps

  if 0:
    def y_rotation_matrix(theta):
        return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                          [0, 1, 0 , 0],
                          [-np.sin(theta), 0, np.cos(theta), 0],
                          [0, 0, 0 , 1]])

    n = 20
    max_iou = 0
    for i in range(n):
        rotated_RT_1 = RT_1@y_rotation_matrix(2*math.pi*i/float(n))
        max_iou = max(max_iou, asymmetric_3d_iou(rotated_RT_1, RT_2, noc_cube_1, noc_cube_2))
  else:
    max_iou = asymmetric_3d_iou(RT_1, RT_2, noc_cube_1, noc_cube_2)

  return max_iou



def compute_auc(rec, max_val=0.1):
  '''https://github.com/wenbowen123/iros20-6d-pose-tracking/blob/2df96b720e8e499b9f0d5fcebfbae2bcfa51ab19/eval_ycb.py#L45
  '''
  if len(rec)==0:
    return 0
  rec = np.sort(np.array(rec))
  n = len(rec)
  prec = np.arange(1,n+1) / float(n)
  rec = rec.reshape(-1)
  prec = prec.reshape(-1)
  index = np.where(rec<max_val)[0]
  rec = rec[index]
  prec = prec[index]

  mrec=[0, *list(rec), max_val]
  mpre=[0, *list(prec), prec[-1]]

  for i in range(1,len(mpre)):
    mpre[i] = max(mpre[i], mpre[i-1])
  mpre = np.array(mpre)
  mrec = np.array(mrec)
  i = np.where(mrec[1:]!=mrec[0:len(mrec)-1])[0] + 1
  ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) / max_val
  return ap


def geodesic_distance(R1,R2):
  cos = (np.trace(R1.dot(R2.T))-1)/2
  cos = np.clip(cos,-1,1)
  return math.acos(cos)


def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud


def depth2xyzmap(depth, K):
  invalid_mask = (depth<0.1)
  H,W = depth.shape[:2]
  vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
  vs = vs.reshape(-1)
  us = us.reshape(-1)
  zs = depth.reshape(-1)
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = pts.reshape(H,W,3).astype(np.float32)
  xyz_map[invalid_mask] = 0
  return xyz_map.astype(np.float32)



def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo


def to_homo_torch(pts):
  '''
  @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
  '''
  ones = torch.ones((*pts.shape[:-1],1)).to(pts.device).float()
  homo = torch.cat((pts, ones),dim=-1)
  return homo


def transform_pts(pts,tf):
  """Transform 2d or 3d points
  @pts: (...,3)
  """
  return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]


def sph2cart(phi, theta, r):
    point_on_sphere = np.zeros(3)
    point_on_sphere[0] = r * math.sin(phi) * math.cos(theta)
    point_on_sphere[1] = r * math.sin(phi) * math.sin(theta)
    point_on_sphere[2] = r * math.cos(phi)
    return point_on_sphere


def chamfer_distance_between_clouds_mutual(pts1,pts2):
  kdtree1 = cKDTree(pts1)
  dists1, indices1 = kdtree1.query(pts2)
  kdtree2 = cKDTree(pts2)
  dists2, indices2 = kdtree2.query(pts1)
  return 0.5*(dists1.mean()+dists2.mean())   #!NOTE should not be mean of all, see https://pdal.io/en/stable/apps/chamfer.html




def trimesh_clean(mesh):
  mesh.merge_vertices()
  mesh.remove_degenerate_faces()
  mesh.remove_duplicate_faces()
  mesh.remove_infinite_values()
  mesh.remove_unreferenced_vertices()
  return mesh


def trimesh_split(mesh, min_edge=1000):
  '''!NOTE mesh.split takes too much memory for large mesh. That's why we have this function
  '''
  components = trimesh.graph.connected_components(mesh.edges, min_len=min_edge, nodes=None, engine=None)
  meshes = []
  for i,c in enumerate(components):
    mask = np.zeros(len(mesh.vertices),dtype=bool)
    mask[c] = 1
    cur_mesh = mesh.copy()
    cur_mesh.update_vertices(mask=mask.astype(bool))
    meshes.append(cur_mesh)
  return meshes


def project_3d_to_2d(pt,K,ob_in_cam):
  pt = pt.reshape(4,1)
  projected = K @ ((ob_in_cam@pt)[:3,:])
  projected = projected.reshape(-1)
  projected = projected/projected[2]
  return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(color, ob_in_cam, scale=0.1, K=np.eye(3), thickness=3, transparency=0.3,is_input_rgb=False):
  '''
  @color: BGR
  '''
  if is_input_rgb:
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
  xx = np.array([1,0,0,1]).astype(float)
  yy = np.array([0,1,0,1]).astype(float)
  zz = np.array([0,0,1,1]).astype(float)
  xx[:3] = xx[:3]*scale
  yy[:3] = yy[:3]*scale
  zz[:3] = zz[:3]*scale
  origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
  xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
  yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
  zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
  line_type = cv2.FILLED
  arrow_len = 0
  tmp = color.copy()
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp = tmp.astype(np.uint8)
  if is_input_rgb:
    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

  return tmp



def mesh_to_usd(mesh, out_file='/home/bowen/debug/1.usd'):
  '''Export mesh to USD file for Omniverse
  @mesh: trimesh
  '''
  import kaolin
  materials_order = torch.zeros((len(mesh.faces), 2)).long()
  tex_img = np.array(mesh.visual.material.image)[...,:3]/255.0
  mat = kaolin.io.materials.PBRMaterial(diffuse_texture=torch.tensor(tex_img).permute(2,0,1))
  stage = kaolin.io.usd.export_mesh(out_file, vertices=torch.tensor(mesh.vertices), faces=torch.tensor(mesh.faces).long(), uvs=torch.tensor(mesh.visual.uv), face_normals=torch.tensor(mesh.face_normals), materials_order=materials_order, materials=[mat], up_axis='Y')


class OctreeManager:
    def __init__(self,pts=None,max_level=None,octree=None):
        if octree is None:
            pts_quantized = kaolin.ops.spc.quantize_points(pts.contiguous(), level=max_level)
            self.octree = kaolin.ops.spc.unbatched_points_to_octree(pts_quantized, max_level, sorted=False)
        else:
            self.octree = octree
        lengths = torch.tensor([len(self.octree)], dtype=torch.int32).cpu()
        self.max_level, self.pyramids, self.exsum = kaolin.ops.spc.scan_octrees(self.octree,lengths)
        self.n_level = self.max_level+1
        self.point_hierarchies = kaolin.ops.spc.generate_points(self.octree, self.pyramids, self.exsum)
        self.point_hierarchy_dual, self.pyramid_dual = kaolin.ops.spc.unbatched_make_dual(self.point_hierarchies, self.pyramids[0])
        self.trinkets, self.pointers_to_parent = kaolin.ops.spc.unbatched_make_trinkets(self.point_hierarchies, self.pyramids[0], self.point_hierarchy_dual, self.pyramid_dual)
        self.n_vox = len(self.point_hierarchies)
        self.n_corners = len(self.point_hierarchy_dual)

    def get_level_corner_quantized_points(self,level):
        start = self.pyramid_dual[...,1,level]
        num = self.pyramid_dual[...,0,level]
        return self.point_hierarchy_dual[start:start+num]

    def get_level_quantized_points(self,level):
        start = self.pyramids[...,1,level]
        num = self.pyramids[...,0,level]
        return self.pyramids[start:start+num]


    def get_trilinear_coeffs(self,x,level):
        quantized = kaolin.ops.spc.quantize_points(x, level)
        coeffs = kaolin.ops.spc.coords_to_trilinear_coeffs(x,quantized,level)   #(N,8)
        return coeffs


    def get_center_ids(self,x,level):
        pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        return pidx


    def get_corners_ids(self,x,level):
        pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        corner_ids = self.trinkets[pidx]
        is_valid = torch.ones(len(x)).bool().to(x.device)
        bad_ids = (pidx<0).nonzero()[:,0]
        is_valid[bad_ids] = 0

        return corner_ids, is_valid


    def trilinear_interpolate(self,x,level,feat):
        '''
        @feat: (N_feature of current level, D)
        '''
        ############!NOTE direct API call cannot back prop well
        # pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        # x = x.unsqueeze(0)
        # interpolated = kaolin.ops.spc.unbatched_interpolate_trilinear(coords=x,pidx=pidx.int(),point_hierarchy=self.point_hierarchies,trinkets=self.trinkets, feats=feat, level=level)[0]
        ##################

        coeffs = self.get_trilinear_coeffs(x,level)  #(N,8)
        corner_ids,is_valid = self.get_corners_ids(x,level)
        # if corner_ids.max()>=feat.shape[0]:
        #     pdb.set_trace()

        corner_feat = feat[corner_ids[is_valid].long()]   #(N,8,D)
        out = torch.zeros((len(x),feat.shape[-1]),device=x.device).float()
        out[is_valid] = torch.sum(coeffs[...,None][is_valid]*corner_feat, dim=1)   #(N,D)

        # corner_feat = feat[corner_ids.long()]   #(N,8,D)
        # out = torch.sum(coeffs[...,None]*corner_feat, dim=1)   #(N,D)

        return out,is_valid

    def draw_boxes(self,level,outfile='/home/bowen/debug/corners.ply'):
        centers = kaolin.ops.spc.unbatched_get_level_points(self.point_hierarchies.reshape(-1,3), self.pyramids.reshape(2,-1), level)
        pts = (centers.float()+0.5)/(2**level)*2-1   #Normalize to [-1,1]
        pcd = toOpen3dCloud(pts.data.cpu().numpy())
        o3d.io.write_point_cloud(outfile.replace("corners","centers"),pcd)

        corners = kaolin.ops.spc.unbatched_get_level_points(self.point_hierarchy_dual, self.pyramid_dual, level)
        pts = corners.float()/(2**level)*2-1   #Normalize to [-1,1]
        pcd = toOpen3dCloud(pts.data.cpu().numpy())
        o3d.io.write_point_cloud(outfile,pcd)


    def ray_trace(self,rays_o,rays_d,level,debug=False):
        """Octree is in normalized [-1,1] world coordinate frame
        'rays_o': ray origin in normalized world coordinate system
        'rays_d': (N,3) unit length ray direction in normalized world coordinate system
        'octree': spc
        @voxel_size: in the scale of [-1,1] space
        Return:
            ray_depths_in_out: traveling times, NOT the Z value
        """
        from mycuda import common

        # Avoid corner cases. issuse in kaolin: https://github.com/NVIDIAGameWorks/kaolin/issues/490 and https://github.com/NVIDIAGameWorks/kaolin/pull/634
        # rays_o = rays_o.clone() + 1e-7

        ray_index, rays_pid, depth_in_out = kaolin.render.spc.unbatched_raytrace(self.octree,self.point_hierarchies,self.pyramids[0],self.exsum,rays_o,rays_d,level=level,return_depth=True,with_exit=True)
        if ray_index.size()[0] == 0:
            print("[WARNING] batch has 0 intersections!!")
            ray_depths_in_out = torch.zeros((rays_o.shape[0],1,2))
            rays_pid = -torch.ones_like(rays_o[:, :1])
            rays_near = torch.zeros_like(rays_o[:, :1])
            rays_far = torch.zeros_like(rays_o[:, :1])
            return rays_near, rays_far, rays_pid, ray_depths_in_out

        intersected_ray_ids,counts = torch.unique_consecutive(ray_index,return_counts=True)
        max_intersections = counts.max().item()
        start_poss = torch.cat([torch.tensor([0], device=counts.device),torch.cumsum(counts[:-1],dim=0)],dim=0)

        ray_depths_in_out = common.postprocessOctreeRayTracing(ray_index.long().contiguous(),depth_in_out.contiguous(),intersected_ray_ids.long().contiguous(),start_poss.long().contiguous(), max_intersections, rays_o.shape[0])

        rays_far = ray_depths_in_out[:,:,1].max(dim=-1)[0].reshape(-1,1)
        rays_near = ray_depths_in_out[:,0,0].reshape(-1,1)

        return rays_near, rays_far, rays_pid, ray_depths_in_out



def get_optimized_poses_in_real_world(poses_normalized, pose_array, sc_factor, translation):
    '''
    @poses_normalized: np array, cam_in_ob (opengl convention), normalized to [-1,1] and centered
    @pose_array: PoseArray, delta poses
    Return:
        cam_in_ob, real-world unit, opencv convention
    '''
    original_poses = poses_normalized.copy()
    original_poses[:, :3, 3] /= sc_factor   # To true world scale
    original_poses[:, :3, 3] -= translation

    # Apply pose transformation
    tf = pose_array.get_matrices(np.arange(len(poses_normalized))).reshape(-1,4,4).data.cpu().numpy()
    optimized_poses = tf@poses_normalized

    optimized_poses = np.array(optimized_poses).astype(np.float32)
    optimized_poses[:, :3, 3] /= sc_factor
    optimized_poses[:, :3, 3] -= translation

    original_init_ob_in_cam = optimized_poses[0].copy()
    offset = np.linalg.inv(original_init_ob_in_cam)@original_poses[0]  # Anchor to the first frame whose pose shouldn't change
    for i in range(len(optimized_poses)):
      new_ob_in_cam = optimized_poses[i]@offset
      optimized_poses[i] = new_ob_in_cam
      optimized_poses[i] = optimized_poses[i]@glcam_in_cvcam

    return optimized_poses,offset


def mesh_to_real_world(mesh,pose_offset,translation,sc_factor):
    '''
    @pose_offset: optimized delta pose of the first frame. Usually it's identity
    '''
    mesh.vertices = mesh.vertices/sc_factor - np.array(translation).reshape(1,3)
    mesh.apply_transform(pose_offset)
    return mesh


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0,255,0), linewidth=2):
  '''
  @bbox: (2,3) min/max
  @line_color: RGB
  '''
  min_xyz = bbox.min(axis=0)
  xmin, ymin, zmin = min_xyz
  max_xyz = bbox.max(axis=0)
  xmax, ymax, zmax = max_xyz

  def draw_line3d(start,end,img):
    pts = np.stack((start,end),axis=0).reshape(-1,3)
    pts = (ob_in_cam@to_homo(pts).T).T[:,:3]   #(2,3)
    projected = (K@pts.T).T
    uv = np.round(projected[:,:2]/projected[:,2].reshape(-1,1)).astype(int)   #(2,2)
    img = cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth)
    return img

  for y in [ymin,ymax]:
    for z in [zmin,zmax]:
      start = np.array([xmin,y,z])
      end = start+np.array([xmax-xmin,0,0])
      img = draw_line3d(start,end,img)

  for x in [xmin,xmax]:
    for z in [zmin,zmax]:
      start = np.array([x,ymin,z])
      end = start+np.array([0,ymax-ymin,0])
      img = draw_line3d(start,end,img)

  for x in [xmin,xmax]:
    for y in [ymin,ymax]:
      start = np.array([x,y,zmin])
      end = start+np.array([0,0,zmax-zmin])
      img = draw_line3d(start,end,img)

  return img
