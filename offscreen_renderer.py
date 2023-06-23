# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys,time
os.environ["PYOPENGL_PLATFORM"] = "egl"
code_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_path)
import open3d as o3d
import numpy as np
from PIL import Image
import cv2,imageio
import time
import trimesh
import pyrender
from Utils import *
from transformations import *
import numpy as np
from PIL import Image
import cv2
import time
import argparse,pickle


cvcam_in_glcam = np.array([[1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1]])

class ModelRendererOffscreen:
  '''https://colab.research.google.com/drive/1Z71mHIc-Sqval92nK290vAsHZRUkCjUx#scrollTo=fIymQapsuWGn
  If off-screen mode, set os.environ["PYOPENGL_PLATFORM"] at very top of the code
  '''
  def __init__(self,model_paths, cam_K, H,W, zfar=2):
    '''
    @window_sizes: H,W
    '''
    self.K = cam_K
    self.scene = pyrender.Scene(ambient_light=[1., 1., 1.],bg_color=[0,0,0])
    self.camera = pyrender.IntrinsicsCamera(fx=cam_K[0,0],fy=cam_K[1,1],cx=cam_K[0,2],cy=cam_K[1,2],znear=0.1,zfar=zfar)
    self.cam_node = self.scene.add(self.camera, pose=np.eye(4))
    self.mesh_nodes = []

    for model_path in model_paths:
      print('model_path',model_path)
      obj_mesh = trimesh.load(model_path)
      mesh = pyrender.Mesh.from_trimesh(obj_mesh)
      mesh_node = self.scene.add(mesh,pose=np.eye(4),parent_node=self.cam_node) # Object pose parent is cam
      self.mesh_nodes.append(mesh_node)

    self.H = H
    self.W = W

    self.r = pyrender.OffscreenRenderer(self.W, self.H)  #!NOTE version>0.1.32 not work https://github.com/mmatl/pyrender/issues/85
    self.glcam_in_cvcam = np.array([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
    self.cvcam_in_glcam = np.linalg.inv(self.glcam_in_cvcam)


  def clear_meshes(self):
    for n in self.mesh_nodes:
      self.scene.remove_node(n)
    self.mesh_nodes = []


  def add_mesh(self, mesh):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_node = self.scene.add(mesh,pose=np.eye(4),parent_node=self.cam_node) # Object pose parent is cam
    self.mesh_nodes.append(mesh_node)


  def add_point_light(self):
    # light = pyrender.SpotLight(color=[0.5,0.5,0.5], intensity=1.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    self.scene.add(light, pose=np.eye(4))  # Same as camera position


  def render(self,ob_in_cvcams):
    for i,ob_in_cvcam in enumerate(ob_in_cvcams):
      ob_in_glcam = self.cvcam_in_glcam.dot(ob_in_cvcam)
      self.scene.set_pose(self.mesh_nodes[i],ob_in_glcam)
    color, depth = self.r.render(self.scene)  # depth: float
    return color, depth

  def renderBatch(self,ob_in_cvcams):
    colors = []
    depths = []
    for ob_in_cvcam in ob_in_cvcams:
      color, depth = self.render(ob_in_cvcam)
      colors.append(color)
      depths.append(depth)
    colors = np.array(colors)
    depths = np.array(depths)
    return colors, depths


class TinyRenderer:
  def __init__(self, H, W, K):
    import pytinyrenderer
    self.H = H
    self.W = W
    self.K = K.copy()
    self.m = None
    self.scene = pytinyrenderer.TinySceneRenderer()
    self.near = 0.01
    self.far = 10
    self.projection_mat = np.array([[2*self.K[0,0]/self.W, -2*self.K[0,1]/self.W, (self.W - 2*self.K[0,2])/self.W, 0],
                                    [0, -2*self.K[1,1]/self.H, (self.H - 2*self.K[1,2])/self.H,0],
                                    [0,0, (-self.far - self.near)/(self.far - self.near), -2*self.far*self.near/(self.far - self.near)],
                                    [0,0,-1,0]]).reshape(4,4)
    self.projection_mat = self.projection_mat.T


  def render(self, ob_in_cam):
    '''
    @ob_in_cam: in opencv camera
    '''
    import pytinyrenderer
    ob_in_glcam = cvcam_in_glcam@ob_in_cam
    viewMatrix = ob_in_glcam.T  # Col-wise
    camera = pytinyrenderer.TinyRenderCamera(viewWidth=self.W, viewHeight=self.H, viewMatrix=viewMatrix.reshape(-1).tolist(), projectionMatrix=self.projection_mat.reshape(-1).tolist())
    self.light = pytinyrenderer.TinyRenderLight(direction=-np.linalg.inv(ob_in_glcam)[:3,2],specular=0.2, ambient=0.8, diffuse=0, distance=5)
    img = self.scene.get_camera_image([self.ob_instance], self.light, camera)
    color = np.array(img.rgb,dtype=np.uint8).reshape(img.height, img.width, -1)[::-1]   # Up-side down
    mask = np.array(img.segmentation_mask,dtype=np.uint8).reshape(img.height, img.width, -1)[::-1]
    mask = mask.sum(axis=-1)>0
    # zbuffer = np.array(img.depthbuffer,dtype=np.float32)
    # depth = (2 * self.near * self.far) / (self.far + self.near - (2 * depth - 1) * (self.far - self.near))
    # z_c = -zbuffer
    # depth = self.far * (self.near + z_c) / (2. * self.far * self.near + self.far * z_c - self.near * z_c)
    # depth = depth.reshape(self.H, self.W)
    # depth = np.zeros((1,1))
    return color, mask


  def add_mesh(self, mesh, color=[200,200,200]):
    texture_img = np.array(color).reshape(1,1,3).astype(np.uint8)
    uvs = np.zeros((len(mesh.vertices),2), dtype=np.float32)
    texture_scaling = 1
    self.m = self.scene.create_mesh(mesh.vertices.reshape(-1).tolist(), mesh.vertex_normals.flatten().tolist(), uvs.flatten().tolist(), mesh.faces.reshape(-1).tolist(), texture_img.reshape(-1).tolist(), texture_img.shape[1], texture_img.shape[0], texture_scaling)
    self.ob_instance = self.scene.create_object_instance(self.m)
    self.scene.set_object_position(self.ob_instance, [0,0,0])
    self.scene.set_object_orientation(self.ob_instance, [0,0,0,1])


  def clear_meshes(self):
    if self.m is not None:
      self.scene.delete_mesh(self.m)
      self.scene.delete_instance(self.ob_instance)
