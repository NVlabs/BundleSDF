# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import dearpygui.dearpygui as dpg


class BundleSdfGui:
  def __init__(self, img_height=300):
    dpg.create_context()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    dpg.create_viewport(large_icon=f'{code_dir}/dpg.jpg')
    dpg.setup_dearpygui()
    self.H = int(img_height)
    self.W = None
    self.K = None
    self.mesh = None
    self.renderer = None
    self.ob_in_cam = None
    self.ob_in_cam_view = None
    self.mesh_min_xyz = None
    self.mesh_max_xyz = None

    with dpg.window(label="",tag="main"):
      with dpg.group(horizontal=True,tag='buttons'):
        dpg.add_button(label="clean_mesh", tag="clean_mesh", callback=self.clean_mesh)
        dpg.add_file_dialog(directory_selector=True, show=False, callback=self.export_mesh, tag="export_mesh_file_dialog", label='export mesh', default_filename='1.obj', height=600, width=1000)
        dpg.add_button(label="export_mesh", tag="export_mesh", callback=lambda: dpg.show_item("export_mesh_file_dialog"))

        def reset_mesh_view():
          self.ob_in_cam_view = self.ob_in_cam
          self.update_render_mesh()

        dpg.add_button(label="reset_mesh_view", callback=reset_mesh_view)

      with dpg.handler_registry():
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self.drag_rotate_pose)
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=self.drag_move_pose)

    dpg.add_group(horizontal=True,tag='row0',parent='main')
    dpg.add_group(horizontal=True,tag='row1',parent='main')
    dpg.add_group(horizontal=True,tag='row2',parent='main')
    dpg.add_group(horizontal=True,tag='row3',parent='main')
    dpg.add_text("frame: 0",tag='frame_id',color=[0,255,0], parent='row3')
    dpg.add_text("keyframe_num: 0",tag='keyframe_num',color=[0,255,0], parent='row3')
    dpg.add_text("nerf_num_frames: X",tag='nerf_num_frames',color=[0,255,0], parent='row3')

    dpg.set_primary_window("main", True)
    dpg.set_viewport_title('BundleSDF')
    dpg.show_viewport()


  def clean_mesh(self):
    try:
      ms = trimesh_split(self.mesh)
      best_size = 0
      best = None
      for m in ms:
        if m.vertices.shape[0]>best_size:
          best_size = m.vertices.shape[0]
          best = m
      self.mesh = trimesh_clean(best)
    except Exception as e:
      logging.info(e)


  def drag_rotate_pose(self, sender, app_data):
    if self.ob_in_cam_view is not None and self.mesh is not None:
      dx = app_data[1]
      dy = app_data[2]
      speed = 0.1
      rx = dy/180.0*np.pi*speed
      ry = -dx/180.0*np.pi*speed
      pts = np.stack([self.mesh_min_xyz, self.mesh_max_xyz], axis=0).reshape(2,3)
      pts = transform_pts(pts, self.ob_in_cam_view)
      center = (pts.max(axis=0) + pts.min(axis=0))/2
      tf = np.eye(4)
      tf[:3,3] = -center
      tf_ = euler_matrix(rx, ry, 0)
      tf = tf_@tf
      tf_  = np.eye(4)
      tf_[:3,3] = center
      tf = tf_@tf
      self.ob_in_cam_view = tf@self.ob_in_cam_view

      self.update_render_mesh()


  def drag_move_pose(self, sender, app_data):
    if self.ob_in_cam_view is not None and self.mesh is not None:
      dx = app_data[1]
      dy = app_data[2]
      speed = 1/self.K[0,0]*self.ob_in_cam_view[2,3]*0.1
      dx = dx*speed
      dy = dy*speed
      tf  = np.eye(4)
      tf[:2,3] = [dx, dy]
      self.ob_in_cam_view = tf@self.ob_in_cam_view

      self.update_render_mesh()


  def update_render_mesh(self):
    if self.ob_in_cam_view is not None and self.mesh is not None:
      color, mask = self.renderer.render(self.ob_in_cam_view)
      # color, depth = self.renderer.render(ob_in_cam=ob_in_cam)
      rgba = np.concatenate((color, np.zeros((self.H,self.W,1))), axis=-1)
      rgba[mask>0,...,3] = 255
      dpg.set_value(f"mesh_render", rgba.reshape(-1)/255.0)


  def export_mesh(self, sender, app_data):
    file = app_data['file_path_name']
    self.mesh.export(file)
    logging.info(f"exported to: {file}")


  def update_frame(self, rgb, mask, ob_in_cam, id_str, K, n_keyframe):
    if self.K is None:
      self.K = K.copy()

    if self.W is None:
      scale = 1/rgb.shape[0]*self.H
      self.W = int(rgb.shape[1]*scale)
      self.K[:2] *= scale

    self.ob_in_cam = ob_in_cam
    if self.ob_in_cam_view is None:
      self.ob_in_cam_view = self.ob_in_cam.copy()
    rgb = cv2.resize(rgb, dsize=(self.W,self.H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, dsize=(self.W,self.H), interpolation=cv2.INTER_NEAREST)
    vis = draw_xyz_axis(rgb[...,::-1], ob_in_cam=ob_in_cam, scale=0.1, K=self.K, transparency=0, thickness=5)
    vis = vis[...,::-1]
    rgba = np.concatenate((vis, np.ones((self.H,self.W,1))*255), axis=-1)
    masked_rgba = np.concatenate((rgb, np.ones((self.H,self.W,1))*255), axis=-1)
    masked_rgba[mask==0,...,:3] = 0

    if dpg.get_value("rgb") is None:
      with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(self.W, self.H, rgba.reshape(-1)/255.0, tag="rgb_init")
        dpg.add_dynamic_texture(self.W, self.H, masked_rgba.reshape(-1)/255.0, tag="masked_rgb_init")
        dpg.add_dynamic_texture(self.W, self.H, rgba.reshape(-1)/255.0, tag="rgb")
        dpg.add_dynamic_texture(self.W, self.H, masked_rgba.reshape(-1)/255.0, tag="masked_rgb")
      dpg.add_image("rgb_init", parent='row0')
      dpg.add_image("masked_rgb_init", parent='row0')
      dpg.add_image("rgb", parent='row1')
      dpg.add_image("masked_rgb", parent='row1')
    else:
      dpg.set_value("rgb", rgba.reshape(-1)/255.0)
      dpg.set_value("masked_rgb", masked_rgba.reshape(-1)/255.0)

    dpg.set_value('frame_id', f"frame_id: {id_str}")
    dpg.set_value('keyframe_num', f"keyframe_num: {n_keyframe}")

    if dpg.get_value("mesh_render") is None:
      with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(self.W, self.H, np.zeros((4),dtype=np.float32), tag=f"mesh_render")
        dpg.add_image(f"mesh_render", parent='row2')

    if self.renderer is not None:
      self.update_render_mesh()





  def update_mesh(self, mesh):
    self.mesh = mesh
    self.clean_mesh()
    if self.renderer is None:
      from offscreen_renderer import TinyRenderer
      self.renderer = TinyRenderer(H=self.H, W=self.W, K=self.K)   #!NOTE other renderer needs openGL context and not work

    self.renderer.clear_meshes()
    self.renderer.add_mesh(self.mesh, color=[255,255,255])
    self.mesh_min_xyz = self.mesh.vertices.min(axis=0)
    self.mesh_max_xyz = self.mesh.vertices.max(axis=0)


  def set_nerf_num_frames(self, nerf_num_frames):
    dpg.set_value('nerf_num_frames', f"nerf_num_frames: {nerf_num_frames}")


if __name__=="__main__":
  gui = BundleSdfGui()

  while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

  dpg.destroy_context()