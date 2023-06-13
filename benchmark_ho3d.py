# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import joblib,argparse
import pandas as pd
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/BundleTrack/scripts')
from data_reader import *


def benchmark_one_video(method,video_dir):
  print('\n',video_dir)
  reader = Ho3dReader(video_dir)
  video_name = reader.get_video_name()
  pred_mesh = None

  benchmark_pose = True
  benchmark_mesh = True

  dir = f'{args.out_dir}/{video_name}'
  pose_files = sorted(glob.glob(f'{dir}/ob_in_cam/*.txt'))
  pred_poses = []
  if len(pose_files)<len(reader.color_files):
    raise RuntimeError(f"Pose file missing: {video_dir}")
  for i in range(len(reader.color_files)):
    pred_poses.append(np.loadtxt(pose_files[i]))
  pred_poses = np.array(pred_poses)
  if benchmark_mesh:
    tmp = sorted(glob.glob(f"{dir}/**/*mesh_real_world.obj",recursive=True))
    if len(tmp)>0:
      pred_mesh = trimesh.load(tmp[-1])
    else:
      pred_mesh_file = sorted(glob.glob(f"{dir}/**/*mesh_normalized_space.obj",recursive=True))[-1]
      print("pred_mesh_file",pred_mesh_file)
      pred_mesh = trimesh.load(pred_mesh_file)
      cfg = yaml.load(open(f"{os.path.dirname(pred_mesh_file)}/config.yml",'r'))
      translation = np.array(cfg['translation'])
      pred_mesh.vertices = pred_mesh.vertices/cfg['sc_factor'] - translation.reshape(1,3)

  gt_poses = []
  ids = []
  for i in range(len(reader.color_files)):
    gt_pose = reader.get_gt_pose(i)
    if gt_pose is None:
      continue
    gt_poses.append(gt_pose)
    ids.append(i)
  ids = np.array(ids)

  gt_poses = np.array(gt_poses)
  pred_poses = np.array(pred_poses)[ids]

  ######### Align first frame
  pred_pose_init_old = pred_poses[0].copy()
  pred_poses = pred_poses@np.linalg.inv(pred_poses[0])@gt_poses[0]

  adi_errs = []
  add_errs = []
  mesh = reader.get_gt_mesh()

  if benchmark_pose:
    for i in range(len(pred_poses)):
      adi = adi_err(pred_poses[i],gt_poses[i],mesh.vertices.copy())
      add = add_err(pred_poses[i],gt_poses[i],mesh.vertices.copy())
      adi_errs.append(adi)
      add_errs.append(add)

  adi_errs = np.array(adi_errs)
  add_errs = np.array(add_errs)
  ADDS_AUC = compute_auc(adi_errs)*100
  ADD_AUC = compute_auc(add_errs)*100

  ############ Mesh
  cd = np.inf
  if benchmark_mesh and pred_mesh is not None:
    pcd = o3d.io.read_point_cloud(f'{reader.video_dir}/visible_mesh.ply')
    pcd = pcd.voxel_down_sample(0.005)
    o3d.io.write_point_cloud(f'{args.log_dir}/gt_{video_name}.ply', pcd)
    gt_pts = np.asarray(pcd.points).copy()

    pred_mesh.apply_transform(pred_pose_init_old)
    pred_mesh.apply_transform(np.linalg.inv(gt_poses[0]))
    pred_mesh.export(f'{args.log_dir}/pred_mesh_{video_name}.obj')


    #######!NOTE need this since some large mesh explode memory
    max_coord = gt_pts.max(axis=0).reshape(1,3) + 0.3
    min_coord = gt_pts.min(axis=0).reshape(1,3) - 0.3
    bad_mask = (pred_mesh.vertices>max_coord).any(axis=-1) | (pred_mesh.vertices<min_coord).any(axis=-1)
    pred_mesh.vertices[bad_mask] = np.inf

    pred_mesh = trimesh_clean(pred_mesh)
    pred_mesh.export(f'{args.log_dir}/pred_mesh_cleaned_{video_name}.obj')

    components = trimesh_split(pred_mesh, min_edge=1000)
    if len(components)==0:
      components = trimesh_split(pred_mesh, min_edge=3)

    best_component = None
    best_size = 0
    for component in components:
      dists = np.linalg.norm(component.vertices,axis=-1)
      if dists.min()>0.1:
        continue
      if len(component.vertices)>best_size:
        best_size = len(component.vertices)
        best_component = component
    pred_mesh = best_component

    pred_mesh.export(f'{args.log_dir}/pred_mesh_biggest_{video_name}.obj')

    pred_pts,_ = trimesh.sample.sample_surface(pred_mesh, 99999, face_weight=None, sample_color=False)

    pcd_pred = toOpen3dCloud(pred_pts)
    pcd_pred = pcd_pred.voxel_down_sample(0.005)
    pcd_gt = toOpen3dCloud(gt_pts)
    thres = 0.02
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd_pred, pcd_gt, thres, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pred_pts_icp = (reg_p2p.transformation@to_homo(pred_pts).T).T[:,:3]
    chamfer_dists = chamfer_distance_between_clouds_mutual(pred_pts_icp, gt_pts)
    cd = chamfer_dists.mean()*100
    print("chamfer_dist(cm)",cd)

    pcd = toOpen3dCloud(gt_pts)
    o3d.io.write_point_cloud(f'{args.log_dir}/gt_pts_{video_name}.ply',pcd)
    pcd = toOpen3dCloud(pred_pts)
    o3d.io.write_point_cloud(f'{args.log_dir}/pred_pts_{video_name}.ply',pcd)


  print(f"video {video_name}, ADD-S_err: {adi_errs.mean()*100:.2f}[cm], ADD_errs: {add_errs.mean()*100:.2f}[cm], ADD-S_AUC: {ADDS_AUC:.2f}, ADD_AUC: {ADD_AUC:.2f}")

  return {f'{method}/{video_name}/ADDS(cm)':adi_errs*100, f'{method}/{video_name}/ADD(cm)':add_errs*100, f'{method}/{video_name}/ADDS_AUC(%)': ADDS_AUC, f'{method}/{video_name}/ADD_AUC(%)': ADD_AUC, f"{method}/{video_name}/chamfer_dist(cm)":cd}



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--video_dirs', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/SM1")
  parser.add_argument('--out_dir', type=str, default=f"/home/bowen/debug/ho3d_ours")
  parser.add_argument('--log_dir', type=str, default=f"/home/bowen/debug/")
  args = parser.parse_args()

  method = 'ours'

  os.makedirs(args.log_dir, exist_ok=True)

  video_dirs = args.video_dirs.split(',')
  out_data = {}
  args = []
  for video_dir in video_dirs:
    out = benchmark_one_video(method, video_dir)
    out_data.update(out)

  out = {}
  for k in out_data.keys():
    metric = k.split('/')[-1]
    if metric not in out:
      out[metric] = []
    if isinstance(out_data[k], np.ndarray):
      out[metric].append(out_data[k].mean())
    else:
      out[metric].append(out_data[k])

  for metric in out:
    print(f'{metric}: {np.array(out[metric]).mean():.3f}')

  with open(f'{args.log_dir}/ho3d_{method}.pkl','wb') as ff:
    print("out_data",out_data.keys())
    pickle.dump(out_data,ff)

  video_names = []
  metrics = []
  for k in out_data:
    tmp = k.split('/')
    video_names.append(tmp[1])
    metrics.append(tmp[2])
  video_names = list(np.unique(video_names))
  metrics = list(np.unique(metrics))
  cols = {'videos': video_names}
  for video_name in video_names:
    for metric in metrics:
      if metric not in cols:
        cols[metric] = []
      k = f'{method}/{video_name}/{metric}'
      v = out_data[k]
      if isinstance(v, np.ndarray):
        v = v.mean()
      cols[metric].append(float(v))

  df = pd.DataFrame(cols, index=[method]*len(video_names))

  mean_dict = {}
  for col in cols:
    if col=='videos':
      continue
    mean_dict[col] = df[col].mean()
  df_mean = pd.DataFrame(mean_dict, index=['ALL'])
  df = pd.concat([df, df_mean])

  df.to_excel(f'{args.log_dir}/ho3d_{method}.xlsx', sheet_name='per_ob')
