# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from bundlesdf import *
import argparse
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/BundleTrack/scripts')
from data_reader import *


def run_one_video(video_dir,out_dir):
  set_seed(0)

  reader = Ho3dReader(video_dir)
  video_name = reader.get_video_name()
  out_folder = f'{out_dir}/{video_name}/'   #!NOTE there has to be a / in the end
  if os.path.exists(f'{out_folder}/ob_in_cam'):
    pose_files = sorted(glob.glob(f'{out_folder}/ob_in_cam/*.txt'))
    if len(pose_files)==len(reader.color_files):
      print(f"{out_folder} done before, skip")
      return

  os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")

  code_dir = os.path.dirname(os.path.realpath(__file__))
  cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_bundletrack['data_dir'] = video_dir
  cfg_bundletrack['SPDLOG'] = 2
  cfg_bundletrack['depth_processing']["zfar"] = 1
  cfg_bundletrack['debug_dir'] = out_folder
  cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{code_dir}/config.yml",'r'))
  cfg_nerf['trunc_start'] = 0.01
  cfg_nerf['trunc'] = 0.01
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
  cfg_nerf['datadir'] = f"{out_folder}/nerf_with_bundletrack_online"
  cfg_nerf['save_dir'] = copy.deepcopy(cfg_nerf['datadir'])
  cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5, use_gui=args.use_gui)

  for i,color_file in enumerate(reader.color_files):
    color = cv2.imread(color_file)
    H,W = color.shape[:2]
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0)
      mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
    else:
      mask = reader.get_mask(i)
      mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
    id_str = reader.id_strs[i]
    tracker.run(color, depth, reader.K, id_str, mask=mask, occ_mask=None)

  tracker.on_finish()
  print(f"Done {video_dir}")


def run_one_video_global_nerf(video_dir,out_dir):
  set_seed(0)

  reader = Ho3dReader(video_dir)
  video_name = reader.get_video_name()
  out_folder = f'{out_dir}/{video_name}/'   #!NOTE there has to be a / in the end

  tracker = BundleSdf(cfg_track_dir=f"{out_folder}/config_bundletrack.yml", cfg_nerf_dir=f"{out_folder}/config_nerf.yml", start_nerf_keyframes=5, use_gui=False)
  tracker.cfg_nerf['n_step'] = 2000
  tracker.cfg_nerf['N_samples'] = 256
  tracker.cfg_nerf['N_samples'] = 128
  tracker.cfg_nerf['down_scale_ratio'] = 1
  tracker.cfg_nerf['finest_res'] = 512
  tracker.cfg_nerf['num_levels'] = 16
  tracker.cfg_nerf['mesh_resolution'] = 0.003

  tracker.cfg_nerf['i_img'] = 500
  tracker.cfg_nerf['i_mesh'] = tracker.cfg_nerf['i_img']
  tracker.cfg_nerf['i_nerf_normals'] = tracker.cfg_nerf['i_img']
  tracker.cfg_nerf['i_save_ray'] = tracker.cfg_nerf['i_img']

  tracker.debug_dir = f'{out_folder}'
  tracker.cfg_nerf['datadir'] = f"{tracker.debug_dir}/nerf_with_bundletrack_online"
  tracker.cfg_nerf['save_dir'] = copy.deepcopy(tracker.cfg_nerf['datadir'])

  tracker.run_global_nerf()

  print(f"Done {video_dir}")


def run_all():
  video_dirs = sorted(glob.glob('/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/*'))

  for video_dir in video_dirs:
    run_one_video(video_dir, out_dir=args.out_dir)


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--video_dirs', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/MPM10")
  parser.add_argument('--out_dir', type=str, default="/home/bowen/debug/ho3d_ours")
  parser.add_argument('--use_segmenter', type=int, default=0)
  parser.add_argument('--use_gui', type=int, default=0)
  args = parser.parse_args()

  use_segmenter = args.use_segmenter
  video_dirs = args.video_dirs.split(',')
  print("video_dirs:\n",video_dirs)
  for video_dir in video_dirs:
    run_one_video(video_dir, args.out_dir)

