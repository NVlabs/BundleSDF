front_matter = """
------------------------------------------------------------------------
Online demo for [LoFTR](https://zju3dv.github.io/loftr/).

This demo is heavily inspired by [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/).
We thank the authors for their execellent work.
------------------------------------------------------------------------
"""

import os
import argparse
from pathlib import Path
import cv2
import torch,imageio
import numpy as np
import matplotlib.cm as cm
from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults
try:
  from demo.utils import (AverageTimer, VideoStreamer,make_matching_plot_fast, make_matching_plot, frame2tensor)
except:
  raise ImportError("This demo requires utils.py from SuperGlue, please use run_demo.sh to start this script.")


torch.set_grad_enabled(False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--weight', type=str, default='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/github/LoFTR/weights/outdoor_ds.ckpt', help="Path to the checkpoint.")
  parser.add_argument('--output_dir', type=str, default='/home/bowen/debug/',help='Directory where to write output frames (If None, no output)')
  parser.add_argument('--resize', type=int, nargs='+', default=[640, 480],help='Resize the input image before running inference. If two numbers, resize to the exact dimensions, if one number, resize the max ''dimension, if -1, do not resize')
  parser.add_argument('--top_k', type=int, default=2000, help="The max vis_range (please refer to the code).")
  parser.add_argument('--bottom_k', type=int, default=0, help="The min vis_range (please refer to the code).")

  opt = parser.parse_args()

  if torch.cuda.is_available():
      device = 'cuda'
  else:
      raise RuntimeError("GPU is required to run this demo.")

  # Initialize LoFTR
  matcher = LoFTR(config=default_cfg)
  matcher.load_state_dict(torch.load(opt.weight)['state_dict'])
  matcher = matcher.eval().to(device=device)

  frameA = imageio.imread(f'/home/bowen/debug/BundleTrack/0229/feature_det_image.png')
  frameB = imageio.imread(f'/home/bowen/debug/BundleTrack/0216/feature_det_image.png')

  if len(frameA.shape)==3:
    frameA = cv2.cvtColor(frameA,cv2.COLOR_RGB2GRAY)
    frameB = cv2.cvtColor(frameB,cv2.COLOR_RGB2GRAY)

  frame_tensor = frame2tensor(frameA, device)
  last_data = {'image0': frame_tensor}

  if opt.output_dir is not None:
      print('==> Will write outputs to {}'.format(opt.output_dir))
      Path(opt.output_dir).mkdir(exist_ok=True)


  vis_range = [opt.bottom_k, opt.top_k]

  frame_tensor = frame2tensor(frameB, device)
  last_data = {**last_data, 'image1': frame_tensor}
  matcher(last_data)

  total_n_matches = len(last_data['mkpts0_f'])
  mkpts0 = last_data['mkpts0_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
  mkpts1 = last_data['mkpts1_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
  mconf = last_data['mconf'].cpu().numpy()[vis_range[0]:vis_range[1]]

  # Normalize confidence.
  if len(mconf) > 0:
      conf_vis_min = 0.
      conf_min = mconf.min()
      conf_max = mconf.max()
      mconf = (mconf - conf_vis_min) / (conf_max - conf_vis_min + 1e-5)

  alpha = 0
  color = cm.jet(mconf, alpha=alpha)

  text = [f'LoFTR','# Matches (showing/total): {}/{}'.format(len(mkpts0), total_n_matches),]
  small_text = [
      f'Showing matches from {vis_range[0]}:{vis_range[1]}',
      f'Confidence Range: {conf_min:.2f}:{conf_max:.2f}',
  ]
  out = make_matching_plot_fast(frameA, frameB, mkpts0, mkpts1, mkpts0, mkpts1, color, text,path=f"{opt.output_dir}/match.png", show_keypoints=True, small_text=small_text)

