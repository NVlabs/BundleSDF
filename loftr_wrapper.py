# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,zmq,pdb,sys,time,torchvision
code_dir = os.path.dirname(os.path.realpath(__file__))
import argparse
import cv2
import torch,imageio
from BundleTrack.LoFTR.src.loftr import *
from Utils import *


class LoftrRunner:
  def __init__(self):
    default_cfg['match_coarse']['thr'] = 0.2
    print("default_cfg",default_cfg)
    self.matcher = LoFTR(config=default_cfg)
    self.matcher.load_state_dict(torch.load(f'{code_dir}/BundleTrack/LoFTR/weights/outdoor_ds.ckpt')['state_dict'])
    self.matcher = self.matcher.eval().cuda()


  @torch.no_grad()
  def predict(self, rgbAs:np.ndarray, rgbBs:np.ndarray):
    '''
    @rgbAs: (N,H,W,C)
    '''
    image0 = torch.from_numpy(rgbAs).permute(0,3,1,2).float().cuda()
    image1 = torch.from_numpy(rgbBs).permute(0,3,1,2).float().cuda()
    if image0.shape[-1]==3:
      image0 = torchvision.transforms.functional.rgb_to_grayscale(image0)
      image1 = torchvision.transforms.functional.rgb_to_grayscale(image1)
    image0 = image0/255.0
    image1 = image1/255.0
    last_data = {'image0': image0, 'image1': image1}
    logging.info(f"image0: {last_data['image0'].shape}")

    batch_size = 64
    ret_keys = ['mkpts0_f','mkpts1_f','mconf','m_bids']
    with torch.cuda.amp.autocast(enabled=True):
      i_b = 0
      for b in range(0,len(last_data['image0']),batch_size):
        tmp = {'image0': last_data['image0'][b:b+batch_size], 'image1': last_data['image1'][b:b+batch_size]}
        with torch.no_grad():
          self.matcher(tmp)
        tmp['m_bids'] += i_b
        for k in ret_keys:
          if k not in last_data:
            last_data[k] = []
          last_data[k].append(tmp[k])
        i_b += len(tmp['image0'])

    logging.info("net forward")

    for k in ret_keys:
      last_data[k] = torch.cat(last_data[k],dim=0)

    total_n_matches = len(last_data['mkpts0_f'])
    mkpts0 = last_data['mkpts0_f'].cpu().numpy()
    mkpts1 = last_data['mkpts1_f'].cpu().numpy()
    mconf = last_data['mconf'].cpu().numpy()
    pair_ids = last_data['m_bids'].cpu().numpy()
    logging.info(f"mconf, {mconf.min()} {mconf.max()}")
    logging.info(f'pair_ids {pair_ids.shape}')
    corres = np.concatenate((mkpts0.reshape(-1,2),mkpts1.reshape(-1,2),mconf.reshape(-1,1)),axis=-1).reshape(-1,5).astype(np.float32)

    logging.info(f'corres: {corres.shape}')
    corres_tmp = []
    for i in range(len(rgbAs)):
      cur_corres = corres[pair_ids==i]
      corres_tmp.append(cur_corres)
    corres = corres_tmp

    del last_data, image0, image1
    torch.cuda.empty_cache()

    return corres