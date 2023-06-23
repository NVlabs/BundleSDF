# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import os
import numpy as np

class Segmenter():
    def __int__(self):
        return

    def run(self, mask_file=None):
        return (cv2.imread(mask_file, -1)>0).astype(np.uint8)
