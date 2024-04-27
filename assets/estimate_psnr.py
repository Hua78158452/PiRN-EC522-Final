import os
import sys
import math
import time
import lpips
import random
import datetime
import functools
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange

from datapipe.datasets import create_dataset
from models.resample import UniformSampler

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils

from utils import util_net
from utils import util_common
from utils import util_image
from tqdm import tqdm

images_lq = util_common.readline_txt(os.path.join("E:/EC522-final-estimate-part/PiRN-main/assets/inference_output/output-OTIS/Pattern16/test_data_lq.txt"))
images_restored = [x for x in images_lq]
images_gt = util_common.readline_txt(os.path.join("E:/EC522-final-estimate-part/PiRN-main/assets/inference_output/output-OTIS/Pattern16/test_data_gt.txt"))
fopen = open("E:/EC522-final-estimate-part/PiRN-main/assets/inference_output/output-OTIS/Pattern16/restored_gt_psnr_ssim.txt", "a")
psnr_lq_gt = 0
ssim_lq_gt = 0
count = 0
print(len(images_lq))

for i in range(1, len(images_lq)+1):
    print(i)

    try:
        print(images_restored[i - 1])
        if os.path.exists(images_lq[i-1]) and os.path.exists(images_gt[0]):
            img_lq = util_image.imread(images_restored[i - 1], chn='bgr', dtype='uint8')
            img_gt = util_image.imread(images_gt[0], chn='bgr', dtype='uint8')

            #img_lq = cv.imread(img_path, cv.IMREAD_COLOR)
            #img_gt = cv.imread(gt_path, cv.IMREAD_COLOR)
        else:
            print(f"File not found: {images_lq} or {images_gt}")
        #img_lq = util_image.imread(images_restored[i - 1], chn='bgr', dtype='uint8')
        #img_gt = util_image.imread(images_gt[i - 1], chn='bgr', dtype='uint8')
        psnr_lq_gt += util_image.calculate_psnr(img_lq, img_gt)
        ssim_lq_gt += util_image.calculate_ssim(img_lq, img_gt)
        count += 1

    # if i % 500 == 0:
    finally:
        print(f"[{count}|{i}] Current PSNR: {psnr_lq_gt/i} and SSIM: {ssim_lq_gt/i}")
        fopen.write(f"[{count}|{i}] Current PSNR: {psnr_lq_gt/i} and SSIM: {ssim_lq_gt/i}\n")
        fopen.flush()
