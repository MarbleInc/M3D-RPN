#!/usr/bin/env python
"""
Custom Marble module exposing detection functionality (such as for running inside a ROS node).
"""

from collections import namedtuple
import sys

from easydict import EasyDict
import numpy as np
import torch
import torch.nn.functional as F

from m3drpn.lib.augmentations import *
from m3drpn.lib.core import *
from m3drpn.lib.nms.gpu_nms import gpu_nms
from m3drpn.lib.rpn_util import *
from m3drpn.lib.util import *


OBJECT_SCORE_THRESHOLD = 0.3
# Preprocessing.
PREPROCESS_IMAGE_MEANS_KITTI = [0.485, 0.456, 0.406]
PREPROCESS_IMAGE_STDS_KITTI = [0.229, 0.224, 0.225]
PREPROCESS_SIZE = 512
PREPROCESS = Preprocess(
    size=[PREPROCESS_SIZE],
    mean=PREPROCESS_IMAGE_MEANS_KITTI,
    stds=PREPROCESS_IMAGE_STDS_KITTI,
)


def load_conf(path):
    with open(path, 'rb') as f:
        conf = EasyDict(pickle.load(f))
    conf.pretrained = None
    return conf


# Detection tuple class.
Detection = namedtuple('Detection', [
    'x1', 'y1', 'x2', 'y2',
    'x3d', 'y3d', 'z3d', 'ry3d', 'alpha',
    'h3d', 'w3d', 'l3d',
    'cls', 'score',
])

def detect(
    net,
    im,
    proj_mat,
    conf,
    object_score_threshold=OBJECT_SCORE_THRESHOLD,
    use_log=True,
):
    """
    Primary detection function. Runs inference and returns detections for a single image. Adapted
    from `test_kitti_3d`; see there for more info.
    """
    # Forward pass.
    aboxes = im_detect_3d(
        im=im,
        net=net,
        rpn_conf=conf,
        preprocess=PREPROCESS,
        p2=proj_mat,
    )

    # Instantiate list of all detections for this image.
    detections = []

    # For each of the top boxes, if score is above threshold, get detection values and add to list
    # of detections.
    for boxind in range(0, min(rpn_conf.nms_topN_post, aboxes.shape[0])):
        box = aboxes[boxind, :]
        score = box[4]
        cls_ind = int(box[5] - 1)
        cls = rpn_conf.lbls[cls_ind]

        if score < object_score_threshold:
            continue

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        width = (x2 - x1 + 1)
        height = (y2 - y1 + 1)

        # get 3D box
        x3d = box[6]
        y3d = box[7]
        z3d = box[8]
        w3d = box[9]
        h3d = box[10]
        l3d = box[11]
        ry3d = box[12]

        # convert alpha into ry3d
        coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
        ry3d = convertAlpha2Rot(ry3d, coord3d[2], coord3d[0])

        step_r = 0.3*math.pi
        r_lim = 0.01
        box_2d = np.array([x1, y1, width, height])

        z3d, ry3d, verts_best = hill_climb(p2, p2_inv, box_2d, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, step_r_init=step_r, r_lim=r_lim)

        # predict a more accurate projection
        coord3d = np.linalg.inv(p2).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
        alpha = convertRot2Alpha(ry3d, coord3d[2], coord3d[0])

        x3d = coord3d[0]
        y3d = coord3d[1]
        z3d = coord3d[2]

        y3d += h3d/2

        # Append new detection for this object to the list of all detections for this image.
        detections.append(
            Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                x3d=x3d, y3d=y3d, z3d=z3d, ry3d=ry3d, alpha=alpha,
                h3d=h3d, w3d=w3d, l3d=l3d,
                cls=cls, score=score,
            )
        )

    return detections
