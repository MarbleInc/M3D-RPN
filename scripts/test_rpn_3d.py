# -----------------------------------------
# python modules
# -----------------------------------------
import argparse
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os

DATA_TYPE_KITTI = 'kitti'
DATA_TYPE_MARBLE = 'marble'

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
# sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# Hack to add top-level package to path so that we can run this script and import all dependencies
# referencing the top-level package.
sys.path.append(
    os.path.dirname(os.path.abspath(__file__)) + '/../../'
)

# -----------------------------------------
# custom modules
# -----------------------------------------
from m3drpn.lib.imdb_util import *

conf_path = './M3D-RPN-Release/m3d_rpn_depth_aware_test_config.pkl'
weights_path = './M3D-RPN-Release/m3d_rpn_depth_aware_test'

# Parse CLI args.
parser = argparse.ArgumentParser()
parser.add_argument('--generate-visualizations', action='store_true')
parser.add_argument(
    '--data-type', type=str, required=True, choices=[DATA_TYPE_KITTI, DATA_TYPE_MARBLE],
)
args = parser.parse_args()
generate_visualizations = args.generate_visualizations
data_type = args.data_type

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None

if data_type == DATA_TYPE_KITTI:
    data_path = os.path.join(os.getcwd(), 'data')
    object_score_threshold = 0.75 # Use Kitti default.
elif data_type == DATA_TYPE_MARBLE:
    data_path = os.path.join(os.getcwd(), 'data_marble')
    object_score_threshold = 0.25 # Reduced threshold to decrease false negatives.

tmp_results_path = os.path.join('output', 'tmp_results')
results_path = os.path.join(tmp_results_path, 'data')
visualizations_path = os.path.join(tmp_results_path, 'plot')

# make directory
mkdir_if_missing(results_path, delete_if_exist=True)
mkdir_if_missing(visualizations_path, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# defaults
init_torch(conf.rng_seed, conf.cuda_seed)

# -----------------------------------------
# setup network
# -----------------------------------------

# net
net = import_module('m3drpn.models.' + conf.model).build(conf)

# load weights
load_weights(net, weights_path, remove_module=True)

# switch modes for evaluation
net.eval()

print(pretty_print('conf', conf))

# -----------------------------------------
# test kitti
# -----------------------------------------

test_kitti_3d(
    conf.dataset_test, net, conf, results_path, data_path, use_log=False,
    generate_visualizations=generate_visualizations, visualizations_path=visualizations_path,
    object_score_threshold=object_score_threshold, data_type=data_type,
)
