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

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

conf_path = '/mnt/aux/ml_models/M3D-RPN/M3D-RPN-Release/m3d_rpn_depth_aware_test_config.pkl'
weights_path = '/mnt/aux/ml_models/M3D-RPN/M3D-RPN-Release/m3d_rpn_depth_aware_test'

# Parse CLI args.
parser = argparse.ArgumentParser()
parser.add_argument('--generate-visualizations', action='store_true')
args = parser.parse_args()
generate_visualizations = args.generate_visualizations

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None

data_path = os.path.join(os.getcwd(), 'data')
tmp_results_path = os.path.join('output', 'tmp_results')
results_path = os.path.join(tmp_results_path, 'data')
visualizations_path = os.path.join(tmp_results_path, 'plot')

# make directory
mkdir_if_missing(results_path, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# defaults
init_torch(conf.rng_seed, conf.cuda_seed)

# -----------------------------------------
# setup network
# -----------------------------------------

# net
net = import_module('models.' + conf.model).build(conf)

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
)
