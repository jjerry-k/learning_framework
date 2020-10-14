import os
import cv2 as cv
from tqdm import tqdm
from tensorflow.keras import utils

dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
data_dir = utils.get_file(origin=dataset_url, fname="BSR", untar=True, cache_dir = './')
root_dir = os.path.join(data_dir, "BSDS500/data")