import os
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader

SAVE_PATH = "../../data"
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
file_name = URL.split("/")[-1]
data = datasets.utils.download_and_extract_archive(URL, SAVE_PATH)
PATH = os.path.join(SAVE_PATH, "flower_photos")

category_list = [i for i in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, i)) ]
print(category_list)

num_classes = len(category_list)
img_size = 128

train_set = torchvision.datasets.ImageFolder()