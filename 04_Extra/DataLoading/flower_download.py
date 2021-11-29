import os
import sys
import time
import urllib.request
from tqdm import tqdm

def report(url):
    file_name = url.split("/")[-1]
    def progbar(blocknr, blocksize, size):
        current = blocknr*blocksize
        sys.stdout.write(f"\rDownloading {file_name} ...... {100.0*current/size:.2f}%")
    return progbar

SAVE_PATH = "./data"
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
file_name = URL.split("/")[-1]
file_path = os.path.join(SAVE_PATH, file_name)

# Data Download
print("In progress to download data ....")
urllib.request.urlretrieve(URL, file_path, report(URL))
print("\Done !")

# Data Extract
print("In progress to extract data ....")

# Data Split
print("In progress to split data ....")
