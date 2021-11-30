import io
import os
import sys
import time
import shutil
import tarfile
import urllib.request

# For progressbar
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
if not os.path.exists(file_path):
    print("In progress to download data ....")
    urllib.request.urlretrieve(URL, file_path, report(URL))
    print()
else:
    print("Already downloaded !")

# Data Extract
if not os.path.exists(os.path.join(SAVE_PATH, "flower_photos")):
    print("In progress to extract data ....")
    tar = tarfile.open(file_path)
    members = tar.getmembers()
    for idx, member in enumerate(members):
        tar.extract(member=member, path=SAVE_PATH)
        sys.stdout.write(f"\rExtracting {file_name} ...... {100.0*(idx+1)/len(members):.2f}%")
    print()
else:
    print("Already extracted !")

# Data Split
print("In progress to split data ....")

flower_photos_path = os.path.join(SAVE_PATH, "flower_photos")
for i, (root, subdir, files) in enumerate(os.walk(flower_photos_path)):
    
    if not i: continue

    flower = root.split("/")[-1]

    print(f"{flower} ......")

    split_ratio = int(0.9 * len(files))   
    
    # Move to train directory
    dst_root = os.path.join(SAVE_PATH, "flower_photos", "train", flower)
    os.makedirs(dst_root, exist_ok=True)
    for file in files[:split_ratio]:
        src_path = os.path.join(root, file)
        dst_path = os.path.join(dst_root, file)
        shutil.move(src_path, dst_path)

    # Move to validation directory
    dst_root = os.path.join(SAVE_PATH, "flower_photos", "validation", flower)
    os.makedirs(dst_root, exist_ok=True)
    for file in files[split_ratio:]:
        src_path = os.path.join(root, file)
        dst_path = os.path.join(dst_root, file)
        shutil.move(src_path, dst_path)

    shutil.rmtree(root)

print("Data preparation done !")