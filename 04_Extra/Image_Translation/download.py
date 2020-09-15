# %%
import os
import shutil
import argparse
import cv2 as cv
from tqdm import tqdm
from tensorflow.keras import utils

URLS = {
    "cityscapes":"https://storage.googleapis.com/kaggle-data-sets/34683%2F47283%2Fupload%2Fcityscapes.tar.gz?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1600348868&Signature=GBgaAVrmtZDZDcTZsku73fpUWpxJFTkddF8gDfK8JqUnX%2FEKAw1KRVjypae8AXEfSaDVz5cotQFPMR61Cp2hobuwrv4rUigaOl4nUPmswFyrEJ2TdyUiqkSt0%2FZOS3NpZNQyHvM23OkS%2BycOlEBrWwQi6MVgtxrYOTUPvuFQWKzCPCuhJc%2B8M30cedFDQVlNWCR6WCco708929pAaDi9%2F6T1cg7naesAehiGR%2BxrHpy3Y4rS44FkFsEuvl2GegDBhgQtp56m%2FNoOLRJywSqtN81%2FCJ96Iv8iNXkUa9%2B3DRQtBqG6jm8P0NTi74vfS2c9jKOBV8H2dIyZwagzMe8XwQ%3D%3D",
    "facades":"https://storage.googleapis.com/kaggle-data-sets/34683%2F47283%2Fupload%2Ffacades.tar.gz?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1600348915&Signature=f42FabfOQvoXnmd8KQUUTIIUNgkaodPl0wjQelA28hSpzD4qE27k7nkUTCfqLrpxvIigPyQkJmIKpGCFUp6TMbBeylQwVSxYdcIFT3Z1rd6yw9QCqULckOTxJ%2B0EhtrxnSzejgtNR%2FuuNmUZDtQwnWeE%2FgJTOG%2BLOgteixozVncV7EvnJ0Im%2FPTfH2PWHXy6gntSyNpdpNdYpyxqKeqRQnvY9NUIzR6ir7YsfNOiahob%2FUB%2BvNZY3%2BX%2FoCRlRy91rb1cjSJfVIiENBD9sqOKpeURQmVF68wWhZjb1%2BPWQBX4moIr1%2B2IjeFCkKkOZ2mcltG%2BFMEpIcSWogWVIyRUrw%3D%3D",
    "maps":"https://storage.googleapis.com/kaggle-data-sets/34683%2F47283%2Fupload%2Fmaps.tar.gz?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1600348965&Signature=jVMS6hKRhRFrH5rQCrYhpzW%2FZfRfHZRzRzlYtdmJvQBIgV9qMs6BCW1IW1y7HbKLV%2BWh16DNaaXQbO%2FDOT5Y5Z3dBw7s20qgOj2rZxZmXjzi1Lq1D5OesSiqy9TucthRoRrcgJCSbFVru6Vo3FE6GiJtyXOPUSBG2k%2FTnkK5ZVtTXMeAUB%2B3tdnsSFrujFN8eHR%2FFkBcDfVOQIxPiMpj%2FB55eF0zf6ISsUFQsfmwJe5LBjerv8nbj90%2B05eD%2Bd1gXMR3TzIvFsBYBYj7JHJabEsRtESq3auwQPmvAhctGXz5zXbB%2BJ4M%2BsLRc8yT9ilwX26nE2QCk99wjZ8vBh7M9Q%3D%3D"
}

def main(args):
    dataset = args.dataset
    URL = URLS[dataset]
    print(f"Start downloading the {dataset} dataset !")
    path_to_zip  = utils.get_file(f"{dataset}.tar", origin=URL, extract=True, cache_dir='./')

    print(f"Downloading Done!")
    PATH = os.path.join(os.path.dirname(path_to_zip), dataset)
    for path, subdir, files in os.walk(PATH):
        if "domain_A" in subdir or not len(files) or "domain" in path: continue
        print(f"{path.split('/')[-1]} directory processing !")
        os.makedirs(os.path.join(path, "domain_A"), exist_ok=True)
        os.makedirs(os.path.join(path, "domain_B"), exist_ok=True)
        for file in tqdm(files):
            if file.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:continue
            img_path = os.path.join(path, file)
            img = cv.imread(img_path)
            h, w, c = img.shape
            cv.imwrite(os.path.join(path, "domain_A", file), img[:, :w//2])
            cv.imwrite(os.path.join(path, "domain_B", file), img[:, w//2:])
            os.remove(img_path)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='facades', type=str, help="")

    args = parser.parse_args()
    assert args.dataset in list(URLS.keys()), f"Please use dataset in {list(URLS.keys())}"
    main(args)
# %%
