# %%
import argparse

def main(args):

    import os
    import cv2 as cv
    from tqdm import tqdm
    from tensorflow.keras import utils

    datatype = args.datatype
    dataset = args.dataset
    if datatype == 'paired':
        URL = f"https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{dataset}.tar.gz"
        print(f"Start downloading the {dataset} dataset !")
        path_to_zip  = utils.get_file(f"{dataset}.tar", origin=URL, extract=True, cache_dir='./')

        print(f"Downloading Done!")
        PATH = os.path.join(os.path.dirname(path_to_zip), dataset)
        for path, subdir, files in os.walk(PATH):
            if "domain_A" in subdir or not len(files) or "domain" in path: continue
            
            data = path.split('/')[-1]
            print(f"{data} directory processing !")

            # os.makedirs(os.path.join(path, "domain_A"), exist_ok=True)
            # os.makedirs(os.path.join(path, "domain_B"), exist_ok=True)
            os.makedirs(os.path.join(PATH, f"{data}A"), exist_ok=True)
            os.makedirs(os.path.join(PATH, f"{data}B"), exist_ok=True)
            
            for file in tqdm(files):
                if file.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:continue
                img_path = os.path.join(path, file)
                img = cv.imread(img_path)
                h, w, c = img.shape
                # cv.imwrite(os.path.join(path, "domain_A", file), img[:, :w//2])
                # cv.imwrite(os.path.join(path, "domain_B", file), img[:, w//2:])
                cv.imwrite(os.path.join(PATH, f"{data}A", file), img[:, :w//2])
                cv.imwrite(os.path.join(PATH, f"{data}B", file), img[:, w//2:])
                os.remove(img_path)
            os.removedirs(path)
    else : 
        URL = f"https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{dataset}.zip"
        print(f"Start downloading the {dataset} dataset !")
        path_to_zip  = utils.get_file(f"{dataset}.tar", origin=URL, extract=True, cache_dir='./')

        print(f"Downloading Done!")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatype", default='paired', type=str, help="")
    parser.add_argument("--dataset", default='facades', type=str, help="")

    args = parser.parse_args()

    datatype_list = ['paired', 'unpaired']
    assert args.datatype in datatype_list, f"Please use dataset in {datatype_list}"
    
    if args.datatype == 'paired':
        data_list = ['cityscapes', 'edges2handbags', 'edges2shoes', 'facades', 'maps']
    else:
        data_list = ["apple2orange", "summer2winter_yosemite", "horse2zebra", "monet2photo", \
                    "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "iphone2dslr_flower", "ae_photos"]

    assert args.dataset in data_list, f"Please use dataset in {data_list}"
    
    dict_args = vars(args)
    for i in dict_args.keys():
        assert dict_args[i]!=None, '"%s" key is None Value!'%i
    print("\n================ Options ================")
    print(f"Dataset : {args.dataset}")
    print("===========================================\n")

    
    main(args)
# %%
