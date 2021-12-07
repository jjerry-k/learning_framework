# %%
import argparse

def main(args):

    import io
    import os
    import sys
    import time
    import shutil
    import tarfile
    import urllib.request

    datatype = args.datatype
    dataset = args.dataset
    SAVE_PATH = "./datasets"
    os.makedirs(SAVE_PATH, exist_ok=True)
    if datatype == 'paired':
        URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset}.tar.gz"
    else : 
        URL = f"http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset}.zip"

    file_name = URL.split("/")[-1]
    file_path = os.path.join(SAVE_PATH, file_name)
    # Data Download
    
    ## For progressbar
    def report(url):
        file_name = url.split("/")[-1]
        def progbar(blocknr, blocksize, size):
            current = blocknr*blocksize
            sys.stdout.write(f"\rDownloading {file_name} ...... {100.0*current/size:.2f}%")
        return progbar
        
    if not os.path.exists(file_path):
        print(f"In progress to download '{dataset}' data ....")
        urllib.request.urlretrieve(URL, file_path, report(URL))
        print()
    else:
        print("Already downloaded !")

    print(f"Downloading Done!")

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
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatype", default='paired', type=str, help="")
    parser.add_argument("--dataset", default='facades', type=str, help="")

    args = parser.parse_args()

    datatype_list = ['paired', 'unpaired']
    assert args.datatype in datatype_list, f"Please use dataset in {datatype_list}"
    
    if args.datatype == 'paired':
        data_list = ['cityscapes', 'edges2handbags', 'edges2shoes', 'facades', 'maps', 'night2day']
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
