# Image Translation

## How to Run

1. Download dataset
``` bash
# In Image_Translation
python download.py --dataset {dataset} # datasets: ['cityscapes', 'facades', 'maps']
```

2. Run what you want (except Neural_Style_Transfer)
``` bash
cd ./{model}/{framework}
python main.py --DATASET {dataset} --IMG_SIZE {img_size} --EPOCHS {epochs} --BATCH_SIZE {batch_size}
``` 


### To-Do List
- [x] Pix2Pix model ver.TensorFlow
- [ ] Pix2Pix model ver.PyTorch
- [x] CycleGAN model ver.TensorFlow
- [ ] CycleGAN model ver.PyTorch
