# DataLoading

---

Framework 별로 Classification을 수행할 때 Data를 Load 하는 방법에 대해서 알아봅니다. 

## Data
- `flower_photos` dataset을 이용합니다. 
- `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips` 와 같이 5개의 class로 구성된 dataset입니다.
- `flower_download.py` script를 실행하면 dataset setting은 자동으로 됩니다.
- 다른 데이터로 직접 setting을 하고 싶으시다면 data download 후 다음과 같이 Directory tree를 구성합니다.
```
dataset name
│
├─── train
│   ├─── class1
│   │   │   ~~~~.jpg
│   │   └───...
│   │
│   ├─── class2
│   │   │   ~~~~..jpg
│   │   └───...
│   │
│   ├─── class3
│   │   │   ~~~~..jpg
│   │   └───...
│   │
│   ├─── class4
│   │   │   ~~~~..jpg
│   │   └───...
│   │
│   └─── class5
│       │   ~~~~..jpg
│       └───...
│   
└─── validation
    ├─── class1
    │   │   ~~~~..jpg
    │   └───...
    │
    ├─── class2
    │   │   ~~~~..jpg
    │   └───...
    │
    ├─── class3
    │   │   ~~~~..jpg
    │   └───...
    │
    ├─── class4
    │   │   ~~~~..jpg
    │   └───...
    │
    └─── class5
        │   ~~~~..jpg
        └───...
```

## PyTorch
- Using `torchvision.datasets.ImageFolder`

- Using `Custom dataset class`