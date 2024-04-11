# **현재 리뉴얼 중입니다!**

# Learning Python A.I Framework

- 본 Repository 는 지극히 개인적인 공부용이라 모든 코드들이 불친절하고 가독성이 좋지 않습니다.
- 대부분 Network만 구성해볼뿐 학습을 해보진 않습니다. (물론 추후에 학습도 해볼 예정)

---
## Framework list 
- PyTorch: 2.2
- TensorFlow: 2.16
- ~~MXNet: 1.9~~ -> 폐기
- JAX: 0.4.26
- MLX: 0.9.1

<details>
<summary> Additional information </summary>

### PyTorch
- 배포 : Facebook
- 공식 홈페이지 : https://pytorch.org
- 주 사용 프레임워크

### TensorFlow
- 배포 : Google
- 공식 홈페이지 : https://www.tensorflow.org
- 서브 프레임워크
- ~~Low level(tf.nn),~~ High level(tf.keras), model subclassing API 로 작성하려함.

### ~~MXNet~~
- ~~배포 : Apache~~
- ~~공식 홈페이지 : https://mxnet.apache.org~~
- ~~그냥...써보려고 함....~~
- ~~Gluon, Module 로 작성하려함.~~
- ~~매우...진행이 느릴 것으로 예상.~~
- [프로젝트 중단](https://attic.apache.org/projects/mxnet.html)

### JAX
- 배포 : Google
- 공식 홈페이지 : https://github.com/google/jax
- 그냥....써보려고....
- Transformer가 jax 기반이길래...

### MLX
- 배포 : Apple
- 공식 홈페이지 : https://github.com/ml-explore/mlx
- 맥북에서 공부하기 위한...방법!

</details>

## Requirement

``` bash
# 공통 설치 패키지
pip install numpy matplotlib scikit-learn Pillow opencv-python tqdm

# 필요에 따라 원하는 Deep Learning 프레임워크 설치
```

## Example List

### 01 Basic Usage

<details>
<summary> Contents </summary>

1. Linear Regression  
[tf.keras](01_Basic/Linear_Regression/tf_keras.py),
[tf.nn](01_Basic/Linear_Regression/tf_nn.py), 
[PyTorch](01_Basic/Linear_Regression/PyTorch.py), 
[MXNet Gluon](01_Basic/Linear_Regression/MXNet_Gluon.py),
[JAX](01_Basic/Linear_Regression/ver_jax.py)

2. Logistic Regression  
[tf.keras](01_Basic/Logistic_Regression/tf_keras.py), 
[tf.nn](01_Basic/Logistic_Regression/tf_nn.py), 
[PyTorch](01_Basic/Logistic_Regression/PyTorch.py), 
[MXNet Gluon](01_Basic/Logistic_Regression/MXNet_Gluon.py)

</details>

### 02 Intermediate

<details>
<summary> Contents </summary>

1. Multi Layer Network  
[tf.keras](02_Intermediate/Multi_Layer_Neural_Network/tf_keras.py), 
[tf.nn](02_Intermediate/Multi_Layer_Neural_Network/tf_nn.py), 
[PyTorch](02_Intermediate/Multi_Layer_Neural_Network/PyTorch.py), 
[MXNet Gluon](02_Intermediate/Multi_Layer_Neural_Network/MXNet_Gluon.py)

2. Simple Convolutional Neural Network  
[tf.keras](02_Intermediate/Simple_Convolutional_Neural_Network/tf_keras.py), 
[tf.nn](02_Intermediate/Simple_Convolutional_Neural_Network/tf_nn.py), 
[PyTorch](02_Intermediate/Simple_Convolutional_Neural_Network/PyTorch.py), 
[MXNet Gluon](02_Intermediate/Simple_Convolutional_Neural_Network/MXNet_Gluon.py)

</details>

### 03 Advance
#### Advance Convolutional Neural Network

<details>
<summary> Contents </summary>

1. VGGNet  (https://arxiv.org/abs/1409.1556)  
[tf.keras](03_Advance/CNN/VGGNet/tf_keras.py), 
[PyTorch](03_Advance/CNN/VGGNet/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/VGGNet/MXNet_Gluon.py)

2. GoogLeNet (https://arxiv.org/abs/1409.4842)  
[tf.keras](03_Advance/CNN/GoogLeNet/tf_keras.py), 
[PyTorch](03_Advance/CNN/GoogLeNet/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/GoogLeNet/MXNet_Gluon.py)

3. ResNet (https://arxiv.org/abs/1512.03385)  
[tf.keras](03_Advance/CNN/ResNet/tf_keras.py), 
[PyTorch](03_Advance/CNN/ResNet/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/ResNet/MXNet_Gluon.py)

4. Inception V2 (https://arxiv.org/abs/1512.00567)  
[tf.keras](03_Advance/CNN/InceptionV2/tf_keras.py), 
[PyTorch](03_Advance/CNN/InceptionV2/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/InceptionV2/MXNet_Gluon.py)

5. Inception V3 (https://arxiv.org/abs/1512.00567)  
[tf.keras](03_Advance/CNN/InceptionV3/tf_keras.py), 
[PyTorch](03_Advance/CNN/InceptionV3/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/InceptionV3/MXNet_Gluon.py)

6. DenseNet (https://arxiv.org/abs/1608.06993)  
[tf.keras](03_Advance/CNN/DenseNet/tf_keras.py), 
[PyTorch](03_Advance/CNN/DenseNet/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/DenseNet/MXNet_Gluon.py)

7. Xception (https://arxiv.org/abs/1610.02357)  
[tf.keras](03_Advance/CNN/Xception/tf_keras.py), 
[PyTorch](03_Advance/CNN/Xception/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/Xception/MXNet_Gluon.py)

8. MobileNet V1 (https://arxiv.org/abs/1704.04861)   
[tf.keras](03_Advance/CNN/MobileNetV1/tf_keras.py), 
[PyTorch](03_Advance/CNN/MobileNetV1/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/MobileNetV1/MXNet_Gluon.py)

9. MobileNet V2 (https://arxiv.org/abs/1801.04381)   
[tf.keras](03_Advance/CNN/MobileNetV2/tf_keras.py), 
[PyTorch](03_Advance/CNN/MobileNetV2/PyTorch.py), 
[MXNet Gluon](03_Advance/CNN/MobileNetV2/MXNet_Gluon.py)

10. MobileNet V3 (https://arxiv.org/abs/1905.02244)   
[tf.keras](03_Advance/CNN/MobileNetV3/tf_keras.py), 
[PyTorch](03_Advance/CNN/MobileNetV3/PyTorch.py)

11. SqueezeNet (https://arxiv.org/abs/1602.07360)  
[tf.keras](03_Advance/CNN/SqueezeNet/tf_keras.py), 
[PyTorch](03_Advance/CNN/SqueezeNet/PyTorch.py)

12. SENet (https://arxiv.org/abs/1709.01507)  
[tf.keras](03_Advance/CNN/SENet/tf_keras.py), 
[PyTorch](03_Advance/CNN/SENet/PyTorch.py)

</details>

#### Segmentation

<details>
<summary> Contents </summary>

1. DeconvNet (http://cvlab.postech.ac.kr/research/deconvnet/)  
[PyTorch](03_Advance/Segmentation/DeconvNet/PyTorch.py)

2. U-Net (https://arxiv.org/abs/1505.04597)  
[tf.keras](03_Advance/Segmentation/U-Net/tf_keras.py), 
[PyTorch](03_Advance/Segmentation/U-Net/PyTorch.py)

</details>

#### Generative Adversarial Network

<details>
<summary> Contents </summary>

1. Vanilla GAN  
[tf.keras](03_Advance/GAN/Vanilla_GAN/tf_keras.py), 
[PyTorch](03_Advance/GAN/Vanilla_GAN/PyTorch.py)

2. LSGAN  
[tf.keras](03_Advance/GAN/LSGAN/tf_keras.py), 
[PyTorch](03_Advance/GAN/LSGAN/PyTorch.py)

3. DCGAN  
[tf.keras](03_Advance/GAN/DCGAN/tf_keras.py),  
[PyTorch](03_Advance/GAN/DCGAN/PyTorch.py)

4. CGAN  
[tf.keras](03_Advance/GAN/CGAN/tf_keras.py),  
[PyTorch](03_Advance/GAN/CGAN/PyTorch.py)

</details>

### 04 Extra

#### Data Loading

<details>
<summary> Contents </summary>

[PyTorch](04_Extra/DataLoading/PyTorch)

[TensorFlow] ( Not Yet )

</details>

#### Transfer Learning ( Not Yet )

<details>
<summary> Contents </summary>

</details>

#### Super Resolution

<details>
<summary> Contents </summary>

1. SRCNN  
[TensorFlow](04_Extra/Super_Resolution/EDSR/TensorFlow), 
[PyTorch](04_Extra/Super_Resolution/SRCNN/PyTorch)

2. VDSR  
[TensorFlow](04_Extra/Super_Resolution/EDSR/TensorFlow), 
[PyTorch](04_Extra/Super_Resolution/VDSR/PyTorch)

3. EDSR  
[TensorFlow](04_Extra/Super_Resolution/EDSR/TensorFlow), 
[PyTorch](04_Extra/Super_Resolution/EDSR/PyTorch)

4. SubPixel  
[TensorFlow](04_Extra/Super_Resolution/EDSR/TensorFlow), 
[PyTorch](04_Extra/Super_Resolution/SubPixel/PyTorch)

</details>

#### Image Translation

<details>
<summary> Contents </summary>

1. Neural Style Transfer  
[PyTorch](04_Extra/Style_Transfer/Neural_Style_Transfer/PyTroch/)

2. Pix2Pix

3. CycleGAN

</details>

#### Attention Module

<details>
<summary> Contents </summary>

1. [BAM](https://arxiv.org/abs/1807.06514)


2. [CBAM](https://arxiv.org/abs/1807.06521)


3. [Transformer](https://arxiv.org/abs/1706.03762)

</details>