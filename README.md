# Learning Python A.I Framework

- 본 Repository 는 지극히 개인적인 공부용이라 모든 코드들이 불친절하고 가독성이 좋지 않습니다.
- 대부분 Network만 구성해볼뿐 학습을 해보진 않습니다. (물론 추후에 학습도 해볼 예정)

---
## Framework list 

### TensorFlow
- 배포 : Google
- 공식 홈페이지 : https://www.tensorflow.org
- 주 사용 프레임워크
- Low level(tf.nn), High level(tf.keras), model subclassing API 로 작성하려함.
- 1.x -> 2.x 로 변겅중.

### PyTorch
- 배포 : Facebook
- 공식 홈페이지 : https://pytorch.org
- 서브 프레임워크

### MXNet
- 배포 : Apache
- 공식 홈페이지 : https://mxnet.apache.org
- 그냥...써보려고 함....
- Gluon, Module 로 작성하려함.
- 매우...진행이 느릴 것으로 예상.


## Example List

### 01 Basic Usage
- Linear Regression  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/01_Basic/Linear_Regression/tf_keras.py),
[tf.nn](https://github.com/jjerry-k/learning_framework/blob/master/01_Basic/Linear_Regression/tf_nn.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/01_Basic/Linear_Regression/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/01_Basic/Linear_Regression/MXNet_Gluon.py)

- Logistic Regression  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/01_Basic/Logistic_Regression/tf_keras.py), 
[tf.nn](https://github.com/jjerry-k/learning_framework/blob/master/01_Basic/Logistic_Regression/tf_nn.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/01_Basic/Logistic_Regression/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/01_Basic/Logistic_Regression/MXNet_Gluon.py)

### 02 Intermediate
- Multi Layer Network  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Multi_Layer_Neural_Network/tf_keras.py), 
[tf.nn](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Multi_Layer_Neural_Network/tf_nn.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Multi_Layer_Neural_Network/PyTorch.py) 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Multi_Layer_Neural_Network/MXNet_Gluon.py)

- Simple Convolutional Neural Network  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Simple_Convolutional_Neural_Network/tf_keras.py), 
[tf.nn](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Simple_Convolutional_Neural_Network/tf_nn.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Simple_Convolutional_Neural_Network/PyTorch.py) 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Simple_Convolutional_Neural_Network/MXNet_Gluon.py)

### 03 Advance
#### Advance Convolutional Neural Network
1. VGGNet  (https://arxiv.org/abs/1409.1556)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/VGGNet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/VGGNet/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/VGGNet/MXNet_Gluon.py)

2. GoogLeNet (https://arxiv.org/abs/1409.4842)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/GoogLeNet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/GoogLeNet/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/GoogLeNet/MXNet_Gluon.py)

3. ResNet (https://arxiv.org/abs/1512.03385)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/ResNet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/ResNet/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/ResNet/MXNet_Gluon.py)

4. Inception V2 (https://arxiv.org/abs/1512.00567)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/InceptionV2/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/InceptionV2/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/InceptionV2/MXNet_Gluon.py)

5. Inception V3 (https://arxiv.org/abs/1512.00567)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/InceptionV3/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/InceptionV3/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/InceptionV3/MXNet_Gluon.py)

6. DenseNet (https://arxiv.org/abs/1608.06993)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/DenseNet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/DenseNet/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/DenseNet/MXNet_Gluon.py)

7. Xception (https://arxiv.org/abs/1610.02357)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/Xception/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/Xception/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/Xception/MXNet_Gluon.py)

8. MobileNet V1 (https://arxiv.org/abs/1704.04861)   
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV1/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV1/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV1/MXNet_Gluon.py)

9. MobileNet V2 (https://arxiv.org/abs/1801.04381)   
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV2/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV2/PyTorch.py), 
[MXNet Gluon](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV2/MXNet_Gluon.py)

10. MobileNet V3 (https://arxiv.org/abs/1905.02244)   
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV3/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV3/PyTorch.py)

11. SqueezeNet (https://arxiv.org/abs/1602.07360)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/SqueezeNet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/SqueezeNet/PyTorch.py)

12. SENet (https://arxiv.org/abs/1709.01507)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/SENet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/SENet/PyTorch.py)


#### Segmentation
1. DeconvNet (http://cvlab.postech.ac.kr/research/deconvnet/)  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/Segmentation/DeconvNet/PyTorch.py)

2. U-Net (https://arxiv.org/abs/1505.04597)  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/Segmentation/U-Net/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/Segmentation/U-Net/PyTorch.py)



#### Generative Adversarial Network
1. Vanilla GAN  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/Vanilla_GAN/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/Vanilla_GAN/PyTorch.py)

2. LSGAN  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/LSGAN/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/LSGAN/PyTorch.py)

3. DCGAN  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/DCGAN/tf_keras.py),  [PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/DCGAN/PyTorch.py)

4. CGAN  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/CGAN/tf_keras.py),  [PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/CGAN/PyTorch.py)


### 04 Extra
#### Super Resolution
1. SRCNN  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/04_Extra/Super_Resolution/SRCNN/PyTorch.py)

2. VDSR  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/04_Extra/Super_Resolution/VDSR/PyTorch.py)

3. EDSR  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/04_Extra/Super_Resolution/EDSR/PyTorch.py)

#### Style Transfer
1. Neural Style Transfer  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/04_Extra/Style_Transfer/PyTroch/)

#### Attention Module

1. [BAM](https://arxiv.org/abs/1807.06514)


2. [CBAM](https://arxiv.org/abs/1807.06521)
