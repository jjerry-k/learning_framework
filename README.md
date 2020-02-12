# Learning Python A.I Framework

- 본 Repository 는 지극히 개인적인 공부용이라 모든 코드들이 불친절하고 가독성이 좋지 않습니다.
- 대부분 Network만 구성해볼뿐 학습을 해보진 않습니다. (물론 추후에 학습도 해볼 예정)

---
## Framework list 

### [Tensorflow](/tensorflow/)
- 배포 : Google
- 공식 홈페이지 : https://www.tensorflow.org
- 주 사용 프레임워크
- Low level(tf.nn), High level(tf.keras) 로 작성하려함.

### [Pytorch](/pytorch/)
- 배포 : Facebook
- 공식 홈페이지 : https://pytorch.org
- 서브 프레임워크

### [MXNet](/mxnet/)
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

- Simple Convolutional Neural Network  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Simple_Convolutional_Neural_Network/tf_keras.py), 
[tf.nn](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Simple_Convolutional_Neural_Network/tf_nn.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/02_Intermediate/Simple_Convolutional_Neural_Network/PyTorch.py)

### 03 Advance
#### Advance Convolutional Neural Network
1. VGGNet  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/VGGNet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/VGGNet/PyTorch.py)

2. ResNet  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/ResNet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/ResNet/PyTorch.py)

3. DenseNet  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/DenseNet/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/DenseNet/PyTorch.py)

4. Xception  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/Xception/tf_keras.py)

5. MobileNet  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/CNN/MobileNetV1/tf_keras.py)

#### Segmentation
1. U-Net  
[tf.keras](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/Segmentation/U-Net/tf_keras.py), 
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/Segmentation/U-Net/PyTorch.py)


#### Generative Adversarial Network
1. Vanilla GAN  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/Vanilla_GAN/PyTorch.py)
2. LSGAN  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/LSGAN/PyTorch.py)
3. DCGAN  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/DCGAN/PyTorch.py)
4. CGAN  
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/03_Advance/GAN/CGAN/PyTorch.py)


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
[PyTorch](https://github.com/jjerry-k/learning_framework/blob/master/04_Extra/Style_Transfer/PyTroch/)Upda