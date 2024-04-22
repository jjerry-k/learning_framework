from tqdm import tqdm

import numpy as np

from mlx import nn
from mlx import core as mx
from mlx import optimizers as optim

from mlx.data import datasets

np.random.seed(777)
mx.random.seed(777)


EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 192
LEARNING_RATE = 1e-4

train_dataset = datasets.load_images_from_folder("../../../data/flower_photos/train")
val_dataset = datasets.load_images_from_folder("../../../data/flower_photos/validation")

mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

def normalize(x):
    return (x - mean) / std

train_loader = (
        train_dataset.shuffle()
        .to_stream()
        .image_resize("image", IMG_SIZE, IMG_SIZE)
        .key_transform("image", normalize)
        .batch(BATCH_SIZE)
    )
val_loader = (
        val_dataset
        .to_stream()
        .image_resize("image", IMG_SIZE, IMG_SIZE)
        .key_transform("image", normalize)
        .batch(BATCH_SIZE)
    )

len_train_loader = int(np.ceil(len(train_dataset)/BATCH_SIZE))
len_val_loader = int(np.ceil(len(val_dataset)/BATCH_SIZE))

# Defining Model
class Inception_Module_A(nn.Module):
    def __init__(self, input_feature, 
                filters_b1, filters_b2_1, filters_b2_2, 
                filters_b3_1, filters_b3_2, filters_b3_3, filters_b4):
        super(Inception_Module_A, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 1), 
            nn.BatchNorm(filters_b1),
            nn.ReLU())

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm(filters_b2_1),
            nn.ReLU(),
            nn.Conv2d(filters_b2_1, filters_b2_2, 3, 1, 1),
            nn.BatchNorm(filters_b2_2),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.BatchNorm(filters_b3_1),
            nn.ReLU(),
            nn.Conv2d(filters_b3_1, filters_b3_2, 3, 1, 1),
            nn.BatchNorm(filters_b3_2),
            nn.ReLU(),
            nn.Conv2d(filters_b3_2, filters_b3_3, 3, 1, 1),
            nn.BatchNorm(filters_b3_3),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
            nn.BatchNorm(filters_b4),
            nn.ReLU()
        )

    def __call__(self, x):
        return mx.concatenate([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], axis=-1)


class Inception_Module_B(nn.Module):
    def __init__(self, input_feature, 
                filters_b1, filters_b2_1, filters_b2_2, filters_b2_3, 
                filters_b3_1, filters_b3_2, filters_b3_3, filters_b3_4, filters_b3_5, 
                filters_b4):
        super(Inception_Module_B, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 1), 
            nn.BatchNorm(filters_b1),
            nn.ReLU())

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm(filters_b2_1),
            nn.ReLU(),
            nn.Conv2d(filters_b2_1, filters_b2_2, (7, 1), 1, (3, 0)),
            nn.BatchNorm(filters_b2_2),
            nn.ReLU(),
            nn.Conv2d(filters_b2_2, filters_b2_3, (1, 7), 1, (0, 3)),
            nn.BatchNorm(filters_b2_3),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.BatchNorm(filters_b3_1),
            nn.ReLU(),
            nn.Conv2d(filters_b3_1, filters_b3_2, (7, 1), 1, (3, 0)),
            nn.BatchNorm(filters_b3_2),
            nn.ReLU(),
            nn.Conv2d(filters_b3_2, filters_b3_3, (1, 7), 1, (0, 3)),
            nn.BatchNorm(filters_b3_3),
            nn.ReLU(),
            nn.Conv2d(filters_b3_3, filters_b3_4, (7, 1), 1, (3, 0)),
            nn.BatchNorm(filters_b3_4),
            nn.ReLU(),
            nn.Conv2d(filters_b3_4, filters_b3_5, (1, 7), 1, (0, 3)),
            nn.BatchNorm(filters_b3_5),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
            nn.BatchNorm(filters_b4),
            nn.ReLU()
        )

    def __call__(self, x):
        return mx.concatenate([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], axis=-1)

class Inception_Module_C(nn.Module):
    def __init__(self, input_feature, 
                filters_b1, filters_b2_1, filters_b2_2, filters_b2_3, 
                filters_b3_1, filters_b3_2, filters_b3_3, filters_b3_4, 
                filters_b4):
        super(Inception_Module_C, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 1), 
            nn.BatchNorm(filters_b1),
            nn.ReLU())

        self.branch2_block_1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm(filters_b2_1),
            nn.ReLU()
        )

        self.branch2_block_2_1 = nn.Sequential(
            nn.Conv2d(filters_b2_1, filters_b2_2, (1, 3), 1, (0, 1)),
            nn.BatchNorm(filters_b2_2),
            nn.ReLU()
        )

        self.branch2_block_2_2 = nn.Sequential(
            nn.Conv2d(filters_b2_1, filters_b2_3, (3, 1), 1, (1, 0)),
            nn.BatchNorm(filters_b2_3),
            nn.ReLU()
        )

        self.branch3_block_1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.BatchNorm(filters_b3_1),
            nn.ReLU(),
            nn.Conv2d(filters_b3_1, filters_b3_2, 3, 1, 1),
            nn.BatchNorm(filters_b3_2),
            nn.ReLU()
        )

        self.branch3_block_2_1 = nn.Sequential(
            nn.Conv2d(filters_b3_2, filters_b3_3, (1, 3), 1, (0, 1)),
            nn.BatchNorm(filters_b3_3),
            nn.ReLU()
        )
        
        self.branch3_block_2_2 = nn.Sequential(
            nn.Conv2d(filters_b3_2, filters_b3_4, (3, 1), 1, (1, 0)),
            nn.BatchNorm(filters_b3_4),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
            nn.BatchNorm(filters_b4),
            nn.ReLU()
        )

    def __call__(self, x):
        block1 = self.branch1(x)
        
        block2 = self.branch2_block_1(x)
        block2 = mx.concatenate([self.branch2_block_2_1(block2), self.branch2_block_2_2(block2)], axis=-1)

        block3 = self.branch3_block_1(x)
        block3 = mx.concatenate([self.branch3_block_2_1(block3), self.branch3_block_2_2(block3)], axis=-1)

        block4 = self.branch4(x)

        return mx.concatenate([block1, block2, block3, block4], axis=-1)


class Grid_Reduction_1(nn.Module):
    def __init__(self, input_feature, filters_b1, 
                filters_b2_1, filters_b2_2, filters_b2_3):
        super(Grid_Reduction_1, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 3, 2),
            nn.BatchNorm(filters_b1),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm(filters_b2_1),
            nn.ReLU(),
            nn.Conv2d(filters_b2_1, filters_b2_2, 3, 1, 1),
            nn.BatchNorm(filters_b2_2),
            nn.ReLU(),
            nn.Conv2d(filters_b2_2, filters_b2_3, 3, 2),
            nn.BatchNorm(filters_b2_3),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, 2)
        )
    def __call__(self, x):
        return mx.concatenate([self.branch1(x), self.branch2(x), self.branch3(x)], axis=-1)

class Grid_Reduction_2(nn.Module):
    def __init__(self, input_feature, filters_b1_1, filters_b1_2, 
                filters_b2_1, filters_b2_2, filters_b2_3, filters_b2_4):
        super(Grid_Reduction_2, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1_1, 1),
            nn.BatchNorm(filters_b1_1),
            nn.ReLU(),
            nn.Conv2d(filters_b1_1, filters_b1_2, 3, 2),
            nn.BatchNorm(filters_b1_2),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm(filters_b2_1),
            nn.ReLU(),
            nn.Conv2d(filters_b2_1, filters_b2_2, (1, 7), 1, (0, 3)),
            nn.BatchNorm(filters_b2_2),
            nn.ReLU(),
            nn.Conv2d(filters_b2_2, filters_b2_3, (7, 1), 1, (3, 0)),
            nn.BatchNorm(filters_b2_3),
            nn.ReLU(),
            nn.Conv2d(filters_b2_3, filters_b2_4, 3, 2),
            nn.BatchNorm(filters_b2_4),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, 2)
        )
    def __call__(self, x):
        return mx.concatenate([self.branch1(x), self.branch2(x), self.branch3(x)], axis=-1)

class Auxiliary_Classifier(nn.Module):
    def __init__(self, input_feature, num_classes):
        super(Auxiliary_Classifier, self).__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(5, 3, 1),
            nn.Conv2d(input_feature, 128, 1),
            nn.BatchNorm(128),
            nn.ReLU(),
        )
        self.linear = nn.Linear(128, num_classes)
    
    def __call__(self, x):
        x = self.block(x)
        x = mx.mean(x, axis=[1, 2]).reshape(x.shape[0], -1)
        x = self.linear(x)
        return x

        
class Model(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000):
        super().__init__()

        self.Stem = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, 2),
            nn.BatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(64, 80, 1),
            nn.BatchNorm(80),
            nn.ReLU(),
            nn.Conv2d(80, 192, 3, 1),
            nn.BatchNorm(192),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        
        self.inception1 = Inception_Module_A(192, 64, 48, 64, 64, 96, 96, 64)
        self.inception2 = Inception_Module_A(288, 64, 48, 64, 64, 96, 96, 64)
        self.inception3 = Inception_Module_A(288, 64, 48, 64, 64, 96, 96, 64)
        self.grid_reduction1 = Grid_Reduction_1(288, 384, 64, 96, 96)    

        self.inception4 = Inception_Module_B(768, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192)
        self.inception5 = Inception_Module_B(768, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192)
        self.inception6 = Inception_Module_B(768, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192)
        self.inception7 = Inception_Module_B(768, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192)
        self.aux = Auxiliary_Classifier(768, num_classes)

        self.grid_reduction2 = Grid_Reduction_2(768, 192, 320, 192, 192, 192, 192)
        self.inception8 = Inception_Module_C(1280, 320, 384, 384, 384, 448, 384, 384, 384, 192)
        self.inception9 = Inception_Module_C(2048, 320, 384, 384, 384, 448, 384, 384, 384, 192)

        self.Classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes)
        )
    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.he_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.he_normal(m.weight)
                nn.init.constant(m.bias, 0)
                
    def __call__(self, x):
        x = self.Stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.grid_reduction1(x)
        
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        aux = self.aux(x)

        x = self.grid_reduction2(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = mx.mean(x, axis=[1, 2]).reshape(x.shape[0], -1)
        x = self.Classifier(x)
        return x, aux

def loss_fn(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x)[0], y))

def eval_fn(model, x, y):
    output = model(x)[0]
    loss = mx.mean(nn.losses.cross_entropy(output, y))
    metric = mx.mean(mx.argmax(output, axis=1) == y)
    return loss, metric

model  = Model(input_channel=3, num_classes=5)
mx.eval(model.parameters())

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    with tqdm(enumerate(train_loader), total=len_train_loader) as pbar:
        pbar.set_description(f"{epoch+1}/{EPOCHS}")
        for i, (batch) in pbar:
            batch_x = mx.array(batch["image"])
            batch_y = mx.array(batch["label"])
            batch_loss, batch_grads = loss_and_grad_fn(model, batch_x, batch_y)
            optimizer.update(model, batch_grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss += batch_loss.item()
            pbar.set_postfix(loss=f"{train_loss/(i+1):.3f}")
    val_loss = 0
    val_acc = 0
    model.eval()
    with tqdm(enumerate(val_loader), total=len_val_loader) as pbar:
        pbar.set_description(f"{epoch+1}/{EPOCHS}")
        for i, (batch) in pbar:
            batch_x = mx.array(batch["image"])
            batch_y = mx.array(batch["label"])
            batch_loss, batch_accuracy = eval_fn(model, batch_x, batch_y)
            val_loss += batch_loss.item()
            val_acc += batch_accuracy.item()
            pbar.set_postfix(val_loss=f"{val_loss/(i+1):.3f}")
        val_acc /= len_val_loader
    pbar.set_postfix(val_loss=f"{val_loss/(i+1):.3f}", val_acc=f"{val_acc:.3f}")
    print(f"{epoch+1}/{EPOCHS}: Train Loss: {train_loss/len_train_loader:.3f}, Val Loss: {val_loss/(i+1):.3f}, Val Accuracy: {val_acc:.3f}\n")    
            
    train_loader.reset()
    val_loader.reset()