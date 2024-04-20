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
class InceptionModule(nn.Module):
    def __init__(self, input_feature, 
                filters_b1, filters_b2_1, filters_b2_2, 
                filters_b3_1, filters_b3_2, filters_b4):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 1), 
            nn.ReLU())

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.ReLU(),
            nn.Conv2d(filters_b2_1, filters_b2_2, 3, 1, 1),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.ReLU(),
            nn.Conv2d(filters_b3_1, filters_b3_2, 5, 1, 2),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
            nn.ReLU()
        )

    def __call__(self, x):
        return mx.concatenate([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], axis=-1)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, input_feature, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(5, 3, 1),
            nn.Conv2d(input_feature, 128, 1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(128, num_classes)
    
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
        x = self.block(x)
        x = mx.mean(x, axis=[1, 2]).reshape(x.shape[0], -1)
        x = self.linear(x)
        return x

        
class Model(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000):
        super().__init__()

        self.Stem = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm(64),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm(192),
            nn.MaxPool2d(3, 2)
        )
        
        self.pool = nn.MaxPool2d(3, 2)

        self.inception1 = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception3 = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.aux1 = AuxiliaryClassifier(512, num_classes)
        self.inception4 = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception5 = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception6 = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.aux2 = AuxiliaryClassifier(528, num_classes)
        self.inception7 = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception8 = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception9 = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
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
        x = self.pool(x)
        
        x = self.inception3(x)
        aux1 = self.aux1(x)

        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        aux2 = self.aux2(x)
        
        x = self.inception7(x)
        x = self.pool(x)

        x = self.inception8(x)
        x = self.inception9(x)

        x = mx.mean(x, axis=[1, 2]).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x, aux1, aux2

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