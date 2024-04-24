from tqdm import tqdm

import numpy as np

from mlx import nn
from mlx import core as mx
from mlx import optimizers as optim

from mlx.data import datasets

np.random.seed(777)
mx.random.seed(777)


EPOCHS = 5
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
class DenseLayer(nn.Module):
    def __init__(self, input_feature, growth_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm(input_feature),
            nn.ReLU(),
            nn.Conv2d(input_feature, growth_rate * 4, 1),
            nn.BatchNorm(growth_rate * 4),
            nn.ReLU(),
            nn.Conv2d(growth_rate * 4, growth_rate, 3, padding=1)
        )

    def __call__(self, x):
        new_features = self.block(x)
        return mx.concatenate([x, new_features], axis=-1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_feature, growth_rate):
        super().__init__()

        layer_list = []
        for i in range(num_layers):
            layer_list.append(DenseLayer(input_feature + (i * growth_rate), growth_rate))

        self.block = nn.Sequential(*layer_list)

    def __call__(self, x):
        return self.block(x)
            
class Transitionlayer(nn.Module):
    def __init__(self, input_feature, reduction):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm(input_feature), 
            nn.ReLU(),
            nn.Conv2d(input_feature, int(input_feature * reduction), kernel_size=1),
            nn.AvgPool2d(2, 2)
        )

    def __call__(self, x):
        return self.block(x)

class Model(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000, num_blocks=121, growth_rate=32):
        super().__init__()

        blocks_dict = {
        121: [6, 12, 24, 16],
        169: [6, 12, 32, 32], 
        201: [6, 12, 48, 32], 
        264: [6, 12, 64, 48]
    }

        assert num_blocks in  blocks_dict.keys(), "Number of layer must be in %s"%blocks_dict.keys()

        self.Stem = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, 2, 3),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        
        layer_list = []
        num_features = 64
        
        for idx, layers in enumerate(blocks_dict[num_blocks]):
            layer_list.append(DenseBlock(layers, num_features, growth_rate))
            num_features = num_features + (layers * growth_rate)
            if idx != 3:
                layer_list.append(Transitionlayer(num_features, 0.5))
                num_features = int(num_features * 0.5)

        self.Main_Block = nn.Sequential(*layer_list)

        self.Block = nn.Sequential(
            nn.BatchNorm(num_features),
            nn.ReLU()
        )
        self.Classifier = nn.Linear(num_features, num_classes)
    
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
        x = self.Main_Block(x)
        x = mx.mean(x, axis=[1, 2]).reshape(x.shape[0], -1)
        x = self.Block(x)
        x = self.Classifier(x)
        return x

def loss_fn(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))

def eval_fn(model, x, y):
    output = model(x)
    loss = mx.mean(nn.losses.cross_entropy(output, y))
    metric = mx.mean(mx.argmax(model(x), axis=1) == y)
    return loss, metric

model  = Model(input_channel=3, num_classes=5, num_blocks=121, growth_rate=32)
mx.eval(model.parameters())

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=LEARNING_RATE)
# %%
for epoch in range(EPOCHS):
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