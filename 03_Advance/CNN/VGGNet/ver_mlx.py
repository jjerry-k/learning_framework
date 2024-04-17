from tqdm import tqdm

import numpy as np

from mlx import nn
from mlx import core as mx
from mlx import optimizers as optim

from mlx.data import datasets

np.random.seed(777)
mx.random.seed(777)


EPOCHS = 5
BATCH_SIZE = 64
IMG_SIZE = 64
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
class Model(nn.Module):
    def __init__(self, input_channel= 3, num_classes=1000, num_layer=16):
        super().__init__()
        
        blocks_dict = {
        11: [1, 1, 2, 2, 2],
        13: [2, 2, 2, 2, 2], 
        16: [2, 2, 3, 3, 3], 
        19: [2, 2, 4, 4, 4]
        }

        num_channel_list = [64, 128, 256, 512, 512]

        assert num_layer in  blocks_dict.keys(), "Number of layer must be in %s"%blocks_dict.keys()

        layer_list = []

        input_features = input_channel
        for idx, num_iter in enumerate(blocks_dict[num_layer]):
            for _ in range(num_iter):
                layer_list.append(nn.Conv2d(input_features, num_channel_list[idx], 3, padding=1))
                layer_list.append(nn.ReLU())
                input_features = num_channel_list[idx]
            layer_list.append(nn.MaxPool2d(2, 2))

        self.vgg = nn.Sequential(*layer_list)
        self.classifier = nn.Linear(512, num_classes)
    
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
        x = self.vgg(x)
        x = mx.mean(x, axis=[1, 2]).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

def loss_fn(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))

def eval_fn(model, x, y):
    output = model(x)
    loss = mx.mean(nn.losses.cross_entropy(output, y))
    metric = mx.mean(mx.argmax(model(x), axis=1) == y)
    return loss, metric

model  = Model(input_channel=3, num_classes=5, num_layer=11)
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