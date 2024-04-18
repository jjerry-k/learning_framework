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
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, output_channel, strides=1, use_branch=True):
        super().__init__()

        self.branch1 = lambda x: x
        if use_branch:
            self.branch1 = nn.Conv2d(in_channel, output_channel, 1, strides)
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, output_channel//4, 1, strides),
            nn.BatchNorm(output_channel//4),
            nn.ReLU(),
            nn.Conv2d(output_channel//4, output_channel//4, 3, 1, padding=1),
            nn.BatchNorm(output_channel//4),
            nn.ReLU(),
            nn.Conv2d(output_channel//4, output_channel, 1, 1),
            nn.BatchNorm(output_channel),        
        )

        self.relu = nn.ReLU()

    def __call__(self, x):
        out = self.branch2(x)
        out = self.relu(out + self.branch1(x))

        return out
    
class Model(nn.Module):
    def __init__(self, input_channel= 3, num_classes=1000, num_layer=16):
        super().__init__()
        
        blocks_dict = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3], 
        152: [3, 8, 36, 3]
        }

        num_channel_list = [256, 512, 1024, 2048]

        assert num_layer in  blocks_dict.keys(), "Number of layer must be in %s"%blocks_dict.keys()

        self.stem = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, 2, 3),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        layer_list = []

        input_features = 64

        for idx, num_iter in enumerate(blocks_dict[num_layer]):
            for j in range(num_iter):
                if j==0:
                    layer_list.append(ResidualBlock(input_features, num_channel_list[idx], strides=2))
                else:
                    layer_list.append(ResidualBlock(input_features, num_channel_list[idx], use_branch=False))
                input_features = num_channel_list[idx]
        self.main_net = nn.Sequential(*layer_list)
        self.classifier = nn.Linear(input_features, num_classes)
    
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
        x = self.stem(x)
        x = self.main_net(x)
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

model  = Model(input_channel=3, num_classes=5, num_layer=50)
mx.eval(model.parameters())

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=LEARNING_RATE, momentum=0.99)
# %%
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