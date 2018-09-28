# Using VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as VGG16_preprocess_input
from quiver_engine.server import launch

model = VGG16()

launch(model, input_folder='./')