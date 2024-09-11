import os
import numpy as np
from tqdm import tqdm
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

from keras import Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

####################################################
# Define U-net architecture
####################################################

SIZE = 128

def unet_model(input_layer, start_neurons):
    # Contraction path
    conv1 = Conv2D(start_neurons, kernel_size=(3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons, kernel_size=(3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons*2, kernel_size=(3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons*2, kernel_size=(3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons*4, kernel_size=(3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons*4, kernel_size=(3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons*8, kernel_size=(3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons*8, kernel_size=(3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons*16, kernel_size=(3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons*16, kernel_size=(3, 3), activation="relu", padding="same")(convm)
    
    # Expansive path
    deconv4 = Conv2DTranspose(start_neurons*8, kernel_size=(3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons*8, kernel_size=(3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons*8, kernel_size=(3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons*4, kernel_size=(3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons*4, kernel_size=(3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons*4, kernel_size=(3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons*2, kernel_size=(3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons*2, kernel_size=(3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons*2, kernel_size=(3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons*1, kernel_size=(3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons, kernel_size=(3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons, kernel_size=(3, 3), activation="relu", padding="same")(uconv1)
    
    # Last conv and output
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

# Compile unet model
input_layer = Input((SIZE, SIZE, 3))
output_layer = unet_model(input_layer = input_layer, start_neurons = 16)

model = Model(input_layer, output_layer)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model_name = 'unet2'
weights_path = os.path.join('/models/unet/result/models/', (model_name + '.weights.h5')) 
model.load_weights(weights_path)
print("model: ", weights_path)

image_name = sys.argv[1]
image_path = '/images/' + image_name
print("image: ", image_path)

image = Image.open(image_path)

# Preprocess the image
image = image.resize((SIZE, SIZE))
image_array = np.asarray(image).astype('float32') / 255.0  # Normalize pixel values if your model expects this
image_array = np.expand_dims(image_array, axis=0)

# Perform inference
prediction = model.predict(image_array)

if prediction.shape[-1] == 1:
    predicted_mask = prediction[0, :, :, 0]  # Remove batch dimension and get the mask
    thresholded_mask = (predicted_mask > 0.5).astype(np.uint8)  # Apply a threshold to get a binary mask

original_image = Image.open(image_path)
original_image = original_image.resize((SIZE, SIZE))
original_image_array = np.asarray(original_image).astype('float32') / 255.0

image_for_model = np.expand_dims(original_image_array, axis=0)  # Add batch dimension
predicted_mask = model.predict(image_for_model)[0, :, :, 0]  # Assuming the model outputs a single-channel mask
thresholded_mask = (predicted_mask > 0.5).astype(np.uint8)
red_mask = np.zeros_like(original_image_array)
red_mask[:, :, 0] = 1  # Set the red channel to maximum
colored_image = np.where(thresholded_mask[:, :, np.newaxis] == 1, red_mask, original_image_array)

dpi = 600  # Change the dpi if needed
figsize = (image.size[0]+100 / dpi, image.size[1]+100 / dpi)

plt.figure(figsize=figsize)  # Adjust figure size as needed
plt.imshow(colored_image)
plt.tight_layout(pad=0)
plt.axis('off')  # Hide the axis
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

image_path = '/images/generated-images/' + image_name

if not os.path.exists('/images' + '/generated-images/'):
    os.makedirs('/images' + '/generated-images')
plt.savefig(image_path)
