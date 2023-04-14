import os

import sys

import numpy as np

import tensorflow as tf

from keras.models import Model

from keras.layers import Input, Conv2D, Dense, Flatten, Lambda

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

# Load the pre-trained VGG-16 model

vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)

# Define the loss function

def style_loss(y_true, y_pred):

    # Calculate the Gram matrix for the style image

    G_style = gram_matrix(y_true)

    # Calculate the Gram matrix for the generated image

    G_generated = gram_matrix(y_pred)

    # Calculate the style loss

    style_loss = tf.reduce_mean(tf.square(G_style - G_generated))

    return style_loss

# Define the content loss

def content_loss(y_true, y_pred):

    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define the total loss

def total_loss(y_true, y_pred):

    # Weight the content and style losses

    alpha = 0.01

    beta = 1.0

    # Calculate the content loss

    content_loss = content_loss(y_true[:, :, :, :3], y_pred[:, :, :, :3])

    # Calculate the style loss

    style_loss = style_loss(y_true[:, :, :, 3:], y_pred[:, :, :, 3:])

    # Calculate the total loss

    total_loss = alpha * content_loss + beta * style_loss

    return total_loss

# Define the input layer

input_layer = Input(shape=(256, 256, 3))

# The VGG-16 model takes an input image of size 224x224x3

# We need to resize the input image to 224x224

x = Lambda(lambda x: tf.image.resize(x, (224, 224)))(input_layer)

# Pass the input image through the VGG-16 model

features = vgg16(x)

# Extract the content layer

content_layer = features[-1]
# Extract the style layers

style_layers = [features[i] for i in [1, 6, 11, 16, 21]]

# Define the output layer

output_layer = Conv2D(filters=3, kernel_size=(3, 3), activation='relu')(content_layer)

# Create the style transfer model

model = Model(input_layer, output_layer)

# Compile the model

model.compile(optimizer=Adam(lr=0.001), loss=total_loss)

# Load the content image

content_image = tf.io.read_file('content.jpg')

content_image = tf.image.decode_jpeg(content_image)

content_image = tf.image.resize(content_image, (256, 256))

content_image = tf.expand_dims(content_image, axis=0)

# Load the style image

style_image = tf.io.read_file('style.jpg')

style_image = tf.image.decode_jpeg(style_image)

style_image = tf.image.resize(style_image, (256, 256))

style_image = tf.expand_dims(style_image, axis=0)
# Generate the stylized image

stylized_image = model.predict(content_image)

# Save the stylized image

tf.io.write_file('stylized_image.jpg', stylized_image)

# Display the stylized image

plt.imshow(stylized_image)

plt.show()

# Allow the user to select a different content image

content_image_path = input("Enter the path to the content image: ")

content_image = tf.io.read_file(content_image_path)

content_image = tf.image.decode_jpeg(content_image)

content_image = tf.image.resize(content_image, (256, 256))

content_image = tf.expand_dims(content_image, axis=0)

# Allow the user to select a different style image

style_image_path = input("Enter the path to the style image: ")

style_image = tf.io.read_file(style_image_path)

style_image = tf.image.decode_jpeg(style_image)

style_image = tf.image.resize(style_image, (256, 256))

style_image = tf.expand_dims(style_image, axis=0)

# Train the model

model.fit(content_image, style_image, epochs=10, steps_per_epoch=100, verbose=1)

# Generate the stylized image

stylized_image = model.predict(content_image)

# Save the stylized image

tf.io.write_file('stylized_image.jpg', stylized_image)

# Display the stylized image

plt.imshow(stylized_image)

plt.show()
# Allow the user to continue generating stylized images

while True:

    # Ask the user if they want to generate another stylized image

    generate_another_image = input("Do you want to generate another stylized image? (y/n) ")

    # If the user says yes, generate another stylized image

    if generate_another_image == "y":

        # Allow the user to select a different content image

        content_image_path = input("Enter the path to the content image: ")

        content_image = tf.io.read_file(content_image_path)

        content_image = tf.image.decode_jpeg(content_image)

        content_image = tf.image.resize(content_image, (256, 256))

        content_image = tf.expand_dims(content_image, axis=0)

        # Allow the user to select a different style image

        style_image_path = input("Enter the path to the style image: ")

        style_image = tf.io.read_file(style_image_path)

        style_image = tf.image.decode_jpeg(style_image)

        style_image = tf.image.resize(style_image, (256, 256))

        style_image = tf.expand_dims(style_image, axis=0)

        # Train the model

        model.fit(content_image, style_image, epochs=10, steps_per_epoch=100, verbose=1)

        # Generate the stylized image

        stylized_image = model.predict(content_image)

        # Save the stylized image

        tf.io.write_file('stylized_image.jpg', stylized_image)

        # Display the stylized image

        plt.imshow(stylized_image)

        plt.show()

    else:

        break
        
