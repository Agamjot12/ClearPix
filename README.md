# ClearPix
Image Super Resolution Using Autoencoders in Keras.

This project demonstrates Image Super Resolution using Autoencoders. The goal of Image Super Resolution is to generate high-resolution images from low-resolution input images. Autoencoders are a type of neural network that is commonly used for unsupervised learning tasks.

## Project Overview

The project is implemented using the TensorFlow and Keras libraries in Python. It consists of two main components: Encoder and Decoder, which together form the Autoencoder architecture.

### Encoder

The Encoder takes a low-resolution input image (256x256x3) as input and applies several convolutional and max-pooling layers to extract relevant features from the image. The Encoder is designed to reduce the spatial dimensions of the input image while increasing the number of channels. It ends with a convolutional layer producing a (None, 64, 64, 256) shaped output.

### Decoder

The Decoder takes the output of the Encoder as input and aims to reconstruct the high-resolution image. It uses Conv2DTranspose layers to upscale the feature maps, thereby increasing the spatial resolution of the output. The Decoder combines skip connections with the feature maps from the Encoder to retain important details during the upsampling process. The final output of the Decoder is a high-resolution image (256x256x3).

## Training

The Autoencoder is trained on a dataset of low-resolution images and their corresponding high-resolution versions. The dataset is loaded from a directory containing image files. The training process involves minimizing the mean squared error loss between the predicted high-resolution images and the ground truth images.

## Model Predictions and Visualizing the Results

After training the Autoencoder, we load the pre-trained model and use it to upscale low-resolution images. The high-resolution predictions are then visualized alongside the original low-resolution images and the ground truth high-resolution images.

## Usage

To use this project, follow the steps below:

1. Install the required libraries using `pip install tensorflow keras scikit-image matplotlib`.

2. Execute the `train_batches` function to load and preprocess the dataset. Adjust the parameters as needed.

3. Run the training process by calling `autoencoder.fit` with appropriate arguments for training the model.

4. Load the pre-trained model using `autoencoder.load_weights` and generate high-resolution images using `autoencoder.predict`.


