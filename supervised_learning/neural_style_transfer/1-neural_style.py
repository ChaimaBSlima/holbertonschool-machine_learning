#!/usr/bin/env python3
""" Task 1 : 1. Load the Model """
import numpy as np
import tensorflow as tf


class NST:
    """
    Neural Style Transfer (NST) class used to apply the style of
    one image to the content of another.

    Attributes:
        style_layers (list):
        Layers from the pretrained model used to extract style features.
        content_layer (str):
        Layer used to extract content features.
        style_image (numpy.ndarray):
        Scaled style image used for the style transfer.
        content_image (numpy.ndarray): Scaled content image used for
        the style transfer.
        alpha (float):
        Weight for content loss.
        beta (float):
        Weight for style loss.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class with style and content images,
        and loss weights.

        Args:
            style_image (numpy.ndarray):
            Style image to apply, must be of shape (h, w, 3).
            content_image (numpy.ndarray):
            Content image to preserve, must be of shape (h, w, 3).
            alpha (float, optional):
            Weight for content loss, default is 1e4.
            beta (float, optional):
            Weight for style loss, default is 1.

        Raises:
            TypeError: If style_image or content_image are not numpy.
            ndarrays or don't have shape (h, w, 3).
            TypeError: If alpha or beta are not non-negative numbers.
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError("style_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.style_image = self.scale_image(style_image)

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.content_image = self.scale_image(content_image)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        else:
            self.alpha = alpha

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        else:
            self.beta = beta

        self.model = None
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image (numpy.ndarray):
            Image to rescale, must have shape (h, w, 3).

        Returns:
            numpy.ndarray: Rescaled image with its largest side of 512 pixels,
            and pixel values between 0 and 1.

        Raises:
            TypeError: If the input image is not a numpy.ndarray
            or doesn't have shape (h, w, 3).
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise (TypeError
                   ("image must be a numpy.ndarray with shape (h, w, 3)"))

        h, w, _ = image.shape

        if w > h:
            w_new = 512
            h_new = int((h * 512) / w)
        else:
            h_new = 512
            w_new = int((w * 512) / h)

        resized_image = tf.image.resize(image,
                                        size=[h_new, w_new],
                                        method='bicubic')

        # Normalize
        resized_image = resized_image / 255.0

        # Limit pixel values between 0 and 1
        resized_image = tf.clip_by_value(resized_image, 0, 1)

        tf_resize_image = tf.expand_dims(resized_image, 0)

        return tf_resize_image

    def load_model(self):
        """
        Loads and modifies the VGG19 model for neural style transfer.

        This method:
        1. Loads the VGG19 model pre-trained on ImageNet without the top
        fully-connected layers.
        2. Freezes the model's weights to prevent further training.
        3. Extracts the outputs from specific layers used for style and
        content extraction.
        4. Replaces all MaxPooling layers in the model with AveragePooling
        layers.
        5. Saves the modified model and reloads it with AveragePooling
        layers for consistent results.

        The model includes outputs for the layers specified in
        `self.style_layers`
        for style extraction and `self.content_layer` for content extraction.

        Returns:
            None: The method assigns the loaded model to `self.model`.
        """
        # Keras API
        modelVGG19 = tf.keras.applications.VGG19(
                include_top=False,
                weights='imagenet'
                )

        modelVGG19.trainable = False

        # selected layers
        selected_layers = self.style_layers + [self.content_layer]

        outputs = [
                modelVGG19.get_layer(name).output for name in selected_layers
                ]

        # construct model
        model = tf.keras.Model([modelVGG19.input], outputs)

        # for replace MaxPooling layer by AveragePooling layer
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model(
                'vgg_base.h5', custom_objects=custom_objects
                )

        self.model = model_avg
