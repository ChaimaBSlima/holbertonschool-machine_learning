#!/usr/bin/env python3
""" Task 4 : 4. Layer Style Cost """
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
        self.gram_style_features, self.content_feature = \
            self.generate_features()

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

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the Gram matrix of an input tensor (layer).

        The Gram matrix is used in style transfer to measure the correlations
        between the features in a given layer of the neural network. It is
        computed by reshaping the input tensor into a 2D matrix and then
        performing matrix multiplication. The result is normalized by the
        total number of elements in the feature map.

        Args:
            input_layer (tf.Tensor or tf.Variable):
                A 4D tensor of shape
                (batch_size, height, width, channels) representing a layer of
                the neural network.

        Returns:
            tf.Tensor: The Gram matrix of shape (1, channels, channels),
            normalized by the number of elements (height * width) in the
            feature map.

        Raises:
            TypeError:
                If input_layer is not a tensor or variable of rank 4.
        """
        if not isinstance(
                input_layer,
                (tf.Tensor, tf.Variable)
                ) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Get the dimensions
        _, h, w, c = input_layer.shape

        # Reshape the tensor to a 2D matrix
        input_layer_reshaped = tf.reshape(input_layer, (h * w, c))

        # Compute the gram matrix
        gram = tf.matmul(input_layer_reshaped,
                         input_layer_reshaped,
                         transpose_a=True)

        # Normalize the gram matrix
        gram_matrix = tf.expand_dims(gram / tf.cast(h * w, tf.float32), axis=0)

        return gram_matrix

    def generate_features(self):
        """
        Extracts and processes style and content features from the style
        and content images using the VGG19 model.

        This method preprocesses the style and content images and feeds them
        into the loaded VGG19 model. For the style image, it computes the
        Gram matrices of the style features from the designated layers.
        For the content image, it extracts the features from the content layer.

        The final Gram matrices (excluding the last style layer) are stored in
        `self.gram_style_features`, while the content features from the last
        content layer are stored in `self.content_feature`.

        Returns:
            tuple: A tuple containing:
                - gram_style_features (list of tf.Tensor): List of
                matrices for the style features extracted from the
                style image.
                - content_feature (tf.Tensor): The feature map extracted
                 from the last content layer of the content image.
        """
        # preprocess style and content image
        preprocess_style = (tf.keras.applications.vgg19.
                            preprocess_input(self.style_image * 255))
        preprocess_content = (
                tf.keras.applications.vgg19.
                preprocess_input(self.content_image * 255))

        # get style and content outputs from VGG19 model
        style_output = self.model(preprocess_style)
        content_output = self.model(preprocess_content)

        # compute Gram matrices for style features
        self.gram_style_features = [self.gram_matrix(style_layer) for
                                    style_layer in style_output]

        # excluding the last element considered more suitable for capturing
        # the style of image
        self.gram_style_features = self.gram_style_features[:-1]

        # select only last network layer
        self.content_feature = content_output[-1]

        return self.gram_style_features, self.content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
    Calculates the style cost for a single layer in the style transfer process.

    The style cost measures the difference between the Gram matrix of the
    layer's style output and the Gram matrix of the target style.
    The smaller the difference, the more closely the current image resembles
    the target style.

        Args:
        style_output (tf.Tensor or tf.Variable): A 4D tensor representing
        the style features output from the current layer of the neural
        network. Must have shape (batch_size, height, width, channels).
        gram_target (tf.Tensor or tf.Variable): The Gram matrix of the target
        style for the given layer, with shape (1, channels, channels).

        Returns:
            tf.Tensor: A scalar tensor representing the style cost for the
            given layer, which is the mean squared difference between
            the current;layer's Gram matrix and the target Gram matrix.

        Raises:
            TypeError: If `style_output` is not a tensor of rank 4,
            or if `gram_target` is not a tensor of the shape
            (1, channels, channels).
        """

        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != [1, c, c]):
            raise TypeError(
                    "gram_target must be a tensor of shape [1, {}, {}]".format(
                        c,
                        c
                        ))

        output_gram_style = self.gram_matrix(style_output)

        # difference between two gram matrix
        layer_style_cost = tf.reduce_mean(
                tf.square(output_gram_style - gram_target))

        return layer_style_cost
