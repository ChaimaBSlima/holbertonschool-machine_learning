#!/usr/bin/env python3
""" Task 9 : 9. Generate Image """
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
        self.gram_style_features, self.content_feature = (
                self.generate_features()
                )

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

        resized_image = tf.image.resize(image, size=[h_new, w_new],
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
        modelVGG19 = tf.keras.applications.VGG19(include_top=False,
                                                 weights='imagenet')

        modelVGG19.trainable = False

        # selected layers
        selected_layers = self.style_layers + [self.content_layer]

        outputs = [
                modelVGG19.get_layer(name).output for name in selected_layers
                ]

        # construct model
        model = tf.keras.Model([modelVGG19.input], outputs)

        # replace MaxPooling layer with AveragePooling layer
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model('vgg_base.h5',
                                               custom_objects=custom_objects)

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
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Get the dimensions
        _, h, w, c = input_layer.shape

        # Reshape the tensor to a 2D matrix
        input_layer_reshaped = tf.reshape(input_layer, (h * w, c))

        # Compute the gram matrix
        gram = tf.matmul(input_layer_reshaped, input_layer_reshaped,
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

    def style_cost(self, style_outputs):
        """
    Calculates the total style cost for all specified style layers.

    This method computes the weighted style cost by comparing the
    style features (Gram matrices) of each specified layer in the generated
    image to those of the target style image. Each layer's contribution
    to the total cost is weighted equally.

    Args:
        style_outputs (list): A list of tensors, each representing the style
            features from one of the specified style layers in the generated
            image. The length of this list must match the number of layers
            specified in `self.style_layers`.

    Returns:
        tf.Tensor: A scalar tensor representing the total style cost, which
        is the weighted sum of the individual style costs from each layer.

        """
        len_style_layer = len(self.style_layers)
        if (not isinstance(style_outputs, list)
                or len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(len_style_layer)
            )

        # uniform initialization
        weight = 1.0 / float(len_style_layer)

        cost_total = sum([weight * self.layer_style_cost(style, target)
                         for style, target
                         in zip(style_outputs, self.gram_style_features)])

        return cost_total

    def content_cost(self, content_output):
        """
    Calculates the content cost between the generated image's content
    and the target content image.

    This method measures the difference in content features between the
    generated image and the target content image, using mean squared error.

    Args:
        content_output (tf.Tensor or tf.Variable): A tensor representing the
            content features extracted from the specified content layer
            of the generated image. This tensor must have the same shape
            as `self.content_feature`.

    Returns:
        tf.Tensor: A scalar tensor representing the content cost,
        indicating the similarity between the content features of
        the generated image and the target content image.
        """
        content_feature_shape = self.content_feature.shape

        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".
                format(content_feature_shape))

        content_cost = (
            tf.reduce_mean(tf.square(content_output - self.content_feature)))

        return content_cost

    def total_cost(self, generated_image):
        """
    Computes the total neural style transfer cost for the generated image,
    combining content and style costs.

    This method processes the `generated_image` by passing it through the
    pre-trained model to extract content and style features. It calculates
    the total cost as a weighted sum of the content and style costs using
    the `alpha` and `beta` factors.

    Args:
        generated_image (tf.Tensor): A tensor representing the generated image
            to be evaluated, with the same shape as `self.content_image`.

    Returns:
        tuple: Contains three elements:
            - total_cost (tf.Tensor): A scalar tensor representing the
              weighted sum of content and style costs.
            - content_cost (tf.Tensor): A scalar tensor representing
              the content cost.
            - style_cost (tf.Tensor): A scalar tensor representing
              the style cost.
        """
        shape_content_image = self.content_image.shape
        shape_content_image = self.content_image.shape

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != shape_content_image):
            raise TypeError("generated_image must be a tensor of shape {}"
                            .format(shape_content_image))

        # preprocess generated img
        preprocess_generated_image = \
            (tf.keras.applications.
             vgg19.preprocess_input(generated_image * 255))

        # calculate content and style for generated image
        generated_output = self.model(preprocess_generated_image)

        # def content and style
        generated_content = generated_output[-1]
        generated_style = generated_output[:-1]

        J_content = self.content_cost(generated_content)
        J_style = self.style_cost(generated_style)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Computes the gradients of the total cost with respect to the
        generated image.

    This method leverages TensorFlow's `GradientTape` to compute the gradient
    of the total neural style transfer cost (combining content and style costs)
    with respect to the input `generated_image`. This gradient information is
    essential for optimizing the generated image iteratively.

    Args:
        generated_image (tf.Tensor): A tensor of shape
            (1, height, width, 3), representing the generated image.
            This tensor must have the same spatial dimensions as the
            `content_image`.

    Returns:
        tuple: Contains four elements:
            - grad (tf.Tensor): The computed gradient tensor of the total cost
              with respect to `generated_image`.
            - total_cost (tf.Tensor): A scalar tensor representing the
              total neural style transfer cost.
            - content_cost (tf.Tensor): A scalar tensor representing the
              content cost.
            - style_cost (tf.Tensor): A scalar tensor representing the
              style cost.

    Raises:
        TypeError: If `generated_image` is not a tensor of shape
            (1, height, width, 3).
        """
        # define shape
        shape_content_image = \
            (1, self.content_image.shape[1], self.content_image.shape[2], 3)

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != shape_content_image):
            raise TypeError("generated_image must be a tensor of shape {}"
                            .format(shape_content_image))

        # create GradientTape context to track operations for automatic
        # differentiation.
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        # calculate gradients of the total cost with respect to generated image
        # using gradient method of tape
        grad = tape.gradient(J_total, generated_image)

        return grad, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """
    Generates an image by minimizing the neural style cost
    function using gradient descent.

    Parameters:
    - iterations (int):
    The number of optimization steps to perform. Must be a positive integer.
    - step (int, optional):
    Interval at which to print the current cost, content cost, and style cost.
    If None, no intermediate outputs are displayed. Must be a positive integer
    less than `iterations`.
    - lr (float):
    Learning rate for the Adam optimizer. Must be a positive float.
    - beta1 (float):
    Exponential decay rate for the first moment estimate in the
    Adam optimizer. Must be a float in the range [0, 1].
    - beta2 (float):
    Exponential decay rate for the second moment estimate in
    the Adam optimizer. Must be a float in the range [0, 1].

    Returns:
    - best_image (numpy.ndarray):
    The generated image that achieves the lowest cost during optimization.
      Its values are clipped between 0 and 1, representing pixel intensity.
    - best_cost (float):
    The total style cost of the generated image.

    Raises:
    - TypeError:
    If any of the parameters have an invalid type.
    - ValueError:
    If `iterations` or `step` are not positive, or if `lr`, `beta1`,
    or `beta2` are outside their valid ranges.

    Process:
    - Initializes the `generated_image` as a copy of `content_image`.
    - Sets `best_cost` to infinity and `best_image` to None initially.
    - Uses the Adam optimizer for gradient descent with the specified
        hyperparameters.
    - Iteratively updates the generated image based on computed
        gradients to minimize style cost.
    - Records and updates the best image and cost during optimization.
    - Prints progress at each specified `step`, if provided.

    Final Output:
    - After all iterations, the `best_image` is clipped between 0 and 1
    for valid pixel intensity representation and returned as a numpy array
    alongside `best_cost`.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and (step < 0 or step > iterations):
            raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr < 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        # intialize image
        generated_image = tf.Variable(self.content_image)

        # intialize best cost and best image
        best_cost = float('inf')
        best_image = None

        # Initialize Adam
        optimizer = tf.optimizers.Adam(lr, beta1, beta2)

        # Optimization loop
        for i in range(iterations + 1):
            # compute gradients and costs
            grads, J_total, J_content, J_style = \
                self.compute_grads(generated_image)

            # use opt
            optimizer.apply_gradients([(grads, generated_image)])

            # selected best cost and best image
            if J_total < best_cost:
                best_cost = float(J_total)
                best_image = generated_image

            # Print step required
            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, style {}"
                      .format(i, J_total, J_content, J_style))

        # remove sup dim
        best_image = best_image[0]
        best_image = tf.clip_by_value(best_image, 0, 1)
        best_image = best_image.numpy()

        return best_image, best_cost
