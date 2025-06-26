# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from absl import flags
from absl import logging # Import for logging
import tensorflow as tf


FLAGS = flags.FLAGS
BATCH_NORM_EPSILON = 1e-5


class BatchNormRelu(tf.keras.layers.Layer):
  """Combined Batch Normalization and ReLU activation layer."""

  def __init__(self,
               relu: bool = True,
               init_zero: bool = False,
               center: bool = True,
               scale: bool = True,
               data_format: str = 'channels_last',
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.relu = relu
    self.init_zero = init_zero # Store for get_config
    self.center = center
    self.scale = scale
    self.data_format = data_format
    self._batch_norm_decay = FLAGS.batch_norm_decay # Store for get_config
    self._global_bn = FLAGS.global_bn # Store for get_config

    gamma_initializer = tf.zeros_initializer() if init_zero else tf.ones_initializer()
    axis = -1 if data_format == 'channels_last' else 1

    if self._global_bn:
      self.bn = tf.keras.layers.SyncBatchNormalization(
          axis=axis,
          momentum=self._batch_norm_decay,
          epsilon=BATCH_NORM_EPSILON,
          center=center,
          scale=scale,
          gamma_initializer=gamma_initializer,
          name="sync_batch_norm"
      )
    else:
      self.bn = tf.keras.layers.BatchNormalization(
          axis=axis,
          momentum=self._batch_norm_decay,
          epsilon=BATCH_NORM_EPSILON,
          center=center,
          scale=scale,
          fused=None,
          gamma_initializer=gamma_initializer,
          name="batch_norm"
      )

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    x = self.bn(inputs, training=training)
    if self.relu:
      x = tf.nn.relu(x)
    return x

  def get_config(self):
    config = super().get_config()
    config.update({
        "relu": self.relu,
        "init_zero": self.init_zero,
        "center": self.center,
        "scale": self.scale,
        "data_format": self.data_format,
        "batch_norm_decay": self._batch_norm_decay, # Save the value used
        "global_bn": self._global_bn, # Save the value used
        # Epsilon is part of BN layer's config already if needed
    })
    return config


# In resnet.py

# ... (other code above DropBlock) ...

class DropBlock(tf.keras.layers.Layer):
  """DropBlock: A regularization method for convolutional networks."""

  def __init__(self,
               keep_prob: float | None,
               dropblock_size: int,
               data_format: str = 'channels_last',
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.keep_prob = keep_prob
    self.dropblock_size = dropblock_size
    self.data_format = data_format

  def call(self, net: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    if not training or self.keep_prob is None or self.keep_prob == 1.0:
      return net

    if self.data_format == 'channels_last':
      _, height, width, _ = net.shape.as_list()
    else: # channels_first
      _, _, height, width = net.shape.as_list()

    feature_map_size = height

    if not (isinstance(self.dropblock_size, int) and self.dropblock_size > 0):
        raise ValueError(f"dropblock_size must be a positive integer, got {self.dropblock_size}")
    
    _dropblock_size = min(self.dropblock_size, feature_map_size)

    denominator_sq = (feature_map_size - _dropblock_size + 1)**2
    if denominator_sq == 0:
        seed_drop_rate = (1.0 - self.keep_prob) if _dropblock_size == feature_map_size else 1.0
    else:
        seed_drop_rate = (1.0 - self.keep_prob) * (feature_map_size**2) / \
                         (_dropblock_size**2) / denominator_sq
    
    grid_range = tf.range(feature_map_size, dtype=tf.int32)
    h_i, w_i = tf.meshgrid(grid_range, grid_range)

    half_block_floor = _dropblock_size // 2
    half_block_ceil = (_dropblock_size - 1) // 2

    valid_block_center_h = tf.logical_and(h_i >= half_block_floor,
                                          h_i < feature_map_size - half_block_ceil)
    valid_block_center_w = tf.logical_and(w_i >= half_block_floor,
                                          w_i < feature_map_size - half_block_ceil)
    valid_block_center = tf.logical_and(valid_block_center_h, valid_block_center_w)

    valid_block_center = tf.expand_dims(valid_block_center, axis=0)
    
    # ***** 여기가 수정된 부분입니다 *****
    axis_to_expand_on = 1 if self.data_format == 'channels_first' else -1 # Corrected: axis=1 for NCHW means C dim
    valid_block_center = tf.expand_dims(
        valid_block_center, axis=axis_to_expand_on)
    # The original SimCLR code had:
    # `axis=-1 if data_format == 'channels_last' else 0)`
    # axis 0 would be batch. For NCHW (C is axis 1), if valid_block_center is (1, H, W)
    # tf.expand_dims(valid_block_center, axis=1) -> (1, 1, H, W) is correct for broadcasting with (N, C, H, W)
    # Let's re-verify the original comment:
    # `valid_block_center = tf.expand_dims(valid_block_center, -1 if data_format == 'channels_last' else 0)`
    # If NCHW, axis 0 would make it (1, 1, H, W), which is suitable for broadcasting with (N,C,H,W) if C=1.
    # But usually, it's (N,C,H,W) and the mask is (1,1,H,W) or (1,C,H,W).
    # If the mask is spatial only, (1,1,H,W) is desired.
    # Given `valid_block_center` is (1, H, W) after first expand_dims:
    # - For channels_last (NHWC): tf.expand_dims(vb, axis=-1) -> (1, H, W, 1). Correct.
    # - For channels_first (NCHW): tf.expand_dims(vb, axis=1) -> (1, 1, H, W). Correct for spatial mask.
    # So the corrected logic `axis_to_expand_on = 1 if self.data_format == 'channels_first' else -1` is what we need.

    randnoise = tf.random.uniform(net.shape, dtype=tf.float32)
    
    block_pattern = (1.0 - tf.cast(valid_block_center, dtype=tf.float32) +
                     (1.0 - seed_drop_rate) + randnoise) >= 1.0
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if _dropblock_size == feature_map_size:
      reduce_axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
      block_pattern = tf.reduce_min(block_pattern, axis=reduce_axes, keepdims=True)
    else:
      if self.data_format == 'channels_last':
        ksize = [1, _dropblock_size, _dropblock_size, 1]
        data_format_str = 'NHWC'
      else:
        ksize = [1, 1, _dropblock_size, _dropblock_size]
        data_format_str = 'NCHW'
      
      block_pattern = -tf.nn.max_pool(
          -block_pattern,
          ksize=ksize,
          strides=[1, 1, 1, 1],
          padding='SAME',
          data_format=data_format_str
      )

    percent_ones = tf.cast(tf.reduce_sum(block_pattern), tf.float32) / \
                   tf.cast(tf.size(input=block_pattern), tf.float32)
    
    net = net * tf.cast(block_pattern, net.dtype)
    net = net / (percent_ones + 1e-12)
    
    return net

  def get_config(self):
    config = super().get_config()
    config.update({
        "keep_prob": self.keep_prob,
        "dropblock_size": self.dropblock_size,
        "data_format": self.data_format,
    })
    return config

# ... (rest of resnet.py) ...

class FixedPadding(tf.keras.layers.Layer):
  """Pads the input along the spatial dimensions independently of input size."""

  def __init__(self,
               kernel_size: int,
               data_format: str = 'channels_last',
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.kernel_size = kernel_size
    self.data_format = data_format

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    pad_total = self.kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if self.data_format == 'channels_first':
      paddings = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
    else: # channels_last
      paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
    
    return tf.pad(tensor=inputs, paddings=paddings)

  def get_config(self):
    config = super().get_config()
    config.update({
        "kernel_size": self.kernel_size,
        "data_format": self.data_format,
    })
    return config


class Conv2dFixedPadding(tf.keras.layers.Layer):
  """2D convolution with fixed padding."""

  def __init__(self,
               filters: int,
               kernel_size: int,
               strides: int,
               data_format: str = 'channels_last',
               use_bias: bool = False,
               kernel_initializer: str | tf.keras.initializers.Initializer = 'VarianceScaling',
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.strides = strides # Store for get_config
    if strides > 1:
      self.fixed_padding = FixedPadding(kernel_size, data_format=data_format, name="fixed_padding")
    else:
      self.fixed_padding = None
      
    self.conv2d = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.get(kernel_initializer),
        data_format=data_format,
        name="conv2d"
    )

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    if self.fixed_padding:
      inputs = self.fixed_padding(inputs, training=training)
    return self.conv2d(inputs)

  def get_config(self):
    config = super().get_config()
    config.update({
        "filters": self.conv2d.filters,
        "kernel_size": self.conv2d.kernel_size[0],
        "strides": self.strides, # Use stored strides
        "data_format": self.conv2d.data_format,
        "use_bias": self.conv2d.use_bias,
        "kernel_initializer": tf.keras.initializers.serialize(self.conv2d.kernel_initializer),
    })
    return config


class IdentityLayer(tf.keras.layers.Layer):
  """Identity layer that simply returns its input."""

  def __init__(self, name: str | None = None, **kwargs):
      super().__init__(name=name, **kwargs)

  def call(self, inputs: tf.Tensor, training: bool | None = None):
    return tf.identity(inputs)


class SK_Conv2D(tf.keras.layers.Layer):
  """Selective kernel convolutional layer (https://arxiv.org/abs/1903.06586)."""

  def __init__(self,
               filters: int,
               strides: int,
               sk_ratio: float,
               min_dim: int = 32,
               data_format: str = 'channels_last',
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.data_format = data_format
    self.filters = filters
    self.strides = strides
    self.sk_ratio = sk_ratio
    self.min_dim = min_dim

    self.conv_main_stream = Conv2dFixedPadding(
        filters=2 * filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format,
        name="sk_main_conv")
    self.bn_relu_main = BatchNormRelu(data_format=data_format, name="sk_main_bn_relu")

    mid_dim = max(int(filters * sk_ratio), min_dim)
    self.attention_conv1 = tf.keras.layers.Conv2D(
        filters=mid_dim,
        kernel_size=1,
        strides=1,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        use_bias=False,
        data_format=data_format,
        name="sk_attention_conv1")
    self.bn_relu_attention = BatchNormRelu(data_format=data_format, name="sk_attention_bn_relu")
    self.attention_conv2 = tf.keras.layers.Conv2D(
        filters=2 * filters, # Output channels for attention weights for 2 branches
        kernel_size=1,
        strides=1,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        use_bias=False,
        data_format=data_format,
        name="sk_attention_conv2")

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    channel_axis = -1 if self.data_format == 'channels_last' else 1
    spatial_axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]

    x = self.conv_main_stream(inputs, training=training)
    x = self.bn_relu_main(x, training=training)
    
    branches = tf.stack(tf.split(x, num_or_size_splits=2, axis=channel_axis), axis=0)

    fused_features = tf.reduce_sum(branches, axis=0)
    gap = tf.reduce_mean(fused_features, axis=spatial_axes, keepdims=True)
    
    attention = self.attention_conv1(gap, training=training)
    attention = self.bn_relu_attention(attention, training=training)
    attention = self.attention_conv2(attention, training=training)
    
    attention_weights = tf.stack(tf.split(attention, num_or_size_splits=2, axis=channel_axis), axis=0)
    attention_weights = tf.nn.softmax(attention_weights, axis=0)

    weighted_branches = branches * attention_weights
    output = tf.reduce_sum(weighted_branches, axis=0)
    
    return output

  def get_config(self):
    config = super().get_config()
    config.update({
        "filters": self.filters,
        "strides": self.strides,
        "sk_ratio": self.sk_ratio,
        "min_dim": self.min_dim,
        "data_format": self.data_format,
    })
    return config


class SE_Layer(tf.keras.layers.Layer):
  """Squeeze and Excitation layer (https://arxiv.org/abs/1709.01507)."""

  def __init__(self,
               filters_in: int, # Number of input filters to the SE Layer
               se_ratio: float,
               data_format: str = 'channels_last',
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.data_format = data_format
    self.filters_in = filters_in # Store for get_config
    self.se_ratio = se_ratio
    
    reduced_filters = max(1, int(filters_in * se_ratio))
    self.se_reduce = tf.keras.layers.Conv2D(
        filters=reduced_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        padding='same',
        data_format=data_format,
        use_bias=True,
        name="se_reduce_conv")
    
    # filters for se_expand is set in build()
    self.se_expand = tf.keras.layers.Conv2D(
        filters=filters_in, # Placeholder, actual set in build
        kernel_size=1,
        strides=1,
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        padding='same',
        data_format=data_format,
        use_bias=True,
        name="se_expand_conv")

  def build(self, input_shape: tf.TensorShape):
    channel_axis = -1 if self.data_format == 'channels_last' else 1
    num_input_channels = input_shape[channel_axis]
    
    if num_input_channels is None:
        raise ValueError("The channel dimension of the inputs to SE_Layer must be defined. "
                         f"Received input_shape={input_shape}")

    self.se_expand.filters = num_input_channels # Set actual filters based on input
    
    # Manually build sublayers if needed (Conv2D usually builds on first call)
    if not self.se_reduce.built:
        # Determine expected input shape for se_reduce (output of GAP)
        gap_output_shape_list = [input_shape[0]] # Batch size
        if self.data_format == 'channels_first':
            gap_output_shape_list.extend([num_input_channels, 1, 1])
        else: # channels_last
            gap_output_shape_list.extend([1, 1, num_input_channels])
        self.se_reduce.build(tf.TensorShape(gap_output_shape_list))

    if not self.se_expand.built:
        # Determine expected input shape for se_expand (output of se_reduce + relu)
        expand_input_shape_list = [input_shape[0]]
        if self.data_format == 'channels_first':
            expand_input_shape_list.extend([self.se_reduce.filters, 1, 1])
        else: # channels_last
            expand_input_shape_list.extend([1, 1, self.se_reduce.filters])
        self.se_expand.build(tf.TensorShape(expand_input_shape_list))
        
    super().build(input_shape)

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    spatial_axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
    
    se_tensor = tf.reduce_mean(inputs, axis=spatial_axes, keepdims=True)
    se_tensor = self.se_reduce(se_tensor)
    se_tensor = tf.nn.relu(se_tensor)
    se_tensor = self.se_expand(se_tensor)
    
    return tf.sigmoid(se_tensor) * inputs

  def get_config(self):
    config = super().get_config()
    config.update({
        "filters_in": self.filters_in,
        "se_ratio": self.se_ratio,
        "data_format": self.data_format,
    })
    return config


class ResidualBlock(tf.keras.layers.Layer):
  """Standard Residual Block with two 3x3 convolutions."""

  def __init__(self,
               filters: int,
               strides: int,
               use_projection: bool = False,
               data_format: str = 'channels_last',
               dropblock_keep_prob: float | None = None, # Not used in this version of ResidualBlock
               dropblock_size: int | None = None,       # Not used
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    # del dropblock_keep_prob # These are not used by the original ResidualBlock
    # del dropblock_size
    self.filters = filters # Store for get_config
    self.strides = strides
    self.use_projection = use_projection
    self.data_format = data_format
    self._se_ratio = FLAGS.se_ratio # Store for get_config

    self.conv2d_bn_layers = []
    self.shortcut_layers = []

    if use_projection:
      if FLAGS.sk_ratio > 0 and strides > 1: # ResNet-D for shortcut
        self.shortcut_layers.append(FixedPadding(kernel_size=2, data_format=data_format, name="shortcut_pad"))
        self.shortcut_layers.append(
            tf.keras.layers.AveragePooling2D(
                pool_size=2,
                strides=strides,
                padding='VALID', # After FixedPadding
                data_format=data_format,
                name="shortcut_avg_pool"))
        self.shortcut_layers.append(
            Conv2dFixedPadding(
                filters=filters, kernel_size=1, strides=1, data_format=data_format, name="shortcut_conv1x1"))
      else: # Standard projection
        self.shortcut_layers.append(
            Conv2dFixedPadding(
                filters=filters, kernel_size=1, strides=strides, data_format=data_format, name="shortcut_conv1x1"))
      self.shortcut_layers.append(
          BatchNormRelu(relu=False, data_format=data_format, name="shortcut_bn"))

    self.conv2d_bn_layers.append(
        Conv2dFixedPadding(
            filters=filters, kernel_size=3, strides=strides, data_format=data_format, name="conv1_3x3"))
    self.conv2d_bn_layers.append(BatchNormRelu(data_format=data_format, name="bn_relu1"))
    self.conv2d_bn_layers.append(
        Conv2dFixedPadding(
            filters=filters, kernel_size=3, strides=1, data_format=data_format, name="conv2_3x3"))
    self.conv2d_bn_layers.append(
        BatchNormRelu(relu=False, init_zero=True, data_format=data_format, name="bn2_init_zero"))
    
    if self._se_ratio > 0:
      self.se_layer = SE_Layer(filters_in=filters, se_ratio=self._se_ratio, data_format=data_format, name="se_layer")
    else:
      self.se_layer = None

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    shortcut = inputs
    for layer in self.shortcut_layers:
      shortcut = layer(shortcut, training=training)

    x = inputs
    for layer in self.conv2d_bn_layers:
      x = layer(x, training=training)

    if self.se_layer:
      x = self.se_layer(x, training=training)

    return tf.nn.relu(x + shortcut)

  def get_config(self):
    config = super().get_config()
    config.update({
        "filters": self.filters,
        "strides": self.strides,
        "use_projection": self.use_projection,
        "data_format": self.data_format,
        "se_ratio": self._se_ratio,
        # Dropblock not used by this class, so not serialized
    })
    return config


class BottleneckBlock(tf.keras.layers.Layer):
  """Bottleneck Residual Block with 1x1, 3x3 (or SK), 1x1 convolutions."""

  def __init__(self,
               filters: int, # Base filters, output is 4*filters
               strides: int,
               use_projection: bool = False,
               data_format: str = 'channels_last',
               dropblock_keep_prob: float | None = None,
               dropblock_size: int | None = None,
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.filters = filters
    self.strides = strides
    self.use_projection = use_projection
    self.data_format = data_format
    self.dropblock_keep_prob = dropblock_keep_prob
    self.dropblock_size = dropblock_size
    self._sk_ratio = FLAGS.sk_ratio # Store for get_config
    self._se_ratio = FLAGS.se_ratio # Store for get_config

    self.projection_layers = []
    filters_out = 4 * filters
    if use_projection:
      if self._sk_ratio > 0 and strides > 1 : # ResNet-D for shortcut
        self.projection_layers.append(FixedPadding(kernel_size=2, data_format=data_format, name="shortcut_pad"))
        self.projection_layers.append(
            tf.keras.layers.AveragePooling2D(
                pool_size=2, strides=strides, padding='VALID', data_format=data_format, name="shortcut_avg_pool"))
        self.projection_layers.append(
            Conv2dFixedPadding(
                filters=filters_out, kernel_size=1, strides=1, data_format=data_format, name="shortcut_conv1x1"))
      else: # Standard projection
        self.projection_layers.append(
            Conv2dFixedPadding(
                filters=filters_out, kernel_size=1, strides=strides, data_format=data_format, name="shortcut_conv1x1"))
      self.projection_layers.append(
          BatchNormRelu(relu=False, data_format=data_format, name="shortcut_bn"))
    
    self.shortcut_dropblock = DropBlock(
        data_format=data_format, keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size, name="shortcut_dropblock")

    self.conv_relu_dropblock_layers = []

    # First 1x1 Conv
    self.conv_relu_dropblock_layers.append(
        Conv2dFixedPadding(filters=filters, kernel_size=1, strides=1, data_format=data_format, name="conv1_1x1"))
    self.conv_relu_dropblock_layers.append(BatchNormRelu(data_format=data_format, name="bn_relu1"))
    self.conv_relu_dropblock_layers.append(
        DropBlock(data_format=data_format, keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size, name="dropblock1"))

    # Middle 3x3 Conv (or SK-Conv)
    if self._sk_ratio > 0:
      self.conv_relu_dropblock_layers.append(
          SK_Conv2D(filters=filters, strides=strides, sk_ratio=self._sk_ratio, data_format=data_format, name="sk_conv_3x3"))
    else:
      self.conv_relu_dropblock_layers.append(
          Conv2dFixedPadding(filters=filters, kernel_size=3, strides=strides, data_format=data_format, name="conv2_3x3"))
      self.conv_relu_dropblock_layers.append(BatchNormRelu(data_format=data_format, name="bn_relu2"))
    self.conv_relu_dropblock_layers.append(
        DropBlock(data_format=data_format, keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size, name="dropblock2"))

    # Final 1x1 Conv
    self.conv_relu_dropblock_layers.append(
        Conv2dFixedPadding(filters=filters_out, kernel_size=1, strides=1, data_format=data_format, name="conv3_1x1"))
    self.conv_relu_dropblock_layers.append(
        BatchNormRelu(relu=False, init_zero=True, data_format=data_format, name="bn3_init_zero"))
    self.conv_relu_dropblock_layers.append(
        DropBlock(data_format=data_format, keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size, name="dropblock3"))

    if self._se_ratio > 0:
      # SE layer applied after the final BN, before addition to shortcut, as per common practice.
      # Input filters to SE layer is filters_out
      self.se_layer = SE_Layer(filters_in=filters_out, se_ratio=self._se_ratio, data_format=data_format, name="se_layer")
    else:
      self.se_layer = None

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    shortcut = inputs
    for layer in self.projection_layers:
      shortcut = layer(shortcut, training=training)
    shortcut = self.shortcut_dropblock(shortcut, training=training)

    x = inputs
    for layer in self.conv_relu_dropblock_layers:
      x = layer(x, training=training)

    if self.se_layer:
      x = self.se_layer(x, training=training)

    return tf.nn.relu(x + shortcut)

  def get_config(self):
    config = super().get_config()
    config.update({
        "filters": self.filters,
        "strides": self.strides,
        "use_projection": self.use_projection,
        "data_format": self.data_format,
        "dropblock_keep_prob": self.dropblock_keep_prob,
        "dropblock_size": self.dropblock_size,
        "sk_ratio": self._sk_ratio,
        "se_ratio": self._se_ratio,
    })
    return config


class BlockGroup(tf.keras.layers.Layer):
  """Group of Residual/Bottleneck Blocks."""

  def __init__(self,
               filters: int,
               block_fn: type[ResidualBlock] | type[BottleneckBlock], # Class type of the block
               blocks: int, # Number of blocks in this group
               strides: int, # Stride for the first block in this group
               data_format: str = 'channels_last',
               dropblock_keep_prob: float | None = None,
               dropblock_size: int | None = None,
               name: str | None = None,
               **kwargs):
    super().__init__(name=name, **kwargs) # Use Keras Layer's name, self._name is not standard
    self.filters = filters
    self.block_fn_name = block_fn.__name__ # Store class name for serialization
    self.blocks = blocks
    self.strides = strides
    self.data_format = data_format
    self.dropblock_keep_prob = dropblock_keep_prob
    self.dropblock_size = dropblock_size
    
    self.block_layers = [] # Renamed from self.layers to avoid Keras conflict
    # First block uses projection and specified strides
    self.block_layers.append(
        block_fn(
            filters=filters,
            strides=strides,
            use_projection=True, # First block always uses projection
            data_format=data_format,
            dropblock_keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
            name=f"{self.name}_block_0" if self.name else "block_0"
        ))

    # Subsequent blocks use stride 1 and no projection (unless input channels change)
    for i in range(1, blocks):
      self.block_layers.append(
          block_fn(
              filters=filters,
              strides=1, # Stride 1 for subsequent blocks
              use_projection=False, # Projection usually not needed if channels match
              data_format=data_format,
              dropblock_keep_prob=dropblock_keep_prob,
              dropblock_size=dropblock_size,
              name=f"{self.name}_block_{i}" if self.name else f"block_{i}"
          ))

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    x = inputs
    for layer in self.block_layers:
      x = layer(x, training=training)
    return tf.identity(x, name=self.name + "_output" if self.name else "block_group_output")

  def get_config(self):
    config = super().get_config()
    config.update({
        "filters": self.filters,
        "block_fn_name": self.block_fn_name,
        "blocks": self.blocks,
        "strides": self.strides,
        "data_format": self.data_format,
        "dropblock_keep_prob": self.dropblock_keep_prob,
        "dropblock_size": self.dropblock_size,
    })
    return config


class Resnet(tf.keras.layers.Layer): # Changed from tf.keras.Model to tf.keras.layers.Layer for consistency
  """ResNet model."""

  def __init__(self,
               block_fn: type[ResidualBlock] | type[BottleneckBlock],
               layers: list[int], # Number of blocks in each of the 4 groups
               width_multiplier: int,
               cifar_stem: bool = False,
               data_format: str = 'channels_last',
               dropblock_keep_probs: list[float | None] | None = None,
               dropblock_size: int | None = None,
               name: str | None = "ResNet",
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.block_fn_name = block_fn.__name__ # Store for get_config
    self.layers_config = layers # Store for get_config
    self.width_multiplier = width_multiplier
    self.cifar_stem = cifar_stem
    self.data_format = data_format
    self.dropblock_keep_probs = dropblock_keep_probs if dropblock_keep_probs else [None] * 4
    self.dropblock_size = dropblock_size

    if len(self.dropblock_keep_probs) != 4:
      raise ValueError('dropblock_keep_probs must be a list of 4 values or None.')

    # Determine trainability of initial blocks based on fine-tuning strategy
    # This logic seems to control `trainable` for the entire layer.
    # Individual block trainability is handled by tf.stop_gradient in call.
    initial_block_trainable = (
        FLAGS.train_mode != 'finetune' or FLAGS.fine_tune_after_block == -1
    )
    # The `trainable` property of the ResNet layer itself can be set.
    # self.trainable = initial_block_trainable (This would freeze the whole ResNet if False)

    self.initial_conv_relu_max_pool = []
    current_filters = 64 * width_multiplier
    if cifar_stem:
      self.initial_conv_relu_max_pool.append(
          Conv2dFixedPadding(filters=current_filters, kernel_size=3, strides=1, data_format=data_format, name="initial_conv_cifar", trainable=initial_block_trainable))
      self.initial_conv_relu_max_pool.append(BatchNormRelu(data_format=data_format, name="initial_bn_relu_cifar", trainable=initial_block_trainable))
      # No explicit max pool in CIFAR stem, pooling happens via strides in block groups.
    else: # Standard ImageNet stem
      if FLAGS.sk_ratio > 0: # ResNet-D stem
        self.initial_conv_relu_max_pool.append(
            Conv2dFixedPadding(filters=current_filters // 2, kernel_size=3, strides=2, data_format=data_format, name="stem_conv1_resnetd", trainable=initial_block_trainable))
        self.initial_conv_relu_max_pool.append(BatchNormRelu(data_format=data_format, name="stem_bn_relu1_resnetd", trainable=initial_block_trainable))
        self.initial_conv_relu_max_pool.append(
            Conv2dFixedPadding(filters=current_filters // 2, kernel_size=3, strides=1, data_format=data_format, name="stem_conv2_resnetd", trainable=initial_block_trainable))
        self.initial_conv_relu_max_pool.append(BatchNormRelu(data_format=data_format, name="stem_bn_relu2_resnetd", trainable=initial_block_trainable))
        self.initial_conv_relu_max_pool.append(
            Conv2dFixedPadding(filters=current_filters, kernel_size=3, strides=1, data_format=data_format, name="stem_conv3_resnetd", trainable=initial_block_trainable))
      else: # Standard ResNet stem
        self.initial_conv_relu_max_pool.append(
            Conv2dFixedPadding(filters=current_filters, kernel_size=7, strides=2, data_format=data_format, name="initial_conv7x7", trainable=initial_block_trainable))
      
      self.initial_conv_relu_max_pool.append(BatchNormRelu(data_format=data_format, name="initial_bn_relu", trainable=initial_block_trainable))
      self.initial_conv_relu_max_pool.append(
          tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='SAME', data_format=data_format, name="initial_max_pool", trainable=initial_block_trainable))

    self.block_groups_list = [] # Renamed from self.block_groups
    # Block group filters: 64, 128, 256, 512 (scaled by width_multiplier)
    # Block group strides: 1, 2, 2, 2 (first block_group usually has stride 1 after stem)
    block_strides = [1, 2, 2, 2]
    block_filters = [64, 128, 256, 512]

    # This loop logic for trainable seems to set based on when fine_tune_after_block is MET
    # Meaning, if fine_tune_after_block = 1, then block_group[0] is frozen, block_group[1] onwards are trainable.
    # The original trainable logic was a bit confusing, it seemed to set `trainable` for future blocks.
    # Let's assume `trainable` here means "can this block ever be trained". Freezing is handled by `stop_gradient`.
    # `initial_block_trainable` sets the default.
    
    # Corrected `trainable` logic based on SimCLR's fine-tuning:
    # A block group is trainable if fine_tune_after_block is less than or equal to its index (0-based).
    # Or if train_mode is not 'finetune'.
    
    for i in range(4): # 4 block groups
        group_trainable = initial_block_trainable # Default trainability
        if FLAGS.train_mode == 'finetune':
            if FLAGS.fine_tune_after_block > i : # If current group index is before the fine_tune_after_block point
                group_trainable = False
            else: # If current group index is at or after the fine_tune_after_block point
                group_trainable = True

        # Ensure the entire layer (sub-layers) get the `trainable` status
        # This is more for initial construction. tf.stop_gradient controls gradient flow.
        # Keras layers' `trainable` property will be recursively applied.
        
        # We pass `trainable` to BlockGroup constructor, which should set its own `trainable` attr.
        # This does not seem to be how Keras expects `trainable` to be used for sub-layers in a list.
        # Instead, one would typically set `layer.trainable = group_trainable` *after* creating the layer.
        # However, SimCLR's `BlockGroup` doesn't use the passed `trainable` arg to set `self.trainable`.
        # The `trainable` arg in SimCLR's BlockGroup was for sub-layers of BlockGroup.
        # Let's simplify: the `tf.stop_gradient` in call is the primary mechanism for freezing.
        # The `trainable` attribute of the BlockGroup itself and its sublayers should be True by default.
        # The initial stem's trainability is handled by `initial_block_trainable`.
        # `layer.trainable=False` on the ResNet layer would freeze everything.

        self.block_groups_list.append(
            BlockGroup(
                filters=block_filters[i] * width_multiplier,
                block_fn=block_fn,
                blocks=layers[i],
                strides=block_strides[i],
                name=f'block_group{i+1}',
                data_format=data_format,
                dropblock_keep_prob=self.dropblock_keep_probs[i],
                dropblock_size=dropblock_size
                # trainable=group_trainable # This was in original, but BlockGroup doesn't use it to set self.trainable
            ))
        # After creating the block_group, one could do:
        # self.block_groups_list[-1].trainable = group_trainable
        # However, the stop_gradient in call() is the main freezing mechanism here.


  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    x = inputs
    for layer in self.initial_conv_relu_max_pool:
      # Pass training explicitly if the layer might need it (BN, DropBlock)
      if isinstance(layer, (BatchNormRelu, DropBlock, tf.keras.layers.MaxPooling2D, Conv2dFixedPadding)):
          x = layer(x, training=training)
      else: # For IdentityLayer or layers not needing training flag
          x = layer(x)

    for i, layer_group in enumerate(self.block_groups_list):
      # Freezing logic based on FLAGS.fine_tune_after_block
      # If current block index `i` matches `fine_tune_after_block`,
      # it means this block (and subsequent ones) should be trained,
      # and blocks *before* this index should be frozen.
      # So, if `fine_tune_after_block` is `i`, gradients stop *before* this block group.
      if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block == i:
        logging.info(f"ResNet: Stopping gradient before block group {i} (0-indexed).")
        x = tf.stop_gradient(x)
      x = layer_group(x, training=training)
    
    # Final stop_gradient if fine_tune_after_block is 4 (meaning all conv blocks are frozen)
    if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block == 4:
      logging.info("ResNet: Stopping gradient after all block groups (all frozen).")
      x = tf.stop_gradient(x)

    # Global Average Pooling
    spatial_axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
    x = tf.reduce_mean(x, axis=spatial_axes)

    return tf.identity(x, 'final_avg_pool')

  def get_config(self):
    config = super().get_config()
    config.update({
        "block_fn_name": self.block_fn_name,
        "layers_config": self.layers_config,
        "width_multiplier": self.width_multiplier,
        "cifar_stem": self.cifar_stem,
        "data_format": self.data_format,
        "dropblock_keep_probs": self.dropblock_keep_probs,
        "dropblock_size": self.dropblock_size,
    })
    return config


def resnet(resnet_depth: int,
           width_multiplier: int,
           cifar_stem: bool = False,
           data_format: str = 'channels_last',
           dropblock_keep_probs: list[float | None] | None = None,
           dropblock_size: int | None = None) -> Resnet:
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
      34: {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
      50: {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},
      101: {'block': BottleneckBlock, 'layers': [3, 4, 23, 3]},
      152: {'block': BottleneckBlock, 'layers': [3, 8, 36, 3]},
      200: {'block': BottleneckBlock, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return Resnet(
      block_fn=params['block'],
      layers=params['layers'],
      width_multiplier=width_multiplier,
      cifar_stem=cifar_stem,
      dropblock_keep_probs=dropblock_keep_probs,
      dropblock_size=dropblock_size,
      data_format=data_format,
      name=f"ResNet{resnet_depth}_w{width_multiplier}"
  )