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
# (License header omitted for brevity but should be kept)
# ==============================================================================
"""Model specification for SimCLR."""

import math
from absl import flags
import tensorflow as tf # Changed from tensorflow.compat.v2 as tf

# Assuming these are local modules updated for TF2
import data_util
import lars_optimizer
import resnet


FLAGS = flags.FLAGS


def build_optimizer(learning_rate: float | tf.keras.optimizers.schedules.LearningRateSchedule):
  """Returns the optimizer based on FLAGS.optimizer.

  Args:
    learning_rate: The learning rate or a learning rate schedule.

  Returns:
    A tf.keras.optimizers.Optimizer instance.
  """
  optimizer_name = FLAGS.optimizer.lower() # Normalize optimizer name
  if optimizer_name == 'momentum':
    return tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=FLAGS.momentum, nesterov=True)
  elif optimizer_name == 'adam':
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif optimizer_name == 'lars':
    # Ensure lars_optimizer.LARSOptimizer is the updated version
    return lars_optimizer.LARSOptimizer(
        learning_rate=learning_rate,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        # Ensure 'head_supervised' is a relevant name scope for exclusion
        exclude_from_weight_decay=[
            'batch_normalization', 'bias', 'head_supervised'
        ])
  else:
    raise ValueError(f'Unknown optimizer {FLAGS.optimizer}')


def add_weight_decay(model: tf.keras.Model, adjust_per_optimizer: bool = True) -> tf.Tensor:
  """Computes L2 weight decay loss.

  Args:
    model: The Keras model.
    adjust_per_optimizer: Boolean, if True, adjust weight decay application
      based on the optimizer type (e.g., LARS handles WD internally for most vars).

  Returns:
    Scalar tensor representing the L2 weight decay loss.
  """
  if FLAGS.weight_decay is None or FLAGS.weight_decay == 0:
      return 0.0

  if adjust_per_optimizer and 'lars' in FLAGS.optimizer.lower():
    # For LARS, weight decay is handled by the optimizer for most variables.
    # Add WD only for 'head_supervised' variables not handled by LARS's WD.
    # This assumes 'head_supervised' variables are not in LARS's exclude_from_weight_decay list OR
    # that we want to apply WD to them regardless. The original logic specifically targets them.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_variables
        if 'head_supervised' in v.name and 'bias' not in v.name # Exclude biases from WD
    ]
    if l2_losses:
      return FLAGS.weight_decay * tf.add_n(l2_losses)
    else:
      return 0.0 # Return a float tensor

  # General case: apply weight decay to variables not excluded by name.
  # TODO(srbs): Think of a way to avoid name-based filtering here. (Original TODO)
  l2_losses = [
      tf.nn.l2_loss(v)
      for v in model.trainable_weights # Using trainable_weights is fine
      if 'batch_normalization' not in v.name and 'bias' not in v.name # Exclude BN params and biases
  ]
  if not l2_losses: # Handle case with no variables for WD
      return 0.0

  loss = FLAGS.weight_decay * tf.add_n(l2_losses)
  return loss


def get_train_steps(num_examples: int) -> int:
  """Determines the number of training steps.

  Args:
    num_examples: The total number of examples in the training dataset.

  Returns:
    The total number of training steps.
  """
  if FLAGS.train_steps and FLAGS.train_steps > 0:
      return FLAGS.train_steps
  else:
      # Ensure FLAGS.train_batch_size is not zero to avoid DivisionByZeroError
      if FLAGS.train_batch_size == 0:
          raise ValueError("train_batch_size cannot be zero.")
      return num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a cosine decay learning rate schedule."""

  def __init__(self, base_learning_rate: float, num_examples: int, name: str | None = None):
    super().__init__() # Changed super call
    self.base_learning_rate = base_learning_rate
    self.num_examples = num_examples
    self._name = name # Keras schedules use 'name' not '_name' for config, but _name is fine for scope

  def __call__(self, step: tf.Tensor) -> tf.Tensor:
    with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
      # Ensure FLAGS.train_batch_size is not zero
      if FLAGS.train_batch_size == 0:
          raise ValueError("train_batch_size cannot be zero for warmup calculation.")
      
      warmup_steps = tf.cast(
          round(FLAGS.warmup_epochs * self.num_examples // FLAGS.train_batch_size),
          dtype=tf.float32
      )

      if FLAGS.learning_rate_scaling == 'linear':
        scaled_lr = self.base_learning_rate * tf.cast(FLAGS.train_batch_size, tf.float32) / 256.0
      elif FLAGS.learning_rate_scaling == 'sqrt':
        scaled_lr = self.base_learning_rate * tf.sqrt(tf.cast(FLAGS.train_batch_size, tf.float32))
      else:
        raise ValueError(f'Unknown learning rate scaling {FLAGS.learning_rate_scaling}')
      
      learning_rate = (step / warmup_steps * scaled_lr if tf.greater(warmup_steps, 0) else scaled_lr)

      total_steps = tf.cast(get_train_steps(self.num_examples), dtype=tf.float32)
      
      # Changed from tf.keras.experimental.CosineDecay
      # Ensure decay_steps is positive
      cosine_decay_steps = tf.maximum(1.0, total_steps - warmup_steps)
      cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
          initial_learning_rate=scaled_lr,
          decay_steps=cosine_decay_steps,
          alpha=0.0 # Cosine decay to 0
      )
      
      # step - warmup_steps should be non-negative for CosineDecay
      decay_step_input = tf.maximum(0.0, step - warmup_steps)
      learning_rate_decayed = cosine_decay(decay_step_input)

      learning_rate = tf.where(step < warmup_steps, learning_rate, learning_rate_decayed)
      return learning_rate

  def get_config(self) -> dict:
    config = {
        'base_learning_rate': self.base_learning_rate,
        'num_examples': self.num_examples,
        # 'name': self._name # Keras schedules typically serialize 'name' if passed to super's init
    }
    if self._name is not None: # Include name if provided
        config['name'] = self._name
    return config


class LinearLayer(tf.keras.layers.Layer):
  """A Keras layer for linear transformation with optional BatchNorm."""

  def __init__(self,
               num_classes: int | callable,
               use_bias: bool = True,
               use_bn: bool = False,
               kernel_initializer: str | tf.keras.initializers.Initializer = tf.keras.initializers.RandomNormal(stddev=0.01),
               name: str = 'linear_layer',
               **kwargs):
    super().__init__(name=name, **kwargs) # Changed super call
    self.num_classes = num_classes
    self.use_bias = use_bias
    self.use_bn = use_bn
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer) # Allow string or initializer

    if self.use_bn:
      # BatchNormRelu typically includes a BN layer.
      # The 'relu=False' means it's just Batch Norm.
      # 'center=use_bias' means BN's beta (offset) is used if use_bias is True.
      self.bn_layer = resnet.BatchNormRelu(relu=False, center=self.use_bias, name="batch_norm")
    
    # Dense layer's use_bias is false if BN is used, as BN provides an offset (beta).
    self.dense_layer = tf.keras.layers.Dense(
        units=1, # Placeholder, will be set in build
        kernel_initializer=self.kernel_initializer,
        use_bias=self.use_bias and not self.use_bn, # Only use Dense bias if no BN or BN doesn't center
        name="dense"
    )


  def build(self, input_shape: tf.TensorShape):
    if callable(self.num_classes):
      _num_classes = self.num_classes(input_shape)
    else:
      _num_classes = self.num_classes
    
    # Reconfigure the dense layer with the correct number of units
    self.dense_layer.units = _num_classes
    # self.dense_layer.build(input_shape) # Dense layer will be built on first call or if build() is called on it
    
    # Explicitly build sub-layers if not done automatically or if needed for weight access
    if not self.dense_layer.built:
        self.dense_layer.build(input_shape)
    if self.use_bn and not self.bn_layer.built:
        # Determine the shape bn_layer expects (output of dense)
        # This can be tricky if num_classes is dynamic before first call.
        # Assuming dense output shape is (batch_size, _num_classes)
        dense_output_shape = tf.TensorShape([input_shape[0], _num_classes])
        self.bn_layer.build(dense_output_shape)

    super().build(input_shape) # Important to call base class build

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    if inputs.shape.rank != 2: # Original used ndims
        raise ValueError(f"Input tensor must have rank 2 (batch_size, features), got rank {inputs.shape.rank}")
    
    x = self.dense_layer(inputs)
    if self.use_bn:
      x = self.bn_layer(x, training=training)
    return x

  def get_config(self) -> dict:
    config = super().get_config()
    config.update({
        'num_classes': self.num_classes if not callable(self.num_classes) else None, # Serialization of callable might be tricky
        'use_bias': self.use_bias,
        'use_bn': self.use_bn,
        'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
    })
    return config


class ProjectionHead(tf.keras.layers.Layer):
  """Projection head for SimCLR, mapping representation h to z."""

  def __init__(self, name: str = "projection_head", **kwargs):
    super().__init__(name=name, **kwargs) # Changed super call
    
    out_dim = FLAGS.proj_out_dim
    self.linear_layers_list = [] # Renamed from self.linear_layers to avoid conflict if base class has it

    if FLAGS.proj_head_mode == 'none':
      pass
    elif FLAGS.proj_head_mode == 'linear':
      self.linear_layers_list = [
          LinearLayer(num_classes=out_dim, use_bias=False, use_bn=True, name='projection_linear_0')
      ]
    elif FLAGS.proj_head_mode == 'nonlinear':
      for j in range(FLAGS.num_proj_layers):
        # Middle layers: use bias, BN, and ReLU (ReLU applied in call)
        # Final layer: use_bias=False, use_bn=True, no ReLU (as per SimCLR)
        is_final_layer = (j == FLAGS.num_proj_layers - 1)
        
        # For middle layers, output dimensionality is same as input to that layer
        # For final layer, output dimensionality is proj_out_dim
        layer_out_dim = out_dim if is_final_layer else (lambda input_shape: int(input_shape[-1]))

        self.linear_layers_list.append(
            LinearLayer(
                num_classes=layer_out_dim,
                use_bias=not is_final_layer, # Bias for middle layers
                use_bn=True, # BN for all layers in projection head
                name=f'projection_nonlinear_{j}'
            )
        )
    else:
      raise ValueError(f'Unknown head projection mode {FLAGS.proj_head_mode}')

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tuple[tf.Tensor, tf.Tensor]:
    if FLAGS.proj_head_mode == 'none':
      # If no projection, output is input, and input for supervised head is also input
      return inputs, inputs 

    hiddens_list = [tf.identity(inputs, 'proj_head_input')]

    if FLAGS.proj_head_mode == 'linear':
      if not self.linear_layers_list: # Should not happen if __init__ is correct
          raise ValueError("Linear projection head selected but no layers were created.")
      # Apply the single linear layer
      processed_hidden = self.linear_layers_list[0](hiddens_list[-1], training=training)
      hiddens_list.append(processed_hidden)
    
    elif FLAGS.proj_head_mode == 'nonlinear':
      current_hidden = hiddens_list[-1]
      for j, layer in enumerate(self.linear_layers_list):
        current_hidden = layer(current_hidden, training=training)
        if j != len(self.linear_layers_list) - 1: # Not the final layer
          current_hidden = tf.nn.relu(current_hidden)
        hiddens_list.append(current_hidden)
    else: # Should be caught in __init__, but defensive
      raise ValueError(f'Unknown head projection mode {FLAGS.proj_head_mode}')

    # The last element in hiddens_list is the final projection output z
    proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
    
    # Select the hidden representation for the supervised fine-tuning head
    # FLAGS.ft_proj_selector is an index into hiddens_list
    if FLAGS.ft_proj_selector < 0 or FLAGS.ft_proj_selector >= len(hiddens_list):
        raise ValueError(f"ft_proj_selector index {FLAGS.ft_proj_selector} out of bounds for hiddens_list (length {len(hiddens_list)})")
    finetune_head_input = hiddens_list[FLAGS.ft_proj_selector]
    
    return proj_head_output, finetune_head_input

  def get_config(self) -> dict:
    config = super().get_config()
    # Add any specific config for ProjectionHead if needed, e.g., FLAGS values used at construction
    # config.update({'proj_head_mode': FLAGS.proj_head_mode, ...}) # Be careful with FLAGS in get_config
    return config


class SupervisedHead(tf.keras.layers.Layer):
  """Supervised head for classification, typically used for fine-tuning or linear evaluation."""

  def __init__(self, num_classes: int, name: str = 'head_supervised', **kwargs):
    super().__init__(name=name, **kwargs) # Changed super call
    self.linear_layer = LinearLayer(num_classes=num_classes, use_bias=True, use_bn=False, name="supervised_linear") # Standard FC layer

  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    outputs = self.linear_layer(inputs, training=training)
    # Name the output logits for easier identification
    return tf.identity(outputs, name='logits_sup')

  def get_config(self) -> dict:
    config = super().get_config()
    # num_classes is part of self.linear_layer's config
    return config


class Model(tf.keras.models.Model):
  """SimCLR model combining a ResNet backbone with projection and/or supervised heads."""

  def __init__(self, num_classes: int | None = None, name: str = "SimCLRModel", **kwargs):
    super().__init__(name=name, **kwargs) # Changed super call

    # Backbone ResNet model
    self.resnet_model = resnet.resnet(
        resnet_depth=FLAGS.resnet_depth,
        width_multiplier=FLAGS.width_multiplier,
        cifar_stem=FLAGS.image_size <= 32  # Use CIFAR-style stem for small images
    )

    # Projection head (for contrastive learning)
    self._projection_head = ProjectionHead() # Name is set in ProjectionHead's __init__

    # Supervised head (for fine-tuning or linear evaluation)
    self.uses_supervised_head = (FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining)
    if self.uses_supervised_head:
      if num_classes is None:
          raise ValueError("num_classes must be provided when using a supervised head.")
      self.supervised_head = SupervisedHead(num_classes)
    else:
      self.supervised_head = None


  # Changed __call__ to call, the standard Keras method
  def call(self, inputs: tf.Tensor, training: bool | None = None) -> tuple[tf.Tensor | None, tf.Tensor | None]:
    """Forward pass of the SimCLR model.

    Args:
      inputs: Input tensor, typically (batch_size, height, width, channels * num_transforms).
              Channels are stacked for multiple augmented views of each image.
      training: Boolean, whether the model is in training mode.

    Returns:
      A tuple (projection_output, supervised_output).
      projection_output: Output of the projection head (z).
      supervised_output: Output of the supervised head (logits for classification).
                         Can be None if not in a mode that uses it.
    """
    if training and FLAGS.train_mode == 'pretrain':
      if FLAGS.fine_tune_after_block > -1:
        # This check is specific to the original SimCLR codebase's capabilities
        raise ValueError('Layer freezing (fine_tune_after_block > -1) is not supported '
                         'during pretraining mode in this implementation.')

    # Ensure input channels are defined (e.g., 3 for RGB)
    if inputs.shape[-1] is None: # Last dimension is channels
      raise ValueError('The input channels dimension must be statically known '
                       f'(got input shape {inputs.shape})')
    
    # Assuming channel dimension stacks multiple (e.g., 2) RGB augmentations
    # e.g., if inputs.shape[-1] == 6, then num_augmentations is 2 (for 3-channel images)
    if inputs.shape[-1] % 3 != 0:
        raise ValueError(f"Input channels ({inputs.shape[-1]}) must be a multiple of 3 (RGB).")
    num_augmentations = inputs.shape[-1] // 3

    if num_augmentations < 1 : # Should be at least 1, usually 2 for SimCLR
        raise ValueError(f"Number of augmentations ({num_augmentations}) derived from input channels must be at least 1.")

    # Split the stacked augmentations into a list of tensors
    # Each tensor in features_list will be (batch_size, height, width, 3)
    features_list = tf.split(inputs, num_or_size_splits=num_augmentations, axis=-1)

    # Apply blur if configured (SimCLR specific augmentation)
    if FLAGS.use_blur and training and FLAGS.train_mode == 'pretrain':
      # Ensure data_util.batch_random_blur is TF2 compatible and expects a list
      features_list = data_util.batch_random_blur(
          features_list,
          FLAGS.image_size, # Assuming this is height
          FLAGS.image_size  # Assuming this is width
      )
    
    # Concatenate augmented views along the batch dimension for parallel processing
    # If features_list has N tensors of (B, H, W, C), result is (N*B, H, W, C)
    features_for_backbone = tf.concat(features_list, axis=0)

    # Pass through the ResNet backbone
    # hiddens shape: (num_augmentations * batch_size, representation_dim)
    hiddens = self.resnet_model(features_for_backbone, training=training)

    # Pass through the projection head
    # projection_head_outputs (z) and supervised_head_inputs (h' for linear eval)
    projection_head_outputs, supervised_head_inputs = self._projection_head(hiddens, training=training)

    supervised_head_outputs = None
    if FLAGS.train_mode == 'finetune':
      if not self.supervised_head:
          raise ValueError("Supervised head not initialized for finetune mode.")
      supervised_head_outputs = self.supervised_head(supervised_head_inputs, training=training)
      # In fine-tuning, typically only supervised output is used for loss. Projection output might be ignored.
      return None, supervised_head_outputs # Or (projection_head_outputs, supervised_head_outputs) if proj output is needed

    elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
      if not self.supervised_head:
          raise ValueError("Supervised head not initialized for lineareval_while_pretraining mode.")
      # Stop gradient to prevent linear eval from affecting backbone pretraining
      frozen_supervised_inputs = tf.stop_gradient(supervised_head_inputs)
      supervised_head_outputs = self.supervised_head(frozen_supervised_inputs, training=training)
      return projection_head_outputs, supervised_head_outputs
    
    elif FLAGS.train_mode == 'pretrain': # Pretrain only, no linear eval
      return projection_head_outputs, None
      
    else: # Fallback for unknown modes or 'eval' mode (which might need specific handling)
        # For a generic 'eval' mode, one might want both outputs or specific ones.
        # This implementation seems to tie evaluation to 'finetune' or 'lineareval_while_pretraining'
        # or implies evaluation happens on the pretrain head's outputs.
        # If supervised_head exists, one might want its output too.
        if self.supervised_head:
            eval_supervised_inputs = tf.stop_gradient(supervised_head_inputs) # Good practice for eval
            supervised_head_outputs = self.supervised_head(eval_supervised_inputs, training=training) # training=False typically for eval
        return projection_head_outputs, supervised_head_outputs

  def get_config(self) -> dict:
    config = super().get_config()
    # Add configuration specific to this model, e.g., num_classes if it affects construction
    # However, num_classes is passed to SupervisedHead, which has its own config.
    # Backbone and head configurations are handled by those layers' get_config.
    config.update({
        "uses_supervised_head": self.uses_supervised_head,
        # Potentially serialize num_classes if it was passed to Model's __init__
        # and needed for re-construction directly by Model.
    })
    if self.uses_supervised_head and self.supervised_head:
        config['num_classes'] = self.supervised_head.linear_layer.num_classes # Example
    return config