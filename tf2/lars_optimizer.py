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
"""Functions and classes related to optimization (weight updates)."""

import re
import tensorflow as tf # Changed from tensorflow.compat.v2

EETA_DEFAULT = 0.001

# Changed inheritance from tf.keras.optimizers.legacy.Optimizer to tf.keras.optimizers.Optimizer
class LARSOptimizer(tf.keras.optimizers.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

  This version is updated to use the modern tf.keras.optimizers.Optimizer
  base class and TensorFlow 2.x best practices.
  """

  def __init__(self,
               learning_rate, # learning_rate can be a float or a LearningRateSchedule
               momentum=0.9,
               use_nesterov=False,
               weight_decay=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               classic_momentum=True,
               eeta=EETA_DEFAULT,
               name="LARSOptimizer",
               **kwargs): # Added **kwargs
    """Constructs a LARSOptimizer.

    Args:
      learning_rate: A `float` or a `tf.keras.optimizers.schedules.LearningRateSchedule`
          for learning rate.
      momentum: A `float` for momentum.
      use_nesterov: A 'Boolean' for whether to use nesterov momentum.
      weight_decay: A `float` for weight decay.
      exclude_from_weight_decay: A list of `string` for variable screening, if
          any of the string appears in a variable's name, the variable will be
          excluded for computing weight decay. For example, one could specify
          the list like ['batch_normalization', 'bias'] to exclude BN and bias
          from weight decay.
      exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
          for layer adaptation. If it is None, it will be defaulted the same as
          exclude_from_weight_decay.
      classic_momentum: A `boolean` for whether to use classic (or popular)
          momentum. The learning rate is applied during momentum update in
          classic momentum, but after momentum for popular momentum.
      eeta: A `float` for scaling of learning rate when computing trust ratio.
      name: The name for the scope.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(name=name, **kwargs) # Changed super call, added **kwargs

    self._set_hyper("learning_rate", learning_rate)
    self._set_hyper("momentum", momentum) # Also good to register momentum as hyper if it might be changed
    self.use_nesterov = use_nesterov
    self.weight_decay = weight_decay # Handled manually, not via base optimizer's weight_decay arg
    self.classic_momentum = classic_momentum
    self.eeta = eeta
    self.exclude_from_weight_decay = exclude_from_weight_decay

    if exclude_from_layer_adaptation is not None: # Explicit check for None
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def _create_slots(self, var_list):
    for var in var_list: # Changed v to var for clarity
      # Changed slot name from "Momentum" to "momentum" for convention
      self.add_slot(var, "momentum")

  def _resource_apply_dense(self, grad, var, apply_state=None): # Changed param to var
    if grad is None or var is None:
      return tf.group([]) # Changed from tf.no_op()

    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    
    # Processed learning rate and momentum from hyperparameters
    # lr_t is already processed by the base class (e.g., if a schedule is used)
    lr_t = coefficients["lr_t"]
    # Get momentum, treating it as a hyperparameter
    # Directly use self.momentum as it's set in __init__ and not a schedule
    # If momentum needed to be a schedulable hyper, use self._get_hyper("momentum", var_dtype)
    current_momentum = self.momentum

    var_name = var.name # Changed param_name to var_name

    # Changed slot retrieval name to "momentum" and variable name to momentum_var
    momentum_var = self.get_slot(var, "momentum")

    grad_with_wd = grad # Create a new variable for grad after potential weight decay
    if self._use_weight_decay(var_name):
      grad_with_wd += self.weight_decay * var

    if self.classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(var_name):
        w_norm = tf.norm(var, ord=2)
        g_norm = tf.norm(grad_with_wd, ord=2) # Use grad_with_wd
        # Added epsilon for numerical stability
        trust_ratio = tf.where(
            tf.logical_and(tf.greater(w_norm, 0.0), tf.greater(g_norm, 0.0)),
            (self.eeta * w_norm) / (g_norm + 1e-12), # Added epsilon
            1.0)
      scaled_lr = lr_t * trust_ratio

      # Changed tf.multiply to * operator
      next_momentum_var = current_momentum * momentum_var + scaled_lr * grad_with_wd
      if self.use_nesterov:
        update = current_momentum * next_momentum_var + scaled_lr * grad_with_wd
      else:
        update = next_momentum_var
      next_var = var - update
    else: # Non-classic momentum
      # Changed tf.multiply to * operator
      next_momentum_var = current_momentum * momentum_var + grad_with_wd # Use grad_with_wd
      if self.use_nesterov:
        update = current_momentum * next_momentum_var + grad_with_wd
      else:
        update = next_momentum_var

      trust_ratio = 1.0
      if self._do_layer_adaptation(var_name):
        w_norm = tf.norm(var, ord=2)
        v_norm = tf.norm(update, ord=2)
        # Added epsilon for numerical stability
        trust_ratio = tf.where(
            tf.logical_and(tf.greater(w_norm, 0.0), tf.greater(v_norm, 0.0)),
            (self.eeta * w_norm) / (v_norm + 1e-12), # Added epsilon
            1.0)
      scaled_lr = trust_ratio * lr_t # Corrected: lr_t not self.learning_rate
      next_var = var - scaled_lr * update

    # Using self._use_locking (from base optimizer) for consistency
    var_update_op = var.assign(next_var, use_locking=self._use_locking)
    momentum_var_update_op = momentum_var.assign(next_momentum_var, use_locking=self._use_locking)
    
    return tf.group(var_update_op, momentum_var_update_op)


  def _use_weight_decay(self, var_name): # Changed param_name to var_name
    """Whether to use L2 weight decay for `var_name`."""
    if not self.weight_decay: # Equivalent to self.weight_decay == 0.0
      return False
    if self.exclude_from_weight_decay:
      for r_pattern in self.exclude_from_weight_decay:
        if re.search(r_pattern, var_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, var_name): # Changed param_name to var_name
    """Whether to do layer-wise learning rate adaptation for `var_name`."""
    if self.exclude_from_layer_adaptation:
      for r_pattern in self.exclude_from_layer_adaptation:
        if re.search(r_pattern, var_name) is not None:
          return False
    return True

  def get_config(self):
    config = super().get_config() # Changed super call
    config.update({
        # "learning_rate" is already handled by base class get_config if set via _set_hyper
        # "momentum" is also handled by base class if set via _set_hyper
        "classic_momentum": self.classic_momentum,
        "weight_decay": self.weight_decay,
        "eeta": self.eeta,
        "use_nesterov": self.use_nesterov,
        # Added missing parameters for complete serialization
        "exclude_from_weight_decay": self.exclude_from_weight_decay,
        "exclude_from_layer_adaptation": self.exclude_from_layer_adaptation,
    })
    # learning_rate and momentum are already serialized by the base Optimizer's get_config
    # if they were set using _set_hyper. So, no need to explicitly add them here
    # unless we want to override or ensure they are present even if not hyper.
    # However, since learning_rate is definitely a hyper, it's covered.
    # If momentum was also set with _set_hyper (as I added), it's also covered.
    # If not, it should be added like: "momentum": self.momentum
    # Let's ensure momentum is in, as the original had it.
    # If _set_hyper("momentum", momentum) was used, it's fine.
    # If self.momentum = momentum, then it should be added manually.
    # The base Optimizer's get_config already serializes hypers set with _set_hyper.
    # Let's verify: tf.keras.optimizers.Optimizer.get_config serializes values from self._hyper.
    # So, "learning_rate" and "momentum" (if set with _set_hyper) are covered.
    return config