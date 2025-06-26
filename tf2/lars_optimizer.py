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
# In lars_optimizer.py

import re
import tensorflow as tf
# import keras # Not strictly needed here if using tf.keras.optimizers.Optimizer

EETA_DEFAULT = 0.001

class LARSOptimizer(tf.keras.optimizers.Optimizer): # Or keras.optimizers.Optimizer if Keras 3 is primary
  """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

  This optimizer is an implementation of LARS (Layer-wise Adaptive Rate Scaling)
  for large batch training, as described in "Large Batch Training of
  Convolutional Networks" by You, Gitman, and Ginsburg.
  (https://arxiv.org/abs/1708.03888)

  It is updated to be compatible with TensorFlow 2.x (including Keras 3).
  """

  def __init__(self,
               learning_rate, # Required by Keras 3 Optimizer base class
               momentum: float = 0.9,
               use_nesterov: bool = False,
               weight_decay: float = 0.0, # This is LARS-specific weight decay
               exclude_from_weight_decay: list[str] | None = None,
               exclude_from_layer_adaptation: list[str] | None = None,
               classic_momentum: bool = True,
               eeta: float = EETA_DEFAULT,
               name: str = "LARSOptimizer",
               **kwargs):
    """Constructs a LARSOptimizer.

    Args:
      learning_rate: A `float` or a `tf.keras.optimizers.schedules.LearningRateSchedule`.
        This is passed to the Keras Optimizer base class.
      momentum: A `float` for momentum. LARS uses this for its momentum updates.
      use_nesterov: A `bool`, whether to use Nesterov momentum.
      weight_decay: A `float` for LARS-specific weight decay. This is handled
        manually by LARS and is distinct from any global `weight_decay`
        parameter the Keras Optimizer base class might have.
      exclude_from_weight_decay: A list of `str` patterns. Variables whose
        names match any of these patterns will be excluded from LARS's
        weight decay.
      exclude_from_layer_adaptation: Similar to `exclude_from_weight_decay`,
        but for LARS's layer adaptation (trust ratio calculation). If None,
        it defaults to `exclude_from_weight_decay`.
      classic_momentum: A `bool`. If True, uses classic momentum where
        learning rate scales gradients before momentum update. Otherwise,
        scales the update after momentum.
      eeta: A `float` for the LARS trust coefficient.
      name: Optional name for the operations created when applying gradients.
      **kwargs: Additional keyword arguments. Passed to the Keras Optimizer
        base class. Note: Avoid passing `weight_decay` via kwargs if you
        want LARS to handle it exclusively, as Keras Optimizer also has a
        `weight_decay` param.
    """
    # Pass learning_rate to the superclass constructor for Keras 3
    # Do NOT pass `weight_decay` here if LARS is to handle it manually,
    # as the base Optimizer also has a `weight_decay` argument which would apply globally.
    # If other kwargs might contain 'weight_decay', consider popping it:
    # lars_specific_weight_decay = weight_decay
    # if 'weight_decay' in kwargs:
    #     logging.warning("LARS received 'weight_decay' in kwargs, which might conflict. "
    #                     "LARS uses its own 'weight_decay' parameter.")
    #     # Decide how to handle: del kwargs['weight_decay'] or use it, or raise error.
    #     # For now, we assume kwargs will not conflict.
    super().__init__(learning_rate=learning_rate, name=name, **kwargs)

    # Register LARS specific hyperparameters with Keras hyperparameter system
    # This helps with serialization and potential scheduling if needed (though not typical for these).
    self._set_hyper("lars_momentum", momentum) # Use a distinct name if 'momentum' is a base class hyper
    self._set_hyper("lars_eeta", eeta)
    # self._set_hyper("lars_weight_decay", weight_decay) # weight_decay is not typically a "hyper" in Keras sense for LARS

    # Store LARS-specific configuration parameters
    self.lars_momentum_val = momentum # Store the python float value
    self.lars_use_nesterov = use_nesterov
    self.lars_weight_decay_val = weight_decay # LARS's own weight decay
    self.lars_classic_momentum = classic_momentum
    self.lars_eeta_val = eeta # Store the python float value

    self.exclude_from_weight_decay = exclude_from_weight_decay
    if exclude_from_layer_adaptation is not None:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay


  def _create_slots(self, var_list: list[tf.Variable]):
    """Creates a 'momentum' slot for each variable."""
    for var in var_list:
      self.add_slot(var, "momentum_lars") # Use a LARS-specific slot name


  def _resource_apply_dense(self, grad: tf.Tensor, var: tf.Variable, apply_state=None):
    """Applies LARS update to a dense variable."""
    if grad is None: # Variable not used in graph if grad is None
      return tf.no_op() # Or tf.group([])

    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    
    # Get processed learning rate (handles schedules, etc.) from base Keras Optimizer
    lr_t = coefficients["lr_t"]
    
    # Get LARS hyperparameters
    # These were stored as direct attributes or can be fetched if set with _set_hyper
    current_momentum = self.lars_momentum_val # Using stored float
    current_eeta = self.lars_eeta_val         # Using stored float

    var_name = var.name
    momentum_var = self.get_slot(var, "momentum_lars")

    grad_with_wd = grad
    if self._use_weight_decay(var_name): # Checks self.lars_weight_decay_val
      grad_with_wd += self.lars_weight_decay_val * var

    # LARS trust ratio and update logic
    if self.lars_classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(var_name):
        w_norm = tf.norm(var, ord=2)
        g_norm = tf.norm(grad_with_wd, ord=2)
        trust_ratio = tf.where(
            tf.logical_and(tf.greater(w_norm, 0.0), tf.greater(g_norm, 0.0)),
            (current_eeta * w_norm) / (g_norm + 1e-12), # Add epsilon for stability
            1.0
        )
      scaled_lr = lr_t * trust_ratio # lr_t is the effective learning rate for this step

      next_momentum_val = current_momentum * momentum_var + scaled_lr * grad_with_wd
      if self.lars_use_nesterov:
        update = current_momentum * next_momentum_val + scaled_lr * grad_with_wd
      else:
        update = next_momentum_val
      
      next_var_val = var - update
    else: # Non-classic momentum (update scaled by LR *after* momentum accumulation)
      next_momentum_val = current_momentum * momentum_var + grad_with_wd
      if self.lars_use_nesterov:
        update_proposal = current_momentum * next_momentum_val + grad_with_wd
      else:
        update_proposal = next_momentum_val
      
      trust_ratio = 1.0
      if self._do_layer_adaptation(var_name):
        w_norm = tf.norm(var, ord=2)
        v_norm = tf.norm(update_proposal, ord=2) # Norm of the proposed update
        trust_ratio = tf.where(
            tf.logical_and(tf.greater(w_norm, 0.0), tf.greater(v_norm, 0.0)),
            (current_eeta * w_norm) / (v_norm + 1e-12), # Add epsilon
            1.0
        )
      scaled_lr = trust_ratio * lr_t # Trust ratio applied to effective learning rate
      next_var_val = var - scaled_lr * update_proposal

    # Assign new values to variable and momentum slot
    # self._use_locking is an attribute from the base Optimizer class
    var_update_op = var.assign(next_var_val, use_locking=self._use_locking)
    momentum_update_op = momentum_var.assign(next_momentum_val, use_locking=self._use_locking)
    
    return tf.group(var_update_op, momentum_update_op)

  def _use_weight_decay(self, var_name: str) -> bool:
    """Determines if LARS weight decay should be applied to this variable."""
    if not self.lars_weight_decay_val or self.lars_weight_decay_val == 0.0:
      return False
    if self.exclude_from_weight_decay:
      for pattern in self.exclude_from_weight_decay:
        if re.search(pattern, var_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, var_name: str) -> bool:
    """Determines if LARS layer adaptation (trust ratio) should be applied."""
    if self.exclude_from_layer_adaptation:
      for pattern in self.exclude_from_layer_adaptation:
        if re.search(pattern, var_name) is not None:
          return False
    return True

  def get_config(self) -> dict:
    """Returns the Keras-serializable configuration of the optimizer."""
    config = super().get_config()
    # `learning_rate` is already handled by the base class's get_config
    # Add LARS-specific parameters.
    config.update({
        "momentum": self.lars_momentum_val, # Or self._serialize_hyperparameter("lars_momentum")
        "use_nesterov": self.lars_use_nesterov,
        "weight_decay": self.lars_weight_decay_val, # LARS-specific weight decay
        "exclude_from_weight_decay": self.exclude_from_weight_decay,
        "exclude_from_layer_adaptation": self.exclude_from_layer_adaptation,
        "classic_momentum": self.lars_classic_momentum,
        "eeta": self.lars_eeta_val, # Or self._serialize_hyperparameter("lars_eeta")
    })
    # Remove Keras base optimizer's own weight_decay if it exists in config and we want LARS to be sole handler
    # This is important if base class adds its own 'weight_decay' to config.
    # if 'weight_decay' in config and config['weight_decay'] != self.lars_weight_decay_val:
    #    # This might happen if the base class has its own WD and it was non-zero.
    #    # We want to ensure our LARS WD is the one saved under 'weight_decay' key,
    #    # or use a distinct key like 'lars_weight_decay'.
    #    # The update above already sets 'weight_decay' to LARS's value.
    #    pass

    return config