# In lars_optimizer.py

import re
import tensorflow as tf
# import keras # Not strictly needed here if using tf.keras.optimizers.Optimizer

EETA_DEFAULT = 0.001

class LARSOptimizer(tf.keras.optimizers.Optimizer):
  """Layer-wise Adaptive Rate Scaling (LARS) optimizer.
  (Your full docstring)
  """

  def __init__(self,
               learning_rate, # Required by Keras 3 Optimizer base class
               momentum: float = 0.9,
               use_nesterov: bool = False,
               weight_decay: float = 0.0, # LARS-specific weight decay
               exclude_from_weight_decay: list[str] | None = None,
               exclude_from_layer_adaptation: list[str] | None = None,
               classic_momentum: bool = True,
               eeta: float = EETA_DEFAULT,
               name: str = "LARSOptimizer",
               **kwargs):
    """Constructs a LARSOptimizer."""
    
    # Keras 3 Optimizer's __init__ might also accept 'momentum'.
    # If it does, you could pass it: super().__init__(learning_rate=learning_rate, momentum=momentum, name=name, **kwargs)
    # For now, let's assume we are managing LARS momentum separately.
    super().__init__(learning_rate=learning_rate, name=name, **kwargs)

    # Store LARS-specific configuration parameters as direct attributes.
    # We are no longer using _set_hyper for these custom LARS params,
    # as Keras 3 might not expose _set_hyper for arbitrary keys or handle them
    # as schedulable hyperparameters unless the base class is designed for it.
    self.lars_momentum = momentum
    self.lars_use_nesterov = use_nesterov
    self.lars_weight_decay = weight_decay # LARS's own weight decay
    self.lars_classic_momentum = classic_momentum
    self.lars_eeta = eeta

    self.exclude_from_weight_decay_list = exclude_from_weight_decay # Renamed to avoid confusion
    if exclude_from_layer_adaptation is not None:
      self.exclude_from_layer_adaptation_list = exclude_from_layer_adaptation # Renamed
    else:
      self.exclude_from_layer_adaptation_list = exclude_from_weight_decay


  def _create_slots(self, var_list: list[tf.Variable]):
    for var in var_list:
      self.add_slot(var, "momentum_lars")


  def _resource_apply_dense(self, grad: tf.Tensor, var: tf.Variable, apply_state=None):
    if grad is None:
      return tf.no_op() 

    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    
    lr_t = coefficients["lr_t"] # Processed learning rate from Keras base
    
    # Use direct attributes for LARS parameters
    current_momentum = self.lars_momentum
    current_eeta = self.lars_eeta

    var_name = var.name
    momentum_var = self.get_slot(var, "momentum_lars")

    grad_with_wd = grad
    if self._use_weight_decay(var_name): # Checks self.lars_weight_decay
      grad_with_wd += self.lars_weight_decay * var

    if self.lars_classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(var_name):
        w_norm = tf.norm(var, ord=2)
        g_norm = tf.norm(grad_with_wd, ord=2)
        trust_ratio = tf.where(
            tf.logical_and(tf.greater(w_norm, 0.0), tf.greater(g_norm, 0.0)),
            (current_eeta * w_norm) / (g_norm + 1e-12),
            1.0
        )
      scaled_lr = lr_t * trust_ratio

      next_momentum_val = current_momentum * momentum_var + scaled_lr * grad_with_wd
      if self.lars_use_nesterov:
        update = current_momentum * next_momentum_val + scaled_lr * grad_with_wd
      else:
        update = next_momentum_val
      next_var_val = var - update
    else: 
      next_momentum_val = current_momentum * momentum_var + grad_with_wd
      if self.lars_use_nesterov:
        update_proposal = current_momentum * next_momentum_val + grad_with_wd
      else:
        update_proposal = next_momentum_val
      
      trust_ratio = 1.0
      if self._do_layer_adaptation(var_name):
        w_norm = tf.norm(var, ord=2)
        v_norm = tf.norm(update_proposal, ord=2)
        trust_ratio = tf.where(
            tf.logical_and(tf.greater(w_norm, 0.0), tf.greater(v_norm, 0.0)),
            (current_eeta * w_norm) / (v_norm + 1e-12),
            1.0
        )
      scaled_lr = trust_ratio * lr_t
      next_var_val = var - scaled_lr * update_proposal

    var_update_op = var.assign(next_var_val, use_locking=self._use_locking)
    momentum_update_op = momentum_var.assign(next_momentum_val, use_locking=self._use_locking)
    
    return tf.group(var_update_op, momentum_update_op)

  def _use_weight_decay(self, var_name: str) -> bool:
    if not self.lars_weight_decay or self.lars_weight_decay == 0.0:
      return False
    if self.exclude_from_weight_decay_list: # Use renamed attribute
      for pattern in self.exclude_from_weight_decay_list:
        if re.search(pattern, var_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, var_name: str) -> bool:
    if self.exclude_from_layer_adaptation_list: # Use renamed attribute
      for pattern in self.exclude_from_layer_adaptation_list:
        if re.search(pattern, var_name) is not None:
          return False
    return True

  def get_config(self) -> dict:
    config = super().get_config()
    # `learning_rate` is handled by the base class's get_config.
    # Add LARS-specific parameters that were stored as direct attributes.
    config.update({
        "momentum": self.lars_momentum,
        "use_nesterov": self.lars_use_nesterov,
        "weight_decay": self.lars_weight_decay, # LARS-specific
        "exclude_from_weight_decay": self.exclude_from_weight_decay_list, # Use renamed
        "exclude_from_layer_adaptation": self.exclude_from_layer_adaptation_list, # Use renamed
        "classic_momentum": self.lars_classic_momentum,
        "eeta": self.lars_eeta,
    })
    return config