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
"""Contrastive loss functions."""

from absl import flags
import tensorflow as tf # Changed from tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

LARGE_NUM = 1e9 # Used to mask out positive samples from the denominator of softmax


def add_supervised_loss(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
  """Computes mean supervised loss over a local batch.

  Args:
    labels: One-hot encoded true labels, shape (batch_size, num_classes).
    logits: Model output logits, shape (batch_size, num_classes).

  Returns:
    A scalar tensor representing the mean categorical cross-entropy loss.
  """
  # `from_logits=True` as model outputs raw scores.
  # `reduction=tf.keras.losses.Reduction.NONE` to get per-sample losses first.
  losses_per_sample = tf.keras.losses.categorical_crossentropy(
      y_true=labels, y_pred=logits, from_logits=True
  )
  # Explicitly take the mean over the batch.
  return tf.reduce_mean(losses_per_sample)


def add_contrastive_loss(
    hidden: tf.Tensor,
    hidden_norm: bool = True,
    temperature: float = 1.0,
    strategy: tf.distribute.Strategy | None = None
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for SimCLR.

  Args:
    hidden: Hidden vector (`Tensor`) of shape (2 * batch_size, dim).
            It contains concatenated representations of two augmented views (view1, view2)
            for each image in the batch.
    hidden_norm: Whether to apply L2 normalization to the hidden vectors.
    temperature: A float for temperature scaling in the softmax.
    strategy: A tf.distribute.Strategy object. If provided, loss is computed
              using cross-replica data.

  Returns:
    loss: A scalar tensor representing the mean contrastive loss.
    logits_con: Logits for the contrastive prediction task (e.g., hidden1 vs all_hidden2).
                Shape: (local_batch_size, global_batch_size_if_distributed_else_local).
    labels_con: Labels for the contrastive prediction task.
                Shape: (local_batch_size, 2 * global_batch_size_if_distributed_else_local).
  """
  # L2 normalize the hidden representations.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, axis=-1)
  
  # Split the hidden representations into two views.
  # hidden1 and hidden2 will each have shape (batch_size, dim).
  hidden1, hidden2 = tf.split(hidden, num_or_size_splits=2, axis=0)
  local_batch_size = tf.shape(hidden1)[0]

  # Prepare labels and gather hidden representations across replicas if distributed.
  if strategy is not None and strategy.num_replicas_in_sync > 1:
    # Gather hidden1 and hidden2 from all replicas.
    hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
    hidden2_large = tpu_cross_replica_concat(hidden2, strategy)
    global_batch_size = tf.shape(hidden1_large)[0] # Total batch size across all replicas

    replica_context = tf.distribute.get_replica_context()
    # replica_id_in_sync_group is already int32. Original had an extra cast.
    replica_id = replica_context.replica_id_in_sync_group
    
    # Create labels for positive pairs. Each replica is responsible for its local batch.
    # labels_idx points to the positive examples (view2_i for view1_i)
    # within the concatenated [hidden1_large, hidden2_large] or [hidden2_large, hidden1_large].
    labels_idx = tf.range(local_batch_size) + replica_id * local_batch_size
    
    # Labels for softmax_cross_entropy: one-hot, indicating the positive pair.
    # The logits will be [similarities_to_hidden2_large, similarities_to_hidden1_large].
    # So, positive pair (hidden1_i vs hidden2_i) is at labels_idx.
    labels_con = tf.one_hot(labels_idx, global_batch_size * 2)
    
    # Masks to remove self-comparisons (e.g., hidden1_i vs hidden1_i) from the denominator.
    # This mask is for logits_aa and logits_bb.
    masks = tf.one_hot(labels_idx, global_batch_size)
  else:
    # Non-distributed case (or single replica).
    hidden1_large = hidden1
    hidden2_large = hidden2
    global_batch_size = local_batch_size # Effective global batch size is local
    
    labels_con = tf.one_hot(tf.range(local_batch_size), local_batch_size * 2)
    masks = tf.one_hot(tf.range(local_batch_size), local_batch_size)

  # Calculate similarity matrices (logits).
  # hidden1 (local) vs hidden1_large (global/local)
  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_aa = logits_aa - masks * LARGE_NUM # Mask out diagonal (self-similarity)

  # hidden2 (local) vs hidden2_large (global/local)
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - masks * LARGE_NUM # Mask out diagonal

  # hidden1 (local) vs hidden2_large (global/local) - These are the key positive pairs
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  
  # hidden2 (local) vs hidden1_large (global/local)
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

  # Compute cross-entropy loss.
  # For loss_a: hidden1_i is compared against [all_hidden2_j, all_hidden1_k].
  # The label points to hidden2_i.
  # Changed from tf.nn.softmax_cross_entropy_with_logits
  loss_a = tf.keras.losses.categorical_crossentropy(
      y_true=labels_con,
      y_pred=tf.concat([logits_ab, logits_aa], axis=1),
      from_logits=True
  )
  # For loss_b: hidden2_i is compared against [all_hidden1_j, all_hidden2_k].
  # The label points to hidden1_i.
  loss_b = tf.keras.losses.categorical_crossentropy(
      y_true=labels_con,
      y_pred=tf.concat([logits_ba, logits_bb], axis=1),
      from_logits=True
  )
  
  loss = tf.reduce_mean(loss_a + loss_b)

  # logits_ab and labels_con can be used for metric calculation (e.g., contrastive accuracy).
  return loss, logits_ab, labels_con


def tpu_cross_replica_concat(tensor: tf.Tensor, strategy: tf.distribute.Strategy | None = None) -> tf.Tensor:
  """Concatenates `tensor` across TPU replicas.

  Args:
    tensor: The tensor to concatenate, specific to the current replica.
    strategy: A `tf.distribute.Strategy`. If None or single replica, returns tensor as is.

  Returns:
    A tensor of the same rank as `tensor`, where the first dimension is
    `num_replicas` times larger (if distributed), containing data from all replicas.
  """
  if strategy is None or strategy.num_replicas_in_sync <= 1:
    return tensor

  num_replicas = strategy.num_replicas_in_sync
  replica_context = tf.distribute.get_replica_context()

  with tf.name_scope('tpu_cross_replica_concat'):
    # Create an expanded tensor with a new leading dimension for replicas.
    # Each replica fills its part; others are zero.
    # E.g., if tensor shape is (B, D), ext_tensor shape is (num_replicas, B, D).
    scatter_indices = tf.expand_dims([replica_context.replica_id_in_sync_group], axis=-1) # Shape (1,1) for scatter_nd
    
    # Shape of the full tensor across replicas before all_reduce
    shape_with_replica_dim = tf.concat([[num_replicas], tf.shape(tensor)], axis=0)
    
    # Scatter local tensor into its slot in the larger tensor
    # tf.expand_dims(tensor, axis=0) makes its shape (1, B, D) to match updates requirements for scatter_nd
    ext_tensor = tf.scatter_nd(
        indices=scatter_indices,
        updates=tf.expand_dims(tensor, axis=0), # Add batch dim for updates
        shape=shape_with_replica_dim
    )

    # All-reduce (sum) to make every replica have the full tensor.
    # As each value is only on one replica and 0 elsewhere, SUM combines them.
    ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, ext_tensor)

    # Reshape to concatenate along the batch dimension.
    # If ext_tensor was (num_replicas, B, D), new shape is (num_replicas * B, D).
    # Original tensor rank
    tensor_rank = tf.rank(tensor)
    
    # Construct new shape: [num_replicas * original_batch_dim, other_dims...]
    # If tensor is scalar (rank 0), result is (num_replicas,)
    # If tensor is rank 1 (B,), result is (num_replicas * B,)
    # If tensor is rank > 1 (B, D1, D2...), result is (num_replicas * B, D1, D2...)
    if tensor_rank == 0: # Scalar case
        new_shape = [num_replicas]
    else: # Tensor with at least one dimension
        # Get shape without the replica dimension (which is ext_tensor.shape[1:])
        original_shape_dims = tf.shape(ext_tensor)[1:]
        first_dim_new_size = original_shape_dims[0] * num_replicas
        
        # If original tensor had rank 1 (e.g. (B,)), original_shape_dims[1:] would error.
        # So we handle rank 1 and rank > 1 carefully.
        if tensor_rank == 1:
            new_shape = [first_dim_new_size]
        else: # rank > 1
            remaining_dims = original_shape_dims[1:]
            new_shape = tf.concat([[first_dim_new_size], remaining_dims], axis=0)

    return tf.reshape(ext_tensor, new_shape)