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
"""Data pipeline."""

import functools
from absl import flags
from absl import logging

import tensorflow as tf # Changed
import tensorflow_datasets as tfds

# Assuming data_util is the previously updated module for image preprocessing
import data_util # Ensure this is the updated data_util.py

FLAGS = flags.FLAGS


def build_input_fn(builder: tfds.core.DatasetBuilder,
                   global_batch_size: int,
                   # topology: tf.tpu.experimental.Topology | None, # Topology not directly used in this function
                   is_training: bool):
  """Builds an input function for `tf.distribute.Strategy.distribute_datasets_from_function`.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size (across all replicas).
    is_training: Whether to build in training mode (for shuffling, repeating, etc.).

  Returns:
    A function that accepts a `tf.distribute.InputContext` and returns a `tf.data.Dataset`.
  """

  def _input_fn(input_context: tf.distribute.InputContext | None) -> tf.data.Dataset:
    """Inner input function that `distribute_datasets_from_function` will call."""
    # Determine per-replica batch size.
    # If input_context is None (e.g. not in a distributed setting or strategy handles it),
    # then per-replica batch size is just the global_batch_size (assuming 1 replica).
    if input_context:
        per_replica_batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        num_input_pipelines = input_context.num_input_pipelines
        logging.info(
            f'Creating dataset for input pipeline {input_context.input_pipeline_id} / {num_input_pipelines}'
        )
    else: # Fallback for non-distributed or when context is not provided by the strategy.
        # This case might need adjustment depending on how `distribute_datasets_from_function`
        # or the direct use of this input_fn handles it without a context.
        # For tf.distribute.MirroredStrategy, context is usually provided.
        # If running on a single device, context might be None.
        per_replica_batch_size = global_batch_size # Assume single pipeline/replica
        num_input_pipelines = 1
        logging.info('InputContext not provided, assuming single pipeline.')


    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', per_replica_batch_size)
    
    preprocess_fn_pretrain = get_preprocess_fn(is_training=is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training=is_training, is_pretrain=False)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
      """Applies preprocessing and augmentation. Produces multiple views if pretraining."""
      # Preprocessing logic based on training mode and pretrain/finetune flags
      if is_training and FLAGS.train_mode == 'pretrain':
        # For pretraining, create two augmented views of the image
        view1 = preprocess_fn_pretrain(image)
        view2 = preprocess_fn_pretrain(image)
        # Concatenate views along the channel dimension.
        # This is specific to how SimCLR's model expects input (e.g., if it splits them back).
        # Alternatively, could return a tuple or dict of views.
        # The original code concatenated.
        processed_image = tf.concat([view1, view2], axis=-1)
      else:
        # For fine-tuning or evaluation, apply fine-tuning/evaluation preprocessing
        processed_image = preprocess_fn_finetune(image)
      
      # One-hot encode the label
      one_hot_label = tf.one_hot(label, depth=num_classes)
      return processed_image, one_hot_label

    # Determine split based on training or evaluation
    split_name = FLAGS.train_split if is_training else FLAGS.eval_split
    
    # Configure TFDS read settings for distributed training
    read_config = tfds.ReadConfig(
        interleave_cycle_length=32, # Default was 16, original SimCLR used 32
        interleave_block_length=1,  # Default was 16, original SimCLR used 1
        input_context=input_context # Critical for sharding data across workers
    )

    dataset = builder.as_dataset(
        split=split_name,
        shuffle_files=is_training, # Shuffle files only during training
        as_supervised=True,        # Returns (image, label) tuples
        read_config=read_config
    )

    if FLAGS.cache_dataset:
      dataset = dataset.cache() # Cache after initial reading and shuffling of files

    if is_training:
      # Apply training-specific dataset transformations
      options = tf.data.Options()
      options.experimental_deterministic = False # For performance
      # options.experimental_slack = True # Might be removed or changed in newer TF
      # Check tf.data.Options() documentation for current performance options
      if hasattr(options, 'experimental_slack'):
            options.experimental_slack = True
      if hasattr(options, 'experimental_threading') and hasattr(options.experimental_threading, 'private_threadpool_size'):
            # Example: Configure inter-op and intra-op parallelism for map functions
            options.experimental_threading.private_threadpool_size = 4 * num_input_pipelines # Or AUTOTUNE
            options.experimental_threading.max_intra_op_parallelism = 1 # Often good for CPU-bound maps

      dataset = dataset.with_options(options)
      
      # Shuffle buffer size: A common heuristic is a multiple of batch_size
      # Ensure buffer_multiplier results in a reasonable shuffle buffer.
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10 # CIFAR vs ImageNet-like
      shuffle_buffer_size = per_replica_batch_size * buffer_multiplier
      dataset = dataset.shuffle(shuffle_buffer_size)
      
      dataset = dataset.repeat(-1) # Repeat indefinitely for training

    # Apply the map_fn for preprocessing and augmentation
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.AUTOTUNE # Changed from experimental.AUTOTUNE
    )
    
    # Batch the dataset
    dataset = dataset.batch(per_replica_batch_size, drop_remainder=is_training)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # Changed from experimental.AUTOTUNE
    
    return dataset

  return _input_fn


def build_distributed_dataset(builder: tfds.core.DatasetBuilder,
                              global_batch_size: int,
                              is_training: bool,
                              strategy: tf.distribute.Strategy,
                              topology: tf.tpu.experimental.Topology | None = None): # Topology can be None
  """Builds a distributed dataset for the given strategy.

  Args:
    builder: TFDS builder for the dataset.
    global_batch_size: The global batch size for training/evaluation.
    is_training: Boolean, whether to build the dataset for training.
    strategy: The tf.distribute.Strategy instance.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
              (Note: topology is not directly used by build_input_fn in this version,
               but kept for API consistency if data_lib needs it elsewhere).

  Returns:
    A `tf.distribute.DistributedDataset`.
  """
  # The `topology` argument is not directly used by `build_input_fn` itself now,
  # but it was in the original signature and might be used by other parts of data_lib.
  # The `input_context` provided by the strategy to `_input_fn` handles sharding.
  _input_fn_for_strategy = build_input_fn(builder, global_batch_size, is_training) # Removed topology
  return strategy.distribute_datasets_from_function(_input_fn_for_strategy)


def get_preprocess_fn(is_training: bool, is_pretrain: bool) -> functools.partial:
  """Gets the preprocessing function for an image.

  Args:
    is_training: Boolean, whether the image is for training.
    is_pretrain: Boolean, whether the image is for pretraining (SimCLR augmentations).

  Returns:
    A functools.partial object wrapping `data_util.preprocess_image`.
  """
  # Determine if test-style cropping should be used (center crop for eval)
  # For small images (e.g., CIFAR), test_crop is typically False even for eval.
  if FLAGS.image_size <= 32: # e.g., CIFAR-10/100
    test_crop_for_eval = False
  else: # e.g., ImageNet
    test_crop_for_eval = True
  
  # Color jitter strength: Apply only during pretraining and if training.
  # For fine-tuning (is_training=True, is_pretrain=False), less or no color jitter might be used.
  # The original code applies color_jitter_strength if is_pretrain is True.
  # is_training flag for preprocess_image controls other augmentations like random crop/flip.
  
  # For pretraining (is_pretrain=True), strong color jitter is used if is_training=True.
  # For fine-tuning (is_pretrain=False), color_jitter_strength is set to 0.
  # For evaluation (is_training=False), color_jitter_strength is also 0.
  
  current_color_jitter_strength = 0.0
  if is_training and is_pretrain: # Strong jitter only for pretraining phase and if in training mode
      current_color_jitter_strength = FLAGS.color_jitter_strength
  elif is_training and not is_pretrain: # Fine-tuning phase, might use milder or no jitter
      # current_color_jitter_strength = FLAGS.finetune_color_jitter_strength # If such a flag existed
      current_color_jitter_strength = 0.0 # Default to no jitter for fine-tuning based on original logic
  # else: eval phase, current_color_jitter_strength remains 0.0

  return functools.partial(
      data_util.preprocess_image, # Assumes data_util.py is updated
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      is_training=is_training, # Controls random crops, flips
      color_jitter_strength=current_color_jitter_strength, # Controls color augmentations
      test_crop=test_crop_for_eval, # Controls whether eval uses center crop
      impl=getattr(FLAGS, 'data_aug_impl', 'simclrv2') # Pass impl if available in FLAGS, default 'simclrv2'
                                                      # Assuming 'data_aug_impl' flag might exist
  )