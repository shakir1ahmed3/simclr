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
"""The main training pipeline."""

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
"""The main training pipeline."""

import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf # Changed
import tensorflow_datasets as tfds

# Assuming these are local modules updated for TF2
import data as data_lib
import metrics
import model as model_lib
import objective as obj_lib


FLAGS = flags.FLAGS


flags.DEFINE_float(
    'learning_rate', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_enum(
    'learning_rate_scaling', 'linear', ['linear', 'sqrt'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'train_batch_size', 512,
    'Batch size for training.')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_steps', 0,
    'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer(
    'eval_steps', 0,
    'Number of steps to eval for. If not provided, evals over entire dataset.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string(
    'eval_split', 'validation',
    'Split for evaluation.')

flags.DEFINE_string(
    'dataset', 'imagenet2012',
    'Name of a dataset.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('lineareval_while_pretraining', True,
                  'Whether to finetune supervised head while pretraining.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

flags.DEFINE_string(
    'master', None,
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_bool(
    'use_tpu', False, # Changed default to False for easier local testing
    'Whether to run on TPU.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_enum(
    'optimizer', 'lars', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Whether to L2 normalize hidden representations for contrastive loss.') # Clarified doc

flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 3,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0, # Corrected: Original comment was ambiguous
    'Which layer of the projection head to use as input for supervised fine-tuning. '
    '0 means the input to the projection head (output of resnet backbone). '
    'Positive N means the N-th HiddensList output from projection head (1-indexed based on list appends).'
    '-1 means the final output of the projection head.')


flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 50,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')


# Removed get_salient_tensors_dict as it relied on tf.compat.v1.get_default_graph()
# and specific tensor names that might change with TF2 Keras model structure.
# Exporting specific tensors for TF Hub would require a TF2-idiomatic way,
# often by returning a dictionary of named outputs from the model's call function
# or by creating specific sub-models for export.


def build_saved_model(model: model_lib.Model, include_projection_head: bool = True):
    """Returns a tf.Module for saving to SavedModel."""

    class SimCLRModelExportable(tf.Module):
        """Saved model for exporting."""

        def __init__(self, model_to_export: model_lib.Model):
            super().__init__(name="SimCLRModelExportable")
            self.model = model_to_export
            # Access trainable variables via self.model.trainable_variables

        # Define a serving signature.
        # This example returns the final average pool output and supervised logits.
        # Adjust as needed for specific Hub export requirements.
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32, name="input_images")])
        def __call__(self, inputs):
            # In serving/inference, training is typically False.
            projection_output, supervised_output = self.model(inputs, training=False)
            
            # Example outputs:
            outputs_dict = {}
            # The actual "final_avg_pool" would be an intermediate output of the ResNet.
            # This requires the ResNet model to be structured to return it or to access it.
            # For simplicity, let's assume supervised_head_inputs IS the avg_pool output for now.
            # If projection_output is (batch, feat_dim) and supervised_output is (batch, num_classes)
            # and if supervised_head_inputs was the avg_pool.
            
            # To get specific intermediate layer outputs in TF2 Keras,
            # you'd typically define a new Model with multiple outputs, e.g.:
            # feature_extractor = tf.keras.Model(inputs=model.input, 
            #                                    outputs=[model.get_layer('final_avg_pool_layer_name').output, ...])
            # Or, design the main model's call method to return a dictionary of tensors.
            # The original get_salient_tensors_dict relied on graph tensor names, which is TF1 style.

            # Placeholder: For a real Hub module, one needs to carefully define these.
            # Let's assume the model's projection head gives 'h' (input to proj) and 'z' (output of proj)
            # And the resnet model itself has an accessible avg_pool output.
            # This part needs careful redesign if get_salient_tensors_dict functionality is critical.
            # For now, let's just return what the model gives.
            if projection_output is not None:
                outputs_dict['projection_output'] = projection_output # This is 'z'
            if supervised_output is not None:
                outputs_dict['logits_sup'] = supervised_output

            # If the model's `_projection_head` call returns `(z, h_for_finetune)`
            # and `h_for_finetune` is the `final_avg_pool` before projection.
            # This logic needs to align with how `model_lib.Model` is structured.
            # The current `model_lib.Model` returns `(projection_head_outputs, supervised_head_outputs)`.
            # `projection_head_outputs` is z.
            # `supervised_head_inputs` (internal to model_lib.Model) is the feature before supervised head.
            # This `supervised_head_inputs` is often the `final_avg_pool` or `h`.

            # To replicate get_salient_tensors_dict, the model would need to be
            # refactored or wrapped to expose these specific tensors by name.
            # E.g., make model.call return a dict, or create a sub-model.
            # For this update, we keep it simple.
            
            return outputs_dict

    module = SimCLRModelExportable(model)
    return module


def save_model_for_export(model_to_save: model_lib.Model, global_step_val: int):
    """Export as SavedModel for fine-tuning and inference."""
    # The `build_saved_model` needs to be adapted for TF2.
    # The original relied on graph tensor names which is fragile.
    # A TF2 approach would be to have the Keras model return a dictionary of named outputs,
    # or to create a new `tf.keras.Model` that exposes the desired intermediate layers.
    
    # Simpler export for now:
    export_dir = os.path.join(FLAGS.model_dir, 'saved_model_export') # Use a different dir name
    checkpoint_export_dir = os.path.join(export_dir, str(global_step_val))
    
    if tf.io.gfile.exists(checkpoint_export_dir):
        logging.info(f"Removing existing export directory: {checkpoint_export_dir}")
        tf.io.gfile.rmtree(checkpoint_export_dir)
    
    # Save the Keras model directly.
    # For more control over signatures, use tf.saved_model.save with a tf.Module.
    # model_to_save.save(checkpoint_export_dir, include_optimizer=False, save_format='tf')
    
    # Using the tf.Module approach (needs build_saved_model to be fully TF2-idiomatic)
    # This part is highly dependent on how `build_saved_model` is implemented
    # to extract salient tensors in a TF2/Keras way.
    # For now, let's assume a basic save of the model, which might not match Hub requirements.
    try:
        # tf.saved_model.save(model_to_save, checkpoint_export_dir) # Direct save of Keras model
        # Or using the custom module (if build_saved_model is fully updated)
        exportable_module = build_saved_model(model_to_save)
        tf.saved_model.save(exportable_module, checkpoint_export_dir)
        logging.info(f"SavedModel exported to {checkpoint_export_dir}")
    except Exception as e:
        logging.error(f"Failed to save model for export: {e}")
        return # Skip cleaning if save failed

    if FLAGS.keep_hub_module_max > 0:
        # Delete old exported SavedModels.
        exported_steps = []
        if tf.io.gfile.exists(export_dir):
            for subdir in tf.io.gfile.listdir(export_dir):
                if subdir.isdigit():
                    exported_steps.append(int(subdir))
        exported_steps.sort()
        for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
            tf.io.gfile.rmtree(os.path.join(export_dir, str(step_to_delete)))


def try_restore_from_checkpoint(model, optimizer): # Removed global_step from args, use optimizer.iterations
    """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
    # Pass optimizer.iterations directly to Checkpoint.
    # It needs to be a tf.Variable. Optimizer.iterations is.
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, epoch=tf.Variable(0), global_step=optimizer.iterations) # Added epoch
    
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=FLAGS.model_dir,
        max_to_keep=FLAGS.keep_checkpoint_max)
    
    latest_ckpt = checkpoint_manager.latest_checkpoint
    if latest_ckpt:
        logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
        # expect_partial() is good for flexibility during development/refactoring
        status = checkpoint.restore(latest_ckpt)
        status.assert_existing_objects_matched().expect_partial() # More specific check
        logging.info(f"Restored model epoch: {checkpoint.epoch.numpy()}, step: {optimizer.iterations.numpy()}")
    elif FLAGS.checkpoint:
        logging.info('Restoring from given checkpoint (weights only): %s', FLAGS.checkpoint)
        # For restoring weights only, create a checkpoint with only the model
        model_checkpoint = tf.train.Checkpoint(model=model)
        status = model_checkpoint.restore(FLAGS.checkpoint)
        status.assert_existing_objects_matched().expect_partial() # Check that model weights were matched

        if FLAGS.zero_init_logits_layer:
            if hasattr(model, 'supervised_head') and model.supervised_head is not None:
                output_layer_parameters = model.supervised_head.trainable_weights
                logging.info('Zero initializing output layer parameters %s',
                             [x.name for x in output_layer_parameters])
                for x_var in output_layer_parameters:
                    x_var.assign(tf.zeros_like(x_var))
            else:
                logging.warning("zero_init_logits_layer is True, but model has no supervised_head or it's None.")
    else:
        logging.info("No checkpoint found. Starting training from scratch.")

    return checkpoint_manager


def json_serializable(val):
  # Check if val is a tf.Tensor, convert to numpy if so.
  if isinstance(val, tf.Tensor):
      val = val.numpy()
  # Check for NumPy types that might not be directly serializable
  if hasattr(val, 'dtype'): # Basic check for numpy types
      if isinstance(val, (tf.dtypes.DType, type(None))): # tf.DType is not serializable
          return False
      if isinstance(val, tf.TensorShape):
          return False # TensorShape is not directly serializable
      # Convert numpy types to Python equivalents
      if isinstance(val, (np.generic,)): # Catches np.int_, np.float_, etc.
          val = val.item()

  try:
    json.dumps(val)
    return True
  except (TypeError, OverflowError):
    return False


def perform_evaluation(model, builder, eval_steps, ckpt_path, strategy, topology): # Renamed ckpt to ckpt_path
  """Perform evaluation."""
  if FLAGS.train_mode == 'pretrain' and not FLAGS.lineareval_while_pretraining:
    logging.info('Skipping eval during pretraining without linear eval.')
    return {} # Return empty dict or None
  logging.info("Attempting to build training dataset iterator...")
  # Build input pipeline.
  ds = data_lib.build_distributed_dataset(builder, FLAGS.eval_batch_size, is_training=False, # explicit is_training
                                          strategy=strategy, topology=topology) # Pass topology
  logging.info("Training dataset iterator created successfully.")
  eval_summary_dir = os.path.join(FLAGS.model_dir, "eval_" + (FLAGS.eval_name or FLAGS.eval_split))
  summary_writer = tf.summary.create_file_writer(eval_summary_dir)

  # Build metrics.
  with strategy.scope():
    regularization_loss_metric = tf.keras.metrics.Mean('eval/regularization_loss')
    # Use CategoricalAccuracy if labels are one-hot, SparseCategoricalAccuracy if integer
    label_top_1_accuracy_metric = tf.keras.metrics.CategoricalAccuracy( # Assuming one-hot labels
        name='eval/label_top_1_accuracy')
    label_top_5_accuracy_metric = tf.keras.metrics.TopKCategoricalAccuracy(
        k=5, name='eval/label_top_5_accuracy') # k=5
    
    all_metrics_list = [
        regularization_loss_metric, label_top_1_accuracy_metric, label_top_5_accuracy_metric
    ]

    # Restore checkpoint.
    logging.info('Restoring from %s for evaluation', ckpt_path)
    # Create a new tf.Variable for global_step for eval, not affecting optimizer's step.
    eval_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="eval_global_step")
    checkpoint_eval = tf.train.Checkpoint(model=model, global_step=eval_global_step)
    status = checkpoint_eval.restore(ckpt_path)
    status.assert_existing_objects_matched().expect_partial() # Ensure model weights are loaded
    
    # Try to extract step from checkpoint path if possible (e.g., "ckpt-12345")
    try:
        step_from_ckpt = int(ckpt_path.split('-')[-1])
        eval_global_step.assign(step_from_ckpt)
    except (ValueError, IndexError):
        logging.warning(f"Could not parse step from checkpoint path: {ckpt_path}. Using 0 for summary step.")
        # eval_global_step remains 0 or its restored value if 'global_step' was in the ckpt.

    logging.info('Performing eval at step %d (from ckpt: %s)', eval_global_step.numpy(), ckpt_path)

  @tf.function
  def distributed_eval_step(dist_inputs):
    per_replica_features, per_replica_labels = dist_inputs
    def single_replica_eval_step(features, labels_dict):
        _, supervised_head_outputs = model(features, training=False)
        assert supervised_head_outputs is not None, "Supervised head output is None during eval"
        
        # Ensure labels_dict['labels'] is in the correct format (e.g., one-hot)
        true_labels_one_hot = labels_dict['labels'] 
        
        # If metrics were SparseCategoricalAccuracy, you would do tf.argmax(true_labels_one_hot, axis=-1)
        # For CategoricalAccuracy and TopKCategoricalAccuracy, one-hot is expected.
        label_top_1_accuracy_metric.update_state(true_labels_one_hot, supervised_head_outputs)
        label_top_5_accuracy_metric.update_state(true_labels_one_hot, supervised_head_outputs)
        
        reg_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True) # Or False if LARS not used
        regularization_loss_metric.update_state(reg_loss)

    strategy.run(single_replica_eval_step, args=(per_replica_features, per_replica_labels))

  iterator = iter(ds)
  for i in tf.range(eval_steps): # Use tf.range for tf.function context
    distributed_eval_step(next(iterator))
    if (i+1) % 100 == 0: # Log progress every 100 steps
        logging.info('Completed eval for %d / %d steps', i.numpy() + 1, eval_steps)
  logging.info('Finished eval for checkpoint: %s', ckpt_path)

  # Write summaries
  cur_eval_step_val = eval_global_step.numpy()
  logging.info('Writing evaluation summaries for step %d', cur_eval_step_val)
  with summary_writer.as_default():
    metrics.log_and_write_metrics_to_summary(all_metrics_list, cur_eval_step_val)
    summary_writer.flush()

  # Record results as JSON.
  result_json_path = os.path.join(FLAGS.model_dir, f'result_eval_{FLAGS.eval_name or FLAGS.eval_split}.json')
  result = {metric.name: metric.result().numpy() for metric in all_metrics_list}
  result['global_step_evald'] = cur_eval_step_val # Step of the checkpoint being evaluated
  logging.info(f"Evaluation results: {result}")
  
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  
  # Also save a versioned result file if eval_name is not specified (for multiple evals)
  # Or use eval_name if provided to make it distinct.
  specific_result_filename = f'result_eval_{FLAGS.eval_name or FLAGS.eval_split}_{cur_eval_step_val}.json'
  result_json_path_step = os.path.join(FLAGS.model_dir, specific_result_filename)
  with tf.io.gfile.GFile(result_json_path_step, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)

  # Only save flags once, not per evaluation.
  # Placed it here for now, but ideally done once at the start of training.
  flag_json_path = os.path.join(FLAGS.model_dir, 'flags_config.json')
  if not tf.io.gfile.exists(flag_json_path):
      with tf.io.gfile.GFile(flag_json_path, 'w') as f:
        serializable_flags = {}
        for key, val in FLAGS.flag_values_dict().items():
          if json_serializable(val):
            serializable_flags[key] = val
        json.dump(serializable_flags, f, indent=4)

  # Save model after evaluation (optional, usually done during training checkpoints)
  # save_model_for_export(model, global_step_val=cur_eval_step_val)

  # Reset metrics for next potential evaluation
  for metric in all_metrics_list:
    metric.reset_states()
    
  return result


# This function was for restoring from pretrain, now integrated into try_restore_from_checkpoint
# def _restore_latest_or_from_pretrain(checkpoint_manager): ...


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Ensure model_dir is set
  if not FLAGS.model_dir:
      raise ValueError("model_dir must be set.")
  tf.io.gfile.makedirs(FLAGS.model_dir) # Create model_dir if it doesn't exist

  builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
  builder.download_and_prepare()
  num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
  num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
  num_classes = builder.info.features['label'].num_classes

  train_steps = model_lib.get_train_steps(num_train_examples) # Uses FLAGS.train_epochs and FLAGS.train_batch_size
  if FLAGS.train_steps > 0 : # Override if FLAGS.train_steps is provided
      train_steps = FLAGS.train_steps
      
  eval_steps = FLAGS.eval_steps or int(
      math.ceil(num_eval_examples / FLAGS.eval_batch_size))
  epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size)) if FLAGS.train_batch_size > 0 else 0

  logging.info('# train examples: %d', num_train_examples)
  logging.info('# train_steps: %d', train_steps)
  logging.info('# eval examples: %d', num_eval_examples)
  logging.info('# eval steps: %d', eval_steps)
  logging.info('# steps_per_epoch: %d', epoch_steps)


  checkpoint_steps = FLAGS.checkpoint_steps
  if checkpoint_steps <= 0 and epoch_steps > 0: # If not set by flag, use checkpoint_epochs
      checkpoint_steps = FLAGS.checkpoint_epochs * epoch_steps
  if checkpoint_steps <= 0 : # Default to a reasonable value if still 0 (e.g. dataset too small)
      checkpoint_steps = 1000 # Fallback checkpointing steps
      logging.warning(f"Checkpoint steps were 0 or invalid, defaulting to {checkpoint_steps}")
  
  logging.info("# checkpoint_steps: %d", checkpoint_steps)


  # TPU / GPU Strategy
  strategy = None
  topology = None # Keep topology for data lib if needed, even for GPU
  if FLAGS.use_tpu:
    if FLAGS.tpu_name:
      cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    elif FLAGS.master: # For local TPUs or direct gRPC address
         cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.master)
    else: # Try to auto-discover
        try:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            logging.error("Could not auto-discover TPU. Please provide tpu_name or master.")
            raise
            
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    logging.info('TPU Topology:')
    logging.info('num_tasks: %d', topology.num_tasks)
    logging.info('num_tpus_per_task: %d', topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(extended=tf.tpu.experimental.TPUExtended(cluster_resolver)) # Use extended for topology
  else:
    # MirroredStrategy for GPUs or CPU
    # strategy = tf.distribute.MirroredStrategy()
    # Or for single device:
    strategy = tf.distribute.get_strategy() # Gets default (Mirrored if >1 GPU, OneDevice if 1 GPU/CPU)
    if isinstance(strategy, tf.distribute.MirroredStrategy):
        logging.info('Running using MirroredStrategy on %d replicas', strategy.num_replicas_in_sync)
    else:
        logging.info('Running on a single device.')


  with strategy.scope():
    model = model_lib.Model(num_classes=num_classes) # Pass num_classes
    # Build LR schedule and optimizer.
    # Learning rate schedule needs to be created within strategy scope if it contains tf.Variables (not typical for schedules)
    learning_rate_schedule = model_lib.WarmUpAndCosineDecay(
        base_learning_rate=FLAGS.learning_rate,
        num_examples=num_train_examples
    )
    optimizer = model_lib.build_optimizer(learning_rate_schedule)


  if FLAGS.mode == 'eval':
    # Checkpoints iterator will yield checkpoint paths
    for ckpt_path in tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15):
      eval_result = perform_evaluation(model, builder, eval_steps, ckpt_path, strategy, topology)
      # Check if training is complete based on the evaluated checkpoint's step
      # This requires parsing the step from ckpt_path or storing it in eval_result
      # Example: if 'global_step_evald' in eval_result and eval_result['global_step_evald'] >= train_steps:
      current_ckpt_step = 0
      try:
          current_ckpt_step = int(ckpt_path.split('-')[-1])
      except: pass
      if current_ckpt_step >= train_steps and train_steps > 0:
          logging.info(f'Evaluation for step {current_ckpt_step} (>=train_steps {train_steps}) complete. Exiting...')
          break # Exit after evaluating the final checkpoint or one beyond
    return # Exit after eval mode is done
  

  # Training or Train then Eval Mode
  train_summary_dir = os.path.join(FLAGS.model_dir, "train")
  summary_writer = tf.summary.create_file_writer(train_summary_dir)
  
  with strategy.scope():
    # Build metrics.
    all_metrics_list_train = []
    weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
    total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
    all_metrics_list_train.extend([weight_decay_metric, total_loss_metric])
    
    if FLAGS.train_mode == 'pretrain':
      contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
      contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc') # Or dedicated Accuracy metric
      contrast_entropy_metric = tf.keras.metrics.Mean('train/contrast_entropy')
      all_metrics_list_train.extend([
          contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric
      ])
    if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
      supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
      supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc') # Or CategoricalAccuracy
      all_metrics_list_train.extend([supervised_loss_metric, supervised_acc_metric])

    # Restore checkpoint if available.
    # Optimizer iterations (global_step) is part of the optimizer state.
    checkpoint_manager = try_restore_from_checkpoint(model, optimizer)

  # Distributed training step function
  @tf.function
  def distributed_train_step(dist_inputs):
    per_replica_features, per_replica_labels = dist_inputs # Unpack iterator output

    def single_replica_train_step(features, labels_dict): # labels_dict {'labels': actual_labels}
      with tf.GradientTape() as tape:
        # Image summary (conditional)
        # Note: optimizer.iterations is a tf.Variable, access .value() or .numpy() outside tf.function
        # Inside tf.function, use it directly as a tensor.
        # current_iter = optimizer.iterations
        # should_record_img_summary = tf.equal((current_iter + 1) % checkpoint_steps, 0)
        # For simplicity, let's not do image summary inside tf.function directly, or do it less frequently.
        # with tf.summary.record_if(should_record_img_summary):
        #    tf.summary.image('image', features[:, :, :, :3], step=current_iter + 1, max_outputs=1)

        projection_head_outputs, supervised_head_outputs = model(features, training=True)
        
        loss = None
        current_loss_components = {} # For debugging or detailed logging

        if projection_head_outputs is not None:
          outputs_proj = projection_head_outputs # Simpler var name
          con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
              outputs_proj,
              hidden_norm=FLAGS.hidden_norm,
              temperature=FLAGS.temperature,
              strategy=strategy) # Pass strategy for cross-replica loss
          loss = con_loss if loss is None else loss + con_loss
          current_loss_components['contrastive'] = con_loss
          
          # Update pretrain metrics (if they exist)
          if FLAGS.train_mode == 'pretrain':
              metrics.update_pretrain_metrics_train(contrast_loss_metric,
                                                    contrast_acc_metric,
                                                    contrast_entropy_metric,
                                                    con_loss, logits_con,
                                                    labels_con)
        
        if supervised_head_outputs is not None:
          outputs_sup = supervised_head_outputs
          true_labels = labels_dict['labels']
          # For lineareval_while_pretraining, labels might need duplication if model processes augmented views separately
          if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # Assuming projection_head_outputs corresponds to 2*batch_size because of augmentations,
            # and supervised_head_inputs was derived from the same.
            # So, true_labels might need to be tiled.
            # This depends on how `model` and `obj_lib.add_supervised_loss` handle it.
            # Original: l = tf.concat([l, l], 0). This assumes supervised_head_outputs has 2*N batch size.
            # Let's assume model's supervised_head_outputs matches shape of true_labels.
            # If `model` handles the duplication of inputs for supervised head, this is fine.
            # The current model_lib.Model's call for 'pretrain' and 'lineareval_while_pretraining' takes
            # hiddens (from 2*N augmented views), gives (z, h_for_sup) from proj_head, then
            # supervised_head(tf.stop_gradient(h_for_sup)).
            # So, supervised_head_outputs will have batch_size 2*N.
            # Thus, true_labels need to be tiled.
            true_labels = tf.concat([true_labels, true_labels], axis=0)
            
          sup_loss = obj_lib.add_supervised_loss(labels=true_labels, logits=outputs_sup)
          loss = sup_loss if loss is None else loss + sup_loss
          current_loss_components['supervised'] = sup_loss
          
          # Update finetune metrics (if they exist)
          if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
              metrics.update_finetune_metrics_train(supervised_loss_metric,
                                                    supervised_acc_metric, sup_loss,
                                                    true_labels, outputs_sup)

        if loss is None: # Should not happen if model is configured correctly for train_mode
            raise ValueError("Loss was not computed. Check model outputs and train_mode.")

        # Weight decay
        wd_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True) # Or False
        weight_decay_metric.update_state(wd_loss)
        loss += wd_loss
        current_loss_components['weight_decay'] = wd_loss
        
        total_loss_metric.update_state(loss)
        
        # Scale loss for multi-replica training
        scaled_loss = loss / strategy.num_replicas_in_sync
      
      # Calculate and apply gradients
      trainable_vars = model.trainable_variables
      grads = tape.gradient(scaled_loss, trainable_vars)
      optimizer.apply_gradients(zip(grads, trainable_vars))
      
      # Log losses for this step (optional, can be verbose)
      # tf.print("Step Losses:", current_loss_components)


    # Training loop
    # Build training dataset iterator
# In run.py, inside main(), right after the scope for checkpoint_manager:

    logging.info("Checkpoint logic complete. Attempting to create data iterator...") # NEW LOG 1
    try:
        train_iterator = iter(data_lib.build_distributed_dataset(
            builder, FLAGS.train_batch_size, is_training=True, strategy=strategy, topology=topology
        ))
        logging.info("Data iterator CREATED successfully.") # NEW LOG 2
    except Exception as e:
        logging.error(f"ERROR CREATING DATA ITERATOR: {e}", exc_info=True) # NEW LOG 3 (if error)
        raise # Stop if iterator fails

    logging.info("Starting training...") # Your existing log
# ... (rest of the training loop) ...
    # Get current step from optimizer.iterations
    # optimizer.iterations is a tf.Variable
    
    while optimizer.iterations.numpy() < train_steps:
        current_step_val = optimizer.iterations.numpy()
        
        # Log image summary (less frequently, outside tf.function)
        if (current_step_val +1) % (checkpoint_steps * 5) == 0: # Even less frequent for images
            with summary_writer.as_default():
                # Get one batch of features for image summary
                # This requires re-building a temporary iterator or getting one sample
                # For simplicity, this is often done with a separate, non-distributed dataset call
                # or by caching one batch. For now, we skip detailed image summary inside the loop.
                # Example:
                # temp_ds_for_img = data_lib.build_dataset(builder, 1, False, strategy, topology).take(1)
                # for img_feat, _ in temp_ds_for_img:
                #    tf.summary.image('image_sample', img_feat[:, :, :, :3], step=optimizer.iterations + 1, max_outputs=1)
                #    break
                pass


        # Run a loop of steps
        # `tf.range` is needed so that this runs in a `tf.while_loop` and is not unrolled in AutoGraph.
        # However, for eager execution style loop, direct python loop is also fine.
        # For performance, tf.function tracing one `train_multiple_steps` is better.
        
        # Define train_multiple_steps for AutoGraph to compile one loop of training steps
        # This function will be traced by tf.function.
        @tf.function
        def train_loop_body(iterator_ref): # Pass iterator by reference (as a list or dict) if needed for re-creation
            for _ in tf.range(checkpoint_steps): # tf.range for XLA compilation
                # Check if we've reached total train_steps
                if optimizer.iterations >= train_steps:
                    break
                # The name_scope was for TF1 graph naming, less critical in TF2 eager/autograph
                # with tf.name_scope(''): 
                images, labels = next(iterator_ref)
                # The data pipeline should yield features and labels in the format expected by distributed_train_step
                # Typically, (per_replica_features, per_replica_labels)
                # The current data_lib.build_distributed_dataset likely returns this.
                distributed_train_step((images, {'labels': labels}))


        # Call the compiled training loop
        try:
            with summary_writer.as_default(): # Ensure summary writer is default for this scope
                 train_loop_body(train_iterator)
        except tf.errors.OutOfRangeError:
            logging.info("Training dataset exhausted. Re-initializing iterator.")
            train_iterator = iter(data_lib.build_distributed_dataset(
                builder, FLAGS.train_batch_size, is_training=True, strategy=strategy, topology=topology
            ))
            # Potentially run the remaining steps if any, or break if epoch-based logic dominates.
            # This example assumes step-based completion.
            # If an epoch finishes mid-loop, the next loop iteration will use the new iterator.
            continue # Continue to the outer while loop check


        cur_step_after_loop = optimizer.iterations.numpy()
        logging.info('Steps executed: %d. Current step: %d / %d', checkpoint_steps, cur_step_after_loop, train_steps)

        # Save checkpoint
        checkpoint_manager.save(checkpoint_number=cur_step_after_loop)
        logging.info(f'Checkpoint saved for step {cur_step_after_loop}.')

        # Log metrics and write summaries
        with summary_writer.as_default():
            metrics.log_and_write_metrics_to_summary(all_metrics_list_train, cur_step_after_loop)
            tf.summary.scalar(
                'learning_rate',
                learning_rate_schedule(tf.cast(optimizer.iterations, dtype=tf.float32)), # Pass step tensor
                step=optimizer.iterations) # Use optimizer.iterations directly (it's a tf.Variable)
            summary_writer.flush()
        
        # Reset training metrics for the next loop
        for metric_obj in all_metrics_list_train:
            metric_obj.reset_states()
            
    logging.info('Training complete.')

    # Final export of the model
    if optimizer.iterations.numpy() >= train_steps:
        save_model_for_export(model, global_step_val=optimizer.iterations.numpy())


    if FLAGS.mode == 'train_then_eval':
      final_ckpt = checkpoint_manager.latest_checkpoint
      if final_ckpt:
          perform_evaluation(model, builder, eval_steps, final_ckpt, strategy, topology)
      else:
          logging.warning("No checkpoint found after training to perform final evaluation.")


if __name__ == '__main__':
  # tf.compat.v1.enable_v2_behavior() # This is default in TF2.
  # For outside compilation of summaries on TPU.
  # Soft device placement is on by default. Explicitly setting can be useful.
  tf.config.set_soft_device_placement(True)
  app.run(main)