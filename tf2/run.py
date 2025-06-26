# run.py

import json
import math
import os
import numpy as np # For json_serializable

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

# Assuming these are local modules updated for TF2
import data as data_lib
import metrics as metrics_lib # Renamed to avoid conflict with Keras metrics module
import model as model_lib
import objective as obj_lib
# lars_optimizer is used within model_lib.build_optimizer

FLAGS = flags.FLAGS

# Define flags (ensure these match your intended configuration)
# General
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'train_then_eval'], 'Run mode.')
flags.DEFINE_string('model_dir', '/tmp/simclr_run', 'Directory to save model and summaries.')
flags.DEFINE_string('data_dir', None, 'Directory for TFDS data. Defaults to ~/tensorflow_datasets.')
flags.DEFINE_integer('seed', 42, 'Random seed for reproducibility.') # Added seed

# Training params
flags.DEFINE_enum('train_mode', 'pretrain', ['pretrain', 'finetune'], 'Training objective.')
flags.DEFINE_float('learning_rate', 0.3, 'Initial learning rate per batch size of 256.')
flags.DEFINE_enum('learning_rate_scaling', 'linear', ['linear', 'sqrt'], 'LR scaling by batch size.')
flags.DEFINE_float('warmup_epochs', 10, 'Number of epochs of warmup.')
flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')
flags.DEFINE_integer('train_batch_size', 512, 'Batch size for training.')
flags.DEFINE_string('train_split', 'train', 'Dataset split for training.')
flags.DEFINE_integer('train_epochs', 100, 'Number of epochs to train for.')
flags.DEFINE_integer('train_steps', 0, 'Number of steps to train for. Overrides train_epochs if > 0.')
flags.DEFINE_enum('optimizer', 'lars', ['momentum', 'adam', 'lars'], 'Optimizer.')
flags.DEFINE_float('momentum', 0.9, 'Momentum parameter for SGD and LARS.')

# Evaluation params
flags.DEFINE_string('eval_split', 'test', 'Dataset split for evaluation. Was "validation".') # Changed default
flags.DEFINE_integer('eval_batch_size', 256, 'Batch size for eval.')
flags.DEFINE_integer('eval_steps', 0, 'Number of steps to eval for. 0 means eval over entire dataset.')
flags.DEFINE_string('eval_name', None, 'Optional name for distinguishing eval runs.')

# Checkpoint and Summary params
flags.DEFINE_integer('checkpoint_epochs', 1, 'Number of epochs between checkpoints.')
flags.DEFINE_integer('checkpoint_steps', 0, 'Number of steps between checkpoints. Overrides checkpoint_epochs if > 0.')
flags.DEFINE_integer('keep_checkpoint_max', 5, 'Maximum number of checkpoints to keep.')
# flags.DEFINE_integer('keep_hub_module_max', 1, 'Maximum number of Hub modules to keep.') # Hub export simplified

# Hardware params
flags.DEFINE_bool('use_tpu', False, 'Whether to run on TPU. Default to False for GPU/CPU.')
flags.DEFINE_string('tpu_name', None, 'Cloud TPU name or grpc address.')
flags.DEFINE_string('tpu_zone', None, 'GCE zone for TPU.')
flags.DEFINE_string('gcp_project', None, 'GCP project for TPU.')
flags.DEFINE_boolean('global_bn', False, 'Aggregate BN stats. Default False for single GPU.') # Default False

# Model params
flags.DEFINE_integer('resnet_depth', 18, 'Depth of ResNet (e.g., 18, 50).') # Default 18 for faster test
flags.DEFINE_integer('width_multiplier', 1, 'Width multiplier for ResNet.')
flags.DEFINE_integer('image_size', 32, 'Input image size (e.g., 32 for CIFAR, 224 for ImageNet).') # Default 32
flags.DEFINE_string('dataset', 'cifar10', 'Name of TFDS dataset.') # Default cifar10

# SimCLR specific params
flags.DEFINE_float('temperature', 0.1, 'Temperature for contrastive loss.')
flags.DEFINE_boolean('hidden_norm', True, 'L2 normalize hidden representations for contrastive loss.')
flags.DEFINE_enum('proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'], 'Projection head type.')
flags.DEFINE_integer('proj_out_dim', 128, 'Output dimension of projection head.')
flags.DEFINE_integer('num_proj_layers', 3, 'Number of layers in non-linear projection head.')
flags.DEFINE_integer('ft_proj_selector', 0, 'Which layer of proj head for finetuning supervised head. 0 means backbone output.')

# Augmentation params
flags.DEFINE_float('color_jitter_strength', 0.5, 'Strength of color jittering.') # Default 0.5
flags.DEFINE_boolean('use_blur', False, 'Use Gaussian blur for augmentation. Default False for CIFAR.') # Default False for CIFAR

# Fine-tuning specific params
flags.DEFINE_bool('lineareval_while_pretraining', True, 'Linear eval during pretraining.')
flags.DEFINE_string('checkpoint', None, 'Path to pre-trained checkpoint for fine-tuning.')
flags.DEFINE_bool('zero_init_logits_layer', False, 'Zero initialize supervised head for fine-tuning.')
flags.DEFINE_integer('fine_tune_after_block', -1, 'ResNet block after which to fine-tune (-1 for all).')

flags.DEFINE_bool('cache_dataset', False, 'Cache dataset in memory.')


def initialize_strategies():
    """Initialize and return the distribution strategy and topology."""
    topology = None
    if FLAGS.use_tpu:
        logging.info("Attempting to connect to TPU: %s", FLAGS.tpu_name or "local")
        if FLAGS.tpu_name:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        else: # Auto-discover (works for local Colab/Kaggle TPUs too)
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        topology = tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info('TPUStrategy initialized. Topology: %s', topology)
    else:
        # For single GPU or CPU, or multi-GPU with MirroredStrategy
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info(f"Found {len(gpus)} GPU(s).")
            # If you want to use only one GPU explicitly, even if multiple are available:
            # tf.config.set_visible_devices(gpus[0], 'GPU')
            # MirroredStrategy will use all available GPUs by default.
            # For single GPU, default strategy or MirroredStrategy works.
            try:
                strategy = tf.distribute.MirroredStrategy()
                logging.info('MirroredStrategy initialized for %d GPU(s).', strategy.num_replicas_in_sync)
            except Exception as e:
                logging.warning(f"Could not initialize MirroredStrategy: {e}. Falling back to default strategy.")
                strategy = tf.distribute.get_strategy() # Default strategy for CPU or single GPU
        else:
            logging.info("No GPUs found. Using default strategy (CPU).")
            strategy = tf.distribute.get_strategy()
            # Ensure global_bn is False if no distribution or single device
            if FLAGS.global_bn and strategy.num_replicas_in_sync <= 1:
                logging.warning("FLAGS.global_bn is True but running on single device. Setting to False.")
                FLAGS.global_bn = False


    logging.info('Running with %d replicas in sync.', strategy.num_replicas_in_sync)
    return strategy, topology

def try_restore_checkpoint(model, optimizer, model_dir):
    """Restores the latest checkpoint if it exists, or from FLAGS.checkpoint for fine-tuning."""
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=optimizer.iterations, # optimizer.iterations is a tf.Variable
        epoch=tf.Variable(0, dtype=tf.int64)
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=model_dir, max_to_keep=FLAGS.keep_checkpoint_max)

    latest_ckpt = checkpoint_manager.latest_checkpoint
    if latest_ckpt:
        logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
        status = checkpoint.restore(latest_ckpt)
        status.assert_existing_objects_matched().expect_partial()
        logging.info(f"Successfully restored model from epoch {checkpoint.epoch.numpy()} at step {optimizer.iterations.numpy()}.")
    elif FLAGS.train_mode == 'finetune' and FLAGS.checkpoint:
        logging.info('Fine-tuning: Restoring model weights from pre-trained checkpoint: %s', FLAGS.checkpoint)
        # Create a new checkpoint object with only the model for loading pre-trained weights
        # This avoids issues with optimizer state or step mismatches from pre-training.
        model_ckpt_for_pretrain = tf.train.Checkpoint(model=model)
        status = model_ckpt_for_pretrain.restore(FLAGS.checkpoint)
        status.expect_partial() # Allow for supervised head not being in pre-trained model
        logging.info("Pre-trained weights restored. Optimizer and step will start from scratch.")
        optimizer.iterations.assign(0) # Reset step for fine-tuning
        checkpoint.epoch.assign(0)      # Reset epoch for fine-tuning

        if FLAGS.zero_init_logits_layer:
            if hasattr(model, 'supervised_head') and model.supervised_head is not None:
                logging.info('Zero initializing supervised head for fine-tuning.')
                for var in model.supervised_head.trainable_variables:
                    var.assign(tf.zeros_like(var))
            else:
                logging.warning("zero_init_logits_layer=True but no supervised_head found on model.")
    else:
        logging.info('No checkpoint found. Starting training from scratch.')
    return checkpoint_manager, checkpoint # Return checkpoint too for epoch updates


def save_current_flags(model_dir):
    """Saves current command-line flags to a JSON file."""
    flag_json_path = os.path.join(model_dir, 'flags_config.json')
    if not tf.io.gfile.exists(flag_json_path): # Save only if it doesn't exist
        serializable_flags = {}
        for key, val_obj in FLAGS.flag_values_dict().items():
            # For absl flags, val_obj is the value itself
            if json_serializable(val_obj):
                serializable_flags[key] = val_obj
            else:
                serializable_flags[key] = str(val_obj) # Store as string if not serializable
        with tf.io.gfile.GFile(flag_json_path, 'w') as f:
            json.dump(serializable_flags, f, indent=4)
        logging.info(f"Saved command-line flags to {flag_json_path}")

def json_serializable(val):
  if isinstance(val, tf.Tensor): val = val.numpy()
  if hasattr(val, 'dtype') and isinstance(val, (np.generic,)): val = val.item()
  if isinstance(val, (tf.dtypes.DType, type(None), tf.TensorShape)): return False
  try:
    json.dumps(val)
    return True
  except (TypeError, OverflowError):
    return False


def perform_evaluation(model, tfds_builder, num_eval_examples, eval_ckpt_path, strategy, current_train_step):
    """Performs evaluation on the evaluation split."""
    logging.info(f"Starting evaluation for checkpoint: {eval_ckpt_path} (Train step: {current_train_step})")
    
    eval_batch_size = FLAGS.eval_batch_size * strategy.num_replicas_in_sync
    eval_steps = FLAGS.eval_steps or math.ceil(num_eval_examples / eval_batch_size)

    ds_eval = data_lib.build_distributed_dataset(
        tfds_builder, FLAGS.eval_batch_size, # Pass per-replica batch size
        is_training=False, strategy=strategy, topology=None # Topology not critical for eval on GPU
    )

    eval_summary_dir = os.path.join(FLAGS.model_dir, f"eval_{FLAGS.eval_split}_{current_train_step}")
    eval_summary_writer = tf.summary.create_file_writer(eval_summary_dir)

    with strategy.scope():
        # Create metrics for evaluation
        eval_supervised_loss_metric = tf.keras.metrics.Mean('eval/supervised_loss')
        eval_label_top_1_accuracy = tf.keras.metrics.CategoricalAccuracy('eval/label_top_1_accuracy')
        eval_label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='eval/label_top_5_accuracy')
        eval_metrics_list = [eval_supervised_loss_metric, eval_label_top_1_accuracy, eval_label_top_5_accuracy]
        
        if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # Also evaluate contrastive metrics if doing linear eval during pretrain
            eval_contrast_loss_metric = tf.keras.metrics.Mean('eval/contrast_loss')
            eval_contrast_acc_metric = tf.keras.metrics.Mean('eval/contrast_acc')
            eval_contrast_entropy_metric = tf.keras.metrics.Mean('eval/contrast_entropy')
            eval_metrics_list.extend([eval_contrast_loss_metric, eval_contrast_acc_metric, eval_contrast_entropy_metric])


        # Restore model from the checkpoint being evaluated
        eval_checkpoint = tf.train.Checkpoint(model=model)
        status = eval_checkpoint.restore(eval_ckpt_path)
        status.assert_existing_objects_matched().expect_partial() # Ensure model weights load
        logging.info(f"Restored model from {eval_ckpt_path} for evaluation.")

    @tf.function
    def distributed_eval_step(dist_inputs):
        per_replica_images, per_replica_labels_dict = dist_inputs
        def replica_eval_step(images, labels_dict):
            projection_outputs, supervised_outputs = model(images, training=False)
            eval_loss = 0.0

            if supervised_outputs is not None:
                sup_loss = obj_lib.add_supervised_loss(labels=labels_dict['labels'], logits=supervised_outputs)
                eval_supervised_loss_metric.update_state(sup_loss)
                eval_label_top_1_accuracy.update_state(labels_dict['labels'], supervised_outputs)
                eval_label_top_5_accuracy.update_state(labels_dict['labels'], supervised_outputs)
                eval_loss += sup_loss
            
            if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining and projection_outputs is not None:
                # For linear eval, contrastive loss is calculated on eval set as well
                # Note: strategy=None here as cross-replica concat might not be intended for eval pass
                # or it needs careful handling if eval batch is smaller.
                # For simplicity, compute contrastive loss on per-replica data.
                con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
                    projection_outputs,
                    hidden_norm=FLAGS.hidden_norm,
                    temperature=FLAGS.temperature,
                    strategy=None) # Pass None or handle eval distribution carefully for contrastive loss
                eval_contrast_loss_metric.update_state(con_loss)
                metrics_lib.update_pretrain_metrics_train( # Using train version for update logic consistency
                    eval_contrast_loss_metric, eval_contrast_acc_metric, eval_contrast_entropy_metric,
                    con_loss, logits_con, labels_con
                )
                eval_loss += con_loss
            
            # Weight decay is not typically part of eval loss, but can be logged.
            # wd_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
            # eval_total_loss_metric.update_state(eval_loss + wd_loss)

        strategy.run(replica_eval_step, args=(per_replica_images, per_replica_labels_dict))

    eval_iterator = iter(ds_eval)
    logging.info(f"Starting evaluation loop for {eval_steps} steps.")
    for step in tf.range(eval_steps):
        try:
            distributed_eval_step(next(eval_iterator))
        except tf.errors.OutOfRangeError:
            logging.warning("Evaluation dataset exhausted before completing all eval_steps.")
            break
        if (step + 1) % 10 == 0: # Log progress
            logging.info(f"Eval step {step.numpy()+1}/{eval_steps} completed.")
    
    logging.info(f"Evaluation finished for checkpoint step {current_train_step}.")
    with eval_summary_writer.as_default():
        metrics_lib.log_and_write_metrics_to_summary(eval_metrics_list, current_train_step)
        eval_summary_writer.flush()

    for metric in eval_metrics_list: # Reset for potential next eval call
        metric.result() # Ensure result is computed if not already
        logging.info(f"{metric.name}: {metric.result().numpy():.4f}")
        metric.reset_states()


def main(argv):
    del argv # Unused.

    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    if not FLAGS.model_dir:
        FLAGS.model_dir = f"/tmp/simclr_{FLAGS.dataset}_{FLAGS.resnet_depth}_{FLAGS.train_mode}"
    tf.io.gfile.makedirs(FLAGS.model_dir)
    save_current_flags(FLAGS.model_dir) # Save flags early

    strategy, topology = initialize_strategies()

    tfds_builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
    tfds_builder.download_and_prepare() # This will download if not present

    num_train_examples = tfds_builder.info.splits[FLAGS.train_split].num_examples
    num_eval_examples = tfds_builder.info.splits[FLAGS.eval_split].num_examples
    num_classes = tfds_builder.info.features['label'].num_classes

    global_train_batch_size = FLAGS.train_batch_size * strategy.num_replicas_in_sync
    
    train_steps = FLAGS.train_steps
    if train_steps <= 0:
        train_steps = num_train_examples * FLAGS.train_epochs // global_train_batch_size
    if train_steps <=0 : # Handle case where calculated steps are 0 (e.g. too few epochs/examples)
        train_steps = 1 # Ensure at least one step if training
        logging.warning(f"Calculated train_steps was <=0, setting to 1.")


    steps_per_epoch = num_train_examples // global_train_batch_size
    if steps_per_epoch == 0 and num_train_examples > 0 : steps_per_epoch = 1


    logging.info('# Train Examples: %d', num_train_examples)
    logging.info('# Eval Examples: %d', num_eval_examples)
    logging.info('# Num Classes: %d', num_classes)
    logging.info('Global Train Batch Size: %d', global_train_batch_size)
    logging.info('Steps per Epoch: %d', steps_per_epoch)
    logging.info('Total Train Steps: %d', train_steps)
    
    checkpoint_steps = FLAGS.checkpoint_steps
    if checkpoint_steps <= 0 and steps_per_epoch > 0:
        checkpoint_steps = FLAGS.checkpoint_epochs * steps_per_epoch
    if checkpoint_steps <= 0 : checkpoint_steps = 1000 # Fallback
    logging.info('Checkpoint Steps: %d', checkpoint_steps)


    with strategy.scope():
        model = model_lib.Model(num_classes=num_classes)
        # Build model with a dummy input to create variables before checkpoint restoration
        # This helps avoid issues with restoring optimizer state if model vars aren't created.
        dummy_input_shape = (FLAGS.train_batch_size, FLAGS.image_size, FLAGS.image_size, 3 * (2 if FLAGS.train_mode=='pretrain' else 1) )
        model.build(input_shape=dummy_input_shape) # Or model(tf.zeros(dummy_input_shape))

        learning_rate_schedule = model_lib.WarmUpAndCosineDecay(
            base_learning_rate=FLAGS.learning_rate,
            num_examples=num_train_examples, # Should be total examples for LR schedule calculation
            # Provide batch size and train_steps for accurate schedule calculation if needed by WarmUpAndCosineDecay
            # The current model_lib.WarmUpAndCosineDecay takes num_examples.
        )
        optimizer = model_lib.build_optimizer(learning_rate_schedule)
        
        # Metrics (created within strategy scope)
        train_metrics_map = {} # Use a map for easier access
        train_metrics_map['weight_decay'] = tf.keras.metrics.Mean('train/weight_decay')
        train_metrics_map['total_loss'] = tf.keras.metrics.Mean('train/total_loss')
        if FLAGS.train_mode == 'pretrain':
            train_metrics_map['contrast_loss'] = tf.keras.metrics.Mean('train/contrast_loss')
            train_metrics_map['contrast_acc'] = tf.keras.metrics.Mean('train/contrast_acc')
            train_metrics_map['contrast_entropy'] = tf.keras.metrics.Mean('train/contrast_entropy')
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            train_metrics_map['supervised_loss'] = tf.keras.metrics.Mean('train/supervised_loss')
            train_metrics_map['supervised_acc'] = tf.keras.metrics.Mean('train/supervised_acc')
        
        all_train_metrics_list = list(train_metrics_map.values())

        # Checkpoint manager and restoration
        checkpoint_manager, main_checkpoint = try_restore_checkpoint(model, optimizer, FLAGS.model_dir)


    if FLAGS.mode == 'eval':
        if not checkpoint_manager.latest_checkpoint:
            logging.error("No checkpoint found in model_dir for evaluation. Exiting.")
            return
        perform_evaluation(
            model, tfds_builder, num_eval_examples,
            checkpoint_manager.latest_checkpoint, strategy,
            optimizer.iterations.numpy() # Pass current train step for eval naming
        )
        return # Exit after eval mode

    # Create training dataset iterator
    ds_train = data_lib.build_distributed_dataset(
        tfds_builder, FLAGS.train_batch_size, # Pass per-replica batch size
        is_training=True, strategy=strategy, topology=topology
    )
    train_iterator = iter(ds_train)
    train_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, "train"))

    @tf.function
    def distributed_train_step_fn(dist_inputs): # Renamed to avoid collision
        """The core training step logic, distributed via strategy.run."""
        per_replica_images, per_replica_labels_dict = dist_inputs

        def replica_train_step(images, labels_dict):
            """Operations executed on each replica."""
            with tf.GradientTape() as tape:
                projection_outputs, supervised_outputs = model(images, training=True)
                
                current_loss = 0.0
                # Calculate losses based on mode
                if FLAGS.train_mode == 'pretrain':
                    if projection_outputs is None:
                        raise ValueError("Projection head output is None during pretrain mode.")
                    con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
                        projection_outputs,
                        hidden_norm=FLAGS.hidden_norm,
                        temperature=FLAGS.temperature,
                        strategy=strategy # Pass strategy for cross-replica loss
                    )
                    current_loss += con_loss
                    metrics_lib.update_pretrain_metrics_train(
                        train_metrics_map['contrast_loss'], train_metrics_map['contrast_acc'],
                        train_metrics_map['contrast_entropy'], con_loss, logits_con, labels_con
                    )
                    if FLAGS.lineareval_while_pretraining and supervised_outputs is not None:
                        # Labels need to be duplicated if supervised head processes augmented views
                        true_labels_sup = tf.concat([labels_dict['labels'], labels_dict['labels']], axis=0)
                        sup_loss_linear = obj_lib.add_supervised_loss(labels=true_labels_sup, logits=supervised_outputs)
                        current_loss += sup_loss_linear # Add to total loss
                        metrics_lib.update_finetune_metrics_train(
                            train_metrics_map['supervised_loss'], train_metrics_map['supervised_acc'],
                            sup_loss_linear, true_labels_sup, supervised_outputs
                        )
                elif FLAGS.train_mode == 'finetune':
                    if supervised_outputs is None:
                        raise ValueError("Supervised head output is None during finetune mode.")
                    sup_loss_finetune = obj_lib.add_supervised_loss(labels=labels_dict['labels'], logits=supervised_outputs)
                    current_loss += sup_loss_finetune
                    metrics_lib.update_finetune_metrics_train(
                        train_metrics_map['supervised_loss'], train_metrics_map['supervised_acc'],
                        sup_loss_finetune, labels_dict['labels'], supervised_outputs
                    )
                
                if current_loss == 0.0 and not (FLAGS.train_mode == 'pretrain' and not FLAGS.lineareval_while_pretraining) :
                     # If only pretrain without linear eval, and proj_outputs was None (model error), this might be 0.
                     # Or if finetune and sup_outputs was None.
                     logging.warning("Main objective loss is zero, check model configuration and outputs.")


                wd_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
                train_metrics_map['weight_decay'].update_state(wd_loss)
                total_loss_for_grads = current_loss + wd_loss
                train_metrics_map['total_loss'].update_state(total_loss_for_grads)
                
                # Scale loss for multi-replica training before applying gradients
                scaled_loss = total_loss_for_grads / strategy.num_replicas_in_sync

            # Apply gradients
            trainable_vars = model.trainable_variables
            grads = tape.gradient(scaled_loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
        
        strategy.run(replica_train_step, args=(per_replica_images, per_replica_labels_dict))


    logging.info("Starting training loop from step %d...", optimizer.iterations.numpy())
    initial_step = optimizer.iterations.numpy() # For calculating steps in this run

    for current_loop_step in range(initial_step, train_steps, checkpoint_steps):
        # Run 'checkpoint_steps' or remaining steps if less than checkpoint_steps
        steps_to_run_in_loop = min(checkpoint_steps, train_steps - optimizer.iterations.numpy())
        if steps_to_run_in_loop <= 0: break # Exit if no more steps needed

        logging.info(f"Running training for {steps_to_run_in_loop} steps (Total progress: {optimizer.iterations.numpy()}/{train_steps})")

        for _ in tf.range(steps_to_run_in_loop): # Use tf.range for potential tf.function benefits
            try:
                batch_images, batch_labels = next(train_iterator)
                distributed_train_step_fn((batch_images, {'labels': batch_labels}))
            except tf.errors.OutOfRangeError: # Should not happen with repeat(-1)
                logging.warning("Training iterator exhausted unexpectedly. Reinitializing.")
                train_iterator = iter(data_lib.build_distributed_dataset(
                    tfds_builder, FLAGS.train_batch_size, True, strategy, topology))
                batch_images, batch_labels = next(train_iterator)
                distributed_train_step_fn((batch_images, {'labels': batch_labels}))

        current_global_step = optimizer.iterations.numpy()
        main_checkpoint.epoch.assign_add(FLAGS.checkpoint_epochs if FLAGS.checkpoint_steps == 0 else checkpoint_steps // steps_per_epoch if steps_per_epoch > 0 else 0)
        
        checkpoint_manager.save(checkpoint_number=current_global_step)
        logging.info(f"Checkpoint saved for step {current_global_step} (Epoch ~{main_checkpoint.epoch.numpy()}).")

        with train_summary_writer.as_default():
            metrics_lib.log_and_write_metrics_to_summary(all_train_metrics_list, current_global_step)
            current_lr = learning_rate_schedule(tf.cast(optimizer.iterations, dtype=tf.float32))
            tf.summary.scalar('learning_rate', current_lr, step=optimizer.iterations)
            tf.summary.scalar('epoch', main_checkpoint.epoch.numpy(), step=optimizer.iterations)
            train_summary_writer.flush()
        
        for metric in all_train_metrics_list:
            metric.reset_states()
        
        # Perform evaluation if in train_then_eval mode and checkpoint interval aligns
        if FLAGS.mode == 'train_then_eval':
            perform_evaluation(
                model, tfds_builder, num_eval_examples,
                checkpoint_manager.latest_checkpoint, strategy,
                current_global_step
            )

    logging.info('Training complete at step %d.', optimizer.iterations.numpy())

    final_checkpoint_path = checkpoint_manager.latest_checkpoint
    if final_checkpoint_path:
        logging.info(f"Final checkpoint saved at {final_checkpoint_path}")
        # save_model_for_export(model, global_step_val=optimizer.iterations.numpy()) # Simplified export
    else:
        logging.warning("No final checkpoint was saved.")


    if FLAGS.mode == 'train_then_eval' and final_checkpoint_path:
        logging.info("Performing final evaluation after training.")
        perform_evaluation(
            model, tfds_builder, num_eval_examples,
            final_checkpoint_path, strategy,
            optimizer.iterations.numpy()
        )


if __name__ == '__main__':
    tf.config.set_soft_device_placement(True) # Good default
    app.run(main)