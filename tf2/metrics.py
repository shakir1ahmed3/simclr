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
"""Training utilities."""

from absl import logging
import tensorflow as tf # Changed from tensorflow.compat.v2 as tf


def update_pretrain_metrics_train(contrast_loss_metric: tf.keras.metrics.Mean,
                                  contrast_acc_metric: tf.keras.metrics.Mean, # Or a dedicated Accuracy metric
                                  contrast_entropy_metric: tf.keras.metrics.Mean,
                                  loss: tf.Tensor,
                                  logits_con: tf.Tensor,
                                  labels_con: tf.Tensor):
  """Updates pretraining metrics for the training step.

  Args:
    contrast_loss_metric: Keras metric for contrastive loss.
    contrast_acc_metric: Keras metric for contrastive accuracy.
    contrast_entropy_metric: Keras metric for contrastive entropy.
    loss: The contrastive loss for the current batch.
    logits_con: The logits from the contrastive head.
    labels_con: The labels for the contrastive task (often an identity matrix).
  """
  contrast_loss_metric.update_state(loss)

  # Contrastive accuracy in SimCLR is often defined as matching the augmented view
  # of an image to itself against other negatives.
  # labels_con is typically an identity matrix, so tf.argmax(labels_con, 1) -> [0, 1, ..., N-1]
  # This calculates if the highest logit corresponds to the "correct" positive pair.
  contrast_acc_val = tf.equal(
      tf.argmax(labels_con, axis=1), tf.argmax(logits_con, axis=1))
  contrast_acc_val = tf.reduce_mean(tf.cast(contrast_acc_val, tf.float32))
  contrast_acc_metric.update_state(contrast_acc_val)

  # Entropy of the softmax probabilities of contrastive predictions
  prob_con = tf.nn.softmax(logits_con)
  # Add small epsilon for numerical stability of log
  entropy_con = -tf.reduce_mean(
      tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), axis=-1))
  contrast_entropy_metric.update_state(entropy_con)


def update_pretrain_metrics_eval(contrast_loss_metric: tf.keras.metrics.Mean,
                                 contrastive_top_1_accuracy_metric: tf.keras.metrics.Metric, # e.g., SparseCategoricalAccuracy or TopKCategoricalAccuracy
                                 contrastive_top_5_accuracy_metric: tf.keras.metrics.TopKCategoricalAccuracy,
                                 contrast_loss: tf.Tensor,
                                 logits_con: tf.Tensor,
                                 labels_con: tf.Tensor): # Typically one-hot or identity matrix
  """Updates pretraining metrics for the evaluation step.

  Args:
    contrast_loss_metric: Keras metric for contrastive loss.
    contrastive_top_1_accuracy_metric: Keras metric for top-1 contrastive accuracy.
    contrastive_top_5_accuracy_metric: Keras metric for top-5 contrastive accuracy.
    contrast_loss: The contrastive loss for the current batch.
    logits_con: The logits from the contrastive head.
    labels_con: The labels for the contrastive task.
  """
  contrast_loss_metric.update_state(contrast_loss)

  # For SimCLR, labels_con is often an identity matrix.
  # tf.argmax(labels_con, axis=1) would yield [0, 1, 2, ..., batch_size-1]
  # This can be directly used with SparseCategoricalAccuracy.
  # If contrastive_top_1_accuracy_metric is SparseCategoricalAccuracy:
  #   true_classes = tf.range(tf.shape(labels_con)[0], dtype=tf.int64)
  #   contrastive_top_1_accuracy_metric.update_state(true_classes, logits_con)
  # If contrastive_top_1_accuracy_metric is CategoricalAccuracy or TopKCategoricalAccuracy (k=1):
  #   contrastive_top_1_accuracy_metric.update_state(labels_con, logits_con)
  # The original code implies labels_con is one-hot and the metric is a simple Accuracy comparing argmaxes.
  # Let's stick to the original logic if the metric is tf.keras.metrics.Accuracy:
  if isinstance(contrastive_top_1_accuracy_metric, tf.keras.metrics.Accuracy):
      contrastive_top_1_accuracy_metric.update_state(
          tf.argmax(labels_con, axis=1), tf.argmax(logits_con, axis=1))
  elif isinstance(contrastive_top_1_accuracy_metric, (tf.keras.metrics.SparseCategoricalAccuracy, tf.keras.metrics.TopKCategoricalAccuracy)):
      # Assuming labels_con makes tf.argmax(labels_con, axis=1) the sparse representation
      # Or, if labels_con is directly usable by TopKCategoricalAccuracy (i.e. one-hot)
      contrastive_top_1_accuracy_metric.update_state(labels_con, logits_con)
  else: # Fallback or if it's a custom metric expecting this
      # This was the original line, assuming labels_con are one-hot and metric is tf.keras.metrics.Accuracy
      # or a custom one that takes argmax of y_true and y_pred.
      true_indices = tf.argmax(labels_con, axis=1)
      pred_indices = tf.argmax(logits_con, axis=1)
      contrastive_top_1_accuracy_metric.update_state(true_indices, pred_indices)


  # This is standard for TopKCategoricalAccuracy when labels_con are one-hot.
  contrastive_top_5_accuracy_metric.update_state(labels_con, logits_con)


def update_finetune_metrics_train(supervised_loss_metric: tf.keras.metrics.Mean,
                                  supervised_acc_metric: tf.keras.metrics.Mean, # Or CategoricalAccuracy
                                  loss: tf.Tensor,
                                  labels: tf.Tensor, # Typically one-hot
                                  logits: tf.Tensor):
  """Updates fine-tuning metrics for the training step.

  Args:
    supervised_loss_metric: Keras metric for supervised loss.
    supervised_acc_metric: Keras metric for supervised accuracy.
    loss: The supervised loss for the current batch.
    labels: The true labels (one-hot encoded).
    logits: The output logits from the model.
  """
  supervised_loss_metric.update_state(loss)

  # Manual accuracy calculation assuming labels are one-hot.
  # If supervised_acc_metric is tf.keras.metrics.CategoricalAccuracy:
  #   supervised_acc_metric.update_state(labels, logits)
  # else (if it's a Mean metric storing pre-calculated accuracy):
  label_acc_val = tf.equal(tf.argmax(labels, axis=1), tf.argmax(logits, axis=1))
  label_acc_val = tf.reduce_mean(tf.cast(label_acc_val, tf.float32))
  supervised_acc_metric.update_state(label_acc_val)


def update_finetune_metrics_eval(label_top_1_accuracy_metric: tf.keras.metrics.Metric, # e.g., CategoricalAccuracy
                                 label_top_5_accuracy_metric: tf.keras.metrics.TopKCategoricalAccuracy,
                                 outputs: tf.Tensor, # Model output logits
                                 labels: tf.Tensor): # True labels, typically one-hot
  """Updates fine-tuning metrics for the evaluation step.

  Args:
    label_top_1_accuracy_metric: Keras metric for top-1 label accuracy.
    label_top_5_accuracy_metric: Keras metric for top-5 label accuracy.
    outputs: The output logits from the model.
    labels: The true labels (one-hot encoded).
  """
  # If label_top_1_accuracy_metric is CategoricalAccuracy or SparseCategoricalAccuracy (if labels were sparse)
  #   label_top_1_accuracy_metric.update_state(labels, outputs)
  # The original implies labels are one-hot and the metric might be tf.keras.metrics.Accuracy
  # or a custom one taking argmaxes.
  if isinstance(label_top_1_accuracy_metric, (tf.keras.metrics.CategoricalAccuracy, tf.keras.metrics.SparseCategoricalAccuracy)):
      label_top_1_accuracy_metric.update_state(labels, outputs) # Assumes labels are in the correct format for the metric
  elif isinstance(label_top_1_accuracy_metric, tf.keras.metrics.Accuracy):
      label_top_1_accuracy_metric.update_state(
          tf.argmax(labels, axis=1), tf.argmax(outputs, axis=1))
  else: # Fallback or if it's a custom metric expecting this
      true_indices = tf.argmax(labels, axis=1)
      pred_indices = tf.argmax(outputs, axis=1)
      label_top_1_accuracy_metric.update_state(true_indices, pred_indices)

  # This is standard for TopKCategoricalAccuracy when labels are one-hot.
  label_top_5_accuracy_metric.update_state(labels, outputs)


def _float_metric_value(metric: tf.keras.metrics.Metric) -> float:
  """Gets the scalar value of a Keras metric.

  Args:
    metric: A tf.keras.metrics.Metric instance.

  Returns:
    The result of the metric as a Python float.
  """
  # .numpy() requires Eager execution or to be called outside tf.function
  # This is typical for metric logging loops.
  return float(metric.result().numpy())


def log_and_write_metrics_to_summary(all_metrics: list[tf.keras.metrics.Metric],
                                     global_step: tf.Tensor | int):
  """Logs metric values and writes them to tf.summary.

  Args:
    all_metrics: A list of Keras metrics.
    global_step: The current training step.
  """
  for metric in all_metrics:
    metric_value = _float_metric_value(metric)
    # Log to console/stdout via absl.logging
    logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
    # Write to TensorBoard
    tf.summary.scalar(metric.name, data=metric_value, step=tf.cast(global_step, tf.int64))