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
"""Data preprocessing and augmentation."""

import functools
import tensorflow as tf # Changed

CROP_PROPORTION = 0.875  # Standard for ImageNet.


def random_apply(func, p: float, x: tf.Tensor) -> tf.Tensor:
  """Randomly apply function func to x with probability p."""
  return tf.cond(
      tf.less(
          tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32), # Ensure minval is float
          tf.cast(p, tf.float32)),
      lambda: func(x),
      lambda: x)


def random_brightness(image: tf.Tensor, max_delta: float, impl: str = 'simclrv2') -> tf.Tensor:
  """A multiplicative vs additive change of brightness."""
  if impl == 'simclrv2':
    factor = tf.random.uniform([], tf.maximum(1.0 - max_delta, 0.0), # Ensure 0.0
                               1.0 + max_delta)
    image = image * factor
  elif impl == 'simclrv1':
    image = tf.image.random_brightness(image, max_delta=max_delta)
  else:
    raise ValueError(f'Unknown impl {impl} for random brightness.')
  return image


def to_grayscale(image: tf.Tensor, keep_channels: bool = True) -> tf.Tensor:
  image = tf.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf.tile(image, [1, 1, 3]) # Assumes image is HWC
  return image


def color_jitter(image: tf.Tensor,
                 strength: float,
                 random_order: bool = True,
                 impl: str = 'simclrv2') -> tf.Tensor:
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.

  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand(
        image, brightness, contrast, saturation, hue, impl=impl)
  else:
    return color_jitter_nonrand(
        image, brightness, contrast, saturation, hue, impl=impl)


def color_jitter_nonrand(image: tf.Tensor,
                         brightness: float = 0.0,
                         contrast: float = 0.0,
                         saturation: float = 0.0,
                         hue: float = 0.0,
                         impl: str = 'simclrv2') -> tf.Tensor:
  """Distorts the color of the image (jittering order is fixed)."""
  with tf.name_scope('distort_color_nonrand'): # Added name_scope
    def apply_transform(i, x, brightness_val, contrast_val, saturation_val, hue_val): # Renamed args
      """Apply the i-th transformation."""
      if brightness_val != 0 and i == 0:
        x = random_brightness(x, max_delta=brightness_val, impl=impl)
      elif contrast_val != 0 and i == 1:
        x = tf.image.random_contrast(
            x, lower=tf.maximum(0.0, 1.0 - contrast_val), upper=1.0 + contrast_val) # Ensure lower >= 0
      elif saturation_val != 0 and i == 2:
        x = tf.image.random_saturation(
            x, lower=tf.maximum(0.0, 1.0 - saturation_val), upper=1.0 + saturation_val) # Ensure lower >= 0
      elif hue_val != 0 and i == 3: # Check i == 3 for hue specifically
        x = tf.image.random_hue(x, max_delta=hue_val)
      return x

    for i_transform in range(4): # Use different loop var
      image = apply_transform(i_transform, image, brightness, contrast, saturation, hue)
      image = tf.clip_by_value(image, 0.0, 1.0) # Ensure 0.0, 1.0
    return image


def color_jitter_rand(image: tf.Tensor,
                      brightness: float = 0.0,
                      contrast: float = 0.0,
                      saturation: float = 0.0,
                      hue: float = 0.0,
                      impl: str = 'simclrv2') -> tf.Tensor:
  """Distorts the color of the image (jittering order is random)."""
  with tf.name_scope('distort_color_rand'): # Added name_scope
    def apply_transform(idx_transform, current_image): # Renamed args
      """Apply the idx_transform-th transformation."""
      if brightness != 0.0 and idx_transform == 0: # Explicitly check for 0.0
          current_image = random_brightness(current_image, max_delta=brightness, impl=impl)
      elif contrast != 0.0 and idx_transform == 1:
          current_image = tf.image.random_contrast(current_image, lower=tf.maximum(0.0, 1.0 - contrast), upper=1.0 + contrast)
      elif saturation != 0.0 and idx_transform == 2:
          current_image = tf.image.random_saturation(current_image, lower=tf.maximum(0.0, 1.0 - saturation), upper=1.0 + saturation)
      elif hue != 0.0 and idx_transform == 3:
          current_image = tf.image.random_hue(current_image, max_delta=hue)
      return current_image

    # Original code used tf.cond chains which can be hard to read.
    # A more direct mapping from permuted index to function:
    transforms = []
    if brightness != 0.0:
        transforms.append(lambda img: random_brightness(img, max_delta=brightness, impl=impl))
    if contrast != 0.0:
        transforms.append(lambda img: tf.image.random_contrast(img, lower=tf.maximum(0.0, 1.0 - contrast), upper=1.0 + contrast))
    if saturation != 0.0:
        transforms.append(lambda img: tf.image.random_saturation(img, lower=tf.maximum(0.0, 1.0 - saturation), upper=1.0 + saturation))
    if hue != 0.0:
        transforms.append(lambda img: tf.image.random_hue(img, max_delta=hue))
    
    # tf.random.shuffle works on the first dimension.
    # Create a list of transform functions and shuffle their application order.
    # This requires dynamic execution or careful tf.cond/tf.case usage if number of active transforms varies.
    # The original permutes indices [0,1,2,3] and then maps these to operations.
    # Let's stick to the original structure using permuted indices for tf.function compatibility.
    
    perm = tf.random.shuffle(tf.range(4))
    for i in range(4): # Loop 4 times
      # Get the operation type based on the permuted index
      op_idx = perm[i]
      # Apply the operation corresponding to op_idx
      if brightness != 0.0 and op_idx == 0:
          image = random_brightness(image, max_delta=brightness, impl=impl)
      elif contrast != 0.0 and op_idx == 1:
          image = tf.image.random_contrast(image, lower=tf.maximum(0.0, 1.0 - contrast), upper=1.0 + contrast)
      elif saturation != 0.0 and op_idx == 2:
          image = tf.image.random_saturation(image, lower=tf.maximum(0.0, 1.0 - saturation), upper=1.0 + saturation)
      elif hue != 0.0 and op_idx == 3:
          image = tf.image.random_hue(image, max_delta=hue)
      
      image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _compute_crop_shape(
    image_height: tf.Tensor, image_width: tf.Tensor, aspect_ratio: float, crop_proportion: float):
  """Compute aspect ratio-preserving shape for central crop."""
  image_width_float = tf.cast(image_width, tf.float32)
  image_height_float = tf.cast(image_height, tf.float32)

  def _requested_aspect_ratio_wider_than_image():
    # Requested aspect ratio (W/H) > image aspect ratio (img_W/img_H)
    # Means requested shape is relatively wider or less tall than the image.
    # To maintain crop_proportion on the 'wider' (requested) dimension,
    # we use crop_proportion of image_width.
    # Crop height will be smaller to match aspect_ratio.
    # crop_width = crop_proportion * image_width_float
    # crop_height = crop_width / aspect_ratio
    crop_height_float = crop_proportion / aspect_ratio * image_width_float
    crop_width_float = crop_proportion * image_width_float
    return tf.cast(tf.math.round(crop_height_float), tf.int32), \
           tf.cast(tf.math.round(crop_width_float), tf.int32)


  def _image_wider_than_requested_aspect_ratio():
    # Image aspect ratio (img_W/img_H) >= requested aspect ratio (W/H)
    # Means image is relatively wider or less tall than requested.
    # To maintain crop_proportion on the 'taller' (requested) dimension,
    # we use crop_proportion of image_height.
    # Crop width will be smaller to match aspect_ratio.
    # crop_height = crop_proportion * image_height_float
    # crop_width = crop_height * aspect_ratio
    crop_height_float = crop_proportion * image_height_float
    crop_width_float = crop_proportion * aspect_ratio * image_height_float
    return tf.cast(tf.math.round(crop_height_float), tf.int32), \
           tf.cast(tf.math.round(crop_width_float), tf.int32)

  return tf.cond(
      aspect_ratio > image_width_float / image_height_float,
      _requested_aspect_ratio_wider_than_image,
      _image_wider_than_requested_aspect_ratio)


def center_crop(image: tf.Tensor, height: int, width: int, crop_proportion: float) -> tf.Tensor:
  """Crops to center of image and rescales to desired size."""
  shape = tf.shape(input=image)
  image_height = shape[0]
  image_width = shape[1]
  # Target aspect ratio for the crop before resize
  target_aspect_ratio = tf.cast(width, tf.float32) / tf.cast(height, tf.float32)
  
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, target_aspect_ratio, crop_proportion)
  
  offset_height = (image_height - crop_height + 1) // 2
  offset_width = (image_width - crop_width + 1) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_height, crop_width)

  # Resize to final output size
  image = tf.image.resize(images=[image], size=[height, width], # Pass image as a batch
                          method=tf.image.ResizeMethod.BICUBIC)[0] # Take first from batch
  return image


def distorted_bounding_box_crop(image: tf.Tensor,
                                bbox: tf.Tensor,
                                min_object_covered: float = 0.1,
                                aspect_ratio_range: tuple[float, float] = (0.75, 1.33),
                                area_range: tuple[float, float] = (0.05, 1.0),
                                max_attempts: int = 100,
                                scope: str | None = None) -> tf.Tensor:
  """Generates cropped_image using one of the bboxes randomly distorted."""
  with tf.name_scope(scope or 'distorted_bounding_box_crop'):
    shape = tf.shape(input=image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        image_size=shape, # Renamed arg
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, target_height, target_width)
    return image


def crop_and_resize(image: tf.Tensor, height: int, width: int) -> tf.Tensor:
  """Make a random crop and resize it to height `height` and width `width`."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  target_aspect_ratio = tf.cast(width, tf.float32) / tf.cast(height, tf.float32)
  image = distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4. * target_aspect_ratio, 4. / 3. * target_aspect_ratio), # Use . for float
      area_range=(0.08, 1.0),
      max_attempts=100,
      scope=None)
  return tf.image.resize(images=[image], size=[height, width], # Pass image as a batch
                         method=tf.image.ResizeMethod.BICUBIC)[0] # Take first


def gaussian_blur(image: tf.Tensor, kernel_size: int, sigma: float | tf.Tensor, padding: str = 'SAME') -> tf.Tensor:
  """Blurs the given image with separable convolution."""
  # Ensure kernel_size is odd
  if isinstance(kernel_size, tf.Tensor):
      radius = kernel_size // 2
  else: # Python int
      radius = int(kernel_size / 2) # Python int division
  
  actual_kernel_size = radius * 2 + 1
  
  x_coords = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
  sigma_float = tf.cast(sigma, dtype=tf.float32)
  
  blur_filter_1d = tf.exp(-tf.pow(x_coords, 2.0) /
                          (2.0 * tf.pow(sigma_float, 2.0)))
  blur_filter_1d = blur_filter_1d / tf.reduce_sum(input_tensor=blur_filter_1d)
  
  # Reshape for depthwise_conv2d: [filter_height, filter_width, in_channels, channel_multiplier]
  # For separable, in_channels is 1 (depthwise), channel_multiplier is num_image_channels.
  # However, tf.nn.depthwise_conv2d's filter has shape [filter_height, filter_width, in_channels, channel_multiplier]
  # where in_channels is image channels, and channel_multiplier is 1.
  
  blur_v = tf.reshape(blur_filter_1d, [actual_kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter_1d, [1, actual_kernel_size, 1, 1])
  
  num_channels = tf.shape(input=image)[-1]
  # Tile to match input channels for depthwise convolution (each channel convolved with its own filter copy)
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1]) # [1, K, C, 1]
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1]) # [K, 1, C, 1]

  expand_batch_dim = (image.shape.rank == 3) # Use rank instead of ndims
  if expand_batch_dim:
    image_batched = tf.expand_dims(image, axis=0)
  else:
    image_batched = image
  
  blurred = tf.nn.depthwise_conv2d(
      input=image_batched, filter=blur_h, strides=[1, 1, 1, 1], padding=padding.upper()) # Use .upper()
  blurred = tf.nn.depthwise_conv2d(
      input=blurred, filter=blur_v, strides=[1, 1, 1, 1], padding=padding.upper())
  
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


def random_crop_with_resize(image: tf.Tensor, height: int, width: int, p: float = 1.0) -> tf.Tensor:
  """Randomly crop and resize an image."""
  def _transform(img_to_transform):
    img_to_transform = crop_and_resize(img_to_transform, height, width)
    return img_to_transform
  return random_apply(_transform, p=p, x=image)


def random_color_jitter(image: tf.Tensor,
                        p: float = 1.0,
                        strength: float = 1.0,
                        impl: str = 'simclrv2') -> tf.Tensor:
  """Applies color jitter and optionally grayscale conversion."""
  def _transform(img_to_transform):
    # Apply color jitter with 80% probability (fixed within this transform)
    # The outer random_apply (with p) controls if this whole block is applied.
    # The strength argument is passed to color_jitter function.
    img_to_transform = random_apply(
        lambda x: color_jitter(x, strength=strength, impl=impl),
        p=0.8, # As per SimCLR paper, 80% for color jitter itself
        x=img_to_transform
    )
    # Apply grayscale with 20% probability
    img_to_transform = random_apply(to_grayscale, p=0.2, x=img_to_transform)
    return img_to_transform
  
  # Apply the entire _transform block (jitter + grayscale) with probability p
  return random_apply(_transform, p=p, x=image)


def random_blur(image: tf.Tensor, height: int, width: int, p: float = 1.0) -> tf.Tensor:
  """Randomly blur an image."""
  del width # width not used for kernel_size calculation, height is used.
  def _transform(img_to_transform):
    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    # Kernel size is 10% of image height, ensure it's odd.
    kernel_val = tf.cast(tf.cast(height, tf.float32) / 10.0, tf.int32)
    kernel_val = kernel_val + (1 - kernel_val % 2) # Make it odd: if even add 1, if odd add 0
    return gaussian_blur(
        img_to_transform, kernel_size=kernel_val, sigma=sigma, padding='SAME')
  return random_apply(_transform, p=p, x=image)


def batch_random_blur(images_list: list[tf.Tensor],
                      height: int,
                      width: int,
                      blur_probability: float = 0.5) -> list[tf.Tensor]:
  """Apply efficient batch data transformations (specifically blur here)."""
  def generate_selector(prob_p, batch_sz):
    shape = [batch_sz, 1, 1, 1] # For broadcasting with HWC images
    selector = tf.cast(
        tf.less(tf.random.uniform(shape, 0.0, 1.0, dtype=tf.float32), prob_p), # Ensure 0.0, 1.0
        tf.float32)
    return selector

  new_images_list_processed = []
  for images_batch in images_list: # Iterate through list of batches
    # Apply blur to all images in the batch first
    # The random_blur function itself has a probability p for applying blur.
    # Here, we want to deterministically blur some images in the batch, controlled by blur_probability.
    # So, we call gaussian_blur directly and then select.
    
    # Generate sigmas for each image in the batch if they are to be blurred
    batch_size_current = tf.shape(images_batch)[0]
    sigmas_for_blur = tf.random.uniform([batch_size_current], 0.1, 2.0, dtype=tf.float32)
    kernel_val = tf.cast(tf.cast(height, tf.float32) / 10.0, tf.int32)
    kernel_val = kernel_val + (1 - kernel_val % 2)

    # Apply blur to each image individually - can be slow.
    # For batch, tf.vectorized_map or manual loop if shapes vary.
    # Or, if sigma and kernel_size are same for all in batch, can do one blur call.
    # The original random_blur blurs an entire batch with *one* random sigma.
    # This batch_random_blur implies *some* images in the batch get blurred.
    
    # Let's assume for simplicity, if a batch is selected for blur, all images in it use one random sigma.
    # This matches behavior if random_blur was called on images_batch directly.
    # The selector then chooses between original and blurred batch.
    
    # Alternative: blur each image if selected, more complex to batch efficiently.
    # Sticking to simpler: blur the whole batch, then select.
    
    # Generate blurred version of the entire batch
    # This applies one random sigma to the entire batch if random_blur is called on the batch.
    # For a per-image random sigma, one would need to loop or use map_fn.
    # Let's keep it simple: one sigma for the blurred version of the batch.
    blurred_images_batch = random_blur(images_batch, height, width, p=1.0) # p=1 to ensure it attempts blur

    selector = generate_selector(blur_probability, batch_size_current)
    images_selected = blurred_images_batch * selector + images_batch * (1.0 - selector) # Ensure 1.0
    images_final = tf.clip_by_value(images_selected, 0.0, 1.0) # Ensure 0.0, 1.0
    new_images_list_processed.append(images_final)

  return new_images_list_processed


def preprocess_for_train(image: tf.Tensor,
                         height: int,
                         width: int,
                         color_jitter_strength: float = 0.0, # Default to 0.0 for clarity
                         crop: bool = True,
                         flip: bool = True,
                         impl: str = 'simclrv2') -> tf.Tensor:
  """Preprocesses the given image for training."""
  if crop:
    image = random_crop_with_resize(image, height, width, p=1.0) # p=1.0 for train crop
  if flip:
    image = tf.image.random_flip_left_right(image)
  if color_jitter_strength > 0.0: # Explicit check
    # random_color_jitter internally has its own probabilities for jitter sub-ops
    image = random_color_jitter(image, p=1.0, strength=color_jitter_strength, impl=impl) # p=1.0 to apply jitter block
  
  # Ensure shape and clip values
  image = tf.reshape(image, [height, width, 3])
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image


def preprocess_for_eval(image: tf.Tensor, height: int, width: int, crop: bool = True) -> tf.Tensor:
  """Preprocesses the given image for evaluation."""
  if crop:
    image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
  
  image = tf.reshape(image, [height, width, 3])
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image


def preprocess_image(image: tf.Tensor,
                     height: int,
                     width: int,
                     is_training: bool = False,
                     color_jitter_strength: float = 0.0, # Default to 0 for eval
                     test_crop: bool = True,
                     impl: str = 'simclrv2' # Added impl here
                     ) -> tf.Tensor:
  """Preprocesses the given image."""
  image = tf.image.convert_image_dtype(image, dtype=tf.float32) # Normalize to [0,1]
  if is_training:
    return preprocess_for_train(
        image, height, width, color_jitter_strength=color_jitter_strength, impl=impl)
  else:
    return preprocess_for_eval(image, height, width, crop=test_crop)