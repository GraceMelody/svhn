from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features['x'], [-1, 32, 32, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

  # Dense Layer
  dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits))

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels,1), predictions=(predictions["classes"]))}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Open the file as readonly
  h5f = h5py.File('SVHN_grey.h5', 'r')
  
  # Load the training, test and validation set
  X_train = h5f['X_train'][:]
  y_train = h5f['y_train'][:]
  X_test = h5f['X_test'][:]
  y_test = h5f['y_test'][:]
  X_val = h5f['X_val'][:]
  y_val = h5f['y_val'][:]
  
  # Close this file
  h5f.close()
  print('ladaad',y_test[2])
  print('Training set', X_train.shape, y_train.shape)
  print('Validation set', X_val.shape, y_val.shape)
  print('Test set', X_test.shape, y_test.shape)

  svhn_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="test/svhnet")

#   tensors_to_log = {"probabilities": "softmax_tensor"}
#   logging_hook = tf.train.LoggingTensorHook(
#       tensors=tensors_to_log, every_n_iter=50)

#   # Train the model
#   train_input_fn = tf.estimator.inputs.numpy_input_fn(
#       x={'x':X_train},
#       y=y_train,
#       batch_size=512,
#       num_epochs=100,
#       shuffle=True)
#   svhn_classifier.train(
#       input_fn=train_input_fn,
#       steps=5000,
#       hooks=[logging_hook])

#   # Evaluate the model and print results
#   eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#       x={"x": X_test}, y=y_test, num_epochs=1, shuffle=False)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": X_test[2]}, num_epochs=1, shuffle=False)
  predict_results = svhn_classifier.predict(input_fn=predict_input_fn,)
  
  final = next(predict_results)
  print('Class detected: ',final['classes'])
  print('Probability: ',max(final['probabilities'])*100,'%')


if __name__ == "__main__":
  tf.app.run()