from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Normal
import sys

tf.flags.DEFINE_integer("N", default=48842, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=14, help="Number of features.")

FLAGS = tf.flags.FLAGS


def preprocess_data(filename='../data/Adult/adult.data'):

    X = np.zeros(FLAGS.N,FLAGS.D-1)
    y = np.zeros(FLAGS.N,1)
    
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            print(', '.join(row))

    print(X.shape)
    print(y.shape)
    
    return X, y


def main(_):
  def neural_network(X):
    h = tf.tanh(tf.matmul(X, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])
  ed.set_seed(42)

  # DATA
  X_train, y_train = preprocess_data(filename='../data/Adult/adult.data')
  sys.exit(0)

  # MODEL
  with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([FLAGS.D, 10]), scale=tf.ones([FLAGS.D, 10]),
                 name="W_0")
    W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]), name="W_1")
    W_2 = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]), name="W_2")
    b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_0")
    b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_1")
    b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")

    X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D], name="X")
    y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(FLAGS.N), name="y")

  # INFERENCE
  with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
      loc = tf.get_variable("loc", [FLAGS.D, 10])
      scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.D, 10]))
      qW_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_1"):
      loc = tf.get_variable("loc", [10, 10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10, 10]))
      qW_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_2"):
      loc = tf.get_variable("loc", [10, 1])
      scale = tf.nn.softplus(tf.get_variable("scale", [10, 1]))
      qW_2 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_0"):
      loc = tf.get_variable("loc", [10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10]))
      qb_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_1"):
      loc = tf.get_variable("loc", [10])
      scale = tf.nn.softplus(tf.get_variable("scale", [10]))
      qb_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_2"):
      loc = tf.get_variable("loc", [1])
      scale = tf.nn.softplus(tf.get_variable("scale", [1]))
      qb_2 = Normal(loc=loc, scale=scale)

  inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                       W_1: qW_1, b_1: qb_1,
                       W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
  inference.run(logdir='log')

if __name__ == "__main__":
  tf.app.run()
