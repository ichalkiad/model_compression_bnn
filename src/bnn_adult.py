from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Normal
import sys
import csv 
from sklearn import preprocessing                                                                                                                                                                                   
                                                                                                                                                                                                                 
 
tf.flags.DEFINE_integer("N", default=48842, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=14, help="Number of features.")

FLAGS = tf.flags.FLAGS


def preprocess_data(filename='../data/Adult/adult.data'):

    workclass = {'Private':0, 'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3, 'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked':7}
    education = {'Bachelors':0, 'Some-college':1, '11th':2, 'HS-grad':3, 'Prof-school':4, 'Assoc-acdm':5, 'Assoc-voc':6, '9th':7, '7th-8th':8, '12th':9, 'Masters':10, '1st-4th':11, '10th':12, 'Doctorate':13, '5th-6th':14, 'Preschool':15}
    marital_status = {'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2, 'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6} 
    occupation = {'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12, 'Armed-Forces':13} 
    relationship = {'Wife':0, 'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5} 
    race = {'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4} 
    sex = {'Female':0, 'Male':1} 
    native_country = {'United-States':0, 'Cambodia':1, 'England':2, 'Puerto-Rico':3, 'Canada':4, 'Germany':5, 'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9, 'South':10, 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14, 'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19, 'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23, 'Dominican-Republic':24, 'Laos':25, 'Ecuador':26, 'Taiwan':27, 'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31, 'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39, 'Holand-Netherlands':40}
    income = {'>50K':1, '<=50K':0}
 
    X = np.zeros((FLAGS.N,FLAGS.D))
    y = np.zeros((FLAGS.N,))
    i = 0
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, skipinitialspace=True, delimiter=',')
        for row in spamreader:
            if '?' in row:
               continue
            elif row==[]:
               continue
            else:
               X[i,0] = float(row[0])
               X[i,1] = float(workclass[row[1]])
               X[i,2] = float(row[2])
               X[i,3] = float(education[row[3]])
               X[i,4] = float(row[4]) 
               X[i,5] = float(marital_status[row[5]])
               X[i,6] = float(occupation[row[6]])
               X[i,7] = float(relationship[row[7]])
               X[i,8] = float(race[row[8]])
               X[i,9] = float(sex[row[9]])
               X[i,10] = float(row[10])
               X[i,11] = float(row[11])
               X[i,12] = float(row[12])
               X[i,13] = float(native_country[row[13]])
               y[i] =  float(income[row[14]])
 
            i = i + 1


    scaler = preprocessing.StandardScaler().fit(X)                                                                                                                                                     
    data_train_scaled = scaler.transform(X)  
   
    return data_train_scaled, y


def main():
  def neural_network(X):
    h = tf.sigmoid(tf.matmul(X, W_0) + b_0)
    h = tf.sigmoid(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])
  ed.set_seed(42)

  # DATA
  X_train, y_train = preprocess_data(filename='../data/Adult/adult.data')

  # MODEL
  with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([FLAGS.D, 5]), scale=tf.ones([FLAGS.D, 5]),
                 name="W_0")
    W_1 = Normal(loc=tf.zeros([5, 5]), scale=tf.ones([5, 5]), name="W_1")
    W_2 = Normal(loc=tf.zeros([5, 1]), scale=tf.ones([5, 1]), name="W_2")
    b_0 = Normal(loc=tf.zeros(5), scale=tf.ones(5), name="b_0")
    b_1 = Normal(loc=tf.zeros(5), scale=tf.ones(5), name="b_1")
    b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")

    X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D], name="X")
    y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(FLAGS.N), name="y")

  # INFERENCE
  with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
      loc = tf.get_variable("loc", [FLAGS.D, 5])
      scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.D, 5]))
      qW_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_1"):
      loc = tf.get_variable("loc", [5, 5])
      scale = tf.nn.softplus(tf.get_variable("scale", [5, 5]))
      qW_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_2"):
      loc = tf.get_variable("loc", [5, 1])
      scale = tf.nn.softplus(tf.get_variable("scale", [5, 1]))
      qW_2 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_0"):
      loc = tf.get_variable("loc", [5])
      scale = tf.nn.softplus(tf.get_variable("scale", [5]))
      qb_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_1"):
      loc = tf.get_variable("loc", [5])
      scale = tf.nn.softplus(tf.get_variable("scale", [5]))
      qb_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_2"):
      loc = tf.get_variable("loc", [1])
      scale = tf.nn.softplus(tf.get_variable("scale", [1]))
      qb_2 = Normal(loc=loc, scale=scale)

  inference = ed.KLqp({W_0: qW_0, b_0: qb_0,W_1: qW_1, b_1: qb_1,W_2: qW_2, b_2: qb_2},  data={X:X_train.astype(float), y:y_train.astype(float)})

  #inference.run()
  inference.initialize(n_iter=5000000,logdir='log')
  sess = ed.get_session()
  init = tf.global_variables_initializer()
  init.run()
  learning_curve = []
  for _ in range(inference.n_iter):
      info_dict = inference.update()
      if _%1000 == 0:
          print(info_dict)
      learning_curve.append(info_dict['loss'])




if __name__ == "__main__":
  main()
