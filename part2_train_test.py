# Creator: Hao Sky Zhou.
# Version: V1.0

# Run time with GPU is: 10175.426006317139 ms. for setting (epoch = 2, H1 = 500, H2 = 300, H1 = 100.)
# Run time with GPU is: 49308.2480430603 ms. for setting (epoch = 10, H1 = 1000, H2 = 500, H1 = 250.)

# Note that we are using python 3.6 instead of python 2.7.

from __future__ import print_function, division

import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import subprocess
import pickle 
import _pickle as cPickle

import boto3
import uuid

import logging
# import watchtower
import datetime
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# os.chdir(path)


def main():

	df = pd.read_csv('content_train.tsv', sep ='\t')

	del df['content_2'], df['content_3'], df['content_4'], df['content_5'], df['content_6'], df['content_7'], df['content_8'], df['content_9'], df['county']

	# df['content_1'] = df['content_1'].fillna(0)
	df = df.fillna(0)

	Y = df[['customer.id','content_1']]
	Y_label = Y['content_1']
	X = df.iloc[:,2:]

	# Create a boolean mask for categorical columns
	categorical_feature_mask = X.dtypes == object

	# Get list of categorical column names
	categorical_columns = X.columns[categorical_feature_mask].tolist()

	# Get list of non-categorical column names
	non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

	X[non_categorical_columns] = X[non_categorical_columns].apply(lambda x: (x-x.mean())/x.std())


	df_dict = X.to_dict("records")

	# Create the DictVectorizer object: dv
	dv = DictVectorizer(sparse=False)

	# Apply dv on df: df_encoded
	df_encoded = dv.fit_transform(df_dict)

	# Print the resulting first five rows
	print(df_encoded[:5,:])

	# Print the vocabulary
	print(dv.vocabulary_)

	Xtrain, Xtest, ytrain, ytest = train_test_split(df_encoded, Y.as_matrix(), test_size=0.3, random_state=123)

	y_train_label = ytrain[:, 1].reshape((len(ytrain), 1))
	y_test_label = ytest[:, 1].reshape((len(ytest), 1))


	max_iter = 2
	print_period = 10

	lr = 0.001
	reg = 0.01

	N, D = Xtrain.shape
	batch_sz = 256
	n_batches = N // batch_sz

	M1 = 500  # number of hidden nodes in layer 1.
	M2 = 300  # number of hidden nodes in layer 1.
	M3 = 100  # number of hidden nodes in layer 1.

	K = y_train_label.shape[1]   # number of label categories.

	W1_init = np.random.randn(D, M1) / np.sqrt(D)
	b1_init = np.zeros(M1)
	W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
	b2_init = np.zeros(M2)
	W3_init = np.random.randn(M2, M3) / np.sqrt(M2)
	b3_init = np.zeros(M3)

	W4_init = np.random.randn(M3, K) / np.sqrt(M3)
	b4_init = np.zeros(K)

	# create tf placeholders for input and target.
	Xtf = tf.placeholder(tf.float32, shape = (None, D), name = 'X')
	Ttf = tf.placeholder(tf.float32, shape = (None, K), name = 'T')

	# create tf variables for weights initializations.
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))

	W4 = tf.Variable(W4_init.astype(np.float32))
	b4 = tf.Variable(b4_init.astype(np.float32))

	# activations in each layer.
	Z1 = tf.nn.relu( tf.matmul(Xtf, W1) + b1)
	Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2)
	Z3 = tf.nn.relu( tf.matmul(Z2, W3) + b3)

	Ylast = tf.matmul(Z3, W4) + b4

	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Ttf, logits=Ylast)
	cost = tf.reduce_mean(cross_entropy)
	trainOp = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

	
	# predictOp = tf.argmax(Ylast, 1)

	predictOp = tf.nn.sigmoid(Ylast)
	correct_pred = tf.equal(tf.round(predictOp), Ttf)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	test_costs = []
	test_erArray = []

	# utilityObj = requiredUtility()
	start_time = time.time()
	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

	with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
		session.run(init)
		writer = tf.summary.FileWriter("./output", session.graph)

		for i in range(max_iter):
			for j in range(n_batches):
				Xbatch = Xtrain[j*batch_sz:min((j*batch_sz + batch_sz), len(Xtrain)),]

				Ybatch = y_train_label[j*batch_sz:min((j*batch_sz + batch_sz), len(y_train_label)),]

				session.run(trainOp, feed_dict = {Xtf: Xbatch, Ttf: Ybatch})

				if j % print_period == 0:
					costTest = session.run(cost, feed_dict = {Xtf: Xtest, Ttf: y_test_label})
					test_costs.append(costTest)
					prediction = session.run(predictOp, feed_dict = {Xtf: Xtest})
					acc = session.run(accuracy, feed_dict = {Xtf: Xtest, Ttf: y_test_label})


					print("Cost and error at each i = %d, j = %d: %.3f, %.3f" %(i, j, costTest, acc))

		prediction = session.run(predictOp, feed_dict = {Xtf: Xtest})

		# probability_output = session.run(probability_final, feed_dict = {Xtf: Xtest})
		result = np.append(prediction, np.reshape( y_test_label, (len(y_test_label), 1)), axis=1)
		np.savetxt("./prediction.csv", result, delimiter=",")
	

		writer.close()

	# -- step 4: Push the model to S3 bucket.



	# Z1_test = tf.nn.relu( tf.matmul(Xtf, W1) + b1)
	# Z2__test = tf.nn.relu( tf.matmul(Z1, W2) + b2)
	# Ylast = tf.matmul(Z2, W3,) + b3

	# plt.plot(erArray)
	# plt.show()

	# plt.plot(costs)
	# plt.show()

if __name__ == '__main__':

	main()