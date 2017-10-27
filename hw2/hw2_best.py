import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import tensorflow as tf

def load_data(train_data_path,train_label_path,test_data_path):
	train_data = pd.read_csv(train_data_path,sep=',',header=0)
	train_data = np.array(train_data.values)
	train_label = pd.read_csv(train_label_path,sep=',',header=0)
	train_label = np.array(train_label.values)
	test_data = pd.read_csv(test_data_path,sep=',',header=0)
	test_data = np.array(test_data.values)

	return (train_data,train_label,test_data)
def normalize(X_all,X_test):
	X_train_test = np.concatenate((X_all,X_test))
	mu = sum(X_train_test)/X_train_test.shape[0]
	mu = np.tile(mu,(X_train_test.shape[0],1))
	sigma = np.std(X_train_test,axis = 0)
	sigma = np.tile(sigma,(X_train_test.shape[0],1))
	X_train_test_normed = (X_train_test - mu)/sigma

	X_all = X_train_test_normed[:X_all.shape[0],:]
	X_test = X_train_test_normed[X_all.shape[0]:,:]

	return (X_all,X_test)

def _shuffle(X,Y):

	ran = np.arange(len(X))
	np.random.shuffle(ran)
	return (X[ran],Y[ran])

def split_valid_set(X_all,Y_all,split_validation_percetage):
	X_all,Y_all = _shuffle(X_all,Y_all)

	total_data_num = len(X_all)
	valid_data_num = int(floor(total_data_num*split_validation_percetage))

	X_valid = X_all[:valid_data_num,:]
	Y_valid = Y_all[:valid_data_num,:]

	X_train = X_all[valid_data_num:,:]
	Y_train = Y_all[valid_data_num:,:]

	return (X_train,Y_train,X_valid,Y_valid)

def transformY(Y):
	Y_ = []
	for i in range(len(Y)):
		if Y[i][0] == 1:
			Y_.append([1,0])
		elif Y[i][0] == 0:   
			Y_.append([0,1])
	return np.array(Y_)
def train(X_all,Y_all,out_path,X_test):


	Y_all = transformY(Y_all)
	
	#split validation set from training set
	split_validation_percetage = 0.1
	X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, split_validation_percetage)
	X_train = np.concatenate((X_train,X_valid))
	Y_train = np.concatenate((Y_train,Y_valid))
	# Initiallize parameter, hyperparameter
	
	l_rate = 0.01
	batch_size = 32
	train_data_size = len(X_train)
	step_num = int(floor(train_data_size / batch_size))
	epoch_num = 100
	save_param_iter = 5

	n_hidden_1 = 1024 # 1st layer number of neurons
	n_hidden_2 = 1024
	num_input = X_train.shape[1]

	X = tf.placeholder('float',[None,num_input])
	Y = tf.placeholder('float',[None,2])

	weights = {
		'h1': tf.Variable(tf.random_normal([num_input,n_hidden_1],stddev=0.001)),
		'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=0.001)),
		'out': tf.Variable(tf.random_normal([n_hidden_2,2],stddev=0.001))
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([2]))
	}
	def neural_net(x):
		# Hidden fully connected layer with 256 neurons
		layer_1 = tf.nn.relu6(tf.add(tf.matmul(x,weights['h1']),biases['b1']))
		layer_2 = tf.nn.relu6(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))
		out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
		return out_layer


	# Construct model
	logits = neural_net(X)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
	optimizer = tf.train.AdagradOptimizer(learning_rate=l_rate)
	train_op = optimizer.minimize(loss_op)

	correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	argm = tf.argmax(logits,1)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(1,epoch_num):

			if (epoch) % save_param_iter == 0:
				print ('epoch = %d'%epoch)
				acc = sess.run(accuracy,feed_dict={X:X_valid,Y:Y_valid})
				print("validation Accuracy= " +  "{:.3f}".format(acc))
			
			X_train,Y_train = _shuffle(X_train,Y_train)

			for step in range(step_num):
				X_t = X_train[step*batch_size:(step+1)*batch_size,:]
				Y_t = Y_train[step*batch_size:(step+1)*batch_size,:]
				sess.run(train_op,feed_dict={X:X_t,Y:Y_t})

		




		result = sess.run(argm,feed_dict={X:X_test})
		
		answer = []
		for index in range(len(X_test)):
			if result[index] == 0:
				answer.append(1)
			elif result[index] == 1:
				answer.append(0)
		answer = np.array(answer)
		
		with open(out_path, 'w') as f:
			f.write('id,label\n')
			for i, v in enumerate(answer):
				f.write('%d,%d\n' %(i+1, v))


def main():
	#load data
	input_train = sys.argv[1]
	label_train = sys.argv[2]
	input_test = sys.argv[3]
	out_path = sys.argv[4]

	X_all, Y_all, X_test = load_data(input_train,label_train,input_test)
	
	#normalize data
	X_all, X_test = normalize(X_all, X_test)
	
	train(X_all,Y_all,out_path,X_test)	
	#infer(X_test,'models','results')

if __name__ == '__main__':
	

	main()