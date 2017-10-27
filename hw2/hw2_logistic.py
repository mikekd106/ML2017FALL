import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd


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

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

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

def valid(weight, bias, X_valid, Y_valid):
	valid_data_size = len(X_valid)
	z = (np.dot(X_valid,np.transpose(weight))+bias)
	y = sigmoid(z)
	y_ = np.around(y)
	result = (np.squeeze(Y_valid) == y_)
	print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    

def train(X_all,Y_all,save_dir):

	#split validation set from training set
	split_validation_percetage = 0.1
	X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, split_validation_percetage)
	
	#X_train = np.concatenate((X_train,X_valid))
	#Y_train = np.concatenate((Y_train,Y_valid))
	# Initiallize parameter, hyperparameter
	weight = np.zeros((106,))
	bias = np.zeros((1,))
	l_rate = 0.01
	batch_size = 32
	train_data_size = len(X_train)
	step_num = int(floor(train_data_size / batch_size))
	epoch_num = 1000
	save_param_iter = 50


    #start training
	total_loss = 0.0
	for epoch in range(1,epoch_num):

		if (epoch) % save_param_iter == 0:
			print('=====Saving Param at epoch %d=====' % epoch)
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			np.savetxt(os.path.join(save_dir, 'w'), weight)
			np.savetxt(os.path.join(save_dir, 'b'), [bias,])
			print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
			total_loss = 0.0
			valid(weight, bias, X_valid, Y_valid)

		X_train,Y_train = _shuffle(X_train,Y_train)

		for step in range(step_num):
			X = X_train[step*batch_size:(step+1)*batch_size,:]
			Y = Y_train[step*batch_size:(step+1)*batch_size,:]

			z = np.dot(X,np.transpose(weight))+bias
			y = sigmoid(z)

			cross_entropy = -1*(np.dot(np.squeeze(Y),np.log(y))+np.dot((1-np.squeeze(Y)),np.log(1-y)))
			total_loss = total_loss + cross_entropy

			w_gra = np.mean(-1*X*(np.squeeze(Y)-y).reshape((batch_size,1)),axis=0)
			b_gra = np.mean(-1*(np.squeeze(Y)-y))

			weight = weight - l_rate*w_gra
			bias = bias -l_rate*b_gra
def infer(X_test, save_dir, out_path):
	test_data_size = len(X_test)

    # Load parameters
	print('=====Loading Param from %s=====' % save_dir)
	w = np.loadtxt(os.path.join(save_dir, 'w'))
	b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
	z = (np.dot(X_test, np.transpose(w)) + b)
	y = sigmoid(z)
	y_ = np.around(y)

	print('=====Write output to %s =====' % out_path)
	with open(out_path, 'w') as f:
		f.write('id,label\n')
		for i, v in  enumerate(y_):
			f.write('%d,%d\n' %(i+1, v))

def main():

	input_train = sys.argv[1]
	label_train = sys.argv[2]
	input_test = sys.argv[3]
	out_path = sys.argv[4]
	#load data
	X_all, Y_all, X_test = load_data(input_train,label_train,input_test)
	
	#normalize data
	X_all, X_test = normalize(X_all, X_test)
	
	train(X_all,Y_all,'models')	
	infer(X_test,'models',out_path)

if __name__ == '__main__':
	

	main()