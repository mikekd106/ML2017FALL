import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor

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


def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
	sigma_inverse = np.linalg.inv(shared_sigma)
	w = np.dot( (mu1-mu2), sigma_inverse)
	x = X_valid.T
	b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
	a = np.dot(w, x) + b
	y = sigmoid(a)
	y_ = np.around(y)
	result = (np.squeeze(Y_valid) == y_)
	print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
	return
def train(X_all, Y_all, save_dir):
    # Split a 10%-validation set from the training set
	valid_set_percentage = 0.1
	X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    
    # Gaussian distribution parameters
	train_data_size = X_train.shape[0]
	cnt1 = 0
	cnt2 = 0

	mu1 = np.zeros((106,))
	mu2 = np.zeros((106,))
	for i in range(train_data_size):
		if Y_train[i] == 1:
			mu1 += X_train[i]
			cnt1 += 1
		else:
			mu2 += X_train[i]
			cnt2 += 1
	mu1 /= cnt1
	mu2 /= cnt2

	sigma1 = np.zeros((106,106))
	sigma2 = np.zeros((106,106))
	for i in range(train_data_size):
		if Y_train[i] == 1:
			sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
		else:
			sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
	sigma1 /= cnt1
	sigma2 /= cnt2
	shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
	N1 = cnt1
	N2 = cnt2

	print('=====Saving Param=====')
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	param_dict = {'mu1':mu1, 'mu2':mu2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
	for key in sorted(param_dict):
		print('Saving %s' % key)
		np.savetxt(os.path.join(save_dir, ('%s' % key)), param_dict[key])
    
	print('=====Validating=====')
	valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)

	

def infer(X_test, save_dir, out_path):
    # Load parameters
	print('=====Loading Param from %s=====' % save_dir)
	mu1 = np.loadtxt(os.path.join(save_dir, 'mu1'))
	mu2 = np.loadtxt(os.path.join(save_dir, 'mu2'))
	shared_sigma = np.loadtxt(os.path.join(save_dir, 'shared_sigma'))
	N1 = np.loadtxt(os.path.join(save_dir, 'N1'))
	N2 = np.loadtxt(os.path.join(save_dir, 'N2'))

    # Predict
	sigma_inverse = np.linalg.inv(shared_sigma)
	w = np.dot( (mu1-mu2), sigma_inverse)
	x = X_test.T
	b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
	a = np.dot(w, x) + b
	y = sigmoid(a)
	y_ = np.around(y)

	print('=====Write output to %s =====' % out_path)
    # Write output
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

