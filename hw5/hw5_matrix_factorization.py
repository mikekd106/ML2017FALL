import matplotlib
matplotlib.use('Agg')
import os
import math
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Flatten, Add, Dot, Concatenate, Dropout
from keras.models import Model
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras import regularizers
import keras
import numpy as np
import pickle
import time,csv,os,sys
from random import shuffle
import argparse
from math import log, floor
import tensorflow as tf
from keras.models import load_model
def draw(x,y):
	y = np.array(y)
	x = np.array(x,dtype=np.float64)
	#perform t-SNE embedding
	vis_data = TSNE(n_components=2).fit_transform(x)
	#plot the result
	vis_x = vis_data[:,0]
	vis_y = vis_data[:,1]

	cm = plt.cm.get_cmap('RdYlBu')
	sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
	plt.colorbar(sc)
	plt.savefig('tsne.png')

def _shuffle(X1,X2,Y):
	ran = np.arange(len(X1))
	np.random.shuffle(ran)
	return (X1[ran],X2[ran],Y[ran])
def split_valid_set(X1_all,X2_all,Y_all,split_validation_percetage=0.01):
	X1_all, X2_all, Y_all = _shuffle(X1_all, X2_all, Y_all)

	total_data_num = len(X1_all)
	valid_data_num = int(floor(total_data_num*split_validation_percetage))

	X1_valid = X1_all[:valid_data_num]
	X2_valid = X2_all[:valid_data_num]
	Y_valid = Y_all[:valid_data_num]

	X1_train = X1_all[valid_data_num:]
	X2_train = X2_all[valid_data_num:]
	Y_train = Y_all[valid_data_num:]

	return (X1_train,X2_train,Y_train,X1_valid,X2_valid,Y_valid)
def normalize(rate):
	rate_mean = np.mean(rate)
	rate_std = np.std(rate)
	rate_normalized = (rate - rate_mean) / rate_std
	normalize_dict = {}
	normalize_dict['mean'] = rate_mean
	normalize_dict['std'] = rate_std
	with open('normalize_dict','wb') as f:
		pickle.dump(normalize_dict,f)
	return rate_normalized
def back_normalize(result):
	normalize_dict = pickle.load(open('normalize_dict','rb'))
	rate_mean = normalize_dict['mean']
	rate_std = normalize_dict['std']

	result = result * rate_std + rate_mean

	return result
def get_model(n_users,n_movies,latent_dims=200):
	user_input = Input(shape=[1])
	movie_input = Input(shape=[1])
	user_vec = Embedding(n_users, latent_dims,embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	movie_vec = Embedding(n_movies, latent_dims,embeddings_initializer='random_normal')(movie_input)
	movie_vec = Flatten()(movie_vec)
	user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
	user_bias = Flatten()(user_bias)
	movie_bias = Embedding(n_movies, 1, embeddings_initializer='zeros')(movie_input)
	movie_bias = Flatten()(movie_bias)
	r_hat = Dot(axes=1)([user_vec, movie_vec])
	r_hat = Add()([r_hat, user_bias, movie_bias])
	model = Model([user_input,movie_input],r_hat)
	model.compile(loss='mse',optimizer='adamax')
	model.summary()
	return model
def get_dnn_model(n_users,n_movies,latent_dims=256):
	user_input = Input(shape=[1])
	movie_input = Input(shape=[1])
	user_vec = Embedding(n_users, latent_dims,embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	movie_vec = Embedding(n_movies, latent_dims,embeddings_initializer='random_normal')(movie_input) #embeddings_regularizer=regularizers.l2(1e-5)
	movie_vec = Flatten()(movie_vec)
	merge_vec = Concatenate()([user_vec, movie_vec])
	hidden_vec = Dense(512,activation='relu')(merge_vec)
	hidden_vec = Dropout(0.2)(hidden_vec)
	hidden_vec = Dense(256,activation='relu')(hidden_vec)
	hidden_vec = Dropout(0.2)(hidden_vec)
	hidden_vec = Dense(128,activation='relu')(hidden_vec)
	output = Dense(1)(hidden_vec)
	model = Model([user_input, movie_input],output)
	model.compile(loss='mse', optimizer='adam')
	model.summary()
	return model
def load_train_data(train_data_path,movie_data_path,users_data_path):
	sers = pd.read_csv(users_data_path, sep='::',engine='python')
	train_data = pd.read_csv(train_data_path, sep=',',engine='python')
	movies = pd.read_csv(movie_data_path,sep='::',engine='python')
	userid = []
	movieid = []
	rate = []
	for i in range(train_data.index.size):
		userid.append(train_data['UserID'][i])
		movieid.append(train_data['MovieID'][i])
		rate.append(train_data['Rating'][i])
	userid = np.array(userid)
	movieid = np.array(movieid)
	rate = np.array(rate)
	return (userid, movieid, rate)
def load_test_data(test_data_path):
	test_data = pd.read_csv(test_data_path, sep=',',engine='python')
	userid = []
	movieid = []
	rate = []
	for i in range(test_data.index.size):
		userid.append(test_data['UserID'][i])
		movieid.append(test_data['MovieID'][i])
	userid = np.array(userid)
	movieid = np.array(movieid)

	return (userid, movieid)
def createlabel_TSNE(movie_emb):
	movies = pd.read_csv('data/movies.csv',sep='::',engine='python')
	movie_dict = {}
	movies_emb = []
	y = []
	movies['Genres'] = movies.Genres.str.split('|')
	for i in range(movies.index.size):
		movie_dict[movies['movieID'][i]] = movies['Genres'][i]
	for i in range(len(movie_emb)):
		if i in movie_dict:
			arr = movie_dict[i]
			if any("Drama" in s for s in arr) or any("Musical" in s for s in arr):
				y.append(0)
				movies_emb.append(movie_emb[i])
			elif any("Thriller" in s for s in arr) or any("Horror" in s for s in arr) or any('Crime' in s for s in arr):
				y.append(1)
				movies_emb.append(movie_emb[i])
			elif any("Adventure" in s for s in arr) or any("Animation" in s for s in arr) or any("Children's" in s for s in arr):
				y.append(2)
				movies_emb.append(movie_emb[i])
	return movies_emb, y
def train(userid, movieid, rate, save_dir):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_model_path = os.path.join(save_dir, 'factorization_model.h5')
	epochs = 12
	batch_size = 128
	checkPoint = ModelCheckpoint(save_model_path,verbose=0,save_best_only=True,monitor='val_loss')
	callback_list = [checkPoint]
	userid_max = max(userid)
	movieid_max = max(movieid)
	model = get_model(userid_max+1,movieid_max+1)
	#model = get_dnn_model(userid_max+1,movieid_max+1)
	userid_train, movieid_train, rate_train, userid_valid, movieid_valid, rate_valid = split_valid_set(userid,movieid,rate)
	try:
		history = model.fit([userid_train,movieid_train],rate_train,epochs=epochs,batch_size=batch_size,
					validation_data=([userid_valid,movieid_valid],rate_valid),callbacks=callback_list)
	except KeyboardInterrupt:
		print()
	'''
	user_emb = np.array(model.layers[2].get_weights()).squeeze()
	print('user embeddings shape:',user_emb.shape)
	movie_emb = np.array(model.layers[3].get_weights()).squeeze()
	print('movie embeddings shape:',movie_emb.shape)
	movie_emb,label = createlabel_TSNE(movie_emb)
	draw(movie_emb,label)
	'''
	return 'train'
def infer(userid, movieid, output_path, save_dir):
	save_model_path = os.path.join(save_dir, 'factorization_model.h5')
	model = load_model(save_model_path)
	
	result = model.predict([userid,movieid])
	#result = back_normalize(result)
	for t in range(len(result)):
		if result[t] < 1.0:
			result[t] = 1.0
		elif result[t] > 5.0:
			result[t] = 5.0
	print('=====Write output to %s =====' % output_path)
	with open(output_path, 'w') as f:
		f.write('TestDataID,Rating\n')
		for i, v in  enumerate(result):
			f.write('%d,%f\n' %(i+1, v))
	return 'infer'
def main(opts):
	train_data_path = opts.train_data_path
	test_data_path = opts.test_data_path
	movie_data_path = opts.movie_data_path
	users_data_path = opts.users_data_path
	save_dir = opts.save_dir
	output_path = opts.output_path
	if opts.train:
		userid, movieid, rate = load_train_data(train_data_path,movie_data_path,users_data_path)
		#rate = normalize(rate)
		train_process = train(userid, movieid, rate, save_dir)
	elif opts.infer:
		userid, movieid = load_test_data(test_data_path)
		infer_process = infer(userid, movieid, output_path, save_dir)
	else:
		print('error: argument --train or --infer not found')
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='translation dask')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--train',action='store_true',default=False,dest='train',help='Input --train to Train')
	group.add_argument('--infer',action='store_true',default=False,dest='infer',help='Input --infer to infer')
	parser.add_argument('--train_data_path',type=str,default='data/train.csv',dest='train_data_path',help='path to training data')
	parser.add_argument('--test_data_path',type=str,default='data/test.csv',dest='test_data_path',help='path to testing data')
	parser.add_argument('--movie_data_path',type=str,default='data/movies.csv',dest='movie_data_path',help='path to movies data')
	parser.add_argument('--users_data_path',type=str,default='data/users.csv',dest='users_data_path',help='path to users data')
	parser.add_argument('--save_dir',type=str,default='models/',dest='save_dir',help='path to save the model parameters')
	parser.add_argument('--output_path',type=str,default='Result_output/result.csv',dest='output_path',help='path to save the model outputs')
	
	opts = parser.parse_args()

	main(opts)