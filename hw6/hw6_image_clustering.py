import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np 
import time,os,sys
from math import log, floor
import pickle
from random import shuffle
import argparse
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras import regularizers
def draw():
	visual_data = np.load('data/visualization.npy')
	visual_data = visual_data.astype(np.float)
	visual_data = visual_data/255
	print(visual_data.shape)
	encoder_file = os.path.join('models/','encoder_model.h5')
	encoder = load_model(encoder_file)
	visual_data_dnn = encoder.predict(visual_data)
	kmeans_model_file = os.path.join('models/', 'kmeans.pkl')
	k_means = pickle.load(open(kmeans_model_file,'rb'))
	label = k_means.predict(visual_data_dnn)
	
	y_tru = []
	for i in range(10000):
		if i < 5000:
			y_tru.append(0)
		else:
			y_tru.append(1)
	y_tru = np.array(y_tru)
	
	y = label
	x = visual_data_dnn
	#perform t-SNE embedding
	vis_data = TSNE(n_components=2).fit_transform(x)
	#plot the result
	vis_x = vis_data[:,0]
	vis_y = vis_data[:,1]

	plt.figure(1)
	cm = plt.cm.get_cmap('RdYlBu')
	sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm, s=10)
	plt.colorbar(sc)
	plt.savefig('tsne.png')
	
	plt.figure(2)
	cm = plt.cm.get_cmap('RdYlBu')
	sc = plt.scatter(vis_x, vis_y, c=y_tru, cmap=cm, s=10)
	plt.colorbar(sc)
	plt.savefig('tsne_all.png')
	
def add_mirroring(image_train):
	image_train = image_train.reshape(image_train.shape[0],28,28,1)
	rotate = np.flip(image_train,axis=2)
	image_train = np.concatenate((image_train,rotate),axis=0)
	image_train = image_train.reshape(image_train.shape[0],784)
	return image_train
def load_train_data(train_data_path):
	train_data = np.load(train_data_path)
	train_data = train_data.astype(np.float)
	train_data = train_data/255
	return train_data
def load_test_data(train_data_path, test_data_path):
	train_data = load_train_data(train_data_path)
	dataFrame = pd.read_csv(test_data_path)
	index1 = np.array(dataFrame.image1_index)
	index2 = np.array(dataFrame.image2_index)
	return train_data, index1, index2
def dimension_reduction_pca(n_components,train_data,save_dir):
	pca = PCA(n_components=n_components)
	train_data_pca = pca.fit_transform(train_data)
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	pca_model_file = os.path.join(save_dir, 'PCA.pkl')
	pickle.dump(pca, open(pca_model_file,'wb'))
	return train_data_pca
def dimension_reduction_dnn(encoding_dim,train_data,save_dir):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	dnn_model_file = os.path.join(save_dir, 'dnn_model.h5')
	encoder_file = os.path.join(save_dir,'encoder_model.h5')
	autoencoder, encoder = get_dnn_model(encoding_dim)
	total_data_num = len(train_data)
	valid_data_num = int(floor(total_data_num*0.01))
	x_test = train_data[:valid_data_num]
	x_train = train_data[valid_data_num:]
	checkPoint = ModelCheckpoint(dnn_model_file,verbose=0,save_best_only=True,monitor='val_loss')
	callback_list = [checkPoint]
	try:
		autoencoder.fit(x_train, x_train,
					epochs=100,
					batch_size=64,
					shuffle=True,
					validation_data=(x_test, x_test),callbacks=callback_list)
	except KeyboardInterrupt:
		print()
	train_data_dnn = encoder.predict(train_data)
	encoder.save(encoder_file)
	return train_data_dnn
def get_dnn_model(encoding_dim):
	input_img = Input(shape=(784,))
	encoded = Dense(128, activation='relu')(input_img)
	encoded = Dense(encoding_dim, activation='relu')(encoded)
	
	decoded = Dense(128, activation='relu')(encoded)
	decoded = Dense(784, activation='sigmoid')(decoded)
	autoencoder = Model(input_img, decoded)
	
	encoder = Model(input_img, encoded)
	'''
	encoded_input = Input(shape=(encoding_dim,))
	decoder_layer = autoencoder.layers[-1]
	decoder = Model(encoded_input, decoder_layer(encoded_input))
	'''
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	
	return autoencoder, encoder
def train_kmeans(train_data, save_dir):
	k_means = KMeans(init='k-means++', n_clusters=2)
	k_means.fit(train_data)
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	kmeans_model_file = os.path.join(save_dir, 'kmeans.pkl')
	pickle.dump(k_means, open(kmeans_model_file,'wb'))
	return 'train_kmeans'
def infer_kmeans(train_data, index1, index2, output_path, save_dir):
	kmeans_model_file = os.path.join(save_dir, 'kmeans.pkl')
	k_means = pickle.load(open(kmeans_model_file,'rb'))

	image1, image2 = get_image_vector(index1,index2,train_data,save_dir)
	pred1 = k_means.predict(image1)
	pred2 = k_means.predict(image2)
	result = []
	for i in range(len(pred1)):
		if pred1[i] == pred2[i]:
			result.append(1)
		else:
			result.append(0)
	write_output(result,output_path)
	return 'infer_kmeans'
def write_output(result,output_path):
	print('=====Write output to %s =====' % output_path)
	with open(output_path, 'w') as f:
		f.write('ID,Ans\n')
		for i, v in  enumerate(result):
			f.write('%d,%d\n' %(i, v))
def get_image_vector(index1,index2,train_data,save_dir):
	#pca_model_file = os.path.join(save_dir, 'PCA.pkl')
	#pca = pickle.load(open(pca_model_file,'rb'))
	#train_data_pca = pca.transform(train_data)
	
	encoder_file = os.path.join(save_dir,'encoder_model.h5')
	encoder = load_model(encoder_file)
	train_data_dnn = encoder.predict(train_data)
	
	image1 = []
	image2 = []
	for i in range(len(index1)):
		image1_index = index1[i]
		image2_index = index2[i]
		image1.append(train_data_dnn[image1_index])
		image2.append(train_data_dnn[image2_index])
	image1 = np.array(image1)
	image2 = np.array(image2)
	return image1, image2
def infer_similarity(train_data, index1, index2, output_path, save_dir):
	image1, image2 = get_image_vector(index1,index2,train_data,save_dir)
	threshold = 0.7
	result = []
	for i in range(len(image1)):
		cos_similarity = cosine_similarity(image1[i:i+1],image2[i:i+1])
		if cos_similarity[0][0] > threshold:
			result.append(1)
		else:
			result.append(0)
	write_output(result,output_path)
	return 'infer_similarity'
def main(opts):
	train_data_path = opts.train_data_path
	test_data_path = opts.test_data_path
	save_dir = opts.save_dir
	output_path = opts.output_path
	if opts.train:
		train_data = load_train_data(train_data_path) #140000x784
		train_data = add_mirroring(train_data)
		train_data_dnn = dimension_reduction_dnn(64,train_data,save_dir)
		#train_data_pca = dimension_reduction_pca(32,train_data,save_dir)
		train_process = train_kmeans(train_data_dnn, save_dir)
		
	elif opts.infer:
		
		train_data, index1, index2 = load_test_data(train_data_path,test_data_path)
		infer_process = infer_kmeans(train_data, index1, index2, output_path, save_dir)
		#infer_process = infer_similarity(train_data, index1, index2, output_path, save_dir)
		

		#draw()
	else:
		print('error: argument --train or --infer not found')


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='image clustering dask')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--train',action='store_true',default=False,dest='train',help='Input --train to Train')
	group.add_argument('--infer',action='store_true',default=False,dest='infer',help='Input --infer to infer')
	parser.add_argument('--train_data_path',type=str,default='data/image.npy',dest='train_data_path',help='path to training data')
	parser.add_argument('--test_data_path',type=str,default='data/test_case.csv',dest='test_data_path',help='path to testing data')
	parser.add_argument('--save_dir',type=str,default='models/',dest='save_dir',help='path to save the model parameters')
	parser.add_argument('--output_path',type=str,default='Result_output/result.csv',dest='output_path',help='path to save the model outputs')
	
	opts = parser.parse_args()

	main(opts)