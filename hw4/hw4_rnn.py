import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle
import time,csv,os,sys
from random import shuffle
import argparse
from math import log, floor
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding, LSTM, Bidirectional,GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
csv.field_size_limit(sys.maxsize)



def create_ngram_set(input_list, ngram_value=2):
	return set(zip(*[input_list[i:] for i in range(ngram_value)]))
def add_ngram(sequences, token_indice, ngram_range=2):
	new_sequences = []
	for input_list in sequences:
		new_list = input_list[:]
		for i in range(len(new_list) - ngram_range + 1):
			for ngram_value in range(2, ngram_range + 1):
				ngram = tuple(new_list[i:i + ngram_value])
				if ngram in token_indice:
					new_list.append(token_indice[ngram])
		new_sequences.append(new_list)
	return new_sequences
def load_train_label_data(train_label_data_path):	
	train_data = []
	train_label = []
	file = open(train_label_data_path,'r')
	csvCursor = csv.reader(file,delimiter=' ')
	for row in csvCursor:
		row = np.delete(row,1)
		train_label.append(int(row[0]))
		train_data.append(' '.join(row[1:]))
	file.close()
	return (train_data ,train_label)
def load_train_unlabel_data(train_unlabel_data_path):
	train_undata = []
	file = open(train_unlabel_data_path,'r')
	csvCursor = csv.reader(file,delimiter=' ')
	for row in csvCursor:
		train_undata.append(' '.join(row))
	file.close()
	return (train_undata)
def load_test_data(test_data_path):
	test_data = []
	with open(test_data_path) as f:
		for line in f:
			onesentence = line.split(',',1)
			test_data.append(onesentence[1].replace('\n',''))
	return test_data[1:]
def preprocess_trainData(train_data,max_features):
	tokenizer = Tokenizer(num_words=max_features,filters='"#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n')
	tokenizer.fit_on_texts(train_data)
	preprocess_trainData = tokenizer.texts_to_sequences(train_data)
	print(len(tokenizer.word_index))
	pickle.dump(tokenizer,open('tokenizer','wb'))
	return preprocess_trainData
def check_Index_before_oov(train_data):
	tokenizer = Tokenizer(filters='"#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n')
	tokenizer.fit_on_texts(train_data )
	arr = tokenizer.word_counts
	b = [item for item in arr.keys() if arr[item] is 5]#15
	dic = tokenizer.word_index
	return dic[b[0]]
def _shuffle(X,Y):
	X = np.array(X)
	Y = np.array(Y)
	ran = np.arange(len(X))
	np.random.shuffle(ran)
	return (X[ran],Y[ran])
def split_valid_set(X_all,Y_all,split_validation_percetage):
	X_all,Y_all = _shuffle(X_all,Y_all)

	total_data_num = len(X_all)
	valid_data_num = int(floor(total_data_num*split_validation_percetage))

	X_valid = X_all[:valid_data_num,:]
	Y_valid = Y_all[:valid_data_num]

	X_train = X_all[valid_data_num:,:]
	Y_train = Y_all[valid_data_num:]

	return (X_train,Y_train,X_valid,Y_valid)
def train(train_data,train_label,save_dir,max_features):
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_model_path = os.path.join(save_dir, 'RNN_model.h5')
	
	ngram_range = 1 #will add bi-grams features
	maxlen = 20
	batch_size = 32
	embedding_dims = 300#250
	epochs = 10
	print(len(train_data), 'train sequences')
	print('Average train sequence length: {}'.format(np.mean(list(map(len, train_data)), dtype=int)))

	if ngram_range > 1:
		print('Adding {}-gram features'.format(ngram_range))
		# Create set of unique n-gram from the training set.
		ngram_set = set()
		for input_list in train_data:
			for i in range(2, ngram_range + 1):
				set_of_ngram = create_ngram_set(input_list, ngram_value=i)
				ngram_set.update(set_of_ngram)

		# Dictionary mapping n-gram token to a unique integer.
		# Integer values are greater than max_features in order
		# to avoid collision with existing features.
		start_index = max_features + 1
		token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
		indice_token = {token_indice[k]: k for k in token_indice}
		with open('mydictionary','wb') as f:
			pickle.dump(token_indice,f)

		# max_features is the highest integer that could be found in the dataset.
		max_features = np.max(list(indice_token.keys())) + 1

		# Augmenting x_train and x_test with n-grams features
		train_data = add_ngram(train_data, token_indice, ngram_range)
		print('Average train sequence length: {}'.format(np.mean(list(map(len,train_data)), dtype=int)))
		
	print('Pad sequences (samples x time)')
	train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
	print('x_train shape:', train_data.shape)
	train_data,train_label,valid_data,valid_label = split_valid_set(train_data,train_label,0.05)
	#train_data = np.concatenate((train_data,valid_data),axis = 0)
	#train_label = np.concatenate((train_label,valid_label),axis = 0)

	checkPoint = ModelCheckpoint(save_model_path,verbose=0,save_best_only=True,monitor='val_acc')
	callback_list = [checkPoint]
	print('Build model...')
	model = Sequential()

	model.add(Embedding(max_features,embedding_dims,input_length=maxlen))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Bidirectional(GRU(256,dropout=0.2, recurrent_dropout=0.2)))
	model.add(Dense(1024,activation='relu'))
	model.add(Dense(1,activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	history = model.fit(train_data,train_label,epochs=epochs,batch_size=batch_size,validation_data=(valid_data,valid_label),callbacks=callback_list)
	score = model.evaluate(valid_data,valid_label,verbose=0)
	print('loss:',score[0])
	print('accuracy:',score[1])
	plt.plot(history.history['acc'],'r',linewidth=3.0)
	plt.plot(history.history['val_acc'],'b',linewidth=3.0)
	plt.legend(['Traing Accuracy','Validation Accuracy'],fontsize=18,loc='upper left')
	plt.xlabel('Epochs',fontsize=16)
	plt.ylabel('Accuracy',fontsize=16)
	plt.title('Accuracy Curves',fontsize=16)
	plt.savefig('RNNaccuracy.png')

	return 'train'
def infer(test_data,save_dir,output_path):
	maxlen = 20
	ngram_range = 1
	model = load_model(os.path.join(save_dir,'RNN_model.h5'))
	tokenizer = pickle.load(open('tokenizer','rb'))
	test_data = tokenizer.texts_to_sequences(test_data)
	print(len(test_data), 'test sequences')
	print('Average test sequence length: {}'.format(np.mean(list(map(len, test_data)), dtype=int)))

	if ngram_range > 1:
		token_indice = pickle.load(open('mydictionary','rb'))
		test_data = add_ngram(test_data, token_indice, ngram_range)
		print('Average test sequence length: {}'.format(np.mean(list(map(len, test_data)), dtype=int)))
		
	test_data = sequence.pad_sequences(test_data, maxlen=maxlen)
	print('x_test shape:', test_data.shape)
	
	answer = model.predict(test_data)
	result = []
	for i in range(len(answer)):
		if answer[i] >= 0.45:
			result.append(1)
		elif answer[i] < 0.45:
			result.append(0)
	print('=====Write output to %s =====' % output_path)
	with open(output_path, 'w') as f:
		f.write('id,label\n')
		for i, v in  enumerate(result):
			f.write('%d,%d\n' %(i, v))
	return 'infer'

def data_augmentation(train_data,train_label,train_undata,save_dir):
	maxlen = 20
	model = load_model(os.path.join(save_dir,'RNN_model.h5'))
	tokenizer = pickle.load(open('tokenizer','rb'))
	train_undata1 = tokenizer.texts_to_sequences(train_undata)
	train_undata1 = sequence.pad_sequences(train_undata1, maxlen=maxlen)
	answer = model.predict(train_undata1)
	augment_label = []
	augment_data = []
	for i in range(len(answer)):
		if answer[i]>=0.85:
			augment_label.append(1)
			augment_data.append(train_undata[i])
		elif answer[i]<=0.15:
			augment_label.append(0)
			augment_data.append(train_undata[i])
	augment_data,augment_label = _shuffle(augment_data,augment_label)
	augment_data = augment_data.tolist()
	augment_label = augment_label.tolist()
	train_data = train_data + augment_data[:len(augment_data)//2]
	train_label = train_label + augment_label[:len(augment_label)//2]
	return (train_data,train_label)
def main(opts):

	train_label_data_path = opts.train_label_data_path
	train_unlabel_data_path = opts.train_unlabel_data_path
	test_data_path = opts.test_data_path
	save_dir = opts.save_dir
	output_path = opts.output_path
	if opts.train:
		train_data,train_label = load_train_label_data(train_label_data_path)
		train_undata = load_train_unlabel_data(train_unlabel_data_path)
		max_features = check_Index_before_oov(train_data)
		print(max_features)
		train_data1 = preprocess_trainData(train_data,max_features)
		train_process = train(train_data1,train_label,save_dir,max_features)
		print('finish first train')
		
		print('begin self learning')
		train_data,train_label = data_augmentation(train_data,train_label,train_undata,save_dir)
		max_features = check_Index_before_oov(train_data)
		print(max_features)
		train_data1 = preprocess_trainData(train_data,max_features)
		train_process = train(train_data1,train_label,save_dir,max_features)
		print('finish self learning')
	elif opts.infer:
		test_data = load_test_data(test_data_path)
		infer_process = infer(test_data,save_dir,output_path)
	else:
		print('error: argument --train or --infer not found')

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='RNN classification')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--train',action='store_true',default=False,dest='train',help='Input --train to Train')
	group.add_argument('--infer',action='store_true',default=False,dest='infer',help='Input --infer to infer')
	parser.add_argument('--train_label_data_path',type=str,default='data/training_label.txt',dest='train_label_data_path',help='path to label training data')
	parser.add_argument('--train_unlabel_data_path',type=str,default='data/training_nolabel.txt',dest='train_unlabel_data_path',help='path to unlabel training data')
	parser.add_argument('--test_data_path',type=str,default='data/testing_data.txt',dest='test_data_path',help='path to testing data')
	parser.add_argument('--save_dir',type=str,default='RNN_params/',dest='save_dir',help='path to save the model parameters')
	parser.add_argument('--output_path',type=str,default='RNN_output/result.csv',dest='output_path',help='path to save the model outputs')
	
	opts = parser.parse_args()

	main(opts)