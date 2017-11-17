import matplotlib
matplotlib.use('Agg')
import numpy as np
import os, sys
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
#from utils import *
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
img_row = 48
img_col = 48

def add_mirroring(image_train,label_train):
	rotate = np.flip(image_train,axis=2)
	image_train = np.concatenate((image_train,rotate),axis=0)
	label_train = np.concatenate((label_train,label_train),axis = 0)
	return image_train,label_train

def add_augmentation(image_train,label_train,augmentation_number):
	datagen = ImageDataGenerator(rotation_range=10,
								width_shift_range=0.15,
								height_shift_range=0.15,
								#horizontal_flip=True,
								fill_mode='nearest')
	datagen.fit(image_train)
	for X_batch, Y_batch in datagen.flow(image_train, label_train, batch_size=augmentation_number):
		image_train = np.concatenate((image_train,X_batch),axis=0)
		label_train = np.concatenate((label_train,Y_batch),axis=0)
		break 
	return (image_train,label_train)
def augmentation(image_train,label_train):
	image_train,label_train = add_mirroring(image_train,label_train)
	length = len(image_train)
	for i in range(3):
		image_train,label_train = add_augmentation(image_train,label_train,length)
	return image_train,label_train
def load_train_data(train_data_path):

	label_data = []
	training_data = []
	
	
	train_set = pd.read_csv(train_data_path,sep=',',header=0)
	dataSize_train = train_set.index.size
	for i in range(dataSize_train):
		label_data.append(train_set['label'][i])
		training_data.append(train_set['feature'][i].split(' '))
	
	training_data = np.array(training_data)
	label_data = np.array(label_data)

	if K.image_data_format() == 'channels_first':
		training_data = training_data.reshape(training_data.shape[0],1,img_row,img_col)	
	elif K.image_data_format() == 'channels_last':
		training_data = training_data.reshape(training_data.shape[0],img_row,img_col,1)
	
	training_data = training_data.astype(np.float)
	label_data = label_data.astype(np.int)
	label_data = keras.utils.to_categorical(label_data,7)
	
	training_data = training_data/255
	
	
	return (training_data, label_data)

def load_test_data(test_data_path):
	testing_data = []

	test_set = pd.read_csv(test_data_path,sep=',',header=0)
	dataSize_test = test_set.index.size
	for i in range(dataSize_test):
		testing_data.append(test_set['feature'][i].split(' '))

	testing_data = np.array(testing_data)

	if K.image_data_format() == 'channels_first':
		testing_data = testing_data.reshape(testing_data.shape[0],1,img_row,img_col)	
	elif K.image_data_format() == 'channels_last':
		testing_data = testing_data.reshape(testing_data.shape[0],img_row,img_col,1)
	testing_data = testing_data.astype(np.float)
	testing_data = testing_data/255

	return testing_data

def _shuffle(X,Y):

	ran = np.arange(len(X))
	np.random.shuffle(ran)
	return (X[ran],Y[ran])
def draw_procedure(history,a,l):
	if l==1:
		#Plot the Loss Curves
		plt.plot(history.history['loss'],'r',linewidth=3.0)
		plt.plot(history.history['val_loss'],'b',linewidth=3.0)
		plt.legend(['Traing Loss','Validation Loss'],fontsize=18,loc='upper left')
		plt.xlabel('Epochs',fontsize=16)
		plt.ylabel('Loss',fontsize=16)
		plt.title('Loss Curves',fontsize=16)
		plt.savefig('CNNloss.png')
	if a==1:
		#Plot the Accuracy Curves
		plt.plot(history.history['acc'],'r',linewidth=3.0)
		plt.plot(history.history['val_acc'],'b',linewidth=3.0)
		plt.legend(['Traing Accuracy','Validation Accuracy'],fontsize=18,loc='upper left')
		plt.xlabel('Epochs',fontsize=16)
		plt.ylabel('Accuracy',fontsize=16)
		plt.title('Accuracy Curves',fontsize=16)
		plt.savefig('CNNaccuracy.png')
	return 'draw'

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
	"""
    This function prints and plots the confusion matrix.
    """
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	return 'confusionmatrix'
def draw_confusion_matrix(X_valid,Y_valid):
	model = load_model('CNN_params/CNN_model.h5')
	answer = model.predict(X_valid)
	y_pred = answer.argmax(axis=1)
	y_test = Y_valid.argmax(axis=1)
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	plt.figure()
	confusionmatrix = plot_confusion_matrix(cnf_matrix, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
	plt.savefig('confusionmatrix.png')
def deprocessimage(x):
    
    #Hint: Normalize and Clip
    
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

    # clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)
	return x
def draw_saliency_map(emotion_classifier,private_pixels):
	input_img = emotion_classifier.input
	img_ids = [9]
	for idx in img_ids:
		val_proba = emotion_classifier.predict(private_pixels[idx])
		pred = val_proba.argmax(axis=-1)
		target = K.mean(emotion_classifier.output[:, pred])
		grads = K.gradients(target, input_img)[0]
		fn = K.function([input_img, K.learning_phase()], [grads])

		imag = np.array([private_pixels[idx]])
		g = fn(image,0)
		heatmap = deprocessimage(g)

		thres = 0.5
		see = private_pixels[idx].reshape(48, 48)
        # for i in range(48):
            # for j in range(48):
                # print heatmap[i][j]
		see[np.where(heatmap <= thres)] = np.mean(see)

		plt.figure()
		plt.imshow(heatmap, cmap=plt.cm.jet)
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		test_dir = 'test_cmap'
		fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

		plt.figure()
		plt.imshow(see,cmap='gray')
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		test_dir ='test_partial'
		fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)
	
	return 'map'

def train(X_train,Y_train,X_valid,Y_valid,save_dir):
	if K.image_data_format() == 'channels_first':
		input_shape = (1,img_row,img_col)
	elif K.image_data_format() == 'channels_last':
		input_shape = (img_row,img_col,1)

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	save_model_path = os.path.join(save_dir, 'CNN_model.h5')
	batch_size = 128
	epochs = 8
	num_classes = 7

	model = Sequential()
	model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())

	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())

	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())

	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	
	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	

	model.add(Dropout(0.5))
	model.add(Flatten())

	model.add(Dense(4096,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(4096,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(num_classes,activation='softmax'))

	checkPoint = ModelCheckpoint(save_model_path,verbose=0,save_best_only=True,monitor='val_acc')
	callback_list = [checkPoint]
	model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.Adadelta(),
				metrics=['accuracy'])
	history = model.fit(X_train,Y_train,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
			validation_data=(X_valid,Y_valid),callbacks=callback_list)
	score = model.evaluate(X_valid,Y_valid,verbose=0)
	print('loss:',score[0])
	print('accuracy:',score[1])
	print('draw procedure')
	#draw = draw_procedure(history,1,0)
	#draw = draw_procedure(history,0,1)
	print('draw Confusion matrix')
	#draw = draw_confusion_matrix(X_valid,Y_valid)
	return 'train'

def infer(testing_data,save_dir,output_path):

	model = load_model(os.path.join(save_dir,'CNN_model.h5'))
	#draw = draw_saliency_map(model,testing_data)
	answer = model.predict(testing_data)
	result = answer.argmax(axis=1)

	print('=====Write output to %s =====' % output_path)
	with open(output_path, 'w') as f:
		f.write('id,label\n')
		for i, v in  enumerate(result):
			f.write('%d,%d\n' %(i, v))

	return 'infer'
	
def split_valid_set(X_all,Y_all,split_validation_percetage):
	X_all,Y_all = _shuffle(X_all,Y_all)

	total_data_num = len(X_all)
	valid_data_num = int(floor(total_data_num*split_validation_percetage))

	X_valid = X_all[:valid_data_num,:,:]
	Y_valid = Y_all[:valid_data_num,:]

	X_train = X_all[valid_data_num:,:,:]
	Y_train = Y_all[valid_data_num:,:]

	return (X_train,Y_train,X_valid,Y_valid)

def main(opts):
	#load data
	#(28709, 48, 48, 1)
	#(7178, 48, 48, 1)
	#(28709, 7)
	train_data_path = opts.train_data_path
	test_data_path = opts.test_data_path
	save_dir = opts.save_dir
	output_path = opts.output_path
	
	if opts.train :
		print('start training')
		training_data, label_data = load_train_data(train_data_path)
		X_train,Y_train,X_valid,Y_valid = split_valid_set(training_data,label_data,0.08)
		X_train,Y_train = augmentation(X_train,Y_train)
		X_valid,Y_valid = augmentation(X_valid,Y_valid)
		#X_train = np.concatenate((X_train,X_valid),axis = 0)
		#Y_train = np.concatenate((Y_train,Y_valid),axis = 0)
		print(X_train.shape)
		train_process = train(X_train,Y_train,X_valid,Y_valid,save_dir)

	elif opts.infer:
		print('start infering')
		testing_data = load_test_data(test_data_path)
		infer_process = infer(testing_data,save_dir,output_path)
	else:
		print('error: argument --train or --infer not found')

	

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='CNN classification')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--train',action='store_true',default=False,dest='train',help='Input --train to Train')
	group.add_argument('--infer',action='store_true',default=False,dest='infer',help='Input --infer to infer')
	parser.add_argument('--train_data_path',type=str,default='data/train.csv',dest='train_data_path',help='path to training data')
	parser.add_argument('--test_data_path',type=str,default='data/test.csv',dest='test_data_path',help='path to testing data')
	parser.add_argument('--save_dir',type=str,default='CNN_params/',dest='save_dir',help='path to save the model parameters')
	parser.add_argument('--output_path',type=str,default='CNN_output/result.csv',dest='output_path',help='path to save the model outputs')
	
	opts = parser.parse_args()

	main(opts)