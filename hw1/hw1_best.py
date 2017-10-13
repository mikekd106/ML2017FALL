import pandas as pd
import numpy as np 
import random
import math
import sys
from numpy.linalg import inv
import csv

import tensorflow as tf

pollution_set = (2,5,7,8,9,11,12,16)

def read_train_data():

	print('go into the read_data function')

	data = []
	for i in range(18):
		data.append([])

	dataFrame = pd.read_csv('train.csv',encoding = "ISO-8859-1")
	totaldays = dataFrame.index.size
	for i in range(totaldays):
		for j in range(24):
			oneFeatureValue = dataFrame[str(j)][i]
			if oneFeatureValue == 'NR':
				oneFeatureValue = 0
			elif float(oneFeatureValue) <0:
				oneFeatureValue = 0
			data[i%18].append(oneFeatureValue)


	return data
def read_test_data():
	print('go into the read_test function')

	data = []
	dataFrame = pd.read_csv('test.csv',header = None)
	indexSizes = dataFrame.index.size
	featureCounts = int(indexSizes/18) 
	
	for i in range(featureCounts):
		data.append([])
		for h in range(9):
			#if h == 0:
			#	continue
			for p in range(18):
				if p in pollution_set:
					oneValue = dataFrame[h+2][18*i+p]
					if oneValue == 'NR':
						oneValue = 0
					elif float(oneValue) < 0:
						oneValue = 0
					data[i].append(oneValue)

	data = np.array(data)
	return data.astype(np.float)

def parse_data(training_data):
	print('go into the parse_data function')	
	train_x = []
	train_y = []
	oneMonthLength = 480

	# 12 months
	for m in range(12):
		# 471 vestors in a month  
		for t in range(471):
			temp_feature = [] 
			# 9 hours
			for h in range(9):
				#if h == 0:
				#	continue
				# 18 pollutions
				for p in range(18):
					if p in pollution_set:  
						temp_feature.append(training_data[p][m*oneMonthLength+t+h])

			train_x.append(temp_feature)
			train_y.append([training_data[9][m*oneMonthLength+t+9]])

	train_x = np.array(train_x)
	train_y = np.array(train_y)

	return train_x.astype(np.float),train_y.astype(np.float)


def write_result(testing_data,weight):
	
	ans = []

	dataTotal = testing_data.shape[0]

	for i in range(dataTotal):
		temp_test = testing_data[i]
		value = np.dot(temp_test,weight)
		ans.append(["id_"+str(i)])
		ans[i].append(int(np.around(value)))

	filename = 'predict.csv'
	text = open(filename,'w+')
	s= csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
		s.writerow(ans[i])
	text.close()

def create_validation(train_x,train_y):

	validation_size = train_x.shape[0]//25

	validation_x = train_x[:validation_size,:]
	validation_y = train_y[:validation_size]

	train_x = train_x[validation_size:,:]
	train_y = train_y[validation_size:]

	return validation_x,validation_y,train_x,train_y

def test_accuracy(validation_x,validation_y,weight):
	ans = []

	dataTotal = validation_x.shape[0]

	for i in range(dataTotal):
		temp_test = validation_x[i]
		value = np.dot(temp_test,weight)
		ans.append(np.around(value))

	s = 0

	for i in range(len(ans)):
		if ans[i] == validation_y[i]:
			s = s + 1
	
	print ('accuracy = %f'% (s/len(ans)*100))



def main():

	training_data = read_train_data()
	train_x,train_y = parse_data(training_data)
	
	# add square
	train_x = np.concatenate((train_x,train_x**2),axis=1)
	#parameter
	learning_rate = 0.001
	num_steps = 30000
	batch_size = 128
	display_step = 100

	# Network Parameters
	n_hidden_1 = 64 # 1st layer number of neurons
	n_hidden_2 = 64 # 2nd layer number of neurons
	num_input = len(pollution_set)*9*2 # input dimension

	# tf Graph input
	X = tf.placeholder('float',[None,num_input])
	Y = tf.placeholder('float',[None,1])

	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([num_input,n_hidden_1],stddev=0.001)),
		'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=0.001)),
		'out': tf.Variable(tf.random_normal([n_hidden_2,1],stddev=0.001))
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([1]))
	}

	
	# Create model
	def neural_net(x):
		# Hidden fully connected layer with 256 neurons
		layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
		layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
		out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
		return out_layer
	#Construct model
	regress = neural_net(X)
	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.square(regress-Y))

	optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Initialize the variables
	init = tf.global_variables_initializer()

	#Start training
	with tf.Session() as sess:
		# Run the initializer
		sess.run(init)
		for step in range(1,num_steps+1):
			for i in range((train_x.shape[0]//batch_size)+1):
				batch_x = train_x[batch_size*i:batch_size*(i+1),:]
				batch_y = train_y[batch_size*i:batch_size*(i+1),:]
				sess.run(train_op,feed_dict={X:batch_x,Y:batch_y})
			if step % display_step == 0 or step == 1:
				batch_x = train_x#[:batch_size*1,:]
				batch_y = train_y#[:batch_size*1,:]
				loss = sess.run(loss_op,feed_dict={X:batch_x,Y:batch_y})
				print ('iteration = %d ,cost = %f' % (step,math.sqrt(loss)))


		print("Optimization Finished")
		testing_data = read_test_data()
		# add square
		testing_data = np.concatenate((testing_data,testing_data**2),axis=1)
		
		# write result
		ans = []

		result = sess.run(regress,feed_dict={X:testing_data})
		for i in range(len(result)):
			value = result[i][0]
			ans.append(["id_"+str(i)])
			ans[i].append(int(np.around(value)))

		filename = 'predict.csv'
		text = open(filename,'w+')
		s= csv.writer(text,delimiter=',',lineterminator='\n')
		s.writerow(["id","value"])
		for i in range(len(ans)):
			s.writerow(ans[i])
		text.close()

	

if __name__ == '__main__':
	main()