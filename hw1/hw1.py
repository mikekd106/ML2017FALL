import pandas as pd
import numpy as np 
import random
import math
import sys
from numpy.linalg import inv
import csv


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
			train_y.append(training_data[9][m*oneMonthLength+t+9])

	train_x = np.array(train_x)
	train_y = np.array(train_y)

	return train_x.astype(np.float),train_y.astype(np.float)

def training(train_x,weight,train_y,repeat,l_r):
	print('go into the training function')
	total_gra = np.zeros(train_x.shape[1])
	train_x_T = np.transpose(train_x)
	print (train_x_T)
	for i in range(repeat):
		y = np.dot(train_x,weight)
		
		loss = y -train_y
		
		cost = np.sum(loss**2)/len(train_x)
		
		s_cost = math.sqrt(cost)
		print ('iteration = %d ,cost = %f' % (i,s_cost))

		gra = 2*np.dot(train_x_T,loss)
		total_gra = total_gra + gra**2
		
		weight = weight - (l_r/np.sqrt(total_gra))*gra

	return weight	

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
	learning_rate = 10
	repeat = 220000
	training_data = read_train_data()
	train_x,train_y = parse_data(training_data)
	
	# add square
	train_x = np.concatenate((train_x,train_x**2),axis=1)
	# add bias
	train_x = np.concatenate((train_x,np.ones((train_x.shape[0],1))), axis=1)
	
	#validation_x,validation_y,train_x,train_y = create_validation(train_x,train_y)
	# init weight
	weight = np.zeros(train_x.shape[1])

	# train weight
	weight = training(train_x,weight,train_y,repeat,learning_rate)
	

	#test_accuracy(validation_x,validation_y,weight)

	testing_data = read_test_data()
	# add square
	testing_data = np.concatenate((testing_data,testing_data**2),axis=1)
	# add bias
	testing_data = np.concatenate((testing_data,np.ones((testing_data.shape[0],1))), axis=1)

	write_result(testing_data,weight)

	np.save('modelFinal_2.npy',weight)

if __name__ == '__main__':
	main()