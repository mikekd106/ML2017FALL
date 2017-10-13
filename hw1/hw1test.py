import sys
import pandas as pd
import numpy as np 
import random
import math
from numpy.linalg import inv
import csv

pollution_set = (2,5,7,8,9,12)


def read_test_data(input_file):
	print('go into the read_test function')

	data = []
	dataFrame = pd.read_csv(input_file,header = None)
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

def write_result(testing_data,weight,out_file):
	
	ans = []

	dataTotal = testing_data.shape[0]

	for i in range(dataTotal):
		temp_test = testing_data[i]
		value = np.dot(temp_test,weight)
		ans.append(["id_"+str(i)])
		ans[i].append(int(np.around(value)))

	text = open(out_file,'w+')
	s= csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
		s.writerow(ans[i])
	text.close()

def main():
	input_file = sys.argv[1]
	out_file = sys.argv[2]


	weight = np.load('modelFinal_1.npy')
	testing_data = read_test_data(input_file)

	# add square
	testing_data = np.concatenate((testing_data,testing_data**2),axis=1)
	# add bias
	testing_data = np.concatenate((testing_data,np.ones((testing_data.shape[0],1))), axis=1)


	write_result(testing_data,weight,out_file)



if __name__ == '__main__':
	main()