import matplotlib
matplotlib.use('Agg')
from skimage import io
from skimage import transform
from numpy.linalg import svd
import numpy as np 
import time,os,sys
from math import log, floor
import pickle
from random import shuffle
import argparse
import matplotlib.pyplot as plt
from PIL import Image
def reconstruction(U,k,img,train_data_path,target_image):
	print(target_image)
	recon_des = io.imread(os.path.join(train_data_path, target_image))
	recon_des = recon_des.flatten()
	recon_des = recon_des.astype(np.float)
	recon_des = recon_des / 255
	weight = []
	for i in range(k):
		eigenfaces = U[:,i]
		w = np.dot(recon_des,eigenfaces)
		weight.append(w)
	recon_img = np.zeros([len(recon_des),])
	for i in range(len(weight)):
		eigenfaces = U[:,i]
		temp = weight[i]*eigenfaces
		recon_img = recon_img + temp
	img_mean = np.mean(img, axis=1)
	recon_img = recon_img + img_mean

	return recon_img
def draw_reconstruct_image(recon_img):
	recon_img -= np.min(recon_img)
	recon_img /= np.max(recon_img)
	recon_img = (recon_img * 255).astype(np.uint8)
	recon_img = recon_img.reshape(600,600,3)
	plt.axis('off')
	plt.imshow(recon_img)
	plt.savefig('temp.png')
	Image.open('temp.png').save('reconstruction.jpg','JPEG')
	return 'draw' 
def main(opts):
	train_data_path = opts.train_data_path
	target_image = opts.target_image
	save_dir = opts.save_dir
	output_path = opts.output_path
	k = 4
	dirs = os.listdir(train_data_path)
	
	for i,f in enumerate(dirs):
		if i == 0:
			img = io.imread(os.path.join(train_data_path, f))
			img = img.flatten()
			img = img.reshape(len(img),1)
			continue
		temp_img = io.imread(os.path.join(train_data_path, f))
		temp_img = temp_img.flatten()
		temp_img = temp_img.reshape(len(temp_img),1)
		img = np.concatenate((img, temp_img), axis=1)
	img = img.astype(np.float)
	img = img / 255
	print(img.shape)
	img_mean = np.mean(img, axis=1)
	img_mean = img_mean.reshape(len(img_mean),1)
	print(img_mean.shape)
	U, s, V = svd(img - img_mean, full_matrices=False)
	#draw_reconstruct_image(U[:,3])
	
	recon_img = reconstruction(U,k,img,train_data_path,target_image)
	draw_reconstruct_image(recon_img)
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='image clustering dask')
	parser.add_argument('--train_data_path',type=str,default='Aberdeen/',dest='train_data_path',help='path to training data')
	parser.add_argument('--target_image',type=str,default='414.jpg',dest='target_image',help='path to testing data')
	parser.add_argument('--save_dir',type=str,default='models/',dest='save_dir',help='path to save the model parameters')
	parser.add_argument('--output_path',type=str,default='Result_output/result.csv',dest='output_path',help='path to save the model outputs')
	
	opts = parser.parse_args()

	main(opts)