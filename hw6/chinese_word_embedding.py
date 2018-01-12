# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import time,os,sys
from math import log, floor
import pickle
from random import shuffle
from adjustText import adjust_text
import argparse
import matplotlib.pyplot as plt
import jieba
from gensim.models import word2vec
from gensim import models
from collections import Counter
import matplotlib as mpl
from PIL import Image
from sklearn.manifold import TSNE
def train_word2vec():
	sentences = word2vec.Text8Corpus("chinese_data_seg.txt")
	model = word2vec.Word2Vec(sentences, size=256, window=6, min_count=3)
	model.save("med250.model.bin")
def visualization():
	model = models.Word2Vec.load('med250.model.bin')
	
	with open('chinese_data_seg.txt') as f:
		
		words = f.read().split()
		counting_maps = Counter(words)
		counting_maps = dict(counting_maps)
		common_counting_maps = dict((k, v)for k, v in counting_maps.items() if v >= 4500)
		common_words = list(common_counting_maps.keys())
	words_vector = []
	for w in (common_words):
		word_vector = model.wv[w]
		words_vector.append(word_vector)
	words_vector = np.array(words_vector)
		
	vis_data = TSNE(n_components=2).fit_transform(words_vector)
	vis_x = vis_data[:,0]
	vis_y = vis_data[:,1]
	
	plot(vis_x, vis_y, common_words)

def plot(Xs, Ys, Texts):
	#font_name = 'sans-serif'
	mpl.rcParams['font.sans-serif'] = ['SimHei', 'sans-serif']
	mpl.rcParams['axes.unicode_minus'] = False
	plt.plot(Xs, Ys, 'o')
	texts = [plt.text(X, Y, Text) for X, Y, Text in zip(Xs, Ys, Texts)]
	plt.title(str(adjust_text(texts, Xs, Ys, arrowprops=dict(arrowstyle='->', color='red'))))
	plt.savefig('emb.png')
	Image.open('emb.png').save('word_embedding.jpg','JPEG')
def main():
	jieba.set_dictionary('Chinese_data/dict.txt.big')
	output = open('chinese_data_seg.txt','w')
	texts_num = 0
	with open('data/all_sents.txt','r') as content :
		for line in content:
			words = jieba.cut(line, cut_all=False)
			for word in words:
				output.write(word +' ')
			texts_num += 1
			if texts_num % 100000 == 0:
				print("已完成前 %d 行的斷詞" % texts_num)
	output.close()
	train_word2vec()
	print('finish training')
	visualization()
if __name__ == '__main__':
	main()

