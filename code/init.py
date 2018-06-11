import numpy as np
import os
import time
import datetime
import random
import global_config
def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = (int)(round(len(data)/batch_size)) 
	for epoch in range(num_epochs):
 		# Shuffle the data at each epochnn
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

def load_data_and_labels_fewshot():
	#69 71
	path = os.getcwd()
	data_path = os.path.dirname(path)+"/data/"
	vec_path = data_path + "words.vec"
	train_path = 	data_path + "train"
	test_path = 	data_path + "test"
	val_path = 		data_path + "valid"
	law_attribution_path = data_path + "attributes"

	num2attr = global_config.num2attr
	count_of_attr = len(num2attr)
	word_embeddings = []
	word2id = {}
	f = open(vec_path, "r")
	content = f.readline()
	while True:
		content = f.readline()
		if content == "":
			break
		content = content.strip().split()
		word2id[content[0]] = len(word2id)
		content = content[1:]
		content = [(float)(i) for i in content]
		word_embeddings.append(content)
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	lists = [0.0 for i in range(len(word_embeddings[0]))]
	word_embeddings.append(lists)
	word_embeddings.append(lists)
	word_embeddings = np.array(word_embeddings, dtype=np.float32)
	print 'word_embedding loaded, size of dict is:',len(word_embeddings)

	relationhash = {}
	namehash = {}
	num_r = {}
	farticlemap = open(law_attribution_path,'r')
	maps = farticlemap.readlines()
	attr_table = [[2 for j in range(count_of_attr)] for i in range(len(maps))]
	for m in maps:
		m = m.strip().strip('\t').split('\t')
		relationhash[m[2]] = int(m[0])
		namehash[int(m[0])] = m[2]
		num_r[int(m[0])] = 0
		temp = [int(m[i]) for i in range(4,14)]
		attr_table[int(m[0])] = temp
	r_list = []
	attr_table = np.array(attr_table)
	print "attribute table loaded."

	def load_data(path):
		xs = []
		ys = []
		y_attrs = []
		f = open(path, "r")
		content = f.readlines()
		random.shuffle(content)
		for i in content:
			z = i.strip().split("\t")
			if(len(z) != 3):
				continue			
			tmp = []
			for j in z[1].strip().split():
				tmp.append(int(j))
			y_attr = []
			for j in z[2].strip().split():
				y_attr.append(int(j))
			if len(y_attr) != global_config.num_of_attr:
				print 'false attr num'
				continue
			xs.append(z[0])
			ys.append(tmp)
			y_attrs.append(y_attr)
		f.close()
		return xs,ys,y_attrs

	x_train,y_train,y_attr_train = load_data(train_path)
	print 'training data loaded, num of training data:',len(x_train)
	x_test,y_test,y_attr_test = load_data(test_path)
	print 'test data loaded, num of test data:',len(x_test)
	x_val,y_val,y_attr_val = load_data(val_path)
	print 'validation data loaded, num of validation data:',len(x_val)

	def process_data(xs,ys):
		res = []
		for i in xrange(0, len(ys)):
			label = [0 for k in range(0, len(relationhash))]
			for j in ys[i]:
				label[j] = 1
			res.append(label)
		res = np.array(res)
		lengths= []
		max_document_length = global_config.DOC_LEN
		size = len(xs)
		for i in xrange(size):
			blank = word2id['BLANK']
			text = [blank for j in xrange(max_document_length)]
			content = xs[i].split()
			for j in xrange(len(content)):
				if(j == max_document_length):
					break
				if not content[j] in word2id:
					text[j] = word2id['UNK']
				else:
					text[j] = word2id[content[j]]
			lengths.append(min(max_document_length,len(content)))
			xs[i] = text
		xs = np.array(xs)
		lengths = np.array(lengths)
		return xs,res,lengths

	x_train,y_train,length_train = process_data(x_train,y_train)
	x_test,y_test,length_test = process_data(x_test,y_test)
	x_val,y_val,length_val = process_data(x_val,y_val)
	print 'all data processed'

	return word2id,word_embeddings,attr_table,x_train,y_train,y_attr_train,x_test,y_test,y_attr_test,x_val,y_val,y_attr_val,namehash,length_train,length_test,length_val
