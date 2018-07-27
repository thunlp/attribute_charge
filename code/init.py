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
	path = os.getcwd()
	data_path = os.path.dirname(path)+"/data/"
	vec_path = data_path + "words.vec"
	train_path = data_path + "train"
	test_path = data_path + "test"
	val_path = 	data_path + "valid"
	law_attribution_path = data_path + "attributes"
	
	num2attr = global_config.num2attr
	count_of_attr = len(num2attr)
	print 'init 1'
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
	print 'word_embedding init',len(word_embeddings)

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
	
	print 'init 2'
	x_train = []
	y_train = []
	y_attr_train = []
	nn = 0

	f = open(train_path, "r")
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
		x_train.append(z[0])
		y_train.append(tmp)
		y_attr_train.append(y_attr)
	f.close()
	print len(x_train)
	print 'init 3'

	x_test = []
	y_test = []
	y_attr_test = []
	la = 0
	f = open(test_path, "r")
	content = f.readlines()
	for i in content:
		z = i.strip().split("\t")
		if(len(z) != 3):
			continue
		tmp = []
		for j in z[1].strip().split():
			tmp.append(int(j))
			num_r[int(j)] += 1

		y_attr = []
		for j in z[2].strip().split():
			y_attr.append(int(j))
		if len(y_attr) != global_config.num_of_attr:
			print 'false attr num'
			continue
		x_test.append(z[0])
		y_test.append(tmp)
		y_attr_test.append(y_attr)
	f.close()
	print len(x_test)
	print 'init 4'

	x_val = []
	y_val = []
	y_attr_val = []
	la = 0
	f = open(val_path, "r")
	content = f.readlines()
	for i in content:
		z = i.strip().split("\t")
		if(len(z) != 3):
			continue
		tmp = []
		for j in z[1].strip().split():
			tmp.append(int(j))
			num_r[int(j)] += 1

		y_attr = []
		for j in z[2].strip().split():
			y_attr.append(int(j))
		if len(y_attr) != global_config.num_of_attr:
			print 'false attr num'
			continue
		x_val.append(z[0])
		y_val.append(tmp)
		y_attr_val.append(y_attr)

	f.close()
	print len(x_val)
	print 'init 5'

	print 'relationhash',len(relationhash)

	res = []
	yz = []
	for i in xrange(0, len(y_test)):
		label = [0 for k in range(0, len(relationhash))]
		for j in y_test[i]:
			label[j] = 1
		res.append(label)
		yz.append(y_test[i])
	y_test = np.array(res)

	res = []
	for i in xrange(0, len(y_train)):
		label = [0 for k in range(0, len(relationhash))]
		for j in y_train[i]:
			label[j] = 1
		res.append(label)
	y_train = np.array(res)

	res = []
	for i in xrange(0, len(y_val)):
		label = [0 for k in range(0, len(relationhash))]
		for j in y_val[i]:
			label[j] = 1
		res.append(label)
	y_val = np.array(res)
	print 'init 6'

	max_document_length = 500
	size = len(x_train)
	size0 = 0
	size1 = 0
	length_train = []
	length_test = []
	length_val = []

	for i in xrange(size):
		blank = word2id['BLANK']
		text = [blank for j in xrange(max_document_length)]
		content = x_train[i].split()
		for j in xrange(len(content)):
			if(j == max_document_length):
				break
			if not content[j] in word2id:
				text[j] = word2id['UNK']
			else:
				text[j] = word2id[content[j]]
		length_train.append(min(max_document_length,len(content)))
		x_train[i] = text

	x_train = np.array(x_train)
	length_train = np.array(length_train)

	size = len(x_test)
	for i in xrange(size):
		blank = word2id['BLANK']
		text = [blank for j in xrange(max_document_length)]
		content = x_test[i].split()
		for j in xrange(len(content)):
			if(j == max_document_length):
				break
			if not content[j] in word2id:
				text[j] = word2id['UNK']
			else:
				text[j] = word2id[content[j]]
		length_test.append(min(max_document_length,len(content)))
		x_test[i] = text
	x_test = np.array(x_test)
	length_test = np.array(length_test)

	size = len(x_val)
	for i in xrange(size):
		blank = word2id['BLANK']
		text = [blank for j in xrange(max_document_length)]
		content = x_val[i].split()
		for j in xrange(len(content)):
			if(j == max_document_length):
				break
			if not content[j] in word2id:
				text[j] = word2id['UNK']
			else:
				text[j] = word2id[content[j]]
		length_val.append(min(max_document_length,len(content)))
		x_val[i] = text
	x_val = np.array(x_val)
	length_val = np.array(length_val)
	print 'init finish'
	return word2id,word_embeddings,attr_table,x_train,y_train,y_attr_train,x_test,y_test,y_attr_test,x_val,y_val,y_attr_val,namehash,length_train,length_test,length_val
