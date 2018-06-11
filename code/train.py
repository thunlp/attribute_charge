import numpy as np
import tensorflow as tf 
import global_config
import model
import random
from prec_recall_counter import PrecRecallCounter
from sys import argv
from init import *
from utils import *
import os
def ismixed(a, b, c):
	#mixed couple, pred, answer
	
	temp = False
	for i in c:
		temp = temp|(i == a[0])
	return temp&(b == a[1])&(b not in c)

def main(_):
	path = os.getcwd()
	father_path = os.path.dirname(path)
	checkpoint_dir = 	father_path + "/checkpoint/"
	lstm_log_dir = 		father_path + "/log/evaluation_charge_log/"
	attr_log_dir = 		father_path + "/log/evaluation_attr_log/"
	val_lstm_log_dir = 	father_path + "/log/validation_charge_log/"
	val_attr_log_dir = 	father_path + "/log/validation_attr_log/"
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(lstm_log_dir):
		os.makedirs(lstm_log_dir)
	if not os.path.exists(attr_log_dir):
		os.makedirs(attr_log_dir)
	if not os.path.exists(val_lstm_log_dir):
		os.makedirs(val_lstm_log_dir)
	if not os.path.exists(val_attr_log_dir):
		os.makedirs(val_attr_log_dir)
	restore  = False
	skiptrain = False
	bs = 32#batch size
	perstep = 500
	eva_number = 0
	val_number = 0
	#init
	print "loading word embedding and data..."
	word2id,word_embeddings,attr_table,x_train,y_train,y_attr_train,x_test,y_test,y_attr_test,x_val,y_val,y_attr_val,namehash,length_train,length_test,length_val = load_data_and_labels_fewshot()
	id2word = {}
	for i in word2id:
		id2word[word2id[i]] = i
	batches = batch_iter(list(zip(x_train, y_train, y_attr_train)), global_config.batch_size, global_config.num_epochs)
	lstm_config = model.lstm_Config()
	lstm_config.num_steps = len(x_train[0])
	lstm_config.hidden_size = len(word_embeddings[0])
	lstm_config.vocab_size = len(word_embeddings)
	lstm_config.num_classes = len(y_train[0])
	lstm_config.num_epochs = 20
	lstm_config.batch_size = bs

	with tf.Graph().as_default():
		tf.set_random_seed(6324)
		tf_config = tf.ConfigProto() 
		tf_config.gpu_options.allow_growth = True 
		sess = tf.Session(config=tf_config)
		print "initializing model" 
		with sess.as_default():
			lstm_initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("lstm_model", reuse=None, initializer = lstm_initializer):
				lstm_model = model.LSTM_MODEL(word_embeddings=word_embeddings,attr_table=attr_table,config = lstm_config)
				lstm_optimizer = tf.train.AdamOptimizer(lstm_config.lr)
				lstm_global_step = tf.Variable(0, name = "lstm_global_step", trainable = False)
				lstm_train_op = lstm_optimizer.minimize(lstm_model.total_loss,global_step = lstm_global_step)
			saver = tf.train.Saver()
			init_op = tf.initialize_all_variables()
			sess.run(init_op)
			best_macro_f1 = 0.0
			if restore:
				f_f1 = open(val_lstm_log_dir+'best_macro_f1','r')
				f1s = f_f1.readlines()
				best_macro_f1 = float(f1s[-1].strip().split(' ')[-1].strip('[').strip(']'))
				f_f1.close()
				ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
				if ckpt and ckpt.model_checkpoint_path:  
					saver.restore(sess, ckpt.model_checkpoint_path)  
				else:  
					pass
			print "initialized"

			batches = batch_iter(list(zip(x_train,y_train,y_attr_train,length_train)),lstm_config.batch_size, lstm_config.num_epochs)

			for batch in batches:
				x_batch, y_batch, y_attr_batch, length_batch = zip(*batch)
				step = lstm_train_step(lstm_train_op, lstm_global_step, lstm_model, sess, x_batch, y_batch, y_attr_batch, length_batch)
				if ((step % perstep) == 0) or (skiptrain):
					new_marco_f1 = evaluation(eva_number,lstm_model,sess,lstm_config,lstm_log_dir,attr_log_dir,y_test,x_test,y_attr_test,length_test)
					eva_number += 1
					#when model get the best performance on test set, validate it on the validation set
					if (new_marco_f1 > best_macro_f1) or skiptrain:
						if not skiptrain:
							saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)
						best_macro_f1 = new_marco_f1
						validation(val_number,lstm_model,sess,lstm_config,val_lstm_log_dir,val_attr_log_dir,y_val,x_val,y_attr_val,length_val)
						val_number += 1
if __name__ == "__main__":
	tf.app.run()








