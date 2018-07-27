#coding:utf-8
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import global_config


class LSTM_MODEL(object):

	def __init__(self, word_embeddings, attr_table, config):

		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.num_attrs = num_attrs = config.num_attrs
		self.size = size =  config.hidden_size
		self.vocab_size = vocab_size = config.vocab_size
		self.num_classes = num_classes = config.num_classes
		self.ave_ratio = ave_ratio = config.ave_ratio
		hits_k = config.hits_k

		self.Y = Y = global_config.value2num['Y']
		self.N = N = global_config.value2num['N']
		self.NA = NA = global_config.value2num['NA']
		self.input_x = tf.placeholder(tf.int32, [batch_size, num_steps])
		self.input_length = tf.placeholder(tf.int32, [batch_size])
		self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes])
		self.unmapped_input_attr = tf.placeholder(tf.int32, [batch_size, num_attrs])
		self.keep_prob = tf.placeholder(tf.float32)

		self.attr_table = tf.Variable(attr_table, trainable = False)
		self.input_attr_mask = 1-tf.cast(tf.equal(NA, self.unmapped_input_attr),tf.int32)
		self.input_attr = tf.multiply(self.unmapped_input_attr, self.input_attr_mask)
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias = 0.0)

		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

		self.initial_state = cell.zero_state(batch_size, tf.float32)

		with tf.device("/cpu:0"), tf.name_scope("lstm_embedding"):
			embedding = tf.Variable(word_embeddings, trainable = False)
			inputs = tf.nn.embedding_lookup(embedding, self.input_x)

		
		outputs,_ = tf.nn.dynamic_rnn(cell,inputs,initial_state = self.initial_state,sequence_length=self.input_length)

		self.attention(outputs, self.size)
		self.attn_weights = tf.concat(1, [tf.expand_dims(temp,1) for temp in self.attention_weights])
		self.attr_preds = tf.concat(1, [tf.expand_dims(temp,1) for temp in self.attn_attr_preds])
		self.attr_loss = tf.concat(1, [tf.expand_dims(temp,1) for temp in self.attn_attr_loss])


		output = tf.expand_dims(tf.reshape(tf.concat(1,values= outputs), [batch_size, -1, size]), -1)

		with tf.name_scope("lstm_maxpool"):
			output_pooling = tf.nn.max_pool(output,
					ksize=[1, num_steps, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
			self.output = tf.reshape(output_pooling, [-1, size])
			
		with tf.name_scope("lstm_output"):
			softmax_w = tf.get_variable("softmax_w", [size+size, num_classes])
			softmax_b = tf.get_variable("softmax_b", [num_classes])
			
			ave_attention_outputs = tf.reduce_mean(tf.concat(1, [tf.expand_dims(temp,1) for temp in self.attention_outputs]),axis=1)
			ave_attention_outputs = ave_attention_outputs*self.ave_ratio
			temp = [self.output]
			temp.append(tf.reshape(ave_attention_outputs, [batch_size, size]))
			self.concat_output = tf.concat(1, temp)
			self.scores = tf.nn.xw_plus_b(self.concat_output, softmax_w, softmax_b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores,labels = self.input_y)
			self.lstm_loss = tf.reduce_mean(losses)
			self.total_attr_loss = tf.reduce_mean(self.attr_loss)
			self.total_loss = self.lstm_loss + config.attr_loss_ratio*self.total_attr_loss

		self.ans = tf.argmax(self.input_y, 1)

		#accuracy
		with tf.name_scope("lstm_accuracy"):
			self.lstm_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions,self.ans),tf.float32))
		
		tops = tf.nn.top_k(self.scores, k=config.top_num)
		self.top_cands = tops.indices
		self.top_cands_values = tops.values
	def attention(self,inputs,attention_size):
		self.attention_outputs = []
		self.attention_weights = []
		self.attn_attr_preds = []
		self.attn_attr_loss = []
		input_projection = tf.contrib.layers.fully_connected(inputs,attention_size,activation_fn=tf.tanh)
		for temp in range(self.num_attrs):
			with tf.name_scope("attr_attention"+str(temp)):
				attention_vector = tf.get_variable(name='attention_vector'+str(temp),
													shape=[attention_size],
													dtype=tf.float32)
				vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_vector), axis=2, keep_dims=True)
				attention_weight = tf.nn.softmax(vector_attn, dim=1)

				self.attention_weights.append(tf.reshape(attention_weight,[tf.cast(attention_weight.get_shape()[0],tf.int32),tf.cast(attention_weight.get_shape()[1],tf.int32)]))
				weighted_output = tf.multiply(inputs,attention_weight)
				attention_output = tf.reduce_sum(weighted_output,axis=1)
				self.attention_outputs.append(attention_output)
				#score
				attr_pred_w = tf.get_variable(name='attn_pred_w'+str(temp),shape=[self.size,2])
				attr_pred_b = tf.get_variable(name='attn_pred_b'+str(temp),shape=[2])
				attr_pred_score = tf.nn.xw_plus_b(attention_output, attr_pred_w, attr_pred_b)
				#pred
				attr_pred = tf.argmax(attr_pred_score,1)
				self.attn_attr_preds.append(attr_pred)
				#loss
				attr_index = tf.slice(self.input_attr, [0,temp], [self.batch_size,1])
				attr_label = tf.reshape(tf.one_hot(attr_index,2,dtype=tf.float32),[self.batch_size,2])
				unmasked_attr_loss = tf.nn.softmax_cross_entropy_with_logits(logits = attr_pred_score,labels = attr_label)
				attr_mask = tf.cast(tf.slice(self.input_attr_mask, [0,temp], [self.batch_size,1]), tf.float32)
				attr_mask = tf.reshape(attr_mask,[self.batch_size])
				attr_loss = tf.multiply(unmasked_attr_loss,attr_mask)
				self.attn_attr_loss.append(attr_loss)

class lstm_Config(object):

	def __init__(self):
		self.num_layers = 2
		self.batch_size = 32
		self.keep_prob = 0.5
		self.num_epochs = 20                                                                                          
		self.num_steps = 50
		self.hidden_size = 50
		self.vocab_size = 10000
		self.num_classes = 550
		self.hits_k = [1]
		self.top_num = 20
		self.lr = 1e-3
		self.num_attrs = 10
		self.lookup_ratio = 0.5
		self.attr_loss_ratio = 1.0
		self.ave_ratio = 10