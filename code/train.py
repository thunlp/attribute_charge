import numpy as np
import tensorflow as tf 
import global_config
import model
import random
from prec_recall_counter import PrecRecallCounter
from sys import argv
from init import *
import os
def ismixed(a, b, c):
	#mixed couple, pred, answer
	
	temp = False
	for i in c:
		temp = temp|(i == a[0])
	return temp&(b == a[1])&(b not in c)
def create_dir(ds):
	for temp in ds:
		if not os.path.exists(temp):
			os.makedirs(temp)
def main(_):

	path = os.getcwd()
	father_path = os.path.dirname(path)
	checkpoint_dir = 	father_path + "/checkpoint/"
	lstm_log_dir = 		father_path + "/log/evaluation_charge_log/"
	attr_log_dir = 		father_path + "/log/evaluation_attr_log/"
	val_lstm_log_dir = 	father_path + "/log/validation_charge_log/"
	val_attr_log_dir = 	father_path + "/log/validation_attr_log/"

	create_dir([checkpoint_dir,lstm_log_dir,attr_log_dir,val_lstm_log_dir,val_attr_log_dir])
	restore  = False
	skiptrain = False
	valmatrix = False
	val_case = False
	mixandmatrix = False
	single_attr_log = False
	bs = 32
	perstep = 500
	eva_number = 0
	val_number = 0
	mixcouple = [69,71]
	mixattr = [2,3,7]
	single_attr = [4,9]
	
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
	lstm_config.num_epochs = 20

	lstm_eval_config = model.lstm_Config()
	lstm_eval_config.keep_prob = 1.0
	lstm_eval_config.num_steps = len(x_train[0])
	lstm_eval_config.hidden_size = len(word_embeddings[0])
	lstm_eval_config.vocab_size = len(word_embeddings)
	lstm_eval_config.num_classes = len(y_train[0])
	lstm_eval_config.batch_size = bs
	lstm_eval_config.num_epochs = 20

	zero_x = [0 for i in range(lstm_config.num_steps)]
	zero_y = [0 for i in range(lstm_config.num_classes)]

	lstm_count_tab = np.array([[0.0 for i in range(lstm_config.num_classes)]for j in range(lstm_config.num_classes)])
	total_tab = np.array([0.0 for i in range(lstm_config.num_classes)])	
	with tf.Graph().as_default():
		tf.set_random_seed(6324)
		tf_config = tf.ConfigProto() 
		tf_config.gpu_options.allow_growth = True 
		sess = tf.Session(config=tf_config) 
		with sess.as_default():
			lstm_initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("lstm_model", reuse=None, initializer = lstm_initializer):
				print 'lstm step1'
				lstm_model = model.LSTM_MODEL(word_embeddings=word_embeddings,attr_table=attr_table,config = lstm_config)
				print 'lstm step2'
				lstm_optimizer = tf.train.AdamOptimizer(lstm_config.lr)
				print 'lstm step3'
				lstm_global_step = tf.Variable(0, name = "lstm_global_step", trainable = False)
				lstm_train_op = lstm_optimizer.minimize(lstm_model.total_loss,global_step = lstm_global_step)
				print 'lstm step4'
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

			def lstm_train_step(x_batch, y_batch, y_attr_batch, length_batch):
				"""
				A single training step
				"""
				feed_dict = {
					lstm_model.input_x: x_batch,
					lstm_model.input_length: length_batch,
					lstm_model.input_y: y_batch,
					lstm_model.unmapped_input_attr: y_attr_batch,
					lstm_model.keep_prob: 0.5,
				}
				_, step, total_loss, lstm_loss, attr_loss = sess.run(
					[lstm_train_op, lstm_global_step, lstm_model.total_loss, lstm_model.lstm_loss, lstm_model.total_attr_loss], feed_dict)
				time_str = datetime.datetime.now().isoformat()
				if step % 50 == 0:
					#print sc
	 				print("{}: step {}, total loss {:g}, lstm_loss {:g}, attr_loss {:g}".format(time_str, step, total_loss,
	 					lstm_loss, attr_loss))
				return step
			def lstm_dev_step(x_batch, y_batch, y_attr_batch, length_batch, writer=None):
				"""
				Evaluates model on a dev set
				"""
				feed_dict = {
					lstm_model.input_x: x_batch,
					lstm_model.input_length: length_batch,
					lstm_model.input_y: y_batch,
					lstm_model.unmapped_input_attr: y_attr_batch,
					lstm_model.keep_prob: 1.0,
				}
				runlist = [lstm_model.predictions,lstm_model.attr_preds,lstm_model.total_loss,lstm_model.lstm_loss,lstm_model.total_attr_loss,lstm_model.attn_weights]
				
				lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = sess.run(runlist, feed_dict=feed_dict)

				return lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights

			batches = batch_iter(list(zip(x_train,y_train,y_attr_train,length_train)),lstm_config.batch_size, lstm_config.num_epochs)

			for batch in batches:
				x_batch, y_batch, y_attr_batch, length_batch = zip(*batch)
				step = lstm_train_step(x_batch, y_batch, y_attr_batch, length_batch)
				if ((step % perstep) == 0) or (skiptrain):

					print 'Evaluation'
					if mixandmatrix:
						f_mix = open(lstm_log_dir+str(eva_number)+'mixed.html','w')
						f_mix.write('<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"/></head>\n')
					if single_attr_log:
						f_single_attr = open(lstm_log_dir+str(eva_number)+'attr.html','w')
						f_single_attr.write('<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"/></head>\n')

					all_count = 0.0
					total_losses,lstm_losses,attr_losses = 0.0,0.0,0.0
					lstm_prc = PrecRecallCounter(lstm_config.num_classes,lstm_log_dir,'lstm',eva_number)
					attr_prc = PrecRecallCounter([2 for temp in range(global_config.num_of_attr)],attr_log_dir,'attr',eva_number)
					lstm_matrix = [[0 for j in range(lstm_config.num_classes)]for i in range(lstm_config.num_classes)]
					num = int(len(y_test)/float(lstm_eval_config.batch_size))
					print num
					for i in range(num):
						if i %100 == 0:
							print i
						begin = i * lstm_eval_config.batch_size
						end = (i+1) * lstm_eval_config.batch_size
						y_batch_t = y_test[begin:end]
						x_batch_t = x_test[begin:end]
						y_attr_batch_t = y_attr_test[begin:end]
						length_batch = length_test[begin:end]

						lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = lstm_dev_step(x_batch_t,y_batch_t,y_attr_batch_t,length_batch)
						total_losses+=t_loss
						lstm_losses+=l_loss
						attr_losses+=a_loss
						for j in range(lstm_eval_config.batch_size):
							indexes = np.flatnonzero(y_batch_t[j])
							lstm_prc.multicount(lstm_p[j], indexes)
							for index in indexes:
								lstm_matrix[index][lstm_p[j]] += 1
							for k in range(global_config.num_of_attr):
								attr_prc.count(attr_p[j][k], y_attr_batch_t[j][k], k)
							if mixandmatrix:
								mixed = ismixed(mixcouple, lstm_p[j], indexes)
								if mixed:
									wordcolor = '<font style="background: rgba(255, 255, 0, %f)">%s</font>\n'
									f_mix.write('<p>'+str(lstm_p[j])+' '+str(indexes)+'</p>\n')
									towrite = ''
									for k in range(global_config.num_of_attr):
										towrite = towrite + str(attr_p[j][k]) + ' '
									f_mix.write('<p>'+towrite+'</p>\n')
									towrite = ''
									for k in range(global_config.num_of_attr):
										towrite = towrite + str(y_attr_batch_t[j][k]) + ' '
									f_mix.write('<p>'+towrite+'</p>\n')
									for c in mixattr:
										f_mix.write(wordcolor%(0,str(c)))
										for w in range(len(x_batch_t[j])):
											if w == length_batch[j]:
												break
											f_mix.write(wordcolor%(attn_weights[j][c][w]/np.max(attn_weights[j][c]),id2word[x_batch_t[j][w]]))
										f_mix.write('<p>---</p>\n')
							if single_attr_log:
								for attr_index in single_attr:
									if (attr_p[j][attr_index] != y_attr_batch_t[j][attr_index])&(y_attr_batch_t[j][attr_index]!=2):
										wordcolor = '<font style="background: rgba(255, 255, 0, %f)">%s</font>\n'
										f_single_attr.write('<p>'+str(indexes)+str(attr_index)+' '+str(attr_p[j][attr_index])+' '+str(y_attr_batch_t[j][attr_index])+'</p>\n')
										for w in range(len(x_batch_t[j])):
											if w == length_batch[j]:
												break
											f_single_attr.write(wordcolor%(attn_weights[j][attr_index][w]/np.max(attn_weights[j][attr_index]),id2word[x_batch_t[j][w]]))
										f_single_attr.write('<p>---</p>\n')

					begin = num * lstm_eval_config.batch_size
					y_batch_t = y_test[begin:]
					x_batch_t = x_test[begin:]
					y_attr_batch_t = y_attr_test[begin:]
					length_batch = length_test[begin:]
					cl = len(y_batch_t)
					for itemp in range(lstm_eval_config.batch_size-cl):
						y_batch_t = np.append(y_batch_t,[y_batch_t[0]],axis=0)
						x_batch_t = np.append(x_batch_t,[x_batch_t[0]],axis=0)
						y_attr_batch_t = np.append(y_attr_batch_t,[y_attr_batch_t[0]],axis=0)
						length_batch = np.append(length_batch,[length_batch[0]],axis=0)
					lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = lstm_dev_step(x_batch_t,y_batch_t,y_attr_batch_t,length_batch)
					total_losses+=t_loss
					lstm_losses+=l_loss
					attr_losses+=a_loss
					for jtemp in range(cl):
						indexes = np.flatnonzero(y_batch_t[jtemp])
						lstm_prc.multicount(lstm_p[jtemp], indexes)
						for index in indexes:
							lstm_matrix[index][lstm_p[jtemp]] += 1
						for k in range(global_config.num_of_attr):
							attr_prc.count(attr_p[jtemp][k], y_attr_batch_t[jtemp][k], k)
							
					lstm_prc.compute()
					attr_prc.compute()

					lstm_prc.output()
					attr_prc.output()

					if (lstm_prc.macro_f1[0] > best_macro_f1) or skiptrain:
						best_macro_f1 = lstm_prc.macro_f1[0]
						f_f1 = open(val_lstm_log_dir+'best_macro_f1','a+')
						f_f1.write('eva:'+str(eva_number)+' '+str(best_macro_f1)+'\n')
						f_f1.close()
						print 'Validation'
						if not skiptrain:
							saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)
						all_count = 0.0
						total_losses,lstm_losses,attr_losses = 0.0,0.0,0.0
						val_lstm_prc = PrecRecallCounter(lstm_config.num_classes,val_lstm_log_dir,'lstm',val_number)
						val_attr_prc = PrecRecallCounter([2 for temp in range(global_config.num_of_attr)],val_attr_log_dir,'attr',val_number)
						val_lstm_matrix = [[0 for j in range(lstm_config.num_classes)]for i in range(lstm_config.num_classes)]
						num = int(len(y_val)/float(lstm_eval_config.batch_size))
						if val_case:
							f_case = open(val_lstm_log_dir+'case'+str(val_number),'w')
						print num
						for i in range(num):
							if i %100 == 0:
								print i
							begin = i * lstm_eval_config.batch_size
							end = (i+1) * lstm_eval_config.batch_size
							y_batch_t = y_val[begin:end]
							x_batch_t = x_val[begin:end]
							y_attr_batch_t = y_attr_val[begin:end]
							length_batch = length_val[begin:end]
							
							lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = lstm_dev_step(x_batch_t,y_batch_t,y_attr_batch_t,length_batch)
							total_losses+=t_loss
							lstm_losses+=l_loss
							attr_losses+=a_loss
							for j in range(lstm_eval_config.batch_size):
								indexes = np.flatnonzero(y_batch_t[j])
								val_lstm_prc.multicount(lstm_p[j], indexes)
								for index in indexes:
									val_lstm_matrix[index][lstm_p[j]] += 1
								for k in range(global_config.num_of_attr):
									val_attr_prc.count(attr_p[j][k], y_attr_batch_t[j][k], k)
								if val_case:
									towrite = str(lstm_p[j])+'\t'+str(indexes[0])+'\t'+str(attr_p[j])+'\t'+str(y_attr_batch_t[j])+'\t'
									for w in range(len(x_batch_t[j])):
										if w == length_batch[j]:
											break
										towrite = towrite + id2word[x_batch_t[j][w]]+' '
									for temp_attr in range(global_config.num_of_attr):
										towrite = towrite + '\t'
										for w in range(len(x_batch_t[j])):
											if w == length_batch[j]:
												break
											towrite = towrite + str(attn_weights[j][temp_attr][w]/np.max(attn_weights[j][temp_attr]))+' '
									towrite = towrite + '\n'
									f_case.write(towrite)
						begin = num * lstm_eval_config.batch_size
						y_batch_t = y_val[begin:]
						x_batch_t = x_val[begin:]
						y_attr_batch_t = y_attr_val[begin:]
						length_batch = length_val[begin:]
						cl = len(y_batch_t)
						for itemp in range(lstm_eval_config.batch_size-cl):
							y_batch_t = np.append(y_batch_t,[y_batch_t[0]],axis=0)
							x_batch_t = np.append(x_batch_t,[x_batch_t[0]],axis=0)
							y_attr_batch_t = np.append(y_attr_batch_t,[y_attr_batch_t[0]],axis=0)
							length_batch = np.append(length_batch,[length_batch[0]],axis=0)
						lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = lstm_dev_step(x_batch_t,y_batch_t,y_attr_batch_t,length_batch)
						total_losses+=t_loss
						lstm_losses+=l_loss
						attr_losses+=a_loss
						for jtemp in range(cl):
							indexes = np.flatnonzero(y_batch_t[jtemp])
							val_lstm_prc.multicount(lstm_p[jtemp], indexes)
							for index in indexes:
								val_lstm_matrix[index][lstm_p[jtemp]] += 1
							for k in range(global_config.num_of_attr):
								val_attr_prc.count(attr_p[jtemp][k], y_attr_batch_t[jtemp][k], k)
							if val_case:
									towrite = str(lstm_p[jtemp])+'\t'+str(indexes[0])+'\t'+str(attr_p[jtemp])+'\t'+str(y_attr_batch_t[jtemp])+'\t'
									for w in range(len(x_batch_t[jtemp])):
										if w == length_batch[jtemp]:
											break
										towrite = towrite + id2word[x_batch_t[jtemp][w]]+' '
									for temp_attr in range(global_config.num_of_attr):
										towrite = towrite + '\t'
										for w in range(len(x_batch_t[jtemp])):
											if w == length_batch[jtemp]:
												break
											towrite = towrite + str(attn_weights[jtemp][temp_attr][w]/np.max(attn_weights[jtemp][temp_attr]))+' '
									towrite = towrite + '\n'
									f_case.write(towrite)

						val_lstm_prc.compute()
						val_attr_prc.compute()
						val_lstm_prc.output()
						val_attr_prc.output()
						if valmatrix:
							fm = open(val_lstm_log_dir+str(val_number)+'matrix','w')
							for i in range(lstm_config.num_classes):
								towrite = ""
								for j in range(lstm_config.num_classes):
									towrite = towrite + str(val_lstm_matrix[i][j])+' '
								towrite = towrite + '\n'
								fm.write(towrite)
							fm.close()
						val_number += 1


					if mixandmatrix:
						fm = open(lstm_log_dir+str(eva_number)+'matrix','w')
						for i in range(lstm_config.num_classes):
							towrite = ""
							for j in range(lstm_config.num_classes):
								towrite = towrite + str(lstm_matrix[i][j])+' '
							towrite = towrite + '\n'
							fm.write(towrite)
						fm.close()

					num = float(num)
					tn = datetime.datetime.now()
					print tn.isoformat()
					print 'loss total:{:g}, lstm:{:g}, attr:{:g}'.format(total_losses/num,lstm_losses/num,attr_losses/num)
					if skiptrain:
						break
					eva_number += 1

if __name__ == "__main__":
	tf.app.run()








