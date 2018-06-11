import datetime
from prec_recall_counter import PrecRecallCounter
import global_config
import numpy as np
def lstm_train_step(lstm_train_op, lstm_global_step, lstm_model, sess, x_batch, y_batch, y_attr_batch, length_batch):
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
def lstm_dev_step(lstm_model, sess, x_batch, y_batch, y_attr_batch, length_batch):
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
def evaluation(eva_number,lstm_model,sess,lstm_config,lstm_log_dir,attr_log_dir,y_test,x_test,y_attr_test,length_test):
	print 'Evaluating'
	all_count = 0.0
	total_losses,lstm_losses,attr_losses = 0.0,0.0,0.0
	lstm_prc = PrecRecallCounter(lstm_config.num_classes,lstm_log_dir,'lstm',eva_number)
	attr_prc = PrecRecallCounter([2 for temp in range(global_config.num_of_attr)],attr_log_dir,'attr',eva_number)
	lstm_matrix = [[0 for j in range(lstm_config.num_classes)]for i in range(lstm_config.num_classes)]
	num = int(len(y_test)/float(lstm_config.batch_size))
	print "num of batches to evaluate:",num
	for i in range(num):
		begin = i * lstm_config.batch_size
		end = (i+1) * lstm_config.batch_size
		y_batch_t = y_test[begin:end]
		x_batch_t = x_test[begin:end]
		y_attr_batch_t = y_attr_test[begin:end]
		length_batch = length_test[begin:end]
	
		lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = lstm_dev_step(lstm_model,sess,x_batch_t,y_batch_t,y_attr_batch_t,length_batch)
		total_losses+=t_loss
		lstm_losses+=l_loss
		attr_losses+=a_loss
		for j in range(lstm_config.batch_size):
			indexes = np.flatnonzero(y_batch_t[j])
			lstm_prc.multicount(lstm_p[j], indexes)
			for index in indexes:
				lstm_matrix[index][lstm_p[j]] += 1
			for k in range(global_config.num_of_attr):
				attr_prc.count(attr_p[j][k], y_attr_batch_t[j][k], k)
	
	begin = num * lstm_config.batch_size
	y_batch_t = y_test[begin:]
	x_batch_t = x_test[begin:]
	y_attr_batch_t = y_attr_test[begin:]
	length_batch = length_test[begin:]
	cl = len(y_batch_t)
	for itemp in range(lstm_config.batch_size-cl):
		y_batch_t = np.append(y_batch_t,[y_batch_t[0]],axis=0)
		x_batch_t = np.append(x_batch_t,[x_batch_t[0]],axis=0)
		y_attr_batch_t = np.append(y_attr_batch_t,[y_attr_batch_t[0]],axis=0)
		length_batch = np.append(length_batch,[length_batch[0]],axis=0)
	lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = lstm_dev_step(lstm_model,sess,x_batch_t,y_batch_t,y_attr_batch_t,length_batch)
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
	num = float(num)
	tn = datetime.datetime.now()
	print tn.isoformat()
	print 'loss total:{:g}, lstm:{:g}, attr:{:g}'.format(total_losses/num,lstm_losses/num,attr_losses/num)
	return lstm_prc.macro_f1[0]
def validation(val_number,lstm_model,sess,lstm_config,val_lstm_log_dir,val_attr_log_dir,y_val,x_val,y_attr_val,length_val):
	print 'Validating'
	all_count = 0.0
	total_losses,lstm_losses,attr_losses = 0.0,0.0,0.0
	val_lstm_prc = PrecRecallCounter(lstm_config.num_classes,val_lstm_log_dir,'lstm',val_number)
	val_attr_prc = PrecRecallCounter([2 for temp in range(global_config.num_of_attr)],val_attr_log_dir,'attr',val_number)
	val_lstm_matrix = [[0 for j in range(lstm_config.num_classes)]for i in range(lstm_config.num_classes)]
	num = int(len(y_val)/float(lstm_config.batch_size))
	print "num of batches to validate:",num
	for i in range(num):
		begin = i * lstm_config.batch_size
		end = (i+1) * lstm_config.batch_size
		y_batch_t = y_val[begin:end]
		x_batch_t = x_val[begin:end]
		y_attr_batch_t = y_attr_val[begin:end]
		length_batch = length_val[begin:end]
		
		lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = lstm_dev_step(lstm_model,sess,x_batch_t,y_batch_t,y_attr_batch_t,length_batch)
		total_losses+=t_loss
		lstm_losses+=l_loss
		attr_losses+=a_loss
		for j in range(lstm_config.batch_size):
			indexes = np.flatnonzero(y_batch_t[j])
			val_lstm_prc.multicount(lstm_p[j], indexes)
			for index in indexes:
				val_lstm_matrix[index][lstm_p[j]] += 1
			for k in range(global_config.num_of_attr):
				val_attr_prc.count(attr_p[j][k], y_attr_batch_t[j][k], k)
	begin = num * lstm_config.batch_size
	y_batch_t = y_val[begin:]
	x_batch_t = x_val[begin:]
	y_attr_batch_t = y_attr_val[begin:]
	length_batch = length_val[begin:]
	cl = len(y_batch_t)
	for itemp in range(lstm_config.batch_size-cl):
		y_batch_t = np.append(y_batch_t,[y_batch_t[0]],axis=0)
		x_batch_t = np.append(x_batch_t,[x_batch_t[0]],axis=0)
		y_attr_batch_t = np.append(y_attr_batch_t,[y_attr_batch_t[0]],axis=0)
		length_batch = np.append(length_batch,[length_batch[0]],axis=0)
	lstm_p,attr_p,t_loss,l_loss,a_loss,attn_weights = lstm_dev_step(lstm_model,sess,x_batch_t,y_batch_t,y_attr_batch_t,length_batch)
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
	
	val_lstm_prc.compute()
	val_attr_prc.compute()
	val_lstm_prc.output()
	val_attr_prc.output()
	
	num = float(num)
	tn = datetime.datetime.now()
	print tn.isoformat()
	print 'loss total:{:g}, lstm:{:g}, attr:{:g}'.format(total_losses/num,lstm_losses/num,attr_losses/num)