import numpy as np
import datetime
class PrecRecallCounter(object):
	def __init__(self, class_nums, log_dir, name, number):
		if type(class_nums) == int:
			self.class_nums = np.array([class_nums])
		elif type(class_nums) == list:
			self.class_nums = np.array(class_nums)
		else:
			print 'input must be int or list'
			exit()
		self.log_dir = log_dir
		self.name = name
		self.number = number

		self.total = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.pred = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.correct = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.accuracy_class = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.recall_class = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.accuracy_object = np.array([0.0 for i in self.class_nums])
		self.accuracy_total = 0.0
		self.macro_acc = np.array([0.0 for i in self.class_nums])
		self.macro_recall = np.array([0.0 for i in self.class_nums])

	def count(self, y, correct, x=0):
		if (correct >= self.class_nums[x]):
			return
		self.total[x][correct] += 1
		self.pred[x][y] += 1
		if correct == y:
			self.correct[x][y] += 1
	def multicount(self, y, indexes, x = 0):
		if len(indexes) == 0:
			return
		for index in indexes:
			self.total[x][index] += 1
			if y == index:
				self.correct[x][y] += 1
		self.pred[x][y] += 1
	def compute(self):
		self.accuracy_class = self.correct/(self.pred+1e-3)
		self.recall_class = self.correct/(self.total+1e-3)
		self.accuracy_object = np.sum(self.correct, axis=1)/(np.sum(self.total, axis=1)+1e-3)
		self.accuracy_total = np.sum(self.correct)/(np.sum(self.total)+1e-3)
		self.macro_acc = np.sum(self.accuracy_class, axis=1)/self.class_nums
		self.macro_recall = np.sum(self.recall_class, axis=1)/self.class_nums
		self.f1_class = 2*self.accuracy_class*self.recall_class/(self.accuracy_class+self.recall_class+1e-10)
		self.macro_f1 = np.sum(self.f1_class, axis = 1)/self.class_nums
	def output(self):
		print self.name
		print 'macro_acc:', np.round(self.macro_acc,4), 'macro_recall:', np.round(self.macro_recall,4), 'micro_acc:', np.round(self.accuracy_total,4)
		
		f = open(self.log_dir + str(self.number), 'w')
		tn = datetime.datetime.now()
		f.write(tn.isoformat()+'\n')
		f.write('macro_acc:'+str(np.round(self.macro_acc,4)) + '\n')
		f.write('macro_recall:' + str(np.round(self.macro_recall,4)) + '\n')
		f.write('macro_f1:' + str(np.round(self.macro_f1,4)) + '\n')
		f.write('micro_acc:' + str(np.round(self.accuracy_total,4)) + '\n')

		f.write('accuracy_class:'+'\n')
		for i in range(len(self.class_nums)):
			towrite = ''
			for j in range(self.class_nums[i]):
				towrite = towrite + str(round(self.accuracy_class[i][j],4)) + ' '
			towrite = towrite + '\n'
			f.write(towrite)

		f.write('recall_class:'+'\n')
		for i in range(len(self.class_nums)):
			towrite = ''
			for j in range(self.class_nums[i]):
				towrite = towrite + str(round(self.recall_class[i][j],4)) + ' '
			towrite = towrite + '\n'
			f.write(towrite)

		f.write('accuracy_object:'+'\n')
		towrite = ''
		for i in range(len(self.class_nums)):
			towrite = towrite + str(round(self.accuracy_object[i],4))+' '
		towrite = towrite + '\n'
		f.write(towrite)

		f.write('pred:'+'\n')
		for i in range(len(self.class_nums)):
			towrite = ''
			for j in range(self.class_nums[i]):
				towrite = towrite + str(self.pred[i][j]) + ' '
			towrite = towrite + '\n'
			f.write(towrite)

		f.write('total:'+'\n')
		for i in range(len(self.class_nums)):
			towrite = ''
			for j in range(self.class_nums[i]):
				towrite = towrite + str(self.total[i][j]) + ' '
			towrite = towrite + '\n'
			f.write(towrite)

		f.close()















