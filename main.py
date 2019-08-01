# -*- coding:utf-8 -*-
import numpy as np

class turnIdx:
	def __init__(self):
		self.idx_hold = 1
		self.idx_dig = 1
		self.idx_change = 1
		self.cont_num = 0

	def cntup(self, action):
		if action == "hold":
			self.idx_hold += 1
		elif action == "dig":
			self.idx_dig += 1
		elif action == "change":
			self.idx_change += 1

	def reset(self):
		self.idx_hold = 1
		self.idx_dig = 1
		self.idx_change = 1
		self.cont_num = 0

	def continuity(self):
		self.cont_num += 1

class interest:
	def __init__(self, queue_size=3):
		self.prev_interest = [''] * queue_size
		self.cnt_interest = 0

	def update(self, inte):
		self.prev_interest.append(self.cnt_interest)
		del self.prev_interest[0]
		self.cnt_interest = inte

	def get_interest(self):
		return self.cnt_interest

	def get_prev_ave_interest(self):
		prev_val = self.prev_interest.append(self.cnt_interest)
		prev_val = [x for x in self.prev_interest if type(x) is not str]
		return np.mean(prev_val)

if __name__ == '__main__':
	pass