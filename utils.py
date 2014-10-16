#!usr/bin/env python
from __future__ import division
import os, sys, time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def find_path():

	if os.path.isdir("/Users/kyle"):
		pth="/Users/kyle/Dropbox/"
		
	elif os.path.isdir("/home/kyle"):
		pth="/home/kyle/Dropbox/"
	
	return pth



def makePivot(df, cols='ssd', index=None, rows=None, values='stop_acc', func=np.mean):

	pvot=pd.pivot_table(df, index=index, rows=rows, columns=cols, values=values, aggfunc=func)

	return pvot



def sub_dists(data, nbins=18, save=True):

	pth=find_path()

	sns.set_style('white')
	sns.set_context('poster')

	for i, sxdata in data.groupby('subj_idx'):
		
		f=plt.figure()
		ax = f.add_subplot(111, xlabel='RT', ylabel='count', title='Go RT distributions')
		
		if len(sxdata.rt)>18:
			ax.hist(sxdata.rt, color='LimeGreen', lw=1.5, bins=nbins, histtype='stepfilled', alpha=0.3)
			ax.grid()
		else:
			continue
		
		if save:
			f.savefig(pth+"CoAx/SS/HDDM/Proactive/sx_rt_dists/sx"+str(i)+'.png', dpi=600)
		else:
			subj_fig.show()


def wrapper(func, *args, **kwargs):
	
	def wrapped():
		
		return func(*args, **kwargs)
	
	return wrapped

class ProgressBar:

	def __init__(self, iterations, have_ipython=True):
		self.iterations = iterations
		self.prog_bar = '[]'
		self.fill_char = '*'
		self.width = 40
		self.__update_amount(0)
		if have_ipython:
			self.animate = self.animate_ipython
		else:
			self.animate = self.animate_noipython

	def animate_ipython(self, iter):
		print '\r', self,
		sys.stdout.flush()
		self.update_iteration(iter + 1)

	def update_iteration(self, elapsed_iter):
		self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
		self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

	def __update_amount(self, new_amount):
		percent_done = int(round((new_amount / 100.0) * 100.0))
		all_full = self.width - 2
		num_hashes = int(round((percent_done / 100.0) * all_full))
		self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
		pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
		pct_string = '%d%%' % percent_done
		self.prog_bar = self.prog_bar[0:pct_place] + \
		    (pct_string + self.prog_bar[pct_place + len(pct_string):])

	def __str__(self):
		return str(self.prog_bar)