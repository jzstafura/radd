#!usr/bin/env python
from __future__ import division
import os
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


def sub_dists(data, nbins=18, save=True):

	pth=find_path()

	sns.set_style('white')
	sns.set_context('poster')

	for i, sxdata in data.groupby('subj_idx'):
		
		f=plt.figure()
		ax = f.add_subplot(111, xlabel='RT', ylabel='count', title='Go RT distributions')
		
		if len(sxdata.rt>18):
			ax.hist(sxdata.rt, color='LimeGreen', lw=1.5, bins=nbins, histtype='stepfilled', alpha=0.)
			ax.grid()
		else:
			continue
		
		if save:
			f.savefig(pth+"CoAx/SS/HDDM/Proactive/sx_rt_dists/sx"+str(i)+'.png', dpi=600)
		else:
			subj_fig.show()
