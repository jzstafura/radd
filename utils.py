#!usr/bin/env python
from __future__ import division
import os, sys, time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
import time
from IPython.display import HTML, Javascript, display


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


class PBinJ:
    
    def __init__(self, n, color='seagreen'):
    	self.n=n
	self.divid=str(uuid.uuid4())
	self.pb=HTML("""
	    <div style="border: 1px solid black; width:500px">
	    <div id="%s" style="background-color:%s; width:0%%">&nbsp;</div>
	    </div> 
	    """ % (self.divid, color))
	self.display=display
	display(self.pb)
  
    def update(self, n):
	display(self.pb)
	for i in range(n):
	    self.update_i(i)
    def update_i(self, i):
	self.display(Javascript("$('div#%s').width('%.2f%%')" % (self.divid, (i/self.n)*100)))
