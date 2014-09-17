#!/usr/local/bin/env python
from __future__ import division
import scipy
from scipy import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow 
import scipy.optimize as opt
import seaborn as sns
import pandas as pd
import os
import ss
import utils

sns.set(font="Helvetica")

def sigmoid(p,x):
    x0,y0,c,k=p
    y = c / (1 + np.exp(k*(x-x0))) + y0
    return y


def residuals(p,x,y):
    return y - sigmoid(p,x)


def res(arr,lower=0.0,upper=1.0):
    arr=arr.copy()
    if lower>upper: lower,upper=upper,lower
    arr -= arr.min()
    arr *= (upper-lower)/arr.max()
    arr += lower
    return arr


def fit_scurves(ysim=None, task='ssRe', showPSE=True, ax=None, labels=None, **kwargs):
	
	plt.ion()
	sns.set(style='white', font="Helvetica")
	
	if task=='ssRe':
		x=np.array([400, 350, 300, 250, 200], dtype='float')
		emp_ReBSL=np.array([.994, .982, .896, .504, .103], dtype='float')
		emp_RePNL=np.array([.993, .985, .925, .594, .181], dtype='float')
		ydata=[emp_ReBSL, emp_RePNL]
		xsim=np.linspace(15, 50, 10000)
		scale_factor=100
	else:
		x=np.array([100, 80, 60, 40, 20, 0], dtype='float')
		emp_ProBSL=np.array([0.931, 0.744, 0.471, 0.240, 0.034, 0.005], dtype='float')
		emp_ProPNL=np.array([0.926, 0.767, 0.485, 0.241, 0.036, 0.006], dtype='float')
		ydata=[emp_ProBSL, emp_ProPNL]
		xsim=np.linspace(-5, 10, 10000)
		scale_factor=10
	
	x=res(-x,lower=x[-1]/10, upper=x[0]/10)
	
	if ysim is None:
		ysim=ydata
		skip_sim_plots=True
		
	ys=[ydata[0], ydata[1], ysim[0], ysim[1]]

	if ax is None:

		f = plt.figure(figsize=(8,9.5)) 
		f.subplots_adjust(top=0.95, wspace=0.12, left=0.19, right=0.98, bottom=0.15)	
		ax = f.add_subplot(111)

	else:
		ax=ax
		x=kwargs['x']
	
	sns.despine()
	
	i=0; empPSE=[]; simPSE=[]

	if "colors" in kwargs:
		colors=kwargs['colors']
	else:
		#colors=['MediumBlue', '#E60000', '#1975FF', '#E6005C']
		colors=['MediumBlue', '#E60000', 'Blue', 'Red']
		#marker_colors=['Blue', 'Red']
	label_list=['Emp BSL','Emp PNL', 'Sim BSL', 'Sim PNL']

	for y in ys:
		
		y=res(y, lower=y[-1], upper=y[0])
		
		print(x)
		print(y)
		
		p_guess=(np.median(x),np.median(y),1.0,1.0)
		p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
		    residuals,p_guess,args=(x,y),full_output=1)#,warning=True)  
    	
		x0,y0,c,k=p
		#print('''\
		#x0 = {x0}
		#y0 = {y0}
		#c = {c}
		#k = {k}
		#'''.format(x0=x0,y0=y0,c=c,k=k))
		
		xp = xsim 
		pxp=sigmoid(p,xp)
		idx = (np.abs(pxp - .5)).argmin()
		
		print "%s PSE = %s\n" % (label_list[i], str(xp[idx]*scale_factor)[:05])
		
		# Plot the results
		if i==2 or i==3:
			ax.plot(xp, pxp, '--', lw=5.0, color=colors[i], label=label_list[i])
			ax.plot(x, y, marker='o', color=colors[i], ms=16, lw=0, mfc='none', mew=2.4, mec=colors[i])
			simPSE.append(xp[idx]/scale_factor)

		else:
			colors=['MediumBlue', '#E60000', 'Blue', 'Red']
			ax.plot(xp, pxp, '-', lw=10, color=colors[i], alpha=.35, label=label_list[i])
			ax.plot(x, y, marker='o', color=colors[i], ms=14, lw=0, alpha=.35)
			empPSE.append(xp[idx]/scale_factor)

		i+=1

	ssPSE=[empPSE, simPSE]
	ax.set_ylim(0, 1.05)
	
	if task=='ssRe':
		ax.set_xlim(18, 45)
		ax.set_xlabel('SSD (ms)', fontsize=34)
		ax.set_xticks(np.arange(20, 45, 5))
		ax.set_xticklabels(np.arange(200, 450, 50), fontsize=26)
	else:
		ax.set_xlim(-1.0, 11.0)
		ax.set_xlabel('P(Go)', fontsize=34)
		ax.set_xticks(np.arange(0, 12, 2))
		ax.set_xticklabels(np.arange(0, 1.2, .20), fontsize=26)
	
	ax.set_yticks(np.arange(0, 1.2, .2))	
	plt.setp(ax.get_yticklabels(), fontsize=26)	
	ax.set_ylabel('P(Inhibit)', fontsize=34, labelpad=11) 
	
	if labels is not None:
		label_list=['Emp BSL','Emp PNL', 'Sim BSL', 'Sim PNL']
		ax.legend(loc=0, fontsize=26)
	
	#yy, locs = plt.yticks()
	#ll = ['%.2f' % a for a in yy]
	#plt.yticks(yy, ll)

	plt.savefig("stopsigmoid_%s%s" % (task, ".png"), format='png', dpi=600)
	plt.savefig("stopsigmoid_%s%s" % (task, ".svg"), rasterized=True, dpi=600)
	
	if showPSE:
		ssrtFig=plotPSE(ssPSE=ssPSE, task=task)

	return ax


def scurves(ysim=None, task='ssRe', pstop=.5, showPSE=True, labels=None, plot_data=False):

	#plt.ion()
	pth=utils.find_path()

	sns.set(style='white', font="Helvetica")
	npoints=len(ysim[0])
	xsim=np.linspace(-5, 110, 10000)
	scale_factor=10

	x=np.array(np.linspace(1, 100, npoints), dtype='float')
	xxticks=x/scale_factor
	xxticklabels=x/100
	xxlim=(-.5, 11)
	xxlabel='P(Go)'

	emp_bsl=np.array([0.931, 0.744, 0.471, 0.240, 0.034, 0.005], dtype='float')[::-1]
	emp_pnl=np.array([0.926, 0.767, 0.485, 0.241, 0.036, 0.006], dtype='float')[::-1]

	if 'ssRe' in task:
		xxlabel='SSD (ms)'
		xxticklabels=np.arange(200, 500, 50)
		emp_bsl=np.array([.994, .982, .896, .504, .103], dtype='float')[::-1]
		emp_pnl=np.array([.993, .985, .925, .594, .181], dtype='float')[::-1]
		
	x=res(-x,lower=x[-1]/10, upper=x[0]/10)

	pse=[]
	f = plt.figure(figsize=(6, 7)) 
	f.subplots_adjust(top=0.95, wspace=0.12, left=0.19, right=0.98, bottom=0.15)	
	ax = f.add_subplot(111)

	sns.despine()
	if plot_data:
		ie=0
		ysim.append(emp_bsl)
		ysim.append(emp_pnl)
	
	colors = sns.blend_palette(["#00A37A", "#4D94B8"], len(ysim))
	title="Stop Curves for Nested Model"
	#colors=sns.blend_palette(["#53FCAL", "#40a368"], len(ysim))
	#colors = sns.blend_palette(["#6600CC", "#66CCFF"], len(ysim))
	#title="Stop Curves for Independent Model"
	
	ai=0
	for i, yi in enumerate(ysim):

		y=res(yi, lower=yi[-1], upper=yi[0])

		p_guess=(np.median(x),np.median(y),1.0,1.0)
		p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
		    residuals,p_guess,args=(x,y),full_output=1)
    	
		x0,y0,c,k=p

		xp = xsim 
		pxp=sigmoid(p,xp)

		idx = (np.abs(pxp - pstop)).argmin()

		# Plot the results
		if plot_data and (i==len(ysim)-1 or i==len(ysim)-2):
			ecolors=['Navy', 'FireBrick']#, '#E60000']; 
			elabels=['Data (BSL)', 'Data (PNL)']
			ax.plot(xp, pxp, '-', lw=8, color=ecolors[ie], alpha=.9)
			ax.plot(x, y, marker='o', color=ecolors[ie], ms=13, lw=0, alpha=.4, label=elabels[ie])
			ie+=1
		else:
			ax.plot(xp, pxp, '-', lw=5.5, color=colors[i], alpha=.95-ai)
			ax.plot(x, y, marker='o', color=colors[i], ms=9, lw=0, alpha=.95-ai)#, label=labels[i])
			ai+=.02

		pse.append(xp[idx]/scale_factor)

	ax.set_xlim(xxlim)
	ax.set_xlabel(xxlabel, fontsize=22)
	ax.set_xticks(xxticks)
	ax.set_xticklabels(xxticklabels, fontsize=18)
	ax.set_ylim(0, 1.05)	

	plt.setp(ax.get_yticklabels(), fontsize=18)	
	ax.set_ylabel('P(Inhibit)', fontsize=22, labelpad=8) 
	ax.legend(loc=0, fontsize=12)
	ax.set_title(title, fontsize=22)
	plt.tight_layout()
	plt.savefig(pth+"CoAx/SS/"+title+".png", dpi=600)



def factorial_scurves(ysim=None, task='ssRe', pstop=.5, showPSE=True, ncurves=5, labels=None, predict_brain=False):
	
	plt.ion()

	sns.set(style='white', font="Helvetica")
	npoints=len(ysim[0])
	xsim=np.linspace(-5, 120, 10000)
	scale_factor=10
	
	x=np.array(np.linspace(10, 100, npoints), dtype='float')
	xxticks=x/scale_factor
	xxticklabels=x/100
	xxlim=(0, 10.5)
	xxlabel='P(Go)'
	
	if task=='ssRe':
		xxlabel='SSD (ms)'
		xxticks=x/scale_factor
		xxticklabels=np.arange(250, 550, 50)
	
	x=res(-x,lower=x[-1]/10, upper=x[0]/10)
	
	pse=[]
	
	f=plt.figure(figsize=(16, 9.5))
	ax = plt.subplot2grid((4, 2), (0, 0), colspan=1, rowspan=4)
	ax2 = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=4)
	sns.despine()
	
	colors = sns.blend_palette(["LimeGreen", "#1919A3"], len(ysim)+1)
	
	if labels is None:
		labels=['C'+str(i) for i in range(len(ysim))]
	
	for i, yi in enumerate(ysim):

		y=res(yi, lower=yi[-1], upper=yi[0])

		p_guess=(np.median(x),np.median(y),1.0,1.0)
		p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
		    residuals,p_guess,args=(x,y),full_output=1)
    	
		x0,y0,c,k=p
		
		xp = xsim 
		pxp=sigmoid(p,xp)
		
		idx = (np.abs(pxp - pstop)).argmin()
		
		if predict_brain and i==0:
			
			x_at_pstop=xp[idx]/scale_factor
			
			return x_at_pstop

		# Plot the results
		ax.plot(xp, pxp, '-', lw=8, color=colors[i], alpha=.6, label=labels[i])
		
		pse.append(xp[idx]/scale_factor)

	ax.vlines(x=xxticks[-2], ymin=0, ymax=1, lw=20, color='k', alpha=.3)
	
	p = FancyArrow(0.432, 0.55, 0.098, 0.0, width=0.005,
		length_includes_head=False, lw=3.5, fc='DarkGray', ec='DarkGray',
		head_width=.02, head_length=.02, 
		shape='full', overhang=0, 
		head_starts_at_zero=False, 
		transform=ax.figure.transFigure, 
		clip_on=False) 
	ax.add_patch(p) 

	emp_ProBSL=np.array([0.931, 0.744, 0.471, 0.240, 0.034, 0.005], dtype='float')
	pbsl=np.array([0.92266667,  0.745, 0.46633333, 0.231, 0.015, 0.00233333], dtype='float')#[::-1]#, 0.00233333], dtype='float')[::-1]
	#ppnl=np.array([0.93533333, 0.752, 0.501, 0.22033333, 0.018, 0.00266667], dtype='float')
	#ax2 = fit_scurves(ysim=[pbsl, ppnl], task='ssPro', showPSE=False, ax=ax2)
	#kwargs = {'line_colors':['Gray', 'Gray'], 'point_colors':colors}
	
	ax2 = basic_curves(ysim=[pbsl, emp_ProBSL], task='ssPro', showPSE=False, ax=ax2, line_colors=['Gray', 'Black'], point_colors=colors[::-1])#**kwargs)	

	ax.set_xlim(xxlim)
	ax.set_xlabel(xxlabel, fontsize=34)
	ax.set_xticks(xxticks)
	ax.set_xticklabels(xxticklabels, fontsize=26)
	ax.set_ylim(0, 1.05)	
		
	ax.set_ylabel('P(Inhibit)', fontsize=34, labelpad=11) 
	ax2.set_ylabel("", fontsize=0)
	plt.setp(ax.get_yticklabels(), fontsize=26)
	plt.setp(ax2.get_yticklabels(), visible=False)

	ax.legend(loc=0, fontsize=26)

	plt.tight_layout(w_pad=4.5, h_pad=2)#h_pad=3, w_pad=1)

	if os.path.isdir("/Users/kyle"):
		f.savefig("/Users/kyle/Dropbox/CoAx/SS/FactorialSims/ReProFactorial_SCurves_%s%s" % (task, ".svg"), rasterized=True, dpi=600)	
		plt.savefig("/Users/kyle/Dropbox/CoAx/SS/FactorialSims/ReProFactorial_SCurves_%s%s" % (task, ".png"), format='png', dpi=600)	
	elif os.path.isdir("/home/kyle"):
		f.savefig("/home/kyle/Dropbox/CoAx/ss/simdata/ReProFactorial_SCurves%s%s" % (task, ".svg"), rasterized=True, dpi=600)	
		plt.savefig("/home/kyle/Dropbox/CoAx/ss/simdata/ReProFactorial_SCurves%s%s" % (task, ".png"), format='png', dpi=600)

	return f

def basic_curves(ysim=None, task='ssRe', showPSE=True, ax=None, labels=None, pstop=.05, **kwargs):
	
	plt.ion()

	sns.set(style='white', font="Helvetica")

	npoints=len(ysim[0])
	
	if task=='ssRe':
		x=np.array([400, 350, 300, 250, 200], dtype='float')
		xsim=np.linspace(15, 50, 10000)
		scale_factor=100
	else:
		x=np.array([100, 80, 60, 40, 20, 0], dtype='float')
		xsim=np.linspace(-5, 10, 10000)
		scale_factor=10
		xxticks=x/scale_factor
		xxticklabels=x/100
		xxlim=(-0.5, 10.5)
		xxlabel='P(Go)'

	x=res(-x,lower=x[-1]/10, upper=x[0]/10)
	
	if ax is None:
		f = plt.figure(figsize=(8,9.5)) 
		f.subplots_adjust(top=0.95, wspace=0.12, left=0.19, right=0.98, bottom=0.15)	
		ax = f.add_subplot(111)
	else:
		pass

	sns.despine()

	if 'line_colors' not in kwargs:
		line_colors = sns.blend_palette(["LimeGreen", "Navy"], len(ysim))
		point_colors = sns.blend_palette(["Black", "Black"], len(ysim))
	else:
		line_colors = kwargs['line_colors']
		point_colors = kwargs['point_colors']

	if labels!=None:
		labels=labels
	else:
		labels=['C'+str(i) for i in range(len(ysim))]

	for i, yi in enumerate(ysim):

		y=res(yi, lower=yi[-1], upper=yi[0])

		p_guess=(np.median(x),np.median(y),1.0,1.0)
		p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
		    residuals,p_guess,args=(x,y),full_output=1)
    	
		x0,y0,c,k=p

		xp = xsim 
		pxp=sigmoid(p,xp)

		idx = (np.abs(pxp - pstop)).argmin()

		# Plot the results
		if i==0:
			ax.plot(xp, pxp, '-', lw=7.5, color=line_colors[i], alpha=.6, label=labels[i])
		else:
			ax.plot(xp, pxp, '--', lw=7.5, color=line_colors[i], alpha=.45, label=labels[i])
		
		ci=0
		#for i, ypoint in enumerate(y):
		for ypoint in y:
			if i==0:
				plt.plot(x[ci], ypoint, marker='o', color=point_colors[ci], ms=19, lw=0, alpha=.55)
			else:
				plt.plot(x[ci], ypoint, marker='o', color=point_colors[ci], ms=22,  mfc='none', mew=2.5, mec=point_colors[ci],  lw=0, alpha=1)
			ci+=1
		#ax.plot(x, y, marker='o', color=colors[i], ms=10, lw=0)
		#pse.append(xp[idx]/scale_factor)

	if 'Re' in task:
		plt.vlines(x=450, ymin=0, ymax=1, lw=20, color='k', alpha=.2)

	ax.set_xlim(xxlim)
	ax.set_xlabel(xxlabel, fontsize=34)
	ax.set_xticks(xxticks)
	ax.set_xticklabels(xxticklabels, fontsize=26)
	ax.set_ylim(-0.05, 1.05)	

	plt.setp(ax.get_yticklabels(), fontsize=26)	
	ax.set_ylabel('P(Inhibit)', fontsize=34, labelpad=11) 
	
	#ax.legend(loc=0, fontsize=26)
	#yy, locs = plt.yticks()
	#ll = ['%.2f' % a for a in yy]
	#plt.yticks(yy, ll)

	#plt.savefig("stopsigmoid_%s%s" % (task, ".png"), format='png', dpi=600)
	#plt.savefig("stopsigmoid_%s%s" % (task, ".svg"), rasterized=True, dpi=600)

	return ax


def go_rt_evs(sim_rt=None, emp_rt=None, emp_err=None, sim_err=None, task='ssRe'):
	
	plt.ion()
	sns.set(style='white', font="Helvetica")
	sns.set_context('poster')
	
	x=np.array([1,2])
	xsim=np.array([1.1, 1.9])
	
	sim_rt=np.array([rt*1000 for rt in sim_rt])
	emp_rt=np.array([rt*1000 for rt in emp_rt])
	sim_err=np.array([rt*1000 for rt in sim_err])
	emp_err=np.array([rt*1000 for rt in emp_err])

	f = plt.figure(figsize=(5.5, 6.5)) 
	ax = f.add_subplot(111)
	sns.despine()
	
	ax.bar(x, emp_rt, yerr=emp_err, color='#441B5F', align='center', error_kw=dict(elinewidth=3, ecolor='black'), alpha=.8)
	childrenLS=ax.get_children()
	barlist=filter(lambda x: isinstance(x, matplotlib.patches.Rectangle), childrenLS)
	barlist[1].set_color('#195E19')
	
	ax.errorbar(xsim, sim_rt, yerr=sim_err, color='k', marker='o', mfc=None, ms=16, lw=3, mew=4, mec='k', ecolor='k', elinewidth=3, capsize=0)
	ax.plot(xsim[0], sim_rt[0], marker='o', color='#9900FF', ms=15)
	ax.plot(xsim[1], sim_rt[1], marker='o', color='#339933', ms=15)

	ax.set_xlim(0.5, 2.5)
	ax.set_xticks(x)
	
	if 'Re' in task:
		ylim=[550, 580]
		ax.set_xticklabels(['BSL', 'PNL'], fontsize=22)	
		ax.set_xlabel('Feedback Condition', fontsize=28)
	else:
		ylim=[520,550]
		ax.set_xticklabels(['Low', 'High'], fontsize=22)
		ax.set_xlabel('Go Probabiltiy', fontsize=28)

	ax.set_ylim(ylim[0], ylim[1])	
	ax.set_yticks(np.arange(ylim[0], ylim[1]+10, 10))
	ax.set_yticklabels(np.arange(ylim[0], ylim[1]+10, 10), fontsize=20)
	ax.set_ylabel('Go RT (ms)', fontsize=28)

	yy, locs = plt.yticks()
	plt.tight_layout()

	plt.savefig("GoRT_%s%s" % (task, ".png"), format='png', dpi=300)
	plt.savefig("GoRT_%s%s" % (task, ".svg"), rasterized=True, dpi=300)


def plot_goRTs(sim_rt=None, task='ssRe'):
	
	plt.ion()
	sns.set(style='white', font="Helvetica")
	
	x=np.array([1,2])
	
	if task=='ssRe':
		emp_rt = np.array([565.625, 573.415])	
		emp_err = np.array([2.9, 2.4])
		ylim=[550, 580]
		inc=10
	elif task=='ssPro':
		emp_rt = np.array([539.154, 539.344])
		emp_err = np.array([1.0, 1.0])
		ylim=[530, 545]
		inc=5
	if sim_rt is None:
		sim_rt = emp_rt
	
	if sim_rt[0]<1:
		sim_rt=np.array([rt*1000 for rt in sim_rt])

	f = plt.figure(figsize=(5.5, 6.5)) 
	f.subplots_adjust(top=0.95, wspace=0.12, left=0.23, right=0.97, bottom=0.10)
	ax = f.add_subplot(111)
	sns.despine()
	
	ax.bar(x, emp_rt, yerr=emp_err, align='center', color='MediumBlue', error_kw=dict(elinewidth=3, ecolor='black'), alpha=.6)
	childrenLS=ax.get_children()
	barlist=filter(lambda x: isinstance(x, matplotlib.patches.Rectangle), childrenLS)
	barlist[1].set_color('#E60000')
	
	ax.plot(x[0], sim_rt[0], color='#1975FF', marker='o', ms=22, lw=0, alpha=.85, mew=2, mec='Navy')
	#ax.plot(x[1], sim_rt[1], color='#FF4D94', marker='o', ms=16, lw=0, alpha=.85)
	ax.plot(x[1], sim_rt[1], color='#E6005C', marker='o', ms=22, lw=0, alpha=.85, mew=2, mec='FireBrick')
	ax.set_ylim(ylim[0], ylim[1])	
	ax.set_yticks(np.arange(ylim[0], ylim[1]+inc, inc))
	ax.set_yticklabels(np.arange(ylim[0], ylim[1]+inc, inc), fontsize=24)
	ax.set_ylabel('Go RT (ms)', fontsize=30)
	
	ax.set_xlim(0.5, 2.5)
	ax.set_xticks(x)
	ax.set_xticklabels(['BSL', 'PNL'], fontsize=28)

	yy, locs = plt.yticks()
	ll = ['%.3f' % a for a in yy]
	plt.yticks(yy, ll)

	plt.savefig("GoRT_%s%s" % (task, ".png"), format='png', dpi=600)
	plt.savefig("GoRT_%s%s" % (task, ".svg"), rasterized=True, dpi=600)


def plotPSE(ssPSE=None, task='ssRe'):
	
	plt.ion()
	sns.set(style='white', font="Helvetica")
	

	if task=='ssRe':
		emp_pse = np.array([[.3491222879, .3614880328],[.3491222879, .3614880328]])	
		emp_err = np.array([0.002834, 0.0027982])
	else:
		emp_pse=np.array([0.380, 0.387])
		emp_err=np.array([0.009, 0.009])

	if ssPSE is None:
		ssPSE = emp_pse
		
	f = plt.figure(figsize=(5.5, 6.5)) 
	f.subplots_adjust(top=0.95, wspace=0.12, left=0.23, right=0.97, bottom=0.10)
	ax = f.add_subplot(111)
	sns.despine()
	
	x=np.array([1,2])
	ax.bar(x, np.array(ssPSE[0]), align='center', color='MediumBlue', yerr=emp_err, error_kw=dict(elinewidth=3, ecolor='black'), alpha=.6)
	childrenLS=ax.get_children()
	barlist=filter(lambda x: isinstance(x, matplotlib.patches.Rectangle), childrenLS)
	barlist[1].set_color('#E60000')
	
	#ax.plot(x, np.array(ssPSE[1]), color='k', marker='o', ms=10, lw=0)'#1975FF', '#E6005C'
	ax.plot(x[0], np.array(ssPSE[1])[0], color='#1975FF', marker='o', ms=22, lw=0, alpha=.85, mew=2, mec='Navy')
	#ax.plot(x[1], np.array(ssPSE[1])[1], color='#FF4D94', marker='o', ms=16, lw=0, alpha=.85)
	ax.plot(x[1], np.array(ssPSE[1])[1], color='#E6005C', marker='o', ms=22, lw=0, alpha=.85, mew=2, mec='FireBrick')
	ax.set_xticks(x)
	ax.set_xticklabels(['BSL', 'PNL'], fontsize=28)
	ax.set_xlim(0.5, 2.5)
	
	if task=='ssRe':
		ax.set_ylim(.335, .365)
		ax.set_yticks(np.arange(.335, .375, .01))
		ax.set_yticklabels(np.arange(.335, .375, .01), fontsize=24)
	else:
		ax.set_ylim(.34, .42)
		ax.set_yticks(np.arange(.34, .44, .02))
		ax.set_yticklabels(np.arange(.34, .44, .02), fontsize=24)
	
	ax.set_ylabel('PSE', fontsize=30)

	yy, locs = plt.yticks()
	ll = ['%.2f' % a for a in yy]
	plt.yticks(yy, ll)

	plt.savefig("pse_%s%s" % (task, ".png"), format='png', dpi=600)
	plt.savefig("pse_%s%s" % (task, ".svg"), rasterized=True, dpi=600)
	
	return f


def predict_neural_integrator(m={}, arr=np.arange(0, 1, .1), task='ssReBSL', plot=True):

	cd=dict()
	
	pglist=[0, .2, .4, .6, .8, 1]
	vlist=[0.073, 0.26, 0.4385, 0.601, 0.93, 1.09]
	ssdlist=[.4, .35, .3, .25, .20]
	simlist=[]

	if 'Re' in task:

		for ssd in ssdlist:
			
			sp={'mu_ss':m['sp']['mu_ss'], 'pGo':.5, 'ssd':ssd, 'ssTer':m['sp']['ssTer'], 'ssTer_var':m['sp']['ssTer_var']}
			gp={'a':m['gp']['a'], 'z':m['gp']['z'], 'v':m['gp']['v'], 'Ter':m['gp']['Ter'], 'st':m['gp']['st'], 'sz':m['gp']['sz'], 
				'eta':m['gp']['eta'], 's2':m['gp']['s2']}

			sim_data=ss.set_model(gParams=gp, sParams=sp, ntrials=m['ntrials'], 
				timebound=m['timebound'], predictBOLD=False, task=task)
			
			simlist.append(sim_data[2])

	elif 'Pro' in task:
		
		for i, pg in enumerate(pglist):
	
			m['gp']['v']=vlist[i]
			m['sp']['v']=pg
	
			sim_data=ss.set_model(gParams=m['gp'], sParams=m['sp'], ntrials=m['ntrials'], 
				timebound=m['timebound'], predictBOLD=False, task=task)
			simlist.append(simdata[1])
	
	for i in arr:
		
		x_at_p=scurves(ysim=np.array(simlist, dtype='float')[::-1], task=task, pstop=i, predict_brain=True)

def integrate_all_signals(df):

	td=dict()
	
	for ttype in df.keys():
		
		df[ttype].dropna(inplace=True)
		df[ttype].index=np.arange(len(df[ttype]))

        	integrals=[]
	        for trace in df[ttype][:]:
			csum_max=pd.Series(trace).cumsum().max()
			integrals.append(csum_max)
		
		td[ttype]=np.mean(integrals)

	return td

def plot_integrator_magnitude(cd):

	plt.ion()
	sns.set(style='white', context='talk', font="Helvetica")

	for i, k in enumerate(cd.keys()):
		
		f = plt.figure() 
		f.suptitle(k, fontsize=16)
		ax = f.add_subplot(111)#len(cd.keys()), 1, i)
		sns.despine()

		xlist=[]; ylist=[]
		for tdk in cd[k].keys():
			xlist.append(tdk)
				#xlist.append(tdk)
		for tdv in cd[k].values():
			ylist.append(tdv)
		
		sns.barplot(x=np.array(xlist), y=np.array(ylist), ci=None, palette="BuGn_d", ax=ax)













