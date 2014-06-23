#!/usr/local/bin/env python
import scipy
from scipy import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import seaborn as sns
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

def fit_scurves(ysim=None, task='ssRe', showPSE=True):
	
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
	#ydata=res(ydata, lower=ydata[-1], upper=ydata[0])
	
	if ysim is None:
		ysim=ydata
		
	ys=[ydata[0], ydata[1], ysim[0], ysim[1]]
	
	f = plt.figure(figsize=(6,7.5)) 
	ax = f.add_subplot(111)
	sns.despine()
	
	i=0; empPSE=[]; simPSE=[]
	for y in ys:
		
		colors=['MediumBlue', '#E60000', '#1975FF', '#E6005C']#'#1975FF', '#FF0066']
		label_list=['EmpBSL','EmpPNL', 'SimBSL', 'SimPNL']
		
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
		
		if task=='ssRe':
			print "%s PSE = %s\n" % (label_list[i], str(xp[idx]*scale_factor)[:05])
		else:
			print "%s PSE = %s\n" % (label_list[i], str(xp[idx]*scale_factor)[:05])
		
		# Plot the results
		if i==2 or i==3:
			ax.plot(xp, pxp, '--', lw=4.5, color=colors[i], label=label_list[i])
			ax.plot(x, y, marker='o', color=colors[i], ms=18, lw=0, mfc='none', mew=2.4, mec=colors[i], alpha=.3)
			simPSE.append(xp[idx]/scale_factor)
			
		else:
			ax.plot(xp, pxp, '-', lw=6, color=colors[i], alpha=.39, label=label_list[i])
			ax.plot(x, y, marker='o', color=colors[i], ms=9, lw=0)
			empPSE.append(xp[idx]/scale_factor)
			
		i+=1
		
	ssPSE=[empPSE, simPSE]
	ax.set_ylim(0, 1.05)
	
	if task=='ssRe':
		ax.set_xlim(18, 45)
		ax.set_xlabel('SSD (ms)', fontsize=24)
		ax.set_xticks(np.arange(20, 45, 5))
		ax.set_xticklabels(np.arange(200, 450, 50), fontsize=18)
	else:
		ax.set_xlim(-1.5, 11.5)
		ax.set_xlabel('P(Go)', fontsize=24)
		ax.set_xticks(np.arange(0, 12, 2))
		ax.set_xticklabels(np.arange(0, 1.2, .20), fontsize=18)
		
	plt.setp(ax.get_yticklabels(), fontsize=18)	
	ax.set_ylabel('P(Stop)', fontsize=24, labelpad=14) 
	ax.legend(loc=0, fontsize=18)
	plt.savefig("evs_stopsigmoid.png", format='png', dpi=600)

	
	if showPSE:
		ssrtFig=plotPSE(ssPSE=ssPSE, task=task)


def plot_goRTs(sim_rt=None, task='ssRe'):
	
	plt.ion()
	sns.set(style='white', font="Helvetica")
	
	x=np.array([1,2])
	
	if task=='ssRe':
		emp_rt = np.array([0.565625, 0.573415])	
		emp_err = np.array([.0029, .0024])
		ylim=[.55, .580]
	elif task=='ssPro':
		emp_rt = np.array([.539154, .539344])
		emp_err = np.array([.001, .001])
		ylim=[.530, .540]
	if sim_rt is None:
		sim_rt = emp_rt
		

	f = plt.figure(figsize=(4, 5)) 
	ax = f.add_subplot(111)
	sns.despine()
	
	ax.bar(x, emp_rt, yerr=emp_err, align='center', color='MediumBlue', ecolor='k', alpha=.6)
	childrenLS=ax.get_children()
	barlist=filter(lambda x: isinstance(x, matplotlib.patches.Rectangle), childrenLS)
	barlist[1].set_color('#E60000')
	
	ax.plot(x[0], sim_rt[0], color='#1975FF', marker='o', ms=16, lw=0, alpha=.8, mew=2, mec='Navy')
	#ax.plot(x[1], sim_rt[1], color='#FF4D94', marker='o', ms=16, lw=0, alpha=.85)
	ax.plot(x[1], sim_rt[1], color='#E6005C', marker='o', ms=16, lw=0, alpha=.8, mew=2, mec='FireBrick')
	ax.set_ylim(ylim[0], ylim[1])	
	ax.set_yticks(np.arange(ylim[0], ylim[1]+.005, .005))
	ax.set_yticklabels(np.arange(ylim[0], ylim[1]+.005, .005), fontsize=14)
	ax.set_ylabel('Go RT (ms)', fontsize=18)
	
	
	ax.set_xlim(0.5, 2.5)
	ax.set_xticks(x)
	ax.set_xticklabels(['BSL', 'PNL'], fontsize=18)
	
def plotPSE(ssPSE=None, task='ssRe'):
	
	plt.ion()
	sns.set(style='white', font="Helvetica")
	#sns.set_style('white')
	

	if task=='ssRe':
		#emp_pse = np.array([.3491222879, .3614880328])	
		emp_err = np.array([0.002834, 0.0027982])
	else:
		#emp_pse=np.array([0.380, 0.387])
		emp_err=np.array([0.009, 0.009])

	if ssPSE is None:
		ssPSE = emp_pse
		
	f = plt.figure(figsize=(4, 5)) 
	ax = f.add_subplot(111)
	sns.despine()
	
	x=np.array([1,2])
	ax.bar(x, np.array(ssPSE[0]), align='center', color='MediumBlue', yerr=emp_err, ecolor='k', alpha=.6)
	childrenLS=ax.get_children()
	barlist=filter(lambda x: isinstance(x, matplotlib.patches.Rectangle), childrenLS)
	barlist[1].set_color('#E60000')
	
	#ax.plot(x, np.array(ssPSE[1]), color='k', marker='o', ms=10, lw=0)'#1975FF', '#E6005C'
	ax.plot(x[0], np.array(ssPSE[1])[0], color='#1975FF', marker='o', ms=16, lw=0, alpha=.8, mew=2, mec='Navy')
	#ax.plot(x[1], np.array(ssPSE[1])[1], color='#FF4D94', marker='o', ms=16, lw=0, alpha=.85)
	ax.plot(x[1], np.array(ssPSE[1])[1], color='#E6005C', marker='o', ms=16, lw=0, alpha=.8, mew=2, mec='FireBrick')
	ax.set_xticks(x)
	ax.set_xticklabels(['BSL', 'PNL'], fontsize=18)
	ax.set_xlim(0.5, 2.5)
	
	if task=='ssRe':
		ax.set_ylim(.335, .365)
		ax.set_yticks(np.arange(.335, .375, .01))
		ax.set_yticklabels(np.arange(.335, .375, .01), fontsize=14)
	else:
		ax.set_ylim(.35, .42)
		ax.set_yticks(np.arange(.35, .43, .01))
		ax.set_yticklabels(np.arange(.35, .43, .01), fontsize=14)
	
	ax.set_ylabel('PSE', fontsize=18)

	return f

def scurves(ysim=None, task='ssRe', showPSE=True, ncurves=5):
	
	plt.ion()
	sns.set(style='white', font="Helvetica")
	
	if task=='ssRe':
		x=np.array([400, 350, 300, 250, 200], dtype='float')
		xsim=np.linspace(15, 50, 10000)
		xxlim=(18, 45)
		xxlabel='SSD (ms)'
		xxticks=np.arange(20, 45, 5)
		xxticklabels=np.arange(200, 450, 50)
		scale_factor=100
	else:
		x=np.array([100, 80, 60, 40, 20, 0], dtype='float')
		xsim=np.linspace(-5, 10, 10000)
		xxlim=(-1.5, 11.5)
		xxlabel='P(Go)'
		xxticks=np.arange(0, 12, 2)
		xxticklabels=np.arange(0, 1.2, .20)
		scale_factor=10
	
	x=res(-x,lower=x[-1]/10, upper=x[0]/10)
	#ydata=res(ydata, lower=ydata[-1], upper=ydata[0])
	
	pse=[]
	
	f = plt.figure(figsize=(6,7.5)) 
	ax = f.add_subplot(111)
	sns.despine()
	
	#colors=['MediumBlue', '#E60000', '#1975FF', '#E6005C']#'#1975FF', '#FF0066']
	colors = sns.blend_palette(["#2C2C2C", "GhostWhite"], len(ysim))
	#label_list=['EmpBSL','EmpPNL', 'SimBSL', 'SimPNL']
	
	for yi in ysim:

		y=res(y, lower=yi[-1], upper=yi[0])
		
		#print(x)
		#print(y)
		
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
		print "PSE = %s\n" % (str(xp[idx]*scale_factor)[:05])
		
		# Plot the results
		ax.plot(xp, pxp, '-', lw=6, color=colors[i], alpha=.5, label=label_list[i])
		ax.plot(x, y, marker='o', color=colors[i], ms=10, lw=0)
		pse.append(xp[idx]/scale_factor)

	if task=='ssRe':
		ax.set_xlim(xxlim)
		ax.set_xlabel(xxlabel, fontsize=24)
		ax.set_xticks(xxticks)
		ax.set_xticklabels(xxticklabels, fontsize=18)
	
	ax.set_ylim(0, 1.05)	
	plt.setp(ax.get_yticklabels(), fontsize=18)	
	ax.set_ylabel('P(Stop)', fontsize=24, labelpad=14) 
	ax.legend(loc=0, fontsize=18)
	plt.savefig("evs_stopsigmoid.png", format='png', dpi=600)

	
	if showPSE:
		ssrtFig=plotPSE(ssPSE=ssPSE, task=task)

	return ax
