from __future__ import division
import os
import seaborn as sns
import pandas as pd
from radd import qsim, psy, utils, ft, simfx, ss
import scipy as scp
from lmfit import Parameters, Minimizer, minimize
import numpy as np
import matplotlib.pyplot as plt


def get_sx_params(sx, sxdf):

	params=sxdf['mean']; params.index=sxdf['param']
	a=params['a']*.1
	z=a*params['z']
	v=params['v(bsl)']*.1

	ssinits=get_ssinits()
	sp={'ssv':ssinits[sx], 'ssTer':0.0, 'ssTer_var':0.0, 'pGo':0.5}
	gp={'a':a, 'v':v, 'z':z, 'Ter':params['t'], 'eta':0.0, 'st':0.0, 'sz':0.0}

	return [gp, sp]


def raddMinFunc(p, gp, sp, ntrials, ydata=[]):

	pth=utils.find_path()+"CoAx/SS/"

	ssdlist=np.arange(.20, .45, .05)
	yhat=[]
	sp['ssv'] = -p['ssv'].value
	gp['v'] = p['v'].value
	gp['a'] = p['a'].value
	gp['z'] = p['z'].value
	gp['Ter'] =  p['Ter'].value

	for ssd in ssdlist:       

		sp['ssd']=ssd
		out=simulate(gp=gp, sp=sp, mfx=radd, ntrials=ntrials, timebound=.650, 
				depHyper=True, task='ssRe', condition_str='bsl') 
		yhat.append(out['stop_acc'])

	yhat=np.array(yhat)

	if ydata==[]:
		return yhat
	else:
		residuals=yhat-ydata
		
		#with open(pth+"fullRaDD/ReBSL/SxStatus.txt", "a") as myfile:
		#	sse="SSE: %s"%str(np.sum([i**2 for i in residuals]))
		#	myfile.write(sse + "\n")

		return residuals


def optRadd(data, paramdf, ntrials):

	pth=utils.find_path()+"CoAx/SS/fullRaDD/ReBSL/"

	fitdf=pd.DataFrame(columns=['subj_idx', 'params', 'ssd', 'residuals', 'ydata', 'yhat', 
			'chisq', 'redchi','sse', 'converged'], index=paramdf.subj_idx.unique())

	pb=utils.PBinJ(len(data.subj_idx.unique()), color="#4168B7"); sx_n=0

	for sx, sxdf in paramdf.groupby('subj_idx'):

		sx_n +=1

		#with open(pth+"SxStatus.txt", "a") as myfile:
		#	myfile.write('--'*80+"\n"+'Subject: '+str(sx)+"\n"+'--'*80+"\n")

		sxdata=data[data['subj_idx']==sx]
		ydata=np.array([ss.anl(sxdata)['stop_acc'] for ssd, sxdata in sxdata.groupby('ssd')][:-1])
		
		gp, sp = get_sx_params(sx, sxdf)

		p=Parameters()

		p.add('ssv', value=sp['ssv'], min=0.0, vary=True)
		p.add('a', value=gp['a'], min=0.0, vary=True)
		p.add('v', value=gp['v'], min=0.0, vary=True)
		p.add('Ter', value=gp['Ter'], min=0.0,max=.649, vary=True)
		p.add('z', value=gp['z'], min=0.0, vary=True)

		theta = Minimizer(raddMinFunc, p, fcn_args=(gp, sp, ntrials, ydata), method='Nelder-Mead', disp=True) 

		theta.fmin(maxfun=1000, ftol=1.e-2, xtol=1.e-2)
		
		if theta.nfev<=1000:
			converged=1
			print "Subject %s: Successfully converged" % str(sx)
		else:
			converged=0
			print "Subject %s: Failed to converge" % str(sx) 
		print "--------------------------------------"

		fitdf.loc[sx]=pd.Series({"subj_idx":sx,'params':{pk:p[pk].value for pk, pv in p.items()}, 
					'ssd':np.arange(.20, .45, .05), "residuals":theta.residual,
					"ydata":ydata.astype('float'),'yhat':ydata+theta.residual,
					"chisq":theta.chisqr,"redchi":theta.redchi, 
					'sse':np.sum([i**2 for i in theta.residual]), 'converged':converged})

		fitdf.to_csv(pth+"OptSxParams_"+str(sx)+".csv")
		
		pb.update_i(sx_n)


def radd(gp, sp, timebound=0.653, ss_trial=False, depHyper=True, s2=.01, tau=.0005, **kwargs):

	mu=gp['v']; TR=gp['Ter']; a=gp['a']; 
	z=gp['z']; ssv=sp['ssv']; ssd=sp['ssd']
	
	ss_started=False
	no_choice_yet=True
	
	if TR>ssd and ss_trial:
		t=ssd	# start the time at ssd

	else:		# or
		t=TR	 # start the time at TR


	if ss_trial: 
		ttype='stop'
	else: 
		ttype='go'
	
	dx=np.sqrt(s2*tau)  		# dx is the step size up or down.
	e=z		     		# starting point
	e_ss=z				#arbitrary (positive) init value
	
	while no_choice_yet: 
		
		# increment the time
		t = t + tau

		if t>=timebound and no_choice_yet:
			choice='stop'
			rt=timebound
			no_choice_yet=False
			break

		if t>=TR:

			r=np.random.random_sample()
			p=0.5*(1 + mu*dx/s2)
			
			# if r < p then move up
			if r < p:
				e = e + dx
			# else move down
			else:
				e = e - dx
			
			if e>=a and no_choice_yet:
				choice='go'
				rt=t
				#mu=0
				no_choice_yet=False

		if ss_trial and t>=ssd:

			r_ss=np.random.random_sample()
			p_ss=0.5*(1 + ssv*dx/s2)
			
			#test if stop signal has started yet.
			#if not, then start at current state
			#of "go/nogo" decision variable
			if not ss_started and depHyper:
				ss_started=True
				e_ss=e
				
			else:
				if r_ss < p_ss:
					e_ss = e_ss + dx
				else:
					e_ss = e_ss - dx

			if e_ss<=0 and no_choice_yet:
				choice='stop'
				rt=t
				#ssv=0
				no_choice_yet=False
	
	if choice==ttype: 
		acc=1.00
	else: 
		acc=0.00

	return {'rt':rt, 'choice':choice, 'trial_type':ttype,
		'acc':acc, 'ssd':sp['ssd'], 'pGo':sp['pGo']}



def simulate(gp=None, sp=None, mfx=radd, ntrials=20000, timebound=0.653, 
	task='ssRe', depHyper=True, condition_str=None):
	
	pStop=1-sp['pGo']
	columns=['rt', 'choice', 'acc', 'trial_type', 'ssd', 'pGo']
	df = pd.DataFrame(columns=columns, index=np.arange(0,ntrials))

	for i in range(ntrials):
		
		ss_trial=False
		
		if np.random.random_sample()<=pStop:
			ss_sstrial=True
					
		sim_out = mfx(gp, sp, depHyper=depHyper, 
			timebound=timebound, ss_trial=ss_trial)	
		
		df.loc[i]=pd.Series({c:sim_out[c] for c in df.columns})

	if condition_str:
		df['condition']=[condition_str]*len(df)
		
	df[['rt', 'acc']]=df[['rt', 'acc']].astype(float)

	return ss.anl(df)


def get_ssinits():

	ssinits={28: 0.73, 29: 0.71, 30: 1.92, 31: 0.6, 32: 0.82, 33: 0.67, 34: 0.61, 35: 0.43, 36: 1.23, 
		37: 0.55, 38: 1.8, 40: 1.75, 41: 1.62, 42: 0.73, 43: 0.6, 44: 1.41, 45: 1.21, 47: 0.52, 48: 0.74, 
		50: 1.54, 51: 0.7, 52: 0.53, 54: 0.54, 55: 0.6, 56: 0.51, 57: 0.63, 58: 1.2, 59: 1.2, 
		60: 0.69, 61: 1.69, 62: 0.6, 63: 1.66, 65: 1.2, 67: 1.39, 68: 0.76, 70: 0.75, 71: 0.84, 72: 0.75, 
		73: 0.54, 74: 0.75, 75: 0.62, 76: 1.02, 77: 0.89, 78: 0.98, 79: 1.93, 80: 1.38, 81: 0.88, 
		82: 1.21, 83: 0.72, 84: 0.89, 85: 1.3, 86: 1.03, 87: 0.86, 88: 1.36, 89: 1.07, 90: 0.5, 95: 1.41, 97: 0.65}
	
	return ssinits

def chisqg(ydata,ymod,sd=None):

      	"""  
	Returns the chi-square error statistic as the sum of squared errors between  
	Ydata(i) and Ymodel(i). If individual standard deviations (array sd) are supplied,   
	then the chi-square error statistic is computed as the sum of squared errors  
	divided by the standard deviations.     Inspired on the IDL procedure linfit.pro.  
	See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.  
	  
	x,y,sd assumed to be Numpy arrays. a,b scalars.  
	Returns the float chisq with the chi-square statistic.  
	  
	Rodrigo Nemmen  
	http://goo.gl/8S1Oo  
      	"""  
      	# Chi-square statistic (Bevington, eq. 6.9)  
	if sd==None:  
		chisq=np.sum((ydata-ymod)**2)  
	else:  
		chisq=np.sum( ((ydata-ymod)/sd)**2 )  

	return chisq 