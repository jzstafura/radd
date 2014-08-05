def scurves(ysim=None, task='ssRe', pstop=.5, showPSE=True, ncurves=5, labels=None, predict_brain=False):
	
	plt.ion()
	if predict_brain:
		ysim=[ysim, ysim]
		#xsim=np.linspace(15, 50, 10000)
		#scale_factor=100
	sns.set(style='white', font="Helvetica")
	npoints=len(ysim[0])
	xsim=np.linspace(-5, 120, 10000)
	scale_factor=10
	
	#if predict_brain:
	#	ysim=[ysim, ysim]
		#xsim=np.linspace(15, 50, 10000)
		#scale_factor=100
	
	x=np.array(np.linspace(10, 100, npoints), dtype='float')
	#xsim=np.linspace(-5, 120, 10000)
	xxticks=x/scale_factor
	xxticklabels=x/100
	xxlim=(0, 10.5)
	xxlabel='P(Go)'
	
	if task=='ssRe':
		xxlabel='SSD'
		xxticks=x/scale_factor
		xxticklabels=np.arange(250, 550, 50)
	
	x=res(-x,lower=x[-1]/10, upper=x[0]/10)
	
	pse=[]

	if 'Re' in task:
		
		f=plt.figure(figsize=(16, 10.5))
		ax = plt.subplot2grid((4, 2), (0, 0), colspan=1, rowspan=4)
		ax2 = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=4)

	else:
		f = plt.figure(figsize=(8,9.5)) 
		f.subplots_adjust(top=0.95, wspace=0.12, left=0.19, right=0.98, bottom=0.15)	
		ax = f.add_subplot(111)


	sns.despine()
	
	colors = sns.blend_palette(["LimeGreen", "Navy"], len(ysim))
	
	if labels!=None:
		labels=labels
	else:
		labels=['C'+str(i) for i in range(len(ysim))]
	
	for i, yi in enumerate(ysim):

		y=res(yi, lower=yi[-1], upper=yi[0])
		#print "y = ", y
		#print "x = ", x
		#print "xsim = ", xsim
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
		ax.plot(xp, pxp, '-', lw=7.5, color=colors[i], alpha=.6, label=labels[i])
		#ax.plot(x, y, marker='o', color=colors[i], ms=10, lw=0)
		
		pse.append(xp[idx]/scale_factor)

	if 'Re' in task:
		
		ax.vlines(x=xxticks[-2], ymin=0, ymax=1, lw=20, color='k', alpha=.2)
		
		p = FancyArrow(0.424, 0.55, 0.085, 0.0, width=0.005,
			length_includes_head=False, lw=3.5, fc='DarkGray', ec='DarkGray',
			head_width=.02, head_length=.02, 
			shape='full', overhang=0, 
			head_starts_at_zero=False, 
			transform=ax.figure.transFigure, 
			clip_on=False) 

		ax.add_patch(p) 

		pbsl_p=[0.92266667,  0.745, 0.46633333, 0.231, 0.015, 0.00233333]
		ppnl_p=[0.93533333, 0.752, 0.501, 0.22033333, 0.018, 0.00266667]
		
		pbsl=np.array(pbsl_p, dtype='float')
		ppnl=np.array(ppnl_p, dtype='float')
		
		ax2 = fit_scurves(ysim=[pbsl, ppnl], task='ssPro', ax=ax2)

		plt.tight_layout(h_pad=3, w_pad=3)
		
		
	ax.set_xlim(xxlim)
	ax.set_xlabel(xxlabel+' (ms)', fontsize=34)
	ax.set_xticks(xxticks)
	ax.set_xticklabels(xxticklabels, fontsize=26)
	ax.set_ylim(0, 1.05)	
	
	plt.setp(ax.get_yticklabels(), fontsize=26)	
	ax.set_ylabel('P(Inhibit)', fontsize=34, labelpad=11) 
	ax.legend(loc=0, fontsize=26)
	
	#yy, locs = ax.yticks()
	#ll = ['%.2f' % a for a in yy]
	#ax.yticks(yy, ll)

	#ax2.set_yticklabels([])
	#ax2.set_ylabel([])
	plt.setp(ax2.get_yticklabels(), visible=False)
	ax2.set_ylabel("", fontsize=0)
	
	if os.path.isdir("/Users/kyle"):
		#plt.savefig("/Users/kyle/ReProFactorial_SCurves%s%s" % (task, ".png"), format='png', dpi=600)
		f.savefig("/Users/kyle/Dropbox/CoAx/SS/FactorialSims/ReProFactorial_SCurves_%s%s" % (task, ".svg"), rasterized=True, dpi=600)	
		plt.savefig("/Users/kyle/Dropbox/CoAx/SS/FactorialSims/ReProFactorial_SCurves_%s%s" % (task, ".png"), format='png', dpi=600)	
	elif os.path.isdir("/home/kyle"):
		f.savefig("/home/kyle/Dropbox/CoAx/ss/simdata/ReProFactorial_SCurves%s%s" % (task, ".svg"), rasterized=True, dpi=600)	
		plt.savefig("/home/kyle/Dropbox/CoAx/ss/simdata/ReProFactorial_SCurves%s%s" % (task, ".png"), format='png', dpi=600)

	pse=pse
	
	return pse