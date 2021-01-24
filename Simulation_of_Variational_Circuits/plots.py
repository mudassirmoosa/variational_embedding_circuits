import matplotlib.pyplot as plt
#import matplotlib.axes as axes
import numpy as np
import pandas

def plot_axes_IdVsCost():
		#axes.Axis.set_axisbelow(True)
	x = range(8)
	# for L=1,Nq=1,d=1
	# for L=1,Nq=2,d=1
	# for L=1,Nq=3,d=1
	# for L=1,Nq=4,d=1
	y = np.array([0.207044,np.nan,np.nan,0.206619,np.nan,np.nan,np.nan,np.nan])
	plt.scatter(x, y, marker='^',facecolors='blue',edgecolors='blue',label='L=1,Nq=4,d=1')
	# for l=2,Nq=1,d=1
	# for l=2,Nq=2,d=1
	# for l=2,Nq=3,d=1
	y = np.array([0.376935,np.nan,0.326575,0.182479,np.nan,np.nan,np.nan,np.nan])
	plt.scatter(x, y, marker='o',facecolors='red',edgecolors='red',label='L=2,Nq=3,d=1')
	# for l=2,Nq=4,d=1
	y = np.array([0.400412,np.nan,np.nan,np.nan,0.593843,0.722007,np.nan,np.nan])
	plt.scatter(x, y, marker='o',facecolors='blue',edgecolors='blue',label='L=2,Nq=4,d=1')
	# for l=3
	# for l=4,Nq=1,d=1
	y = np.array([0.116092,0.103657,0.312526,np.nan,np.nan,np.nan,np.nan,np.nan])
	plt.scatter(x, y, marker='s',facecolors='purple',edgecolors='purple',label='L=4,Nq=1,d=1')
	# for l=4,Nq=2,d=1
	y = np.array([np.nan,np.nan,0.375075,0.325434,np.nan,np.nan,0.398591,0.660803])
	plt.scatter(x, y, marker='s',facecolors='green',edgecolors='green',label='L=4,Nq=2,d=1')
	# for l=4,Nq=3,d=1
	# for l=4,Nq=4,d=1

	# for l=1,Nq=1..3,d=2
	# for l=1,Nq=4,d=2
	y = np.array([np.nan,np.nan,np.nan,np.nan,0.748411,np.nan,np.nan,np.nan])
	plt.scatter(x, y,marker='^',facecolors='none',edgecolors='blue',label='L=1,Nq=4,d=2')
	# for l=2,Nq=1,d=2
	# for l=2,Nq=2,d=2
	y = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,0.270515,0.92881])
	plt.scatter(x, y,marker='o',facecolors='none',edgecolors='green',label='L=2,Nq=2,d=2')
	# for l=2,Nq=3,d=2
	# for l=2,Nq=4,d=2
	y = np.array([np.nan,np.nan,np.nan,np.nan,0.719350,np.nan,np.nan,0.568995])
	plt.scatter(x, y,marker='o',facecolors='none',edgecolors='blue',label='L=2,Nq=4,d=2')
	# for l=3
	# for l=4,Nq=2,d=2
	y = np.array([0.482175,np.nan,np.nan,np.nan,np.nan,np.nan,0.469099,0.398838])
	plt.scatter(x, y,marker='s',facecolors='none',edgecolors='green',label='L=4,Nq=2,d=2')

	plt.grid(b=True, which='both', color='#666666', linestyle='--')

	plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left')
	plt.xlabel("Circuit ID", fontsize=13)
	plt.ylabel("Cost Function After 300 Steps", fontsize=13)
	plt.show()



#L,Nq,d,ID
data = np.full((4,4,2,8), np.nan)
data[0,3,0,0]=0.207044
data[0,3,0,3]=0.206619
data[1,2,0,0]=0.376935
data[1,2,0,2]=0.326575
data[1,2,0,3]=0.182479
data[1,3,0,0]=0.400412
data[1,3,0,4]=0.593843
data[1,3,0,5]=0.722007
data[3,0,0,0]=0.116092
data[3,0,0,1]=0.103657
data[3,0,0,2]=0.312526


data[0,1,0,0]=np.nan
data[0,1,0,1]=np.nan
data[0,1,0,2]=np.nan
data[0,1,0,3]=np.nan
data[0,1,0,4]=np.nan
data[0,1,0,5]=np.nan
data[0,1,0,6]=np.nan
data[0,1,0,7]=np.nan

data[1,1,0,0]=np.nan
data[1,1,0,1]=np.nan
data[1,1,0,2]=np.nan
data[1,1,0,3]=np.nan
data[1,1,0,4]=np.nan
data[1,1,0,5]=np.nan
data[1,1,0,6]=np.nan
data[1,1,0,7]=np.nan

data[3,1,0,0]=np.nan
data[3,1,0,1]=np.nan
data[3,1,0,2]=0.375075
data[3,1,0,3]=0.325434
data[3,1,0,4]=np.nan
data[3,1,0,5]=np.nan
data[3,1,0,6]=0.398591
data[3,1,0,7]=0.660803

data[0,3,1,4]=0.748411

data[1,1,1,6]=0.270515
data[1,1,1,7]=0.92881

data[1,3,1,4]=0.719350
data[1,3,1,7]=0.568995

data[3,1,1,0]=0.482175
data[3,1,1,6]=0.469099
data[3,1,1,7]=0.398838

def define_plot_legend(x_Axis,L=None,Nq=None,data_length=None,ID=None):
	#ID == marker, Nq==color, d==filled/nonfilled
	if x_Axis == 'ID':
		if L==0:
			marker='^'
		elif L==1:
			marker='o'
		elif L==2:
			marker='+'
		else:
			marker='s'
		if Nq==0:
			edgecolors='purple'
		elif Nq==1:
			edgecolors='green'
		elif Nq==2:
			edgecolors='red'
		else:
			edgecolors='blue'
		if data_length==0:
			facecolors=edgecolors
		else:
			facecolors='none'
	elif x_Axis=='L':
		if ID==0:
			marker='^'
		elif ID==1:
			marker='o'
		elif ID==2:
			marker='+'
		elif ID==3:
			marker='s'
		elif ID==4:
			marker="D"
		elif ID==5:
			marker="p"
		elif ID==6:
			marker="d"
		elif ID==7:
			marker="v"
		if Nq==0:
			edgecolors='purple'
		elif Nq==1:
			edgecolors='green'
		elif Nq==2:
			edgecolors='red'
		else:
			edgecolors='blue'
		if data_length==0:
			facecolors=edgecolors
		else:
			facecolors='none'
	elif x_Axis=='Nq':
		if ID==0:
			marker='^'
		elif ID==1:
			marker='o'
		elif ID==2:
			marker='+'
		elif ID==3:
			marker='s'
		elif ID==4:
			marker="D"
		elif ID==5:
			marker="p"
		elif ID==6:
			marker="d"
		elif ID==7:
			marker="v"
		if L==0:
			edgecolors='purple'
		elif L==1:
			edgecolors='green'
		elif L==2:
			edgecolors='red'
		else:
			edgecolors='blue'
		if data_length==0:
			facecolors=edgecolors
		else:
			facecolors='none'
	else:
		print("Incorrect label for X axis")
		return
	return marker,facecolors,edgecolors



		



def plotting(data,x_Axis,draw_line=False):
	L = len(data)
	Nq = len(data[0])
	data_length = len(data[0][0])
	ID = len(data[0][0][0])
	print(L,Nq,data_length,ID)
	if x_Axis == "ID":
		for l in range(L):
			for n in range(Nq):
				for d in range(data_length):
					if np.isnan(data[l,n,d,:]).all():
						continue
					marker,facecolors,edgecolors=define_plot_legend(x_Axis=x_Axis,L=l,Nq=n,data_length=d,ID=None)
					x = np.arange(1,ID+1)
					plt.scatter(x, data[l,n,d,:],s = 70,marker=marker,facecolors=facecolors,edgecolors=edgecolors,\
						label='L={},Nq={},d={}'.format(l+1,n+1,d+1))
					if draw_line:
						datamask = np.isfinite(data[l,n,d,:].astype(np.double))
						plt.plot(x, data[l,n,d,datamask], color=edgecolors)
					plt.xlabel("Circuit ID", fontsize=13)
	elif x_Axis == "L":
		for id_ in range(ID):
			for n in range(Nq):
				for d in range(data_length):
					#n=1
					if np.isnan(data[:,n,d,id_]).all():
						continue
					marker,facecolors,edgecolors=define_plot_legend(x_Axis=x_Axis,Nq=n,data_length=d,ID=id_)
					x = np.arange(L-1)
					my_xticks = ['1','2','4']
					data_ = np.array([data[0,n,d,id_],data[1,n,d,id_],data[3,n,d,id_]]).astype(np.double)
					plt.xticks(x, my_xticks)
					plt.scatter(x, data_,s = 70, marker=marker,\
						facecolors=facecolors,edgecolors=edgecolors,\
						label='ID={},Nq={},d={}'.format(id_+1,n+1,d+1))
					if draw_line:
						datamask = np.isfinite(data_)
						plt.plot(x[datamask], data_[datamask], color=edgecolors)
					#for i in range(L): plt.annotate(np.around(data[i,n,d,id_], 8), (x[i], data[i,n,d,id_]))
					plt.xlabel("Number of Layers", fontsize=13)
	elif x_Axis == "Nq":
		for id_ in range(ID):
			for l in range(L):
				for d in range(data_length):
					if np.isnan(data[l,:,d,id_]).all():
						continue
					marker,facecolors,edgecolors=define_plot_legend(x_Axis=x_Axis,L=l,data_length=d,ID=id_)
					x = np.arange(Nq)
					my_xticks = ['1','2','3','4']
					data_ = data[l,:,d,id_].astype(np.double)
					plt.xticks(x, my_xticks)
					plt.scatter(x, data_,marker=marker,\
						facecolors=facecolors,edgecolors=edgecolors,\
						label='ID={},L={},d={}'.format(id_+1,l+1,d+1))
					if draw_line:
						datamask = np.isfinite(data_)
						plt.plot(x[datamask], data_[datamask], color=edgecolors)
					#for i in range(L): plt.annotate(np.around(data[i,n,d,id_], 8), (x[i], data[i,n,d,id_]))
					plt.xlabel("Number of Layers", fontsize=13)
					
	plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left')
	plt.ylabel("Cost Function After 300 Steps", fontsize=13)				
	plt.grid(b=True, which='both', color='#666666', linestyle='--')				
	plt.show()

#plotting(data)


plotting(data,x_Axis='ID')
#print(data[1,3,0,:])

	








