import matplotlib.pyplot as plt
#import matplotlib.axes as axes
import numpy as np
import pandas

def plot_axes_IdVsCost():
		#axes.Axis.set_axisbelow(True)
	x = np.array([1,2,3,4,5,6,7,8])
	my_xticks = ['1','2','3','4','5','6','7','8']
	plt.xticks(x, my_xticks)
	# for L=1,Nq=1,d=1
	# for L=1,Nq=2,d=1
	# for L=1,Nq=3,d=1
	# for L=1,Nq=4,d=1
	y = np.array([0.207044,np.nan,np.nan,0.206619,np.nan,np.nan,np.nan,np.nan])
	plt.scatter(x, y, marker='^',color='blue',label='L=1,Nq=4,d=1')
	# for l=2,Nq=1,d=1
	# for l=2,Nq=2,d=1
	# for l=2,Nq=3,d=1
	y = np.array([0.376935,np.nan,0.326575,0.182479,np.nan,np.nan,np.nan,np.nan])
	plt.scatter(x, y, marker='o',color='red',label='L=2,Nq=3,d=1')
	# for l=2,Nq=4,d=1
	y = np.array([0.400412,np.nan,np.nan,np.nan,0.593843,0.722007,np.nan,np.nan])
	plt.scatter(x, y, marker='o',color='blue',label='L=2,Nq=4,d=1')
	# for l=3
	# for l=4,Nq=1,d=1
	y = np.array([0.116092,0.103657,0.312526,np.nan,np.nan,np.nan,np.nan,np.nan])
	plt.scatter(x, y, marker='s',color='purple',label='L=4,Nq=1,d=1')
	# for l=4,Nq=2,d=1
	y = np.array([np.nan,np.nan,0.375075,0.325434,np.nan,np.nan,0.398591,0.660803])
	plt.scatter(x, y, marker='s',color='green',label='L=4,Nq=2,d=1')
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


def plot_axes_LayersVsCost():
	#ID == marker, Nq==color, d==filled/nonfilled
	#x = np.array([1,2,3,4])
	x = np.array([1,2,3,4])
	#my_xticks = ['1','2','3','4']
	#plt.xticks(x, my_xticks)
	# for ID=1,Nq=1,d=1
	y = np.array([np.nan,np.nan,np.nan,0.116092])
	plt.scatter(x, y, marker='^',color='blue',label='ID=1,Nq=4,d=1')
	# for ID=1,Nq=2,d=1
	# for ID=1,Nq=3,d=1
	y = np.array([np.nan,0.376935,np.nan,np.nan])
	plt.scatter(x, y, marker='^',color='red',label='ID=1,Nq=4,d=1')
	# for ID=1,Nq=4,d=1
	y = np.array([0.207044,0.400412,np.nan,np.nan])
	plt.scatter(x, y, marker='^',color='blue',label='ID=1,Nq=4,d=1')
	#___________________________________ID=2________________________________
	# for ID=2,Nq=1,d=1
	y = np.array([np.nan,np.nan,np.nan,0.103657])
	plt.scatter(x, y, marker='o',color='blue',label='ID=2,Nq=4,d=1')
	# for ID=2,Nq=2,d=1
	# for ID=2,Nq=3,d=1
	# for ID=2,Nq=4,d=1
	#___________________________________ID=3________________________________
	# for ID=3,Nq=1,d=1
	y = np.array([np.nan,np.nan,np.nan,0.312526])
	plt.scatter(x, y, marker='s',color='blue',label='ID=3,Nq=4,d=1')
	# for ID=3,Nq=2,d=1
	y = np.array([np.nan,np.nan,np.nan,0.375075])
	plt.scatter(x, y, marker='s',color='green',label='ID=3,Nq=4,d=1')
	# for ID=3,Nq=3,d=1
	y = np.array([np.nan,0.326575,np.nan,np.nan])
	plt.scatter(x, y, marker='s',color='red',label='ID=3,Nq=4,d=1')

	# for l=2,Nq=2,d=1
	# for l=2,Nq=3,d=1
	y = np.array([0.376935,np.nan,0.326575,0.182479,np.nan,np.nan,np.nan,np.nan])
	plt.scatter(x, y, marker='o',color='red',label='L=2,Nq=3,d=1')
	# for l=2,Nq=4,d=1
	y = np.array([0.400412,np.nan,np.nan,np.nan,0.593843,0.722007,np.nan,np.nan])
	plt.scatter(x, y, marker='o',color='blue',label='L=2,Nq=4,d=1')
	# for l=3
	# for l=4,Nq=1,d=1
	y = np.array([0.116092,0.103657,0.312526,np.nan,np.nan,np.nan,np.nan,np.nan])
	plt.scatter(x, y, marker='s',color='purple',label='L=4,Nq=1,d=1')
	# for l=4,Nq=2,d=1
	y = np.array([np.nan,np.nan,0.375075,0.325434,np.nan,np.nan,0.398591,0.660803])
	plt.scatter(x, y, marker='s',color='green',label='L=4,Nq=2,d=1')
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

#plot_axes_IdVsCost()


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
data[3,1,0,2]=0.375075
data[3,1,0,3]=0.325434
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




print(data)







