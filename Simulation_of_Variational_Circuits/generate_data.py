import numpy as np

def generate_data(finename_X, finename_Y, number_of_data):
	X = []
	Y = []
	for i in range(40):
	    x =[]
	    y = 0
	    r1 = np.random.uniform(0,1)
	    if r1 < 0.5:
	        x.append(np.random.uniform(-1.0,1.0))
	        x.append(np.random.uniform(-1.0,1.0))
	        y = -1.0
	    else:
	        r2 = np.random.uniform(0,1)
	        y = 1.0
	        if r2 < 0.5:
	            x.append(np.random.uniform(-2.0,-1.0))
	            x.append(np.random.uniform(-2.0,-1.0))
	        else:
	            x.append(np.random.uniform(1.0,2.0))
	            x.append(np.random.uniform(1.0,2.0))
	    X.append(x)
	    Y.append(y)
	    
	    np.savetxt('./data/{}'.format(finename_X), X)
	    np.savetxt('./data/{}'.format(finename_Y), Y)
