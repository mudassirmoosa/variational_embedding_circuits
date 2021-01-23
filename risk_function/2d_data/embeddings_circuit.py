import pennylane as qml
from pennylane import numpy as np


# This function generates a 2-dimensional data set.
# The argument 'size' determines the number of 
# data points generated.
# The argument 'margin' determines the gap between
# the data points of different classes.
# For i = 1,2, we transform the x_i to x_i + margin * sign(x_i).

def generate_data(size,margin):
    X = []
    Y = []
    for i in range(size):
        x1 = 2.0*np.random.random() - 1.0
        x2 = 2.0*np.random.random() - 1.0
        y = x1*x2
        
                
        if x1 < 0:
            x1 = x1 - margin
        else:
            x1 = x1 + margin
        if x2 < 0:
            x2 = x2 - margin
        else:
            x2 = x2 + margin
        
        X.append([x1,x2])
        if y>0:
            Y.append(1)
        else: 
            Y.append(-1)
            
    return X, Y



def embedding_circuit(x,weights,wires):
    
    no_qubits = len(wires)
    no_layers = len(weights)
    
    for l in range(no_layers):
        p = l%2
        xp = x[p]
        params = weights[l]
        
        for i in range(no_qubits):
            qml.RX(xp,wires=wires[i])
        
        for i in range(no_qubits):
            qml.RY(params[i],wires=wires[i])
    
    if no_layers%2 == 0:
        xp = x[0]*x[1]
    elif no_layers%2 == 1:
        xp = x[1]
    for w in range(no_qubits):
        qml.RX(xp,wires=wires[w])


# In[ ]:




