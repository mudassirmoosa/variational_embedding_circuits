#!/usr/bin/env python
# coding: utf-8

import pennylane as qml
from pennylane import numpy as np

# This function generates a QAOA type circuit. 

def embedding_circuit(x,weights,wires):
    
    no_qubits = len(wires)
        
    for params_layer in weights:
        for i in range(no_qubits):
            qml.RX(x,wires=wires[i])
        
        for i in range(no_qubits):
            qml.RY(params_layer[i],wires=wires[i])
                        
    for w in range(no_qubits):
        qml.RX(x,wires=wires[w])


# In[ ]:




