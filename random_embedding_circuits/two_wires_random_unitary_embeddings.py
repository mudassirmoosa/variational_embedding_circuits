#!/usr/bin/env python

import pennylane as qml
from pennylane import numpy as np


# The function `random_gate_sequence` generates a sequence of rotation operators by sampling uniformaly from $\{ RX,RY,RZ\}$. It takes as an input **num_wires** and **num_layers** as an `int` and returns an array of shape (num_layer, 2$*$num_wires).

def random_gate_sequence(num_wires,num_layers):
    
    gate_set = [qml.RX, qml.RY, qml.RZ]
    
    gate_sequence = []
    
    for i in range(num_layers):
        gate = [np.random.choice(gate_set) for i in range(2*num_wires)]
        gate_sequence.append(gate)
    
    return gate_sequence


# The function `random_embedding_circuit` generates a variational circuit that embeds the data $x$ into a quantum state $|x>$. This circuit will act of $N$ wires and it will consist of $L$ layers. In addition to the data input $x$, this function will take the following inputs:
# 
# 1. **weights**: an array of shape $(L,N)$. 
# 2. **wires**: a list of size $N$. This will determine which wires the circuit acts on.
# 3. **gate_sequence**: an array of shape $(L,2N)$. This will determine what gates are part of the embedding circuit.
# 
# Note that each layer will consists of a random Pauli rotation by amount $x$ on each wire followed by a controlled-Z gate between wires $i$ and $i+1$ for $i = \{1,2,...,N-1\}$ followed by another round of random Pauli rotations by amount $\theta \in $ **weights**. In addition to $L$ layers of this form, the circuit will start off with a $RY(\pi/4)$ acting on each wire and it will end with a $RX(x)$ acting on each wire. This circuit is inspired by [Mclean et al; 2018](https://arxiv.org/pdf/1803.11173.pdf).



def random_embedding_circuit(x,weights,wires,gate_sequence):
    
    no_qubits = len(wires)
    
    for w in wires:
        qml.RY(np.pi/4,wires=w)
        
    for params_layer, gates_layer in zip(weights,gate_sequence):
        for i in range(no_qubits):
            gates_layer[i](x,wires=wires[i])
        
        for i in range(no_qubits - 1):
            qml.CZ(wires=wires[i:i+2])
        
        for i in range(no_qubits):
            gates_layer[no_qubits+i](params_layer[i],wires=wires[i])
            
        for i in range(no_qubits - 1):
            qml.CZ(wires=wires[i:i+2])
            
    for w in wires:
        qml.RX(x,wires=w)



