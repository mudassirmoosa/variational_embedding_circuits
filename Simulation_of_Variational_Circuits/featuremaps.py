"""
Feature maps
************

This module contains feature maps. Each feature map function
takes an input vector x and weights, and constructs a circuit that maps
these two to a quantum state. The feature map function can be called in a qnode.

A feature map has the following positional arguments: weights, x, wires. It can have optional
keyword arguments.

Each feature map comes with a function that generates initial parameters
for that particular feature map.
"""
import numpy as np
import pennylane as qml


def _entanglerZ(w_, w1, w2):
    qml.CNOT(wires=[w2, w1])
    qml.RZ(2*w_, wires=w1)
    qml.CNOT(wires=[w2, w1])



def qaoa(weights, x, wires, n_layers=1, circuit_ID = 1):
    """
    1-d Ising-coupling QAOA feature map, according to arXiv1812.11075.

    Example one layer, 4 wires, 2 inputs:

       |0> - R_x(x1) - |^| -------- |_| - R_y(w7)  -
       |0> - R_x(x2) - |_|-|^| ---------- R_y(w8)  -
       |0> - ___H___ ------|_|-|^| ------ R_y(w9)  -
       |0> - ___H___ ----------|_| -|^| - R_y(w10) -

    After the last layer, another block of R_x(x_i) rotations is applied.

    :param weights: trainable weights of shape 2*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    """
    n_wires = len(wires)

    if n_wires == 1:
        n_weights_needed = n_layers
    elif n_wires == 2:
        n_weights_needed = 3 * n_layers
    else:
        n_weights_needed = 2 * n_wires * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(n_wires):
            # Either feed in feature
            if i < len(x):
                if circuit_ID == 1:
                    qml.RX(x[i], wires=wires[i])
                elif circuit_ID == 2:
                    qml.RY(x[i], wires=wires[i])
            # or a Hadamard
            else:
                qml.Hadamard(wires=wires[i])

        # 1-d nearest neighbour coupling
        if n_wires == 1:
            if circuit_ID == 1:
                qml.RY(weights[l], wires=wires[0])
            elif circuit_ID == 2:
                qml.RX(weights[l], wires=wires[0])
            
        elif n_wires == 2:
            _entanglerZ(weights[l * 3 + 2], wires[0], wires[1])
            # local fields
            for i in range(n_wires):
                if circuit_ID == 1:
                    qml.RY(weights[l * 3 + i], wires=wires[i])
                elif circuit_ID == 2:
                    qml.RX(weights[l * 3 + i], wires=wires[i])
        else:
            for i in range(n_wires):
                if i < n_wires-1:
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[i + 1])
                else:
                    # enforce periodic boundary condition
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[0])
            # local fields
            for i in range(n_wires):
                if circuit_ID == 1:
                    qml.RY(weights[l * 2 * n_wires + n_wires + i], wires=wires[i])
                elif circuit_ID == 2:
                    qml.RX(weights[l * 2 * n_wires + n_wires + i], wires=wires[i])

    # repeat feature encoding once more at the end
    for i in range(n_wires):
        # Either feed in feature
        if i < len(x):
            if circuit_ID == 1:
                qml.RX(x[i], wires=wires[i])
            elif circuit_ID == 2:
                qml.RY(x[i], wires=wires[i])
        # or a Hadamard
        else:
            qml.Hadamard(wires=wires[i])



def pars_qaoa(n_wires, n_layers=1):
    """
    Initial weight generator for 1-d qaoa feature map
    :param n_wires: number of wires
    :param n_layers: number of layers
    :return: array of weights
    """
    if n_wires == 1:
        return 0.001*np.ones(n_layers)
    elif n_wires == 2:
        return 0.001 * np.ones(n_layers * 3)
    elif n_wires == 4:
        return 0.001 * np.ones(n_wires * n_layers * 2)
    return 0.001*np.ones(n_layers * n_wires * 2)




def shallow_circuit(weights, x, wires, n_layers=1,circuit_ID=1):
    """
    Circuits are designed based on paper arXiv:1905.10876.

    Example one layer, 4 wires, 2 inputs:

       |0> - R_x(x1) - |^| -------- |_| - R_y(w5)  -
       |0> - R_x(x2) - |_|-|^| ---------- R_y(w6)  -
       |0> - ___H___ ------|_|-|^| ------ R_y(w7)  -
       |0> - ___H___ ----------|_| -|^| - R_y(w8) -

    After the last layer, another block of R_x(x_i) rotations is applied.

    :param weights: trainable weights of shape 2*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    :param circuit_ID: the ID of the circuit based on 
    """
    n_wires = len(wires)

    if n_wires == 1:
        n_weights_needed = n_layers
    elif n_wires == 2:
        n_weights_needed = 3 * n_layers
    else:
        n_weights_needed = 2 * n_wires * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(n_wires):
            # Either feed in feature
            if i < len(x):
                if circuit_ID == 18 or circuit_ID == 19:
                    qml.RX(x[i], wires=wires[i])

                elif circuit_ID == 11 or circuit_ID == 12:
                    qml.RY(x[i], wires=wires[i])
                else:
                    raise ValueError("Wrong circuit_ID: It should be between 1-19, got {}.".format(circuit_ID))
            else:
                qml.Hadamard(wires=wires[i])

        # 1-d nearest neighbour coupling
        if n_wires == 1:
            if circuit_ID == 18 or circuit_ID == 19:
                qml.RZ(weights[l], wires=wires[0])
            
        elif n_wires == 2:
            # local fields
            for i in range(n_wires):
                if circuit_ID == 18 or circuit_ID == 19:
                    qml.RZ(weights[l * 3 + i], wires=wires[i])
                else:
                    raise ValueError("Wrong circuit_ID: It should be between 1-19, got {}.".format(circuit_ID))
            if circuit_ID == 18:
                qml.CRZ(weights[l * 3 + 2], wires=[wires[1], wires[0]])
            elif circuit_ID == 19:
                qml.CRX(weights[l * 3 + 2], wires=[wires[1], wires[0]])
        else:
            # local fields
            for i in range(n_wires):
                if circuit_ID == 18 or circuit_ID == 19:
                    qml.RZ(weights[l * 2 * n_wires + i], wires=wires[i])

            for i in range(n_wires):
                if i == 0:
                    if  circuit_ID == 18:
                        qml.CRZ(weights[l * 2 * n_wires + n_wires + i], wires=[wires[n_wires-1], wires[0]])
                    elif  circuit_ID == 19:
                        qml.CRX(weights[l * 2 * n_wires + n_wires + i], wires=[wires[n_wires-1], wires[0]])
                elif i < n_wires-1:
                    if  circuit_ID == 18:
                        qml.CRZ(weights[l * 2 * n_wires + n_wires + i], wires=[wires[i], wires[i + 1]])
                    elif  circuit_ID == 19:
                        qml.CRX(weights[l * 2 * n_wires + n_wires + i], wires=[wires[i], wires[i + 1]])

    # repeat feature encoding once more at the end
    for i in range(n_wires):
        # Either feed in feature
        if i < len(x):
            if circuit_ID == 18 or circuit_ID == 19:
                qml.RX(x[i], wires=wires[i])
        # or a Hadamard
        else:
            qml.Hadamard(wires=wires[i])



def HVA_XXZ(weights, x, wires, n_layers=1):
    """
    1-d Ising-coupling QAOA feature map, according to arXiv1812.11075.

    :param weights: trainable weights of shape 2*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    """
    n_wires = len(wires)

    if n_wires == 1:
        n_weights_needed = n_layers
    elif n_wires == 2:
        n_weights_needed = 3 * n_layers
    else:
        n_weights_needed = 2 * n_wires * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(n_wires):
            # Either feed in feature
            if i < len(x):
                qml.RX(x[i], wires=wires[i])
            # or a Hadamard
            else:
                qml.Hadamard(wires=wires[i])

        # 1-d nearest neighbour coupling
        if n_wires == 1:
            qml.RY(weights[l], wires=wires[0])
        elif n_wires == 2:
            _entanglerZ(weights[l * 3 + 2], wires[0], wires[1])
            # local fields
            for i in range(n_wires):
                qml.RY(weights[l * 3 + i], wires=wires[i])
        else:
            for i in range(n_wires):
                if i < n_wires-1:
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[i + 1])
                else:
                    # enforce periodic boundary condition
                    _entanglerZ(weights[l * 2 * n_wires + i], wires[i], wires[0])
            # local fields
            for i in range(n_wires):
                qml.RY(weights[l * 2 * n_wires + n_wires + i], wires=wires[i])

    # repeat feature encoding once more at the end
    for i in range(n_wires):
        # Either feed in feature
        if i < len(x):
            qml.RX(x[i], wires=wires[i])
        # or a Hadamard
        else:
            qml.Hadamard(wires=wires[i])

def HVA_TFIM_2D_data(weights, x, wires, n_layers=1, types = 1):
    """
    1-d Ising-coupling HVA_TFIM feature map, according to 2008.02941v2.11075.

    :param weights: trainable weights of shape 2*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    """
    wires = range(0, 4)
    n_wires = len(wires)
    if types == 1:
        n_weights_needed = 4 * n_layers
    elif types == 2:
        n_weights_needed = 2 * n_layers
    else:
        n_weights_needed = 6 * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(n_wires):
            qml.Hadamard(wires=wires[i])

        if types == 1:
            _entanglerZ(x[0], wires[0], wires[1])
            _entanglerZ(x[1], wires[2], wires[3])
            _entanglerZ(weights[l * 4 ], wires[0], wires[3])
            _entanglerZ(weights[l * 4 + 1], wires[1], wires[2])
        elif types == 2:
            _entanglerZ(weights[l * 2], wires[0], wires[1])
            _entanglerZ(weights[l * 2], wires[2], wires[3])
            _entanglerZ(weights[l * 2], wires[0], wires[3])
            _entanglerZ(weights[l * 2], wires[1], wires[2])
        else:
            _entanglerZ(weights[l * 6 ], wires[0], wires[1])
            _entanglerZ(weights[l * 6 + 1], wires[2], wires[3])
            _entanglerZ(weights[l * 6 + 2], wires[0], wires[3])
            _entanglerZ(weights[l * 6 + 3], wires[1], wires[2])

    # repeat feature encoding once more at the end
        # Either feed in feature
        qml.RX(x[0], wires=wires[0])
        qml.RX(x[1], wires=wires[2])
        if types == 1:
            qml.RX(weights[l * 4 + 2], wires=wires[1])
            qml.RX(weights[l * 4 + 3], wires=wires[3])
        elif types == 2:
            qml.RX(weights[l * 2 + 1], wires=wires[1])
            qml.RX(weights[l * 2 + 1], wires=wires[3])

        else:
            qml.RX(weights[l * 6 + 4], wires=wires[1])
            qml.RX(weights[l * 6 + 5], wires=wires[3])

def HVA_TFIM_1D_data(weights, x, wires, n_layers=1, types = 1):
    """
    1-d Ising-coupling HVA_TFIM feature map, according to 2008.02941v2.11075.

    :param weights: trainable weights of shape 2*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    """
    wires = range(0, 4)
    n_wires = len(wires)
    if types == 1:
        n_weights_needed = 6 * n_layers # Data encoded via first and last layer
    elif types == 2:
        n_weights_needed = 2 * n_layers #all zz have same params, all rx have same params
    else:
        n_weights_needed = 7 * n_layers #encode layer just in last layer

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(n_wires):
            qml.Hadamard(wires=wires[i])

        if types == 1:
            _entanglerZ(x[0], wires[0], wires[1])
            _entanglerZ(weights[l * 6 ], wires[2], wires[3])
            _entanglerZ(weights[l * 6 + 1], wires[0], wires[3])
            _entanglerZ(weights[l * 6 + 2], wires[1], wires[2])
        elif types == 2:
            _entanglerZ(weights[l * 2], wires[0], wires[1])
            _entanglerZ(weights[l * 2], wires[2], wires[3])
            _entanglerZ(weights[l * 2], wires[0], wires[3])
            _entanglerZ(weights[l * 2], wires[1], wires[2])
        else:
            _entanglerZ(weights[l * 7 ], wires[0], wires[1])
            _entanglerZ(weights[l * 7 + 1], wires[2], wires[3])
            _entanglerZ(weights[l * 7 + 2], wires[0], wires[3])
            _entanglerZ(weights[l * 7 + 3], wires[1], wires[2])

    # repeat feature encoding once more at the end
        # Either feed in feature
        qml.RX(x[0], wires=wires[0])
        if types == 1:

            qml.RX(weights[l * 6 + 3], wires=wires[1])
            qml.RX(weights[l * 6 + 4], wires=wires[2])
            qml.RX(weights[l * 6 + 5], wires=wires[3])
        elif types == 2:
            qml.RX(weights[l * 2 + 1], wires=wires[1])
            qml.RX(weights[l * 2 + 1], wires=wires[2])
            qml.RX(weights[l * 2 + 1], wires=wires[3])

        else:
            qml.RX(weights[l * 7 + 4], wires=wires[1])
            qml.RX(weights[l * 7 + 5], wires=wires[1])
            qml.RX(weights[l * 7 + 6], wires=wires[3])

def VQC(weights, x, wires, n_layers=1, types = 1):
    """ Circuits ID = 5, 6 in arXiv:1905.10876 paper
    :param weights: trainable weights of shape 2*n_layers*n_wires
    :param 1d x: input, len(x) is <= len(wires)
    :param wires: list of wires on which the feature map acts
    :param n_layers: number of repetitions of the first layer
    """
    data_size = len(x)
    n_wires = len(wires)
    weights_each_layer  = (n_wires*(n_wires+3) - 2*data_size)
    n_weights_needed = weights_each_layer * n_layers

    if len(x) > n_wires:
        raise ValueError("Feat map can encode at most {} features (which is the "
                         "number of wires), got {}.".format(n_wires, len(x)))

    if len(weights) != n_weights_needed:
        raise ValueError("Feat map needs {} weights, got {}."
                         .format(n_weights_needed, len(weights)))

    for l in range(n_layers):

        # inputs
        for i in range(data_size):
            qml.RX(x[i], wires=wires[i])

        for i in range(n_wires-data_size):
            qml.RX(weights[weights_each_layer*l+i], wires=wires[i+data_size])

        for i in range(n_wires):
            qml.RZ(weights[weights_each_layer*l+n_wires-data_size+i], wires=wires[i])

        for i in reversed(range(n_wires)):
            for j in reversed(range(n_wires)):
                if j == i:
                    continue
                if types == 1: #type 6 in Aspuru's paper
                    qml.CRX(weights[weights_each_layer*l+2*n_wires-data_size+i*(n_wires-1)+j],wires=[wires[i],wires[j]])
                if types == 2: #type 5 in Aspuru's paper
                    qml.CRZ(weights[weights_each_layer*l+2*n_wires-data_size+i*(n_wires-1)+j],wires=[wires[i],wires[j]])


        for i in range(data_size):
            qml.RX(x[i], wires=wires[i])

        for i in range(n_wires-data_size):
            qml.RX(weights[weights_each_layer*l+n_wires*(n_wires+1)-data_size+i], wires=wires[i+data_size])

        for i in range(n_wires):
            qml.RZ(weights[weights_each_layer*l+n_wires*(n_wires+2)-2*data_size+i], wires=wires[i])


def pars_HVA(n_layers=1,types=1):
    """
    Initial weight generator for 1-d qaoa feature map
    :param n_wires: number of wires
    :param n_layers: number of layers
    :return: array of weights
    """
    if types == 1:
        return 0.001*np.ones(n_layers * 4)
    elif types == 2:
        return 0.001*np.ones(n_layers * 2)
    else:
        return 0.001*np.ones(n_layers * 6)

def pars_HVA_TFIM_1D_data(n_layers=1,types=1):
    """
    Initial weight generator for 1-d qaoa feature map
    :param n_wires: number of wires
    :param n_layers: number of layers
    :return: array of weights
    """
    if types == 1:
        return 0.001*np.ones(n_layers * 6)
    elif types == 2:
        return 0.001*np.ones(n_layers * 2)
    else:
        return 0.001*np.ones(n_layers * 7)

def pars_VQC(x_dim, n_wires, n_layers=1, types = 1):

    weights_each_layer  = (n_wires*(n_wires+3) - 2*x_dim)

    return np.random.uniform(0,2*np.pi)*np.ones(n_layers * weights_each_layer)



