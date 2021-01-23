"""
Fidelity classifier
===================

Implements the fidelity classifier.

``predict()`` returns the predicted label or continuous output for a new input
``accuracy()`` returns the accuracy on a test set

The 'exact' implementation computes overlap of ket vectors numerically.
The 'circuit' implementation performs a swap test on all data pairs.

"""
import pennylane as qml
from pennylane import numpy as np
import dill as pickle  # to load featuremap


def negate(item):
    if isinstance(item, list):
        return [-i for i in item]
    else:
        return -item


def cphase_inv(k):
    gate = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(np.complex(0, -2*np.pi/2**k))]]
    return np.array(gate)


def _fast(x_new, A_samples, B_samples, featmap, pars, n_inp):
    """
    Implements the fidelity measurement circuit using the "overlap with 0" trick.
    """
    # Allocate registers
    dev = qml.device('default.qubit', wires=n_inp)
    # Identify input register wires
    wires = list(range(n_inp))
    Proj0 = np.zeros((2**n_inp, 2**n_inp))
    Proj0[0, 0] = 1

    @qml.qnode(dev)
    def circuit(weights, x1=None, x2=None):
        # Apply embedding
        featmap(weights, x1, wires)
        # Apply inverse embedding
        featmap(negate(weights), negate(x2), wires)
        # Measure overlap with |0..0>
        return qml.expval(qml.Hermitian(Proj0, wires=wires))

    # Compute mean overlap with A
    overlap_A = 0
    for a in A_samples:
        overlap_A += circuit(pars, x1=a, x2=x_new)
    overlap_A = overlap_A/len(A_samples)

    # Compute mean overlap with B
    overlap_B = 0
    for b in B_samples:
        overlap_B += circuit(pars, x1=b, x2=x_new)
    overlap_B = overlap_B/len(B_samples)

    return overlap_A, overlap_B


def _circuit(x_new, A_samples, B_samples, featmap, pars, n_inp):
    """
    Implements the fidelity measurement circuit using samples of class A and B.
    """
    # Allocate registers
    n_qubits = 2*n_inp + 1  # Total number of qubits
    dev = qml.device('default.qubit', wires=n_qubits)
    # Identify input register wires
    wires_x1 = list(range(1, n_inp+1))
    wires_x2 = list(range(n_inp+1, 2*n_inp+1))

    @qml.qnode(dev)
    def circuit(weights, x1=None, x2=None):
        # Load the two inputs into two different registers
        featmap(weights, x1, wires_x1)
        featmap(weights, x2, wires_x2)

        # Do a SWAP test
        qml.Hadamard(wires=0)
        for k in range(n_inp):
            qml.CSWAP(wires=[0, k + 1, n_inp + k + 1])
        qml.Hadamard(wires=0)

        # Measure overlap by checking ancilla
        return qml.expval(qml.PauliZ(0))

    # Compute mean overlap with A
    overlap_A = 0
    for a in A_samples:
        overlap_A += circuit(pars, x1=a, x2=x_new)
    overlap_A = overlap_A/len(A_samples)

    # Compute mean overlap with B
    overlap_B = 0
    for b in B_samples:
        overlap_B += circuit(pars, x1=b, x2=x_new)
    overlap_B = overlap_B/len(B_samples)

    return overlap_A, overlap_B


def _exact(x_new, A_samples, B_samples, featmap, n_inp, pars):
    """Calculates the analytical result of the fidelity measurement,

        overlap_A = \sum_i p_A |<\phi(x_new)|\phi(a_i)>|^2,
        overlap_B = \sum_i p_B |<\phi(x_new)|\phi(b_i)>|^2,

        using numpy as well as pennylane to simulate the feature map.
     """

    dev = qml.device('default.qubit', wires=n_inp)

    @qml.qnode(dev)
    def fm(weights, x=None):
        """Circuit to get the state after feature map"""
        featmap(weights, x, range(n_inp))
        return qml.expval(qml.PauliZ(0))

    # Compute feature states for A
    A_states = []
    for a in A_samples:
        fm(pars, x=a)
        phi_a = dev._state
        A_states.append(phi_a)

    # Compute feature states for B
    B_states = []
    for b in B_samples:
        fm(pars, x=b)
        phi_b = dev._state
        B_states.append(phi_b)

    # Get feature state for new input
    fm(pars, x=x_new)
    phi_x = dev._state

    # Put together
    overlap_A = sum([np.abs(np.vdot(phi_x, phi_a)) ** 2 for phi_a in A_states])
    overlap_A = overlap_A/len(A_states)

    overlap_B = sum([np.abs(np.vdot(phi_x, phi_b)) ** 2 for phi_b in B_states])
    overlap_B = overlap_B/len(B_states)

    return overlap_A, overlap_B


def predict(x_new, path_to_featmap, n_samples=None,
            probs_A=None, probs_B=None, binary=True, implementation=None, seed=None):
    """
    Predicts which class the new input is from, using either exact numerical simulation
    or a simulated quantum circuit.

    As a convention, the class labeled by +1 is 'A', the class labeled by -1 is 'B'.

    :param x_new: new input to predict label for
    :param path_to_featmap: Where to load featmap from.
    :param n_samples: How many samples to use, if None, use full class (simulating perfect measurement)
    :param probs_A: Probabilities with which to draw each samples from A. If None, use uniform.
    :param probs_B: Probabilities with which to draw each samples from B. If None, use uniform.
    :param binary: If True, return probability, else return value {-1, 1}
    :param implementation: String that chooses the background implementation. Can be 'exact',
        'fast' or 'circuit'
    :return: probability or prediction of class for x_new
    """

    if seed is not None:
        np.random.seed(seed)

    # Load settings from result of featmap learning function
    settings = np.load(path_to_featmap, allow_pickle=True).item()
    featmap = pickle.loads(settings['featmap'])
    pars = settings['pars']
    n_inp = settings['n_wires']
    X = settings['X']
    Y = settings['Y']
    A = X[Y == 1]
    B = X[Y == -1]

    if probs_A is not None and len(probs_A) != len(A):
        raise ValueError("Length of probs_A and A have to be the same, got {} and {}."
                         .format(len(probs_A), len(A)))
    if probs_B is not None and len(probs_B) != len(B):
        raise ValueError("Length of probs_B and B have to be the same, got {} and {}."
                         .format(len(probs_B), len(B)))

    # Sample subsets from A and B
    if n_samples is None:
        # Consider all samples from A, B
        A_samples = A
        B_samples = B
    else:
        selectA = np.random.choice(range(len(A)), size=(n_samples,), replace=True, p=probs_A)
        A_samples = A[selectA]
        selectB = np.random.choice(range(len(B)), size=(n_samples,), replace=True, p=probs_B)
        B_samples = B[selectB]

    if implementation == "exact":
        overlap_A, overlap_B = _exact(x_new=x_new, A_samples=A_samples, B_samples=B_samples,
                                      featmap=featmap, n_inp=n_inp, pars=pars)
    elif implementation == "circuit":
        overlap_A, overlap_B = _circuit(x_new=x_new, A_samples=A_samples, B_samples=B_samples,
                                        featmap=featmap, pars=pars, n_inp=n_inp)
    elif implementation == "fast":
        overlap_A, overlap_B = _fast(x_new=x_new, A_samples=A_samples, B_samples=B_samples,
                                     featmap=featmap, pars=pars, n_inp=n_inp)
    else:
        raise ValueError("Implementation not recognized.")

    if binary:
        if overlap_A > overlap_B:
            return 1
        elif overlap_A < overlap_B:
            return -1
        else:
            return 0
    else:
        return overlap_A - overlap_B


def accuracy(X, Y, path_to_featmap, n_samples=None, probs_A=None, probs_B=None,
             implementation=None, seed=None):
    """
    Computes the ratio of correctly classified samples to all samples.

    :param X: Array of test inputs
    :param Y: 1-d array of test labels
    :param path_to_featmap: Where to load featmap from.
    :param n_samples: How many samples to use, if None, use full class (simulating perfect measurement)
    :param probs_A: Probabilities with which to draw each samples from A. If None, use uniform.
    :param probs_B: Probabilities with which to draw each samples from B. If None, use uniform.
    :param implementation: String that chooses the background implementation.
    :return: accuracy of predictions on test set
    """

    acc = []
    for x_test, y_test in zip(X, Y):
        y_pred = predict(x_new=x_test,
                         path_to_featmap=path_to_featmap,
                         n_samples=n_samples,
                         probs_A=probs_A,
                         probs_B=probs_B,
                         binary=True,
                         implementation=implementation,
                         seed=seed)

        if y_test == y_pred:
            acc.append(1)
        else:
            acc.append(0)

    return sum(acc)/len(acc)

