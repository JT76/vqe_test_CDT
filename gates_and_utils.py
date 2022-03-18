import numpy as np
import matplotlib.pyplot as plt

def Rx(qubit_index, num_qubits, theta):
    """Return a matrix representation of the Rx gate
    """

    rx = np.array([[np.cos(theta/2.), -1.0j*np.sin(theta/2.)],
                   [-1.0j*np.sin(theta/2.), np.cos(theta/2.)]])
    identity = np.array([[1.0 , 0.0],
                         [0.0, 1.0]])
    if qubit_index==0:
        gate = rx
    else:
        gate = identity  
    for i in range(1, num_qubits):
        if qubit_index == i:
            gate = np.kron(gate, rx)
        else:
            gate = np.kron(gate, identity)
    return gate

def Ry(qubit_index, num_qubits, theta):
    """Return a matrix representation of the Ry gate
    """
    ry = np.array([[np.cos(theta/2.), -np.sin(theta/2.)],
                   [np.sin(theta/2.), np.cos(theta/2.)]])
    identity = np.array([[1.0 , 0.0],
                         [0.0, 1.0]])
    if qubit_index==0:
        gate = ry
    else:
        gate = identity  
    for i in range(1, num_qubits):
        if qubit_index == i:
            gate = np.kron(gate, ry)
        else:
            gate = np.kron(gate, identity)
    return gate

def Rz(qubit_index, num_qubits, theta):
    """Return a matrix representation of the Rz gate
    """
    rz = np.array([[np.exp(-1.0j*theta/2.), 0.],
                   [0., np.exp(1.0j*theta/2.)]])
    identity = np.array([[1.0 , 0.0],
                         [0.0, 1.0]])
    if qubit_index==0:
        gate = rz
    else:
        gate = identity  
    for i in range(1, num_qubits):
        if qubit_index == i:
            gate = np.kron(gate, rz)
        else:
            gate = np.kron(gate, identity)
    return gate

def CNOT(control, target, num_qubits):
    """Return a matrix representation of the CNOT gate
    """
    cnot = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 1., 0.]])
    identity = np.array([[1.0 , 0.0],
                         [0.0, 1.0]])
    assert target == control + 1
    min_range = 1
    if control==0:
        gate = cnot
        min_range += 1
    else:
        gate = identity 

    for i in range(min_range, num_qubits):
        if control == i:
            gate = np.kron(gate, cnot)
            i = i + 1 
        elif target == i:
            gate = gate 
        else:
            gate = np.kron(gate, identity)
    return gate

def CZ(control, target, num_qubits):
    """Return a matrix representation of the CZ gate
    """
    cz = np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., -1.]])
    assert target == control + 1
    identity = np.array([[1.0 , 0.0],
                         [0.0, 1.0]])
    min_range = 1
    if control==0:
        gate = cz
        min_range += 1
    else:
        gate = identity 
 
    for i in range(min_range, num_qubits):
        if control == i:
            gate = np.kron(gate, cz)
            i = i + 1 
        elif target == i:
            gate = gate 
        else:
            gate = np.kron(gate, identity)
    return gate



def plot_vqe_result(value_list, target):
    plt.plot(value_list, label = 'VQE estimation')
    plt.axhline(y=target, linestyle='dotted', color='red', label='Ground energy')
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.show()  



