import numpy as np
from hamiltonian import get_hamil, nth_state
from gates_and_utils import Rx, Ry, Rz, CNOT, CZ, plot_vqe_result
import time

def ansatz(hamiltonian, thetas, depth):
    """The hardware efficient ansatz (HEA) function takes as inputs the modelled molecular 
       hamiltonian, information about the structure of the ansatz (its depth). Here depth is 
       understood as the number of HEA layers (depth=1 implies 1 Rx and 1 Ry on each qubit, 
       followed by a ladder of entangling gates). The functions assumes that the input state
       is always the zero state | 0 > for all qubits.

    Args:
        hamiltonian: (Matrix, complex) Matrix representing the hamiltonian of the molecule studied.
        thetas: (1-D array, float) Array of parameters for the parametrised gates of the ansatz, 
                ordering suggested: 
                    First layer, first qubit, first gate.
                    First layer, first qubit, second gate.
                    First layer, second qubit, first gate.
                    etc.
        depth: (int) Number of HEA ansatz layer for this ansatz

    Returns:
        expect_val: (real, float) The expectation value of the hamiltonian with respect to the 
                    parametrised state produced by the ansatz. Because we work in density matrix 
                    formalism, we do not need to worry about the number of measurement or finite
                    sampling noise.  

    """

    num_qubits = int(np.log2(hamiltonian.shape[0]))
    state = zero_state(num_qubits)

    # TODO: complete the ansatz function

    return expect_val

def zero_state(num_qubits):
    """This function produces a vector model for the zero state of the dimension of 
       the entire qubit space considered

    Args:
        num_qubits: (int) number of qubits used to model the wave function of ground state  

    Returns:
        complete_zero_state: (1-D array, float) vector representation of the zero state 
                             over all qubits

    """
    zero_state = np.array([1., 0.])
    complete_zero_state = np.array([1., 0.])
    for i in range(1, num_qubits):
        complete_zero_state = np.kron(complete_zero_state, zero_state)
    return complete_zero_state 

def compute_gradient(ansatz_func, hamiltonian, thetas, depth, theta_index):
    """This function computes the analytical gradient for one parameters given an ansatz

    Args:
        ansatz_func: (function) Function representing the ansatz for which the gradient is computed
        hamiltonian: (Matrix, complex) Matrix representing the hamiltonian of the molecule studied.
                     Used as input for the ansatz function
        thetas: (1-D array, float) Array of parameters for the parametrised gates of the ansatz
        depth: (int) Number of HEA ansatz layer for this ansatz
        theta_index: (int) Index of the parameter in thetas for which the gradient is computed 

    Returns:
        gradient: (float) Gradient of the expectation value of the hamiltonian w.r.t. the 
                  parametrised state for a given parameter

    """
    num_thetas = int(len(thetas))
    thetas_plus = np.copy(thetas)
    thetas_plus[theta_index]=thetas[theta_index] + np.pi/2. 

    thetas_minus = np.copy(thetas)
    thetas_minus[theta_index]=thetas[theta_index] - np.pi/2.

    gradient = 0.5*(ansatz_func(hamiltonian, thetas_plus, depth) 
                    - ansatz_func(hamiltonian, thetas_minus, depth))
    return gradient

def gradient_descent(ansatz_func, hamiltonian, thetas, depth, step):
    """Gradient descent optimiser for the ansatz function. Requires a function to compute gradient.

    Args:
        ansatz_func: (function) Function representing the ansatz for which the gradient is computed
        hamiltonian: (Matrix, complex) Matrix representing the hamiltonian of the molecule studied.
                     Used as input for the ansatz function
        thetas: (1-D array, float) Array of parameters for the parametrised gates of the ansatz
        depth: (int) Number of HEA ansatz layer for this ansatz
        step: (float) Value specifying the proportion of the gradient applied to each parameter at
              each iteration of the algorithm

    Returns:
        thetas_new: (1D-array, float) Updated set of parameters for the ansatz, following one 
                    iteration of the gradient descent optimiser

    """

    gradient_vector = []
    num_thetas = int(len(thetas))
    for i in range(num_thetas):
        gradient_vector.append(compute_gradient(ansatz_func,
                                                hamiltonian, 
                                                thetas, 
                                                depth, i))
    thetas_new = thetas - step*np.array(gradient_vector)
    return thetas_new


def rotosolver(ansatz_func, hamiltonian, thetas, depth):
    """RotoSolver optimiser for the ansatz function. The RotoSolver optimiser relies on the 
       sinusoidal properties of the value of the expectation of an operator following a given 
       parameter to find the minimum.

    Args:
        ansatz_func: (function) Function representing the ansatz for which parameters are updated.
        hamiltonian: (Matrix, complex) Matrix representing the hamiltonian of the molecule studied.
                     Used as input for the ansatz function
        thetas: (1-D array, float) Array of parameters for the parametrised gates of the ansatz
        depth: (int) Number of HEA ansatz layer for this ansatz

    Returns:
        thetas_new: (1D-array, float) Updated set of parameters for the ansatz, following one 
                    iteration of the RotoSolver optimiser
    """
    num_thetas = int(len(thetas))
    for i in range(num_thetas):
        thetas_plus = np.copy(thetas)
        thetas_minus = np.copy(thetas)
        thetas_new = np.copy(thetas)       

        thetas_plus[i]=thetas[i] + np.pi/2. 
        thetas_minus[i]=thetas[i] - np.pi/2.

        expect_val = np.real(ansatz_func(hamiltonian, thetas, depth))
        expect_val_minus = np.real(ansatz_func(hamiltonian, thetas_minus, depth))
        expect_val_plus = np.real(ansatz_func(hamiltonian, thetas_plus, depth))

        c = (expect_val_minus + expect_val_plus) / 2.0
        b = np.arctan2(expect_val - c, expect_val_plus - c) - thetas_new[i]
        thetas_new[i] = -b - np.pi / 2.0
        thetas = np.copy(thetas_new)
    return thetas

def vqe(molecule, ansatz_func, num_iter, depth, optimiser_func, 
        optimiser_args={'optimiser_args': None}, print_all=False, plot_vqe=False):
    """Run the Variational Quantum Eigensolver (VQE) for a specified number of iteration given 
       an ansatz and an optimiser. 

    Args:
        molecule: (string) Molecule for which the ground state will be computed. Molecules available
                  in the hamiltonian.py file include H2, LiH and BeH2.
        ansatz_func: (function) Function representing the ansatz representing the wave-function

        num_iter: (int) Number of iteration of the VQE to be conducted
        depth: (int) Number of ansatz layer for the VQE
        optimiser_func: (function) Function representing the optimiser used to updated the 
                        parameters at each VQE iteration
        optimiser_args: (dict) Dictionary holding the parameters specific to the optimiser used
        print_all: (bool) Flag to control whether results are printed for all iterations or 
                   only the last one
        plot_vqe: (bool) Flag to control whether results are plotted for all iterations
        
    Returns:
        expect_val: (real, float) Final expectation value computed by the VQE - representing the 
                    ground energy of the molecule
    """

    hamiltonian = get_hamil(molecule)
    num_qubits = int(np.log2(hamiltonian.shape[0]))
    num_thetas  = #TODO specify the number of parameters

    ground_energy, ground_state = nth_state(hamiltonian, 0)
    print('---------------------------------------------')
    print('Target for ground energy:           ' + str(round(np.real(ground_energy), 6)))
    value_saved = []
    thetas = np.random.uniform(-np.pi, np.pi, num_thetas).astype('float32')
    begin_vqe = time.time()
    for iteration in range(num_iter):
        begin_iter = time.time()
        thetas = optimiser_func(ansatz_func, hamiltonian, thetas, depth, **optimiser_args)
        expect_val = ansatz_func(hamiltonian, thetas, depth)
        end_iter = time.time()
        runtime_iter = (round(end_iter - begin_iter, 6))    
        if print_all:
            print('---------------------------------------------')
            print('Iteration number:    ' + str(iteration))
            print('Expectation value:  ' + str(round(expect_val, 6)))
            print('Error to target:     ' + str(round(abs(expect_val - ground_energy), 6)))
            print('Runtime:             ' + str(runtime_iter))
        value_saved.append(expect_val)
    end_vqe = time.time()
    runtime_vqe = (round(end_vqe - begin_vqe, 6))
    print('---------------------------------------------')
    print('Final VQE estimated ground energy:  '  + str(round(expect_val, 6)))
    print('Error to target:                     ' + str(round(abs(expect_val - ground_energy), 6)))
    print('Runtime:                             ' + str(runtime_vqe))
    print('END of VQE \n')
    if plot_vqe: plot_vqe_result(value_saved, ground_energy)
    return expect_val






# Test code for your ansatz. It should work once your ansatz code is completed, however note 
# that the number of iteration and the depth of the ansatz might not be ideal for H2, LiH or BeH2

print('\nGradient descent optimiser: ')
vqe(molecule='H2', 
    ansatz_func=ansatz, 
    num_iter = 100, 
    depth=4, 
    optimiser_func=gradient_descent, 
    optimiser_args={'step': 0.1},
    print_all=False)

print('RotoSolver optimiser: ')
vqe(molecule='H2', 
    ansatz_func=ansatz, 
    num_iter = 100, 
    depth=4, 
    optimiser_func=rotosolver,
    print_all=False)









