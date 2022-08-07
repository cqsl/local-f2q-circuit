# import scipy
import pennylane as qml
import numpy as np
from src.gates import CH, isingZZZ
from src.grid import get_x_pairs, get_y_pairs


def _hopping_map_x(wires):
    qml.CNOT(wires=(wires[1], wires[0]))
    CH(wires=wires)
    qml.CNOT(wires=(wires[1], wires[0]))


def _hopping_map_y(p_wires, a_wires):
    # map the physical qubits
    _hopping_map_x(p_wires)  # reuse
    qml.S(wires=p_wires[0])
    # map the auxiliary qubits
    qml.RX(-np.pi / 2, wires=a_wires[0])
    qml.RY(np.pi / 2, wires=a_wires[1])


def time_evo_x(p_wires, a_wires, theta):
    """Evolution operator of the x hopping"""
    qml.adjoint(_hopping_map_x)(p_wires)
    wires1 = [p_wires[0], a_wires[1]]
    qml.IsingZZ(theta, wires1)
    wires2 = [p_wires[1], a_wires[1]]
    qml.IsingZZ(-theta, wires2)
    _hopping_map_x(p_wires)


# def horizontal_exact(p_wires, a_wires, t):
#     def _circ1():
#         qml.PauliX(p_wires[0])
#         qml.PauliX(p_wires[1])
#         qml.PauliZ(a_wires[1])

#     def _circ2():
#         qml.PauliY(p_wires[0])
#         qml.PauliY(p_wires[1])
#         qml.PauliZ(a_wires[1])

#     m1 = matrix(_circ1, n_wires=3)()
#     m2 = matrix(_circ2, n_wires=3)()
#     H = -(m1 + m2) / 2
#     return scipy.linalg.expm(-1j * t * H)


def time_evo_y(p_wires, a_wires, theta):
    """Evolution operator of the y hopping"""
    qml.adjoint(_hopping_map_y)(p_wires, a_wires)

    wires1 = [*a_wires, p_wires[0]]
    isingZZZ(theta, wires=wires1)
    wires2 = [*a_wires, p_wires[1]]
    isingZZZ(-theta, wires=wires2)

    _hopping_map_y(p_wires, a_wires)


# def vertical_exact(p_wires, a_wires, t):
#     def _circ1():
#         qml.PauliX(p_wires[0])
#         qml.PauliY(p_wires[1])
#         qml.PauliY(a_wires[0])
#         qml.PauliX(a_wires[1])

#     def _circ2():
#         qml.PauliY(p_wires[0])
#         qml.PauliX(p_wires[1])
#         qml.PauliY(a_wires[0])
#         qml.PauliX(a_wires[1])

#     m1 = matrix(_circ1, n_wires=4)()
#     m2 = matrix(_circ2, n_wires=4)()
#     H = -(-m1 + m2) / 2
#     return scipy.linalg.expm(-1j * t * H)


def time_evo_inter(p_wires, theta):
    """Evolution operator of the interaction term"""
    qml.ControlledPhaseShift(-theta, p_wires)


# def interaction_exact(p_wires, t):
#     m0 = matrix(qml.Identity, n_wires=2)(p_wires[0])
#     m1 = matrix(qml.PauliZ, n_wires=2)(p_wires[0])
#     m2 = matrix(qml.PauliZ, n_wires=2)(p_wires[1])
#     H = (m0 - m1) @ (m0 - m2) / 4
#     return scipy.linalg.expm(-1j * t * H)


def time_evo_circuit(qubit_grid, t_hop, V_coul, n_steps=1, delta_t=1e-3, parallel=True):
    """Full time evolution circuit"""
    for _ in range(n_steps):
        if t_hop != 0.0:
            for p_wires, a_wires in zip(
                get_y_pairs(qubit_grid[0], parallel=parallel),
                get_y_pairs(qubit_grid[1], parallel=parallel),
            ):
                time_evo_y(p_wires, a_wires, t_hop * delta_t)
            for p_wires, a_wires in zip(
                get_x_pairs(qubit_grid[0], parallel=parallel),
                get_x_pairs(qubit_grid[1], parallel=parallel),
            ):
                time_evo_x(p_wires, a_wires, t_hop * delta_t)
        if V_coul != 0.0:
            for p_wires in get_y_pairs(qubit_grid[0], parallel=parallel):
                time_evo_inter(p_wires, V_coul * delta_t)
            for p_wires in get_x_pairs(qubit_grid[0], parallel=parallel):
                time_evo_inter(p_wires, V_coul * delta_t)


def hv_ansatz_circuit(qubit_grid, params, n_steps=1, parallel=True):
    """Variational ansatz based on the time evolution"""
    param_counter = 0
    for _ in range(n_steps):
        for p_wires, a_wires in zip(
            get_y_pairs(qubit_grid[0], parallel=parallel),
            get_y_pairs(qubit_grid[1], parallel=parallel),
        ):
            time_evo_y(p_wires, a_wires, params[param_counter])
            param_counter += 1
            time_evo_inter(p_wires, params[param_counter])
            param_counter += 1
        for p_wires, a_wires in zip(
            get_x_pairs(qubit_grid[0], parallel=parallel),
            get_x_pairs(qubit_grid[1], parallel=parallel),
        ):
            time_evo_x(p_wires, a_wires, params[param_counter])
            param_counter += 1
            time_evo_inter(p_wires, params[param_counter])
            param_counter += 1

    assert (
        param_counter == params.size
    ), f"ended up at {param_counter}, while {params.size} parameters were provided"


def get_n_params_hv(qubit_grid, n_layers):
    """Number of parameters needed to run the HV ansatz"""
    Lx = len(get_x_pairs(qubit_grid[0]))
    Ly = len(get_y_pairs(qubit_grid[0]))
    n_pars = 2 * (Lx + Ly)  # number of edges
    n_pars = n_pars * n_layers
    return n_pars
