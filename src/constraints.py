import pennylane as qml
from src.utils import state, init_state, same_states, print_state
import numpy as np


def Gr_circuit(p_wires, a_wires):
    """The Gauss operator"""
    qml.PauliX(a_wires[0])
    qml.PauliY(a_wires[1])
    qml.PauliX(a_wires[2])
    qml.PauliY(a_wires[3])
    # here we need the auxiliary system as well
    qml.PauliZ(p_wires[-1])
    qml.PauliZ(p_wires[0])


def Gr_proj_state(st, p_wires, a_wires):
    n_wires = int(np.log2(st.size))
    return state(Gr_circuit, n_wires, in_state=st)(p_wires, a_wires)


def verify_Gr(
    circuit, n_wires, p_wires, a_wires, *circuit_args, circuit_kwargs={}, in_state=None
):
    """Method to verify the Gr constraints on a circuit"""
    if in_state is None:
        in_state = init_state(n_wires)
    out_state = state(circuit, n_wires, in_state=in_state)(
        *circuit_args, **circuit_kwargs
    )
    out_Gr_state = Gr_proj_state(out_state, p_wires, a_wires)
    if not same_states(out_state, out_Gr_state):
        print("ERROR: different states found for Gr on wires:", p_wires, a_wires)
        print("STATE 1:")
        print_state(out_state)
        print("STATE 2:")
        print_state(out_Gr_state)
        raise ValueError("")


def verify_G_locals(
    circuit,
    n_wires,
    plaqs_wires,
    plaqs_wires_aux,
    *circuit_args,
    circuit_kwargs={},
    in_state=None
):
    """Run over all plaquettes and check all Gauss constraints"""
    # verify by verifying each Gr individually
    for p_wires, a_wires in zip(plaqs_wires, plaqs_wires_aux):
        verify_Gr(
            circuit,
            n_wires,
            p_wires,
            a_wires,
            *circuit_args,
            circuit_kwargs=circuit_kwargs,
            in_state=in_state
        )
