import jax.numpy as jnp
import pennylane as qml
from src.gates import subspace_Z, subspace_X, subspace_Y, Rgate
from src.grid import get_x_pairs, get_y_pairs
import os


def prepare_periodicity_state(parity_flips):
    """Prepare the periodicity state by flipping qubits in the auxiliary system"""
    for i in parity_flips:
        qml.PauliX(i)


def prepare_single_Gr_state(a_wires, change_first=False):
    """
    prepare a plaquette state into a Gr eigenstate is +1 eigenvalue
    assuming the physical system is in |0> state
    """
    if change_first:
        qml.PauliX(a_wires[0])  # put back in 0 state temporarily
    qml.Hadamard(a_wires[0])
    qml.CY(wires=[a_wires[0], a_wires[1]])
    qml.CNOT(wires=[a_wires[0], a_wires[2]])
    qml.CY(wires=[a_wires[0], a_wires[3]])
    if change_first:
        qml.PauliX(a_wires[0])  # put back in 1 state


def prepare_full_G_state(plaqs_wires_aux, parity_flips, periodicity_state=True):
    """Create the full vacuum state"""
    if periodicity_state:
        prepare_periodicity_state(parity_flips)
    for a_wires in reversed(plaqs_wires_aux):
        prepare_single_Gr_state(a_wires, change_first=a_wires[0] in parity_flips)


def create_pair(p_wires, a_wires):
    """create particles on the lattice (bosonized), on positions next to each other
    p_wires: (r, r+x) wires
    a_wires: (r, r+x) auxiliary wires
    """
    Lx = int(os.environ["Lx"])
    Ly = int(os.environ["Ly"])
    assert (p_wires[1] > p_wires[0]) or (
        p_wires[1] == p_wires[0] - Lx + 1
    ), f"got {p_wires}"
    qml.PauliX(p_wires[0])
    qml.PauliX(p_wires[1])
    qml.PauliZ(a_wires[1])


def hop_particles_x(p_wires, a_wires, theta, phi):
    """Move particles around (bosonized)"""
    Lx = int(os.environ["Lx"])
    Ly = int(os.environ["Ly"])
    assert (p_wires[1] > p_wires[0]) or (
        p_wires[1] == p_wires[0] - Lx + 1
    ), f"got {p_wires}"
    assert (a_wires[1] > a_wires[0]) or (
        a_wires[1] == a_wires[0] - Lx + 1
    ), f"got {a_wires}"

    qml.CNOT(wires=[p_wires[1], p_wires[0]])
    subspace_Z(p_wires, a_wires[1])
    Rgate(theta, phi, p_wires[1], dagger=True)
    qml.CNOT(wires=[p_wires[0], p_wires[1]])
    Rgate(theta, phi, p_wires[1], dagger=False)
    subspace_Z(p_wires, a_wires[1])
    qml.CNOT(wires=[p_wires[1], p_wires[0]])


def hop_particles_y(p_wires, a_wires, theta, phi):
    """Move particles around (bosonized)"""
    Lx = int(os.environ["Lx"])
    Ly = int(os.environ["Ly"])
    assert (p_wires[0] > p_wires[1]) or (
        p_wires[1] == p_wires[0] + (Lx * Ly - Lx)
    ), f"got {p_wires}"
    assert (a_wires[0] > a_wires[1]) or (
        a_wires[1] == a_wires[0] + (Lx * Ly - Lx)
    ), f"got {a_wires}"

    # we need to insert a i factor when the pos qubit is in |0> state
    qml.CNOT(wires=[p_wires[1], p_wires[0]])
    qml.CZ(wires=[p_wires[0], p_wires[1]])
    subspace_Y(p_wires, a_wires[0])
    subspace_X(p_wires, a_wires[1])
    Rgate(theta, phi, p_wires[1], dagger=True)
    qml.CNOT(wires=[p_wires[0], p_wires[1]])
    qml.CY(wires=[p_wires[0], p_wires[1]])
    Rgate(theta, phi, p_wires[1], dagger=False)
    subspace_X(p_wires, a_wires[1])
    subspace_Y(p_wires, a_wires[0])
    qml.CNOT(wires=[p_wires[1], p_wires[0]])


def hop_layer(qubit_grid, thetas, phis, parallel=False):
    """Create a full layer of hopping on all edges"""
    idx = 0
    for p_wires, a_wires in zip(
        get_y_pairs(qubit_grid[0], parallel=parallel),
        get_y_pairs(qubit_grid[1], parallel=parallel),
    ):
        theta = thetas[idx]
        phi = phis[idx]
        hop_particles_y(p_wires, a_wires, theta, phi)
        idx += 1
    for p_wires, a_wires in zip(
        get_x_pairs(qubit_grid[0], parallel=parallel),
        get_x_pairs(qubit_grid[1], parallel=parallel),
    ):
        theta = thetas[idx]
        phi = phis[idx]
        hop_particles_x(p_wires, a_wires, theta, phi)
        idx += 1
    assert (
        idx == thetas.size
    ), f"ended at index {idx}, while received {thetas.size} thetas"
    assert idx == phis.size, f"ended at index {idx}, while received {phis.size} phis"


def get_n_params(qubit_grid, n_layers):
    """Get number of parameters needed to run the VQE"""
    return (
        len(get_x_pairs(qubit_grid[0])) + len(get_y_pairs(qubit_grid[0]))
    ) * n_layers


def move_ansatz(qubit_grid, thetas, phis, n_layers=1):
    """Create one full layer of A(theta, phi) operators"""
    assert thetas.shape[0] >= n_layers
    assert phis.shape[0] >= n_layers

    def _check_same(a, b):
        if a != b:
            raise ValueError(f"{a} != {b}")

    _check_same(len(thetas), len(phis))
    _check_same(len(thetas), get_n_params(qubit_grid, n_layers))
    thetas_blocks = jnp.split(thetas, n_layers)
    phis_blocks = jnp.split(phis, n_layers)
    for il in range(n_layers):
        hop_layer(qubit_grid, thetas_blocks[il], phis_blocks[il])


def prepare_vacuum(plaqs_wires_aux, parity_flips):
    prepare_full_G_state(plaqs_wires_aux, parity_flips, periodicity_state=True)


def construct_ansatz(
    init_idxs,
    plaqs_wires_aux,
    parity_flips,
    qubit_grid,
    thetas,
    phis,
    n_layers=1,
):
    """Construct the full ansatz: vacuum + variational part"""
    prepare_vacuum(plaqs_wires_aux, parity_flips)
    for idx in init_idxs:
        p_wires_pair, a_wires_pair = [
            (qubit_grid[i, idx[0][0], idx[0][1]], qubit_grid[i, idx[1][0], idx[1][1]])
            for i in range(qubit_grid.shape[0])
        ]
        create_pair(p_wires_pair, a_wires_pair)
    move_ansatz(qubit_grid, thetas, phis, n_layers=n_layers)
