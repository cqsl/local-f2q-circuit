import pennylane as qml
from src.grid import get_x_pairs, get_y_pairs


def fermihubbard(t, V, qubit_grid):
    """Spinless fermi-hubbard model"""
    phys = qubit_grid[0]
    aux = qubit_grid[1]
    coeffs = []
    ops = []

    if t != 0:

        for (r_p, rx_p), (r_a, rx_a) in zip(get_x_pairs(phys), get_x_pairs(aux)):
            coeffs.append(-t / 2)
            ops.append(qml.PauliY(r_p) @ qml.PauliY(rx_p) @ qml.PauliZ(rx_a))
            coeffs.append(-t / 2)
            ops.append(qml.PauliX(r_p) @ qml.PauliX(rx_p) @ qml.PauliZ(rx_a))

        for (r_p, ry_p), (r_a, ry_a) in zip(get_y_pairs(phys), get_y_pairs(aux)):
            coeffs.append(t / 2)
            ops.append(
                qml.PauliX(r_p) @ qml.PauliY(ry_p) @ qml.PauliY(r_a) @ qml.PauliX(ry_a)
            )
            coeffs.append(-t / 2)
            ops.append(
                qml.PauliY(r_p) @ qml.PauliX(ry_p) @ qml.PauliY(r_a) @ qml.PauliX(ry_a)
            )

    if V != 0:
        all_pairs = get_x_pairs(phys) + get_y_pairs(phys)
        for r_p, rd_p in all_pairs:
            coeffs.append(V / 4)
            ops.append(qml.Identity(0))
            coeffs.append(-V / 4)
            ops.append(qml.PauliZ(r_p))
            coeffs.append(-V / 4)
            ops.append(qml.PauliZ(rd_p))

        for r_p, rd_p in all_pairs:
            coeffs.append(V / 4)
            ops.append(qml.PauliZ(r_p) @ qml.PauliZ(rd_p))

    return qml.Hamiltonian(coeffs, ops, simplify=True)
