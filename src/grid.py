import numpy as np


def shape_from_grid(grid):
    """Get the grid dimensions from the qubits array"""
    Ly, Lx = grid.shape[-2:]
    return Lx, Ly


def get_x_pairs(grid, x=None, parallel=False):
    """Return all edges along the x direction"""
    Lx, Ly = shape_from_grid(grid)
    pairs = []
    do_parallel = parallel and Lx > 3
    if x is None:
        if do_parallel:
            for ix in range(0, Lx if Lx > 2 else 1, 2):
                pairs += get_x_pairs(grid, x=ix)
            for ix in range(1, Lx if Lx > 2 else 1, 2):
                pairs += get_x_pairs(grid, x=ix)
        else:
            for ix in range(Lx if Lx > 2 else 1):
                pairs += get_x_pairs(grid, x=ix)
        return pairs
    else:
        ix = x
        ixx = (ix + 1) % Lx
        for iy in range(Ly):
            pairs.append((grid[iy, ix], grid[iy, ixx]))
        return pairs


def get_y_pairs(grid, y=None, parallel=True):
    """Return all edges along the y direction"""
    Lx, Ly = shape_from_grid(grid)
    pairs = []
    do_parallel = parallel and Ly > 3
    if y is None:
        if do_parallel:
            for iy in range(0, Ly if Ly > 2 else 1, 2):
                pairs += get_y_pairs(grid, y=iy)
            for iy in range(1, Ly if Ly > 2 else 1, 2):
                pairs += get_y_pairs(grid, y=iy)
        else:
            for iy in range(Ly if Ly > 2 else 1):
                pairs += get_y_pairs(grid, y=iy)
        return pairs
    else:
        iy = y
        iyy = (iy + 1) % Ly
        for ix in range(Lx):
            # this order is important, must be consistent!
            pairs.append((grid[iyy, ix], grid[iy, ix]))

        return pairs


def get_plaquettes(qubit_grid):
    """Get all plaquettes"""
    Lx, Ly = shape_from_grid(qubit_grid)
    n_qubits_per_sys = Lx * Ly
    plaqs_wires = [
        qubit_grid[0, [i, i, i + 1, i + 1], [j, j + 1, j + 1, j]]
        for i in range(Ly - 1)
        for j in range(Lx - 1)
    ]
    plaqs_wires_aux = [p + n_qubits_per_sys for p in plaqs_wires]
    return plaqs_wires, plaqs_wires_aux


def get_qubit_grid(Lx, Ly):
    """Create a grid of qubits and number the qubits"""
    n_qubits = 2 * Lx * Ly
    qubit_grid = np.arange(n_qubits).reshape(2, Ly, Lx)
    return qubit_grid
