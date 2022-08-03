import jax.numpy as jnp
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# helper functions


def matrix(circuit, n_wires=None):
    wire_order = None if n_wires is None else np.arange(n_wires)

    def _matrix(*args, **kwargs):
        return qml.matrix(circuit, wire_order=wire_order)(*args, **kwargs)

    return _matrix


def _round_matrix(m, eps=1e-14):
    m = np.array(m)
    m[np.abs(m) < eps] = 0
    return m


def plot_matrix(m):
    mr = _round_matrix(m.real)
    im = plt.matshow(mr)
    plt.colorbar(im)
    plt.show()
    mi = _round_matrix(m.imag)
    im = plt.matshow(mi)
    plt.colorbar(im)
    plt.show()


def draw(circuit, n_wires):
    def _draw(*args, **kwargs):
        dev = qml.device("default.qubit", wires=n_wires)
        #             result = circuit(*args, **kwargs)

        def _new_circuit(*args, **kwargs):
            circuit(*args, **kwargs)
            qml.Barrier()
            measurements = []
            for i in range(n_wires):
                measurements.append(qml.expval(qml.PauliZ(i)))
            return measurements

        qnode = qml.QNode(_new_circuit, dev)
        print(qml.draw(qnode)(*args, **kwargs))

    return _draw


def state(circuit, n_wires, in_state=None):
    if in_state is not None:
        assert isinstance(in_state, (jnp.ndarray, np.ndarray))
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def _circuit(*args, **kwargs):
        if in_state is not None:
            qml.QubitStateVector(in_state, wires=np.arange(n_wires))
        circuit(*args, **kwargs)
        return qml.state()

    return _circuit


def init_state(n_wires):
    def _circuit():
        pass

    return state(_circuit, n_wires)()


def normalize_state(s):
    norm = np.sqrt(np.sum(np.conjugate(s) * s))
    return s / norm


def qasm(circ, n_wires, *args, **kwargs):
    dev = qml.device("qiskit.aer", wires=n_wires)

    @qml.qnode(dev)
    def _circuit(*args, **kwargs):
        circ(*args, **kwargs)
        qml.Barrier()
        measurements = []
        for i in range(n_wires):
            measurements.append(qml.expval(qml.PauliZ(i)))
        return measurements

    _circuit(*args, **kwargs)
    return dev._circuit.decompose().qasm(formatted=True)


def decomp_and_draw(circ, n_wires, *args, **kwargs):
    dev = qml.device("qiskit.aer", wires=n_wires)

    @qml.qnode(dev)
    def _circuit(*args, **kwargs):
        circ(*args, **kwargs)
        qml.Barrier()
        measurements = []
        for i in range(n_wires):
            measurements.append(qml.expval(qml.PauliZ(i)))
        return measurements

    _circuit(*args, **kwargs)
    return dev._circuit.compose().draw()


def _print_float(x):
    if np.isclose(x, 0):
        s = ""
    else:
        s = f"{x:1.6f}"
    return s.rjust(9, " ")


def _print_complex(x):
    if not np.iscomplexobj(x):
        return _print_float(x)
    return f"({_print_float(x.real)} + {_print_float(x.imag)}j)"


def test_state(n_wires):
    def _circuit():
        pass

    #         qml.PauliX(0)
    #         qml.PauliX(1)
    return state(_circuit, n_wires)()


def print_state(state):
    state = state.flatten()
    n_wires = int(np.log2(state.size))
    for i in range(state.size):
        b = state[i]
        if np.isclose(b, 0):
            continue
        else:
            bs = np.binary_repr(i, n_wires)
            print(f"{bs}  {_print_complex(b)}")


def state_from_string(s_str):
    n_wires = len(s_str)
    state = np.zeros((2**n_wires,))
    s_ints = list(map(int, list(s_str)))
    basis = 2 ** np.arange(n_wires)[::-1]
    idx = np.sum(basis * s_ints)
    state[idx] = 1
    return state


def same_states(a, b):
    """same state, with same phase"""
    return np.allclose(a, b)
