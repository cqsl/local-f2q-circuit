import pennylane as qml
import numpy as np
from pennylane.wires import Wires
from pennylane.operation import Operation


class CH(Operation):
    """Conditional Hadamard"""

    num_wires = 2
    num_params = 0
    basis = "Z"

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "H"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        s2 = 1 / np.sqrt(2)
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, s2, s2], [0, 0, s2, -s2]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        return np.array([-1.0, 1.0, 1.0, 1.0])

    def adjoint(self):
        # the adjoint operator of this gate simply negates the angle
        return CH(wires=self.wires)
        # return FlipAndRotate(-self.parameters[0], self.wires[0], self.wires[1], do_flip=self.hyperparameters["do_flip"])

    @staticmethod
    def compute_decomposition(wires):
        decomp_ops = [qml.CZ(wires=wires), qml.CRY(np.pi / 2, wires=wires)]
        return decomp_ops

    @property
    def control_wires(self):
        return Wires(self.wires[0])


def Rgate(theta, phi, wire, dagger=False):
    if dagger:
        sign = -1
        qml.RZ(sign * (phi + np.pi), wire)
        qml.RY(sign * (theta + np.pi / 2), wire)
    else:
        sign = +1
        # change order also
        qml.RY(sign * (theta + np.pi / 2), wire)
        qml.RZ(sign * (phi + np.pi), wire)


def CCX(wires):
    """Doubly conditional X gate"""
    qml.Toffoli(wires=wires)


def CCY(wires):
    """Doubly conditional Y gate"""
    qml.RZ(-np.pi / 2, wires=wires[-1])
    CCX(wires)
    qml.RZ(np.pi / 2, wires=wires[-1])


def CCZ(wires):
    """Doubly conditional Z gate"""
    qml.RY(np.pi / 2, wires=wires[-1])
    CCX(wires)
    qml.RY(-np.pi / 2, wires=wires[-1])


def subspace_X(p_wires, a_wire):
    """X operation only in the single-excitation subspace"""
    CCX(wires=[p_wires[0], p_wires[1], a_wire])


def subspace_Y(p_wires, a_wire):
    """Y operation only in the single-excitation subspace"""
    CCY(wires=[p_wires[0], p_wires[1], a_wire])


def subspace_Z(p_wires, a_wire):
    """Z operation only in the single-excitation subspace"""
    CCZ(wires=[p_wires[0], p_wires[1], a_wire])


def isingZZZ(theta, wires):
    """Three qubit Z rotation"""
    qml.CNOT(wires=(wires[0], wires[1]))
    qml.CNOT(wires=(wires[1], wires[2]))
    qml.RZ(theta, wires=wires[2])
    qml.CNOT(wires=(wires[1], wires[2]))
    qml.CNOT(wires=(wires[0], wires[1]))
