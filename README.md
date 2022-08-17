# local-f2q-circuit

Code that implements the circuits of the paper:
"Quantum circuits for solving local fermion-to-qubit mappings", by Jannes Nys and Giuseppe Carleo (http://arxiv.org/abs/2208.07192)

A detailed discussion of VQE +  time evolution is given in example.py.
It demonstrates how to construct a valid vacuum and variational state, and how all our circuits maintain the constraints.

To run these, you need:
* jupytext (to open the example vqe_and_tevo.py as a jupyter notebook)
* pennylane
* jax
* numpy
* matplotlib
* optax