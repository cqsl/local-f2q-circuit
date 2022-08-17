# local-f2q-circuit

## Intro
Code that implements the circuits of the paper:
"Quantum circuits for solving local fermion-to-qubit mappings", by Jannes Nys and Giuseppe Carleo (http://arxiv.org/abs/2208.07192)

## Code
A detailed discussion of VQE +  time evolution is given in example.py.
It demonstrates how to construct a valid vacuum and variational state, and how all our circuits maintain the constraints.

## Requirements
To run these, you need:
* jupytext (to open the example vqe_and_tevo.py as a jupyter notebook)
* pennylane
* jax
* numpy
* matplotlib
* optax

## How to cite

```text
@misc{nys2022quantumcircuitslocal,
  doi = {10.48550/ARXIV.2208.07192},
  url = {https://arxiv.org/abs/2208.07192},
  author = {Nys, Jannes and Carleo, Giuseppe},
  title = {Quantum circuits for solving local fermion-to-qubit mappings},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```