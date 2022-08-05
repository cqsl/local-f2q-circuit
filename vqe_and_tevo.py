# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Circuits for local fermion-to-qubit mappings in 2D

# +
import optax
from functools import partial
from collections import defaultdict
import os
import jax.numpy as jnp
import jax
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# from jax.config import config
# config.update("jax_enable_x64", True)

from src.tevo import time_evo_circuit
from src.hamiltonian import fermihubbard
from src.utils import draw, state, print_state, same_states
from src.ansatz import prepare_full_G_state, prepare_single_Gr_state, prepare_periodicity_state, create_pair, prepare_vacuum, hop_particles_x, hop_particles_y, get_n_params, construct_ansatz
from src.constraints import verify_G_locals, Gr_circuit
from src.grid import get_qubit_grid, get_plaquettes

# %load_ext autoreload
# %autoreload 2

# +
# grid dimensions
Lx = 4
Ly = 2

# you can ignore the following, this is to execute some checks in the hopping
os.environ["Lx"] = str(Lx)
os.environ["Ly"] = str(Ly)

n_qubits_per_sys = Lx*Ly
n_qubits = n_qubits_per_sys*2
# -

# create a list of qubits that represents the grid
qubit_grid = get_qubit_grid(Lx, Ly)
qubit_grid

# define the plaquettes of the physical and auxiliary system
plaqs_wires, plaqs_wires_aux = get_plaquettes(qubit_grid)

# ## Creating the initial state

# we prepare the vacuum to the correct periodicity
parity_flips = (8, 9, 10, 15)
parity_flips

# prepare_full_G_state creates the vacuum state, where every plaquette is in a Gr eigenstate with eigenvalue +1
draw(prepare_full_G_state, n_qubits)(plaqs_wires_aux, parity_flips)

# we can verify that all Gr = 1 constraints are fulfilled
# for this, we define the function "verify_G_locals" which takes a circuit as input 
# and checks the constraint for all plaquettes
verify_G_locals(prepare_full_G_state, n_qubits, 
                plaqs_wires, plaqs_wires_aux, plaqs_wires_aux, parity_flips, in_state=None)

# +
# to make this more explicit, we will prepare each plaquette sequentially 
# in the correct eigenstate and verify that this is indeed fulfilled

# first fulfil the periodicity constraint
s_prev = state(prepare_periodicity_state, n_qubits)(parity_flips)
print("Periocity state:")
print_state(s_prev)
print()

# now iteratively loop over the plaquettes
for pi in reversed(range(len(plaqs_wires))):
    print("="*10, "new plaquette", "="*10)
    s = state(prepare_single_Gr_state, n_qubits, in_state=s_prev)(
        plaqs_wires_aux[pi], change_first=plaqs_wires_aux[pi][0] in parity_flips
    )
    print("prepared state")
    print_state(s)
    s_true = state(Gr_circuit, n_qubits, in_state=s)(
        plaqs_wires[pi], plaqs_wires_aux[pi])
    print("--- after applying Gr to the state ---")
    print_state(s_true)

    if not same_states(s, s_true):
        print("!!! ERROR: different states found for Gr on wires:",
              plaqs_wires[pi], plaqs_wires_aux[pi])

    s_prev = s
    print()
# -

# we create the vacuum state once to use it for testing later
vac_state_prepared = state(prepare_vacuum, n_qubits)(plaqs_wires_aux, parity_flips)

# print it to make sure it's the vacuum
print_state(vac_state_prepared)

# ## Insert fermion pairs

# +
# from now on, we insert particles in the following grid positions
init_idxs = [((0, 0), (0, 1))]

# get the corresponding qubit numbers
out = [
    (qubit_grid[i, init_idxs[0][0][0], init_idxs[0][0][1]], 
     qubit_grid[i, init_idxs[0][1][0], init_idxs[0][1][1]]) for i in (0,1)]
init_qubits, init_aux_qubits = out
init_qubits, init_aux_qubits
# -

# let's verify that when we insert fermions, we maintain the Gauss constraints
verify_G_locals(
    create_pair, n_qubits,
    plaqs_wires, plaqs_wires_aux,
    init_qubits, init_aux_qubits,
    in_state=vac_state_prepared)

# let's create the state once to continue
pair_in_vacuum_state = state(create_pair, n_qubits, in_state=vac_state_prepared)(init_qubits, init_aux_qubits)
print_state(pair_in_vacuum_state)

# ## Variational ansatz

qubit_grid

# +
# now let's test if we fulfil the Gauss laws if we hop the particles around along the x axis
# if there's no error, there is no problem
theta = np.random.uniform(high=np.pi)
phi = np.random.uniform(high=np.pi)

# let's hop along the following edge
test_qubits = (1, 2), (9, 10)

verify_G_locals(
    hop_particles_x, n_qubits,
    plaqs_wires, plaqs_wires_aux,
    *test_qubits, 
    theta, phi,
    in_state=pair_in_vacuum_state)

# +
# same for the y axis

# let's hop along the following edge
test_qubits = (0, 4), (8, 12)

verify_G_locals(
    hop_particles_y, n_qubits,
    plaqs_wires, plaqs_wires_aux,
    *test_qubits, 
    theta, phi,
    in_state=pair_in_vacuum_state)

# +
# let's test everything together for a depth 1 circuit
depth_test = 1
np_test = get_n_params(qubit_grid, depth_test)
print("number of parameters (overestimation) = ", np_test)
thetas = np.random.normal(size=(np_test,))
phis = np.random.normal(size=(np_test,))

# get the final quantum state from the parametrized circuit
verify_G_locals(
    construct_ansatz, n_qubits,
    plaqs_wires, plaqs_wires_aux,
    init_idxs, plaqs_wires_aux, parity_flips, qubit_grid, 
    thetas, phis, 
    circuit_kwargs={'n_layers': depth_test},
    in_state=None)
# -


# # VQE

# +
# create the hamiltonian to optimize (here: free fermions)
t_fh = 1.0
V_fh = 0.0

ham0 = fermihubbard(1, 1, qubit_grid)
print(ham0)

# +
# we will create an ansatz of depth 1 iterations
depth = 1

dev = qml.device("default.qubit.jax", wires=np.arange(n_qubits))

@qml.qnode(dev, interface="jax")
def cost_fn(params, ham):
    """ Cost function to minimize: energy of the system """
    thetas, phis = jnp.split(params, 2)
    construct_ansatz(init_idxs, plaqs_wires_aux, parity_flips, qubit_grid, thetas, phis, n_layers=depth)
    return qml.expval(ham)


# +
max_iterations = 50000
conv_tol = 1e-12
learning_rate = 1e-3

# random initial parameters
n_thetas = get_n_params(qubit_grid, depth)
print("Number of parameters (overestimation):", 2*n_thetas)
thetas = np.random.normal(size=(n_thetas,), scale=1e-3)
phis = np.random.normal(size=(n_thetas,), scale=1e-3)

# we minimize the energy
cost_fn_ham = lambda p: cost_fn(p, ham0)

params = jnp.concatenate((thetas, phis))

value_and_grad_circuit = jax.value_and_grad(cost_fn_ham, argnums=0)
value_and_grad_circuit = jax.jit(value_and_grad_circuit)

prev_energy = value_and_grad_circuit(params)[0]
print(f"Initial cost:   {prev_energy:.16f}")

opt = optax.adam(learning_rate)
opt_state = opt.init(params)

energy_logs = []
for n_iter in range(max_iterations):
    energy, grads = value_and_grad_circuit(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    energy_logs.append(energy)
        
    conv = np.abs(energy - prev_energy)

    print(f"Step = {n_iter},  Energy = {energy:.16f}")
    if n_iter > 0 and conv <= conv_tol and conv_tol != 0:
        print("Convergence reached")
        break
    prev_energy = energy

print(f"Tuned cost: {value_and_grad_circuit(params)[0]}")
# -

np.savetxt(f"vqe_params.txt", np.array(params))

plt.plot(energy_logs)
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.show()

# # Time evolution on this state (quench)

# we will simulate the time-evolution over n_tevo_steps with a new Hamiltonian (quench)
t_quench = 1.0
V_quench = 3.0

# +
dev = qml.device("default.qubit", wires=np.arange(n_qubits))

# we use the previous state as the initial state to time-evolve
# we fill in the parameters to simplify things
def init_circuit_prepper(params, depth, init_idxs):
    thetas, phis = jnp.split(params, 2)

    def _init_circuit():
        construct_ansatz(init_idxs, plaqs_wires_aux,
                         parity_flips, qubit_grid, thetas, phis, n_layers=depth)
    return _init_circuit

# actual time evolution circuit
def time_evo(delta_t=1e-3, n_steps=1):
    time_evo_circuit(qubit_grid, t_quench, V_quench,
                     delta_t=delta_t, n_steps=n_steps)

@qml.qnode(dev, interface="jax", diff_method='best')
def tevo_circuit_measurement(init_circuit, delta_t=1e-3, n_steps=1):
    # new circuit that time evolves the previous one
    init_circuit()
    time_evo(delta_t=delta_t, n_steps=n_steps)
    # let's measure the pauli Z operator for each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(Lx*Ly)]


# -

# load the vqe parameters
params = np.loadtxt("vqe_params.txt")
params

init_circuit = init_circuit_prepper(params, depth, init_idxs)

# time step size
delta_t = 1e-1
# number of trotter steps to simulate
n_trotter = np.arange(0, 20, 1)
n_trotter

# +
print("Time evolution")
print("--------------")

tevo_circuit = partial(tevo_circuit_measurement, init_circuit)

evolution_data = defaultdict(list)
for n_steps in n_trotter:
    print("Number of Trotter steps:", n_steps)
    measurements = tevo_circuit(delta_t=delta_t, n_steps=n_steps)
    
    # save the data in a structured way
    for i in range(len(measurements)):
        evolution_data[f"n{i}"].append((1-measurements[i])/2) # go to occupation numbers
    evolution_data["n_steps"].append(n_steps)
    evolution_data["time"].append(n_steps*delta_t)

# +
for i in range(len(measurements)):
    plt.plot(evolution_data["time"], evolution_data[f"n{i}"])
    
plt.xlabel("Time")
plt.ylabel("<n_i>")
plt.show()
# -


