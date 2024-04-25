# GKR
This repo contains a toy implementation of the GKR protocol, as modified and explicated in Chapter 4.6 of Thaler's book [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf).
## Guide
- circuit.py contains a circuit class and a createCircuit function, which creates a circuit from a text file.
- interactor_GKR.py contains an interactor class, which is a parent class of the prover and verifier class. The interactor class has a (deep) copy of the circuit as an internal variable.
- prover_GKR.py contains the prover class and verifier_GKR.py contains the verifier class.
- command_GKR.py contains the command of the protocol, where everything is run.
- circuit_data_i.csv are csv files that contain the data of circuits.
- deep_circuit_1.csv is a text file with the data of a deep, narrow circuit; deep_circuit_1.py transforms this into an instantiation of a circuit.

There are a couple other assorted files in this repo, which I wrote to understand other parts of Chapter 4 of the [book](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf).
- sumcheck_general.py contains a general implementation of the sumcheck protocol.
- sumcheck_efficient.py contains a partial implementation of the sumcheck protocol aimed at *super efficient* matrix multiplication. There is also an application to counting triangles. This corresponds to implementations of the protocols in 4.4 and 4.5 of the [book](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf).
