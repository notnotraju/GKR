#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:48:48 2022

@author: raju
"""



import numpy as np
#import math
#import random
import circuit


"""
Interactor class is a parent class for interactive agents.
In particular, it is a class that will have both a prover and a verifier
subclass. The goal of writing it is to uniformize the methods that the
prover and the verifier have in common. To initialize, it takes a circuit.
The internal variables are the following:
    circ: a (deep) copy of the circuit. the point is both the prover and the
        verifier will have access to this, but they should be different copies
        because the prover will actually compute the circuit (which amounts
        to filling out various parts of the layers)
    d: depth
    p: prime
    k: a list of length d such that 2^k[i] is the size of layer i of the 
        circuit
    random_vectors: a list of the random vectors for which the prover claims 
        to know:
                            \tilde{W}_i(random_vectors[i]) 
        (this is the claim at the beginning of sumcheck at each layer)
    evaluations_of_random_vectors: a list of the prover's claimed evaluations: 
        \tilde{W}_i(random_vectors[i])
    sumcheck_random_elements: a list of lists. The ith list is the list of
        random elements the verifier has sent the prover to run the sumcheck
        protocol for layer i.
        NOTE: in sumcheck, the verifier sends nothing to the prover at step 0.
        in our convention, sumcheck_random_elements[i][j-1] is the randomness
        that the verifier sends after step j on layer i.
    polynomials: a list of lists of ``polynomials''. more precisely, the ith
        element is a list of the (quadratc) polynomials that the verifier 
        sends doing sumcheck on layer i. Here, a polynomial is given by a list
        of integers of length 3 (they are quadratic): [a, b, c] corresponds to
        a + bx + cx^2.
    lines: a list of ``lines''. more precisely, at the end of the layer i
        sumcheck, we reduce two eval_MLEs to a single one via linear
        interpolation. the ith entry of lines is a *function*
            F_p -> F_p^{k_{i+1}}, 
        which is the line between b* and c* and the end of layer i.
"""

class Interactor:
    def __init__(self, C: circuit):
        self.circ = C.deepcopy()
        self.d = self.circ.get_depth()
        self.p = self.circ.get_p()
        self.k = self.circ.get_k()
        self.random_vectors = [] 
        self.evaluations_of_random_vectors = []

        self.sumcheck_random_elements = [[] for i in range(self.d)] 
        
        self.polynomials = [[] for i in range(self.d + 1)] 

        self.lines = []
    
    
    def get_circ(self):
        return self.circ
    def get_depth(self):
        return self.d
    def get_p(self):
        return self.p
    def get_k(self):
        return self.k
    def get_random_vectors(self):
        return self.random_vectors
    def get_random_vector(self, i):
        return self.get_random_vectors()[i]
    def get_evaluations_of_RV(self):
        return self.evaluations_of_random_vectors
    def get_evaluation_of_RV(self, i):
        return self.get_evaluations_of_RV()[i]
    def get_sumcheck_random_elements(self):
        return self.sumcheck_random_elements
    def get_layer_i_sumcheck_random_elements(self, i):
        return self.sumcheck_random_elements[i]
    def get_sumcheck_random_element(self, i, s):
        return self.get_layer_i_sumcheck_random_elements(i)[s-1]
    def get_polynomials(self):
        return self.polynomials
    def get_specific_polynomial(self, i, s):
        assert i>=0 and i<=self.get_depth(), "i is out of bounds"
        assert s>=0 and s<len(self.get_polynomials()[i]), "s is out of bounds"
        return self.get_polynomials()[i][s]
    def get_add_and_mult(self, i):
        return self.circ.get_add_and_mult(i)
    def get_lines(self):
        return self.lines
    def get_line(self, i):
        lines = self.get_lines()
        assert 0<= i <len(lines)
        return lines[i]
    def append_evaluations_RV(self, val):
        assert type(val)==int, "type of input is not an integer"
        self.evaluations_of_random_vectors.append(val)
    def append_RV(self, vec):
        self.random_vectors.append(vec)
    
    # NOTE TO SELF: not sure if I need this or not!! maybe.
    # SRE stands for ``sumcheck_random_element''
    def append_element_SRE(self, i, random_element):
        assert 0<=i and i<self.get_depth()
        self.sumcheck_random_elements[i].append(random_element)
    def append_sumcheck_polynomial(self, i, poly):
        assert 0<= i and i<=self.get_depth(), "i is out of bounds" #not sure about right hand bound
        self.polynomials[i].append(poly)
    def append_line(self, f):
        self.lines.append(f)

        
# at the end of the layer i sumcheck protocol, the verifier is left to ``compute'' or verify two
# MLEs: the two input vectors are the first and second half respectively of
# sumcheck_random_elements for layer i. Call these two elements b and c.
# compute_line returns a function F_p-->F_p^{k[i+1]} that is a line in between these two points.
    def compute_line(self, i):
        k = self.get_k()
        p = self.get_p()

        layer_i_random_elements = self.get_layer_i_sumcheck_random_elements(i)
        assert len(layer_i_random_elements) == 2 * k[i+1],\
            "the number of random elements the verifier has added to layer i is not 2*k[i+1]\
                instead, {} is not {}".format(len(layer_i_random_elements), 2 * k[i+1])
        b = layer_i_random_elements[:k[i+1]]
        c = layer_i_random_elements[k[i+1]:]
        def line(x):
            assert type(x)==int, "input to line function must be an integer"
            np_answer = (np.array(b) + (np.array(c)-np.array(b))* x) %p
            return tuple(int(c) for c in np_answer)
        return line

    