#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 20:58:15 2022

@author: raju
"""




from interactor_GKR import Interactor

import sumcheck_util as SU
import circuit


"""
Prover class is a subclass of the interactor. Given the base interactor, we also
compute the circuit. There are no extra internal variables.
"""
class Prover(Interactor):
    def __init__(self, C: circuit):
        Interactor.__init__(self, C)

        self.circ.compute_circuit()
        
    def output_layer_communication(self):
        return self.circ.get_W(0)
    
    """
    receive_random_vector
    INPUT: i (int), r_i (tuple)
    receive_random_vector appends the random vector r_i to random_vectors
    and also appends the correct evalution of W_i.
    Here, i is the layer.
    """
    def receive_random_vector(self, i: int, r_i: tuple):
        d = self.d
        k = self.k
        p = self.p
        assert i>=0 and i<=d, "the layer must be between 0 and d-1"
        self.random_vectors.append(r_i)
        D_i = self.circ.get_W(i)
        evaluation_at_random_vector = SU.eval_MLE(D_i, r_i, k[i], p)
        self.append_evaluations_RV(evaluation_at_random_vector)


    """
    sum_fi 
    INPUTS: i (integer), s (step)
    OUTPUTS: a quadratic polynomial, in the form of a list, i.e. [a, b, c],
        which corresponds to a + bx + cx^2
    
    sum_fi computes the relevant partial boolean hypercube sum that the prover
    must compute at the sth step of the sumcheck protocol on the ith layer. the function returns
    a quadratic polynomial.
    
    the sumcheck is done on a function fi, but we do not independently compute fi; rather,
    we use an optimization.
    here, f^{(i)}_{random_vector[i]} is a function on 2 * k[i+1] variables. (there is an 
    implicit parameter, random_vector[i]!! without this implicit parameter,
    the function would be of k[i]+ 2 *k[i+1] variables, but we fix the first k[i]
    inputs!!)
    """    
    def sum_fi(self, i: int, s: int):
        circ = self.get_circ()
        d = circ.get_depth()
        p = circ.get_p()
        k = circ.get_k()
        assert i>=0 and i<d, "i is out of bounds"
        assert i<len(self.get_random_vectors()), "haven't reached this layer yet"
        assert s>=1 and s<=2 * k[i+1], "the step s in sumcheck is out of bounds"
        poly_values = [0,0,0] #initialize the values of my poly at the inputs 0, 1, 2
#        Li = circ.get_layer(i)
        N = k[i] + 2* k[i+1]
        # the following is the main content of the partial boolean hypercube sum
        # the goal is to compute poly_values. We use a fast evaluation trick:
        # involving summing over the gates. (This is explained in 4.6.5, method 2,
        # of Thaler's book as of 09/04/2022.)
        for gate in range(2**k[i]):
            gate_inputs = circ.get_inputs(i, gate)
            # a is a boolean string in {0,1}^k[i] \times {0,1}^{2*k[i+1]}
            # which consists (or runs over) triples where the second two parts
            # are input gates to the first piece.
            a = SU.int_to_bin(gate, k[i]) +\
                SU.int_to_bin(gate_inputs[0],k[i+1]) +\
                SU.int_to_bin(gate_inputs[1],k[i+1])
            for x in range(3):
                current_random_elements = self.get_layer_i_sumcheck_random_elements(i)
                # bc = (r_0, ..., r_{s-2}, {0/1/2}, b_{s},..b_{k[i]}-1) (as in Method
                # 2 in 4.6.5, with appropriate indices changed)
                # the point is we only need to sum over such tuples that agree with
                # the last (however many) bits of a!! 
                bc = tuple(current_random_elements[:s-1])+\
                    (x,)+\
                    tuple(a[-(2*(k[i+1]) - s):])
                b = bc[:k[i+1]]
                c = bc[k[i+1]:]
  
                z = tuple(self.get_random_vector(i)) + bc
                W_iplus1 = circ.get_W(i+1)
                W_iplus1_at_b = SU.eval_MLE(W_iplus1, b, k[i+1], p)
                W_iplus1_at_c = SU.eval_MLE(W_iplus1, c, k[i+1], p)
                gate_type = circ.get_type(i, gate)
                if gate_type == "add":
                    poly_values[x] = (poly_values[x]+\
                        SU.chi(a, z, N, p) *\
                        (W_iplus1_at_b + W_iplus1_at_c)) %p
                elif gate_type == "mult":
                    poly_values[x] = (poly_values[x]+\
                        SU.chi(a, z, N, p) *\
                        (W_iplus1_at_b * W_iplus1_at_c)) %p
        poly = SU.quadratic_interpolate(poly_values, p)
        return poly
          
    """
    partial_sumcheck
    INPUT: i (integer), s (integer), random_element (integer)
    OUTPUT: poly (list)
    
    performs step s of sumcheck for layer i of the circuit.
    here, sumcheck is done with respect to the polynomial in Eq. 4.17 (section 4.6.3: Protocol overview)
    of Thaler's book. this polynomial, which we will call f^{i}_{r_i}, is a polynomial in
    2 * k[i] variables
    i is the layer in the circuit, s is the step in the sumcheck protocol
    random_element is the random element of F_p that the verifier sends. (in
    case of the first communication, we'll  default this to 0.)
    returns a quadratic polynomial in the form [a, b, c], which corresponds to
    a + bx + cx^2
    """

    def partial_sumcheck(self, i: int, s: int, random_element: int):
        d = self.d
        k = self.k
        assert i>=0 and i<d, "i is out of bounds"

        assert s>=0 and s<= 2* k[i+1], "step must be between 0 and 2*k_{i+1}"
        RV_i = self.get_random_vector(i)
        
        # at step = 0, we are starting the sumcheck protocol and
        # have to send back the actual value of 
        # \tilde{W}_i(value_of_random_vectors[i]).
        # (NOTE: no sum is required.)
        if s == 0:
            new_evaluation = self.get_evaluation_of_RV(i)
            print("The multi-linear extension of W_{} at {} is {}".\
                  format(i, RV_i, new_evaluation))
            return [new_evaluation, 0, 0] # do I need to do this?? Probably...
        elif s == 1:
            poly = self.sum_fi(i, s)
            self.append_sumcheck_polynomial(i, poly)
            return poly
        elif s<= 2*k[i+1]:
            self.append_element_SRE(i, random_element)
            poly = self.sum_fi(i,s)
            self.append_sumcheck_polynomial(i, poly)
            return poly
          
    # NOTE: we're appending the last random element, to fill out the sumcheck random elements. Write in specs!
    """
    send_Wi_on_line
    INPUTS: i (integer), random_element (integer)
    outputs: poly (list)
    At the end of the layer i sumcheck protocol, the Prover must send the
    polynomial \tilde{W}_{i} on the line between bstar and cstar that both
    the prover and the verifier independently compute. (This is implemented
    in the Interactor class.)
    On input i (layer #) and random_element (the last randomness that the
    Verifier sends), we append the random_element to our SRE, and then compute
    the univariate polynomial poly that is \tilde{W}_i|_{line}.
    
    NOTE: we have to append the random_element because in our implementation,
    when the verifier sends a random element, only in the next call does the
    prover add it to its internal state.
    """
    def send_Wi_on_line(self, i: int, random_element: int):
        p = self.get_p()
        k = self.get_k()

        self.append_element_SRE(i,random_element)
        line = self.compute_line(i)
        W_iplus1 = self.circ.get_W(i+1)
        poly = SU.polynomial_interpolation(\
                [[N, SU.eval_MLE(W_iplus1, line(N), k[i+1], p)] for N in range(k[i+1]+1)],
                k[i+1], p)
        # deg_of_poly = len(poly)-1
        # string_of_poly = "+".join(\
        #                           ["{}*x^{}".format(poly[k],deg_of_poly - k) for k in range(deg_of_poly+1)])
        # print("The univariate polynomial that the prover sends at the end of step {} on the line is: {}".\
        #       format(i, string_of_poly))
        return poly
    
    """
    send_final_Wd_evaluation
    OUTPUT: \tilde{W}_d(random_vector[d])
    This is what the prover's sends at the end of the protocol: it is
    the value of \tilde{W}_d (i.e., the MLE of the input) on the random vector that
    has been generated by the protocol. 
    """
    def send_final_Wd_evaluation(self):
        d = self.get_depth()
        return self.get_evaluation_of_RV(d)
    
    
    

            
        
        
        
