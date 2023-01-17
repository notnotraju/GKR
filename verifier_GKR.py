#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 21:49:06 2022

@author: raju
"""



import numpy as np
# import math
# import random


import sumcheck_util as SU
import circuit
from interactor_GKR import Interactor
"""
Verifier is a subclass of interactor. 
The only extra internal variable is: claimed_values_at_end_of_layer. This is a list,
whose i^th entry is the value that the prover implicit claims is \tilde{W}_i(random_vector[i])
through the protocol (i.e., it is the evaluation of a univariate polynomial at 
a random number that the verifier chooses.)
"""
class Verifier(Interactor):
    def __init__(self, C: circuit):
        Interactor.__init__(self, C)
        self.claimed_values_at_end_of_layer = []
    def get_claimed_values_at_end_of_layer(self):
        return self.claimed_values_at_end_of_layer
    def get_claimed_value_at_end_of_layer(self, i):
        assert i>=0 and i< 2 * self.get_depth(), "the layer i must be between {} and {}".\
            format(1, 2*self.get_depth())
        return self.claimed_values_at_end_of_layer[i]
    def append_claimed_values_at_end_of_layer(self, val):
        self.claimed_values_at_end_of_layer.append(val)
    def output_layer_communication(self, D):
        k = self.circ.get_k() # k is a list of the log (base 2) of number of gates
                                #in each layer. 
        p = self.p
        first_random_vector = tuple([np.random.randint(0,p) for i in range(k[0])])
        self.random_vectors.append(first_random_vector)
        value_at_first_random_vector = SU.eval_MLE(D, first_random_vector, k[0], p)
        self.append_evaluations_RV(value_at_first_random_vector)
        return first_random_vector
    
    
    """
    partial_sumcheck_check
    INPUTS: i (integer), s (integer), poly (list)
    OUTPUTS: new_random_element
    
    This is the verifier's side of the sumcheck protocol. Here, i is the layer,
    s is the step in the sumcheck within the layer, and poly is the last thing that
    the prover sent to us. (For the initial message, which is simply a number, we
    had the prover send it over as [val, 0, 0], i.e., the constant quadratic polynomial)
    
    if the checks are satisfied, we have the verifier send fresh randomness.
    """
    
    def partial_sumcheck_check(self, i: int, s: int, poly: list):
        p = self.p
        d = self.d
        k = self.get_circ().get_k()
        assert i>=0 and i<d, "i is out of bounds" 
        assert s>=0 and s<= 2* k[i+1], "step must be between 0 and 2*k_{i+1}"
        # we will separate out three cases: s = 0, s = 1, and all other cases.
        if s == 0:
            # poly represents the *value* that the prover is claiming. I.e.,
            # Prover claims \tilde{W}_i(last_random_vector) = val (in the form of
            # [val, 0, 0]). He will spend the rest of the sumcheck locally verifying this
            # and eventually reducing it to a claim about \tilde{W}_{i+1}
            if i>= 2:
                assert poly[0] == self.get_claimed_value_at_end_of_layer(i-1),\
                    "The claimed value at the end of step {}, {} does not match with what the prover just sent, {}".\
                        format(i-1, self.get_claimed_value_at_end_of_layer(i-1), poly[0])
            self.append_sumcheck_polynomial(i, poly)     
            #if s == 0, don't return anything
            return 0
        elif s == 1:
            #first, check compatibility of the 0th and first poly.
            sum_new_poly_at_0_1 = (SU.quadratic_evaluation(poly, 0, p) + SU.quadratic_evaluation(poly, 1, p)) %p
            old_value = SU.quadratic_evaluation(self.get_specific_polynomial(i, s-1), 0, p)
            assert sum_new_poly_at_0_1 == old_value % p, \
                "the first check failed, {} is not equal to {}".format(\
                                         sum_new_poly_at_0_1, old_value)
#            print("layer {} step 1 succeeded!".format(i))
            self.append_sumcheck_polynomial(i, poly)
            new_random_element = np.random.randint(0,p)
            self.append_element_SRE(i, new_random_element)
            return new_random_element
        elif 1 < s <= 2 * k[i+1]:
            r = self.get_sumcheck_random_element(i, s-1)
            sum_new_poly_at_0_1 = (SU.quadratic_evaluation(poly, 0, p) + SU.quadratic_evaluation(poly, 1, p)) %p
            old_value = SU.quadratic_evaluation(self.get_specific_polynomial(i, s-1), r, p)
            assert sum_new_poly_at_0_1 == old_value % p, \
                "the check failed at step {}, {} is not equal to {}".format(\
                                         s, sum_new_poly_at_0_1, old_value)
#            print("layer {} step {} succeeded!".format(i, s))
            self.append_sumcheck_polynomial(i, poly)
            new_random_element = np.random.randint(0,p)
            self.append_element_SRE(i, new_random_element)
            
            # if we are at the last step of sumcheck for layer i, then
            # compute+append the line between bstar and cstar.
            if s == 2 * k[i+1]:
                f = self.compute_line(i)
                self.append_line(f)

            return new_random_element

    """
    reduce_two_to_one
    INPUTS: i (integer), poly (list)
    OUTPUTS: new_random_vector (tuple)
    At the end of the sumcheck protocol for layer i, we have just received a
    polynomial, poly, that the prover claims to be \tilde{W}_{i+1} restricted to the line
    between bstar and cstar. More precisely:
        poly(0) is claimed to be \tilde{W}_{i+1}(bstar) and
        poly(1) is claimed to be \tilde{W}_{i+1}(cstar)
    We may use this to construct a "current" claim about what f^i_{RV_i} is at
    (bstar, cstar). There is a second, "old" claimed value: the last quadratic
    polynomial sent, evaluated at the last random element the verifier picked.
    If this passes, then the verifier picks a random element e in F_p, gets
    a new_random_vector via the line function, and binds the Prover to poly(e).
    We have therefore reduced ourselves to a statement about \tilde{W}_{i+1}.
    """
    def reduce_two_to_one(self, i: int, poly: list):
        p = self.get_p()
        k = self.get_k()
        circ = self.get_circ()
        vals = [SU.polynomial_evaluation(poly, i, p) for i in range(2)]
        SRE_layer_i= self.get_layer_i_sumcheck_random_elements(i)
        bstar = tuple(SRE_layer_i[:k[i+1]])
        cstar = tuple(SRE_layer_i[k[i+1]:])
        RV_i = tuple(self.get_random_vector(i))
        last_poly = self.get_specific_polynomial(i,2*k[i+1])
        add_bstar_cstar = circ.eval_MLE_add(i, RV_i + bstar + cstar)
        mult_bstar_cstar = circ.eval_MLE_mult(i, RV_i + bstar + cstar)
        
        # compute what the prover claims f_i(SRE_layer_i) is based on 
        # what the prover claims W_{i+1}(bstar) and W_{i+1}(cstar) are.
        # (this is via the polynomial that the prover sends!!)
        
        current_claimed_value_of_fi = \
            (add_bstar_cstar * (vals[0] + vals[1]) +\
            mult_bstar_cstar * (vals[0] * vals[1])) % p
        old_claimed_value_of_fi = SU.quadratic_evaluation(last_poly, SRE_layer_i[-1], p)
        assert current_claimed_value_of_fi == old_claimed_value_of_fi,\
            "The first check at the end of sumcheck for layer {} failed:\
            there is an imcompatibility between the last polynomial and the claimed\
            values of \tilde{W}_{i+1}(bstar) and \tilde{W}_{i+1}(cstar)\
                {}!={}".format(i, current_claimed_value_of_fi, old_claimed_value_of_fi)
        
        print("The two claimed values of f^{} (with random vector {}) at {} agree: {} and {}".\
              format(i, self.get_random_vector(i), SRE_layer_i, old_claimed_value_of_fi,\
              current_claimed_value_of_fi))
        line = self.get_line(i)
        final_random_element_in_layer = np.random.randint(0,p)
        new_random_vector = line(final_random_element_in_layer)
        self.append_RV(new_random_vector)
        self.append_claimed_values_at_end_of_layer(\
            SU.polynomial_evaluation(poly, final_random_element_in_layer, p))
        
        return new_random_vector
    """
    final_verifier_check
    OUTPUTS: True if we get to the end
    Here, we compare the two claimed values of \tilde{W}_d(RV[d])
    """
    def final_verifier_check(self):
        d = self.get_depth()
        circ = self.get_circ()
        p = self.get_p()
        k = self.get_k()
        Wd = circ.get_W(d)
        RV_d = tuple(self.get_random_vector(d))
        last_claimed_value = self.get_claimed_value_at_end_of_layer(d-1)
        actual_value_at_RV = SU.eval_MLE(Wd, RV_d, k[d], p)
        assert last_claimed_value == actual_value_at_RV,\
            "{} is not equal to {}".format(last_claimed_value, actual_value_at_RV)
        return True
        
        
