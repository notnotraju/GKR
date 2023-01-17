#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 21:38:43 2022

@author: raju
"""



#import numpy as np
#import math
#import random
#import time
#import copy
#import matplotlib.pyplot as plt
#import csv

import sumcheck_util as SU
import circuit
import prover_GKR as P_GKR
import verifier_GKR as V_GKR


"""execute GKR for a circuit C"""

def execute(C):
    k = C.get_k()
    d = C.get_depth()
    # initialize prover and verifier
    prover_inst = P_GKR.Prover(C)
    verifier_inst = V_GKR.Verifier(C)
    # prover_output_communication is the first message the prover sends.
    # This is a dictionary of the output values, in dictionary form: {0,1}^{k[0]}->F_p
    prover_output_communication = prover_inst.output_layer_communication()
    # verifier runs ``output layer communication'' with input the dictionary 
    # that prover just sent. returns a random vector r_0 in F_p^{k[0]}
    random_vector_0 = verifier_inst.output_layer_communication(prover_output_communication)
    prover_inst.receive_random_vector(0, random_vector_0)
    print("At layer 0, the random value provided by the verifier is", random_vector_0)
    print("The value of the multilinear extension at ", random_vector_0," is", prover_inst.get_evaluations_of_RV()[0])

    # iterate over the layers
    for i in range(d):
        r = 0
        for s in range(2 * k[i+1] + 1):
            prover_msg = prover_inst.partial_sumcheck(i, s, r)
            string_of_prover_msg =\
                "+".join(\
                         ["{}*x^{}".format(prover_msg[l], l)\
                          for l in [2,1,0]])
            print("at layer {} step {}, the polynomial the prover sends is is {}".format(i ,s, string_of_prover_msg))
            r = verifier_inst.partial_sumcheck_check(i, s, prover_msg)
            if s!=0:
                print("at layer {} step {}, verifier's randomness is {}".format(i, s, r))
        # end_of_sumcheck_poly is what the prover claims \tilde{W}_i restricted to the line is.
        end_of_sumcheck_poly = prover_inst.send_Wi_on_line(i, r)
        print("The univariate polynomial that the prover sends at the end of step {} on the line is: {}".\
              format(i, SU.string_of_polynomial(end_of_sumcheck_poly)))
        new_random_vector = verifier_inst.reduce_two_to_one(i, end_of_sumcheck_poly)
        prover_inst.receive_random_vector(i+1, new_random_vector)
    verifier_inst.final_verifier_check()
    print("we win!!!")

C = [circuit.createCircuit("circuitdata-{}.csv".format(i),10007) for i in range(1,5)] 
