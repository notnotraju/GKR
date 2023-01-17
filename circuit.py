#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 21:41:58 2022

@author: raju
"""


import math
import copy
import csv
import numpy as np

import sumcheck_util as SU




"""
class: circuit
A circuit is a data structure. It has the following internal
variables:
    d: the depth of the circuit. (this does not include the input layer!)
    k: a list of length d+1, where the i^{th} layer has 2^k[i] gates. (the input layer corresponds to k[d])
    L: a list of dictionaries that represents the circuit. We call these dictionaries layers.
        for i<d, the dictionary L[i] takes keys numbers between 0 and 2^{k[i]}-1 (representing gates)
        the corresponding value is a list:
            ["add"/"mult", [input gates], [val]], where [input gates] is a list of 
            length two and [val] is either a null list or a list of length 1, 
            depending on whether or not we have already computed the value
        if i==d, on key a gate the value is ("input", [], [val]), where the middle list
        is *always* the empty list (as there are no inputs to an input gate)

    p: a prime number.
These four variables are fixed at creation of a circuit.

we initialize a circuit with d, k, and p: L is initially set to a list of empty dictionaries.
To populate a circuit, we have methods "add_layer" and "add_input_layer", which take in a layer
(a dictionary L_i that has the right key-value structure), and we set self.L[i]:=L_i.

We also have a createCircuit function, which builds a circuit from a text file. The specifications will be
explained above this function.
    """
class Circuit:
    def __init__(self, d: int, k: list, p: int):
        assert d == len(k) - 1, "depth is not the same as the length of the list k"
        self.L = [dict() for i in range(d+1)]
        self.k = copy.deepcopy(k)
        self.d = d
        self.p = p
        
    def deepcopy(self):
        C = Circuit(self.get_depth(), copy.deepcopy(self.get_k()), self.get_p())
        C.L = copy.deepcopy(self.get_circuitData())
        return C
    
    def get_circuitData(self):
        return self.L
    
    def get_k(self):
        return self.k
    
    def get_p(self):
        return self.p
    
    def get_L(self):
        return self.L
    
    def get_layer(self,i):
        d = self.get_depth()
        assert i>= 0 and i<=d, "layer has to be between 0 and d"
        return self.L[i]
    
    def get_depth(self):
        return self.d
    
    # the type is either "add", "mult", or "input"
    def get_type(self, i, gate):
        d = self.get_depth()
        Di = self.get_layer(i)
        assert gate in Di, "gate is not in the layer you picked"
        return Di[gate][0]
    
    def get_value(self, i, gate):
        d = self.get_depth()
        assert i>=0 and i<=d, "layer has to be between 0 and d"
        Di = self.get_layer(i)
        assert gate in Di, "gate is not in the input layer you picked"
        assert len(Di[gate][2])==1, "value has not yet been filled in"
        return Di[gate][2][0] 
    
    def get_inputs(self, i, gate):
        d = self.get_depth()
        assert i<d, "layer must be a non-input layer of the circuit"
        Di = self.get_layer(i)
        assert gate in Di, "gate is not in the input layer you picked"
        return Di[gate][1]
    
    def get_input_values(self, i, gate):
        input_gate1, input_gate2 = self.get_inputs(i, gate)
        return self.get_value(i+1, input_gate1), self.get_value(i+1, input_gate2)
    
    
    """get_add_and_mult is a method that takes in a (non-input) layer and returns 2 dictionaries: 
        the addition dictionary (which encodes the add gates together with their inputs); and
        the multiplication dictionary (which encodes the multiplication gates together with their inputs)
    the format of the dictionary is the following: 
        {0,1}^k_i \times {0,1}^k_{i+1} \times {0,1}^k_{i+1} ---->{0,1}
        where we think of each tuple as representing a gate, either in layer i or layer i+1
    """
    def get_add_and_mult(self, i):
        d = self.get_depth()
        assert i>=0 and i<d, "layer can be neither an input layer nor out of bounds"
        Di = self.get_layer(i)
        k = self.get_k()
        k_i = k[i]
        k_iplus1 = k[i+1]
        
        add_i = dict()
        mult_i = dict()
        # addi is going to take a boolean tuple of length k_i + 2 * k_iplus1,
        # and spit out either 0 or 1. It spits out 1 if it is an add gate with the right inputs
        # else it spits out zero.
        
        for gate in range(2**k_i):
            for a in range(2**k_iplus1):
                for b in range(2**k_iplus1):
                    key = SU.int_to_bin(gate,k_i) + SU.int_to_bin(a, k_iplus1) + \
                        SU.int_to_bin(b, k_iplus1)
                    add_i[key] = 0
                    mult_i[key] = 0
            gate_type = self.get_type(i, gate)
            a, b = self.get_inputs(i, gate)
            key = SU.int_to_bin(gate,k_i) + SU.int_to_bin(a, k_iplus1) + \
                    SU.int_to_bin(b, k_iplus1)
            if gate_type == "add":
                add_i[key] = 1
            elif gate_type == "mult":
                mult_i[key] = 1
        return add_i, mult_i
    
    # this will allow fast computation of eval_MLE for the \tilde{add}
    # and \tilde{mult}. #FINISH THIS (will be used in f_i code, and also)
    # verifier code. NOTE: this is not a general ``eval_MLE'' type
    # of thing, it relies on the fact that we are stacking computation
    # and mostly computing \tilde{add} of \emph{structured} inputs,
    # that e.g. end in boolean.
    # NOTE AS OF JULY 23: this is probably not exactly necessary, or rather
    # it needs to be modified. Just as in MatMul, it is
    def eval_MLE_add(self, i, x):

        add_i = self.get_add_and_mult(i)[0]
        k = self.get_k()
        p = self.get_p()
        N = k[i] + 2 * k[i+1]
        assert N == len(x), "length of vector is not correct"
        answer = 0
        for gate in range(2**k[i]):
            if self.get_type(i, gate) == "add":
                
                first_input, second_input = self.get_inputs(i, gate)
                
                w = SU.int_to_bin(gate, k[i]) +\
                    SU.int_to_bin(first_input, k[i+1])+\
                     SU.int_to_bin(second_input, k[i+1])

                answer = (answer + SU.chi(w, x, N, p)) % p
        return answer

    def eval_MLE_mult(self, i, x):

        add_i = self.get_add_and_mult(i)[1]
        k = self.get_k()
        p = self.get_p()
        N = k[i] + 2 * k[i+1]
        assert N == len(x), "length of vector is not correct"
        answer = 0
        for gate in range(2**k[i]):
            if self.get_type(i, gate) == "mult":
                first_input, second_input = self.get_inputs(i, gate)
                
                w = SU.int_to_bin(gate, k[i]) +\
                    SU.int_to_bin(first_input, k[i+1])+\
                     SU.int_to_bin(second_input, k[i+1])

                answer = (answer + SU.chi(w, x, N, p)) % p
        return answer
    
    # the method get_W does the following. Given a layer i, we return the function
    # on the boolean hypercube given by the the values of the gates on that layer.
    def get_W(self, i):
        d = self.get_depth()
        assert i>=0 and i<= d, "layer must be within bounds"
        k = self.get_k()
        k_i = k[i]
        W_i = dict()
        for gate in range(2**k_i):
            W_i[SU.int_to_bin(gate, k_i)] = self.get_value(i, gate)
        return W_i
                
                        
                        

                    
    # place_value takes in a layer and a gate and inserts a value at that gate.
    def place_value(self, i, gate, val):
        d = self.get_depth()
        assert i>=0 and i<=d, "layer has to be between 0 and d"
        Di = self.get_layer(i)
        assert gate in Di, "gate is not in the layer you picked"
        Di[gate][2] = [val]
        
        
    # add_layer takes in a (non-input) layer and a dictionary, checks that the
    # the dictionary has the correct form to be a layer, and then adds to the circuit.
    def add_layer(self, i: int, D_i: dict):
        d = self.get_depth()
        k = self.get_k()
        assert 0 <= i < d, "layer is out of bounds. use add_input_layer\
                                if you wish do have i == d."
        #gates is a list of the keys, gates_info is a list of the values
        gates = list(D_i.keys())
        gates_info = list(D_i.values())
        
        #check the typing and information of gates_info
        for info in gates_info:
            assert len(info) == 3, "one of the values has the wrong length" 
            assert info[0] in ["add", "mult", "input"], "one of the values has an invalid gate type"
            assert len(info[1])==2 and 0 <= info[1][0] < 2**k[i+1]\
                       and 0 <= info[1][1] < 2**k[i+1],\
                       "there is a problem with the input gate numbering"
            assert len(info[2]) <= 1, "there are too many values!"
        gates.sort()  
        assert gates == [i for i in range(0,2**self.get_k()[i])], "Dictionary has the wrong number of elements"
        self.L[i] = copy.deepcopy(D_i)
    # add_input_layer adds an input layer.
    def add_input_layer(self, D_inp: dict):
        d = self.get_depth()
        k = self.get_k()
        gates = list(D_inp.keys())
        gates_info = list(D_inp.values())
        
        #check the typing and information of gates_info
        for info in gates_info:
            assert len(info) == 3, "one of the values has the wrong length" 
            assert info[0] == "input", "one of the values has an invalid input type"

            assert len(info[1])==0,"input gates can have no input"

            assert len(info[2]) == 1, "each input gate has exactly one value"
        gates.sort()  
        assert gates == [i for i in range(0,2**k[d])], "Dictionary has the wrong number of elements"
        self.L[d] = copy.deepcopy(D_inp)
        
    # compute_layer takes in a layer and computes the values of that layer from the previous layer.
    # this in particular involves the place_value method, to add the value to a gate in our layer.
    # if the previous layer has not yet been filled in, we return an assert error.
    def compute_layer(self, i: int):
        p = self.get_p()
        d = self.get_depth()
        assert i>=0 or i<d, "layer must be between 0 and d-1"
        Di = self.get_layer(i)
        for gate in Di.keys():
            if self.get_type(i, gate) == "add":
                input_value1, input_value2 = self.get_input_values(i, gate)
                self.place_value(i, gate, (input_value1+input_value2) % p)
            elif self.get_type(i, gate) == "mult":
                input_value1, input_value2 = self.get_input_values(i, gate)
                self.place_value(i, gate, (input_value1 * input_value2) % p)
    # compute_circuit simply computes fills in all the values of the circuit layer by layer, using
    # compute layer
    def compute_circuit(self):
        d = self.get_depth()
        for i in range(d-1,-1, -1):
            self.compute_layer(i)
            
    
        
                
    
    
# example_circuit is an example of a test circuit. 

def example_circuit():
    a = Circuit(1,[2,2], 97)
    a.add_layer(0, {1:["add",[1,1],[]], 2:["mult",[2,1],[]], 3:["add",[3,0],[]], 0:["add",[0,1],[]]})
    a.add_input_layer({0:["input",[],[1]],\
                       1:["input",[],[3]],\
                       2:["input",[],[4]],\
                       3:["input",[],[10]]})
    return a

def build_random_circuit_of_depth_d(d, max_k, max_input, p):
    assert d>0 and max_k>0 and max_k <10,\
        "depth must be positive, and max_k must be in between 0 and 10."
    k = [np.random.randint(1,max_k+1) for i in range(d+1)]
    C = Circuit(d, k, p)
#    print(k)
    random_gate_determiner = ["add", "mult"]
    
    for i in range(d):
#        print("adding layer {}".format(i))
        layer_i = dict()
        for gate in range(2**k[i]):
            coin = np.random.randint(0,2)
            current_input_gates = [np.random.randint(0,2**k[i+1])\
                                   for bit in range(2)]
#            print(current_input_gates)
            layer_i[gate] =\
                [random_gate_determiner[coin], current_input_gates, []]
        C.add_layer(i, layer_i)
    
    input_layer = dict()
    for gate in range(2**k[d]):
        current_input_value = np.random.randint(0,max_input)
        input_layer[gate] = ["input",[],[current_input_value]]
    C.add_input_layer(input_layer)
    return C
        
    


"""createCircuit takes in a file name and a prime and returns a circuit.

The file must be in the following format.
line i: i, # of gates, type, input, input, type, input, input, ...
        (# of gates is 2^k[i], in the language of our circuit)
final line: d, # of gates, "input", value, "input", value, ...

"""
def createCircuit(fileName: str, p: int)-> Circuit:
    file = open(fileName, mode ='r')
    csvFile = csv.reader(file)
    circuitData =[]
    k = []
    for line in csvFile:
        circuitData.append(line)
        k.append(int(math.log2(int(line[1]))))
    d = len(circuitData) - 1
    C = Circuit(d, k,p)
    
    # check to make sure circuitData passes some basic sanity checks.
    # check each non-input layer. If it passes, add the layer to C.
    for i in range(d):
        line = circuitData[i]
        assert i == int(line[0]), "the {}th line of the document does not start with {}".format(i,i)
        numGates = int(line[1])
        layer = dict()
        for j in range(numGates):
            gateType = line[2 + 3*j].strip()

            assert gateType in ["mult", "add"],\
            "The only allowable gates are mult and add"
            gateInput1 = int(line[2 + 3*j +1])
            gateInput2 = int(line[2 + 3*j +2])
            layer[j] = [gateType, [gateInput1, gateInput2], []]
        
        C.add_layer(i, layer)
    
    # check input layer: i =d . If it passes, add the layer to C.
    for i in range(d,d+1):
        line = circuitData[i]
        assert i == int(line[0]), "the {}th line of the document does not start with {}".format(i,i)
        numGates = int(line[1])
        layer = dict()
        for j in range(numGates):
            gateType = line[2 + 2 * j].strip()
            assert gateType == "input",\
            "On the last layer, only have input gates!"
            gateValue = int(line[2 + 2*j +1])
            layer[j] = [gateType, [], [gateValue]]
        C.add_input_layer(layer)
    file.close()
    return(C)

            

    
    
    
    
    
    