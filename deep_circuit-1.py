#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:35:10 2022

@author: raju
"""


import csv
def write_deep_circuit(d: int):
    f = open("deep_circuit-1.csv","w")
    circuit_in_a_list = []
    for i in range(d):
        circuit_in_a_list.append("{},2,add,0,1,mult,1,1\n".format(2*i))
        circuit_in_a_list.append("{},2,mult,0,0,add,0,1\n".format(2*i+1))
    circuit_in_a_list.append("{},2,input,1,input,2".format(2*d))
    for layer in circuit_in_a_list:
        f.write(layer)
    f.close()
    