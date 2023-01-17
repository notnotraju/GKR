#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 21:25:47 2022

@author: raju
"""

import numpy as np
import math
import random




from timeit import default_timer as timer

"""
int_to_bin 
Inputs: i (integer), d (integer)
Outputs: tuple of the binary representation of i, with a total
    of d digits, if information is correctly constructed.

"""
def int_to_bin(i: int, d: int)->tuple:
    if i<0 or 2**d<i:
        print("out of bounds")
#        return tuple([])
    assert i>= 0 and i<= 2**d
    
    str_bin = bin(i)[2:] # bin = '0b...'
    if len(str_bin)<d:
        str_bin = '0'*(d-len(str_bin)) + str_bin
    #added to correctly deal with d = 0:
    if d==0:
        return tuple([])
    return tuple([int(i) for i in str_bin])
    
    
    
    

"""
build_function_from_matrix
INPUTS: an np array M and an integer n, such that M has dimension nxn.
OUTPUT: function_on_hypercube (dictionary), 2 * digit (integer)
        function_on_dictionary encodes the values of the matrix as a function on a 
        boolean hypercube {0,1}^digits x {0,1}^digits. More precisely, the keys
        of function_on_hypercube are boolean tuples of length 2* digits,
        where digits is the ceiling of log_2(n)
        the second output is 2 * digits = dimension of boolean hypercube.

"""
    
    
def build_function_from_matrix(M: np.array, n: int)->(dict,int):
    
    assert M.shape == (n,n), "The matrix M has the wrong dimensions"
    
    #new_mat_dim is the padded matrix dimension, which means that we will pad
    #until the dimensions are powers of 2.
    new_mat_dim = 2 ** (math.ceil(math.log2(n)))
    digits = math.ceil(math.log2(n))
    function_on_hypercube = dict()
    for i in range(new_mat_dim):
        for j in range(new_mat_dim):
            #index is the concatenation of i and j in binary,
            #in tuple form. That is, it is a tuple of length
            #2 * new_dim. (note: the binary expansions of i and
            #j have leading 0s.)
            index = int_to_bin(i,digits)+int_to_bin(j,digits)
            if i<n and j<n:
                function_on_hypercube[index] = M[i][j]
            else:
                function_on_hypercube[index] = 0
    #return function as a dictionary, and the "dimension", i.e,
    #the function is on {0,1}^{2*new_dim}, we return function as dict,
    #and the dimension of the boolean hypercube.
    return function_on_hypercube, 2 * digits





"""
quadratic_interpolate
INPUT: values, p
        values is a list of three integers
        p is a prime.
OUTPUT:
        answer (list), which is a list of length 3
        that represents the coefficients of the unique
        quadratic polynomial q such that:
            q(0)=values[0], q(1)=values[1], and q(2)=values[2]
        the elements of answer are in increasing degree order (i.e.,
        the constant coefficient is the first)

"""
def quadratic_interpolate(values: list, p: int)->list:
    assert len(values) == 3, "the list values does not have 3 elements"

    # answer will be in terms of lowest to highest.
    A = [values[0] * pow(2,-1, mod=p)%p,\
         -values[1]%p,\
         values[2]* pow(2,-1, mod=p)%p  ]
    answer = [values[0]%p, (-3*A[0]-2*A[1]-A[2]) %p, (A[0]+A[1]+A[2])%p]
    return answer



"""
quadratic_evaluation
INPUT: g (list), x, p
        where g are the coefficients of a quadratic polynomial
        x is an integer, and p is a prime
OUTPUT: g(x) mod p
"""
def quadratic_evaluation(g: list, x: int, p: int)->int:
    assert len(g)==3, "the list of coefficients of the polynomial does not have\
                        only 3 entries"
    return (g[0]+g[1]*x + g[2]*(x**2)) % p


def Lagrange_basis(xcoords: list, n: int, p: int)->list:
    assert len(xcoords) <= n+1, "n is too big to be uniquely determined by a list of this length"
    LB = []
    for i in range(n+1):
        current_polynomial = [1]
        current_denom = 1
        for j in range(n+1):
            if j != i:
#                print(j)
#                print(len(xcoords))
                current_polynomial = np.polymul(current_polynomial, [1, -xcoords[j]])
                current_denom = (current_denom * pow(xcoords[i]-xcoords[j],-1,p)) % p
        current_polynomial = (current_polynomial * current_denom) % p
        LB.append(current_polynomial)
    return LB
        
    
"""polynomial_interpolation takes in a list of (xcoord,ycoord), a degree
and a prime number p, and spits out the np.polynomial that interpolates
"""    

def polynomial_interpolation(values: list, n: int, p: int):
    xcoords = [pair[0] for pair in values]
    ycoords = [pair[1] for pair in values]
    LB = Lagrange_basis(xcoords, n, p)
    answer = [0]
    for i in range(len(LB)):
        weighted_poly = (LB[i] * ycoords[i]) % p
        answer = np.polyadd(answer, weighted_poly) % p
    return answer

def polynomial_evaluation(poly, x, p):
    reverse_poly = poly[::-1]
    answer = 0
    for i in range(len(poly)):
        answer = (answer + pow(x,i,p) * reverse_poly[i] )% p
    return answer

def string_of_polynomial(poly):
    deg_of_poly = len(poly)-1
    string_of_poly = "+".join(\
                    ["{}*x^{}".format(poly[k],deg_of_poly - k)\
                    for k in range(deg_of_poly+1)])
    return string_of_poly

    
"""chi
INPUTS: L (dict), a (boolean tuple), z (tuple of integers), N, p (prime)
    this returns the value of the MLE of the ``delta'' function L[a]* \delta_a
    at z. In other words, this is simply:
    \prod_{i=1..N} (a[i]*z[i] + (1-a[i])*(1-z[i])) mod p
"""
def chi(a: tuple, z: tuple, N: int, p: int):
    answer = 1
    for i in range(N):
        next_term = a[i] * z[i] + (1-a[i])*(1-z[i]) % p
        answer = answer * next_term % p
    return answer

"""
eval_MLE
INTPUTS: L (dict), r (tuple or list), N (int), p (int)
        Here, L is the dictionary that has keys in (0,1)^N
        r will be the vector in F_p^n that we are evaluating
        our multi-linear extension on
        N is the dimension of the boolean hypercube
        p is the prime number, with respect to which we work
OUTPUTS: answer
        which is \tilde{L}(r), i.e., the value of the (unique)
        MLE on input r.
        
NOTE: this algorithm may be found in Section 3.5 of Thaler's book.
"""

def eval_MLE(L: dict, r: tuple, N: int, p: int)->int:
    answer = 0
    for w in L:
        answer = (answer + L[w] * chi(w, r, N, p)) % p
    return answer


      


"""
DP_eval_MLE
INTPUTS: L (dict), r (tuple or list), N (int), p (int)
        Here, L is the dictionary that has keys in (0,1)^N
        r will be the vector in F_p^n that we are evaluating
        our multi-linear extension on
        N is the dimension of the boolean hypercube
        p is the prime number, with respect to which we work
OUTPUTS: answer
        which is \tilde{L}(r), i.e., the value of the (unique)
        MLE on input r.
        
NOTE: this algorithm may be found in Section 3.5 of Thaler's book. It differs
from the above in that it uses dynamic programming. It saves a log factor in
time but uses linear space.
"""    
def DP_eval_MLE(L: dict, r: tuple, N: int, p: int)->int:
    answer = 0
    chi_values = [1]
    for i in range(N):
        temp = []
        for j in range(2**i):
            temp.append((1-r[i])*chi_values[j] % p)
            temp.append((r[i])*chi_values[j] % p)
        chi_values = temp

    for key in L:
        dec =0
        for i in range(N):
            dec = dec + 2**(N-i-1)*key[i]
        answer = (answer+L[key]*chi_values[dec]) % p
    return answer
