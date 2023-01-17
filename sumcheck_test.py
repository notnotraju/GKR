#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 21:46:52 2022

@author: raju
"""



import numpy as np
import math
import itertools
import random
import time
import matplotlib.pyplot as plt
import sumcheck_util as util



from timeit import default_timer as timer

import sumcheck_general as SG
import sumcheck_efficient as SE



###TEST SUITE#####

##generate a random adjacency matrix for simple undirected graph
##on n nodes. output is a numpy matrix with coefficients in {0,1}.
def gen_random_adj(n):
    M=np.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(i):
            r = random.randint(0,1)
            M[i][j]=r
            M[j][i]=r
    return M

#built a random integer matrix with elements in [0,p)
def rand_int_mat(n,p):
    M = np.zeros((n,n),dtype = int)
    for i in range(n):
        for j in range(n):
            M[i][j] = np.random.randint(0,p)
    return M



#here, l is the length of r_vec, which will be bounded by 3* N//2 in the case of
#count_tri_fn




#final test for "counting triangles" IP.
def test_count_triangles(n,p):
    M = gen_random_adj(n)
    L,N = util.build_function_from_matrix(M,n)
    SG.Verifier(count_tri_fn,[L],N,3 * N//2,2,(),p)
    print(M)

#tests eval_MLE. we especially used the timer to figure out how long it took.
def test_eval_MLE(n,p):
    M = gen_random_adj(n)
    L,N = util.build_function_from_matrix(M,n)
    r_vec =[]
    for i in range(N):
        r = random.randint(0,p)
        r_vec.append(r)
    start = timer()
    #tests seem to indicate that the following could be either eval_MLE
    #or DP_eval_MLE
    answer = util.eval_MLE(L,r_vec,N,p)
    end = timer()
    print("time elapsed is: ",end-start)
    print("dimension of boolean hypercube is: ",N)
    print("r_vec is: ",r_vec)
    print("value is: ",answer)
    return end-start


def growth_MLE(k,p):
    times=[]
    for i in range(1,k):
        times.append(test_eval_MLE(i,p))
    squares = [(i)**2 for i in range(1,k)]
    plt.plot(squares,times)
#    plt.plot([i for i in range(1,k)],times)



#tried the triangle counting function on the complete graph.
def test_complete_graph_triangle(n,p):
    M=np.ones((n,n),dtype=int)-np.identity(n,dtype=int)
    L,N = util.build_function_from_matrix(M,n)
    print(M)
    start = timer()
    SG.Verifier(count_tri_fn,[L],N,3 * N//2,2,(),p)
    end = timer()
    print("total time elapsed is: ", end-start,"seconds")



def test_square_mat(n,p):
    M = rand_int_mat(n,p)
    L,N = util.build_function_from_matrix(M,n)
    i = util.int_to_bin(np.random.randint(0,2**(N//2)),N//2)
    j = util.int_to_bin(np.random.randint(0,2**(N//2)),N//2)
    print("The matrix is: ", M)
    SG.Verifier(square_mat_fn, [L],N, N//2, 2,i+j,p)
    
    i = util.int_to_bin(np.random.randint(0,n),N//2)
    j = util.int_to_bin(np.random.randint(0,n),N//2)
    print("now, the index is: ", i+j)
    Verifier(square_mat_fn, [L],N, N//2, 2,i+j,p)


#length of params will be l. We will generate a random matrix, and random
#parameters. flag tells us if we are replacing the first or the last entries.
#this is in the SE module
def test_initialize_dict_with_params(n, l, flag, p):
    #initialize_dict_with_params(L, N, params, p)
    M = rand_int_mat(n,p)
    params = []
    for i in range(l):
        params.append(np.random.randint(0,p))
    L, N = util.build_function_from_matrix(M,n)
    print("Original matrix is: ",M)
    print("dimension of dictionary is: ",N)
    print("parameters are ",params)
    
    new_dict = SE.initialize_dict_with_params(L,N,params,flag, p)
    r = params.copy()
    if flag ==0:
        for i in range(N-l):
            r.append(np.random.randint(0,2))
    elif flag == 1:
        r.reverse()
        for i in range(N-l):
            r.append(np.random.randint(0,2))
        r.reverse()
    print("random vector is: ",r)
    print("evaluation at r is", util.eval_MLE(L,r,N,p))
    print("my dictionary tells me: ", new_dict[tuple(r)])
    return new_dict



def test_mat_mul(n,p):
    M0 = rand_int_mat(n,p)
    L0,N = util.build_function_from_matrix(M0,n)
    M1 = rand_int_mat(n,p)
    L1,N = util.build_function_from_matrix(M1,n)
    
    i = util.int_to_bin(np.random.randint(0,2**(N//2)),N//2)
    j = util.int_to_bin(np.random.randint(0,2**(N//2)),N//2)
    print("The matrices are: ")
    print(M0)
    print(M1)
    print("The indices are: ",i,j)
    SG.Verifier(mat_mul_fn, [L0,L1],N, N//2, 2,i+j,p)
    
    i = util.int_to_bin(np.random.randint(0,n),N//2)
    j = util.int_to_bin(np.random.randint(0,n),N//2)
    print("now, the index is: ", i+j)
    SG.Verifier(mat_mul_fn, [L0,L1],N, N//2, 2,i+j,p)



#test Prover_Mat_Mul(L, N, params, p) (before verifier!!!)
def test_fast_mat_mul(n,p):
    M0 = rand_int_mat(n,p)
    M1 = rand_int_mat(n,p)
    L0, N = util.build_function_from_matrix(M0,n)
    L1, N = util.build_function_from_matrix(M1,n)
    params = []
    for i in range(N):
        params.append(np.random.randint(0,p))
    #params are the random vectors that I choose to evaluate my \tlde{f}_{AB} at.
    
    SE.Prover_Mat_Mul([L0,L1],N,params,p)
    
def test_fast_complete_graph_triangle(n,p):
    M=np.ones((n,n),dtype=int)-np.identity(n,dtype=int)
    print(M)
    L,N = util.build_function_from_matrix(M,n)
    SE.Prover_Count_Tri(M,n,p)

