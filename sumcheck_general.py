#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 21:35:48 2022

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


###MULTILINEAR SUMCHECK -- using Thaler's DP algorithm for optimizing prover time
###as opposed to the usual sumcheck protocol, we want to avoid using sage, polynomial
###rings, etc. To allow this to be possible, we will only code it for *multilinear polynomials*
###of course, it is an interesting question: how does one encode/input a multilinear polynomial
###part of the trick is, for us, we will input a dictionary (or other DS) of values on the 
###boolean hypercube. 

#the following, count_tri_fn, is the simple function that will be used to count triangles.
#(NOTE: here we are note explicitly computing the coefficients of the polynomial; 
#rather, we give a procedure for *evaluating* tri_fn.)

#HERE, L is the output of "build function from matrix".
#params is empty tuple. this is so that the *same* sumcheck protocol
#can be used for both count_tri_fn *and* for mat_mul_fn

#all_L is a list of all relevant dictionaries passed to the function
def count_tri_fn(all_L,N, full_input, params, p):
    if len(all_L)!=1:
        print("You passed",len(all_L),"dictionaries, which is not correct.")
        return False
    L = all_L[0]
    m = len(full_input)
    if m%3 !=0:
        print("the length of the input is not divisible by 3!!")
        print(m)
        return False 
    n = m// 3
    if N != 2* n:
        print("the length of the input is not correct, it should be",3*N/2)
        return False
#    print("made it so far")
    x = tuple(full_input[:n])
#    print(x)
    y = tuple(full_input[n:2*n])
#    print(y)
    z = tuple(full_input[2*n:])
#    print(z)
    #note, we use DP_eval_MLE with 2*n because we have twice as many variables!! 
#    print(DP_eval_MLE(L,x+y,2*n,p))
    #as a test, changed DP_eval_MLE to eval_MLE
    return (util.eval_MLE(L,x+y,2*n,p)%p)*(util.eval_MLE(L,y+z,2*n,p)%p)*util.eval_MLE(L,x+z,2*n,p) % p



#mat_mul code. here, all_L should be a list of two dictionaries. params will be (i,j),
#written in int_to_bin format
def mat_mul_fn(all_L,N,full_input,params,p):
    if len(all_L)!=2:
        print("Number of dictionaries passed is ",len(all_L),"which is not correct.")
        return False
    L0 = all_L[0]
    L1 = all_L[1]
    m = len(full_input)
    if 2*m != N:
        print("twice the length of the input, {}, is not {}!!".format(m,N))
        return False
    l = len(params)
    if l != N:
        print("the length of the parameters, {}, is not {}!!".format(l,N))
    i = params[:N//2]
    j = params[N//2:]
    return (util.eval_MLE(L0,i+tuple(full_input),N, p)*util.eval_MLE(L1,tuple(full_input)+j,N,p))


#params will be (i,j), written in int_to_bin format.
#full_input will just be the input k.
def square_mat_fn(all_L,N,full_input,params, p):
    if len(all_L)!=1:
        print("You passed",len(all_L),"dictionaries, which is not correct.")
        return False
    L = all_L[0]
    m = len(full_input)
    if 2*m != N:
        print("twice the length of the input, {}, is not {}!!".format(m,N))
        return False
    l = len(params)
    if l != N:
        print("the length of the parameters, {}, is not {}!!".format(l,N))
    i = params[:N//2]
    j = params[N//2:]


    return (util.eval_MLE(L,i+tuple(full_input),N, p)*util.eval_MLE(L,tuple(full_input)+j,N,p))


##Given a polynomial poly (which we can evaluate/query) in m variables (whose maximal deg
#in any given variable is d), and
##an initial ``r_vec'', we wish to return the data of a univariate polynomial
##of degree d, which represents:
##\sum poly(r_vec,X_{rlen},\vec{b}), where the sum is over bit-strings b of length
##k =N - len(r_Vec)-1. Here, the "free variable" of the polynomial is therefore
##X_{len(r_vec)}.
    
##to return this data, we choose to return the values of this polynomial at:
##0,1,\dots,d, d+1 points, which of course uniquely determines a degree d polynomial
##more concretely, we return a list of length d+1 whose ith term is:
##\sum poly(r_vec,i,\vec{b}).
    

###NOTE: as of yet, we are still dealing with d=2#####


def partial_boolean_hypercube_sum_fast(poly, L, N, m, d, r_vec, params, p):
    #here, L will be function on boolean hypercube, n will be dimension of boolean
    #hypercube (in particular, we imagine that we are padding our matrix.)
    
    k = m - len(r_vec)-1
    #for debugging, I also sometimes had k printed, so I could see how the calculation was proceeding.
#    print("k is :",k)
    rlen = len(r_vec)
    rtup = tuple(r_vec)
    univariate_values = []
    start = timer()
#    start_time = time.time()

    for i in range(d+1):
        running_sum = 0
        #loop to sum the values p(rvec,x_{rlen},bitstring)
        for b in range(2**k):
            
#            print(int_to_bin(b,k))
#            print(b)
#            print(rtup+(i,)+int_to_bin(b,k))
            
            running_sum = (running_sum + poly(L, N, rtup+(i,)+util.int_to_bin(b,k), params, p))%p
        univariate_values.append(running_sum % p)
    end = timer()
#   here is a useful timer, wwhich I've commented out.
#    print("time elapsed for all 2^{} queries is:".format(k),end-start,"seconds")
#        print(i)
#    end_time = time.time()
#    end = timer()
#    print("total time elapsed for partial_boolean_hypercube_sum_fast",\
#          end - start)
    return util.quadratic_interpolate(univariate_values,p)



def full_boolean_hypercube_sum_fast(poly, L, N, m, d, params, p):
    running_sum = 0
    for b in range(2**m):
        running_sum = (running_sum + poly(L,N,util.int_to_bin(b,m),params, p)) %p
    return running_sum

        
###prover code, for general sumcheck protocol. Here is the meaning of the variables.
###(note: Prover is not the command here)
###poly is the polynomial (in our applications, either count_tri_fn or mat_mul)
###we assume that our polynomial may be easily constructed from multilinear functions.
###L is a list of dictionaries, with keys {0,1}^N, and values the function values.
###N is the dimension of the boolean hypercube. (We assume each dictionary has the
###same input size)
###m is the number of variables in poly
###d is the max degree of any term of poly. in our applications, we have only
###implemented d = 1,2
###p is the prime number
###step is the step # (recall that Verifier is the command for this protocol)
###r_vec is the random vector that the verifier keeps supplying
###params are any additional parameters to our original (multilinear) functions


def Prover(poly, L, N, m, d, p, step, r_vec, params):
    if step==0:
        #if we are at the 0th step, send the proposed
        #sum of g over the full boolean hypercube
        return full_boolean_hypercube_sum_fast(poly, L, N, m, d, params, p)
        
    else:
        #steps are from 0 to m
        if step >m or step <0:
            return False
        #length of r_vec should be step - 1
        if len(r_vec) != step -1:
            return False
        
        
        #return the univariate polynomial which is the sum of 
        #g(r_vec,x_{rlen},bit_string), where bit_string ranges over
        #the partial boolean hypercube. The output is a univariate polynomial
        #in x_{rlen}, communicated by its list of coefficients
        return partial_boolean_hypercube_sum_fast(poly, L, N, m, d, r_vec, params, p)

#verifier is the command.
#m is number of variables of poly
#d is the maximal degree of any given variables. We assume that this is <=2
#(this is reflected in various other parts of our code)
#TODO: improve this!!!
def Verifier(poly, L, N, m, d, params, p):
    r_vec=[]
    g_prev=[]
    for step in range(m+1):
#        print("step is", step)
        if step ==0:
            g_prev.append([Prover(poly,L,N,m,d,p,step,r_vec,params)])
            print("The sum over the boolean hypercube is: ",g_prev[0][0])
            #the Prover outputs what it claims is the sum of g over the
            #full boolean hypercube: {0,1}^n inside of F_q^n.
        elif step==1:

            new_g = Prover(poly,L,N,m,d,p,step,r_vec, params)
#            print(new_g)
            #new_g is what the prover claims is the *sum* of g(x_0,bit_string), where
            #bit_string ranges over {0,1}^{n-1}. This will be a univariate polynomial
            #in x_0
            
            #compute the sum of new_g on the relevant bit strings, compare to g_prev[0]
            #i.e., new_g(0,0,...,0)+new_g(1,0,0,...,0) should be the first claimed sum,
            #which is g_prev[0].
            if g_prev[0][0] != util.quadratic_evaluation(new_g,0,p)+util.quadratic_evaluation(new_g,1,p):
                return False
            print("step",step,"succeeded")
            g_prev.append(new_g)
            r = random.randint(0,p)
            r_vec.append(r)
#            print("random value is",r)
        else:
            new_g = Prover(poly,L,N,m,d,p,step,r_vec, params)
#            print(new_g)
            #compute the sum of new_g on relevant bit strings, compare to g_{s-1}
            #evaluated at x_{s-2} = r_vec[-1]. (recall: g_{s-1} is a univariate polynomial
            #in x_{s-2})
            
            #MORE PRECISELY:
            #g_prev[-1] is the univariate polynomial that the Prover sent
            #i.e., the "proof" at step s-1. (NOTE: we are at step s now.)
            #hence, previous_predicted_value is g_prev[-1](r_{last})

            #Here, r_last is r_vec[-1]; note also that s-1 = len(r_vec).
            
            #NOTE: we have not yet appended the new prover value: we want to check
            #its validity first
            previous_predicted_value = util.quadratic_evaluation(g_prev[-1],r_vec[-1],p)
#            print(previous_predicted_value)
            #new_g is meant to be: sum of g(r_0,...,r_{s-2},x_{s-1},bitstring)
            #as bitstring ranges. 
            #new_predicted value: plug in x_{s-1}=0, x_{s}=1, add. I get a field element. Compare
            #with previous_predicted_value.
            new_predicted_value = (util.quadratic_evaluation(new_g,0,p)+util.quadratic_evaluation(new_g,1,p)) % p
#            print(new_predicted_value)

            if previous_predicted_value != new_predicted_value:
                print("incompatibility between old printed value and new value")
                return False
            g_prev.append(new_g)
            print("step",step,"succeeded")
            r = random.randint(0,p)
            r_vec.append(r)
#            print("random value is",r)
        if step==m:
            #extra check for s==n.
            #look at the last thing prover sent, plug in r_n
            #compare to g(r_vec)
            #these two things should be equal
#            print(len(r_vec))

            poly_at_r_vec = poly(L,N, r_vec, params, p) %p
            g_n_at_r_n = util.quadratic_evaluation(g_prev[-1],r_vec[-1],p) %p
            if poly_at_r_vec!= g_n_at_r_n:
                print("last step failed")
                return False
    print("You win!!")
    #print the sequence of polynomials
    #print the random elements we chose.
    print(g_prev)
    print(r_vec)
    return True
      
