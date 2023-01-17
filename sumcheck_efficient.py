#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 21:44:04 2022

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

###For the purpose of today, we are concerned with efficient ``matrix multiplication''.
###as opposed to our previous implementations, we will not rely so heavily on partial/full boolean sum,
###or more generally, ``independently computing the MLE''.
###instead, the logic must be slightly changed. The reason is that the prover will use *prior steps*
###to deduce each stage. Fortunately, we only need *one prior stage*, which means that the verifier can pass
###the dictionary of the step before!! Anyways, here the prover is the command.


###NOW: we BUILD our SUMCHECK IP protocol for MatMul, for which we assume #############




###today, I am changing the Prover code from before. In particular, L will
###*not* be the full dictionary. It will rather be the special dictionary
###Thaler constructs, with things like:
### f(r_1,0,0), f(r_1,0,1),...
    
    
###to implement this, we first build a general purpose function: DP_new_dict_from_old
###input is a dictionary
###of the form: f(r_1,..,r_{k-1},{0,1,2},b_{k+1},...), and r_k
###and output is a dictionary of the values:
###f(r_1,...,r_k,{0,1,2},b_{k+2},...)
###for this, we will be assuming that f is a MLE.
    
###d is still 2 in these applications
    
###similarly, we build an iterative initialize_dict_with_params.
###(the functionality is identical, but we are passed a vector params, and
###we )
    
##this is only done for f_A. For f_B, something will have to be reversed.
##will eventually add a flag to select for what to do! however, this will require
##updating the test suite code, which I don't want to do right now. 
##now: flag 
def initialize_dict_with_params(L, N, params, flag, p):
    old_L = L
    new_L = dict()
    
    for i in range(len(params)):
        
        digits_in_b = N-i-1

#        print("the number of digits_in_b is", digits_in_b)
        for b in range(2**(digits_in_b)):
            tuple_b = util.int_to_bin(b,digits_in_b)
            if flag == 0: #flag = 0 means in f_A
                elt = params[i]
                old_params = tuple(params[:i])
                new_L[old_params + (elt,) + tuple_b]= (\
                      (1-elt) * old_L[old_params + (0,) + tuple_b] + \
                      elt * old_L[old_params + (1,) + tuple_b] ) %p
                      
            elif flag ==1:
                l = len(params)
                elt = params[l - 1 - i]
                if i == 0:
                    old_params = tuple()
                else:
                    old_params = tuple(params[l-i : ])
                new_L[tuple_b + (elt,) + old_params] = (\
                      (1-elt) * old_L[tuple_b + (0,) + old_params] + \
                      elt * old_L[tuple_b + (1,) + old_params]) % p
                
        old_L = new_L
        new_L = dict()
    return old_L




#the following is the key code in the fast matrix multiplication protocol (due to
#Thaler): it is a dynamic programming algorithm for updating the dictionary. We are given
#an old dictionary, which corresponds to the values of the function at params, r_vec[:-1], and some bit strings
#using DP, we build a new dictionary, which yields the values of the function at params, r_vec, and some bit_strings.
#flag tells us the order of params_r_vec
#if flag == 0 , then it is params, r_vec (i.e., (x,r_1))
#if flag == 1, then it is r_vec,params (i.e., (r_3, y))
#this is important to distinguish between the f_A and f_B cases!



def new_dict_from_old(old_L, N, params, r_vec, flag, p):
    new_L = dict()
    param_len = len(params)
    #params is a tple!!
    r_len = len(r_vec)
    digits_in_b = N - param_len - r_len
    
    for b in range(2**(digits_in_b)):
        tuple_b = util.int_to_bin(b, digits_in_b)
        elt = r_vec[-1]
        old_r_vec = tuple(r_vec[:-1])
        if flag ==0: #flag = 0 means working with f_A

            new_L[params + old_r_vec + (elt,) + tuple_b] = (\
                  (1-elt)* old_L[params + old_r_vec + (0,) + tuple_b]+\
                  elt * old_L[params + old_r_vec + (1,) + tuple_b]) % p
        elif flag == 1: #flag = 1 means working with f_B
            new_L[old_r_vec + (elt,) + tuple_b + params] = (\
                  (1-elt)*old_L[old_r_vec + (0,) + tuple_b + params] +\
                  elt * old_L[old_r_vec + (1,) + tuple_b + params])%p
    return new_L
                  


#L is a list of two dictionaries, corresponding to A and B respectively
#N is of course 2*digits, i.e., the dimension of the vector space on which
#\tilde{f}_A and \tilde{f}_B are (multilinear) functions
#params is necessary! remember, we are somehow doing a sumcheck for
#\tilde{f}_C(params). if params is (int_to_bin(i),int_to_bin(j))
#this is simply the (i,j) entry of C = A*B. 
        
    
###running into problem: Prover_Mat_Mul must remember state!!!
###in particular, we have now chosen the Prover (so, Prover_Mat_Mul)
###is the command. 
def Prover_Mat_Mul(L, N, params, p):
    step = 0
    #g is a list of the things I will send the verifier
    g = []
    #r_vec is a list of the randomness, that the verifier provides.
    #note: for the purpose of testing this on June 18, 2022, we're going to have
    #the prover generate the randoness.
    r_vec = []
    
    #Mat_Mul corresponds to a polynomial of degree N//2.
    
    #namely, g(z): = f_A(params[:N//2],z)*f_B(z,params[N//2:])
    #if params[0:N//2] and params[N//2:] are boolean, then this corresponds
    #to a matrix entry in the product matrix. 
    
    m = N//2

    param0 = tuple(params[:m])
    param1 = tuple(params[m:])
    
    #we initialize the two dictionaries we are using. 
    #at the start, current_L_A contains the values of \tilde{f}_A(param0 + b)
    #where b ranges over all m-bit boolean strings. 
    current_L_A = initialize_dict_with_params(L[0], N, param0, 0, p)
    #at the start, current_L_B contains the values of \tilde{f}_B(b + param1)
    #where b ranges over all m-bit boolean strings. 
    current_L_B = initialize_dict_with_params(L[1], N, param1, 1, p)
    
    for step in range(m+1): 
        print("We are currently on step: ",step)
        digits_in_b = m - step
        #GOAL: at the end of each step, generate some ``proof'' for
        #the verifier to check. Then we will run the verifier, who will check
        #it, and continue.
        
        #what I need to send depends on the step.
        #if step = 0, I just claim some number, which is the sum over the boolean
        #hypercube. print this value.
        
        if step == 0:
            running_sum = 0
            for b in range(2**m):
                running_sum = running_sum + current_L_A[param0+util.int_to_bin(b,m)]*\
                current_L_B[util.int_to_bin(b,m)+param1] #updating with g(b), where
                running_sum = running_sum % p       #b ranges over {0,1}^m
            print("The sum over the boolean hypercube is:", running_sum)
            print("Here, the parameters are:",param0,"and",param1)
            old_poly = [running_sum,0,0]
            #here, old_poly is the constant poly with value running_sum. This is
            #simply to keep syntax symmetric. We need this for check with step 1.
            g.append(old_poly)
            
        #if step = 1, I send what I claim is \sum_{b}(g(z_1,b_2,..,b_m)), which is a 
        #polynomial (quadratic) in z_1. (The sum is over b_j in {0,1})
        #this sum concretely looks like: 
        #sum over b_j in {0,1} (dim m-1) of:
        #f_A(param0, z_1,b_2,..b_m) * f_B(z_1,b_2,...,b_m,param1)
        #the above is g(z_1,b_2,..,b_m)

        
        ###REWRITE ALL STEPS 1 to m (except last part) IN UNIFORM WAY.
        else:
#            print("first dictionary is:", current_L_A)
#            print("second dictionary is: ", current_L_B)
            running_sum_0 = 0 #will represent running sum of g(r_vec, 0,b_{step},...)
            running_sum_1 = 0 #will represent running sum of g(r_vec, 1,b_{step},...)
            running_sum_2 = 0 ##will represent running sum of g(r_vec, 2,b_{step},...)
            ###NOT SURE ABOUT PRECISE INDEX in the text ``b_{step}'' above, but it won't
            ###matter, I don't need to know the exact index.
            rtup = tuple(r_vec)
            for b in range(2**(digits_in_b)):
                #print(b)
                f_A0 = current_L_A[ param0 + rtup + (0,) + util.int_to_bin(b,digits_in_b) ] #f_A(rtup, 0,b)
                f_B0 = current_L_B[ rtup  + (0,) + util.int_to_bin(b,digits_in_b) + param1 ] #f_B(rtup, 0,b)
                f_A1 = current_L_A[ param0 + rtup + (1,) + util.int_to_bin(b,digits_in_b) ] #f_A(rtup, 1,b)
                f_B1 = current_L_B[ rtup + (1,) + util.int_to_bin(b,digits_in_b) + param1 ] #f_B(rtup, 1,b)
                f_A2 = 2 * f_A1 - f_A0 #f_A(param0, 2,rtup, b)
                f_B2 = 2 * f_B1 - f_B0 #f_B(2, rtup, b, param1)
                                        # above is an ERROR, should be f_B(rtup, 2, b, param1)
                                        #as of 22:04, June 18th, I've "fixed this error",

                #formulas for f_*(2,b) only hold because both functions are multilinear.
                # do the update phase
            
                running_sum_0 = (running_sum_0+ f_A0 * f_B0)%p
                #computes sum of g(0,\vec{b}) over all b in {0,1}^{m - 1}
                running_sum_1 = (running_sum_1 + f_A1 * f_B1)%p
                #computes sum of g(1,\vec{b}) over all b in {0,1}^{m - 1}
                running_sum_2 = (running_sum_2 + f_A2 * f_B2)%p
            new_poly = util.quadratic_interpolate([running_sum_0, running_sum_1, running_sum_2],p)
            g.append(new_poly)
            #JUN 18 2022: commenting the line below out, so I can test the prover code.
            #r = Verifier_Mat_Mul(old_poly, new_poly, L, N, step, params, p)
            # If all goes to plan and step is in [1,m-1], the verifier will return 
            # r, a random value.

            r,did_it_succeed = Verifier_Mat_Mul(L, N, step, params, r_vec, g, p)
            if did_it_succeed == False:
                return False
            r_vec.append(r)
            
            current_L_A = new_dict_from_old(current_L_A, N, tuple(param0), r_vec, 0, p)
            current_L_B = new_dict_from_old(current_L_B, N, tuple(param1), r_vec, 1, p)
#            print("the value of randoness we choose is: ", r)
#            print("the polynomial that we output is", new_poly)
            
        
        #if 1<step (verifier will act differently at the mth step)
        #the prover will pick a random number r, and add it to r_vec.
        ###MSK NOTE: we can have the verifier do this by just adding a flag. 
        #the prover then builds (via DP) the new dictionary, and computes
        #the three sums that we need to via the new dictionary.
        #the prover sends the last polynomial and the new polynomial to the verifier.
        #(prover will remember all of the polynomials.)
        #if verifier checks out, next step.
        
        #if we are at step == m, then we do the following *extra*
        #send r_vec to verifier. Verifier will query g at r_vec, and also has a second
        #predicted value from g_prev[-1], which was just sent.it compares these two,
        #if they are the same, the verifier returns true, and we also return true.
        #maybe then prover prints out the proof in a nice format.
    #we print the proof
    print("The polynomials are:",g)
    print("The randomness is:", r_vec)
    #finally, we return the actual value we were claiming, just so we can feed
    #into Prover_Count_Tri protocol
    return g[0][0]
    
    
def Verifier_Mat_Mul(L, N, step, params, r_vec, g, p):
    if step <= 0:
        return True
    m = N// 2
    if step > m:
        print("too many steps!!")
        return False
    g_current = g[-1]
    g_last = g[-2]
    
    current_guess = (util.quadratic_evaluation(g_current, 0, p) +\
                     util.quadratic_evaluation(g_current,1,p)) %p
    if step ==1:
        old_guess = g_last[0]% p #claimed value of the sum over the boolean hypercube
                                #because g_prev[0] is [constant, 0, 0]
    else:
        old_guess = util.quadratic_evaluation(g_last, r_vec[-1], p) % p
    
    
    if current_guess == old_guess:
        r = np.random.randint(0,p) #r is the randomness that the verifier chooses
        if step < m:
            return r, True
        else:
            current_poly_eval = util.quadratic_evaluation(g_current,r,p)%p
            rtup = tuple(r_vec)+(r,)
#            print("dictionary is", L[0])
#            print("params is", params[0:m])
#            print("rtup is", rtup)
            function_query = (util.eval_MLE(L[0], tuple(params[0:m]) + rtup, N, p) *\
            util.eval_MLE(L[1],rtup+tuple(params[m:]),N,p)) % p
            if function_query == current_poly_eval:
                print("The proof is correct!")
                return r, True
            else:
                print("The last step of the proof failed")
                return r, False
    else:
        print("Step", step, "of the proof failed")
        return 0,False

    
    
#due to annoying technical issues, we will write our "prover_count_tri"
#separately! Here, A will be the adjacency matrix.
#NOT DONE YET!!!
#here, my function will be h: F_p^N-->F_p, given by \tilde{f}_{A^2} * \tilde{f}_A
#as each is multi-linear, this function is multi-quadratic. (This means total degree
#in each variable is 2.)
def Prover_Count_Tri(A,n, p):
    step = 0
    L0, N = util.build_function_from_matrix(np.matmul(A,A),n)
    L1, N = util.build_function_from_matrix(A,n)
#    print(np.matmul(A,A))
    #g is a list of the things I will send the verifier
    g = []
    #r_vec is a list of the randomness, that the verifier provides.

    r_vec = []    
    m = N//2

    current_L_A2 = L0
    current_L_A = L1
    
    for step in range(N+1): 
        print("We are currently on step: ",step, "in the counting_triangles protocol")
        digits_in_b = N - step
        #GOAL: at the end of each step, generate some ``proof'' for
        #the verifier to check. Then we will run the verifier, who will check
        #it, and continue.
        
        #what I need to send depends on the step.
        #if step = 0, I just claim some number, which is the sum over the boolean
        #hypercube. print this value.
        
        if step == 0:
            running_sum = 0

            for b in range(2**N):
                running_sum = (running_sum + current_L_A2[util.int_to_bin(b,N)]*\
                current_L_A[util.int_to_bin(b,N)])%p 
                #compute the total sum over the boolean hypercube of the function
                #h
            print("We claim that the number of triangles * 6 is:", running_sum)
#            print("Here, the parameters are:",param0,"and",param1)
            old_poly = [running_sum,0,0]
            #here, old_poly is the constant poly with value running_sum. This is
            #simply to keep syntax symmetric. We need this for check with step 1.
            g.append(old_poly)
            
        #if step = 1, I send what I claim is \sum_{b}(h(z_1,b_2,..,b_m)), which is a 
        #polynomial (quadratic) in z_1. (The sum is over b_j in {0,1})
        #this sum concretely looks like: 
        #sum over b_j in {0,1} (dim m-1) of:
        #f_A(param0, z_1,b_2,..b_m) * f_B(z_1,b_2,...,b_m,param1)
        #the above is h(z_1,b_2,..,b_m)

        else:
            running_sum_0 = 0 #will represent running sum of h(r_vec, 0,b_{step},...)
            running_sum_1 = 0 #will represent running sum of h(r_vec, 1,b_{step},...)
            running_sum_2 = 0 ##will represent running sum of h(r_vec, 2,b_{step},...)
            ###NOT SURE ABOUT PRECISE INDEX in the text ``b_{step}'' above, but it won't
            ###matter, I don't need to know the exact index.
            rtup = tuple(r_vec)
            for b in range(2**(digits_in_b)):
                #print(b)
                f_A2_0 = current_L_A2[ rtup + (0,) + util.int_to_bin(b,digits_in_b) ] #f_A2(rtup, 0,b)
                f_A_0 = current_L_A[ rtup  + (0,) + util.int_to_bin(b,digits_in_b)  ] #f_A(rtup, 0,b)
                f_A2_1 = current_L_A2[ rtup + (1,) + util.int_to_bin(b,digits_in_b) ] #f_A2(rtup, 1,b)
                f_A_1 = current_L_A[ rtup + (1,) + util.int_to_bin(b,digits_in_b)  ] #f_A(rtup, 1,b)
                f_A2_2 = 2 * f_A2_1 - f_A2_0 
                f_A_2 = 2 * f_A_1 - f_A_0 

                #formulas for f_*(2,b) only hold because both functions are multilinear.
                # do the update phase
            
                running_sum_0 = (running_sum_0+ f_A2_0 * f_A_0)%p
                #computes sum of g(0,\vec{b}) over all b in {0,1}^{m - 1}
                running_sum_1 = (running_sum_1 + f_A2_1 * f_A_1)%p
                #computes sum of g(1,\vec{b}) over all b in {0,1}^{m - 1}
                running_sum_2 = (running_sum_2 + f_A2_2 * f_A_2)%p
            new_poly = util.quadratic_interpolate([running_sum_0, running_sum_1, running_sum_2],p)
            g.append(new_poly)
            #JUN 18 2022: commenting the line below out, so I can test the prover code.
            #r = Verifier_Mat_Mul(old_poly, new_poly, L, N, step, params, p)
            # If all goes to plan and step is in [1,m-1], the verifier will return 
            # r, a random value.
            #MODIFY VERIFIER CODE!!! ONLY PASSING L1
            r,did_it_succeed = Verifier_Count_Tri(L1, N, step, r_vec, g, p)
            if did_it_succeed == False:
                return False
            r_vec.append(r)
            
            current_L_A2 = new_dict_from_old(current_L_A2, N, tuple(), r_vec, 0, p)
            current_L_A = new_dict_from_old(current_L_A, N, tuple(), r_vec, 0, p)
            print("the value of randoness we choose is: ", r)
            print("the polynomial that we output is", new_poly)
            
        
        #if 1<step (verifier will act differently at the mth step)
        #the prover will pick a random number r, and add it to r_vec.
        ###MSK NOTE: we can have the verifier do this by just adding a flag. 
        #the prover then builds (via DP) the new dictionary, and computes
        #the three sums that we need to via the new dictionary.
        #the prover sends the last polynomial and the new polynomial to the verifier.
        #(prover will remember all of the polynomials.)
        #if verifier checks out, next step.
        
        #if we are at step == m, then we do the following *extra*
        #send r_vec to verifier. Verifier will query g at r_vec, and also has a second
        #predicted value from g_prev[-1], which was just sent.it compares these two,
        #if they are the same, the verifier returns true, and we also return true.
        #maybe then prover prints out the proof in a nice format.
    
#Verifier for count triangles
def Verifier_Count_Tri(L1, N, step, r_vec, g, p):
    if step <= 0:
        return True
    m = N// 2
    if step > N:
        print("too many steps!!")
        return False
    g_current = g[-1]
    g_last = g[-2]
    
    current_guess = (util.quadratic_evaluation(g_current, 0, p) +\
                     util.quadratic_evaluation(g_current,1,p)) %p
    if step ==1:
        old_guess = g_last[0]% p #claimed value of the sum over the boolean hypercube
                                #because g_prev[0] is [constant, 0, 0]
    else:
        old_guess = util.quadratic_evaluation(g_last, r_vec[-1], p) % p
    
    
    if current_guess == old_guess:
        r = np.random.randint(0,p) #r is the randomness that the verifier chooses
        if step < N:
            return r, True
        else:
            current_poly_eval = util.quadratic_evaluation(g_current,r,p)%p
            rtup = tuple(r_vec)+(r,)
#            print("dictionary is", L[0])
#            print("params is", params[0:m])
#            print("rtup is", rtup)
            #NOW, need to do 2 queries. The first is just the normal check on
            #\tilde{f}_A(r_vec)
            print("rtup is", rtup)
            A_query = util.eval_MLE(L1, rtup, N, p) %p
            A2_query = Prover_Mat_Mul([L1,L1],N,rtup,p)%p
#            print("tilde{f}_A(rtup) is:", function_query)
#            print("g_current at", r,"is", current_poly_eval)
#            print("g_current is", g_current)
            if A_query * A2_query %p == current_poly_eval % p:
                print("The protocol works!!!")
                return r,True
            else:
                print("The last step of the count_triangles protocol failed")
                return r, False
    else:
        print("Step", step, "of the counting_triangles protocol failed")
        return 0,False
