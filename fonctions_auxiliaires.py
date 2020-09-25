# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:45:38 2020

@author: jwehounou
"""
import time
import numpy as np


STATE1_STRATS = ("CONS+", "CONS-")                                             # strategies possibles pour l'etat 1 de a_i
STATE2_STRATS = ("DIS", "CONS-")                                               # strategies possibles pour l'etat 2 de a_i
STATE3_STRATS = ("DIS", "PROD")

#------------------------------------------------------------------------------
#           definitions of functions
#------------------------------------------------------------------------------

def fct_positive(sum_list1, sum_list2):
    """
    sum_list1 : sum of items in the list1
    sum_list2 : sum of items in the list2
    
    difference between sum of list1 et sum of list2 such as :
         diff = 0 if sum_list1 - sum_list2 <= 0
         diff = sum_list1 - sum_list2 if sum_list1 - sum_list2 > 0

        diff = 0 if sum_list1 - sum_list2 <= 0 else sum_list1 - sum_list2
    Returns
    -------
    return 0 or sum_list1 - sum_list2
    
    """
    
    # boolean = sum_list1 - sum_list2 > 0
    # diff = boolean * (sum_list1 - sum_list2)
    diff = 0 if sum_list1 - sum_list2 <= 0 else sum_list1 - sum_list2
    return diff

def generate_energy_unit_price_SG(pi_hp_plus, pi_hp_minus):
    """
    generate intern cost and intern price of one unit of energy inside SG

    Returns
    -------
    pi_0_plus, pi_0_minus

    """
    rd_num = np.random.random()
    pi_0_plus = pi_hp_plus * rd_num
    pi_0_minus = pi_hp_minus * rd_num
    return pi_0_plus, pi_0_minus
     

#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------    
def test_fct_positive():
    N = 100
    OK, NOK = 0, 0
    for n in range(N):
        list1 = np.random.randint(1, 10, 10)
        list2 =  np.random.randint(1, 10, 10)
        diff = fct_positive(sum_list1=sum(list1), sum_list2=sum(list2))
        
        if sum(list1)>sum(list2) and diff != 0:
            OK += 1
            # print("OK1, n={} {}>{} => diff={}"
            #       .format(n, sum(list1), sum(list2), diff))
        elif sum(list1)<sum(list2) and diff == 0:
            OK += 1
            # print("OK2, n={} {}<{} => diff={}"\
            #       .format(n, sum(list1), sum(list2), diff))
        elif sum(list1)<sum(list2) and diff != 0:
            NOK += 1
            # print("NOK1, n={} {}<{} => diff={}"\
            #       .format(n, sum(list1), sum(list2), diff))
        elif sum(list1)>sum(list2) and diff == 0:
            NOK += 1
            # print("NOK2, n={} {}>{} => diff={}"\
            #       .format(n, sum(list1), sum(list2), diff))
                
    print("fct_positive: %OK={}, %NOK={}".format(OK/(OK+NOK), NOK/(OK+NOK)))
      
    
def test_generate_energy_unit_price_SG():
    N = 10
    pi_hp_plus = np.random.random_sample(N) * 20
    pi_hp_minus = np.random.random_sample(N) * 20
    res = np.array(list(map(generate_energy_unit_price_SG, 
                                     *(pi_hp_plus, pi_hp_minus))))
    
    pi_0_plus, pi_0_minus = res[:,0], res[:,1]
    if (pi_0_plus<pi_hp_plus).all() and (pi_0_minus<pi_hp_minus).all():
        print("generate_energy_unit_price_SG: OK")
    else:
        print("generate_energy_unit_price_SG: NOK")
    
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------    
if __name__ == "__main__":
    ti = time.time()
    test_fct_positive()
    test_generate_energy_unit_price_SG()
    print("runtime = {}".format(time.time() - ti))