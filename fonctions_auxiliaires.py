# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:45:38 2020

@author: jwehounou
"""
import os
import time
import json
import string
import random
import numpy as np


STATE1_STRATS = ("CONS+", "CONS-")                                             # strategies possibles pour l'etat 1 de a_i
STATE2_STRATS = ("DIS", "CONS-")                                               # strategies possibles pour l'etat 2 de a_i
STATE3_STRATS = ("DIS", "PROD")

#------------------------------------------------------------------------------
#           definitions of class
#------------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
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
     
def generate_Cis_Pis_Sis(n_items, low_1, high_1, low_2, high_2):
    """
    generate Cis, Pis, Sis and Si_maxs.
    each variable has 1*n_items shape
    low_1 and high_1 are the limit of Ci items generated
    low_2 and high_2 are the limit of Pi, Si, Si_max items generated
    
    return:
        Cis, Pis, Si_maxs, Sis
    """
    Cis = np.random.uniform(low=low_1, high=high_1, 
                            size=(1, n_items))
    
    low = low_2; high = high_2
    # Pi
    inters = map(lambda x: (low*x, high*x), Cis.reshape(-1))
    Pis = np.array([np.random.uniform(low=low_item, high=high_item) 
                    for (low_item,high_item) in inters]).reshape((1,-1))
    # Si
    inters = map(lambda x: (low*x, high*x), Pis.reshape(-1))
    Si_maxs = np.array([np.random.uniform(low=low_item, high=high_item) 
                    for (low_item,high_item) in inters]).reshape((1,-1))
    inters = map(lambda x: (low*x, high*x), Si_maxs.reshape(-1))
    Sis = np.array([np.random.uniform(low=low_item, high=high_item) 
                    for (low_item,high_item) in inters]).reshape((1,-1))
    
    ## code initial 
    # Cis = np.random.uniform(low=LOW_VAL_Ci, high=HIGH_VAL_Ci, 
    #                         size=(1, M_PLAYERS))
    
    # low = sys_inputs['case'][0]; high = sys_inputs['case'][1]
    # # Pi
    # inters = map(lambda x: (low*x, high*x), Cis.reshape(-1))
    # Pis = np.array([np.random.uniform(low=low_item, high=high_item) 
    #                 for (low_item,high_item) in inters]).reshape((1,-1))
    # # Si
    # inters = map(lambda x: (low*x, high*x), Pis.reshape(-1))
    # Si_maxs = np.array([np.random.uniform(low=low_item, high=high_item) 
    #                 for (low_item,high_item) in inters]).reshape((1,-1))
    # inters = map(lambda x: (low*x, high*x), Si_maxs.reshape(-1))
    # Sis = np.array([np.random.uniform(low=low_item, high=high_item) 
    #                 for (low_item,high_item) in inters]).reshape((1,-1))
    return Cis, Pis, Si_maxs, Sis

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def find_path_to_variables(name_dir, ext=".npy", threshold= 0.89, n_depth=2):
    """
    create the complet path to variables of extensions .npy

    Parameters
    ----------
    name_dir : TYPE
        DESCRIPTION.
    ext : String
        DESCRIPTION.
        extension of variables
    threshold: float
        DESCRIPTION.
        percent of specified files in a directory 
    depth: integer
        DESCRIPTION.
        number of subdirectories we have to open
        
    Returns
    -------
    path_to_variables: String.

    """
    dirs = []
    dirs.append(name_dir)
    boolean = True
    depth = 0
    while boolean:
        depth += 1
        reps = os.listdir(name_dir)
        rep = reps[np.random.randint(0,len(reps))]
        dirs.append(rep)
        #print("dirs = {}, rep={}".format(dirs, os.path.join(*dirs) ))
        files = os.listdir(os.path.join(*dirs))
        located_files = [fn for fn in files if fn.endswith(".npy")]
        if round(len(located_files)/len(files)) >= threshold \
            or depth == n_depth:
            boolean = False
        else:
            name_dir = os.path.join(*dirs)
            
    path_to_variables = os.path.join(*dirs)
    #print('dirs = {}, path_to_variables={}, type={}'.format(dirs, path_to_variables, type( path_to_variables)))
    
    return path_to_variables
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
    
def test_find_path_to_variables():
    name_dir = "tests"
    depth = 2
    ext = "npy"
    find_path_to_variables(name_dir, ext, depth)
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------    
if __name__ == "__main__":
    ti = time.time()
    test_fct_positive()
    test_generate_energy_unit_price_SG()
    
    path_file = test_find_path_to_variables()
    
    print("runtime = {}".format(time.time() - ti))