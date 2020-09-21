# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:29:14 2020

@author: jwehounou
"""
import time
import numpy as np
import smartgrids_actors as sg

N_INSTANCE = 10

#------------------------------------------------------------------------------
#           definitions of functions
#------------------------------------------------------------------------------
def compute_energy_unit_price(pi_0_plus, pi_0_minus, 
                              pi_hp_plus, pi_hp_minus,
                              In_sg, Out_sg):
    """
    compute the unit price of energy benefit and energy cost 
    
    pi_0_plus: the intern benefit price of one unit of energy inside SG
    pi_0_minus: the intern cost price of one unit of energy inside SG
    pi_hp_plus: the intern benefit price of one unit of energy between SG and HP
    pi_hp_minus: the intern cost price of one unit of energy between SG and HP
    Out_sg: the total amount of energy relative to the consumption of the SG
    In_sg: the total amount of energy relative to the production of the SG
    
    Returns
    -------
    bo: the benefit of one unit of energy in SG.
    co: the cost of one unit of energy in SG.

    """
    c0 = pi_0_minus if In_sg >= Out_sg \
                    else (Out_sg - In_sg)*pi_hp_minus + In_sg*pi_0_minus
    b0 = pi_0_plus if In_sg < Out_sg \
                    else In_sg * pi_0_plus + (- Out_sg + In_sg)*pi_0_plus
    
    return b0, c0

def compute_utility_ai(gamma_i, b0, c0, prod_i, cons_i, r_i):
    """
    compute the utility of an agent 

    Returns
    -------
    V_i, cst_i, ben_i

    """
    ben_i = b0 * prod_i + gamma_i * r_i
    cst_i = c0 * cons_i
    V_i = ben_i - cst_i
    return V_i, cst_i, ben_i



#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------    
def test_compute_energy_unit_price():
    pi_0_plus, pi_0_minus, pi_hp_plus, pi_hp_minus = np.random.randn(4,1)
    In_sg, Out_sg = np.random.randint(2, 10, 2)
    
    b0, c0 = compute_energy_unit_price(pi_0_plus, pi_0_minus, 
                                       pi_hp_plus, pi_hp_minus,
                                       In_sg, Out_sg)
    print("b0={}, c0={}".format(b0[0], c0[0]))
    
def test_compute_utility_ai():
    Pis = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    Cis = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    Sis = np.random.randint(1, 15, N_INSTANCE) + np.random.randn(1, N_INSTANCE) 
    Si_maxs = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE) 
    gamma_is = np.random.randint(1, 3, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    
    OK = 0
    for ag in np.concatenate((Pis, Cis, Sis, Si_maxs, gamma_is)).T:
        # print("ag={}".format(ag))
        ai = sg.Agent(*ag)
        
        pi_0_plus, pi_0_minus, pi_hp_plus, pi_hp_minus = np.random.randn(4,1)
        In_sg, Out_sg = np.random.randint(2, 10, 2)
        b0, c0 = compute_energy_unit_price(pi_0_plus, pi_0_minus, 
                                           pi_hp_plus, pi_hp_minus,
                                           In_sg, Out_sg)
        # print("b0={}, c0={}, pi_0_plus={}".format(b0[0], c0[0], pi_0_plus))
        
        state_ai, mode_i, prod_i, cons_i, r_i = ai.select_mode_compute_r_i(
                                                    ai.identify_state())
        # print("state_ai={}, mode_i={}, prod_i={}, cons_i={}, r_i={}"
              # .format(state_ai, mode_i, prod_i, cons_i, r_i))
        
        V_i, ben_i, cst_i = compute_utility_ai(ai.gamma_i, b0, c0, 
                                               prod_i, cons_i, r_i)
        
        # print("V_i={}, OK={}".format(V_i,OK))
        if V_i[0] != np.inf or V_i[0] != np.nan:
            OK += 1
    
    print("test_compute_utility_ai OK = {}".format(round(OK/N_INSTANCE)))
        
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_compute_energy_unit_price()
    test_compute_utility_ai()
    print("runtime = {}".format(time.time() - ti))