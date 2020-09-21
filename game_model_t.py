# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:29:14 2020

@author: jwehounou
"""
import time
import math
import numpy as np
import smartgrids_actors as sg
import fonctions_auxiliaires as fct_aux

N_INSTANCE = 10
M_PLAYERS = 10

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

def select_storage_politic(ai, state_ai, mode_i, Ci_t_plus_1, Pi_t_plus_1, 
                           pi_0_plus, pi_0_minus, pi_hp_plus, pi_hp_minus):
    """
    choose gamma_i following the rules on the document.

    Returns
    -------
    gamma_i.

    """
    Si_minus, Si_plus = 0, 0
    X = pi_0_minus; Y = pi_hp_minus
    if state_ai == "state1":
        Si_minus = 0 if mode_i == "CONS+" else 0
        Si_plus = ai.get_Si() if mode_i == "CONS-" else 0
    elif state_ai == "state2":
        Si_minus = ai.get_Si() - (ai.get_Ci() - ai.get_Pi()) \
            if mode_i == "DIS" else 0
        Si_plus = ai.get_Si() if mode_i == "CONS-" else 0
    elif state_ai == "state3":
        Si_minus = ai.get_Si() if mode_i == "PROD" else 0
        Si_plus = max(ai.get_Si_max(), 
                      ai.get_Si() + (ai.get_Pi() - ai.get_Ci()))
    else:
        Si_minus, Si_plus = np.inf, np.inf
        
    
    if fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_minus:
        ai.set_gamma_i(X-1)
    elif fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_plus:
        ai.set_gamma_i(Y-1)
    elif fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_minus and \
        fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_plus:
            res = (fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1)- Si_minus)\
                   / (Si_plus - Si_minus)
            Z = X + (Y-X)*res
            ai.set_gamma_i(math.floor(Z))
    
def generate_players():
    """
    generate N_PLAYERS+1 players.

    Returns
    -------
    list of N+1 players of type Agents

    """
    
    return None

def create_strategy_profile():
    """
    create a serie of players having a decision of state and a mode. 
    gamma_i is chosen following the formula on the document.
        
    
    Returns
    -------
    a list of agents.

    """
    return None    
    

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
        
        V_i, ben_i, cst_i = compute_utility_ai(ai.get_gamma_i(), b0, c0, 
                                               prod_i, cons_i, r_i)
        
        # print("V_i={}, OK={}".format(V_i,OK))
        if V_i[0] != np.inf or V_i[0] != np.nan:
            OK += 1
    
    print("test_compute_utility_ai OK = {}".format(round(OK/N_INSTANCE)))
       
def test_select_storage_politic():
    Pis = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    Cis = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    Sis = np.random.randint(1, 15, N_INSTANCE) + np.random.randn(1, N_INSTANCE) 
    Si_maxs = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE) 
    gamma_is = np.random.randint(1, 3, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    
    pi_0_plus, pi_0_minus, pi_hp_plus, pi_hp_minus = np.random.randn(4,1)
    In_sg, Out_sg = np.random.randint(2, 10, 2)
    
    
    OK = 0; cpt = 0
    for ag in np.concatenate((Pis, Cis, Sis, Si_maxs, gamma_is)).T:
        # print("ag={}".format(ag))
        cpt += 1
        ai = sg.Agent(*ag)
        pi_0_plus, pi_0_minus, pi_hp_plus, pi_hp_minus = np.random.randn(4,1)
        In_sg, Out_sg = np.random.randint(2, 10, 2)
        b0, c0 = compute_energy_unit_price(pi_0_plus, pi_0_minus, 
                                           pi_hp_plus, pi_hp_minus,
                                           In_sg, Out_sg)
        # print("b0={}, c0={}, pi_0_plus={}".format(b0[0], c0[0], pi_0_plus))
        
        Pi_t_plus_1 = (np.random.randint(1, 30, 1) + np.random.randn())[0]
        Ci_t_plus_1 = (np.random.randint(1, 30, 1) + np.random.randn())[0]
        
        state_ai, mode_i, prod_i, cons_i, r_i = ai.select_mode_compute_r_i(
                                                    ai.identify_state())
        gamma_i_old = ai.get_gamma_i()
        # print("Avant types: state_ai={} state_ai={},".format(type(state_ai), state_ai)+ 
        #       " mode_i={} mode_i={},".format(type(mode_i), mode_i)+
        #       " prod_i={} prod_i={},".format(type(prod_i), prod_i)+
        #       " cons_i={} cons_i={},".format(type(cons_i), cons_i)+
        #       " r_i={} r_i={},".format(type(r_i), r_i)+
        #       " gamma_i={} gamma_i={}.".format(type(ai.get_gamma_i()), ai.get_gamma_i()))
              
        select_storage_politic(ai, state_ai, mode_i, Ci_t_plus_1, Pi_t_plus_1, 
                           pi_0_plus, pi_0_minus, pi_hp_plus, pi_hp_minus)
        new_gamma_i = ai.get_gamma_i()
        if gamma_i_old != new_gamma_i:
            OK += 1
            # print("gamma_i: cpt={}, old ={}, new ={}, type={} \n".format(
            #         cpt, gamma_i_old, new_gamma_i, type(new_gamma_i)))
            
    print("test_select_storage_politic: OK = {}".format(round(OK/N_INSTANCE)))
        
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_compute_energy_unit_price()
    test_compute_utility_ai()
    test_select_storage_politic()
    print("runtime = {}".format(time.time() - ti))