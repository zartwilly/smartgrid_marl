# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:33:06 2020

@author: jwehounou
"""
import time
import math
import numpy as np
# import smartgrids_actors as sg
import fonctions_auxiliaires as fct_aux

#------------------------------------------------------------------------------
#                       definition of constantes
#------------------------------------------------------------------------------
N_INSTANCE = 10
M_PLAYERS = 10
CASE1 = (0.75, 1.5)
CASE2 = (0.4, 0.75)
CASE3 = (0, 0.3)

#------------------------------------------------------------------------------
# ebouche de solution generale
#                   definitions of functions
#------------------------------------------------------------------------------
def initialize_game_create_agents_t0(sys_inputs):
    """
    initialize a game by
        * create N+1 agents
        * attribute characteristic values of agents such as agents become players
            characteristic values are :
                - name_ai, state_ai, mode_i, prod_i, cons_i, r_i

    Parameters
    ----------
    sys_inputs : TYPE
        DESCRIPTION.

    Returns
    -------
    arr_pls, arr_val_pls.

    """
    return None

def compute_prod_cons_SG(arr_val_pls):
    """
    compute the production In_sg and the consumption Out_sg in the SG.

    Parameters
    ----------
    arr_val_pls : numpy array (N+1, 6)
        DESCRIPTION.

    Returns
    -------
    In_sg, Out_sg: Integer.

    """
    return None

def determine_new_pricing_sg(prod_is, cons_is, pi_hp_plus, pi_hp_minus):
    """
    determine the new price of energy in the SG from the amounts produced 
    (prod_is), consumpted (cons_is) and the price of one unit of energy 
    exchanged with the HP

    Parameters
    ----------
    prod_is : array of 1 x N
        DESCRIPTION.
    cons_is : array of 1 x N
        DESCRIPTION.
    pi_hp_plus : a float 
        DESCRIPTION.
    pi_hp_minus : float
        DESCRIPTION.

    Returns
    -------
    new_pi_0_plus, new_pi_0_minus.

    """
    return None

def update_player(arr_pls, list_valeurs_by_variable):
    """
    

    Parameters
    ----------
    arr_pls : TYPE
        DESCRIPTION.
    list_valeurs_by_variable : list of tuples
        DESCRIPTION.
        EXEMPLE
            [(1,new_gamma_i_t_plus_1_s)])
            1 denotes update gamma_i and generate/update prod_i or cons_i from gamma_i
            [(2, state_ais)]
            2 denotes update variable state_ai without generate/update prod_i or cons_i
    Returns
    -------
    arr_pls.

    """

def game_model_SG_T(T, pi_hp_plus, pi_hp_minus, pi_0_plus, pi_0_minus):
    """
    create the game model divised in T periods such as players maximize their 
    profits on each period t

    diff between agent and player:
        an agent becomes a player iff he has state_ai, mode_i, prod_i, cons_i, 
        r_is
        
    form of arr_pls : N x 6
        one row: [AG, state_ai, mode_i, prod_i, cons_i, r_i]
    Returns
    -------
    None.

    """
    S_0, S_1, = 0, 0;
    # TODO:how to determine Ci_t_plus_1_s, Pi_t_plus_1_s?
    ## generate random values of Ci_t_plus_1_s, Pi_t_plus_1_s
    C_T_plus_1, P_T_plus_1 = generate_random_values()
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, 
                      "C_T_plus_1_s":C_T_plus_1_s, "P_T_plus_1s":P_T_plus_1_s,
                      "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                      "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls = initialize_game_create_agents_t0(sys_inputs)
    B0s, C0s, BENs, CSTs = [], [], [], []
    pi_sg_plus, pi_sg_minus = 0, 0
    for t in range(0, T):
        # compute In_sg, Out_sg
        In_sg, Out_sg = compute_prod_cons_SG(arr_pls)
        # compute price of an energy unit price for cost and benefit players
        b0, c0 = compute_energy_unit_price(
                        pi_0_plus, pi_0_minus, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg)
        B0s.append(b0);
        C0s.append(C0);
        # compute cost and benefit players by energy exchanged.
        gamma_is = extract_values_to_array(arr_pls, attribut_position=0)        
        bens, csts = compute_utility_players(arr_pls, gamma_is)
        BENs.append(bens), CSTs.append(csts)
        
        #compute the new prices pi_0_plus and pi_0_minus from a pricing model in the document
        cons_is = extract_values_to_array(arr_pls, attribut_position=4)
        prod_is = extract_values_to_array(arr_pls, attribut_position=3)
        pi_0_plus, pi_0_minus = determine_new_pricing_sg(
                                    prod_is, cons_is,
                                    pi_hp_plus, pi_hp_minus)
        # update CONS_i for a each player
        arr_pls = update_player(arr_pls, [(2,cons_is), (2,prod_is)])
        
        
        # definition of the new storage politic
        new_gamma_i_t_plus_1_s = select_storage_politic(
                                    arr_pls, state_ais, mode_is, 
                                    Ci_t_plus_1_s[t], Pi_t_plus_1_s[t], 
                                    pi_0_plus, pi_0_minus)
        arr_pls = update_player(arr_pls, [(1,new_gamma_i_t_plus_1_s)])
        # choose a new state, new mode at t+1
        state_ais, mode_is = select_mode_compute_r_i(arr_pls)
        arr_pls = update_player(arr_pls, [(2,state_ais), (2,mode_is)])
        
        # update prices of the SG
        pi_sg_plus, pi_sg_minus = pi_0_plus, pi_0_minus
        
    # compute utility of 
    RUs = compute_real_utility(arr_pls, pi_sg_plus, pi_sg_minus)
    
    return None

def game_model_SG():
    """
    create a game for one period T = [1..T]

    Returns
    -------
    None.

    """
    return None
    
#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------    
