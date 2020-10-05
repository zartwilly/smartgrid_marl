# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:33:06 2020

@author: jwehounou
"""
import os
import sys
import time
import math
import json
import numpy as np
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import execution_game as exec_game

from datetime import datetime
from pathlib import Path


#------------------------------------------------------------------------------
#                       definition of constantes
#------------------------------------------------------------------------------
# N_INSTANCE = 10
# M_PLAYERS = 10
# CHOICE_RU = 1
# CASE1 = (0.75, 1.5)
# CASE2 = (0.4, 0.75)
CASE3 = (0, 0.3)
# LOW_VAL_Ci = 1 
# HIGH_VAL_Ci = 30
# NUM_PERIODS = 5
INDEX_ATTRS = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, "gamma_i":4, 
               "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9}
#------------------------------------------------------------------------------
# ebouche de solution generale
#                   definitions of functions
#------------------------------------------------------------------------------
def generate_random_values(zero=0):
    """
    at time t=0, the values of C_T_plus_1, P_T_plus_1 are null. 
    see decription of smartgrid system model document.

    if zero=0 then this is zero array else not null array
    Returns
    -------
    array of shape (2,).

    """
    C_T_plus_1, P_T_plus_1 = np.array((0,0))
    if zero != 0:
        C_T_plus_1 = np.random.random_sample()
        P_T_plus_1 = np.random.random_sample()
    return  C_T_plus_1, P_T_plus_1

def initialize_game_create_agents_t0(sys_inputs):    
    """
    initialize a game by create M_PLAYERS agents called players and affect 
    random values (see the case 1, 2, 3) for all players and for all period 
    time (NUM_PERIODS)

    Parameters
    ----------
    sys_inputs : dict
        contain variables like hp prices, sg prices, storage capacity 
        at t=0 and t=1, the future production and consumption, 
        the case of the game (more production, more consumption,).

    Returns
    -------
     arr_pls:  array of players of shape 1*M M*T
     arr_pls_M_T: array of players of shape M_PLAYERS*NUM_PERIODS*9
     pl_m^t contains a list of 
             Ci, Pi, Si, Si_max, gamma_i, prod_i, cons_i, r_i, state_i, mode_i
     m \in [0,M-1] and t \in [0,T-1] 
    """
    # declaration variables
    arr_pls = np.array([]) 
    arr_pls_M_T = [] # np.array([])
    Ci_t_plus_1, Pi_t_plus_1 = sys_inputs["Ci_t_plus_1"], sys_inputs["Pi_t_plus_1"] 
    pi_0_plus, pi_0_minus = sys_inputs["pi_0_plus"], sys_inputs["pi_0_minus"] 
    pi_hp_plus, pi_hp_minus = sys_inputs["pi_hp_plus"], sys_inputs["pi_hp_minus"]
    
    # create the M players
    Cis, Pis, Si_maxs, Sis = fct_aux.generate_Cis_Pis_Sis(
                                n_items = exec_game.M_PLAYERS, 
                                low_1 = exec_game.LOW_VAL_Ci, 
                                high_1 = exec_game.HIGH_VAL_Ci,
                                low_2 = sys_inputs['case'][0],
                                high_2 = sys_inputs['case'][1]
                                )
    
    gamma_is = np.zeros(shape=(1, exec_game.M_PLAYERS))
    prod_is = np.zeros(shape=(1, exec_game.M_PLAYERS))
    cons_is = np.zeros(shape=(1, exec_game.M_PLAYERS))
    r_is = np.zeros(shape=(1, exec_game.M_PLAYERS))
    state_is = np.array([None]*exec_game.M_PLAYERS).reshape((1,-1))
    for ag in np.concatenate((Pis, Cis, Sis, Si_maxs, gamma_is, prod_is, 
                              cons_is, r_is, state_is)).T:
        pl = players.Player(*ag)
        pl.get_state_i()
        pl.select_mode_i()
        pl.update_prod_cons_r_i()
        pl.select_storage_politic(Ci_t_plus_1, Pi_t_plus_1, 
                                  pi_0_plus, pi_0_minus, 
                                  pi_hp_plus, pi_hp_minus)
        arr_pls = np.append(arr_pls, pl) #arr_pls.append(pl)
        
        a_i_t_s = []
        Ci = pl.get_Ci(); Pi = pl.get_Pi(); Si = pl.get_Si(); 
        Si_max = pl.get_Si_max(); gamma_i = pl.get_gamma_i(); 
        prod_i = pl.get_prod_i(); cons_i = pl.get_cons_i(); 
        r_i = pl.get_r_i(); state_i = pl.get_state_i()
        mode_i = pl.get_cons_i()
        a_i_t_s.append([Ci, Pi, Si, Si_max, gamma_i, 
                        prod_i, cons_i, r_i, state_i, mode_i])
        
        # for each player, generate the attribut values of players a each time.
        Cis, Pis, Si_maxs, Sis = fct_aux.generate_Cis_Pis_Sis(
                                    n_items = exec_game.NUM_PERIODS, 
                                    low_1 = exec_game.LOW_VAL_Ci, 
                                    high_1 = exec_game.HIGH_VAL_Ci,
                                    low_2 = sys_inputs['case'][0],
                                    high_2 = sys_inputs['case'][1]
                                    )
        for (Ci, Pi, Si, Si_max) in np.concatenate((Cis, Pis, Sis, Si_maxs)).T:
            pl.set_Ci(Ci, update=False)
            pl.set_Pi(Pi, update=False)
            pl.set_Si(Si, update=False)
            pl.set_Si_max(Si_max, update=False)
            pl.get_state_i()
            pl.select_mode_i()
            pl.update_prod_cons_r_i()
            # TODO IMPORTANT
            # est il necessaire to update S_i^t pdt la phase d'initialisation des Pi^t, Ci^t, Si^t, Si^t_max,...
            pl.select_storage_politic(Ci_t_plus_1, Pi_t_plus_1, 
                                  pi_0_plus, pi_0_minus, 
                                  pi_hp_plus, pi_hp_minus)
            
            gamma_i = pl.get_gamma_i(); prod_i = pl.get_prod_i() 
            cons_i = pl.get_cons_i(); r_i = pl.get_r_i(); 
            state_i = pl.get_state_i(); mode_i = pl.get_mode_i()
            # print("Type: gamma_i={}:{}, cons_i={}:{}, prod_i={}, r_i={}, Pi={}, Ci={}".format(
            #     type(gamma_i), gamma_i, type(cons_i), cons_i, type(prod_i), type(r_i), type(Pi), type(Ci)))
            a_i_t_s.append([Ci, Pi, Si, Si_max, gamma_i, 
                            prod_i, cons_i, r_i, state_i, mode_i])
            
        # TODO a resoudre cela
        # arr_pls_M_T = np.array(a_i_t_s) \
        #     if len(arr_pls_M_T)==0 \
        #     else np.concatenate((arr_pls_M_T, np.array(a_i_t_s)),axis=0) #arr_pls_M_T.append(a_i_t_s)
        # # arr_pls_M_T.reshape(M_PLAYERS,NUM_PERIODS+1,-1) #--> FALSE
        arr_pls_M_T.append(a_i_t_s)
        
    arr_pls_M_T = np.array(arr_pls_M_T, dtype=object)
    return arr_pls, arr_pls_M_T

def compute_prod_cons_SG(arr_pls_M_T, t):
    """
    

    Parameters
    ----------
    arr_pls_M_T : array of shape M_PLAYERS*NUM_PERIODS+1*9
        DESCRIPTION.
    t : integer
        DESCRIPTION.

    Returns
    -------
    In_sg, Out_sg : float, float.
    
    """
    In_sg = sum( arr_pls_M_T[:,t, INDEX_ATTRS["prod_i"]] )
    Out_sg = sum( arr_pls_M_T[:,t, INDEX_ATTRS["cons_i"]] )
    return In_sg, Out_sg
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

def extract_values_to_array(arr_pls_M_T, list_t,
                            attribut_position=INDEX_ATTRS["gamma_i"]):
    """
    extract list of values for one variable at time t or a list of period 
    for all players

    Parameters
    ----------
    arr_pls_M_T :  array of shape M_PLAYERS*NUM_PERIODS*9
        DESCRIPTION.
    list_t : list of specified NUM_PERIODS; len(list_t) <= NUM_PERIODS
        DESCRIPTION
    attribut_position : Integer, optional
        DESCRIPTION. The default is 4 = INDEX_ATTRS["gamma_i"]. 
        exemple, thes positions of gamma_i, prod_i, cons_i and r_i are 
                respectively 4, 5, 6, 7

    Returns
    -------
    numpy array of shape=(M_PLAYERS,) if type of list_t = integer.
    numpy array of shape=(M_PLAYERS,list_t) if type of list_t = list

    """
    vals = arr_pls_M_T[:, list_t, attribut_position] \
        if type(list_t) is int \
        else arr_pls_M_T[:, list_t, attribut_position]
    return vals
    
def compute_utility_players(arr_pls_M_T, gamma_is, t, b0, c0):
    """
    calculate the benefit and the cost of each player at time t

    Parameters
    ----------
    arr_pls_M_T : array of shape M_PLAYERS*NUM_PERIODS*9
        DESCRIPTION.
    gamma_is :  array of shape (M_PLAYERS,)
        DESCRIPTION.
    t : integer
        DESCRIPTION.
    b0 : float
        benefit per unit.
    c0 : float
        cost per unit.

    Returns
    -------
    bens: benefits of M_PLAYERS, shape (M_PLAYERS,).
    csts: costs of M_PLAYERS, shape (M_PLAYERS,)
    """
    bens = b0 * arr_pls_M_T[:, t, INDEX_ATTRS["prod_i"]] \
            + gamma_is * arr_pls_M_T[:, t, INDEX_ATTRS["r_i"]]
    csts = c0 * arr_pls_M_T[:, t, INDEX_ATTRS["cons_i"]]
    return bens, csts

def determine_new_pricing_sg(prod_is_0_t, cons_is_0_t, 
                             pi_hp_plus, pi_hp_minus, t, dbg=False):
    """
    # TODO quel valeur je donne a new_pi_0_plus, new_pi_0_minus quand sum_cons = sum_prod = 0
    determine the new price of energy in the SG from the amounts produced 
    (prod_is), consumpted (cons_is) and the price of one unit of energy 
    exchanged with the HP

    Parameters
    ----------
    prod_is_0_t : array of (M_PLAYERS, t)
        DESCRIPTION.
    cons_is_0_t : array of (M_PLAYERS, t)
        DESCRIPTION.
    pi_hp_plus : a float 
        Energy unit price for exporting from SG to HP.
    pi_hp_minus : float
        Energy unit price for importing from HP to SG.
    t : integer
        period of time
    Returns
    -------
    new_pi_0_plus, new_pi_0_minus: float, float.

    """
    """
    sum_ener_res_plus : sum of residual energy (diff between prod_i and cons_i) 
                        for all periods i.e 0 to t 
    sum_ener_res_minus : sum of residual energy (diff between cons_i and prod_i) 
                            for all periods i.e 0 to t
    sum_prod : sum of production for all players for all periods i.e 0 to t-1
    sum_cons : sum of consumption for all players for all periods i.e 0 to t-1
    """

    sum_ener_res_plus = 0            # sum of residual energy (diff between prod_i and cons_i) for all periods i.e 0 to t 
    sum_ener_res_minus = 0           # sum of residual energy (diff between cons_i and prod_i) for all periods i.e 0 to t 
    sum_prod = 0                     # sum of production for all players for all periods i.e 0 to t-1
    sum_cons = 0                     # sum of consumption for all players for all periods i.e 0 to t-1
    for t_prime in range(0, t+1):
        sum_prod_i_tPrime = sum(prod_is_0_t[:, t_prime])
        sum_cons_i_tPrime = sum(cons_is_0_t[:, t_prime])
        sum_prod += sum_prod_i_tPrime
        sum_cons += sum_ener_res_minus
        sum_ener_res_plus += fct_aux.fct_positive(sum_prod_i_tPrime, 
                                                  sum_cons_i_tPrime)
        sum_ener_res_minus += fct_aux.fct_positive(sum_cons_i_tPrime, 
                                                   sum_prod_i_tPrime)
        
        # print("t_prime={}, sum_prod={}, sum_cons={}, sum_ener_res_plus={}, sum_ener_res_minus={}".format(
        #     t_prime, sum_prod, sum_cons, sum_ener_res_plus, sum_ener_res_minus))
        
    new_pi_0_minus = round( pi_hp_minus*sum_ener_res_minus / sum_cons, 3) \
                        if sum_cons != 0 else np.nan
    new_pi_0_plus = round( pi_hp_plus*sum_ener_res_plus / sum_prod, 3) \
                        if sum_prod != 0 else np.nan
                            
    return new_pi_0_plus, new_pi_0_minus

def update_player(arr_pls, arr_pls_M_T, t, list_valeurs_by_variable):
    """
    TODO : la mise a jour des players NE SE FAIT PAS.
    update attributs of players in arr_pls. 
    Values of attributs come from list_valeurs_by_variable
    
    Parameters
    ----------
    arr_pls : array of players with a shape (M_PLAYERS,) 
        DESCRIPTION.
    arr_pls_M_T: array of shape M_PLAYERS*NUM_PERIODS*10
        DESCRIPTION.
    t : a time of NUM_PERIODS
        DESCRIPTION.
    list_valeurs_by_variable : list of tuples
        DESCRIPTION.
        EXEMPLE
            [(4,new_gamma_i_t_plus_1_s)])
            4 denotes update gamma_i and it is a position of the list 
            [Ci, Pi, Si, Si_max, gamma_i, prod_i, cons_i, r_i, state_i, mode_i]
            [(1,new_gamma_i_t_plus_1_s)])
            1 denotes update gamma_i and generate/update prod_i or cons_i from gamma_i
            [(2, state_ais)]
            2 denotes update variable state_ai without generate/update prod_i or cons_i
    Returns
    -------
    arr_pls, arr_pls_M_T.

    """
    #arr_pls_new = np.array([])
    for tup_variable in list_valeurs_by_variable:
        # print("tup_variable={}".format(tup_variable))
        num_attr, vals = tup_variable
        if num_attr == INDEX_ATTRS["gamma_i"]:
            for num_pl,pl in enumerate(arr_pls):
                pl.set_gamma_i(vals[num_pl])
                arr_pls_M_T[num_pl,t,num_attr] = vals[num_pl]
        if num_attr == INDEX_ATTRS["prod_i"]:
            for num_pl,pl in enumerate(arr_pls):
                pl.set_prod_i(vals[num_pl])
                arr_pls_M_T[num_pl,t,num_attr] = vals[num_pl]
        if num_attr == INDEX_ATTRS["cons_i"]:
            for num_pl,pl in enumerate(arr_pls):
                pl.set_cons_i(vals[num_pl])
                arr_pls_M_T[num_pl,t,num_attr] = vals[num_pl]
        if num_attr == INDEX_ATTRS["r_i"]:
            for num_pl,pl in enumerate(arr_pls):
                pl.set_r_i(vals[num_pl])
                arr_pls_M_T[num_pl,t,num_attr] = vals[num_pl]
        if num_attr == INDEX_ATTRS["state_i"]:
            for num_pl,pl in enumerate(arr_pls):
                pl.set_state_i(vals[num_pl])
                # pl.state_i = vals[num_pl]
                # print("num_pl ={}, new_state={}, pl={}".format(num_pl, pl.get_state_i(),pl))
                arr_pls_M_T[num_pl,t,num_attr] = vals[num_pl]
                # arr_pls_new = np.append(arr_pls_new, pl)
        if num_attr == INDEX_ATTRS["mode_i"]:
            for num_pl,pl in enumerate(arr_pls):
                pl.set_mode_i(vals[num_pl])
                arr_pls_M_T[num_pl,t,num_attr] = vals[num_pl]
        # arr_pls_new = np.append(arr_pls_new, pl)    
    return arr_pls, arr_pls_M_T

def determinate_Ci_Pi_t_plus_1(arr_pls_M_T, t):
    """
    determinate the values of Ci and Pi at t+1
    
    Parameters
    ----------
    arr_pls_M_T : array of shape  M_PLAYERS*NUM_PERIODS*10
    t: integer
    
    Returns
    -------
    Ci_t_plus_1_s: array of Ci of shape (M_PLAYERS,) 
    Pi_t_plus_1_s: array of Pi of shape (M_PLAYERS,)
    """
    Ci_t_plus_1_s = np.zeros(shape=(exec_game.M_PLAYERS,))
    Pi_t_plus_1_s = np.zeros(shape=(exec_game.M_PLAYERS,))
    if t != 0 and t != exec_game.NUM_PERIODS:
        Ci_t_plus_1_s = arr_pls_M_T[:, t+1, INDEX_ATTRS["Ci"]]
        Pi_t_plus_1_s = arr_pls_M_T[:, t+1, INDEX_ATTRS["Pi"]]
    elif t == exec_game.NUM_PERIODS:
        Ci_t_plus_1_s = np.zeros(shape=(exec_game.M_PLAYERS,))
        Pi_t_plus_1_s = np.zeros(shape=(exec_game.M_PLAYERS,))
    elif t == 0:
        Ci_t_plus_1_s = np.zeros(shape=(exec_game.M_PLAYERS,))
        Pi_t_plus_1_s = arr_pls_M_T[:, t+1, INDEX_ATTRS["Pi"]]
    return Ci_t_plus_1_s, Pi_t_plus_1_s

def select_storage_politic_players(arr_pls, state_ais, mode_is, 
                                   Ci_t_plus_1_s, Pi_t_plus_1_s, 
                                   pi_0_plus, pi_0_minus,
                                   pi_hp_plus, pi_hp_minus):
    """
    choose the storage politic gamma_i of the M_PLAYERS players  
    following the rules on the smartgrid system model document 
    (see storage politic).

    Parameters
    ----------
    arr_pls : array of shape (M_players,)
        DESCRIPTION.
    state_ais : states of players with a shape (M_players,)
        DESCRIPTION.
    mode_is : modes of players with a shape (M_players,)
        DESCRIPTION.
    Ci_t_plus_1_s : consumption of M_players players at t+1 with a shape (M_players,)
        DESCRIPTION.
    Pi_t_plus_1_s : production of M_players players at t+1 with a shape (M_players,)
        DESCRIPTION.
    pi_0_plus : float, new production price inside SG
        DESCRIPTION.
    pi_0_minus : float, new consumption price inside SG 
        DESCRIPTION.
    pi_hp_plus : float, price for exporting energy from SG to HP
        DESCRIPTION.
    pi_hp_minus : float, price for importing energy from HP to SG 
        DESCRIPTION.

    Returns
    -------
    new_gamma_is: array of shape (M_players,).

    """
    new_gamma_is = np.array([])
    for tu in zip(arr_pls, state_ais, mode_is, Ci_t_plus_1_s, Pi_t_plus_1_s):
        pl = tu[0]; state_i = tu[1]; #mode_i = tu[2]; 
        Ci_t_plus_1 = tu[3]; Pi_t_plus_1 = tu[4]
        
        #update pl
        pl.set_state_i(state_i)
        # pl.set_mode_i(mode_i) ---> pas besoin
        # compute gamma_i
        pl.select_storage_politic(Ci_t_plus_1, Pi_t_plus_1, 
                               pi_0_plus, pi_0_minus, 
                               pi_hp_plus, pi_hp_minus)
        # append new gamma in new_gamma_is
        new_gamma_i = pl.get_gamma_i()
        new_gamma_is = np.append(new_gamma_is, new_gamma_i)
        
    return new_gamma_is

def select_mode_compute_r_i(arr_pls, arr_pls_M_T, t):
    """
    select a mode_i and compute r_i for a time t+1

    Parameters
    ----------
    arr_pls : array of shape (M_players,)
        DESCRIPTION.
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*10
        DESCRIPTION.
    t : integer 
        DESCRIPTION.

    NB: [Ci, Pi, Si, Si_max, gamma_i, prod_i, cons_i, r_i, state_i, mode_i]
    Returns
    -------
    state_ais, mode_is.

    """
    
    Cis = arr_pls_M_T[:, t, INDEX_ATTRS["Ci"]]
    Pis = arr_pls_M_T[:, t, INDEX_ATTRS["Pi"]]
    Sis = arr_pls_M_T[:, t, INDEX_ATTRS["Si"]]
    Si_maxs = arr_pls_M_T[:, t, INDEX_ATTRS["Si_max"]]
    gamma_is = arr_pls_M_T[:, t, INDEX_ATTRS["gamma_i"]]
    prod_is = arr_pls_M_T[:, t, INDEX_ATTRS["prod_i"]]
    cons_is = arr_pls_M_T[:, t,INDEX_ATTRS["cons_i"]]
    
    if t == exec_game.NUM_PERIODS-1:
        return arr_pls, arr_pls_M_T
    else:
        for num_pl, pl in enumerate(arr_pls):
            pl.set_Ci(Cis[num_pl], update=False)
            pl.set_Pi(Pis[num_pl], update=False)
            pl.set_Si(Sis[num_pl], update=False)
            pl.set_Si_max(Si_maxs[num_pl], update=False)
            pl.set_gamma_i(gamma_is[num_pl])
            pl.set_prod_i(prod_is[num_pl], update=False)
            pl.set_cons_i(cons_is[num_pl], update=False)
            
            pl.select_mode_i()
            pl.update_prod_cons_r_i()
        
            # update arr_pls_M_T at mode, r_i
            arr_pls_M_T[num_pl, t+1, INDEX_ATTRS["r_i"]] = pl.get_r_i()
            arr_pls_M_T[num_pl, t+1, INDEX_ATTRS["state_i"]] = pl.get_state_i()
            arr_pls_M_T[num_pl, t+1, INDEX_ATTRS["mode_i"]] = pl.get_mode_i()
    
    return arr_pls, arr_pls_M_T
            
def compute_real_utility(arr_pls_M_T, BENs, CSTs, B0s, C0s,
                         pi_sg_plus, pi_sg_minus, choice=2):
    """
    compute the real utility of all players

    Parameters
    ----------
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*10
        DESCRIPTION.
    BENs : array of M_PLAYERS*NUM_PERIODS.
        list of M_PLAYERS items. each item is a list of NUM_PERIODS benefits .
    CSTs : array of M_PLAYERS*NUM_PERIODS.
        list of M_PLAYERS items. each item is a list of NUM_PERIODS costs. 
    B0s : array of (NUM_PERIODS,). 
        list of NUM_PERIODS items. each item is a benefit at a period t. 
    C0s : array of (NUM_PERIODS,).
        list of NUM_PERIODS items. each item is a benefit at a period t
    pi_sg_plus_s : list of energy price exported to HP. NUM_PERIODS items
        DESCRIPTION.
    pi_sg_minus_s : list of energy price imported from HP. NUM_PERIODS items
        DESCRIPTION.

    Returns
    -------
    RUs : array of M_PLAYERS utilities, shape=(M_PLAYERS,).

    """
    
    RUs = None
    if choice == 1:
        RUs = BENs - CSTs
    else:
        B_is =  arr_pls_M_T[:,:, INDEX_ATTRS["prod_i"]] * B0s    # array of (M_PLAYERS,)
        C_is =  arr_pls_M_T[:,:, INDEX_ATTRS["cons_i"]] * C0s    # array of (M_PLAYERS,)
        RUs = B_is - C_is
    return RUs.sum(axis=1)

##------------------ OLD test game model a t --> debut ----------------------------    
def game_model_SG_t_old(arr_pls, arr_pls_M_T, t, 
                    pi_0_plus, pi_0_minus, 
                    pi_hp_plus, pi_hp_minus):
    """
    create the game model at time t

    parameters:
    -------
    arr_pls : array of shape (M_players,)
        DESCRIPTION.
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*10
        DESCRIPTION.
    t : integer 
        DESCRIPTION.    
    Returns
    -------
    arr_pls, arr_pls_M_T,
    b0: integer. benefit price for exporting energy from SG to HP
    c0: integer. cost price for exporting energy from HP to SG
    bens, csts: (M_PLAYERS,). benefit and cost prices depending of gamma_is
    new_pi_0_plus, new_pi_0_minus, pi_sg_plus, pi_sg_minus : integer. new price inside SG
    """
    # compute In_sg, Out_sg
    In_sg, Out_sg = compute_prod_cons_SG(arr_pls_M_T, t)
    # compute price of an energy unit price for cost and benefit players
    b0, c0 = compute_energy_unit_price(
                    pi_0_plus, pi_0_minus, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg)
    #B0s.append(b0);
    #C0s.append(c0);
    # compute cost and benefit players by energy exchanged.
    gamma_is = extract_values_to_array(arr_pls_M_T, t, attribut_position=INDEX_ATTRS["gamma_i"])        
    bens, csts = compute_utility_players(arr_pls_M_T, gamma_is, t, b0, c0)
    #BENs.append(bens); CSTs.append(csts)
    
    #compute the new prices pi_0_plus and pi_0_minus from a pricing model in the document
    prod_is_0_t = extract_values_to_array(
                    arr_pls_M_T, 
                    range(0, t+1), 
                    attribut_position=INDEX_ATTRS["prod_i"])
    cons_is_0_t = extract_values_to_array(
                    arr_pls_M_T, 
                    range(0, t+1), 
                    attribut_position=INDEX_ATTRS["cons_i"])
    new_pi_0_plus, new_pi_0_minus = determine_new_pricing_sg(
                                        prod_is_0_t, cons_is_0_t,
                                        pi_hp_plus, pi_hp_minus, t)
    new_pi_0_plus = pi_0_plus if new_pi_0_plus is np.nan else new_pi_0_plus
    new_pi_0_minus = pi_0_minus if new_pi_0_minus is np.nan else new_pi_0_minus
    
    #pi_0_plus, pi_0_minus = new_pi_0_plus, new_pi_0_minus
    # update cons_i, prod_i for a each player
    prod_is = extract_values_to_array(arr_pls_M_T, t, 
                                      attribut_position=INDEX_ATTRS["prod_i"])
    cons_is = extract_values_to_array(arr_pls_M_T, t,
                                      attribut_position=INDEX_ATTRS["cons_i"])
    arr_pls, arr_pls_M_T = update_player(arr_pls, arr_pls_M_T, t,
                                         [(INDEX_ATTRS["prod_i"],cons_is), 
                                          (INDEX_ATTRS["cons_i"],prod_is)])
    
    
    # determination of Ci_t_plus_1_s, Pi_t_plus_1_s at t+1
    Ci_t_plus_1_s, Pi_t_plus_1_s = determinate_Ci_Pi_t_plus_1(
                                    arr_pls_M_T, t)
    # definition of the new storage politic
    state_ais = extract_values_to_array(arr_pls_M_T, t, 
                                        attribut_position=INDEX_ATTRS["state_i"])
    mode_is = extract_values_to_array(arr_pls_M_T, t, 
                                      attribut_position=INDEX_ATTRS["mode_i"])
    new_gamma_i_t_plus_1_s = select_storage_politic_players(
                                arr_pls, state_ais, mode_is, 
                                Ci_t_plus_1_s, Pi_t_plus_1_s, 
                                new_pi_0_plus, new_pi_0_minus,
                                pi_hp_plus, pi_hp_minus)
    arr_pls, arr_pls_M_T = update_player(arr_pls, arr_pls_M_T, t,
                                         [(INDEX_ATTRS["gamma_i"],
                                           new_gamma_i_t_plus_1_s)])
    # choose a new state, new mode at t+1 and update arr_pls et arr_pls_M_T
    arr_pls, arr_pls_M_T = select_mode_compute_r_i(arr_pls, arr_pls_M_T, t)
    
    # update prices of the SG
    new_pi_sg_plus, new_pi_sg_minus = new_pi_0_plus, new_pi_0_minus
    
    return arr_pls, arr_pls_M_T, b0, c0, bens, csts, \
            new_pi_0_plus, new_pi_0_minus, new_pi_sg_plus, new_pi_sg_minus
    

    
def game_model_SG_old(pi_hp_plus, pi_hp_minus, pi_0_plus, pi_0_minus, case):
    """
    create a game for one period T = [1..T]

    Returns
    -------
    None.

    """
    S_0, S_1, = 0, 0;
    # TODO:how to determine Ci_t_plus_1_s, Pi_t_plus_1_s?
    ## generate random values of Ci_t_plus_1_s, Pi_t_plus_1_s
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    B0s, C0s, BENs, CSTs = [], [], [], []
    pi_sg_plus_s, pi_sg_minus_s = [], [] 
    pi_sg_plus, pi_sg_minus = 0, 0
    
    for t in range(0, exec_game.NUM_PERIODS):
        new_arr_pls, new_arr_pls_M_T, \
        b0, c0, bens, csts, \
        new_pi_0_plus, new_pi_0_minus, \
        new_pi_sg_plus, new_pi_sg_minus = game_model_SG_t_old(
                                                arr_pls, arr_pls_M_T, t, 
                                                pi_0_plus, pi_0_minus, 
                                                pi_hp_plus, pi_hp_minus)
        # update new_arr_pls, new_arr_pls_M_T
        arr_pls, arr_pls_M_T = new_arr_pls, new_arr_pls_M_T
        # add to variable vectors
        B0s.append(b0); C0s.append(c0);
        BENs.append(bens); CSTs.append(csts)
        pi_sg_plus, pi_sg_minus = new_pi_sg_plus, new_pi_sg_minus
        pi_sg_plus_s.append(pi_sg_plus); pi_sg_minus_s.append(pi_sg_minus)
        pi_0_plus, pi_0_minus = new_pi_0_plus, new_pi_0_minus
        
    # compute real utility of all players
    BENs = np.array(BENs, dtype=object)      # array of M_PLAYERS*NUM_PERIODS
    CSTs = np.array(CSTs, dtype=object)      # array of M_PLAYERS*NUM_PERIODS
    B0s = np.array(B0s, dtype=object)        # array of (NUM_PERIODS,)
    C0s = np.array(C0s, dtype=object)        # array of (NUM_PERIODS,)
    RUs = compute_real_utility(arr_pls_M_T, BENs, CSTs, B0s, C0s,
                               pi_sg_plus_s, pi_sg_minus_s, 
                               exec_game.CHOICE_RU)
    
    return RUs

def game_model_SG_T(T, pi_hp_plus, pi_hp_minus, pi_0_plus, pi_0_minus, case):
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
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    B0s, C0s, BENs, CSTs = [], [], [], []
    pi_sg_plus, pi_sg_minus = 0, 0
    for t in range(0, T):
        # compute In_sg, Out_sg
        In_sg, Out_sg = compute_prod_cons_SG(arr_pls_M_T, t)
        # compute price of an energy unit price for cost and benefit players
        b0, c0 = compute_energy_unit_price(
                        pi_0_plus, pi_0_minus, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg)
        B0s.append(b0);
        C0s.append(c0);
        # compute cost and benefit players by energy exchanged.
        gamma_is = extract_values_to_array(arr_pls_M_T, t, attribut_position=INDEX_ATTRS["gamma_i"])        
        bens, csts = compute_utility_players(arr_pls_M_T, gamma_is, t, b0, c0)
        BENs.append(bens); CSTs.append(csts)
        
        #compute the new prices pi_0_plus and pi_0_minus from a pricing model in the document
        prod_is_0_t = extract_values_to_array(
                        arr_pls_M_T, 
                        range(0,t+1), 
                        attribut_position=INDEX_ATTRS["prod_i"])
        cons_is_0_t = extract_values_to_array(
                        arr_pls_M_T, 
                        range(0,t+1), 
                        attribut_position=INDEX_ATTRS["cons_i"])
        new_pi_0_plus, new_pi_0_minus = determine_new_pricing_sg(
                                            prod_is_0_t, cons_is_0_t,
                                            pi_hp_plus, pi_hp_minus, t)
        pi_0_plus, pi_0_minus = new_pi_0_plus, new_pi_0_minus
        # update cons_i, prod_i for a each player
        prod_is = extract_values_to_array(
                        arr_pls_M_T, t, 
                        attribut_position=INDEX_ATTRS["prod_i"])
        cons_is = extract_values_to_array(
                        arr_pls_M_T, t, 
                        attribut_position=INDEX_ATTRS["cons_i"])
        arr_pls, arr_pls_M_T = update_player(arr_pls, arr_pls_M_T, t,
                                             [(INDEX_ATTRS["prod_i"],cons_is), 
                                              (INDEX_ATTRS["cons_i"],prod_is)])
        
        
        # determination of Ci_t_plus_1_s, Pi_t_plus_1_s at t+1
        Ci_t_plus_1_s, Pi_t_plus_1_s = determinate_Ci_Pi_t_plus_1(
                                        arr_pls_M_T, t)
        # definition of the new storage politic
        state_ais = extract_values_to_array(arr_pls_M_T, t, attribut_position=8)
        mode_is = extract_values_to_array(arr_pls_M_T, t, attribut_position=9)
        new_gamma_i_t_plus_1_s = select_storage_politic_players(
                                    arr_pls, state_ais, mode_is, 
                                    Ci_t_plus_1_s, Pi_t_plus_1_s, 
                                    pi_0_plus, pi_0_minus,
                                    pi_hp_plus, pi_hp_minus)
        arr_pls, arr_pls_M_T = update_player(arr_pls, arr_pls_M_T, t,
                                             [(INDEX_ATTRS["gamma_i"],
                                               new_gamma_i_t_plus_1_s)])
        # choose a new state, new mode at t+1 and update arr_pls et arr_pls_M_T
        arr_pls, arr_pls_M_T = select_mode_compute_r_i(arr_pls, arr_pls_M_T, t)
        
        # update prices of the SG
        pi_sg_plus, pi_sg_minus = pi_0_plus, pi_0_minus
        
    # compute real utility of all players
    BENs = np.array(BENs, dtype=object)      # array of M_PLAYERS*NUM_PERIODS
    CSTs = np.array(CSTs, dtype=object)      # array of M_PLAYERS*NUM_PERIODS
    B0s = np.array(B0s, dtype=object)        # array of (NUM_PERIODS,)
    C0s = np.array(C0s, dtype=object)        # array of (NUM_PERIODS,)
    RUs = compute_real_utility(arr_pls_M_T, BENs, CSTs, B0s, C0s,
                               pi_sg_plus, pi_sg_minus, exec_game.CHOICE_RU)
    
    return RUs

##------------------ OLD test game model a t --> fin   ----------------------------    


def game_model_SG_t(arr_pls, arr_pls_M_T, t, 
                        pi_0_plus_t, pi_0_minus_t, 
                        pi_hp_plus, pi_hp_minus):
    """
    create the game model at time t

    parameters:
    -------
    arr_pls : array of shape (M_players,)
        DESCRIPTION.
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*INDEX_ATTRS
        DESCRIPTION.
    t : integer 
        DESCRIPTION.
    pi_0_plus_t : integer, exportation price at time t from pl to SG
        DESCRIPTION.
    pi_0_minus_t: integer, importation price at time t from SG to pl
        DESCRIPTION.
    pi_hp_plus: integer, exportation price at time t from SG to HP
        DESCRIPTION
    pi_hp_minus: integer, importation price at time t from HP to SG
        DESCRIPTION
    Returns
    -------
    arr_pls, arr_pls_M_T,
    b0: integer. benefit price for exporting energy from SG to HP
    c0: integer. cost price for exporting energy from HP to SG
    bens, csts: (M_PLAYERS,). benefit and cost prices depending of gamma_is
    new_pi_0_plus, new_pi_0_minus, pi_sg_plus, pi_sg_minus : integer. new price inside SG
    """
    ## update attributs of players because of modification of gamma_i 
    ## depending of Pi_t_plus_1 and Ci_t_plus_1
    # update P_i, C_i, Si_max, Si attributs of a player belonging to arr_pls
    arr_pls, arr_pls_M_T \
        = update_player(
            arr_pls, arr_pls_M_T, t,
            [(INDEX_ATTRS["Pi"], arr_pls_M_T[:,t,INDEX_ATTRS["Pi"]]),
             (INDEX_ATTRS["Ci"], arr_pls_M_T[:,t,INDEX_ATTRS["Ci"]]),
             (INDEX_ATTRS["Si"], arr_pls_M_T[:,t,INDEX_ATTRS["Si"]]),
             (INDEX_ATTRS["Si_max"], arr_pls_M_T[:,t,INDEX_ATTRS["Si_max"]])
             ])
    # calculer P_i^{t+1} et C_i^{t+1}
    Pi_t_plus_1_s = arr_pls_M_T[:,t,INDEX_ATTRS["Pi"]]
    Ci_t_plus_1_s = arr_pls_M_T[:,t,INDEX_ATTRS["Ci"]]
    
    new_gamma_is = np.array([])
    for num_pl, pl in enumerate(arr_pls):
        # select state_i and mode_i
        state_i = pl.get_state_i()
        pl.select_mode_i()
        # compute gamma_i
        pl.select_storage_politic(Ci_t_plus_1_s[num_pl], 
                                  Pi_t_plus_1_s[num_pl], 
                                  pi_0_plus_t, pi_0_minus_t, 
                                  pi_hp_plus, pi_hp_minus)
        # append new gamma in new_gamma_is
        new_gamma_i = pl.get_gamma_i()
        new_gamma_is = np.append(new_gamma_is, new_gamma_i)
        # compute prod_i, cons_i and r_i 
        pl.update_prod_cons_r_i()
        # update arr_pls_M_T
        arr_pls_M_T[num_pl, t, INDEX_ATTRS["state_i"]] = state_i
        arr_pls_M_T[num_pl, t, INDEX_ATTRS["mode_i"]] = pl.get_mode_i()
        arr_pls_M_T[num_pl, t, INDEX_ATTRS["prod_i"]] = pl.get_prod_i()
        arr_pls_M_T[num_pl, t, INDEX_ATTRS["cons_i"]] = pl.get_cons_i()
        arr_pls_M_T[num_pl, t, INDEX_ATTRS["r_i"]] = pl.get_r_i()
        
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = compute_prod_cons_SG(arr_pls_M_T, t)
    # compute prices of an energy unit price for cost and benefit players
    b0, c0 = compute_energy_unit_price(
                    pi_0_plus_t, pi_0_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg)
    # compute cost and benefit players by energy exchanged.
    gamma_is = extract_values_to_array(arr_pls_M_T, t, 
                                       attribut_position=INDEX_ATTRS["gamma_i"])        
    bens, csts = compute_utility_players(arr_pls_M_T, gamma_is, t, b0, c0)
    # compute the new prices pi_0_plus_t_plus_1 and pi_0_minus_t_plus_1 from a pricing model in the document
    diff_energy_cons_t = 0
    diff_energy_prod_t = 0
    for k in range(0,t+1):
        prod_is_k = arr_pls_M_T[:,k,INDEX_ATTRS["prod_i"]]
        cons_is_k = arr_pls_M_T[:,k,INDEX_ATTRS["cons_i"]]
        diff_energy_cons_k = fct_aux.fct_positive(sum(cons_is_k), sum(prod_is_k))
        diff_energy_cons_t += diff_energy_cons_k
        diff_energy_prod_k = fct_aux.fct_positive(sum(prod_is_k), sum(cons_is_k))
        diff_energy_prod_t += diff_energy_prod_k
    sum_cons = sum(sum(arr_pls_M_T[:, :t, INDEX_ATTRS["cons_i"] ]))
    pi_sg_minus_t = round( pi_hp_minus*diff_energy_cons_t / sum_cons, 3) \
                        if sum_cons != 0 else np.nan
    sum_prod = sum(sum(arr_pls_M_T[:, :t, INDEX_ATTRS["prod_i"] ]))
    pi_sg_plus_t = round( pi_hp_plus*diff_energy_prod_t / sum_prod, 3) \
                        if sum_prod != 0 else np.nan
    
    pi_0_plus_t_plus_1 = pi_0_plus_t if pi_sg_minus_t is np.nan else pi_sg_minus_t
    pi_0_minus_t_plus_1 = pi_0_minus_t if pi_sg_plus_t is np.nan else pi_sg_plus_t
                            
    
    return arr_pls, arr_pls_M_T, \
            b0, c0, \
            bens, csts, \
            pi_0_minus_t_plus_1, pi_0_plus_t_plus_1
        
def game_model_SG(pi_hp_plus, pi_hp_minus, pi_0_plus, pi_0_minus, case):
    """
    create a game for all periods of time NUM_PERIODS = [1..T]

    Returns
    -------
    arr_pls_M_T, RUs, B0s, C0s, BENs, CSTs, pi_sg_plus_s, pi_sg_minus_s.
    
    arr_pls_M_T: array of players with a shape M_PLAYERS*NUM_PERIODS*INDEX_ATTRS
    BENs: array of M_PLAYERS*NUM_PERIODS
    CSTs: array of M_PLAYERS*NUM_PERIODS
    B0s: array of (NUM_PERIODS,)
    C0s: array of (NUM_PERIODS,)
    pi_sg_plus_s: array of (NUM_PERIODS,)
    pi_sg_minus_s: array of (NUM_PERIODS,)

    """
    # create players and its attribut values for all values.
    S_0, S_1, = 0, 0;
    # TODO:how to determine Ci_t_plus_1_s, Pi_t_plus_1_s?
    ## generate random values of Ci_t_plus_1_s, Pi_t_plus_1_s
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    # initialize arrays containing values for each period t < NUM_PERIODS
    B0s, C0s, BENs, CSTs = [], [], [], []
    pi_sg_plus_s, pi_sg_minus_s = [], [] 
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    
    # run for all NUM_PERIODS
    for t in range(0, exec_game.NUM_PERIODS):
        # determine the value of pi_0_plus_t and pi_0_minus_t
        pi_0_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t
        pi_0_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t
        
        pi_sg_plus_s.append(pi_0_plus_t) 
        pi_sg_minus_s.append(pi_0_minus_t) 
        
        arr_pls, arr_pls_M_T, \
        b0, c0, \
        bens, csts, \
        pi_0_minus_t_plus_1, pi_0_plus_t_plus_1 \
            = game_model_SG_t(arr_pls, arr_pls_M_T, t, 
                        pi_0_plus_t, pi_0_minus_t, 
                        pi_hp_plus, pi_hp_minus)
        pi_sg_plus_t = pi_0_minus_t_plus_1
        pi_sg_minus_t = pi_0_minus_t_plus_1
        
        # update arrays containing values for each period t < NUM_PERIODS
        B0s.append(b0); C0s.append(c0);
        BENs.append(bens); CSTs.append(csts)
        
    # compute real utility of all players
    BENs = np.array(BENs, dtype=object).T      # array of M_PLAYERS*NUM_PERIODS
    CSTs = np.array(CSTs, dtype=object).T      # array of M_PLAYERS*NUM_PERIODS
    B0s = np.array(B0s, dtype=object)        # array of (NUM_PERIODS,)
    C0s = np.array(C0s, dtype=object)        # array of (NUM_PERIODS,)
    pi_sg_plus_s = np.array(pi_sg_plus_s, dtype=object)     # array of (NUM_PERIODS,)
    pi_sg_minus_s = np.array(pi_sg_minus_s, dtype=object)   # array of (NUM_PERIODS,)
    print("pi_sg_plus_s={},pi_sg_minus_s={},".format(pi_sg_plus_s.shape,pi_sg_minus_s.shape))
    RUs = compute_real_utility(arr_pls_M_T, BENs, CSTs, B0s, C0s,
                               pi_sg_plus_s, pi_sg_minus_s, 
                               exec_game.CHOICE_RU)
        
    return arr_pls_M_T, RUs, B0s, C0s, BENs, CSTs, pi_sg_plus_s, pi_sg_minus_s

def run_game_model_SG(pi_hp_plus, pi_hp_minus, 
                      pi_0_plus, pi_0_minus, 
                      case, path_to_save):
    """
    execute the game of SG model with some parameters

    Parameters
    ----------
    pi_hp_plus : integer
        DESCRIPTION.
    pi_hp_minus : integer
        DESCRIPTION.
    pi_0_plus : integer
        DESCRIPTION.
    pi_0_minus : integer
        DESCRIPTION.
    case : tuple
        DESCRIPTION.
    path_to_save : string, path to save json variables.
        DESCRIPTION

    Returns
    -------
    None.

    """
    # pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    # pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s \
        = game_model_SG(pi_hp_plus, pi_hp_minus, 
                        pi_0_plus, pi_0_minus, 
                        case=case)
    # # save arrays to json
    # json_arr_pls_M_T = json.dumps(arr_pls_M_T, cls=fct_aux.NumpyEncoder)
    # json_RUs = json.dumps(RUs, cls=fct_aux.NumpyEncoder)
    # json_B0s = json.dumps(B0s, cls=fct_aux.NumpyEncoder)
    # json_C0s = json.dumps(C0s, cls=fct_aux.NumpyEncoder)
    # json_BENs = json.dumps(BENs, cls=fct_aux.NumpyEncoder)
    # json_CSTs = json.dumps(CSTs, cls=fct_aux.NumpyEncoder)
    # # save locally
    # with open(os.path.join(path_to_save, "arr_pls_M_T.json"),'w') as f:
    #     f.dump(json_arr_pls_M_T)
    # with open(os.path.join(path_to_save, "RUs.json"),'w') as f:
    #     f.dump(json_RUs)
    # with open(os.path.join(path_to_save, "B0s.json"),'w') as f:
    #     f.dump(json_B0s)
    # with open(os.path.join(path_to_save, "C0s"),'w') as f:
    #     f.dump(json_C0s)
    # with open(os.path.join(path_to_save, "BENs.json"),'w') as f:
    #     f.dump(json_BENs)
    # with open(os.path.join(path_to_save, "CSTs.json"),'w') as f:
    #     f.dump(json_CSTs)
    # save locally
    np.save(os.path.join(path_to_save, "arr_pls_M_T.npy"), arr_pls_M_T)
    np.save(os.path.join(path_to_save, "RUs.npy"), RUs)
    np.save(os.path.join(path_to_save, "B0s.npy"), B0s)
    np.save(os.path.join(path_to_save, "C0s.npy"), C0s)
    np.save(os.path.join(path_to_save, "BENs.npy"), BENs)
    np.save(os.path.join(path_to_save, "CSTs.npy"), CSTs)
    np.save(os.path.join(path_to_save, "pi_sg_minus_s.npy"), pi_sg_minus_s)
    np.save(os.path.join(path_to_save, "pi_sg_plus_s.npy"), pi_sg_plus_s)
    
#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------    
def test_initialize_game_create_agents_t0():
    """
    Si OK alors arr_pls.shape = (M_PLAYERS,)
                et arr_pls_M_T.shape = (M_PLAYERS, NUM_PERIODS, 10)

    Returns
    -------
    None.

    """
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    print("shape: arr_pls={}, arr_pls_M_T={}, size={}".format(
            arr_pls.shape, arr_pls_M_T.shape, sys.getsizeof(arr_pls_M_T) ))
    if arr_pls.shape == (exec_game.M_PLAYERS,) \
        and arr_pls_M_T.shape == (exec_game.M_PLAYERS, 
                                  exec_game.NUM_PERIODS+1, 
                                  len(INDEX_ATTRS)):
            print("test_initialize_game_create_agents_t0: OK")
            print("shape: arr_pls={}, arr_pls_M_T={}".format(
            arr_pls.shape, arr_pls_M_T.shape))
    else:
        print("test_initialize_game_create_agents_t0: NOK")
        print("shape: arr_pls={}, arr_pls_M_T={}".format(
            arr_pls.shape, arr_pls_M_T.shape))
    
def test_compute_prod_cons_SG():
    """
    compute the percent of unbalanced grid (production, consumption, balanced)
    at each time.

    Returns
    -------
    None.

    """
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    production = 0; consumption = 0; balanced = 0
    for t in range(0, exec_game.NUM_PERIODS):
        In_sg_t, Out_sg_t = compute_prod_cons_SG(arr_pls_M_T, t)
        if In_sg_t > Out_sg_t:
            production += 1
        elif In_sg_t < Out_sg_t:
            consumption += 1
        else:
            balanced += 1
    
    print("SG: production={}, consumption={}, balanced={}".format(
            round(production/exec_game.NUM_PERIODS,2), 
            round(consumption/exec_game.NUM_PERIODS,2),
            round(balanced/exec_game.NUM_PERIODS,2) ))
        
def test_compute_energy_unit_price():
    pi_0_plus, pi_0_minus, pi_hp_plus, pi_hp_minus = np.random.randn(4,1)
    In_sg, Out_sg = np.random.randint(2, len(INDEX_ATTRS), 2)
    
    b0, c0 = compute_energy_unit_price(pi_0_plus, pi_0_minus, 
                                       pi_hp_plus, pi_hp_minus,
                                       In_sg, Out_sg)
    print("b0={}, c0={}".format(b0[0], c0[0]))
    
def test_extract_values_to_array():
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    # t integer
    OK_int = 0
    for t in range(0, exec_game.NUM_PERIODS):    
        vals = extract_values_to_array(arr_pls_M_T, t, 
                                       attribut_position=INDEX_ATTRS["gamma_i"])
        if vals.shape == (exec_game.M_PLAYERS,):
            OK_int += 1
            #print("test_extract_values_to_array: OK")
        else:
            pass
            #print("test_extract_values_to_array: NOK")
    print("t:integer ==> shape gamma_is: {}".format(vals.shape))
    print("test_extract_values_to_array: OK_int={}".format(
        round(OK_int/exec_game.NUM_PERIODS,2)))
    
    # list_t : list of periods
    list_t = range(1,2+1)
    vals = extract_values_to_array(arr_pls_M_T, list_t, 
                                   attribut_position=INDEX_ATTRS["gamma_i"])
    print("t:list ==> shape gamma_is: {}".format(vals.shape))
    if vals.shape == (exec_game.M_PLAYERS, 2):
        print("test_extract_values_to_array: OK_list")
    else:
        print("test_extract_values_to_array: NOK_list")
    
    
def test_compute_utility_players():
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    OK = 0
    for t in range(0, exec_game.NUM_PERIODS):
        b0, c0 = np.random.randn(), np.random.randn()
        gamma_is = extract_values_to_array(
                        arr_pls_M_T, t, 
                        attribut_position=INDEX_ATTRS["gamma_i"])
        bens, csts = compute_utility_players(arr_pls_M_T, gamma_is, t, b0, c0)
        
        if bens.shape == (exec_game.M_PLAYERS,) \
            and csts.shape == (exec_game.M_PLAYERS,):
            print("bens={}, csts={}, gamma_is={}".format(
                    bens.shape, csts.shape, gamma_is.shape))
            OK += 1
    print("test_compute_utility_players: rp={}".format(
            round(OK/exec_game.NUM_PERIODS,2)))
    
def test_determine_new_pricing_sg():
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    t = np.random.randint(0, exec_game.NUM_PERIODS)
    
    prod_is_0_t = extract_values_to_array(
                    arr_pls_M_T, range(0,t+1), 
                    attribut_position=INDEX_ATTRS["prod_i"])
    cons_is_0_t = extract_values_to_array(
                    arr_pls_M_T, range(0,t+1), 
                    attribut_position=INDEX_ATTRS["cons_i"])
    print("t = {}, shapes prod_is_0_t:{}, cons_is_0_t={}".format(
            t, prod_is_0_t.shape, cons_is_0_t.shape))
    
    new_pi_0_plus, new_pi_0_minus = determine_new_pricing_sg(
                                    prod_is_0_t, cons_is_0_t,
                                    pi_hp_plus, pi_hp_minus, t)
    print("test_determine_new_pricing_sg: OK pi_plus") \
        if new_pi_0_plus != pi_0_plus \
        else print("test_determine_new_pricing_sg: NOK pi_plus")
    print("test_determine_new_pricing_sg: OK pi_minus") \
        if new_pi_0_minus != pi_0_minus \
        else print("test_determine_new_pricing_sg: NOK pi_minus")

def test_update_player():
    """
    short case of helping 
    
    class Player:
        cpt_player = 0
        def __init__(self, state_i):
        self.name = ("").join(["a",str(self.cpt_player)])
        self.state_i = state_i
        Player.cpt_player += 1
    
        def get_state_i(self):
            return self.state_i
        
        def set_state_i(self, new_state)
            self.state_i = new_state

    M = 3
    arr_players = []
    states = "A,B,C,D,E,F,G,H,I,J".split(",")
    for ag in np.concatenate((states)).T
        pl = Player(*ag)
        arr_players = np.append(arr_players, pl)
    for player in arr_players:
        print("Before update: player state={}".format(player.get_state_i()))
        
    def update_state(arr_players):
        states = "A,B,C,D,E,F,G,H,I,J".lower().split(",")
        for num, pl in enumerate(arr_players):
            pl.set_state_i(states[num])
        return arr_players
    
    arr_players = update_state(arr_players)
    for player in arr_players:
        print("After update: player state={}".format(player.get_state_i()))
    
    Returns
    -------
    None.

    """
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    t = np.random.randint(0, exec_game.NUM_PERIODS)
    
    #str_vals = "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,R,S,T,U,V,W,X,Z".split(",")
    str_vals = [item for item in fct_aux.get_random_string(exec_game.M_PLAYERS)]
    num_attr = 8
    list_valeurs_by_variable = [(num_attr, str_vals)]
    arr_pls, arr_pls_M_T = update_player(arr_pls, arr_pls_M_T, t,
                                         list_valeurs_by_variable)
    OK = 0; OK_pls_M_T = 0
    for num_pl,pl in enumerate(arr_pls):
        # print("pl.state_i={}".format(pl.get_state_i()))
        if str(pl.get_state_i()) is str and len(pl.get_state_i()) == 1:
            OK += 1
        if arr_pls_M_T[num_pl,t,num_attr] == str_vals[num_pl]:
            OK_pls_M_T += 1
    print("test_update_player: rp_OK= {}, rp_OK_pls_M_T={}".format(
            round(OK/exec_game.M_PLAYERS,3), 
            round(OK_pls_M_T/exec_game.M_PLAYERS,3)))
    
def test_select_storage_politic_players():
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    t = np.random.randint(0, exec_game.NUM_PERIODS)
    
    Ci_t_plus_1_s, Pi_t_plus_1_s = determinate_Ci_Pi_t_plus_1(
                                        arr_pls_M_T, t)
    # definition of the new storage politic
    state_ais = extract_values_to_array(arr_pls_M_T, t, 
                                        attribut_position=INDEX_ATTRS["state_i"])
    mode_is = extract_values_to_array(arr_pls_M_T, t, 
                                      attribut_position=INDEX_ATTRS["mode_i"])
    new_gamma_i_t_plus_1_s = select_storage_politic_players(
                                    arr_pls, state_ais, mode_is, 
                                    Ci_t_plus_1_s, Pi_t_plus_1_s, 
                                    pi_0_plus, pi_0_minus,
                                    pi_hp_plus, pi_hp_minus)
    OK = 0
    for tu in zip(arr_pls, new_gamma_i_t_plus_1_s):
        pl = tu[0]
        new_gamma_i = tu[1]
        if pl.get_gamma_i() != new_gamma_i:
            OK += 1
    print("test_select_storage_politic_players: rp_OK={}".format( 
            round(OK/exec_game.M_PLAYERS,3)))
    
def test_select_mode_compute_r_i():
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    t = np.random.randint(0, exec_game.NUM_PERIODS)
    
    # replace r_i values by np.nan
    arr_pls_M_T[:,t,INDEX_ATTRS["r_i"]] = np.nan
    
    arr_pls, arr_pls_M_T = select_mode_compute_r_i(arr_pls, arr_pls_M_T, t)
    
    if t == exec_game.NUM_PERIODS-1:
        print("aucune modification de r_i, mode_i, state_i, t=NUM_PERIODS")
    else:
        r_is = arr_pls_M_T[:,t+1,INDEX_ATTRS["r_i"]]
        arr_nan = np.empty(shape=(exec_game.M_PLAYERS))
        arr_nan[:] = None
        if (arr_nan == r_is).all():
            print("test_select_mode_compute_r_i: NOK")
        else:
            print("test_select_mode_compute_r_i: OK")
            # print("r_is = {}".format(r_is))
    
def test_compute_real_utility(dbg=True):
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs) 

    t = exec_game.NUM_PERIODS+1 #np.random.randint(0,NUM_PERIODS)

    BENs = np.random.rand(exec_game.M_PLAYERS,t)
    CSTs = np.random.rand(exec_game.M_PLAYERS,t)
    B0s = np.random.rand(t)
    C0s = np.random.rand(t)

    RUs_1 = compute_real_utility(arr_pls_M_T, BENs, CSTs, B0s, C0s, 
                               pi_sg_plus=0, pi_sg_minus=0, choice=1)
    RUs_2 = compute_real_utility(arr_pls_M_T, BENs, CSTs, B0s, C0s, 
                               pi_sg_plus=0, pi_sg_minus=0, choice=2)
    print("test_compute_real_utility: RUs_1 shape:{}, RUs_2 shape:{}".format(
            RUs_1.shape, RUs_2.shape))
    if dbg:
        print("test_compute_real_utility: RUs_1 shape:{}, sum={}".format(RUs_1.shape, RUs_1))
        print("test_compute_real_utility: RUs_2 shape:{}, sum={}".format(RUs_2.shape, RUs_2))

def test_game_model_SG_t_old():
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs) 

    t = np.random.randint(0, exec_game.NUM_PERIODS)
    #t = 0, NUM_PERIODS # 0: ERROR update gamma_i; NUM_PERIODS: error out of bounds update arr_pls_M_T
    print("test_game_model_SG_t_old t={} => debut".format(t))
        
    new_arr_pls, new_arr_pls_M_T, \
    b0, c0, bens, csts, \
    new_pi_0_plus, new_pi_0_minus, \
    new_pi_sg_plus, new_pi_sg_minus = game_model_SG_t_old(
                                              arr_pls, arr_pls_M_T, t, 
                                              pi_0_plus, pi_0_minus, 
                                              pi_hp_plus, pi_hp_minus)
    ## add to variable vectors
    BENs, CSTs = [], []
    B0s, C0s = [], []
    B0s.append(b0); C0s.append(c0);
    BENs.append(bens); CSTs.append(csts)
    pi_sg_plus, pi_sg_minus = new_pi_sg_plus, new_pi_sg_minus
    pi_0_plus, pi_0_minus = new_pi_0_plus, new_pi_0_minus
    ##
    
    print("test_game_model_SG_t_old t={} => Pas d'erreur".format(t))

def test_game_model_SG_t(dbg=True):
    """
    TODO
    Message a NOK parce que la fonction update_players et 
    la mise a jour des attributs ne se font pas.

    Returns
    -------
    None.

    """
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus_t, pi_0_minus_t = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus_t, "pi_0_minus":pi_0_minus_t}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs) 

    t = np.random.randint(0, exec_game.NUM_PERIODS)
    # modification Ci, Pi de arr_pls_M_T
    arr_pls_M_T[:,t,INDEX_ATTRS["Pi"]] = np.ones(shape=(exec_game.M_PLAYERS,))
    arr_pls_M_T[:,t,INDEX_ATTRS["Ci"]] = np.ones(shape=(exec_game.M_PLAYERS,))
    
    new_arr_pls, new_arr_pls_M_T, \
    b0, c0, bens, csts, \
    pi_0_minus_t_plus_1, pi_0_plus_t_plus_1 \
        = game_model_SG_t(arr_pls, arr_pls_M_T, t+1, 
                                pi_0_plus_t, pi_0_minus_t, 
                                pi_hp_plus, pi_hp_minus)
    gamma_is = arr_pls_M_T[:,t,INDEX_ATTRS["gamma_i"]]
    new_gamma_is = new_arr_pls_M_T[:,t,INDEX_ATTRS["gamma_i"]]    
        
    if dbg:
        print("_____test_game_model_SG_t_____")
        print("t={}, b0={}, c0={}, bens={}, csts={}".format(t,b0,c0,bens,csts))
        print("pi_0_minus_t={}, pi_0_plus_t={}, pi_0_minus_t_plus_1={}, pi_0_plus_t_plus_1={}".format(
               pi_0_minus_t, pi_0_plus_t,pi_0_minus_t_plus_1,pi_0_plus_t_plus_1))
        print("gamma_is={} \n new_gamma_is={}".format(gamma_is, new_gamma_is))
        print("new_arr_pls_M_T={}".format(new_arr_pls_M_T[:,t,INDEX_ATTRS["Pi"]]))
        
    if (new_gamma_is == gamma_is).all():
        print("test_game_model_SG_t: t={} NOK".format(t))
    else:
        print("test_game_model_SG_t: t={} OK".format(t))
    
def test_game_model_SG_old(dbg=True):
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    
    RUs = game_model_SG_old(pi_hp_plus, pi_hp_minus, 
                            pi_0_plus, pi_0_minus, 
                            case=CASE3)
    
    print("test_game_model_SG_old: RUs_1 shape:{}".format(RUs.shape))
    if dbg:
        print("sum:{}".format(RUs))
    
def test_game_model_SG(dbg=True):
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s \
        = game_model_SG(pi_hp_plus, pi_hp_minus, 
                        pi_0_plus, pi_0_minus, 
                        case=CASE3)
    

    print("test_game_model_SG: RUs_1 shape:{}".format(RUs.shape))
    if dbg:
        print(" RUs_1 sum={}".format(RUs))
          
        
def test_run_game_model_SG():
    # create test directory tests and put file in the directory simu_date_hhmm
    name_dir = "tests"; date_hhmm = datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join(name_dir, "simu_"+date_hhmm)
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    
    case = CASE3
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    
    run_game_model_SG(pi_hp_plus, pi_hp_minus, 
                      pi_0_plus, pi_0_minus, 
                      case, path_to_save)
    
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_initialize_game_create_agents_t0()
    test_compute_prod_cons_SG()
    test_compute_energy_unit_price()
    test_extract_values_to_array()
    test_compute_utility_players()
    test_determine_new_pricing_sg()
    test_update_player()
    test_select_storage_politic_players()
    test_select_mode_compute_r_i()
    test_compute_real_utility(False)
    test_game_model_SG_t_old()
    test_game_model_SG_old(False)
    
    test_game_model_SG_t(False)
    test_game_model_SG(False)
    
    test_run_game_model_SG()
    print("runtime = {}".format(time.time() - ti))    
    