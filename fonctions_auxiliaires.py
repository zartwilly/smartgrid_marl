# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:45:38 2020

@author: jwehounou
"""
import os
import sys
import time
import math
import json
import string
import random
import numpy as np
import pandas as pd
import itertools as it
import smartgrids_players as players

from datetime import datetime
from pathlib import Path

#------------------------------------------------------------------------------
#                       definition of constantes
#------------------------------------------------------------------------------
M_PLAYERS = 10
NUM_PERIODS = 50

CHOICE_RU = 1
N_DECIMALS = 2
NB_REPEAT_K_MAX = 4
STOP_LEARNING_PROBA = 0.9

Ci_LOW = 10
Ci_HIGH = 60

global PI_0_PLUS_INIT, PI_0_MINUS_INIT
PI_0_PLUS_INIT = 4 #20
PI_0_MINUS_INIT = 3 #10

SET_AC = ["setA", "setC"]
SET_ABC = ["setA", "setB", "setC"]
SET_AB1B2C = ["setA", "setB1",  "setB2", "setC"]
STATES = ["state1", "state2", "state3"]

STATE1_STRATS = ("CONS+", "CONS-")                                             # strategies possibles pour l'etat 1 de a_i
STATE2_STRATS = ("DIS", "CONS-")                                               # strategies possibles pour l'etat 2 de a_i
STATE3_STRATS = ("DIS", "PROD")

CASE1 = (1.7, 2.0) #(0.75, 1.5)
CASE2 = (0.4, 0.75)
CASE3 = (0, 0.3)

PROFIL_H = (0.6, 0.2, 0.2)
PROFIL_M = (0.2, 0.6, 0.2)
PROFIL_L = (0.2, 0.2, 0.6)

INDEX_ATTRS = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, "gamma_i":4, 
               "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9,
               "Profili":10, "Casei":11, "R_i_old":12, "Si_old":13, 
               "balanced_pl_i": 14, "formule":15}

NON_PLAYING_PLAYERS = {"PLAY":1, "NOT_PLAY":0}
ALGO_NAMES_BF = ["BEST-BRUTE-FORCE", "BAD-BRUTE-FORCE", "MIDDLE-BRUTE-FORCE"]
ALGO_NAMES_NASH = ["BEST-NASH", "BAD-NASH", "MIDDLE-NASH"]

# manual debug constants
MANUEL_DBG_GAMMA_I = np.random.randint(low=2, high=21, size=1)[0]              #5
MANUEL_DBG_PI_SG_PLUS_T_K = 8
MANUEL_DBG_PI_SG_MINUS_T_K = 10
MANUEL_DBG_PI_0_PLUS_T_K = 4 
MANUEL_DBG_PI_0_MINUS_T_K = 3

RACINE_PLAYER = "player"

#_________________            AUTOMATE CONSTANCES           ________________

#AUTOMATE_FILENAME_ARR_PLAYERS_ROOT = "arr_pl_M_T_players_set1_{}_repSet1_{}_set2_{}_repSet2_{}_periods_{}.npy"
AUTOMATE_FILENAME_ARR_PLAYERS_ROOT = "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}.npy"
AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C = "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAC = "arr_pl_M_T_players_setA_{}_setC_{}_periods_{}.npy"


AUTOMATE_INDEX_ATTRS_DBG = {"Ci":0, "Pi":1, "Si_init":24, "Si":2, "Si_max":3, 
               "gamma_i":4, 
               "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9,
               "Profili":10, "Casei":11, "R_i_old":12, "Si_old":13, 
               "balanced_pl_i": 14, "formule":15, "Si_minus":16,
               "Si_plus":17, "u_i": 18, "bg_i": 19,
               "S1_p_i_j_k": 20, "S2_p_i_j_k": 21, 
               "non_playing_players":22, "set":23}

AUTOMATE_INDEX_ATTRS = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, 
               "gamma_i":4, 
               "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9,
               "Profili":10, "Casei":11, "R_i_old":12, "Si_old":13, 
               "balanced_pl_i": 14, "formule":15, "Si_minus":16,
               "Si_plus":17, "u_i": 18, "bg_i": 19,
               "S1_p_i_j_k": 20, "S2_p_i_j_k": 21, 
               "non_playing_players":22, "set":23}

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

# _________        determine opposite mode_i: debut       _____________________
def find_out_opposite_mode(state_i, mode_i):
    """
    look for the opposite mode of the player.
    for example, 
    if state_i = state1, the possible modes are CONS+ and CONS-
    the opposite mode of CONS+ is CONS- and this of CONS- is CONS+
    """
    mode_i_bar = None
    if state_i == STATES[0] \
        and mode_i == STATE1_STRATS[0]:                                        # state1, CONS+
        mode_i_bar = STATE1_STRATS[1]                                          # CONS-
    elif state_i == STATES[0] \
        and mode_i == STATE1_STRATS[1]:                                        # state1, CONS-
        mode_i_bar = STATE1_STRATS[0]                                          # CONS+
    elif state_i == STATES[1] \
        and mode_i == STATE2_STRATS[0]:                                        # state2, DIS
        mode_i_bar = STATE2_STRATS[1]                                          # CONS-
    elif state_i == STATES[1] \
        and mode_i == STATE2_STRATS[1]:                                        # state2, CONS-
        mode_i_bar = STATE2_STRATS[0]                                          # DIS
    elif state_i == STATES[2] \
        and mode_i == STATE3_STRATS[0]:                                        # state3, DIS
        mode_i_bar = STATE3_STRATS[1]                                          # PROD
    elif state_i == STATES[2] \
        and mode_i == STATE3_STRATS[1]:                                        # state3, PROD
        mode_i_bar = STATE3_STRATS[0]                                          # DIS

    return mode_i_bar
# _________        determine opposite mode_i: fin         _____________________

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

def compute_utility_players(arr_pl_M_T, gamma_is, t, b0, c0):
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
    bens = b0 * arr_pl_M_T[:, t, INDEX_ATTRS["prod_i"]] \
            + gamma_is * arr_pl_M_T[:, t, INDEX_ATTRS["r_i"]]
    csts = c0 * arr_pl_M_T[:, t, INDEX_ATTRS["cons_i"]]
    bens = np.around(np.array(bens, dtype=float), N_DECIMALS)
    csts = np.around(np.array(csts, dtype=float), N_DECIMALS)
    return bens, csts 

def compute_prod_cons_SG(arr_pl_M_T, t):
    """
    compute the production In_sg and the consumption Out_sg in the SG.

    Parameters
    ----------
    arr_pl_M_T : array of shape (M_PLAYERS,NUM_PERIODS,len(INDEX_ATTRS))
        DESCRIPTION.
    t : integer
        DESCRIPTION.

    Returns
    -------
    In_sg, Out_sg : float, float.
    
    """
    In_sg = sum( arr_pl_M_T[:, t, INDEX_ATTRS["prod_i"]].astype(np.float64) )
    Out_sg = sum( arr_pl_M_T[:, t, INDEX_ATTRS["cons_i"]].astype(np.float64) )
    return In_sg, Out_sg
 
def compute_gamma_4_period_t(arr_pl_M_T_K_vars, t, 
                             pi_0_plus, pi_0_minus,
                             pi_hp_plus, pi_hp_minus, dbg=False):
    """
    compute gamma_i for all players 
    
    arr_pl_M_T_K_vars: shape (m_players, t_periods, len(vars)) or 
                             (m_players, t_periods, k_steps, len(vars))
    """
    m_players = arr_pl_M_T_K_vars.shape[0]
    t_periods = arr_pl_M_T_K_vars.shape[1]
    
    arr_pl_vars = None
    arr_pl_vars = arr_pl_M_T_K_vars.copy()
    
    if len(arr_pl_M_T_K_vars.shape) == 3:
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["state_i"]]
            Pi = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si"]] \
                    if t == 0 \
                    else arr_pl_vars[num_pl_i, t-1, AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si_max"]]
            Ci_t_plus_1 = arr_pl_vars[num_pl_i, 
                                       t+1, 
                                       AUTOMATE_INDEX_ATTRS["Ci"]] \
                            if t+1<t_periods \
                            else 0
            Pi_t_plus_1 = arr_pl_vars[num_pl_i, 
                                      t+1, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]] \
                            if t+1 < t_periods \
                            else 0
            Si_t_minus, Si_t_plus = None, None
            X, Y = None, None
            if state_i == STATES[0]:                                   # state1 or Deficit
                Si_t_minus = 0
                Si_t_plus = Si
                X = pi_0_minus
                Y = pi_hp_minus
            elif state_i == STATES[1]:                                 # state2 or Self
                Si_t_minus = Si - (Ci - Pi)
                Si_t_plus = Si
                X = pi_0_minus
                Y = pi_hp_minus
            elif state_i == STATES[2]:                                 # state3 or Surplus
                Si_t_minus = Si
                Si_t_plus = max(Si_max, Si+(Pi-Ci))
                X = pi_0_plus
                Y = pi_hp_plus
                
            gamma_i = None
            Si_t_plus_1 = fct_positive(Ci_t_plus_1, Pi_t_plus_1)
            if Si_t_plus_1 < Si_t_minus:
                gamma_i = X - 1
            elif Si_t_plus_1 >= Si_t_plus:
                gamma_i = Y + 1
            elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                res = ( Si_t_plus_1 - Si_t_minus) / (Si_t_plus - Si_t_minus)
                Z = X + (Y-X)*res
                gamma_i = math.floor(Z)  
                
            arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_i
            arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si_minus"]] = Si_t_minus
            arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si_plus"]] = Si_t_plus
            
            bool_gamma_i = (gamma_i >= min(pi_0_minus, pi_0_plus)-1) \
                            & (gamma_i <= max(pi_hp_minus, pi_hp_plus)+1)
            print("GAMMA : t={}, player={}, val={}, bool_gamma_i={}"\
                  .format(t, num_pl_i, gamma_i, bool_gamma_i)) if dbg else None
                
    elif len(arr_pl_M_T_K_vars.shape) == 4:
        arr_pl_vars = arr_pl_M_T_K_vars.copy()
        k = 0
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_vars[num_pl_i, t, k, 
                                  AUTOMATE_INDEX_ATTRS["state_i"]]
            Pi = arr_pl_vars[num_pl_i, t, k,
                                  AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_vars[num_pl_i, t, k,
                                  AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_vars[num_pl_i, t, k,
                                  AUTOMATE_INDEX_ATTRS["Si"]] \
                    if t == 0 \
                    else arr_pl_vars[num_pl_i, t-1, k,
                                  AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_vars[num_pl_i, t, k,
                                  AUTOMATE_INDEX_ATTRS["Si_max"]]
            Ci_t_plus_1 = arr_pl_vars[num_pl_i, 
                                       t+1, k,
                                       AUTOMATE_INDEX_ATTRS["Ci"]] \
                            if t+1<t_periods \
                            else 0
            Pi_t_plus_1 = arr_pl_vars[num_pl_i, 
                                      t+1, k,
                                      AUTOMATE_INDEX_ATTRS["Pi"]] \
                            if t+1 < t_periods \
                            else 0
            Si_t_minus, Si_t_plus = None, None
            X, Y = None, None
            if state_i == STATES[0]:                                           # state1 or Deficit
                Si_t_minus = 0
                Si_t_plus = Si
                X = pi_0_minus
                Y = pi_hp_minus
            elif state_i == STATES[1]:                                         # state2 or Self
                Si_t_minus = Si - (Ci - Pi)
                Si_t_plus = Si
                X = pi_0_minus
                Y = pi_hp_minus
            elif state_i == STATES[2]:                                         # state3 or Surplus
                Si_t_minus = Si
                Si_t_plus = max(Si_max, Si+(Pi-Ci))
                X = pi_0_plus
                Y = pi_hp_plus
                
            gamma_i = None
            Si_t_plus_1 = fct_positive(Ci_t_plus_1, Pi_t_plus_1)
            if Si_t_plus_1 < Si_t_minus:
                gamma_i = X-1
            elif Si_t_plus_1 >= Si_t_plus:
                gamma_i = Y+1
            elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                res = ( Si_t_plus_1 - Si_t_minus) / (Si_t_plus - Si_t_minus)
                Z = X + (Y-X)*res
                gamma_i = math.floor(Z)  
                
            arr_pl_vars[num_pl_i, t, :, 
                        AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_i
            arr_pl_vars[num_pl_i, t, :, 
                        AUTOMATE_INDEX_ATTRS["Si_minus"]] = Si_t_minus
            arr_pl_vars[num_pl_i, t, :, 
                        AUTOMATE_INDEX_ATTRS["Si_plus"]] = Si_t_plus
            arr_pl_vars[num_pl_i, t, :, 
                        AUTOMATE_INDEX_ATTRS["S_i"]] = Si
            
            bool_gamma_i = (gamma_i >= min(pi_0_minus, pi_0_plus)-1) \
                            & (gamma_i <= max(pi_hp_minus, pi_hp_plus)+1)
            print("GAMMA : t={}, player={}, val={}, bool_gamma_i={}, {}"\
                  .format(t, num_pl_i, gamma_i, bool_gamma_i, state_i)) \
                if dbg else None

    return arr_pl_vars
  
# new version of compute gamma and state ---> debut
def update_variables(arr_pl_M_T_K_vars, variables, shape_arr_pl,
                     num_pl_i, t, k, gamma_i, Si,
                     pi_0_minus, pi_0_plus, 
                     pi_hp_minus, pi_hp_plus, dbg):
    
    # ____              update cell arrays: debut               _______
    if shape_arr_pl == 4:
        for (var,val) in variables:
            arr_pl_M_T_K_vars[num_pl_i, t, :,
                        AUTOMATE_INDEX_ATTRS[var]] = val
            
    elif shape_arr_pl == 3:
        for (var,val) in variables:
            arr_pl_M_T_K_vars[num_pl_i, t,
                    AUTOMATE_INDEX_ATTRS[var]] = val
    # ____              update cell arrays: fin                 _______
    
    bool_gamma_i = (gamma_i >= min(pi_0_minus, pi_0_plus)-1) \
                    & (gamma_i <= max(pi_hp_minus, pi_hp_plus)+1)
    print("GAMMA : t={}, player={}, val={}, bool_gamma_i={}"\
          .format(t, num_pl_i, gamma_i, bool_gamma_i)) if dbg else None

    Si_t_minus_1 = None
    if shape_arr_pl == 3 and t > 0:
        Si_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, t-1, 
                               AUTOMATE_INDEX_ATTRS["Si"]] 
    elif shape_arr_pl == 3 and t == 0:
        Si_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, t, 
                               AUTOMATE_INDEX_ATTRS["Si"]]
    elif shape_arr_pl == 4 and t > 0:
        Si_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, t-1, k, 
                               AUTOMATE_INDEX_ATTRS["Si"]] 
    elif shape_arr_pl == 4 and t == 0:
        Si_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, t, k,
                               AUTOMATE_INDEX_ATTRS["Si"]]
        
    print("Si_t_minus_1={}, Si={}".format(Si_t_minus_1, Si)) \
        if dbg else None
            
    return arr_pl_M_T_K_vars
    
def get_values_Pi_Ci_Si_Simax_Pi1_Ci1(arr_pl_M_T_K_vars, 
                                      num_pl_i, t, k,
                                      t_periods,
                                      shape_arr_pl):
    """
    return the values of Pi, Ci, Si, Si_max, Pi_t_plus_1, Ci_t_plus_1 from arrays
    """
    Pi = arr_pl_M_T_K_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Pi"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Pi"]]
    Ci = arr_pl_M_T_K_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Ci"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Ci"]]
    Si_max = arr_pl_M_T_K_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si_max"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Si_max"]]
    Si = None
    if t > 0:
        Si = arr_pl_M_T_K_vars[num_pl_i, t-1, 
                                       AUTOMATE_INDEX_ATTRS["Si"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t-1, k,
                                       AUTOMATE_INDEX_ATTRS["Si"]]
    else:
        Si = arr_pl_M_T_K_vars[num_pl_i, t, 
                                       AUTOMATE_INDEX_ATTRS["Si"]] \
        if shape_arr_pl == 3 \
        else arr_pl_M_T_K_vars[num_pl_i, t, k,
                                       AUTOMATE_INDEX_ATTRS["Si"]]
    Ci_t_plus_1, Pi_t_plus_1 = None, None
    if t+1 > t_periods:
        Ci_t_plus_1 = arr_pl_M_T_K_vars[num_pl_i, t+1, 
                                        AUTOMATE_INDEX_ATTRS["Ci"]] \
            if shape_arr_pl == 3 \
            else arr_pl_M_T_K_vars[num_pl_i, t+1, k,
                                   AUTOMATE_INDEX_ATTRS["Ci"]]
        Pi_t_plus_1 = arr_pl_M_T_K_vars[num_pl_i, t+1,
                                        AUTOMATE_INDEX_ATTRS["Pi"]] \
            if shape_arr_pl == 3 \
            else arr_pl_M_T_K_vars[num_pl_i, t+1, k,
                                   AUTOMATE_INDEX_ATTRS["Pi"]]
    else:
        Ci_t_plus_1 = arr_pl_M_T_K_vars[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["Ci"]] \
            if shape_arr_pl == 3 \
            else arr_pl_M_T_K_vars[num_pl_i, t, k,
                                   AUTOMATE_INDEX_ATTRS["Ci"]]
        Pi_t_plus_1 = arr_pl_M_T_K_vars[num_pl_i, t,
                                        AUTOMATE_INDEX_ATTRS["Pi"]] \
            if shape_arr_pl == 3 \
            else arr_pl_M_T_K_vars[num_pl_i, t, k,
                                   AUTOMATE_INDEX_ATTRS["Pi"]]
    
    return Pi, Ci, Si, Si_max, Pi_t_plus_1, Ci_t_plus_1

def compute_gamma_state_4_period_t(arr_pl_M_T_K_vars, t, 
                                   pi_0_plus, pi_0_minus,
                                   pi_hp_plus, pi_hp_minus, 
                                   gamma_version=1,
                                   manual_debug=False, dbg=False):
    """        
    compute gamma_i and state for all players 
    
    arr_pl_M_T_K_vars: shape (m_players, t_periods, len(vars)) or 
                             (m_players, t_periods, k_steps, len(vars))
    """
    m_players = arr_pl_M_T_K_vars.shape[0]
    t_periods = arr_pl_M_T_K_vars.shape[1]
    k = 0
    shape_arr_pl = len(arr_pl_M_T_K_vars.shape)
    
    Cis_t_plus_1 = arr_pl_M_T_K_vars[:, t, k, AUTOMATE_INDEX_ATTRS["Ci"]] \
                    if shape_arr_pl == 4 \
                    else arr_pl_M_T_K_vars[:, t, AUTOMATE_INDEX_ATTRS["Ci"]]
    Pis_t_plus_1 = arr_pl_M_T_K_vars[:, t, k, AUTOMATE_INDEX_ATTRS["Pi"]] \
                    if shape_arr_pl == 4 \
                    else arr_pl_M_T_K_vars[:, t, AUTOMATE_INDEX_ATTRS["Pi"]] 
    Cis_Pis_t_plus_1 = Cis_t_plus_1 - Pis_t_plus_1
    Cis_Pis_t_plus_1[Cis_Pis_t_plus_1 < 0] = 0
    GC_t = np.sum(Cis_Pis_t_plus_1)
    
    # initialisation of variables for gamma_version = 2
    state_is = np.empty(shape=(m_players,), dtype=object)
    Sis = np.zeros(shape=(m_players,))
    GSis_t_minus = np.zeros(shape=(m_players,)) 
    GSis_t_plus = np.zeros(shape=(m_players,))
    Xis = np.zeros(shape=(m_players,)) 
    Yis = np.zeros(shape=(m_players,))
    
    for num_pl_i in range(0, m_players):
        Pi, Ci, Si, Si_max, Pi_t_plus_1, Ci_t_plus_1 \
            = get_values_Pi_Ci_Si_Simax_Pi1_Ci1(
                arr_pl_M_T_K_vars, 
                num_pl_i, t, k,
                t_periods,
                shape_arr_pl)
        
        prod_i, cons_i, r_i, gamma_i, state_i = 0, 0, 0, 0, ""
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        state_i = pl_i.find_out_state_i()
        state_is[num_pl_i] = state_i
        Si_t_plus_1, Si_t_minus, Si_t_plus = None, None, None
        Xi, Yi = None, None
        if state_i == STATES[0]:                                               # state1 or Deficit
            Si_t_minus = 0
            Si_t_plus = Si
            Xi = pi_0_minus
            Yi = pi_hp_minus
        elif state_i == STATES[1]:                                             # state2 or Self
            Si_t_minus = Si - (Ci - Pi)
            Si_t_plus = Si
            Xi = pi_0_minus
            Yi = pi_hp_minus
        elif state_i == STATES[2]:                                             # state3 or Surplus
            Si_t_minus = Si
            Si_t_plus = max(Si_max, Si+(Pi-Ci))
            Xi = pi_0_plus
            Yi = pi_hp_plus
        Sis[num_pl_i] = Si
        GSis_t_minus[num_pl_i] = Si_t_minus
        GSis_t_plus[num_pl_i] = Si_t_plus
        Xis[num_pl_i] = Xi; Yis[num_pl_i] = Yi
        
        gamma_i, gamma_i_min, gamma_i_max, gamma_i_mid  = None, None, None, None
        res_mid = None
        if manual_debug:
            gamma_i_min = MANUEL_DBG_GAMMA_I
            gamma_i_mid = MANUEL_DBG_GAMMA_I
            gamma_i_max = MANUEL_DBG_GAMMA_I
            gamma_i = MANUEL_DBG_GAMMA_I
        else:
            Si_t_plus_1 = fct_positive(Ci_t_plus_1, Pi_t_plus_1)
            gamma_i_min = Xi - 1
            gamma_i_max = Yi + 1
            # print("Pi={}, Ci={}, Si={}, Si_t_plus_1={}, Si_t_minus={}, Si_t_plus={}".format(Pi, 
            #         Ci, Si, Si_t_plus_1, Si_t_minus, Si_t_plus))
            if Si_t_plus_1 < Si_t_minus:
                # Xi - 1
                gamma_i = gamma_i_min
            elif Si_t_plus_1 >= Si_t_plus:
                # Yi + 1
                gamma_i = gamma_i_max
            elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                res_mid = ( Si_t_plus_1 - Si_t_minus) / \
                        (Si_t_plus - Si_t_minus)
                Z = Xi + (Yi-Xi)*res_mid
                gamma_i_mid = int(np.floor(Z))
                gamma_i = gamma_i_mid
              
        # print("num_pl_i={},gamma_i={}, gamma_i_min={}, state_i={}, gamma_V{}".format(num_pl_i, 
        #         gamma_i, gamma_i_min, state_i, gamma_version))
        if gamma_version == 0:
            gamma_i = 0
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus, pi_hp_plus, dbg)
        elif gamma_version == 1:
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus, pi_hp_plus, dbg)
            
        elif gamma_version == 3:
            gamma_i = None
            if manual_debug:
                gamma_i = MANUEL_DBG_GAMMA_I
            elif Si_t_plus_1 < Si_t_minus:
                gamma_i = gamma_i_min
            else :
                gamma_i = gamma_i_max
                
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus, pi_hp_plus, dbg)
            
        elif gamma_version == 4:
            gamma_i = None
            if manual_debug:
                gamma_i = MANUEL_DBG_GAMMA_I
            else:
                if Si_t_plus_1 < Si_t_minus:
                    # Xi - 1
                    gamma_i = gamma_i_min
                elif Si_t_plus_1 >= Si_t_plus:
                    # Yi + 1
                    gamma_i = gamma_i_max
                elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                    res_mid = ( Si_t_plus_1 - Si_t_minus) / \
                            (Si_t_plus - Si_t_minus)
                    Z = Xi + (Yi-Xi) * np.sqrt(res_mid)
                    gamma_i_mid = int(np.floor(Z))
                    gamma_i = gamma_i_mid
                
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_pl_M_T_K_vars = update_variables(
                                    arr_pl_M_T_K_vars, variables, shape_arr_pl,
                                    num_pl_i, t, k, gamma_i, Si,
                                    pi_0_minus, pi_0_plus, 
                                    pi_hp_minus, pi_hp_plus, dbg)
        
    if gamma_version == 2:
        GS_t_minus = np.sum(GSis_t_minus)
        GS_t_plus = np.sum(GSis_t_plus)
        gamma_is = None
        if GC_t <= GS_t_minus:
            gamma_is = Xis - 1
        elif GC_t > GS_t_plus:
            gamma_is = Yis + 1
        else:
            frac = (GC_t - GS_t_minus) / (GS_t_plus - GS_t_minus)
            res_is = Xis + (Yis-Xis)*frac
            gamma_is = np.floor(res_is)
            
        # ____              update cell arrays: debut               _______
        variables = [("Si", Sis), ("state_i", state_is), 
                     ("Si_minus", GSis_t_minus), ("Si_plus", GSis_t_plus)]
        if shape_arr_pl == 3:
            for (var,vals) in variables:
                arr_pl_M_T_K_vars[:, t,
                        AUTOMATE_INDEX_ATTRS[var]] = vals
            if manual_debug:
                arr_pl_M_T_K_vars[:, t,
                        AUTOMATE_INDEX_ATTRS["gamma_i"]] = MANUEL_DBG_GAMMA_I
            else:
                arr_pl_M_T_K_vars[:, t, 
                                  AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_is
        elif shape_arr_pl == 4:
            # turn Sis 1D to 2D
            arrs_Sis_2D = []; arrs_state_is_2D = []; arrs_GSis_t_minus_2D = []
            arrs_GSis_t_plus_2D = []; arrs_gamma_is_2D = []; 
            arrs_manuel_dbg_gamma_i_2D = []; 
            manuel_dbg_gamma_is = [MANUEL_DBG_GAMMA_I for _ in range(0, m_players)]
            k_steps = arr_pl_M_T_K_vars.shape[2]
            for k in range(0, k_steps):
                arrs_Sis_2D.append(list(Sis))
                arrs_state_is_2D.append(list(state_is))
                arrs_GSis_t_minus_2D.append(list(GSis_t_minus))
                arrs_GSis_t_plus_2D.append(list(GSis_t_plus))
                arrs_gamma_is_2D.append(list(gamma_is))
                arrs_manuel_dbg_gamma_i_2D.append(manuel_dbg_gamma_is)
            Sis_2D = np.array(arrs_Sis_2D, dtype=object).T
            state_is_2D = np.array(arrs_state_is_2D, dtype=object).T
            GSis_t_minus_2D = np.array(arrs_GSis_t_minus_2D, dtype=object).T
            GSis_t_plus_2D = np.array(arrs_GSis_t_plus_2D, dtype=object).T
            gamma_is_2D = np.array(arrs_gamma_is_2D, dtype=object).T
            manuel_dbg_gamma_i_2D = np.array(arrs_manuel_dbg_gamma_i_2D, 
                                             dtype=object).T
            
            arr_pl_M_T_K_vars[:, t, :, AUTOMATE_INDEX_ATTRS["Si"]] = Sis_2D
            arr_pl_M_T_K_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["state_i"]] = state_is_2D          
            arr_pl_M_T_K_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["Si_minus"]] = GSis_t_minus_2D
            arr_pl_M_T_K_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["Si_plus"]] = GSis_t_plus_2D 
            if manual_debug:
                arr_pl_M_T_K_vars[:, t, :,
                            AUTOMATE_INDEX_ATTRS["gamma_i"]] = manuel_dbg_gamma_i_2D
            else:
                arr_pl_M_T_K_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_is_2D
        
        # ____              update cell arrays: fin               _______
        
        bool_gamma_is = (gamma_is >= min(pi_0_minus, pi_0_plus)-1) \
                            & (gamma_is <= max(pi_hp_minus, pi_hp_plus)+1)
        print("GAMMA : t={}, val={}, bool_gamma_is={}"\
              .format(t, gamma_is, bool_gamma_is)) if dbg else None
        GSis_t_minus_1 = arr_pl_M_T_K_vars[
                            :, t, AUTOMATE_INDEX_ATTRS["Si"]] \
                        if shape_arr_pl == 3 \
                        else arr_pl_M_T_K_vars[
                            :, t, k, AUTOMATE_INDEX_ATTRS["Si"]]
        print("GSis_t_minus_1={}, Sis={}".format(GSis_t_minus_1, Sis)) \
            if dbg else None
        
    return arr_pl_M_T_K_vars
    
# new version of compute gamma and state ---> fin

def compute_gamma_state_4_period_t_OLD(arr_pl_M_T_K_vars, t, 
                                   pi_0_plus, pi_0_minus,
                                   pi_hp_plus, pi_hp_minus, 
                                   gamma_version=1,
                                   manual_debug=False, dbg=False):
    """
    # TODO: ajouter manual_debug parmi les parametres de la fonction puis 
        faire un if manual_dbg : gamma = np.random.randint(low=2, high=21, size=1)[0]
        
    compute gamma_i for all players 
    
    arr_pl_M_T_K_vars: shape (m_players, t_periods, len(vars)) or 
                             (m_players, t_periods, k_steps, len(vars))
    """
    m_players = arr_pl_M_T_K_vars.shape[0]
    t_periods = arr_pl_M_T_K_vars.shape[1]
    
    arr_pl_vars = None
    arr_pl_vars = arr_pl_M_T_K_vars.copy()
    
    if len(arr_pl_M_T_K_vars.shape) == 3:
        if gamma_version == 1:
            for num_pl_i in range(0, m_players):
                Pi = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Pi"]]
                Ci = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Ci"]]
                Si = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si"]] \
                        if t == 0 \
                        else arr_pl_vars[num_pl_i, t-1, AUTOMATE_INDEX_ATTRS["Si"]]
                Si_max = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si_max"]]
                Ci_t_plus_1 = arr_pl_vars[num_pl_i, 
                                           t+1, 
                                           AUTOMATE_INDEX_ATTRS["Ci"]] \
                                if t+1<t_periods \
                                else 0
                Pi_t_plus_1 = arr_pl_vars[num_pl_i, 
                                          t+1, 
                                          AUTOMATE_INDEX_ATTRS["Pi"]] \
                                if t+1 < t_periods \
                                else 0
                prod_i, cons_i, r_i, gamma_i, state_i = 0, 0, 0, 0, ""
                pl_i = None
                pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                      prod_i, cons_i, r_i, state_i)
                state_i = pl_i.find_out_state_i()
                
                Si_t_minus, Si_t_plus = None, None
                X, Y = None, None
                if state_i == STATES[0]:                                            # state1 or Deficit
                    Si_t_minus = 0
                    Si_t_plus = Si
                    X = pi_0_minus
                    Y = pi_hp_minus
                elif state_i == STATES[1]:                                          # state2 or Self
                    Si_t_minus = Si - (Ci - Pi)
                    Si_t_plus = Si
                    X = pi_0_minus
                    Y = pi_hp_minus
                elif state_i == STATES[2]:                                          # state3 or Surplus
                    Si_t_minus = Si
                    Si_t_plus = max(Si_max, Si+(Pi-Ci))
                    X = pi_0_plus
                    Y = pi_hp_plus
                    
                gamma_i = None
                if manual_debug:
                    gamma_i = MANUEL_DBG_GAMMA_I
                else:
                    Si_t_plus_1 = fct_positive(Ci_t_plus_1, Pi_t_plus_1)
                    if Si_t_plus_1 < Si_t_minus:
                        gamma_i = X - 1
                    elif Si_t_plus_1 >= Si_t_plus:
                        gamma_i = Y + 1
                    elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                        res = ( Si_t_plus_1 - Si_t_minus) / \
                                (Si_t_plus - Si_t_minus)
                        Z = X + (Y-X)*res
                        gamma_i = math.floor(Z)
                                    
                arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si"]] = Si
                arr_pl_vars[num_pl_i, t, 
                            AUTOMATE_INDEX_ATTRS["state_i"]] = state_i            
                arr_pl_vars[num_pl_i, t, 
                            AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_i
                arr_pl_vars[num_pl_i, t, 
                            AUTOMATE_INDEX_ATTRS["Si_minus"]] = Si_t_minus
                arr_pl_vars[num_pl_i, t, 
                            AUTOMATE_INDEX_ATTRS["Si_plus"]] = Si_t_plus
                
                bool_gamma_i = (gamma_i >= min(pi_0_minus, pi_0_plus)-1) \
                                & (gamma_i <= max(pi_hp_minus, pi_hp_plus)+1)
                print("GAMMA : t={}, player={}, val={}, bool_gamma_i={}"\
                      .format(t, num_pl_i, gamma_i, bool_gamma_i)) if dbg else None
                Si_t_minus_1 = arr_pl_vars[num_pl_i, t, 
                                           AUTOMATE_INDEX_ATTRS["Si"]] \
                                if t == 0 \
                                else arr_pl_vars[num_pl_i, t-1, 
                                                 AUTOMATE_INDEX_ATTRS["Si"]]
                print("Si_t_minus_1={}, Si={}".format(Si_t_minus_1, Si)) \
                    if dbg else None
                            
        elif gamma_version==2:
            Cis_t_plus_1 = arr_pl_vars[:,t+1, AUTOMATE_INDEX_ATTRS["Ci"]] \
                            if t+1 < t_periods \
                            else np.zeros(shape=(m_players,))
            Pis_t_plus_1 = arr_pl_vars[:,t+1, AUTOMATE_INDEX_ATTRS["Pi"]] \
                            if t+1 < t_periods \
                            else np.zeros(shape=(m_players,))
            Ci_Pis_t_plus_1 = Cis_t_plus_1 - Pis_t_plus_1
            Ci_Pis_t_plus_1[Ci_Pis_t_plus_1 < 0] = 0
            GC_t = np.sum(Ci_Pis_t_plus_1)
            
            state_is = np.empty(shape=(m_players,), dtype=object)
            Sis = np.zeros(shape=(m_players,))
            GSis_t_minus = np.zeros(shape=(m_players,)) 
            GSis_t_plus = np.zeros(shape=(m_players,))
            Xis = np.zeros(shape=(m_players,)) 
            Yis = np.zeros(shape=(m_players,))
            for num_pl_i in range(0, m_players):
                Pi = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Pi"]]
                Ci = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Ci"]]
                Si = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si"]] \
                        if t == 0 \
                        else arr_pl_vars[num_pl_i, t-1, AUTOMATE_INDEX_ATTRS["Si"]]
                Si_max = arr_pl_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si_max"]]
                Sis[num_pl_i] = Si
                prod_i, cons_i, r_i, gamma_i, state_i = 0, 0, 0, 0, ""
                pl_i = None
                pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                      prod_i, cons_i, r_i, state_i)
                state_i = pl_i.find_out_state_i()
                state_is[num_pl_i] = state_i
                
                Si_t_minus, Si_t_plus = None, None
                Xi, Yi = None, None
                if state_i == STATES[0]:                                            # state1 or Deficit
                    Si_t_minus = 0
                    Si_t_plus = Si
                    Xi = pi_0_minus
                    Yi = pi_hp_minus
                elif state_i == STATES[1]:                                          # state2 or Self
                    Si_t_minus = Si - (Ci - Pi)
                    Si_t_plus = Si
                    Xi = pi_0_minus
                    Yi = pi_hp_minus
                elif state_i == STATES[2]:                                          # state3 or Surplus
                    Si_t_minus = Si
                    Si_t_plus = max(Si_max, Si+(Pi-Ci))
                    Xi = pi_0_plus
                    Yi = pi_hp_plus
                GSis_t_minus[num_pl_i] = Si_t_minus
                GSis_t_plus[num_pl_i] = Si_t_plus
                Xis[num_pl_i] = Xi; Yis[num_pl_i] = Yi
            
            GS_t_minus = np.sum(GSis_t_minus)
            GS_t_plus = np.sum(GSis_t_plus)
            gamma_is = None
            if GC_t <= GS_t_minus:
                gamma_is = Xis - 1
            elif GC_t > GS_t_plus:
                gamma_is = Yis + 1
            else:
                frac = (GC_t - GS_t_minus) / (GS_t_plus - GS_t_minus)
                res_is = Xis + (Yis-Xis)*frac
                gamma_is = np.floor(res_is)
                
            arr_pl_vars[:, t, AUTOMATE_INDEX_ATTRS["Si"]] = Sis
            arr_pl_vars[:, t, 
                        AUTOMATE_INDEX_ATTRS["state_i"]] = state_is            
            arr_pl_vars[:, t, 
                        AUTOMATE_INDEX_ATTRS["Si_minus"]] = GSis_t_minus
            arr_pl_vars[:, t, 
                        AUTOMATE_INDEX_ATTRS["Si_plus"]] = GSis_t_plus
            if manual_debug:
                arr_pl_vars[:, t, 
                            AUTOMATE_INDEX_ATTRS["gamma_i"]] = MANUEL_DBG_GAMMA_I
            else:
                arr_pl_vars[:, t, 
                        AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_is
            
            bool_gamma_is = (gamma_is >= min(pi_0_minus, pi_0_plus)-1) \
                            & (gamma_is <= max(pi_hp_minus, pi_hp_plus)+1)
            print("GAMMA : t={}, val={}, bool_gamma_is={}"\
                  .format(t, gamma_is, bool_gamma_is)) if dbg else None
            GSis_t_minus_1 = arr_pl_vars[:, t, AUTOMATE_INDEX_ATTRS["Si"]] \
                            if t == 0 \
                            else arr_pl_vars[:, t-1, AUTOMATE_INDEX_ATTRS["Si"]]
            print("GSis_t_minus_1={}, Sis={}".format(GSis_t_minus_1, Sis)) \
                if dbg else None
                
    elif len(arr_pl_M_T_K_vars.shape) == 4:
        arr_pl_vars = arr_pl_M_T_K_vars.copy()
        k = 0; k_steps = arr_pl_M_T_K_vars.shape[2]
        if gamma_version == 1: 
            for num_pl_i in range(0, m_players):
                Pi = arr_pl_vars[num_pl_i, t, k,
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
                Ci = arr_pl_vars[num_pl_i, t, k,
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
                Si = arr_pl_vars[num_pl_i, t, k,
                                      AUTOMATE_INDEX_ATTRS["Si"]] \
                        if t == 0 \
                        else arr_pl_vars[num_pl_i, t-1, k,
                                      AUTOMATE_INDEX_ATTRS["Si"]]
                Si_max = arr_pl_vars[num_pl_i, t, k,
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
                Ci_t_plus_1 = arr_pl_vars[num_pl_i, 
                                           t+1, k,
                                           AUTOMATE_INDEX_ATTRS["Ci"]] \
                                if t+1<t_periods \
                                else 0
                Pi_t_plus_1 = arr_pl_vars[num_pl_i, 
                                          t+1, k,
                                          AUTOMATE_INDEX_ATTRS["Pi"]] \
                                if t+1 < t_periods \
                                else 0
                                
                prod_i, cons_i, r_i, gamma_i, state_i = 0, 0, 0, 0, ""
                pl_i = None
                pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                      prod_i, cons_i, r_i, state_i)
                state_i = pl_i.find_out_state_i()
                
                Si_t_minus, Si_t_plus = None, None
                X, Y = None, None
                if gamma_version == 1:
                    if state_i == STATES[0]:                                   # state1 or Deficit
                        Si_t_minus = 0
                        Si_t_plus = Si
                        X = pi_0_minus
                        Y = pi_hp_minus
                    elif state_i == STATES[1]:                                 # state2 or Self
                        Si_t_minus = Si - (Ci - Pi)
                        Si_t_plus = Si
                        X = pi_0_minus
                        Y = pi_hp_minus
                    elif state_i == STATES[2]:                                 # state3 or Surplus
                        Si_t_minus = Si
                        Si_t_plus = max(Si_max, Si+(Pi-Ci))
                        X = pi_0_plus
                        Y = pi_hp_plus
                        
                    gamma_i = None
                    if manual_debug:
                        gamma_i = MANUEL_DBG_GAMMA_I
                    else:
                        Si_t_plus_1 = fct_positive(Ci_t_plus_1, Pi_t_plus_1)
                        if Si_t_plus_1 < Si_t_minus:
                            gamma_i = X-1
                        elif Si_t_plus_1 >= Si_t_plus:
                            gamma_i = Y+1
                        elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                            res = ( Si_t_plus_1 - Si_t_minus) / (Si_t_plus - Si_t_minus)
                            Z = X + (Y-X)*res
                            gamma_i = math.floor(Z)  
                        
                    arr_pl_vars[num_pl_i, t, :, 
                                AUTOMATE_INDEX_ATTRS["state_i"]] = state_i
                    arr_pl_vars[num_pl_i, t, :, 
                                AUTOMATE_INDEX_ATTRS["Si"]] = Si
                    arr_pl_vars[num_pl_i, t, :, 
                                AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_i
                    arr_pl_vars[num_pl_i, t, :, 
                                AUTOMATE_INDEX_ATTRS["Si_minus"]] = Si_t_minus
                    arr_pl_vars[num_pl_i, t, :, 
                                AUTOMATE_INDEX_ATTRS["Si_plus"]] = Si_t_plus
                    
                    bool_gamma_i = (gamma_i >= min(pi_0_minus, pi_0_plus)-1) \
                                    & (gamma_i <= max(pi_hp_minus, pi_hp_plus)+1)
                    print("GAMMA : t={}, player={}, val={}, bool_gamma_i={}, {}"\
                          .format(t, num_pl_i, gamma_i, bool_gamma_i, state_i)) \
                        if dbg else None
                    Si_t_minus_1 = arr_pl_vars[num_pl_i, t, k, 
                                               AUTOMATE_INDEX_ATTRS["Si"]] \
                                    if t == 0 \
                                    else arr_pl_vars[num_pl_i, t-1, k,
                                                     AUTOMATE_INDEX_ATTRS["Si"]]
                    print("Si_t_minus_1={}, Si={}".format(Si_t_minus_1, Si)) \
                        if dbg else None
                            
        elif gamma_version == 2:
            Cis_t_plus_1 = arr_pl_vars[:,t+1, k, AUTOMATE_INDEX_ATTRS["Ci"]] \
                            if t+1 < t_periods \
                            else np.zeros(shape=(m_players,))
            Pis_t_plus_1 = arr_pl_vars[:,t+1, k, AUTOMATE_INDEX_ATTRS["Pi"]] \
                            if t+1 < t_periods \
                            else np.zeros(shape=(m_players,))
            Ci_Pis_t_plus_1 = Cis_t_plus_1 - Pis_t_plus_1
            Ci_Pis_t_plus_1[Ci_Pis_t_plus_1 < 0] = 0
            GC_t = np.sum(Ci_Pis_t_plus_1)
            
            state_is = np.empty(shape=(m_players,), dtype=object)
            Sis = np.zeros(shape=(m_players,))
            GSis_t_minus = np.zeros(shape=(m_players,)) 
            GSis_t_plus = np.zeros(shape=(m_players,))
            Xis = np.zeros(shape=(m_players,))
            Yis = np.zeros(shape=(m_players,))
            for num_pl_i in range(0, m_players):
                Pi = arr_pl_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Pi"]]
                Ci = arr_pl_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Ci"]]
                Si = arr_pl_vars[num_pl_i, t, k, AUTOMATE_INDEX_ATTRS["Si"]] \
                        if t == 0 \
                        else arr_pl_vars[num_pl_i, t-1, k, 
                                         AUTOMATE_INDEX_ATTRS["Si"]]
                Si_max = arr_pl_vars[num_pl_i, t, k, 
                                     AUTOMATE_INDEX_ATTRS["Si_max"]]
                Sis[num_pl_i] = Si
                prod_i, cons_i, r_i, gamma_i, state_i = 0, 0, 0, 0, ""
                pl_i = None
                pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                      prod_i, cons_i, r_i, state_i)
                state_i = pl_i.find_out_state_i()
                state_is[num_pl_i] = state_i
                
                Si_t_minus, Si_t_plus = None, None
                Xi, Yi = None, None
                if state_i == STATES[0]:                                            # state1 or Deficit
                    Si_t_minus = 0
                    Si_t_plus = Si
                    Xi = pi_0_minus
                    Yi = pi_hp_minus
                elif state_i == STATES[1]:                                          # state2 or Self
                    Si_t_minus = Si - (Ci - Pi)
                    Si_t_plus = Si
                    Xi = pi_0_minus
                    Yi = pi_hp_minus
                elif state_i == STATES[2]:                                          # state3 or Surplus
                    Si_t_minus = Si
                    Si_t_plus = max(Si_max, Si+(Pi-Ci))
                    Xi = pi_0_plus
                    Yi = pi_hp_plus
                GSis_t_minus[num_pl_i] = Si_t_minus
                GSis_t_plus[num_pl_i] = Si_t_plus
                Xis[num_pl_i] = Xi; Yis[num_pl_i] = Yi
            
            GS_t_minus = np.sum(GSis_t_minus)
            GS_t_plus = np.sum(GSis_t_plus)
            if GC_t <= GS_t_minus:
                gamma_is = Xis - 1
            elif GC_t > GS_t_plus:
                gamma_is = Yis + 1
            else:
                frac = (GC_t - GS_t_minus) / (GS_t_plus - GS_t_minus)
                res_is = Xis + (Yis-Xis)*frac
                gamma_is = np.floor(res_is)
                
            #TODO turn Sis 1D to 2D
            arrs_Sis_2D = []; arrs_state_is_2D = []; arrs_GSis_t_minus_2D = []
            arrs_GSis_t_plus_2D = []; arrs_gamma_is_2D = []; 
            arrs_manuel_dbg_gamma_i_2D = []; 
            manuel_dbg_gamma_is = [MANUEL_DBG_GAMMA_I for _ in range(0, m_players)]
            for k in range(0, k_steps):
                arrs_Sis_2D.append(list(Sis))
                arrs_state_is_2D.append(list(state_is))
                arrs_GSis_t_minus_2D.append(list(GSis_t_minus))
                arrs_GSis_t_plus_2D.append(list(GSis_t_plus))
                arrs_gamma_is_2D.append(list(gamma_is))
                arrs_manuel_dbg_gamma_i_2D.append(manuel_dbg_gamma_is)
            Sis_2D = np.array(arrs_Sis_2D, dtype=object).T
            state_is_2D = np.array(arrs_state_is_2D, dtype=object).T
            GSis_t_minus_2D = np.array(arrs_GSis_t_minus_2D, dtype=object).T
            GSis_t_plus_2D = np.array(arrs_GSis_t_plus_2D, dtype=object).T
            gamma_is_2D = np.array(arrs_gamma_is_2D, dtype=object).T
            manuel_dbg_gamma_i_2D = np.array(arrs_manuel_dbg_gamma_i_2D, 
                                             dtype=object).T
            
            arr_pl_vars[:, t, :, AUTOMATE_INDEX_ATTRS["Si"]] = Sis_2D
            arr_pl_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["state_i"]] = state_is_2D          
            arr_pl_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["Si_minus"]] = GSis_t_minus_2D
            arr_pl_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["Si_plus"]] = GSis_t_plus_2D 
            if manual_debug:
                arr_pl_vars[:, t, :,
                            AUTOMATE_INDEX_ATTRS["gamma_i"]] = manuel_dbg_gamma_i_2D
            else:
                arr_pl_vars[:, t, :,
                        AUTOMATE_INDEX_ATTRS["gamma_i"]] = gamma_is_2D
        
            bool_gamma_is = (gamma_is >= min(pi_0_minus, pi_0_plus)-1) \
                            & (gamma_is <= max(pi_hp_minus, pi_hp_plus)+1)
            print("GAMMA : t={}, val={}, bool_gamma_is={}"\
                  .format(t, gamma_is, bool_gamma_is)) if dbg else None
            GSis_t_minus_1 = arr_pl_vars[:, t, k, AUTOMATE_INDEX_ATTRS["Si"]] \
                            if t == 0 \
                            else arr_pl_vars[:, t-1, k, 
                                             AUTOMATE_INDEX_ATTRS["Si"]]
            print("GSis_t_minus_1={}, Sis={}".format(GSis_t_minus_1, Sis)) \
                if dbg else None

    return arr_pl_vars
      
# ______________        compute prices: debut       ___________________________ 

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
    c0 = pi_0_minus \
        if In_sg >= Out_sg \
        else ((Out_sg - In_sg)*pi_hp_minus + In_sg*pi_0_minus)/Out_sg
    b0 = pi_0_plus \
        if In_sg < Out_sg \
        else (In_sg * pi_0_plus + (- Out_sg + In_sg)*pi_0_plus)/In_sg
   
    return round(b0, N_DECIMALS), round(c0, N_DECIMALS)

def compute_prices_br(arr_pl_M_T_K_vars, t, k,
                   pi_sg_plus_t_minus_1_k, pi_sg_minus_t_minus_1_k,
                   pi_hp_plus, pi_hp_minus, manual_debug, dbg):
    """
    compute the best response prices' and benefits/costs variables: 
        ben_i, cst_i
        pi_sg_plus_t_k, pi_sg_minus_t_k
        pi_0_plus_t_k, pi_0_minus_t_k
    """
    pi_sg_plus_t_k_new, pi_sg_minus_t_k_new \
        = determine_new_pricing_sg(
            arr_pl_M_T_K_vars[:,:,k,:], 
            pi_hp_plus, 
            pi_hp_minus, 
            t, 
            dbg=dbg)
    print("pi_sg_plus_{}_{}_new={}, pi_sg_minus_{}_{}_new={}".format(
        t, k, pi_sg_plus_t_k_new, t, k, pi_sg_minus_t_k_new)) if dbg else None
          
    pi_sg_plus_t_k = pi_sg_plus_t_minus_1_k \
                            if pi_sg_plus_t_k_new is np.nan \
                            else pi_sg_plus_t_k_new
    pi_sg_minus_t_k = pi_sg_minus_t_minus_1_k \
                            if pi_sg_minus_t_k_new is np.nan \
                            else pi_sg_minus_t_k_new
    pi_0_plus_t_k = round(pi_sg_minus_t_minus_1_k*pi_hp_plus/pi_hp_minus, 
                          N_DECIMALS)
    pi_0_minus_t_k = pi_sg_minus_t_minus_1_k
    
    if manual_debug:
        pi_sg_plus_t_k = MANUEL_DBG_PI_SG_PLUS_T_K #8
        pi_sg_minus_t_k = MANUEL_DBG_PI_SG_MINUS_T_K #10
        pi_0_plus_t_k = MANUEL_DBG_PI_0_MINUS_T_K #2 
        pi_0_minus_t_k = MANUEL_DBG_PI_0_MINUS_T_K #3
        
    print("pi_sg_minus_t_minus_1_k={}, pi_0_plus_t_k={}, pi_0_minus_t_k={},"\
          .format(pi_sg_minus_t_minus_1_k, pi_0_plus_t_k, pi_0_minus_t_k)) \
        if dbg else None
    print("pi_sg_plus_t_k={}, pi_sg_minus_t_k={} \n"\
          .format(pi_sg_plus_t_k, pi_sg_minus_t_k)) \
        if dbg else None
    
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = compute_prod_cons_SG(
                        arr_pl_M_T_K_vars[:,:,k,:], t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t_k, c0_t_k = compute_energy_unit_price(
                        pi_0_plus_t_k, pi_0_minus_t_k, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg) 
    
    # compute ben, cst of shapes (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_K_vars[:, t, k, INDEX_ATTRS["gamma_i"]]
    bens_t_k, csts_t_k = compute_utility_players(
                            arr_pl_M_T_K_vars[:,t,:,:], 
                            gamma_is, 
                            k, 
                            b0_t_k, 
                            c0_t_k)
    
    return pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k 

def determine_new_pricing_sg(arr_pl_M_T, pi_hp_plus, pi_hp_minus, t, dbg=False):
    diff_energy_cons_t = 0
    diff_energy_prod_t = 0
    T = t+1
    for k in range(0, T):
        energ_k_prod = \
            fct_positive(
            sum_list1=sum(arr_pl_M_T[:, k, INDEX_ATTRS["prod_i"]]),
            sum_list2=sum(arr_pl_M_T[:, k, INDEX_ATTRS["cons_i"]])
                    )
        energ_k_cons = \
            fct_positive(
            sum_list1=sum(arr_pl_M_T[:, k, INDEX_ATTRS["cons_i"]]),
            sum_list2=sum(arr_pl_M_T[:, k, INDEX_ATTRS["prod_i"]])
                    )
            
        diff_energy_cons_t += energ_k_cons
        diff_energy_prod_t += energ_k_prod
        print("Price t={}, energ_k_prod={}, energ_k_cons={}".format(
            k, energ_k_prod, energ_k_cons)) if dbg else None
        bool_ = arr_pl_M_T[:, k, INDEX_ATTRS["prod_i"]]>0
        unique,counts=np.unique(bool_,return_counts=True)
        sum_prod_k = round(np.sum(arr_pl_M_T[:, k, INDEX_ATTRS["prod_i"]]), 
                           N_DECIMALS)
        sum_cons_k = round(np.sum(arr_pl_M_T[:, k, INDEX_ATTRS["cons_i"]]),
                           N_DECIMALS)
        diff_sum_prod_cons_k = sum_prod_k - sum_cons_k
        print("t={}, k={}, unique:{}, counts={}, sum_prod_k={}, sum_cons_k={}, diff_sum_k={}".format(
                t,k,unique, counts, sum_prod_k, sum_cons_k, diff_sum_prod_cons_k)) \
            if dbg==True else None
    
    sum_cons = sum(sum(arr_pl_M_T[:, :T, INDEX_ATTRS["cons_i"]].astype(np.float64)))
    sum_prod = sum(sum(arr_pl_M_T[:, :T, INDEX_ATTRS["prod_i"]].astype(np.float64)))
    
    print("NAN: cons={}, prod={}".format(
            np.isnan(arr_pl_M_T[:, :T, INDEX_ATTRS["cons_i"]].astype(np.float64)).any(),
            np.isnan(arr_pl_M_T[:, :T, INDEX_ATTRS["prod_i"]].astype(np.float64)).any())
        ) if dbg else None
    arr_cons = np.argwhere(np.isnan(arr_pl_M_T[:, :T, INDEX_ATTRS["cons_i"]].astype(np.float64)))
    arr_prod = np.argwhere(np.isnan(arr_pl_M_T[:, :T, INDEX_ATTRS["prod_i"]].astype(np.float64)))
    
    if arr_cons.size != 0:
        for arr in arr_cons:
            print("{}-->state:{}, Pi={}, Ci={}, Si={}".format(
                arr, arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["state_i"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Pi"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Ci"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Si"]]))
    
    # print("sum_cons={}, sum_prod={}".format(sum_cons, sum_prod))
    # print("diff_energy_cons_t={}, diff_energy_prod_t={}".format(diff_energy_cons_t, diff_energy_prod_t))
    new_pi_sg_minus_t = round(pi_hp_minus*diff_energy_cons_t / sum_cons, N_DECIMALS)  \
                    if sum_cons != 0 else np.nan
    new_pi_sg_plus_t = round(pi_hp_plus*diff_energy_prod_t / sum_prod, N_DECIMALS) \
                        if sum_prod != 0 else np.nan
    # print("new_pi_sg_minus_t={}, new_pi_sg_plus_t={}".format(new_pi_sg_minus_t, new_pi_sg_plus_t))                    
                            
    return new_pi_sg_plus_t, new_pi_sg_minus_t

def compute_prices_inside_SG(arr_pl_M_T_vars_modif, t,
                                pi_hp_plus, pi_hp_minus,
                                pi_0_plus_t, pi_0_minus_t,
                                manual_debug, dbg):
    
    # compute the new prices pi_sg_plus_t, pi_sg_minus_t
    # from a pricing model in the document
    pi_sg_plus_t, pi_sg_minus_t = determine_new_pricing_sg(
                                            arr_pl_M_T_vars_modif.copy(), 
                                            pi_hp_plus, 
                                            pi_hp_minus, 
                                            t, dbg=dbg)
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = compute_prod_cons_SG(arr_pl_M_T_vars_modif.copy(), t)
    # print("In_sg={}, Out_sg={}".format(In_sg, Out_sg ))
    
    # compute prices of an energy unit price for cost and benefit players
    b0_t, c0_t = compute_energy_unit_price(
                    pi_0_plus_t, pi_0_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg)
    
    # compute ben, cst of shape (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_vars_modif[:, t, AUTOMATE_INDEX_ATTRS["gamma_i"]]
    bens_t, csts_t = compute_utility_players(arr_pl_M_T_vars_modif, 
                                              gamma_is, 
                                              t, 
                                              b0_t, 
                                              c0_t)
    
    return b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t
            
            
# ______________        compute prices: fin         ___________________________


# ______________        save variables: debut       ___________________________ 
def save_variables(path_to_save, arr_pl_M_T_K_vars, 
                   b0_s_T_K, c0_s_T_K,
                   B_is_M, C_is_M, 
                   BENs_M_T_K, CSTs_M_T_K, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T_K, pi_sg_plus_T_K, 
                   pi_0_minus_T_K, pi_0_plus_T_K,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res,
                   algo="LRI",
                   dico_best_steps=dict()):
    
    if algo is None:
        path_to_save = path_to_save \
                        if path_to_save != "tests" \
                        else os.path.join(
                                    path_to_save, 
                                    "simu_"+datetime.now()\
                                        .strftime("%d%m_%H%M"))
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
    else:
        path_to_save = path_to_save \
                        if path_to_save != "tests" \
                        else os.path.join(
                                    path_to_save, 
                                    algo+"_simu_"+datetime.now()\
                                        .strftime("%d%m_%H%M"))
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
    
        
    np.save(os.path.join(path_to_save, "arr_pl_M_T_K_vars.npy"), 
            arr_pl_M_T_K_vars)
    np.save(os.path.join(path_to_save, "b0_s_T_K.npy"), b0_s_T_K)
    np.save(os.path.join(path_to_save, "c0_s_T_K.npy"), c0_s_T_K)
    np.save(os.path.join(path_to_save, "B_is_M.npy"), B_is_M)
    np.save(os.path.join(path_to_save, "C_is_M.npy"), C_is_M)
    np.save(os.path.join(path_to_save, "BENs_M_T_K.npy"), BENs_M_T_K)
    np.save(os.path.join(path_to_save, "CSTs_M_T_K.npy"), CSTs_M_T_K)
    np.save(os.path.join(path_to_save, "BB_is_M.npy"), BB_is_M)
    np.save(os.path.join(path_to_save, "CC_is_M.npy"), CC_is_M)
    np.save(os.path.join(path_to_save, "RU_is_M.npy"), RU_is_M)
    np.save(os.path.join(path_to_save, "pi_sg_minus_T_K.npy"), pi_sg_minus_T_K)
    np.save(os.path.join(path_to_save, "pi_sg_plus_T_K.npy"), pi_sg_plus_T_K)
    np.save(os.path.join(path_to_save, "pi_0_minus_T_K.npy"), pi_0_minus_T_K)
    np.save(os.path.join(path_to_save, "pi_0_plus_T_K.npy"), pi_0_plus_T_K)
    np.save(os.path.join(path_to_save, "pi_hp_plus_s.npy"), pi_hp_plus_s)
    np.save(os.path.join(path_to_save, "pi_hp_minus_s.npy"), pi_hp_minus_s)
    pd.DataFrame.from_dict(dico_stats_res)\
        .to_csv(os.path.join(path_to_save, "stats_res.csv"))
    pd.DataFrame.from_dict(dico_best_steps)\
        .to_csv(os.path.join(path_to_save,
                              "best_learning_steps.csv"), 
                  index=True)
    pd.DataFrame.from_dict(dico_best_steps, orient="columns")\
        .to_excel(os.path.join(path_to_save,
                               "best_learning_steps.xlsx"), 
                  index=True)
        
    
    print("$$$$ saved variables. $$$$")
    
def save_instances_games(arr_pl_M_T, name_file_arr_pl, path_to_save):
    """
    Store players instances of the game so that:
        the players' numbers, the periods' numbers, the scenario and the prob_Ci
        define a game.
        
    Parameters:
        ----------
    arr_pl_M_T : array of shape (M_PLAYERS,NUM_PERIODS,len(INDEX_ATTRS))
        DESCRIPTION.
    name_file_arr_pl : string 
        DESCRIPTION.
        Name of file saving arr_pl_M_T
    path_to_save : string
        DESCRIPTION.
        path to save instances
        
    """
    
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(path_to_save, name_file_arr_pl), 
            arr_pl_M_T)
    
# ______________        save variables: debut       ___________________________

# __________        resume game on excel file :   debut          ______________
def resume_game_on_excel_file(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                              t, m_players, t_periods, k_steps, 
                              scenario="scenario1", learning_rate=0.1, 
                              prob_Ci = 0.5):
    """
    resume_game_on_excel_file(df_arr_M_T_Ks, 
                              t = t, 
                              m_players=m_players, 
                              t_periods=t_periods, 
                              k_steps=k_steps,
                              scenario=scenario, 
                              learning_rate=learning_rate, 
                              prob_Ci=prob_Ci)
    """
   
    print('shape: df_arr_M_T_Ks = {} '.format(df_arr_M_T_Ks.shape))
    df_arr_M_T_Ks["pl_i"] = df_arr_M_T_Ks['pl_i'].astype(float);
    df_ben_cst_M_T_K["pl_i"] = df_ben_cst_M_T_K['pl_i'].astype(float);
    

    learning_algos = ["LRI1","LRI2"]
    algo_names = learning_algos\
                    +[ALGO_NAMES_BF[0]] \
                    + [ALGO_NAMES_NASH[0]] 
    
    
    # initial array from INSTANCES_GAMES
    path_to_variable = os.path.join(
                        "tests", "INSTANCES_GAMES", scenario, str(prob_Ci)
                        )
    arr_name = "arr_pl_M_T_players_{}_periods_{}.npy".format(m_players, t_periods)
    arr_pl_M_T_vars = np.load(os.path.join(path_to_variable, arr_name),
                                            allow_pickle=True)
    INDEX_ATTRS_INIT = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, "gamma_i":4, 
               "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9,
               "Profili":10, "Casei":11, "R_i_old":12, "Si_old":13, 
               "balanced_pl_i": 14, "formule":15}
    arr_cols = list(INDEX_ATTRS_INIT.keys())
    df_arr_M_T_vars = pd.DataFrame(arr_pl_M_T_vars[:,t,:],
                                   columns=arr_cols)
    df_arr_M_T_vars = df_arr_M_T_vars.reset_index()
    df_arr_M_T_vars.rename(columns={"index":"pl_i"}, inplace=True)
    
    print("shape: arr_pl_M_T_K={}, df_arr_M_T_vars={}".format(
            arr_pl_M_T_vars.shape, df_arr_M_T_vars.shape))
    
    cols = ["state_i","algo","pl_i","k","Ci","Pi","Si","Si_max","gamma_i",
            "prod_i","cons_i","Profili","Casei","bg_i","u_i","prob_mode_state_i",
            "mode_i"] #,"","","","","",]
    
    df = pd.DataFrame(columns=cols)
    
    k_step_chosen = k_steps
    states = list(df_arr_M_T_Ks['state_i'].unique())
    for state_i in states:
        for algo_name in algo_names:
            mask_algo_name = (df_arr_M_T_Ks.state_i == state_i) \
                                & (df_arr_M_T_Ks.algo == algo_name) \
                                & (df_arr_M_T_Ks.k == k_step_chosen-1) \
                                & (df_arr_M_T_Ks.scenario == scenario)
            df_al = df_arr_M_T_Ks[mask_algo_name].copy()
            df_al = df_al[cols]
            
            #mask arr
            mask_arr = (df_arr_M_T_vars.state_i == state_i)
            df_arr = df_arr_M_T_vars[mask_arr].copy()
            df_arr = df_arr[["pl_i","Si"]]
            df_arr.rename(columns={"Si":"initial_Si"}, inplace=True)
                        
            mask_ben_cst = (df_ben_cst_M_T_K.state_i == state_i) \
                                & (df_ben_cst_M_T_K.algo == algo_name) \
                                & (df_ben_cst_M_T_K.k == k_step_chosen-1) \
                                & (df_ben_cst_M_T_K.scenario == scenario)
            df_ben = df_ben_cst_M_T_K[mask_ben_cst]
            df_ben = df_ben[["pl_i","ben","cst"]]
            
            
            df_al = pd.merge(left=df_al, right=df_ben, 
                          left_on='pl_i', right_on='pl_i')
            df_al = pd.merge(left=df_al, right=df_arr, 
                          left_on='pl_i', right_on='pl_i')
            
            
            df = pd.concat([df, df_al], axis=0)
            
    # save to the excel file
    df.rename(columns={"mode_i":"strategie","prob_mode_state_i":'p_i_j_k'}, 
              inplace=True)
    cols = df.columns.tolist()
    cols.insert(6, cols.pop(19))
    df = df[cols]
    path_to_save = os.path.join(*["files_debug"])
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df.to_excel(os.path.join(
                *[path_to_save,
                  "resume_game_rate_{}.xlsx".format(learning_rate)]), 
                index=False )
    
    return arr_pl_M_T_vars, df_arr_M_T_vars, df

def resume_game_on_excel_file_automate_OLD(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                              df_b0_c0_pisg_pi0_T_K, t=1,
                              set1_m_players=10, set1_stateId0_m_players=0.75, 
                              set2_m_players=6, set2_stateId0_m_players=0.42, 
                              t_periods=2, k_steps=250, learning_rate=0.1, 
                              price="0.0002_0.33"):
    """
    resume_game_on_excel_file_automate(df_arr_M_T_Ks, 
                              t = t, 
                              set1_m_players=10, set1_stateId0_m_players=0.75, 
                              set2_m_players=6, set2_stateId0_m_players=0.42, 
                              t_periods=2, k_steps=250, learning_rate=0.1)
    """
   
    print('shape: df_arr_M_T_Ks = {} '.format(df_arr_M_T_Ks.shape))
    df_arr_M_T_Ks["pl_i"] = df_arr_M_T_Ks['pl_i'].astype(float);
    df_ben_cst_M_T_K["pl_i"] = df_ben_cst_M_T_K['pl_i'].astype(float);
    df_arr_M_T_Ks["rate"] = df_arr_M_T_Ks['rate'].astype(float);
    df_b0_c0_pisg_pi0_T_K["rate"] = df_b0_c0_pisg_pi0_T_K['rate'].astype(float);
    
    

    learning_algos = ["LRI1","LRI2"]
    algo_names = learning_algos \
                    + [ALGO_NAMES_BF[0]] \
                    + ["DETERMINIST"] \
                    + [ALGO_NAMES_BF[1]] \
                    + [ALGO_NAMES_NASH[0]] 
    
    
    # initial array from INSTANCES_GAMES
    path_to_variable = os.path.join(
                        "tests", "AUTOMATE_INSTANCES_GAMES"
                        )
    arr_name = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        set1_m_players, set1_stateId0_m_players, 
                        set2_m_players, set2_stateId0_m_players, 
                        t_periods)
    arr_pl_M_T_vars = np.load(os.path.join(path_to_variable, arr_name),
                                            allow_pickle=True)
    arr_cols = list(AUTOMATE_INDEX_ATTRS.keys())
    df_arr_M_T_vars = pd.DataFrame(arr_pl_M_T_vars[:,t,:],
                                   columns=arr_cols)
    df_arr_M_T_vars = df_arr_M_T_vars.reset_index()
    df_arr_M_T_vars.rename(columns={"index":"pl_i"}, inplace=True)
    
    print("shape: arr_pl_M_T_K={}, df_arr_M_T_vars={}".format(
            arr_pl_M_T_vars.shape, df_arr_M_T_vars.shape))
    
    cols = ["state_i","algo","pl_i","k","Ci","Pi","Si","Si_max","gamma_i",
            "prod_i","cons_i","Profili","Casei","bg_i","u_i","S1_p_i_j_k",
            "S2_p_i_j_k", "mode_i", "r_i", "set"] #,"","","","","",]
    cols_b0_c0 = ["b0","c0", 'pi_0_minus','pi_0_plus','pi_sg_minus','pi_sg_plus']
    
    df = pd.DataFrame(columns=cols)
    
    k_step_chosen = k_steps
    states = list(df_arr_M_T_Ks['state_i'].unique())
    for state_i in states:
        for algo_name in algo_names:
            mask_algo_name = (df_arr_M_T_Ks.state_i == state_i) \
                                & (df_arr_M_T_Ks.algo == algo_name) \
                                & (df_arr_M_T_Ks.k == k_step_chosen-1) \
                                & ((df_arr_M_T_Ks.rate == learning_rate) | 
                                   (df_arr_M_T_Ks.rate == 0))  \
                                & (df_arr_M_T_Ks.prices == price)
            mask_b0_c0 = (df_b0_c0_pisg_pi0_T_K.t == t) \
                            & (df_b0_c0_pisg_pi0_T_K.algo == algo_name) \
                            & (df_b0_c0_pisg_pi0_T_K.k == k_step_chosen-1) \
                            & ((df_b0_c0_pisg_pi0_T_K.rate == learning_rate) | 
                               (df_b0_c0_pisg_pi0_T_K.rate == 0))  \
                            & (df_b0_c0_pisg_pi0_T_K.prices == price)
                            
            df_al = df_arr_M_T_Ks[mask_algo_name].copy()
            df_al = df_al[cols]
            print("shape: {}, df_al={}".format(state_i, df_al.shape))
            df_al_b0_c0 = df_b0_c0_pisg_pi0_T_K[mask_b0_c0].copy()
            df_al_b0_c0 = df_al_b0_c0[cols_b0_c0].reset_index()
            df_al_b0_c0.drop('index', axis=1, inplace=True)
            # df_al_b0_c0 = df_al_b0_c0.loc[df_al_b0_c0.index.repeat(df_al.shape[0])]
            # df_al_b0_c0.reset_index().drop('index', axis=1, inplace=True)
            
            
            #mask arr
            mask_arr = (df_arr_M_T_vars.state_i == state_i)
            df_arr = df_arr_M_T_vars[mask_arr].copy()
            df_arr = df_arr[["pl_i","Si"]]
            df_arr.rename(columns={"Si":"initial_Si"}, inplace=True)
                        
            mask_ben_cst = (df_ben_cst_M_T_K.state_i == state_i) \
                                & (df_ben_cst_M_T_K.algo == algo_name) \
                                & (df_ben_cst_M_T_K.k == k_step_chosen-1) \
                                & ((df_ben_cst_M_T_K.rate == learning_rate) | 
                                   (df_ben_cst_M_T_K.rate == 0))  \
                                & (df_ben_cst_M_T_K.prices == price) 
            df_ben = df_ben_cst_M_T_K[mask_ben_cst]
            df_ben = df_ben[["pl_i","ben","cst"]]
            print("shape: df_ben={}".format(df_ben.shape))
            
            
            df_al = pd.merge(left=df_al, right=df_ben, 
                          left_on='pl_i', right_on='pl_i').copy()
            df_al = pd.merge(left=df_al, right=df_arr, 
                          left_on='pl_i', right_on='pl_i').copy()
            df_al_b0_c0 = df_al_b0_c0.loc[df_al_b0_c0.index.repeat(df_al.shape[0])]
            df_al_b0_c0.reset_index().drop('index', axis=1, inplace=True)
            print("shape: {} df_al={}, df_al_b0_c0={}".format(algo_name, 
                  df_al.shape, df_al_b0_c0.shape))
            print("rows: {} df_al={}, df_al_b0_c0={}".format(algo_name, 
                  len(df_al.index), len(df_al_b0_c0.index)))
            df_al = pd.concat([df_al, df_al_b0_c0], axis=0)
            
            
            df = pd.concat([df, df_al], axis=0)
            
            
            
    # save to the excel file
    df.rename(columns={"mode_i":"strategie"}, inplace=True)
    cols = df.columns.tolist()
    cols.insert(6, cols.pop(21))
    df = df[cols]
    path_to_save = os.path.join(*["files_debug"])
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df.to_excel(os.path.join(
                *[path_to_save,
                  "resume_game_rate_{}.xlsx".format(learning_rate)]), 
                index=False )
    
    return arr_pl_M_T_vars, df_arr_M_T_vars, df

def resume_game_on_excel_file_automate(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                              df_b0_c0_pisg_pi0_T_K, t=1,
                              setA_m_players=10, setB_m_players=6, 
                              setC_m_players=5, 
                              t_periods=2, k_steps=250, learning_rate=0.1, 
                              price="0.0002_0.33"):
   
    print('shape: df_arr_M_T_Ks = {} '.format(df_arr_M_T_Ks.shape))
    df_arr_M_T_Ks["pl_i"] = df_arr_M_T_Ks['pl_i'].astype(float);
    df_ben_cst_M_T_K["pl_i"] = df_ben_cst_M_T_K['pl_i'].astype(float);
    df_arr_M_T_Ks["rate"] = df_arr_M_T_Ks['rate'].astype(float);
    df_b0_c0_pisg_pi0_T_K["rate"] = df_b0_c0_pisg_pi0_T_K['rate'].astype(float);
    
    

    learning_algos = ["LRI1","LRI2"]
    algo_names = learning_algos \
                    + [ALGO_NAMES_BF[0]] \
                    + ["DETERMINIST"] \
                    + [ALGO_NAMES_BF[1]] \
                    + [ALGO_NAMES_NASH[0]] 
    
    
    # initial array from INSTANCES_GAMES
    path_to_variable = os.path.join(
                        "tests", "AUTOMATE_INSTANCES_GAMES"
                        )
    arr_name = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods)
    arr_pl_M_T_vars = np.load(os.path.join(path_to_variable, arr_name),
                                            allow_pickle=True)
    arr_cols = list(AUTOMATE_INDEX_ATTRS.keys())
    df_arr_M_T_vars = pd.DataFrame(arr_pl_M_T_vars[:,t,:],
                                   columns=arr_cols)
    df_arr_M_T_vars = df_arr_M_T_vars.reset_index()
    df_arr_M_T_vars.rename(columns={"index":"pl_i"}, inplace=True)
    
    print("shape: arr_pl_M_T_K={}, df_arr_M_T_vars={}".format(
            arr_pl_M_T_vars.shape, df_arr_M_T_vars.shape))
    
    cols = ["state_i","algo","pl_i","k","Ci","Pi","Si","Si_max","gamma_i",
            "prod_i","cons_i","Profili","Casei","bg_i","u_i","S1_p_i_j_k",
            "S2_p_i_j_k", "mode_i", "r_i", "set"] #,"","","","","",]
    cols_b0_c0 = ["b0","c0", 'pi_0_minus','pi_0_plus','pi_sg_minus','pi_sg_plus']
    
    df = pd.DataFrame(columns=cols)
    
    k_step_chosen = k_steps
    states = list(df_arr_M_T_Ks['state_i'].unique())
    for state_i in states:
        for algo_name in algo_names:
            mask_algo_name = (df_arr_M_T_Ks.state_i == state_i) \
                                & (df_arr_M_T_Ks.algo == algo_name) \
                                & (df_arr_M_T_Ks.k == k_step_chosen-1) \
                                & ((df_arr_M_T_Ks.rate == learning_rate) | 
                                   (df_arr_M_T_Ks.rate == 0))  \
                                & (df_arr_M_T_Ks.prices == price)
            mask_b0_c0 = (df_b0_c0_pisg_pi0_T_K.t == t) \
                            & (df_b0_c0_pisg_pi0_T_K.algo == algo_name) \
                            & (df_b0_c0_pisg_pi0_T_K.k == k_step_chosen-1) \
                            & ((df_b0_c0_pisg_pi0_T_K.rate == learning_rate) | 
                               (df_b0_c0_pisg_pi0_T_K.rate == 0))  \
                            & (df_b0_c0_pisg_pi0_T_K.prices == price)
                            
            df_al = df_arr_M_T_Ks[mask_algo_name].copy()
            df_al = df_al[cols]
            print("shape: {}, df_al={}".format(state_i, df_al.shape))
            df_al_b0_c0 = df_b0_c0_pisg_pi0_T_K[mask_b0_c0].copy()
            df_al_b0_c0 = df_al_b0_c0[cols_b0_c0].reset_index()
            df_al_b0_c0.drop('index', axis=1, inplace=True)
            # df_al_b0_c0 = df_al_b0_c0.loc[df_al_b0_c0.index.repeat(df_al.shape[0])]
            # df_al_b0_c0.reset_index().drop('index', axis=1, inplace=True)
            
            
            #mask arr
            mask_arr = (df_arr_M_T_vars.state_i == state_i)
            df_arr = df_arr_M_T_vars[mask_arr].copy()
            df_arr = df_arr[["pl_i","Si"]]
            df_arr.rename(columns={"Si":"initial_Si"}, inplace=True)
                        
            mask_ben_cst = (df_ben_cst_M_T_K.state_i == state_i) \
                                & (df_ben_cst_M_T_K.algo == algo_name) \
                                & (df_ben_cst_M_T_K.k == k_step_chosen-1) \
                                & ((df_ben_cst_M_T_K.rate == learning_rate) | 
                                   (df_ben_cst_M_T_K.rate == 0))  \
                                & (df_ben_cst_M_T_K.prices == price) 
            df_ben = df_ben_cst_M_T_K[mask_ben_cst]
            df_ben = df_ben[["pl_i","ben","cst"]]
            print("shape: df_ben={}".format(df_ben.shape))
            
            
            df_al = pd.merge(left=df_al, right=df_ben, 
                          left_on='pl_i', right_on='pl_i').copy()
            df_al = pd.merge(left=df_al, right=df_arr, 
                          left_on='pl_i', right_on='pl_i').copy()
            df_al_b0_c0 = df_al_b0_c0.loc[df_al_b0_c0.index.repeat(df_al.shape[0])]
            df_al_b0_c0.reset_index().drop('index', axis=1, inplace=True)
            print("shape: {} df_al={}, df_al_b0_c0={}".format(algo_name, 
                  df_al.shape, df_al_b0_c0.shape))
            print("rows: {} df_al={}, df_al_b0_c0={}".format(algo_name, 
                  len(df_al.index), len(df_al_b0_c0.index)))
            df_al = pd.concat([df_al, df_al_b0_c0], axis=0)
            
            
            df = pd.concat([df, df_al], axis=0)
            
            
            
    # save to the excel file
    df.rename(columns={"mode_i":"strategie"}, inplace=True)
    cols = df.columns.tolist()
    cols.insert(6, cols.pop(21))
    df = df[cols]
    path_to_save = os.path.join(*["files_debug"])
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df.to_excel(os.path.join(
                *[path_to_save,
                  "resume_game_rate_{}.xlsx".format(learning_rate)]), 
                index=False )
    
    return arr_pl_M_T_vars, df_arr_M_T_vars, df
# __________        resume game on excel file :   fin           ______________

# __________    generate Cis, Pis, Si_maxs and Sis --> debut   ________________
def generate_Pi_Ci_Si_Simax_by_profil_scenario(
                                    m_players=3, num_periods=5, 
                                    scenario="scenario1", prob_Ci=0.3, 
                                    Ci_low=Ci_LOW, Ci_high=Ci_HIGH):
    """
    create the initial values of all players at all time intervals

    Parameters
    ----------
    m_players : Integer optional
        DESCRIPTION. The default is 3.
        the number of players
    num_periods : Integer, optional
        DESCRIPTION. The default is 5.
        the number of time intervals 
    scenario : String, optional
        DESCRIPTION. The default is "scenario1".
        indicate the scenario to play
    prob_Ci : float, optional
        DESCRIPTION. The default is 0.3.
        the probability of choosing the type of players' consumption
    Ci_low : float, optional
        DESCRIPTION. The default is 10.
        the min value of the consumption
    Ci_high : float, optional
        DESCRIPTION. The default is 60.
        the max value of the consumption
    Returns
    -------
    arr_pl_M_T : array of shape (M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))
        DESCRIPTION.

    """
    arr_pl_M_T = []
    for num_pl, prob in enumerate(np.random.uniform(0, 1, size=m_players)):
        Ci = None; profili = None
        if prob <= prob_Ci:
            Ci = Ci_low
            Si_max = 0.8 * Ci
            if scenario == "scenario1":
                profili = PROFIL_L
            elif scenario == "scenario2":
                profili = PROFIL_H
            elif scenario == "scenario3":
                profili = PROFIL_M
        else:
            Ci = Ci_high
            Si_max = 0.5 * Ci
            if scenario == "scenario1":
                profili = PROFIL_H
            elif scenario == "scenario2":
                profili = PROFIL_M
            elif scenario == "scenario3":
                profili = np.random.default_rng().choice(
                            p=[0.5, 0.5],
                            a=[PROFIL_H, PROFIL_M])
            
        profil_casei = None
        prob_casei = np.random.uniform(0,1)
        if prob_casei < profili[0]:
            profil_casei = CASE1
        elif prob_casei >= profili[0] \
            and prob_casei < profili[0]+profili[1]:
            profil_casei = CASE2
        else:
            profil_casei = CASE3
        # profil_casei = None
        # if prob < profili[0]:
        #     profil_casei = CASE1
        # elif prob >= profili[0] \
        #     and prob < profili[0]+profili[1]:
        #     profil_casei = CASE2
        # else:
        #     profil_casei = CASE3
                
        min_val_profil = profil_casei[0]*Ci 
        max_val_profil = profil_casei[1]*Ci
        
        Pi_s = list( np.around(np.random.uniform(
                                low=min_val_profil, 
                                high=max_val_profil, 
                                size=(num_periods,)
                                ), decimals=N_DECIMALS) )
        Si_s = list( np.around(np.random.uniform(0,1,size=(num_periods,))*Si_max,
                               decimals=N_DECIMALS))
        Si_s[0] = 0; 
        
        str_profili_s = ["_".join(map(str, profili))] * num_periods
        str_casei_s = ["_".join(map(str, profil_casei))] * num_periods
        
        # building list of list 
        Ci_s = [Ci] * num_periods
        Si_max_s = [Si_max] * num_periods
        gamma_i_s, r_i_s = [0]*num_periods, [0]*num_periods
        prod_i_s, cons_i_s = [0]*num_periods, [0]*num_periods
        mode_i_s = [""]*num_periods
        R_i_old_s = [round(x - y, 2) for x, y in zip(Si_max_s, Si_s)]
        Si_old_s = [0]*num_periods
        balanced_pl_i_s = [False]*num_periods
        formules = [""]*num_periods
        
        state_i_s = []
        for t, tu in enumerate(zip(Ci_s, Pi_s, Si_s, Si_max_s)):
            gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
            tu_Pi, tu_Ci, tu_Si, tu_Si_max = tu[1], tu[0], tu[2], tu[3]
            pl_i = players.Player(tu_Pi, tu_Ci,tu_Si, tu_Si_max, gamma_i, 
                            prod_i, cons_i, r_i, state_i)
            state_i = pl_i.find_out_state_i()
            if t == 0 and state_i == STATES[0]:
                if tu_Ci-tu_Pi-1 == 0:
                    tu_Pi -= 1
                high = min(tu_Si_max, tu_Ci-tu_Pi-1)
                tu_Si = np.random.uniform(low=1, high=high)
            elif t==0 and state_i == STATES[1]:
                low = min(tu_Ci-tu_Pi, tu_Si_max)
                high = min(2*(tu_Ci-tu_Pi), tu_Si_max)
                tu_Si = np.random.uniform(low=low, high=high)
            elif t==0 and state_i == STATES[2]:
                low = 1
                high = tu_Si_max
                tu_Si = np.random.uniform(low=low, high=high)
            Si_s[t] = tu_Si
            state_i_s.append(state_i)
        
        init_values_i_s = list(zip(Ci_s, Pi_s, Si_s, Si_max_s, gamma_i_s, 
                                   prod_i_s, cons_i_s, r_i_s, state_i_s, 
                                   mode_i_s, str_profili_s, str_casei_s, 
                                   R_i_old_s, Si_old_s, balanced_pl_i_s, 
                                   formules))
        arr_pl_M_T.append(init_values_i_s)
    
    arr_pl_M_T = np.array(arr_pl_M_T, dtype=object)
    
    return arr_pl_M_T

def generer_Pi_Ci_Si_Simax_for_all_scenarios(scenarios=["scenario1"], 
                                    m_players=3, num_periods=5, 
                                    prob_Ci=0.3, Ci_low=Ci_LOW, Ci_high=Ci_HIGH):
    """
    create the variables for all scenarios

    Parameters
    ----------
    scenarios : list of String, optional
        DESCRIPTION. The default is [].
    m_players : Integer optional
        DESCRIPTION. The default is 3.
        the number of players
    num_periods : Integer, optional
        DESCRIPTION. The default is 5.
        the number of time intervals 
    prob_Ci : float, optional
        DESCRIPTION. The default is 0.3.
        the probability of choosing the type of players' consumption
    Ci_low : float, optional
        DESCRIPTION. The default is 10.
        the min value of the consumption
    Ci_high : float, optional
        DESCRIPTION. The default is 60.
        the max value of the consumption

    Returns
    -------
    list of arrays. 
    Each array has a shape ((M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))).

    """
    l_arr_pl_M_T = []
    for scenario in scenarios:
        arr_pl_M_T = generate_Pi_Ci_Si_Simax_by_profil_scenario(
                            m_players=m_players, num_periods=num_periods, 
                            scenario=scenario, prob_Ci=prob_Ci, 
                            Ci_low=Ci_low, Ci_high=Ci_high)
        l_arr_pl_M_T.append(arr_pl_M_T)
        
    return l_arr_pl_M_T

# __________    generate Cis, Pis, Si_maxs and Sis --> fin   ________________

###############################################################################
#            generate Pi, Ci, Si, state by automate --> debut
###############################################################################

def generate_Pi_Ci_Si_Simax_by_automate(set1_m_players, set2_m_players, 
                                        t_periods, 
                                        set1_states=None, 
                                        set2_states=None,
                                        set1_stateId0_m_players=15,
                                        set2_stateId0_m_players=5):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider set1 = {state1, state2} and set2={state2, state3}
        set1_stateId0 = state1, set1_stateId1 = state2
        set2_stateId0 = state2, set2_stateId1 = state3
    Returns
    -------
    None.

    """
    set1_states = [STATES[0], STATES[1]] \
                    if set1_states == None else set1_states
    set2_states = [STATES[1], STATES[2]] \
                    if set2_states == None else set2_states
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = set1_m_players + set2_m_players
    list_players = range(0, m_players)
    
    set1_players = list(np.random.choice(list(list_players), 
                                    size=set1_m_players, 
                                    replace=False))
    if type(set1_stateId0_m_players) is int:
        set1_stateId0_m_players = set1_stateId0_m_players
    else:
        set1_stateId0_m_players = int(np.rint(set1_m_players
                                              *set1_stateId0_m_players))
    set1_stateId0_players = list(np.random.choice(set1_players, 
                                        size=set1_stateId0_m_players, 
                                        replace=False))
    set1_stateId1_players = list(set(set1_players) \
                                 - set(set1_stateId0_players))
    
    set2_players = list(set(list_players) - set(set1_players))
    if type(set2_stateId0_m_players) is int:
        set2_stateId0_m_players = set2_stateId0_m_players
    else:
        set2_stateId0_m_players = int(np.rint(set2_m_players
                                              *set2_stateId0_m_players))
    set2_stateId0_players = list(np.random.choice(set2_players, 
                                    size=set2_stateId0_m_players, 
                                    replace=False))
    set2_stateId1_players = list(set(set2_players) \
                                 - set(set2_stateId0_players))
    if len(set(set1_players).intersection(set2_players)) == 0:
        print("set1 != set2 --> OK")
    else:
        print("set1 != set2 --> NOK")
    if len(set(set1_stateId0_players)\
           .intersection(set(set1_stateId1_players))) == 0:
        print("set1: stateId0={}:{}, stateId1={}:{}--> OK".format(
            set1_states[0], len(set1_stateId0_players), 
            set1_states[1], len(set1_stateId1_players) ))
    else:
        print("set1: stateId0={}:{}, stateId1={}:{}--> NOK".format(
            set1_states[0], len(set1_stateId0_players), 
            set1_states[1], len(set1_stateId1_players) ))
        
    if len(set(set2_stateId0_players)\
           .intersection(set(set2_stateId1_players))) == 0:
        print("set2: stateId0={}:{}, stateId1={}:{}--> OK".format(
            set2_states[0], len(set2_stateId0_players), 
            set2_states[1], len(set2_stateId1_players) ))
    else:
        print("set2: stateId0={}:{}, stateId1={}:{}--> NOK".format(
            set2_states[0], len(set2_stateId0_players), 
            set2_states[1], len(set2_stateId1_players) ))
        
    # ____ generation of sub set of players in set1 and set2 : fin   _________
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((set1_m_players+set2_m_players,
                                  t_periods,
                                  len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[set1_stateId0_players, t, 
                    AUTOMATE_INDEX_ATTRS["state_i"]] = set1_states[0]                    # state1 or Deficit
    arr_pl_M_T_vars[set1_stateId1_players,t, 
                    AUTOMATE_INDEX_ATTRS["state_i"]] = set1_states[1]                    # state2 or Self
    arr_pl_M_T_vars[set1_players, :, 
                    AUTOMATE_INDEX_ATTRS["set"]] = "set1"
    
    arr_pl_M_T_vars[set2_stateId0_players, t, 
                    INDEX_ATTRS["state_i"]] = set2_states[0]                    # state2 or Self
    arr_pl_M_T_vars[set2_stateId1_players,t, 
                    INDEX_ATTRS["state_i"]] = set2_states[1]                    # state3 or Surplus
    arr_pl_M_T_vars[set2_players, :, 
                    AUTOMATE_INDEX_ATTRS["set"]] = "set2"
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, set1_m_players+set2_m_players):
            state_i = arr_pl_M_T_vars[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["state_i"]]
            
            # compute values and inject in arr_pl_M_T
            Pi_t, Ci_t, Si_t, Si_t_max = None, None, None, None
            Si_t_max = 20
            if state_i == STATES[0]:                                            # state1 or Deficit
                Si_t = 3
                Ci_t = 10 + Si_t
                x = np.random.randint(low=2, high=6, size=1)[0]
                Pi_t = x + math.ceil(Si_t/2)
            elif state_i == STATES[1]:                                          # state2 or Self
                Si_t = 4
                y = np.random.randint(low=20, high=30, size=1)[0]
                Ci_t = y + math.ceil(Si_t/2)
                Pi_t = Ci_t - math.ceil(Si_t/2)
            elif state_i == STATES[2]:                                          # state3 or Surplus
                Si_t = 10
                Ci_t = 30
                x = np.random.randint(low=31, high=40, size=1)[0]
                Pi_t = x + math.ceil(Si_t/2)
                
                
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si", Si_t), 
                    ("Si_max", Si_t_max), ("mode_i","")]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # determine state of player for t+1
            state_i_t_plus_1 = None
            if num_pl_i in set1_players and state_i == set1_states[0]:           # set1_states[0] = Deficit
                state_i_t_plus_1 = np.random.choice(set1_states, p=[0.7,0.3])
            if num_pl_i in set1_players and state_i == set1_states[1]:           # set1_states[1] = Self
                state_i_t_plus_1 = np.random.choice(set1_states, p=[0.5,0.5])
            if num_pl_i in set2_players and state_i == set2_states[0]:           # set2_states[0] = Self
                state_i_t_plus_1 = np.random.choice(set2_states, p=[0.4,0.6])
            if num_pl_i in set2_players and state_i == set2_states[1]:           # set2_states[1] = Surplus
                state_i_t_plus_1 = np.random.choice(set2_states, p=[0.4,0.6])
            
            if t < t_periods-1:
                arr_pl_M_T_vars[
                    num_pl_i, t+1, 
                    AUTOMATE_INDEX_ATTRS["state_i"]] = state_i_t_plus_1
            
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance(set1_m_players, set2_m_players, 
                            t_periods, 
                            set1_states, 
                            set2_states,
                            set1_stateId0_m_players,
                            set2_stateId0_m_players, 
                            path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {state1, state2} : set of players' states 
    set2 = {state2, state3}
    
    Parameters
    ----------
    set1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to set1.
    set2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to set2.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    set1_states : set
        DESCRIPTION.
        set of states creating the group set1
    set2_states: set
        DESCRIPTION.
        set of states creating the group set1
    set1_stateId0_m_players : Integer
        DESCRIPTION. 
        Number of players in the set1 having the state equal to the first state in set1
    set2_stateId0_m_players : Integer
        DESCRIPTION. 
        Number of players in the set2 having the state equal to the first state in set2
    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    # filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
    #                     set1_m_players, set2_m_players, set1_stateId0_m_players,
    #                     set2_stateId0_m_players, t_periods)
    "arr_pl_M_T_players_set1_{}_repSet1_{}_set2_{}_repSet2_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        set1_m_players, set1_stateId0_m_players, 
                        set2_m_players, set2_stateId0_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_Si_Simax_by_automate(
                    set1_m_players, set2_m_players, 
                    t_periods, 
                    set1_states, 
                    set2_states,
                    set1_stateId0_m_players,
                    set2_stateId0_m_players)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_Si_Simax_by_automate(
                    set1_m_players, set2_m_players, 
                    t_periods, 
                    set1_states, 
                    set2_states,
                    set1_stateId0_m_players,
                    set2_stateId0_m_players)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
###############################################################################
#            generate Pi, Ci, Si, state by automate --> fin
###############################################################################

###############################################################################
#            generate Pi, Ci, Si, state by automate for 2, 4 players --> debut
###############################################################################

def generate_Pi_Ci_Si_Simax_by_automate_2_4players(m_players=2, t_periods=2):
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    if m_players == 2:
       arr_pl_M_T_vars[0, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[0]
       arr_pl_M_T_vars[1, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[2]
    elif m_players == 3:
       arr_pl_M_T_vars[0, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[0]
       arr_pl_M_T_vars[1, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[1]
       arr_pl_M_T_vars[2, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[2]
       
       arr_pl_M_T_vars[0, :, AUTOMATE_INDEX_ATTRS["set"]] = 1
       arr_pl_M_T_vars[1, :, AUTOMATE_INDEX_ATTRS["set"]] = 1
       arr_pl_M_T_vars[2, :, AUTOMATE_INDEX_ATTRS["set"]] = 2
    elif m_players == 4:
       arr_pl_M_T_vars[0, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[0]
       arr_pl_M_T_vars[1, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[1]
       arr_pl_M_T_vars[2, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[1]
       arr_pl_M_T_vars[3, :, AUTOMATE_INDEX_ATTRS["state_i"]] = STATES[2]
       
       arr_pl_M_T_vars[0, :, AUTOMATE_INDEX_ATTRS["set"]] = 1
       arr_pl_M_T_vars[1, :, AUTOMATE_INDEX_ATTRS["set"]] = 1
       arr_pl_M_T_vars[2, :, AUTOMATE_INDEX_ATTRS["set"]] = 2
       arr_pl_M_T_vars[3, :, AUTOMATE_INDEX_ATTRS["set"]] = 2
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_M_T_vars[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["state_i"]]
            
            # compute values and inject in arr_pl_M_T
            Pi_t, Ci_t, Si_t, Si_t_max = None, None, None, None
            Si_t_max = 20
            if state_i == STATES[0]:                                            # state1 or Deficit
                Si_t = 3
                Ci_t = 10 + Si_t
                x = np.random.randint(low=2, high=6, size=1)[0]
                Pi_t = x + math.ceil(Si_t/2)
            elif state_i == STATES[1]:                                          # state2 or Self
                Si_t = 4
                y = np.random.randint(low=20, high=30, size=1)[0]
                Ci_t = y + math.ceil(Si_t/2)
                Pi_t = Ci_t - math.ceil(Si_t/2)
            elif state_i == STATES[2]:                                          # state3 or Surplus
                Si_t = 10
                Ci_t = 30
                x = np.random.randint(low=31, high=40, size=1)[0]
                Pi_t = x + math.ceil(Si_t/2)
                
                
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si", Si_t), 
                    ("Si_max", Si_t_max), ("mode_i","")]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars

def get_or_create_instance_2_4players(m_players=2, t_periods=2,
                                      path_to_arr_pl_M_T="", 
                                      used_instances=True):
    """
    get instance if it exists else create instance.

    set1 = {state1, state2} : set of players' states 
    set2 = {state2, state3}
    
    Parameters
    ----------
    m_players : integer
        DESCRIPTION.
        Number of players.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    filename_arr_pl = "arr_pl_M_T_players_{}_periods_{}_DBG.npy".format(
                        m_players, t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    print("path_to_arr_pl_M_T={}".format(path_to_arr_pl_M_T))
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_Si_Simax_by_automate_2_4players(m_players, 
                                                                 t_periods)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_Si_Simax_by_automate_2_4players(m_players, 
                                                                 t_periods)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars   

###############################################################################
#            generate Pi, Ci, Si, state by automate for 2, 4 players --> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate --> debut
###############################################################################
def generate_Pi_Ci_one_period(setA_m_players=15, 
                               setB_m_players=10, 
                               setC_m_players=10, 
                               t_periods=1, 
                               scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA belonging to state1
             setB_m_players = number of players in setB belonging to state2
             setC_m_players = number of players in setC belonging to state3
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B : float [0,1] - moving transition probability from A to B 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB_id_players)
                       .intersection(
                           set(setC_id_players)))) == 0 \
        else print("generation players par setA, setB, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setB_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[1]                  # setB
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, setA_m_players+setB_m_players+setC_m_players):
            prob = np.random.random(size=1)[0]
            Pi_t, Ci_t, Si_t = None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_ABC[0]:                                             # setA = State1 or Deficit
                Si_t = 0
                Ci_t = 15
                Pi_t = np.random.randint(low=5, high=10, size=1)[0]
                
            elif setX == SET_ABC[1] and prob<0.5:                              # setB1 = State2 or Self
                Si_t = 6
                Ci_t = 10
                Pi_t = np.random.randint(low=5, high=8, size=1)[0]
                
            elif setX == SET_ABC[1] and prob>=0.5:                              # setB1 State2 or Self
                Si_t = 8
                Ci_t = 31
                Pi_t = np.random.randint(low=21, high=30, size=1)[0]
                
            elif setX == SET_ABC[2]:                                           # setC State3 or Surplus
                Si_t = 8
                Ci_t = 20
                Pi_t = np.random.randint(low=21, high=30, size=1)[0]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i","")]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB_m_players = number of players in setB
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B : float [0,1] - moving transition probability from A to B 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB_id_players)
                       .intersection(
                           set(setC_id_players)))) == 0 \
        else print("generation players par setA, setB, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((setA_m_players+setB_m_players+setC_m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[0]                  # setA
    arr_pl_M_T_vars[setB_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[1]                  # setB
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_ABC[2]                  # setC
    
    (prob_A_A, prob_A_B, prob_A_C) = scenario[0]
    (prob_B_A, prob_B_B, prob_B_C) = scenario[1]
    (prob_C_A, prob_C_B, prob_C_C) = scenario[2]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, setA_m_players+setB_m_players+setC_m_players):
            Pi_t, Ci_t, Si_t = None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_ABC[0]:                                             # setA
                Si_t = 3
                Ci_t = 10
                x = np.random.randint(low=2, high=8, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_ABC, p=[prob_A_A, 
                                                              prob_A_B, 
                                                              prob_A_C])
            elif setX == SET_ABC[1]:                                           # setB
                Si_t = 7
                Ci_t = 20
                x = np.random.randint(low=12, high=20, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3 
                set_i_t_plus_1 = np.random.choice(SET_ABC, p=[prob_B_A, 
                                                              prob_B_B, 
                                                              prob_B_C])
            elif setX == SET_ABC[2]:                                           # setC
                Si_t = 10
                Ci_t = 30
                x = np.random.randint(low=26, high=35, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                set_i_t_plus_1 = np.random.choice(SET_ABC, p=[prob_C_A, 
                                                              prob_C_B, 
                                                              prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE(setA_m_players, 
                                      setB_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {state1, state2} : set of players' states 
    set2 = {state2, state3}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period(setA_m_players, 
                                      setB_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {state1, state2} : set of players' states 
    set2 = {state2, state3}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
                with prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0,
                     prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3,
                     prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7; 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT.format(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period(setA_m_players, 
                               setB_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB_t, nb_setC_t = 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_ABC[0]:                                             # setA
                Pis = [2,8]; Cis = [10]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_ABC[1]:                                           # setB
                Pis = [12,20]; Cis = [20]; Sis = [4]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_ABC[2]:                                           # setC
                Pis = [26,35]; Cis = [30]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB_t, nb_setC_t = 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_ABC[0]:                                             # setA
                Pis = [5,10]; Cis = [15]; Sis = [0]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_ABC[1]:                                           # setB
                Pis_1 = [5,8]; Cis_1 = [10]; Sis_1 = [6]
                Pis_2 = [21,30]; Cis_2 = [31]; Sis_2 = [8]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if (Pi >= Pis_1[0] and Pi <= Pis_1[1]) \
                                    or (Pi >= Pis_2[0] and Pi <= Pis_2[1]) else 0
                cpt_t_Pi_nok += 1 if (Pi < Pis_1[0] or Pi > Pis_1[1]) \
                                     and (Pi < Pis_2[0] or Pi > Pis_2[1]) else 0
                cpt_t_Ci_ok += 1 if (Ci in Cis_1 or Ci in Cis_2)  else 0
                cpt_t_Ci_nok += 1 if (Ci not in Cis_1 and Ci not in Cis_2) else 0
                cpt_t_Si_ok += 1 if Si in Sis_1 or Si in Sis_2 else 0
                cpt_t_Si_nok += 1 if Si not in Sis_1 and Si not in Sis_2 else 0
                
            elif setX == SET_ABC[2]:                                           # setC
                Pis = [21,30]; Cis = [20]; Sis = [8]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate --> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AB1B2C --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAB1B2C(setA_m_players, 
                                           setB1_m_players, 
                                           setB2_m_players, 
                                           setC_m_players, 
                                           t_periods, 
                                           scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Ci_t = 13
                x = np.random.randint(low=2, high=6, size=1)[0]
                Pi_t = x + 2
                state_i = STATES[0]
                
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                y = np.random.randint(low=20, high=30, size=1)[0]
                Ci_t = y + 3
                Pi_t = Ci_t -2
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[2]:                                        # setB2
                Si_t = 4
                y = np.random.randint(low=20, high=30, size=1)[0]
                Ci_t = y + 3
                Pi_t = Ci_t -2
                state_i = STATES[1]
                
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Ci_t = 30
                x = np.random.randint(low=25, high=35, size=1)[0]
                Pi_t = x
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAB1B2C(setA_m_players, 
                                           setB1_m_players, 
                                           setB2_m_players, 
                                           setC_m_players, 
                                           t_periods, 
                                           scenario=None, 
                                           scenario_name="scenario1"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setB1_m_players = number of players in setB1
             setB2_m_players = number of players in setB2
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
                with prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
                     prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
                     prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
                     prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_B1 : float [0,1] - moving transition probability from A to B1 
                prob_A_B2 : float [0,1] - moving transition probability from A to B2 
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setB1_m_players + setB2_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setB1_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB1_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players))
    setB2_id_players = list(np.random.choice(list(remain_players), 
                                            size=setB2_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setB1_id_players)
                          - set(setB2_id_players)
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setB1, setB2, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setB1_id_players)
                       .intersection(
                           set(setB2_id_players)
                           .intersection(
                               set(setC_id_players)
                           ))
                       )
                ) == 0 \
        else print("generation players par setA, setB1, setB2, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[0]               # setA
    arr_pl_M_T_vars[setB1_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[1]               # setB1
    arr_pl_M_T_vars[setB2_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[2]               # setB2
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AB1B2C[3]               # setC
    
    (prob_A_A, prob_A_B1, prob_A_B2, prob_A_C) = scenario[0]
    (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C) = scenario[1]
    (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C) = scenario[2]
    (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C) = scenario[3]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AB1B2C[0]:                                          # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 14
                x = np.random.randint(low=2, high=4, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_A_A, 
                                                                 prob_A_B1,
                                                                 prob_A_B2,
                                                                 prob_A_C])
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Si_t = 4
                Si_t_max = 10
                Ci_t = 10
                x = np.random.randint(low=10, high=12, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B1_A = 0.3; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B1_A, 
                                                                 prob_B1_B1, 
                                                                 prob_B1_B2,                     
                                                                 prob_B1_C])
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario1":       # setB2
                Si_t = 10
                Si_t_max = 20
                Ci_t = 22
                x = np.random.randint(low=18, high=22, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario2":       # setB2
                Si_t = 10
                Si_t_max = 20
                Ci_t = 26
                x = np.random.randint(low=8, high=18, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_B2_A, 
                                                                 prob_B2_B1, 
                                                                 prob_B2_B2,                     
                                                                 prob_B2_C])
            elif setX == SET_AB1B2C[3]:                                        # setC
                Si_t = 10
                Si_t_max = 20
                Ci_t = 20
                x = np.random.randint(low=24, high=30, size=1)[0]
                Pi_t = x
                # player' set at t+1 prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AB1B2C, p=[prob_C_A, 
                                                                 prob_C_B1,
                                                                 prob_C_B2,
                                                                 prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
                
            # print("t={}, pl_i={}, Pi,Ci,Si={}".format(t, num_pl_i, 
            #         arr_pl_M_T_vars[num_pl_i, t,[AUTOMATE_INDEX_ATTRS["Pi"], 
            #                                      AUTOMATE_INDEX_ATTRS["Ci"],
            #                                      AUTOMATE_INDEX_ATTRS["Si"], 
            #                                      AUTOMATE_INDEX_ATTRS["set"]]]
            #                                           ))
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {state1, state2} : set of players' states 
    set2 = {state2, state3}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAB1B2C(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAB1B2C(setA_m_players, 
                                      setB1_m_players, 
                                      setB2_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {state1, state2} : set of players' states 
    set2 = {state2, state3}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setB1_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB1.
    setB2_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setB2.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
            prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
            prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
            prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setB1_{}_setB2_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAB1B2C(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAB1B2C(setA_m_players, 
                               setB1_m_players, 
                               setB2_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAB1B2C(arr_pl_M_T_vars_init, 
                                           scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2,4]; Cis = [14]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[1]:                                        # setB1
                Pis = [10,12]; Cis = [10]; Sis = [4]; Sis_max=[10]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario1":       # setB2
                Pis = [18,22]; Cis = [22]; Sis = [10]; Sis_max=[20]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[2] and scenario_name == "scenario2":       # setB2
                Pis = [8,18]; Cis = [26]; Sis = [10]; Sis_max=[20]
                nb_setB2_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AB1B2C[3]:                                        # setC
                Pis = [30,40]; Cis = [20]; Sis = [10]; Sis_max=[20]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB1_t,  nb_setB2_t, nb_setC_t = 0, 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AB1B2C[0]:                                          # setA
                Pis = [2+2,6+2]; Cis = [13]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[1]:                                        # setB
                Pis = [20+3-2,30+3-2]; Cis = [20+3, 30+3]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            
            elif setX == SET_AB1B2C[2]:                                        # setB
                Pis = [20+3-2,30+3-2]; Cis = [20+3, 30+3]; Sis = [4]
                nb_setB1_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[1] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[1] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
            elif setX == SET_AB1B2C[2]:                                        # setC
                Pis = [25,35]; Cis = [30]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci >= Cis[0] and Ci <= Cis[0] else 0
                cpt_t_Ci_nok += 1 if Ci < Cis[0] and Ci > Cis[0] else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB1={}, setB2={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB1_t, nb_setB2_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C --> fin
###############################################################################

###############################################################################
#            generate Pi, Ci by automate SET_AC --> debut
###############################################################################
def generate_Pi_Ci_one_period_SETAC(setA_m_players, 
                                    setC_m_players, 
                                    t_periods, 
                                    scenario=None):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
                with prob_A_A = 0.6; prob_A_C = 0.4;
                     prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setC_id_players)
                    )
            ) == 0 \
        else print("generation players par setA, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[0]                   # setA
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[1]                   # setC
    
    (prob_A_A, prob_A_C) = scenario[0]
    (prob_C_A, prob_C_C) = scenario[1]
    
    Si_t_max = 20
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, state_i = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            if setX == SET_AC[0]:                                              # setA
                Si_t = 3
                Ci_t = 20
                Pi_t = 10
                state_i = STATES[0]
                
            elif setX == SET_AC[1]:                                            # setC
                Si_t = 10
                Ci_t = 24
                Pi_t = 32
                state_i = STATES[2]

            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i", state_i)]
            for col, val in cols:
                arr_pl_M_T_vars[num_pl_i, t, 
                                AUTOMATE_INDEX_ATTRS[col]] = val
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars    

def generate_Pi_Ci_by_automate_SETAC(setA_m_players,  
                                    setC_m_players, 
                                    t_periods, 
                                    scenario=None,
                                    scenario_name="scenario0"):
    """
    generate the variables' values for each player using the automata 
    defined in the section 5.1
    
    consider setA_m_players = number of players in setA
             setC_m_players = number of players in setC
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
                with prob_A_A = 0.6; prob_A_C = 0.4;
                     prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A
                prob_A_C : float [0,1] - moving transition probability from A to C
    Returns
    -------
    None.

    """
                        
    # ____ generation of sub set of players in set1 and set2 : debut _________
    m_players = setA_m_players + setC_m_players
    list_players = range(0, m_players)
    
    setA_id_players = list(np.random.choice(list(list_players), 
                                            size=setA_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) - set(setA_id_players))
    setC_id_players = list(np.random.choice(list(remain_players), 
                                            size=setC_m_players, 
                                            replace=False))
    remain_players = list(set(list_players) 
                          - set(setA_id_players) 
                          - set(setC_id_players))
    print("Remain_players: {} -> OK ".format(remain_players)) \
        if len(remain_players) == 0 \
        else print("Remain_players: {} -> NOK ".format(remain_players))
    print("generation players par setA, setC = OK") \
        if len(set(setA_id_players)
                   .intersection(
                       set(setC_id_players)
                       )
                ) == 0 \
        else print("generation players par setA, setC = NOK")
        
    # ____ generation of sub set of players in setA, setB and setC : fin   ____
    
    # ____          creation of arr_pl_M_T_vars : debut             _________
    arr_pl_M_T_vars = np.zeros((m_players,
                                t_periods,
                                len(AUTOMATE_INDEX_ATTRS.keys())),
                                 dtype=object)
    # ____          creation of arr_pl_M_T_vars : fin               _________
    
    # ____ attribution of players' states in arr_pl_M_T_vars : debut _________
    t = 0
    arr_pl_M_T_vars[setA_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[0]                   # setA
    arr_pl_M_T_vars[setC_id_players, t, 
                    AUTOMATE_INDEX_ATTRS["set"]] = SET_AC[1]                   # setC
    
    (prob_A_A, prob_A_C) = scenario[0]
    (prob_C_A, prob_C_C) = scenario[1]
    
    for t in range(0, t_periods):
        for num_pl_i in range(0, m_players):
            Pi_t, Ci_t, Si_t, Si_t_max  = None, None, None, None
            setX = arr_pl_M_T_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["set"]]
            set_i_t_plus_1 = None
            if setX == SET_AC[0]:                                              # setA
                Si_t = 3
                Si_t_max = 10
                Ci_t = 20
                Pi_t = 10
                # player' set at t+1 prob_A_A = 0.6; prob_A_C = 0.4 
                set_i_t_plus_1 = np.random.choice(SET_AC, p=[prob_A_A,
                                                             prob_A_C])
            elif setX == SET_AC[1]:                                            # setC
                Si_t = 10
                Si_t_max = 20
                Ci_t = 25
                Pi_t = 35
                # player' set at t+1 prob_C_A = 0.4; prob_C_C = 0.6; 
                set_i_t_plus_1 = np.random.choice(SET_AC, p=[prob_C_A,
                                                             prob_C_C])
                
            # update arrays cells with variables
            cols = [("Pi",Pi_t), ("Ci",Ci_t), ("Si",Si_t), ("Si_max",Si_t_max), 
                    ("mode_i",""), ("state_i",""), ("set",set_i_t_plus_1)]
            for col, val in cols:
                if col != "set":
                    arr_pl_M_T_vars[num_pl_i, t, 
                                    AUTOMATE_INDEX_ATTRS[col]] = val
                else:
                    if t < t_periods-1:
                        arr_pl_M_T_vars[
                            num_pl_i, t+1, 
                            AUTOMATE_INDEX_ATTRS["set"]] = set_i_t_plus_1
    
    # ____ attribution of players' states in arr_pl_M_T_vars : fin   _________
    
    return arr_pl_M_T_vars
    
def get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC(setA_m_players,
                                      setC_m_players, 
                                      t_periods, 
                                      scenario, 
                                      scenario_name,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {state1, state2} : set of players' states 
    set2 = {state2, state3}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_C = 0.4;
            prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAC.format(
                        setA_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAC(setA_m_players,  
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_by_automate_SETAC(setA_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario, 
                               scenario_name)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def get_or_create_instance_Pi_Ci_one_period_SETAC(setA_m_players, 
                                      setC_m_players, 
                                      t_periods, 
                                      scenario,
                                      path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    set1 = {state1, state2} : set of players' states 
    set2 = {state2, state3}
    
    Parameters
    ----------
    setA_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setA.
    setC_m_players : integer
        DESCRIPTION.
        Number of players having their states belonging to setC.
    t_periods : integer
        DESCRIPTION.
        Number of periods in the game
    scenario : list of tuple. 
                each tuple is the moving transition from one state to the other sates
        DESCRIPTION
        contains the transition probability of each state
        exple  [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
        with 
            prob_A_A = 0.6; prob_A_C = 0.4;
            prob_C_A = 0.4; prob_C_C = 0.6 
                and 
                prob_A_A : float [0,1] - moving transition probability from A to A

    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to save/get array arr_pl_M_T
        example: tests/AUTOMATE_INSTANCES_GAMES/\
                    arr_pl_M_T_players_set1_{m_players_set1}_set2_{m_players_set2}\
                        _periods_{t_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_vars = None
    "arr_pl_M_T_players_setA_{}_setC_{}_periods_{}.npy"
    filename_arr_pl = AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAC.format(
                        setA_m_players, setC_m_players, 
                        t_periods)
    path_to_save = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    path_to_arr_pl_M_T = os.path.join(*[path_to_arr_pl_M_T,filename_arr_pl])
    
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_vars \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAC(setA_m_players, 
                               setC_m_players, 
                               t_periods, 
                               scenario)
            
            save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                                 path_to_save=path_to_save)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_vars \
                = generate_Pi_Ci_one_period_SETAC(setA_m_players,
                               setC_m_players, 
                               t_periods, 
                               scenario)
        save_instances_games(arr_pl_M_T_vars, filename_arr_pl, 
                             path_to_save=path_to_save)
        print("NO PREVIOUS INSTANCE GENERATED: CREATE NOW !!!")
            
    return arr_pl_M_T_vars    
    
def checkout_values_Pi_Ci_arr_pl_SETAC(arr_pl_M_T_vars_init, scenario_name):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        cpt_t_Simax_ok = 0
        nb_setA_t, nb_setC_t = 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si_max"]]
            
            if setX == SET_AC[0]:                                              # setA
                Pis = [10]; Cis = [20]; Sis = [3]; Sis_max=[10]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
            elif setX == SET_AC[1]:                                            # setC
                Pis = [35]; Cis = [25]; Sis = [10]; Sis_max=[20]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                cpt_t_Simax_ok += 1 if Si_max in Sis_max else 0
                cpt_t_Si_nok += 1 if Si_max not in Sis_max else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))
    
def checkout_values_Pi_Ci_arr_pl_one_period_SETAC(arr_pl_M_T_vars_init):
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setC_t = 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars_init[num_pl_i, t, 
                                        AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_AC[0]:                                              # setA
                Pis = [10]; Cis = [20]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_AC[1]:                                            # setC
                Pis = [32]; Cis = [24]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi in Pis else 0
                cpt_t_Pi_nok += 1 if Pi not in Pis else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
    print("arr_pl_M_T_vars_init={}".format(arr_pl_M_T_vars_init.shape))

###############################################################################
#            generate Pi, Ci, Si by automate SET_AB1B2C --> fin
###############################################################################

###############################################################################
#           balanced players 4 mode profile at t and k --> debut   
###############################################################################
def balanced_player_game_4_mode_profil(arr_pl_M_T_vars_modif, 
                                        mode_profile,
                                        t,
                                        pi_0_plus_t, pi_0_minus_t, 
                                        pi_hp_plus, pi_hp_minus,
                                        manual_debug, dbg):
    
    dico_gamma_players_t = dict()
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_vars_modif[num_pl_i, t,
                                   AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_vars_modif[num_pl_i, t,
                                   AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                   AUTOMATE_INDEX_ATTRS['Si']] 
        Si_max = arr_pl_M_T_vars_modif[num_pl_i, t,
                                       AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                        AUTOMATE_INDEX_ATTRS['gamma_i']]
        prod_i, cons_i, r_i = 0, 0, 0
        state_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                        AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                            # update R_i_old
                
        # select mode for player num_pl_i
        mode_i = mode_profile[num_pl_i]
        pl_i.set_mode_i(mode_i)
        
        # compute cons, prod, r_i
        pl_i.update_prod_cons_r_i()
        
        # is pl_i balanced?
        boolean, formule = balanced_player(pl_i, thres=0.1)
        
        # update variables in arr_pl_M_T_k
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), ("mode_i", pl_i.get_mode_i()), 
                ("balanced_pl_i", boolean), ("formule", formule)]
        for col, val in tup_cols_values:
            arr_pl_M_T_vars_modif[num_pl_i, t,
                                    AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_vars_modif, dico_gamma_players_t

def balanced_player_game_t_4_mode_profil_prices_SG(
                        arr_pl_M_T_vars_modif, 
                        mode_profile,
                        t,
                        pi_hp_plus, pi_hp_minus, 
                        pi_0_plus_t, pi_0_minus_t,
                        random_mode,
                        manual_debug, dbg=False):
    """
    """
    # find mode, prod, cons, r_i
    arr_pl_M_T_vars_modif, dico_gamma_players_t \
        = balanced_player_game_4_mode_profil(
            arr_pl_M_T_vars_modif.copy(), 
            mode_profile,
            t,
            pi_0_plus_t, pi_0_minus_t, 
            pi_hp_plus, pi_hp_minus,
            manual_debug, dbg)
    
    # compute pi_sg_{plus,minus}_t_k, pi_0_{plus,minus}_t_k
    b0_t, c0_t, \
    bens_t, csts_t, \
    pi_sg_plus_t, pi_sg_minus_t, \
        = compute_prices_inside_SG(arr_pl_M_T_vars_modif, t,
                                           pi_hp_plus, pi_hp_minus,
                                           pi_0_plus_t, pi_0_minus_t,
                                           manual_debug, dbg)
        
    return arr_pl_M_T_vars_modif, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            dico_gamma_players_t

###############################################################################
#           balanced players 4 mode profile at t and k --> fin   
###############################################################################


# __________    look for whether pli is balanced or not --> debut  ____________

def balanced_player(pl_i, thres=0.1, dbg=False):
    """
    verify if pl_i is whether balanced or unbalanced

    Parameters
    ----------
    pl_i : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    Pi = pl_i.get_Pi(); Ci = pl_i.get_Ci(); Si_new = pl_i.get_Si(); 
    Si_max = pl_i.get_Si_max(); R_i_old = pl_i.get_R_i_old()
    state_i = pl_i.get_state_i(); 
    mode_i = pl_i.get_mode_i()
    cons_i = pl_i.get_cons_i(); prod_i = pl_i.get_prod_i()
    Si_old = pl_i.get_Si_old()
    
    if dbg:
        print("_____ balanced_player Pi={}, Ci={}, Si={}, Si_max={}, state_i={}, mode_i={}"\
              .format(pl_i.get_Pi(), pl_i.get_Ci(), pl_i.get_Si(), 
                      pl_i.get_Si_max(), pl_i.get_state_i(), 
                      pl_i.get_mode_i())) 
    boolean = None
    if state_i == "state1" and mode_i == "CONS+":
        boolean = True if np.abs(Pi+(Si_old-Si_new)+cons_i - Ci)<thres else False
        formule = "Pi+(Si_old-Si_new)+cons_i - Ci"
        res = Pi+(Si_old-Si_new)+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Ci':np.round(Ci,2),
                'Si_new':np.round(Si_new,2), 'Si_max':np.round(Si_max,2), 
                'cons_i':np.round(cons_i,2), 'R_i_old': np.round(R_i_old,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "state1" and mode_i == "CONS-":
        boolean = True if np.abs(Pi+cons_i - Ci)<thres else False
        formule = "Pi+cons_i - Ci"
        res = Pi+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Si_new':np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "state2" and mode_i == "DIS":
        boolean = True if np.abs(Pi+(Si_old-Si_new) - Ci)<thres else False
        formule = "Pi+(Si_old-Si_new) - Ci"
        res = Pi+(Si_old-Si_new) - Ci
        dico = {'Pi':np.round(Pi,2), 'Si_new':np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "state2" and mode_i == "CONS-":
        boolean = True if np.abs(Pi+cons_i - Ci)<thres else False
        formule = "Pi+cons_i - Ci"
        res = Pi+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Si_new':np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "state3" and mode_i == "PROD":
        boolean = True if np.abs(Pi - Ci-prod_i)<thres else False
        formule = "Pi - Ci-prod_i"
        res = Pi - Ci-prod_i
        dico = {'Pi':np.round(Pi,2), 'Si_new':np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                "prod_i": np.round(prod_i,2), 
                'cons_i': np.round(cons_i,2), 
                'Ci': np.round(Ci,2), "state_i": state_i, 
                "mode_i": mode_i, "formule": formule, 
                "res": res}
    elif state_i == "state3" and mode_i == "DIS":
        boolean = True if np.abs(Pi - Ci-(Si_max-Si_old)-prod_i)<thres else False
        formule = "Pi - Ci-(Si_max-Si_old)-prod_i"
        res = Pi - Ci-(Si_max-Si_old)-prod_i
        dico = {'Pi': np.round(Pi,2), 'Si_new': np.round(Si_new,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                "prod_i": np.round(prod_i,2), 
                'cons_i': np.round(cons_i,2), 
                'Ci': np.round(Ci,2), "state_i": state_i, 
                "mode_i": mode_i, "formule": formule, 
                    "res": res, }
    return boolean, formule

# __________    look for whether pli is balanced or not --> fin  ____________

# __________    reupdate states, find possibles modes --> debut     _________
def reupdate_state_players_OLD(arr_pl_M_T_K_vars, t=0, k=0):
    """
    after remarking that some players have 2 states during the game, 
    I decide to write this function to set uniformly the players' state for all
    periods and all learning step

    Parameters
    ----------
    arr_pl_M_T_K_vars : TYPE, optional
        DESCRIPTION. The default is None.
    t : TYPE, optional
        DESCRIPTION. The default is 0.
    k : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """

    m_players = arr_pl_M_T_K_vars.shape[0]
    
    # print("AVANT MODIFICATION")
    # for num_pl_i in range(0, m_players):
    #     res = np.array(
    #             np.unique(
    #                 arr_pl_M_T_K_vars_dbg[1,t,:,fct_aux.INDEX_ATTRS["state_i"]], 
    #                 return_counts=True)).T
    #     print("num_pl_i={} res={}".format(num_pl_i, res))
        
    
    for num_pl_i in range(0, m_players):
        Ci = round(
                arr_pl_M_T_K_vars[num_pl_i, t, k, INDEX_ATTRS["Ci"]], 
                N_DECIMALS)
        Pi = round(
                arr_pl_M_T_K_vars[num_pl_i, t, k, INDEX_ATTRS["Pi"]],
                N_DECIMALS)
        Si = round(
                arr_pl_M_T_K_vars[num_pl_i, t, k, INDEX_ATTRS["Si"]],
                N_DECIMALS)
        Si_max = round(
                    arr_pl_M_T_K_vars[num_pl_i, t, k, 
                                      INDEX_ATTRS["Si_max"]],
                    N_DECIMALS)
        gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                            prod_i, cons_i, r_i, state_i)
        
        # get mode_i, state_i and update R_i_old
        state_i = pl_i.find_out_state_i()
        col = "state_i"
        arr_pl_M_T_K_vars[num_pl_i,:,:,INDEX_ATTRS[col]] = state_i
        
    # print("AVANT MODIFICATION")
    # for num_pl_i in range(0, m_players):
    #     res = np.array(
    #             np.unique(
    #                 arr_pl_M_T_K_vars[1,t,:,INDEX_ATTRS["state_i"]], 
    #                 return_counts=True)).T
    #     print("num_pl_i={} res={}".format(num_pl_i, res))
        
    return arr_pl_M_T_K_vars

def reupdate_state_players(arr_pl_M_T_K_vars, t=0, k=0):
    """
    after remarking that some players have 2 states during the game, 
    I decide to write this function to set uniformly the players' state for all
    periods and all learning step

    Parameters
    ----------
    arr_pl_M_T_K_vars : TYPE, optional
        DESCRIPTION. The default is None.
    t : TYPE, optional
        DESCRIPTION. The default is 0.
    k : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """

    m_players = arr_pl_M_T_K_vars.shape[0]
    possibles_modes = list()
    
    arr_pl_vars = None
    if len(arr_pl_M_T_K_vars.shape) == 3:
        arr_pl_vars = arr_pl_M_T_K_vars
        for num_pl_i in range(0, m_players):
            Ci = round(arr_pl_vars[num_pl_i, t, INDEX_ATTRS["Ci"]], 
                       N_DECIMALS)
            Pi = round(arr_pl_vars[num_pl_i, t, INDEX_ATTRS["Pi"]], 
                       N_DECIMALS)
            Si = round(arr_pl_vars[num_pl_i, t, INDEX_ATTRS["Si"]], 
                       N_DECIMALS)
            Si_max = round(arr_pl_vars[num_pl_i, t, INDEX_ATTRS["Si_max"]],
                           N_DECIMALS)
            gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
            pl_i = None
            pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                prod_i, cons_i, r_i, state_i)
            
            # get mode_i, state_i and update R_i_old
            state_i = pl_i.find_out_state_i()
            col = "state_i"
            arr_pl_vars[num_pl_i,:,INDEX_ATTRS[col]] = state_i
            if state_i == "state1":
                possibles_modes.append(STATE1_STRATS)
            elif state_i == "state2":
                possibles_modes.append(STATE2_STRATS)
            elif state_i == "state3":
                possibles_modes.append(STATE3_STRATS)
            # print("3: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
                
    elif len(arr_pl_M_T_K_vars.shape) == 4:
        arr_pl_vars = arr_pl_M_T_K_vars
        for num_pl_i in range(0, m_players):
            Ci = round(arr_pl_vars[num_pl_i, t, k, INDEX_ATTRS["Ci"]], 
                       N_DECIMALS)
            Pi = round(arr_pl_vars[num_pl_i, t, k, INDEX_ATTRS["Pi"]], 
                       N_DECIMALS)
            Si = round(arr_pl_vars[num_pl_i, t, k, INDEX_ATTRS["Si"]], 
                       N_DECIMALS)
            Si_max = round(arr_pl_vars[num_pl_i, t, k, INDEX_ATTRS["Si_max"]],
                        N_DECIMALS)
            gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
            pl_i = None
            pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                prod_i, cons_i, r_i, state_i)
            
            # get mode_i, state_i and update R_i_old
            state_i = pl_i.find_out_state_i()
            col = "state_i"
            arr_pl_vars[num_pl_i,:,:,INDEX_ATTRS[col]] = state_i
            if state_i == "state1":
                possibles_modes.append(STATE1_STRATS)
            elif state_i == "state2":
                possibles_modes.append(STATE2_STRATS)
            elif state_i == "state3":
                possibles_modes.append(STATE3_STRATS)
            # print("4: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
    else:
        print("STATE_i: NOTHING TO UPDATE.")
        
    return arr_pl_vars, possibles_modes

def possibles_modes_players_automate(arr_pl_M_T_K_vars, t=0, k=0):
    """
    after remarking that some players have 2 states during the game, 
    I decide to write this function to set uniformly the players' state for all
    periods and all learning step

    Parameters
    ----------
    arr_pl_M_T_K_vars : TYPE, optional
        DESCRIPTION. The default is None.
    t : TYPE, optional
        DESCRIPTION. The default is 0.
    k : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """

    m_players = arr_pl_M_T_K_vars.shape[0]
    possibles_modes = list()
    
    arr_pl_vars = None
    if len(arr_pl_M_T_K_vars.shape) == 3:
        arr_pl_vars = arr_pl_M_T_K_vars
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_vars[num_pl_i, t, 
                                  AUTOMATE_INDEX_ATTRS["state_i"]] 
            
            # get mode_i
            if state_i == "state1":
                possibles_modes.append(STATE1_STRATS)
            elif state_i == "state2":
                possibles_modes.append(STATE2_STRATS)
            elif state_i == "state3":
                possibles_modes.append(STATE3_STRATS)
            # print("3: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
                
    elif len(arr_pl_M_T_K_vars.shape) == 4:
        arr_pl_vars = arr_pl_M_T_K_vars
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_vars[num_pl_i, t, k, 
                                  AUTOMATE_INDEX_ATTRS["state_i"]]
            
            # get mode_i
            if state_i == "state1":
                possibles_modes.append(STATE1_STRATS)
            elif state_i == "state2":
                possibles_modes.append(STATE2_STRATS)
            elif state_i == "state3":
                possibles_modes.append(STATE3_STRATS)
            # print("4: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
    else:
        print("STATE_i: NOTHING TO UPDATE.")
        
    return possibles_modes

# __________    reupdate states, find possibles modes --> fin       _________

def compute_real_money_SG(arr_pls_M_T, pi_sg_plus_s, pi_sg_minus_s, 
                          INDEX_ATTRS):
    """
    compute real cost (CC)/benefit (BB) and real money (RU) inside the SG

    Parameters
    ----------
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*len(INDEX_ATTRS)
        DESCRIPTION.
    pi_sg_plus_s : list of energy price exported to HP. NUM_PERIODS items
        DESCRIPTION.
    pi_sg_minus_s : list of energy price imported from HP. NUM_PERIODS items
        DESCRIPTION.

    Returns
    -------
    BB_i: real benefits' array of M_PLAYERS, 
    CC_i: real costs' array of M_PLAYERS, 
    RU_i: real money's array of M_PLAYERS.

    """
    BB, CC, RU = [], [], []
    for num_pl in range(0, arr_pls_M_T.shape[0]):
        CONS_pl = arr_pls_M_T[num_pl, :, INDEX_ATTRS["cons_i"]]
        PROD_pl = arr_pls_M_T[num_pl, :, INDEX_ATTRS["prod_i"]]
        BB_pl = pi_sg_plus_s[-1] * sum(PROD_pl)
        CC_pl = pi_sg_minus_s[-1] * sum(CONS_pl)
        ru_pl = BB_pl - CC_pl
        
        BB.append(BB_pl); CC.append(CC_pl);RU.append(ru_pl)
        
    return BB, CC, RU
    
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
    ext : String, Optional
        DESCRIPTION.
        extension of variables
    threshold: float, Optional
        DESCRIPTION.
        percent of specified files in a directory 
    depth: integer, Optional
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
        located_files = [fn for fn in files if fn.endswith(ext)]
        if round(len(located_files)/len(files)) >= threshold \
            or depth == n_depth:
            boolean = False
        else:
            name_dir = os.path.join(*dirs)
            
    path_to_variables = os.path.join(*dirs)
      
    return path_to_variables

def one_hot_string_without_set_classe(array):
    """
    convert an array of string assuming that all items in array are 
    inside the set classe

    Parameters
    ----------
    array : (n_items,)
        DESCRIPTION.

    Returns
    -------
    onehot : array of (array.shape[0], set(array))
        DESCRIPTION.

    """
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

def one_hot_string_with_set_classe(array, classes):
    """
    convert an array of string assuming that all items in array are not
    inside the set classe

    Parameters
    ----------
    array : (n_items,)
        DESCRIPTION.

    Returns
    -------
    onehot : array of (array.shape[0], set(classes))
        DESCRIPTION.

    """
    # define a mapping of chars to integers
    string_to_int = dict((c, i) for i, c in enumerate(classes))
    int_to_string = dict((i, c) for i, c in enumerate(classes))
    # integer encode input data
    integer_encoded = [string_to_int[string_] for string_ in array]
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        string_ = [0 for _ in range(len(classes))]
        string_[value] = 1
        onehot_encoded.append(string_)
    
    return onehot_encoded

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
    
def test_compute_real_money_SG():  
    m_players = 5
    num_periods = 5
    INDEX_ATTRS = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, "gamma_i":4, 
                   "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9}
    arr_pls_M_T = [] #np.ones(shape=(m_players, num_periods), dtype=object)
    for (num_pl, t) in it.product(range(m_players), range(num_periods)):
        arr_pls_M_T.append([t]*10)
    arr_pls_M_T = [arr_pls_M_T[i:i+num_periods] for i in range(0, len(arr_pls_M_T), num_periods)]
    arr_pls_M_T = np.array(arr_pls_M_T, dtype=object)
    #print("arr_pls_M_T = {} \n {}".format(arr_pls_M_T.shape, arr_pls_M_T))
    pi_sg_minus_s = [2]*num_periods
    pi_sg_plus_s = [3]*num_periods
    BB, CC, RU = compute_real_money_SG(arr_pls_M_T, 
                                       pi_sg_plus_s, pi_sg_minus_s, 
                                       INDEX_ATTRS)
    print("BB={},CC={},RU={}".format(BB, CC, RU))
    if sum(BB)/m_players == BB[0] \
        and sum(CC)/m_players == CC[0] \
        and sum(RU)/m_players == RU[0]:
        print("True")
    else:
        print("False")
    
def test_compute_utility_players():
    
    arr_pls_M_T = generate_Pi_Ci_Si_Simax_by_profil_scenario(
                            m_players=M_PLAYERS, num_periods=NUM_PERIODS, 
                            scenario="scenario1", prob_Ci=0.3, 
                            Ci_low=Ci_LOW, Ci_high=Ci_HIGH)
    
    OK = 0
    for t in range(0, NUM_PERIODS):
        b0, c0 = np.random.randn(), np.random.randn()
        gamma_is = arr_pls_M_T[:,t, INDEX_ATTRS["gamma_i"]]
            
        bens, csts = compute_utility_players(arr_pls_M_T, gamma_is, t, b0, c0)
        
        if bens.shape == (M_PLAYERS,) \
            and csts.shape == (M_PLAYERS,):
            print("bens={}, csts={}, gamma_is={}".format(
                    bens.shape, csts.shape, gamma_is.shape))
            OK += 1
    print("test_compute_utility_players: rp={}".format(
            round(OK/NUM_PERIODS,2)))
    
def test_compute_prod_cons_SG():
    """
    compute the percent of unbalanced grid (production, consumption, balanced)
    at each time.

    Returns
    -------
    None.

    """
    
    arr_pl_M_T = generate_Pi_Ci_Si_Simax_by_profil_scenario(
                            m_players=M_PLAYERS, num_periods=NUM_PERIODS, 
                            scenario="scenario1", prob_Ci=0.3, 
                            Ci_low=Ci_LOW, Ci_high=Ci_HIGH)
    
    production = 0; consumption = 0; balanced = 0
    for t in range(0, NUM_PERIODS):
        In_sg_t, Out_sg_t = compute_prod_cons_SG(arr_pl_M_T, t)
        if In_sg_t > Out_sg_t:
            production += 1
        elif In_sg_t < Out_sg_t:
            consumption += 1
        else:
            balanced += 1
    
    print("SG: production={}, consumption={}, balanced={}".format(
            round(production/NUM_PERIODS,2), 
            round(consumption/NUM_PERIODS,2),
            round(balanced/NUM_PERIODS,2) ))
  
def test_compute_energy_unit_price():
    pi_0_plus, pi_0_minus, pi_hp_plus, pi_hp_minus = np.random.randn(4)
    In_sg, Out_sg = np.random.randint(2, len(INDEX_ATTRS), 2)
    
    b0, c0 = compute_energy_unit_price(pi_0_plus, pi_0_minus, 
                                       pi_hp_plus, pi_hp_minus,
                                       In_sg, Out_sg)
    print("b0={}, c0={}".format(b0, c0))
    
def test_generate_Pi_Ci_Si_Simax_by_profil_scenario():
    
    arr_pl_M_T = generate_Pi_Ci_Si_Simax_by_profil_scenario(
                            m_players=30, num_periods=5, 
                            scenario="scenario1", prob_Ci=0.3, 
                            Ci_low=Ci_LOW, Ci_high=Ci_HIGH)

    # compter le nombre players ayant Ci = 10 et Ci = 60
    cis_weak = arr_pl_M_T[arr_pl_M_T[:, 1, INDEX_ATTRS["Ci"]] == Ci_LOW].shape[0]
    cis_strong = arr_pl_M_T[arr_pl_M_T[:, 1, INDEX_ATTRS["Ci"]] == Ci_HIGH].shape[0]
    
    print("___ arr_pl_M_T : {}, Ci_weak={}, Ci_strong={}".format(
            arr_pl_M_T.shape, round(cis_weak/arr_pl_M_T.shape[0],2), 
            round(cis_strong/arr_pl_M_T.shape[0],2)))
    return arr_pl_M_T

def test_generer_Pi_Ci_Si_Simax_for_all_scenarios():
    m_players=100; num_periods=250
    l_arr_pl_M_T = []
    l_arr_pl_M_T = generer_Pi_Ci_Si_Simax_for_all_scenarios(
                        scenarios=["scenario1", "scenario2", "scenario3"], 
                        m_players=m_players, num_periods=num_periods, 
                        prob_Ci=0.3, Ci_low=Ci_LOW, Ci_high=Ci_HIGH)
    cpt_true = 0
    for arr_pl_M_T in l_arr_pl_M_T:
        if arr_pl_M_T.shape == (m_players, num_periods, len(INDEX_ATTRS)):
            cpt_true += 1
            
    if cpt_true == len(l_arr_pl_M_T):
        print("___ generer_Pi_Ci_Si_Simax_for_all_scenarios ___ OK")
        print(" m_players={}, num_periods={}".format(m_players, num_periods))
        print(" memsize \n scenario1:{} Mo, scenario2:{} Mo, scenario3:{} Mo".format(
                sys.getsizeof(l_arr_pl_M_T[0])/(1024*1024), 
                sys.getsizeof(l_arr_pl_M_T[1])/(1024*1024),
                sys.getsizeof(l_arr_pl_M_T[2])/(1024*1024)))
    else:
        print("___ generer_Pi_Ci_Si_Simax_for_all_scenarios ___ NOK")
        
def test_get_or_create_instance_2_4players():
    m_players=2; t_periods=2
    
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"]); 
    used_instances=True
    
    arr_pl_M_T_vars = get_or_create_instance_2_4players(m_players, t_periods,
                                      path_to_arr_pl_M_T, 
                                      used_instances)
    
    print("shape arr_pl_M_T_vars={}".format(arr_pl_M_T_vars.shape))
    
def test_get_or_create_instance_Pi_Ci_etat_AUTOMATE():
    
    setA_m_players = 10; setB_m_players = 6; setC_m_players = 5 
    t_periods = 20 
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False#True
    prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario = [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
    
    arr_pl_M_T_vars = get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, 
                        scenario,
                        path_to_arr_pl_M_T, used_instances)
    m_players = arr_pl_M_T_vars.shape[0]
    t_periods = arr_pl_M_T_vars.shape[1]
    for t in range(0, t_periods):
        cpt_t_Pi_nok, cpt_t_Pi_ok = 0, 0
        cpt_t_Ci_ok, cpt_t_Ci_nok = 0, 0
        cpt_t_Si_ok, cpt_t_Si_nok = 0, 0
        nb_setA_t, nb_setB_t, nb_setC_t = 0, 0, 0
        for num_pl_i in range(0, m_players):
            setX = arr_pl_M_T_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars[num_pl_i, t, AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == SET_ABC[0]:                                              # setA
                Pis = [2,8]; Cis = [10]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_ABC[1]:                                            # setB
                Pis = [12,20]; Cis = [20]; Sis = [4]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == SET_ABC[2]:                                            # setC
                Pis = [26,35]; Cis = [30]; Sis = [10]
                nb_setC_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
                
        # print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Pi_NOK={}, Ci_OK={}, Ci_NOK={}, Si_NOK={}, Pi={}, Ci={}, Si={}".format(
        #         t, nb_setA_t, nb_setB_t, nb_setC_t, cpt_t_Pi_ok, cpt_t_Pi_nok,
        #         cpt_t_Ci_ok, cpt_t_Ci_nok, cpt_t_Si_nok, Pi, Ci, Si))
        print("t={}, setA={}, setB={}, setC={}, Pi_OK={}, Ci_OK={}, Si_OK={}".format(
                t, nb_setA_t, nb_setB_t, nb_setC_t, 
                round(cpt_t_Pi_ok/m_players,2), round(cpt_t_Ci_ok/m_players,2), 
                1-round(cpt_t_Si_nok/m_players,2) ))
            
                
    print("arr_pl_M_T_vars={}".format(arr_pl_M_T_vars.shape))

def test_get_or_create_instance_Pi_Ci_one_period():
    
    print(" \n \n arr_one period = ")
    setA_m_players = 15; setB_m_players = 10; setC_m_players = 10 
    t_periods = 1 
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = False#True #False#True
    prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario = [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
    
    arr_pl_M_T_vars = get_or_create_instance_Pi_Ci_one_period(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, 
                        scenario,
                        path_to_arr_pl_M_T, used_instances)
    checkout_values_Pi_Ci_arr_pl_one_period(arr_pl_M_T_vars)
                
    print("arr_pl_M_T_vars={}".format(arr_pl_M_T_vars.shape))

def test_get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C():
    
    setA_m_players = 10; setB1_m_players = 3; 
    setB2_m_players = 5; setC_m_players = 8
    t_periods = 20 
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = False#True
    prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
    prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
    prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
    prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6;
    scenario = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
    scenario_name = "scenario1"
    
    arr_pl_M_T_vars = get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, 
                        scenario, 
                        scenario_name,
                        path_to_arr_pl_M_T, used_instances)
    checkout_values_Pi_Ci_arr_pl_SETAB1B2C(arr_pl_M_T_vars, scenario_name)
    print("arr_pl_M_T_vars={}".format(arr_pl_M_T_vars.shape))

def test_get_or_create_instance_Pi_Ci_one_period_SETAB1B2C():
    
    print(" \n \n arr_one period = ")
    setA_m_players = 15; setB1_m_players = 5; 
    setB2_m_players = 5; setC_m_players = 10;
    t_periods = 1 
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = False#True
    prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
    prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
    prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
    prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6;
    scenario = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
    
    
    arr_pl_M_T_vars = get_or_create_instance_Pi_Ci_one_period_SETAB1B2C(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players,  
                        t_periods, 
                        scenario,
                        path_to_arr_pl_M_T, used_instances)
    checkout_values_Pi_Ci_arr_pl_one_period_SETAB1B2C(arr_pl_M_T_vars)
                
    print("arr_pl_M_T_vars={}".format(arr_pl_M_T_vars.shape))
    
def test_get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC():
    
    setA_m_players = 10; setC_m_players = 10
    t_periods = 20 
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = False#True
    prob_A_A = 0.6; prob_A_C = 0.4;
    prob_C_A = 0.4; prob_C_C = 0.6;
    scenario = [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
    scenario_name = "scenario0"
    
    arr_pl_M_T_vars = get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC(
                        setA_m_players, setC_m_players, 
                        t_periods, 
                        scenario,
                        scenario_name,
                        path_to_arr_pl_M_T, used_instances)
    checkout_values_Pi_Ci_arr_pl_SETAC(arr_pl_M_T_vars, scenario_name)
    print("arr_pl_M_T_vars={}".format(arr_pl_M_T_vars.shape))

def test_get_or_create_instance_Pi_Ci_one_period_SETAC():
    
    print(" \n \n arr_one period = ")
    setA_m_players = 10; setC_m_players = 10;
    t_periods = 1 
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = False#True
    prob_A_A = 0.6; prob_A_C = 0.4;
    prob_C_A = 0.4; prob_C_C = 0.6;
    scenario = [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
    
    arr_pl_M_T_vars = get_or_create_instance_Pi_Ci_one_period_SETAC(
                        setA_m_players, setC_m_players,  
                        t_periods, 
                        scenario,
                        path_to_arr_pl_M_T, used_instances)
    checkout_values_Pi_Ci_arr_pl_one_period_SETAC(arr_pl_M_T_vars)
                
    print("arr_pl_M_T_vars={}".format(arr_pl_M_T_vars.shape))

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------    
if __name__ == "__main__":
    ti = time.time()
    #test_fct_positive()
    #test_generate_energy_unit_price_SG()
    
    # test_compute_utility_players()
    # test_compute_real_money_SG()
    # test_compute_prod_cons_SG()
    # test_compute_energy_unit_price()
    
    #arrs = test_generate_Pi_Ci_Si_Simax_by_profil_scenario()
    
    #test_generer_Pi_Ci_Si_Simax_for_all_scenarios()
    
    #test_get_or_create_instance_2_4players()
    
    # test_get_or_create_instance_Pi_Ci_etat_AUTOMATE()
    
    test_get_or_create_instance_Pi_Ci_one_period()
    
    # test_get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C()
    # test_get_or_create_instance_Pi_Ci_one_period_SETAB1B2C()
    
    # test_get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC()
    # test_get_or_create_instance_Pi_Ci_one_period_SETAC()
    
    print("runtime = {}".format(time.time() - ti))