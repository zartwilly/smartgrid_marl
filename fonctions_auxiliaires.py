# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:45:38 2020

@author: jwehounou
"""
import os
import sys
import time
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

Ci_LOW = 10
Ci_HIGH = 60

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
            + gamma_is * arr_pl_M_T[:, t, INDEX_ATTRS["r_i"]] + 1
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
    In_sg = sum( arr_pl_M_T[:, t, INDEX_ATTRS["prod_i"]] )
    Out_sg = sum( arr_pl_M_T[:, t, INDEX_ATTRS["cons_i"]] )
    return In_sg, Out_sg
    
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

def determine_new_pricing_sg(arr_pl_M_T, pi_hp_plus, pi_hp_minus, t, dbg=False):
    diff_energy_cons_t = 0
    diff_energy_prod_t = 0
    for k in range(0, t+1):
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
    
    sum_cons = sum(sum(arr_pl_M_T[:, :t+1, INDEX_ATTRS["cons_i"]].astype(np.float64)))
    sum_prod = sum(sum(arr_pl_M_T[:, :t+1, INDEX_ATTRS["prod_i"]].astype(np.float64)))
    
    print("NAN: cons={}, prod={}".format(
            np.isnan(arr_pl_M_T[:, :t+1, INDEX_ATTRS["cons_i"]].astype(np.float64)).any(),
            np.isnan(arr_pl_M_T[:, :t+1, INDEX_ATTRS["prod_i"]].astype(np.float64)).any())
        ) if dbg else None
    arr_cons = np.argwhere(np.isnan(arr_pl_M_T[:, :t+1, INDEX_ATTRS["cons_i"]].astype(np.float64)))
    arr_prod = np.argwhere(np.isnan(arr_pl_M_T[:, :t+1, INDEX_ATTRS["prod_i"]].astype(np.float64)))
    
    if arr_cons.size != 0:
        for arr in arr_cons:
            print("{}-->state:{}, Pi={}, Ci={}, Si={}".format(
                arr, arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["state_i"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Pi"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Ci"]],
                arr_pl_M_T[arr[0], arr[1], INDEX_ATTRS["Si"]]))
    
    new_pi_sg_minus_t = round(pi_hp_minus*diff_energy_cons_t / sum_cons, N_DECIMALS)  \
                    if sum_cons != 0 else np.nan
    new_pi_sg_plus_t = round(pi_hp_plus*diff_energy_prod_t / sum_prod, N_DECIMALS) \
                        if sum_prod != 0 else np.nan
                            
    return new_pi_sg_plus_t, new_pi_sg_minus_t


def save_variables(path_to_save, arr_pl_M_T_K_vars, 
                   b0_s_T_K, c0_s_T_K,
                   B_is_M, C_is_M, 
                   BENs_M_T_K, CSTs_M_T_K, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T_K, pi_sg_plus_T_K, 
                   pi_0_minus_T_K, pi_0_plus_T_K,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res,
                   algo="LRI"):
    
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

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------    
if __name__ == "__main__":
    ti = time.time()
    test_fct_positive()
    test_generate_energy_unit_price_SG()
    
    # test_compute_utility_players()
    # test_compute_real_money_SG()
    # test_compute_prod_cons_SG()
    # test_compute_energy_unit_price()
    
    arrs = test_generate_Pi_Ci_Si_Simax_by_profil_scenario()
    
    #test_generer_Pi_Ci_Si_Simax_for_all_scenarios()
    
    print("runtime = {}".format(time.time() - ti))