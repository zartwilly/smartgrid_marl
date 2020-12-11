# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:42:13 2020

@author: jwehounou

refactored on Fri Nov  6 15:49:06 2020

"""

import os
import time
import math

import numpy as np
import pandas as pd
import itertools as it
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import deterministic_game_model as detGameModel


from datetime import datetime
from pathlib import Path

###############################################################################
#                   definition  des constantes
#
###############################################################################
NON_PLAYING_PLAYERS = {"PLAY":1, "NOT_PLAY":0}

###############################################################################
#                   definition  des fonctions
#
###############################################################################

def find_out_min_max_bg(arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, 
                        t, k, indices_playing_players_pl_is):
    """
    discover to min and max values of players' benefits bg at time t and 
    at step k

    Parameters
    ----------
    arr_pl_M_T_K_vars : array of shape (M_PLAYERS, NUM_PERIODS, K_STEPS, len(vars))
        DESCRIPTION.
    arr_bg_i_nb_repeat_k : array of shape (M_PLAYERS, NB_REPEAT_K_MAX)
        DESCRIPTION
        array containing bg_i for all time when algo repeats at step k.
    t : integer
        DESCRIPTION.
        one time instant
    k : integer
        DESCRIPTION.
        one step of learning

    Returns
    -------
    bg_min_i_t_0_to_k : array of shape (M_PLAYERS,)
        DESCRIPTION.
        the minimum benefit of each player from 0 to k
    bg_max_i_t_0_to_k : array of shape (M_PLAYERS,)
        DESCRIPTION.
        the maximum benefit of each player from 0 to k

    """
    if np.isnan(
            np.array(
                arr_bg_i_nb_repeat_k[:,:], 
                dtype=np.float64)
            ).all():
        bg_max_i_t_0_to_k \
            = np.nanmax(
                arr_pl_M_T_K_vars[
                    :,t,
                    0:k+1, 
                    fct_aux.INDEX_ATTRS["bg_i"]], 
                axis=1)
        bg_min_i_t_0_to_k \
            = np.nanmin(
                arr_pl_M_T_K_vars[
                    :,t,
                    0:k+1, 
                    fct_aux.INDEX_ATTRS["bg_i"]], 
                axis=1)
    else:
        bg_i = arr_pl_M_T_K_vars[
                    :,t,
                    k, 
                    fct_aux.INDEX_ATTRS["bg_i"]]
        bg_i = bg_i.reshape(-1, 1)
        merge_bg_i_arr_bg_i_nb_repeat_k \
            = np.concatenate( 
                    (bg_i,arr_bg_i_nb_repeat_k), 
                    axis=1)
        bg_max_i_t_0_to_k \
            = np.nanmax(
                merge_bg_i_arr_bg_i_nb_repeat_k, 
                axis=1)
        bg_min_i_t_0_to_k \
            = np.nanmin( 
                merge_bg_i_arr_bg_i_nb_repeat_k, 
                axis=1)
    
    return bg_min_i_t_0_to_k, bg_max_i_t_0_to_k


def utility_function_version1(arr_pl_M_T_K_vars, 
                              arr_bg_i_nb_repeat_k,
                              bens_t_k, csts_t_k, 
                              t, k, m_players, indices_non_playing_players, 
                              learning_rate):
    """
    compute the utility of players following the version 1 in document

    Parameters
    ----------
    arr_pl_M_T_K_vars : array of (M_PLAYERS, NUM_PERIODS, K_STEPS, len(vars))
        DESCRIPTION.
    arr_bg_i_nb_repeat_k : array of shape (M_PLAYERS, NB_REPEAT_K_MAX)
        DESCRIPTION
        array containing bg_i for all time when algo repeats at step k.
    bens_t_k : array of shape (M_PLAYERS,)
        DESCRIPTION.
        benefit of players at time t and step k
    csts_t_k : array of shape (M_PLAYERS,)
        DESCRIPTION.
        cost of players at time t and step k
    t : integer
        DESCRIPTION.
        one time instant
    k : integer
        DESCRIPTION.
        one step 
    learning_rate : float
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_K_vars : array of (M_PLAYERS, NUM_PERIODS, K_STEPS, len(vars))
        DESCRIPTION.
    bool_bg_i_min_eq_max : boolean
        DESCRIPTION.
        False if min(bg_i) not equal to max(bg_i)

    """
    # compute bg_i
    indices_playing_players_pl_is \
        = [num_pl_i for num_pl_i in range(0, arr_pl_M_T_K_vars.shape[0]) 
                    if num_pl_i not in indices_non_playing_players]
    for num_pl_i in indices_playing_players_pl_is:
        state = arr_pl_M_T_K_vars[
                    num_pl_i,
                    t, k,
                    fct_aux.INDEX_ATTRS["state_i"]]
        if state == fct_aux.STATES[2]:
            arr_pl_M_T_K_vars[
                num_pl_i,
                t, k,
                fct_aux.INDEX_ATTRS["bg_i"]] = bens_t_k[num_pl_i]
            #print("pl_{}, prod={}, cons={}, r_i={}, gamma_i={}, R_i_old={}, Si_old={}, Si={}, Si_max={}, Pi={}, Ci={}, ben={}, cst={}".format(num_pl_i, round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["prod_i"]],3), round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["cons_i"]],3), round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["r_i"]],3), round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["gamma_i"]],3), round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["R_i_old"]],3), arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["Si_old"]], arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["Si"]], arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["Si_max"]], arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["Pi"]], arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["Ci"]], bens_t_k[num_pl_i], csts_t_k[num_pl_i]))
        else:
            arr_pl_M_T_K_vars[
                num_pl_i,
                t, k,
                fct_aux.INDEX_ATTRS["bg_i"]] \
                = fct_aux.fct_positive(csts_t_k[num_pl_i] , 
                                       bens_t_k[num_pl_i])
            #print("pl_{}, prod={}, cons={}, r_i={}, gamma_i={}, R_i_old={}, Si_old={}, Si={}, Si_max={}, cst={}, ben={}".format(num_pl_i, round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["prod_i"]],3), round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["cons_i"]],3), round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["r_i"]],3), round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["gamma_i"]],3), round(arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["R_i_old"]],3), arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["Si_old"]], arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["Si"]], arr_pl_M_T_K_vars[num_pl_i, t, k,fct_aux.INDEX_ATTRS["Si_max"]], csts_t_k[num_pl_i], bens_t_k[num_pl_i]))
    
    # bg_min_i_t_0_to_k, bg_max_i_t_0_to_k
    bool_bg_i_min_eq_max = False     # False -> any player doesn't have min == max bg
    bg_min_i_t_0_to_k, \
    bg_max_i_t_0_to_k \
        = find_out_min_max_bg(arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, 
                              t, k, indices_playing_players_pl_is)
    bg_min_i_t_0_to_k = np.array(bg_min_i_t_0_to_k, dtype=float)
    bg_max_i_t_0_to_k = np.array(bg_max_i_t_0_to_k, dtype=float)
    comp_min_max_bg = np.isclose(bg_min_i_t_0_to_k,
                                  bg_max_i_t_0_to_k, 
                                  equal_nan=False,
                                  atol=pow(10,-fct_aux.N_DECIMALS))
            
    if comp_min_max_bg.any() == True:
        # print("V1 indices_non_playing_players_old={}".format(indices_non_playing_players))
        # print("V1 bg_i min == max for players {} --->ERROR".format(
        #         np.argwhere(comp_min_max_bg).reshape(-1)))
        bool_bg_i_min_eq_max = True
        indices_non_playing_players_new = np.argwhere(comp_min_max_bg)\
                                            .reshape(-1)
        indices_non_playing_players \
            = set([*indices_non_playing_players,
                   *list(indices_non_playing_players_new)])
        # for num_pl_i in indices_non_playing_players:
        #     state_i = arr_pl_M_T_K_vars[num_pl_i,t,k,fct_aux.INDEX_ATTRS["state_i"]]
        #     mode_i = arr_pl_M_T_K_vars[num_pl_i,t,k,fct_aux.INDEX_ATTRS["mode_i"]]
            # print("#### 11 num_pl_i={}, state={}, mode={}".format(num_pl_i, state_i, mode_i))
        
        return arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, \
                bool_bg_i_min_eq_max, list(indices_non_playing_players),\
                bg_min_i_t_0_to_k, bg_max_i_t_0_to_k
        
    bg_i_t_k = arr_pl_M_T_K_vars[
                :,
                t, k,
                fct_aux.INDEX_ATTRS["bg_i"]]
        
    u_i_t_k = np.empty(shape=(m_players,)); u_i_t_k.fill(np.nan)
    for num_pl_i in indices_playing_players_pl_is:
        state = arr_pl_M_T_K_vars[
                    num_pl_i,
                    t, k,
                    fct_aux.INDEX_ATTRS["state_i"]]
        if state == fct_aux.STATES[2]:
            u_i_t_k[num_pl_i] = 1 - (bg_max_i_t_0_to_k[num_pl_i] 
                                     - bg_i_t_k[num_pl_i])/ \
                                (bg_max_i_t_0_to_k[num_pl_i] 
                                 - bg_min_i_t_0_to_k[num_pl_i]) 
        else:
            u_i_t_k[num_pl_i] = (bg_max_i_t_0_to_k[num_pl_i] 
                                 - bg_i_t_k[num_pl_i]) / \
                              (bg_max_i_t_0_to_k[num_pl_i] 
                               - bg_min_i_t_0_to_k[num_pl_i])
        
    p_i_t_k = arr_pl_M_T_K_vars[
                        :,
                        t, k,
                        fct_aux.INDEX_ATTRS["prob_mode_state_i"]]\
            if k == 0 \
            else arr_pl_M_T_K_vars[
                        :,
                        t, k-1,
                        fct_aux.INDEX_ATTRS["prob_mode_state_i"]]

    # p_i_t_k_new = p_i_t_k + learning_rate * u_i_t_k * (1 - p_i_t_k)
    p_i_t_k_new = np.empty(shape=(m_players,)); p_i_t_k_new.fill(np.nan)
    for num_pl_i in range(0,arr_pl_M_T_K_vars.shape[0]):
        if np.isnan(u_i_t_k[num_pl_i]):
            p_i_t_k_new[num_pl_i] = p_i_t_k[num_pl_i]
        else:
            p_i_t_k_new[num_pl_i] = p_i_t_k[num_pl_i] \
                                    + learning_rate \
                                        * u_i_t_k[num_pl_i] \
                                            * (1 - p_i_t_k[num_pl_i])
            
    u_i_t_k = np.around(np.array(u_i_t_k, dtype=float), fct_aux.N_DECIMALS)
    p_i_t_k_new = np.around(np.array(p_i_t_k_new, dtype=float),
                             fct_aux.N_DECIMALS)
    arr_pl_M_T_K_vars[
            :,
            t, k,
            fct_aux.INDEX_ATTRS["prob_mode_state_i"]] = p_i_t_k_new
    arr_pl_M_T_K_vars[
            :,
            t, k,
            fct_aux.INDEX_ATTRS["u_i"]] = u_i_t_k
    
    return arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, \
            bool_bg_i_min_eq_max, indices_non_playing_players,\
            bg_min_i_t_0_to_k, bg_max_i_t_0_to_k
         
def utility_function_version2(arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k,
                              b0_t_k, c0_t_k,
                              bens_t_k, csts_t_k, 
                              pi_hp_minus, pi_0_minus_t_k,
                              t, k, m_players, indices_non_playing_players,
                              learning_rate):
    """
    compute the utility of players following the version 1 in document

    Parameters
    ----------
    arr_pl_M_T_K_vars : array of (M_PLAYERS, NUM_PERIODS, K_STEPS, len(vars))
        DESCRIPTION.
    arr_bg_i_nb_repeat_k : array of shape (M_PLAYERS, NB_REPEAT_K_MAX)
        DESCRIPTION
        array containing bg_i for all time when algo repeats at step k.
    b0_t_k : float
        DESCRIPTION.
        unit energy price of benefit
    c0_t_k : float
        DESCRIPTION.
        unit energy price of cost
    bens_t_k : array of shape (M_PLAYERS,)
        DESCRIPTION.
        benefit of players at time t and step k
    csts_t_k : array of shape (M_PLAYERS,)
        DESCRIPTION.
        cost of players at time t and step k
    pi_hp_minus: float
        DESCRIPTION.
        the price of imported (purchased) energy from HP to SG
    pi_0_minus_t_k: float
        DESCRIPTION.
        the price of imported (purchased) energy from SG to players
    t : integer
        DESCRIPTION.
        one time instant
    k : integer
        DESCRIPTION.
        one step 
    learning_rate : float
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_K_vars : array of (M_PLAYERS, NUM_PERIODS, K_STEPS, len(vars))
        DESCRIPTION.
    bool_bg_i_min_eq_max : boolean
        DESCRIPTION.
        False if min(bg_i) not equal to max(bg_i)

    """
    indices_playing_players_pl_is \
        = [num_pl_i for num_pl_i in range(0, arr_pl_M_T_K_vars.shape[0]) 
                    if num_pl_i not in indices_non_playing_players]
    # I_m, I_M
    P_i_t_s = arr_pl_M_T_K_vars[
                arr_pl_M_T_K_vars[:,t,k,
                    fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[2]
                ][:,t,k,fct_aux.INDEX_ATTRS["Pi"]]
    C_i_t_s = arr_pl_M_T_K_vars[
                arr_pl_M_T_K_vars[:,t,k,
                    fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[2]
                ][:,t,k,fct_aux.INDEX_ATTRS["Ci"]]
    S_i_t_s = arr_pl_M_T_K_vars[
                arr_pl_M_T_K_vars[:,t,k,
                    fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[2]
                ][:,t,k,fct_aux.INDEX_ATTRS["Si"]]
    Si_max_t_s = arr_pl_M_T_K_vars[
                arr_pl_M_T_K_vars[:,t,k,
                    fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[2]
                ][:,t,k,
                  fct_aux.INDEX_ATTRS["Si_max"]]
    ## I_m
    P_C_S_i_t_s = P_i_t_s - (C_i_t_s + (Si_max_t_s - S_i_t_s))
    P_C_S_i_t_s[P_C_S_i_t_s < 0] = 0
    I_m = np.sum(P_C_S_i_t_s, axis=0) 
    ## I_M
    P_C_i_t_s = P_i_t_s - C_i_t_s
    I_M = np.sum(P_C_i_t_s, axis=0)
    
    # O_m, O_M
    ## O_m
    P_i_t_s = arr_pl_M_T_K_vars[
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       ][:, t, k,
                         fct_aux.INDEX_ATTRS["Pi"]]
    C_i_t_s = arr_pl_M_T_K_vars[
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       ][:, t, k,
                         fct_aux.INDEX_ATTRS["Ci"]]
    S_i_t_s = arr_pl_M_T_K_vars[
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       ][:, t, k,
                         fct_aux.INDEX_ATTRS["Si"]]
    C_P_S_i_t_s = C_i_t_s - (P_i_t_s + S_i_t_s)
    O_m = np.sum(C_P_S_i_t_s, axis=0)
    ## O_M
    P_i_t_s = arr_pl_M_T_K_vars[
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       | 
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[1])
                       ][:, t, k,
                         fct_aux.INDEX_ATTRS["Pi"]]
    C_i_t_s = arr_pl_M_T_K_vars[
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       | 
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[1])
                       ][:, t, k,
                         fct_aux.INDEX_ATTRS["Ci"]]
    S_i_t_s = arr_pl_M_T_K_vars[
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       | 
                    (arr_pl_M_T_K_vars[:, t, k,
                        fct_aux.INDEX_ATTRS["state_i"]] == fct_aux.STATES[1])
                       ][:, t, k,
                         fct_aux.INDEX_ATTRS["Si"]]
    C_P_i_t_s = C_i_t_s - P_i_t_s
    O_M = np.sum(C_P_i_t_s, axis=0)
    
    # c_0_M
    frac = ( (O_M - I_m) * pi_hp_minus + I_M * pi_0_minus_t_k ) / O_m
    c_0_M = min(frac, pi_0_minus_t_k)
    c_0_M = round(c_0_M, fct_aux.N_DECIMALS)
    # print("O_M={}, O_m={}, I_M={}, I_m={}, c0_t_nstep={}".format(O_M, O_m, I_M, I_m, c0_t_k))
    # print("pi_0_minus_t_k={}, frac={}, c_0_M = {}, c0_t_k<=c_0_M={}".format(
    #     pi_0_minus_t_k, frac, c_0_M, c0_t_k<=c_0_M))
    
    # bg_i
    for num_pl_i in indices_playing_players_pl_is:
        bg_i = None
        bg_i = bens_t_k[num_pl_i] - csts_t_k[num_pl_i] \
                + (c_0_M \
                   * fct_aux.fct_positive(
                       arr_pl_M_T_K_vars[num_pl_i, t, k, 
                                         fct_aux.INDEX_ATTRS["Ci"]],
                       arr_pl_M_T_K_vars[num_pl_i, t, k, 
                                         fct_aux.INDEX_ATTRS["Pi"]]
                       ))
        bg_i = round(bg_i, fct_aux.N_DECIMALS)
        arr_pl_M_T_K_vars[num_pl_i, t, k, fct_aux.INDEX_ATTRS["bg_i"]] = bg_i
    
    # bg_min_i_t_0_to_k, bg_max_i_t_0_to_k
    bool_bg_i_min_eq_max = False     # False -> any player doesn't have min == max bg
    bg_min_i_t_0_to_k, \
    bg_max_i_t_0_to_k \
        = find_out_min_max_bg(arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, 
                              t, k, indices_playing_players_pl_is)
    bg_min_i_t_0_to_k = np.array(bg_min_i_t_0_to_k, dtype=float)
    bg_max_i_t_0_to_k = np.array(bg_max_i_t_0_to_k, dtype=float)

    comp_min_max_bg = np.isclose(bg_min_i_t_0_to_k,
                                  bg_max_i_t_0_to_k, 
                                  equal_nan=False,
                                  atol=pow(10,-fct_aux.N_DECIMALS))
    if comp_min_max_bg.any() == True:
        # print("V1 bg_i min == max for players {} --->ERROR".format(
        #         np.argwhere(comp_min_max_bg).reshape(-1)))
        bool_bg_i_min_eq_max = True
        indices_non_playing_players_new = np.argwhere(comp_min_max_bg).reshape(-1)
        indices_non_playing_players \
            = set([*indices_non_playing_players,
                   *list(indices_non_playing_players_new)])
    
        return arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, \
                bool_bg_i_min_eq_max, list(indices_non_playing_players),\
                bg_min_i_t_0_to_k, bg_max_i_t_0_to_k
    
    # u_i_t_k on shape (M_PLAYERS,)
    bg_i_t_k = arr_pl_M_T_K_vars[:, t, k, 
                                 fct_aux.INDEX_ATTRS["bg_i"]]
    u_i_t_k = 1 - (bg_max_i_t_0_to_k - bg_i_t_k)\
                        /(bg_max_i_t_0_to_k - bg_min_i_t_0_to_k)
    
    p_i_t_k = arr_pl_M_T_K_vars[
                        :,
                        t, k,
                        fct_aux.INDEX_ATTRS["prob_mode_state_i"]] \
            if k == 0 \
            else arr_pl_M_T_K_vars[
                        :,
                        t, k-1,
                        fct_aux.INDEX_ATTRS["prob_mode_state_i"]]
    
    p_i_t_k_new =  p_i_t_k + learning_rate * u_i_t_k * (1 - p_i_t_k)
    
    u_i_t_k = np.around(np.array(u_i_t_k, dtype=float), fct_aux.N_DECIMALS)
    p_i_t_k_new = np.around(np.array(p_i_t_k_new, dtype=float),
                             fct_aux.N_DECIMALS)
     
    arr_pl_M_T_K_vars[
        :,
        t, k,
        fct_aux.INDEX_ATTRS["prob_mode_state_i"]] = p_i_t_k_new
    arr_pl_M_T_K_vars[
        :,
        t, k,
        fct_aux.INDEX_ATTRS["u_i"]] = u_i_t_k
   
    
    return arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, \
            bool_bg_i_min_eq_max, indices_non_playing_players,\
            bg_min_i_t_0_to_k, bg_max_i_t_0_to_k

def update_probs_modes_states_by_defined_utility_funtion(
                arr_pl_M_T_K_vars, 
                arr_bg_i_nb_repeat_k,
                t, k,
                b0_t_k, c0_t_k,
                bens_t_k, csts_t_k,
                pi_hp_minus,
                pi_0_plus_t_k, pi_0_minus_t_k,
                m_players, indices_non_playing_players,
                learning_rate,
                utility_function_version=1):
    bool_bg_i_min_eq_max = False
    bg_min_i_t_0_to_k, bg_max_i_t_0_to_k = None, None
    if utility_function_version==1:
        # version 1 of utility function 
        arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, \
        bool_bg_i_min_eq_max, indices_non_playing_players, \
        bg_min_i_t_0_to_k, bg_max_i_t_0_to_k \
            = utility_function_version1(
                              arr_pl_M_T_K_vars, 
                              arr_bg_i_nb_repeat_k, 
                              bens_t_k, csts_t_k, 
                              t, k, m_players, indices_non_playing_players,
                              learning_rate)
    else:
        # version 2 of utility function 
        arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, \
        bool_bg_i_min_eq_max, indices_non_playing_players, \
        bg_min_i_t_0_to_k, bg_max_i_t_0_to_k \
            = utility_function_version2(
                                arr_pl_M_T_K_vars, 
                                arr_bg_i_nb_repeat_k, 
                                b0_t_k, c0_t_k,
                                bens_t_k, csts_t_k, 
                                pi_hp_minus, pi_0_minus_t_k,
                                t, k, m_players, indices_non_playing_players,
                                learning_rate)
        
    return arr_pl_M_T_K_vars, arr_bg_i_nb_repeat_k, \
            bool_bg_i_min_eq_max, indices_non_playing_players, \
            bg_min_i_t_0_to_k, bg_max_i_t_0_to_k

def balanced_player_game_t(arr_pl_M_T_K_vars, t, k, 
                           pi_hp_plus, pi_hp_minus, 
                           pi_sg_plus_t_minus_1_k, pi_sg_minus_t_minus_1_k,
                           m_players, indices_non_playing_players, 
                           num_periods, nb_repeat_k, dbg=False):
    
    cpt_error_gamma = 0; cpt_balanced = 0;
    dico_state_mode_i_t_k = {}; dico_balanced_pl_i_t_k = {}
    
    indices_playing_players_pl_is \
        = [num_pl_i for num_pl_i in range(0, m_players) 
                    if num_pl_i not in indices_non_playing_players]
    for num_pl_i in indices_playing_players_pl_is:
        Ci = round(
                arr_pl_M_T_K_vars[num_pl_i, t, k, fct_aux.INDEX_ATTRS["Ci"]], 
                fct_aux.N_DECIMALS)
        Pi = round(
                arr_pl_M_T_K_vars[num_pl_i, t, k, fct_aux.INDEX_ATTRS["Pi"]],
                fct_aux.N_DECIMALS)
        Si = round(
                arr_pl_M_T_K_vars[num_pl_i, t, k, fct_aux.INDEX_ATTRS["Si"]],
                fct_aux.N_DECIMALS)\
            if k == 0 \
            else \
                round(
                arr_pl_M_T_K_vars[num_pl_i, t, k-1, fct_aux.INDEX_ATTRS["Si"]],
                fct_aux.N_DECIMALS)
        Si_max = round(
                    arr_pl_M_T_K_vars[num_pl_i, t, k, 
                                      fct_aux.INDEX_ATTRS["Si_max"]],
                    fct_aux.N_DECIMALS)
        gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                            prod_i, cons_i, r_i, state_i)
        
        # get mode_i, state_i and update R_i_old
        pl_i.set_R_i_old(Si_max-Si)
        state_i = pl_i.find_out_state_i()
        ## update Si at t==0
        
        p_i_t_k = arr_pl_M_T_K_vars[num_pl_i, 
                                    t, k, 
                                    fct_aux.INDEX_ATTRS["prob_mode_state_i"]] \
            if k == 0 \
            else arr_pl_M_T_K_vars[num_pl_i, 
                                    t, k-1, 
                                    fct_aux.INDEX_ATTRS["prob_mode_state_i"]]
                
        pl_i.select_mode_i(p_i=p_i_t_k)
        
        pl_i.update_prod_cons_r_i()
        
        # balancing
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        cpt_balanced += round(1/m_players, 2) if boolean else 0
        dico_balanced_pl_i_t_k["cpt"] = cpt_balanced
        if "player" in dico_balanced_pl_i_t_k and boolean is False:
            dico_balanced_pl_i_t_k['player'].append(num_pl_i)
        elif boolean is False:
            dico_balanced_pl_i_t_k['player'] = [num_pl_i]
        
        print("_____ pl_{} : balanced={}_____".format(num_pl_i, boolean)) \
            if dbg else None
        if boolean and dbg:
            print("Pi={}, Ci={}, Si_old={}, Si={}, Si_max={}, state_i={}, mode_i={}"\
              .format(
               pl_i.get_Pi(), pl_i.get_Ci(), pl_i.get_Si_old(), pl_i.get_Si(),
               pl_i.get_Si_max(), pl_i.get_state_i(), pl_i.get_mode_i() 
            ))
            print("====> prod_i={}, cons_i={}, new_S_i={}, new_Si_old={}, R_i_old={}, r_i={}".format(
            round(pl_i.get_prod_i(),2), round(pl_i.get_cons_i(),2),
            round(pl_i.get_Si(),2), round(pl_i.get_Si_old(),2), 
            round(pl_i.get_R_i_old(),2), round(pl_i.get_r_i(),2) ))
        
        # compute gamma_i
        Pi_t_plus_1_k \
            = arr_pl_M_T_K_vars[num_pl_i, t+1, k, fct_aux.INDEX_ATTRS["Pi"]] \
                if t+1 < num_periods \
                else 0
        Ci_t_plus_1_k \
            = arr_pl_M_T_K_vars[num_pl_i, t+1, k, fct_aux.INDEX_ATTRS["Ci"]] \
                if t+1 < num_periods \
                else 0
                     
        pl_i.select_storage_politic(
            Ci_t_plus_1 = Ci_t_plus_1_k, 
            Pi_t_plus_1 = Pi_t_plus_1_k, 
            pi_0_plus = pi_sg_plus_t_minus_1_k,
            pi_0_minus = pi_sg_minus_t_minus_1_k,
            pi_hp_plus = pi_hp_plus, 
            pi_hp_minus = pi_hp_minus)
        gamma_i = pl_i.get_gamma_i()
        if gamma_i >= min(pi_sg_minus_t_minus_1_k, pi_sg_plus_t_minus_1_k)-1 \
            and gamma_i <= max(pi_hp_minus, pi_hp_plus):
            pass
        else :
            cpt_error_gamma = round(1/m_players, fct_aux.N_DECIMALS)
            dico_state_mode_i_t_k["cpt"] = \
                dico_state_mode_i_t_k["cpt"] + cpt_error_gamma \
                if "cpt" in dico_state_mode_i_t_k \
                else cpt_error_gamma
            dico_state_mode_i_t_k[(pl_i.state_i, pl_i.mode_i)] \
                = dico_state_mode_i_t_k[(pl_i.state_i, pl_i.mode_i)] + 1 \
                if (pl_i.state_i, pl_i.mode_i) in dico_state_mode_i_t_k \
                else 1
        
        # update variables in arr_pl_M_T_k
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()),
                ("gamma_i", pl_i.get_gamma_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), ("state_i", pl_i.get_state_i()),
                ("mode_i", pl_i.get_mode_i()), ("prob_mode_state_i", p_i_t_k),
                ("balanced_pl_i", boolean), ("formule", formule)]
        for col,val in tup_cols_values:
            arr_pl_M_T_K_vars[num_pl_i, t, k,
                              fct_aux.INDEX_ATTRS[col]] = val
            
    dico_stats_res_t_k = (round(cpt_balanced/m_players, fct_aux.N_DECIMALS),
                         round(cpt_error_gamma/m_players, fct_aux.N_DECIMALS), 
                         dico_state_mode_i_t_k)
    dico_stats_res_t_k = {"balanced": dico_balanced_pl_i_t_k, 
                         "gamma_i": dico_state_mode_i_t_k}    
        
    # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
    # from a pricing model in the document
    pi_sg_plus_t_k_new, pi_sg_minus_t_k_new \
        = fct_aux.determine_new_pricing_sg(
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
    # TODO question a Dominique voir document section 2.2
    """
    je pense que
    pi_0_plus_t_k = round(pi_sg_minus_t_k*pi_hp_plus/pi_hp_minus, 2)
    pi_0_minus_t_k = pi_sg_minus_t_k
    """
    pi_0_plus_t_k = round(pi_sg_minus_t_minus_1_k*pi_hp_plus/pi_hp_minus, 
                          fct_aux.N_DECIMALS)
    pi_0_minus_t_k = pi_sg_minus_t_minus_1_k
    print("pi_sg_minus_t_minus_1_k={}, pi_0_plus_t_k={}, pi_0_minus_t_k={},".format(
        pi_sg_minus_t_minus_1_k, pi_0_plus_t_k, pi_0_minus_t_k)) \
        if dbg else None
    print("pi_sg_plus_t_k={}, pi_sg_minus_t_k={} \n".format(pi_sg_plus_t_k, 
                                                            pi_sg_minus_t_k)) \
        if dbg else None
    
    
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_M_T_K_vars[:,:,k,:], t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t_k, c0_t_k = fct_aux.compute_energy_unit_price(
                    pi_0_plus_t_k, pi_0_minus_t_k, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg) 
    
    # compute ben, cst of shapes (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_K_vars[:, t, k, fct_aux.INDEX_ATTRS["gamma_i"]]
    bens_t_k, csts_t_k = fct_aux.compute_utility_players(
                            arr_pl_M_T_K_vars[:,t,:,:], 
                            gamma_is, 
                            k, 
                            b0_t_k, 
                            c0_t_k)
    print('#### bens_t_k={}, csts_t_k={}'.format(
            bens_t_k.shape, csts_t_k.shape)) \
        if dbg else None
    
    return arr_pl_M_T_K_vars, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k, \
            dico_stats_res_t_k
    
# ______________            debug LRI   ---> debut      _______________________
def lri_balanced_player_game(arr_pl_M_T,
                             pi_hp_plus=0.10, 
                             pi_hp_minus=0.15,
                             m_players=3, 
                             num_periods=4, 
                             k_steps=5,
                             prob_Ci=0.3, 
                             learning_rate=0.01,
                             probs_modes_states=[0.5, 0.5, 0.5],
                             scenario="scenario1",
                             utility_function_version=1,
                             path_to_save="tests", dbg=False):
    
    print("{}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, p_i_t_k={} ---> debut \n".format(
            scenario, prob_Ci, pi_hp_plus, pi_hp_minus, probs_modes_states))
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_plus_T_K.fill(np.nan)
    pi_sg_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_minus_T_K.fill(np.nan)
    pi_sg_plus_t_k, pi_sg_minus_t_k = 0, 0
    
    pi_0_plus_T_K = np.empty(shape=(num_periods, k_steps)); 
    pi_0_plus_T_K.fill(np.nan)
    pi_0_minus_T_K = np.empty(shape=(num_periods, k_steps)); 
    pi_0_minus_T_K.fill(np.nan)
    
    b0_s_T_K = np.empty(shape=(num_periods, k_steps)); b0_s_T_K.fill(np.nan)
    c0_s_T_K = np.empty(shape=(num_periods, k_steps)); c0_s_T_K.fill(np.nan)
    BENs_M_T_K = np.empty(shape=(m_players, num_periods, k_steps)); 
    BENs_M_T_K.fill(np.nan)
    CSTs_M_T_K = np.empty(shape=(m_players, num_periods, k_steps)); 
    CSTs_M_T_K.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)); C_is_M.fill(np.nan)
    
    dico_stats_res = dict()
    
    fct_aux.INDEX_ATTRS["prob_mode_state_i"] = 16
    fct_aux.INDEX_ATTRS["u_i"] = 17
    fct_aux.INDEX_ATTRS["bg_i"] = 18
    fct_aux.INDEX_ATTRS["non_playing_players"] = 19
    # fct_aux.INDEX_ATTRS[""] = 20
    nb_vars_2_add = 4
    # _______ variables' initialization --> fin   ________________
    
    # ____   turn arr_pl_M_T in a array of 4 dimensions   ____
    ## good time 21.3 ns for k_steps = 1000
    arrs = []
    for k in range(0, k_steps):
        arrs.append(list(arr_pl_M_T))
    arrs = np.array(arrs, dtype=object)
    arrs = np.transpose(arrs, [1,2,0,3])
    ## good but slow 21.4 ns for k_steps = 1000
    # arrs = np.broadcast_to(
    #                         arr_pl_M_T, (k_steps,) + arr_pl_M_T.shape);
    # arrs = np.transpose(arrs, [1,2,0,3])
    # return arrs
    
    ## add initial values for the new attributs
    arr_pl_M_T_K_vars = np.zeros((arrs.shape[0],
                             arrs.shape[1],
                             arrs.shape[2],
                             arrs.shape[3]+nb_vars_2_add), 
                            dtype=object)
    arr_pl_M_T_K_vars[:,:,:,:-nb_vars_2_add] = arrs
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["u_i"]] = np.nan
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["bg_i"]] = np.nan
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["prob_mode_state_i"]] = 0.5
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["non_playing_players"]] \
        = NON_PLAYING_PLAYERS["PLAY"]
    
    arr_pl_M_T_K_vars_modif = arr_pl_M_T_K_vars.copy()
    
    # ____      run balanced sg for all num_periods at any k_step     ________
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    for t in range(0, num_periods):
        print("******* t = {}, p1p2p3={} *******".format(t, 
                np.unique(arr_pl_M_T_K_vars_modif[:,t,:,
                            fct_aux.INDEX_ATTRS["prob_mode_state_i"]]))) 
        
        pi_sg_plus_t_minus_1 = pi_hp_plus-1 if t == 0 \
                                            else pi_sg_plus_t_minus_1
        pi_sg_minus_t_minus_1 = pi_hp_minus-1 if t == 0 \
                                            else pi_sg_minus_t_minus_1
        pi_sg_plus_t_k_minus_1 = None
        pi_sg_minus_t_k_minus_1 = None
        indices_non_playing_players = []      # indices of non-playing players because bg_min = bg_max
        nb_repeat_k = 0
        k = 0
        arr_bg_i_nb_repeat_k = np.empty(
                                shape=(m_players,fct_aux.NB_REPEAT_K_MAX))
        arr_bg_i_nb_repeat_k.fill(np.nan)
        while (k < k_steps):
            print("------- t = {}, k = {}, repeat_k = {}, p1p2p3={}, players={} -------".format(
                    t, k, nb_repeat_k,
                    np.unique(arr_pl_M_T_K_vars_modif[:,t,:,
                            fct_aux.INDEX_ATTRS["prob_mode_state_i"]]),
                    arr_pl_M_T_K_vars_modif.shape[0]\
                        -len(indices_non_playing_players) )) \
                if dbg else None
            print("------- pi_sg_plus_t_k_minus_1={}, pi_sg_minus_t_k_minus_1={} -------".format(
                    pi_sg_plus_t_k_minus_1, pi_sg_minus_t_k_minus_1)) \
                if dbg else None
            
            pi_sg_plus_t_k_minus_1 = pi_sg_plus_t_minus_1 \
                                        if k == 0 \
                                        else pi_sg_plus_t_k_minus_1
            pi_sg_minus_t_k_minus_1 = pi_sg_minus_t_minus_1 \
                                        if k == 0 \
                                        else pi_sg_minus_t_k_minus_1
                                            
            ## balanced_player_game_t
            arr_pl_M_T_K_vars_modif, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k, \
            dico_stats_res_t_k \
                = balanced_player_game_t(arr_pl_M_T_K_vars_modif, t, k, 
                           pi_hp_plus, pi_hp_minus, 
                           pi_sg_plus_t_k_minus_1, pi_sg_minus_t_k_minus_1,
                           m_players, indices_non_playing_players, num_periods, 
                           nb_repeat_k, dbg=False)
            
            ## update pi_sg_minus_t_k_minus_1 and pi_sg_plus_t_k_minus_1
            pi_sg_minus_t_k_minus_1 = pi_sg_minus_t_k
            pi_sg_plus_t_k_minus_1 = pi_sg_plus_t_k
            
            ## update variables at each step because they must have to converge in the best case
            #### update pi_sg_plus, pi_sg_minus of shape (NUM_PERIODS,K_STEPS)
            pi_sg_minus_T_K[t,k] = pi_sg_minus_t_k
            pi_sg_plus_T_K[t,k] = pi_sg_plus_t_k
            pi_0_minus_T_K[t,k] = pi_0_minus_t_k
            pi_0_plus_T_K[t,k] = pi_0_plus_t_k
            #### update b0_s, c0_s of shape (NUM_PERIODS,K_STEPS) 
            b0_s_T_K[t,k] = b0_t_k
            c0_s_T_K[t,k] = c0_t_k
            #### update BENs, CSTs of shape (M_PLAYERS,NUM_PERIODS,K_STEPS)
            #### shape: bens_t_k: (M_PLAYERS,)
            BENs_M_T_K[:,t,k] = bens_t_k
            CSTs_M_T_K[:,t,k] = csts_t_k
            
            ## compute new strategies probabilities by using utility fonction
            print("bens_t_k={}, csts_t_k={}".format(bens_t_k.shape, csts_t_k.shape)) \
                if dbg else None
            arr_pl_M_T_K_vars_modif, arr_bg_i_nb_repeat_k, \
            bool_bg_i_min_eq_max, indices_non_playing_players_new, \
            bg_min_i_t_0_to_k, bg_max_i_t_0_to_k \
                = update_probs_modes_states_by_defined_utility_funtion(
                    arr_pl_M_T_K_vars_modif, 
                    arr_bg_i_nb_repeat_k,
                    t, k,
                    b0_t_k, c0_t_k,
                    bens_t_k, csts_t_k,
                    pi_hp_minus,
                    pi_0_plus_t_k, pi_0_minus_t_k,
                    m_players, indices_non_playing_players,
                    learning_rate, 
                    utility_function_version)
                
            if bool_bg_i_min_eq_max and nb_repeat_k != fct_aux.NB_REPEAT_K_MAX:
                k = k
                arr_bg_i_nb_repeat_k[:,nb_repeat_k] \
                    = arr_pl_M_T_K_vars_modif[
                        :,
                        t, k,
                        fct_aux.INDEX_ATTRS["bg_i"]]
                nb_repeat_k += 1
                arr_pl_M_T_K_vars_modif[:,t,k,:] \
                    = arr_pl_M_T_K_vars[:,t,k,:].copy()
                # print("REPEAT t={}, k={}, repeat_k={}, min(bg_i)==max(bg_i)".format(
                #         t, k, nb_repeat_k))
            elif bool_bg_i_min_eq_max and nb_repeat_k == fct_aux.NB_REPEAT_K_MAX:
                ## test indices_non_playing_players --> debut
                indices_non_playing_players = indices_non_playing_players_new 
                ### add marker to not playing players from k+1 to k_steps
                arr_pl_M_T_K_vars[
                        indices_non_playing_players,
                        t,k+1:k_steps,
                        fct_aux.INDEX_ATTRS["non_playing_players"]] \
                            = NON_PLAYING_PLAYERS["NOT_PLAY"]
                ### update bg_i, mode_i, u_i_t_k, p_i_t_k for not playing players from k+1 to k_steps 
                for var in ["bg_i", "mode_i", "prod_i", "cons_i",
                            "u_i", "prob_mode_state_i"]:
                    # TODO change prob_mode_state_i by p_i_t_k
                    # print("SHAPE NON_PLAYING: t={},k={},k_steps={} var_modifs_non_players={}, k:k_steps={}".format(
                    #     t,k,k_steps,
                    #     arr_pl_M_T_K_vars_modif[indices_non_playing_players,
                    #                             t,k,fct_aux.INDEX_ATTRS[var]], 
                    #     arr_pl_M_T_K_vars_modif[indices_non_playing_players,t,
                    #                             k:k_steps,fct_aux.INDEX_ATTRS[var]]))
                    
                    arr_pl_M_T_K_vars_modif[
                        indices_non_playing_players,
                        t,k+1:k_steps,
                        fct_aux.INDEX_ATTRS[var]] \
                        = arr_pl_M_T_K_vars_modif[
                                    indices_non_playing_players,
                                    t,k,
                                    fct_aux.INDEX_ATTRS[var]].reshape(-1,1)
                    arr_pl_M_T_K_vars[
                        indices_non_playing_players,
                        t,k+1:k_steps,
                        fct_aux.INDEX_ATTRS[var]] \
                                = arr_pl_M_T_K_vars_modif[
                                    indices_non_playing_players,
                                    t,k,
                                    fct_aux.INDEX_ATTRS[var]].reshape(-1,1)
                    # print("AFTER UPDATE var={}, k+1={}->k-step={}; arr_={}, \n arr_modif={}".format(
                    #     var, k+1, k_steps,
                    #     arr_pl_M_T_K_vars[
                    #         indices_non_playing_players,
                    #         t,k+1:k_steps,
                    #         fct_aux.INDEX_ATTRS[var]], 
                    #     arr_pl_M_T_K_vars_modif[
                    #         indices_non_playing_players,
                    #         t,k+1:k_steps,
                    #         fct_aux.INDEX_ATTRS[var]]
                    #     ))
                ### update of k, nb_repeat_k and arr_bg_i_nb_repeat_k
                k = k+1
                nb_repeat_k = 0
                remain_m_players = m_players - len(indices_non_playing_players)
                arr_bg_i_nb_repeat_k \
                    = np.empty(
                        shape=(m_players, fct_aux.NB_REPEAT_K_MAX)
                        )
                arr_bg_i_nb_repeat_k.fill(np.nan)
                
            else:
                arr_pl_M_T_K_vars[:,t,k,:] \
                    = arr_pl_M_T_K_vars_modif[:,t,k,:].copy()
                
                k = k+1
                nb_repeat_k = 0
                arr_bg_i_nb_repeat_k \
                    = np.empty(
                        shape=(m_players, fct_aux.NB_REPEAT_K_MAX)
                        )
                arr_bg_i_nb_repeat_k.fill(np.nan)
                indices_non_playing_players = indices_non_playing_players_new
            
        # update pi_sg_plus_t_minus_1 and pi_sg_minus_t_minus_1
        pi_sg_plus_t_minus_1 = pi_sg_plus_T_K[t,k_steps-1]
        pi_sg_minus_t_minus_1 = pi_sg_minus_T_K[t,k_steps-1]
        
       
    # __________        compute prices variables         ______________________
    ## B_is, C_is of shape (M_PLAYERS, )
    CONS_is = np.sum(arr_pl_M_T_K_vars[
                        :, :,
                        k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]], 
                     axis=1)
    PROD_is = np.sum(arr_pl_M_T_K_vars[
                        :, :, 
                        k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]], 
                     axis=1)
    prod_i_T = arr_pl_M_T_K_vars[:,:, k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_T = arr_pl_M_T_K_vars[:,:, k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is_M = np.sum(b0_s_T_K[:,k_steps-1] * prod_i_T, axis=1)
    C_is_M = np.sum(c0_s_T_K[:,k_steps-1] * cons_i_T, axis=1)
    
    ## BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    BB_is_M = pi_sg_plus_T_K[-1,-1] * PROD_is #np.sum(PROD_is)
    # for num_pl, bb_i in enumerate(BB_is_M):
    #     if bb_i != 0:
    #         print("player {}, BB_i={}".format(num_pl, bb_i))
    CC_is_M = pi_sg_minus_T_K[-1,-1] * CONS_is #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * num_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * num_periods, dtype=object)
    
    #__________      save computed variables locally      _____________________
    fct_aux.save_variables(path_to_save, arr_pl_M_T_K_vars, 
                   b0_s_T_K, c0_s_T_K, B_is_M, C_is_M, BENs_M_T_K, CSTs_M_T_K, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T_K, pi_sg_plus_T_K, 
                   pi_0_minus_T_K, pi_0_plus_T_K,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo="LRI")
    
    print("{}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, p_i_t_k={} ---> fin \n".format(
            scenario, prob_Ci, pi_hp_plus, pi_hp_minus, probs_modes_states))
    
    return arr_pl_M_T_K_vars
# ______________            debug LRI   ---> fin        _______________________

###############################################################################
#                   definition  des unittests
#
###############################################################################
def test_lri_balanced_player_game():
    pi_hp_plus = 0.10; pi_hp_minus = 0.15
    pi_hp_plus = 0.2*pow(10,-3); pi_hp_minus = 0.33
    m_players = 3; num_periods = 5; k_steps = 4
    # m_players = 20; num_periods = 15; k_steps = 250
    Ci_low = fct_aux.Ci_LOW; Ci_high = fct_aux.Ci_HIGH
    prob_Ci = 0.3; learning_rate = 0.05;
    probs_modes_states = [0.5, 0.5, 0.5]
    scenario = "scenario1"; 
    utility_function_version = 2 ; path_to_save = "tests"
    
    fct_aux.N_DECIMALS = 3
    fct_aux.NB_REPEAT_K_MAX = 5 #15#5#3#10# 7
    
    # ____   generation initial variables for all players at any time   ____
    arr_pl_M_T = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                                    m_players=m_players, 
                                    num_periods=num_periods, 
                                    scenario=scenario, prob_Ci=prob_Ci, 
                                    Ci_low=Ci_low, Ci_high=Ci_high)
    
    arr_M_T_K_vars = \
    lri_balanced_player_game(arr_pl_M_T,
                             pi_hp_plus=pi_hp_plus, 
                             pi_hp_minus=pi_hp_minus,
                             m_players=m_players, 
                             num_periods=num_periods, 
                             k_steps=k_steps, 
                             prob_Ci=prob_Ci, 
                             learning_rate=learning_rate,
                             probs_modes_states=probs_modes_states,
                             scenario=scenario,
                             utility_function_version=utility_function_version,
                             path_to_save=path_to_save, dbg=False)
    
    return arr_M_T_K_vars

###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    arr_pl_M_T_K_vars = test_lri_balanced_player_game()
    
    print("runtime = {}".format(time.time() - ti))