# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:12:47 2021

@author: jwehounou
"""
import os
import time
import math

import numpy as np
import itertools as it
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

###############################################################################
#                   definition  des fonctions
#
###############################################################################

#_______________   definition of learning functions: debut  ___________________
def find_out_min_max_bg(arr_pl_M_T_K_vars_modif_new, 
                        arr_bg_i_nb_repeat_k, 
                        t, k):
    """
    discover to min and max values of players' benefits bg at time t and 
    at step k

    Parameters
    ----------
    arr_pl_M_T_K_vars : array of shape (M_PLAYERS, T_PERIODS, K_STEPS, len(vars))
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
                arr_pl_M_T_K_vars_modif_new[
                    :,t,
                    0:k+1, 
                    fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]], 
                axis=1)
        bg_min_i_t_0_to_k \
            = np.nanmin(
                arr_pl_M_T_K_vars_modif_new[
                    :,t,
                    0:k+1, 
                    fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]], 
                axis=1)
    else:
        bg_i = arr_pl_M_T_K_vars_modif_new[
                    :,t,
                    k, 
                    fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]]
        bg_i = bg_i.reshape(-1, 1)
        merge_bg_i_arr_bg_i_nb_repeat_k \
            = np.concatenate( 
                    (bg_i, arr_bg_i_nb_repeat_k), 
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
    

def utility_function_version1(arr_pl_M_T_K_vars_modif_new, 
                                arr_bg_i_nb_repeat_k, 
                                bens_t_k, csts_t_k, 
                                t, k, 
                                nb_repeat_k,
                                learning_rate):
    
    """
    compute the utility of players following the version 1 in the document

    Parameters
    ----------
    arr_pl_M_T_K_vars : array of (M_PLAYERS, T_PERIODS, K_STEPS, len(vars))
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
    indices_non_playing_players : list
        DESCRIPTION.
        indices of players having min(bg_i) == max(bg_i)
    """
    
    
    # compute stock maximal
    stock_max \
        = np.max(
            arr_pl_M_T_K_vars_modif_new[:,t,k,
                                        fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] 
            * arr_pl_M_T_K_vars_modif_new[:,t,k,
                                          fct_aux.AUTOMATE_INDEX_ATTRS["gamma_i"]],
            axis=0)
        
    m_players = arr_pl_M_T_K_vars_modif_new.shape[0]
    # compute bg_i
    for num_pl_i in range(0, m_players):
        state_i = arr_pl_M_T_K_vars_modif_new[
                    num_pl_i,
                    t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
        if state_i == fct_aux.STATES[2]:
            arr_pl_M_T_K_vars_modif_new[
                num_pl_i,
                t, k,
                fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = bens_t_k[num_pl_i]
        else:
            arr_pl_M_T_K_vars_modif_new[
                num_pl_i,
                t, k,
                fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] \
                = csts_t_k[num_pl_i] - bens_t_k[num_pl_i] + stock_max
                
    # bg_min_i_t_0_to_k, bg_max_i_t_0_to_k
    bool_bg_i_min_eq_max = False     # False -> any players have min bg != max bg
    bg_min_i_t_0_to_k, \
    bg_max_i_t_0_to_k \
        = find_out_min_max_bg(arr_pl_M_T_K_vars_modif_new, 
                              arr_bg_i_nb_repeat_k, 
                              t, k)
    bg_min_i_t_0_to_k = np.array(bg_min_i_t_0_to_k, dtype=float)
    bg_max_i_t_0_to_k = np.array(bg_max_i_t_0_to_k, dtype=float)
    comp_min_max_bg = np.isclose(bg_min_i_t_0_to_k,
                                  bg_max_i_t_0_to_k, 
                                  equal_nan=False,
                                  atol=pow(10,-fct_aux.N_DECIMALS))
            
    indices_non_playing_players = np.argwhere(comp_min_max_bg)\
                                            .reshape(-1)
    indices_non_playing_players = set(indices_non_playing_players)
    
    if comp_min_max_bg.any() == True \
        and nb_repeat_k != fct_aux.NB_REPEAT_K_MAX:
        # print("V1 indices_non_playing_players_old={}".format(indices_non_playing_players))
        # print("V1 bg_i min == max for players {} --->ERROR".format(
        #         np.argwhere(comp_min_max_bg).reshape(-1)))
        bool_bg_i_min_eq_max = True
        
        # for num_pl_i in indices_non_playing_players:
        #     state_i = arr_pl_M_T_K_vars[num_pl_i,t,k,fct_aux.INDEX_ATTRS["state_i"]]
        #     mode_i = arr_pl_M_T_K_vars[num_pl_i,t,k,fct_aux.INDEX_ATTRS["mode_i"]]
            # print("#### 11 num_pl_i={}, state={}, mode={}".format(num_pl_i, state_i, mode_i))
        
        return arr_pl_M_T_K_vars_modif_new, arr_bg_i_nb_repeat_k, \
                bool_bg_i_min_eq_max, list(indices_non_playing_players)
     
    bg_i_t_k = arr_pl_M_T_K_vars_modif_new[
                :,
                t, k,
                fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]]
        
    u_i_t_k = np.empty(shape=(m_players,)); u_i_t_k.fill(np.nan)
    for num_pl_i in range(0, m_players):
        state_i = arr_pl_M_T_K_vars_modif_new[
                    num_pl_i,
                    t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
        if state_i == fct_aux.STATES[2]:
            u_i_t_k[num_pl_i] = 1 - (bg_max_i_t_0_to_k[num_pl_i] 
                                     - bg_i_t_k[num_pl_i])/ \
                                (bg_max_i_t_0_to_k[num_pl_i] 
                                 - bg_min_i_t_0_to_k[num_pl_i]) 
        else:
            u_i_t_k[num_pl_i] = (bg_max_i_t_0_to_k[num_pl_i] 
                                 - bg_i_t_k[num_pl_i]) / \
                              (bg_max_i_t_0_to_k[num_pl_i] 
                               - bg_min_i_t_0_to_k[num_pl_i])
    
    u_i_t_k[u_i_t_k == np.inf] = 0
    u_i_t_k[u_i_t_k == -np.inf] = 0
    where_is_nan = np.isnan(list(u_i_t_k))
    u_i_t_k[where_is_nan] = 0
    
    p_i_j_k_minus_1 = arr_pl_M_T_K_vars_modif_new[
                        :,
                        t, k,
                        fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]]\
            if k == 0 \
            else arr_pl_M_T_K_vars_modif_new[
                        :,
                        t, k-1,
                        fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]]

    # p_i_j_k = p_i_j_k_1 + learning_rate * u_i_t_k * (1 - p_i_j_k_1)
    p_i_j_k = np.empty(shape=(m_players,)); p_i_j_k.fill(np.nan)
    for num_pl_i in range(0, m_players):
        if np.isnan(u_i_t_k[num_pl_i]):
            p_i_j_k[num_pl_i] = p_i_j_k_minus_1[num_pl_i]
        else:
            p_i_j_k[num_pl_i] = p_i_j_k_minus_1[num_pl_i] \
                                + learning_rate \
                                    * u_i_t_k[num_pl_i] \
                                        * (1 - p_i_j_k_minus_1[num_pl_i])
    
    u_i_t_k = np.around(np.array(u_i_t_k, dtype=float), fct_aux.N_DECIMALS)
    p_i_j_k = np.around(np.array(p_i_j_k, dtype=float),
                             fct_aux.N_DECIMALS)
    
    arr_pl_M_T_K_vars_modif_new[
            :,
            t, k,
            fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] = p_i_j_k
    arr_pl_M_T_K_vars_modif_new[
            :,
            t, k,
            fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = u_i_t_k
    arr_pl_M_T_K_vars_modif_new[
            list(indices_non_playing_players),
            t,k,
            fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
            = fct_aux.NON_PLAYING_PLAYERS["NOT_PLAY"]
    
    
    return arr_pl_M_T_K_vars_modif_new, \
            arr_bg_i_nb_repeat_k, \
            bool_bg_i_min_eq_max, \
            list(indices_non_playing_players)
    
def utility_function_version2(arr_pl_M_T_K_vars_modif_new, 
                                arr_bg_i_nb_repeat_k, 
                                b0_t_k, c0_t_k,
                                bens_t_k, csts_t_k, 
                                pi_hp_minus, pi_0_minus_t_k,
                                t, k, 
                                nb_repeat_k,
                                learning_rate, dbg=False):
    """
    compute the utility of players following the version 1 in document

    Parameters
    ----------
    arr_pl_M_T_K_vars : array of (M_PLAYERS, T_PERIODS, K_STEPS, len(vars))
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
    arr_pl_M_T_K_vars : array of (M_PLAYERS, T_PERIODS, K_STEPS, len(vars))
        DESCRIPTION.
    bool_bg_i_min_eq_max : boolean
        DESCRIPTION.
        False if min(bg_i) not equal to max(bg_i)

    """
    
    # I_m, I_M
    P_i_t_s = arr_pl_M_T_K_vars_modif_new[
                arr_pl_M_T_K_vars_modif_new[
                    :,t,k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[2]
                ][:,t,k,fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
    C_i_t_s = arr_pl_M_T_K_vars_modif_new[
                arr_pl_M_T_K_vars_modif_new[:,t,k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[2]
                ][:,t,k,fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
    S_i_t_s = arr_pl_M_T_K_vars_modif_new[
                arr_pl_M_T_K_vars_modif_new[:,t,k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[2]
                ][:,t,k,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
    Si_max_t_s = arr_pl_M_T_K_vars_modif_new[
                arr_pl_M_T_K_vars_modif_new[:,t,k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[2]
                ][:,t,k,
                  fct_aux.AUTOMATE_INDEX_ATTRS["Si_max"]]
    
    #print("P_i_t_s={}, C_i_t_s={}, \n Si_max_t_s={}, S_i_t_s={}".format(P_i_t_s, C_i_t_s, Si_max_t_s, S_i_t_s))
                  
    ## I_m
    P_C_S_i_t_s = P_i_t_s - (C_i_t_s + (Si_max_t_s - S_i_t_s))
    P_C_S_i_t_s[P_C_S_i_t_s < 0] = 0
    I_m = np.sum(P_C_S_i_t_s, axis=0) 
    ## I_M
    P_C_i_t_s = P_i_t_s - C_i_t_s
    I_M = np.sum(P_C_i_t_s, axis=0)
    
    # O_m, O_M
    ## O_m
    P_i_t_s = arr_pl_M_T_K_vars_modif_new[
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       ][:, t, k,
                         fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
    C_i_t_s = arr_pl_M_T_K_vars_modif_new[
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       ][:, t, k,
                         fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
    S_i_t_s = arr_pl_M_T_K_vars_modif_new[
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       ][:, t, k,
                         fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
    C_P_S_i_t_s = C_i_t_s - (P_i_t_s + S_i_t_s)
    O_m = np.sum(C_P_S_i_t_s, axis=0)
    ## O_M
    P_i_t_s = arr_pl_M_T_K_vars_modif_new[
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       | 
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[1])
                ][:, t, k,
                  fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
    C_i_t_s = arr_pl_M_T_K_vars_modif_new[
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       | 
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[1])
                ][:, t, k,
                  fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
    S_i_t_s = arr_pl_M_T_K_vars_modif_new[
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[0]) 
                       | 
                (arr_pl_M_T_K_vars_modif_new[:, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == fct_aux.STATES[1])
                ][:, t, k,
                  fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
    C_P_i_t_s = C_i_t_s - P_i_t_s
    O_M = np.sum(C_P_i_t_s, axis=0)
    
    # ***** verification I_m <= IN_sg <= I_M et O_m <= OUT_sg <= O_M *****
    IN_sg = np.sum(arr_pl_M_T_K_vars_modif_new[
                        :,t,k,
                        fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]], axis=0)
    OUT_sg = np.sum(arr_pl_M_T_K_vars_modif_new[
                        :,t,k,
                        fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]], axis=0)
    
    if dbg:
        if I_m <= IN_sg and IN_sg <= I_M:
            print("LRI2, t={},k={}: I_m <= IN_sg <= I_M? ---> OK".format(t,k))
            if dbg:
               print("LRI2 t={},k={}: I_m={} <= IN_sg={} <= I_M={} ---> OK"\
                     .format(t,k, round(I_m,2), round(IN_sg,2), round(I_M,2))) 
        else:
            print("LRI2: I_m <= IN_sg <= I_M? t={},k={} ---> NOK".format(t,k))
            print("LRI2 t={},k={}: I_m={} <= IN_sg={} <= I_M={} ---> NOK"\
                     .format(t,k, round(I_m,2), round(IN_sg,2), round(I_M,2)))
            if dbg:
               print("LRI2 t={},k={}: I_m={} <= IN_sg={} <= I_M={} ---> OK"\
                     .format(t,k, round(I_m,2), round(IN_sg,2), round(I_M,2)))
        if O_m <= OUT_sg and OUT_sg <= O_M:
            print("LRI2, t={},k={}: O_m <= OUT_sg <= O_M? ---> OK".format(t,k))
            if dbg:
               print("LRI2 t={},k={}: O_m={} <= OUT_sg={} <= O_M={} ---> OK"\
                     .format(t,k, round(O_m,2), round(OUT_sg,2), round(O_M,2))) 
        else:
            print("LRI2: O_m <= OUT_sg <= O_M? t={},k={} ---> NOK".format(t,k))
            if dbg:
               print("LRI2 t={},k={}: O_m={} <= OUT_sg={} <= O_M={} ---> OK"\
                     .format(t,k, round(O_m,2), round(OUT_sg,2), round(O_M,2))) 
               
    # c_0_M
    frac = ( (O_M - I_m) * pi_hp_minus + I_M * pi_0_minus_t_k ) / O_m
    c_0_M = min(frac, pi_0_minus_t_k)
    c_0_M = round(c_0_M, fct_aux.N_DECIMALS)
    #print("c_0_M = {}, pi_0_minus_t_k={}".format(c_0_M, pi_0_minus_t_k))

    # bg_i
    for num_pl_i in range(0, arr_pl_M_T_K_vars_modif_new.shape[0]):
        bg_i = None
        bg_i = bens_t_k[num_pl_i] - csts_t_k[num_pl_i] \
                + (c_0_M \
                   * fct_aux.fct_positive(
                       arr_pl_M_T_K_vars_modif_new[num_pl_i, t, k, 
                                         fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]],
                       arr_pl_M_T_K_vars_modif_new[num_pl_i, t, k, 
                                         fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
                       ))
        bg_i = round(bg_i, fct_aux.N_DECIMALS)
        arr_pl_M_T_K_vars_modif_new[num_pl_i, t, k, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = bg_i

    # bg_min_i_t_0_to_k, bg_max_i_t_0_to_k
    bool_bg_i_min_eq_max = False     # False -> any players have min bg != max bg
    bg_min_i_t_0_to_k, \
    bg_max_i_t_0_to_k \
        = find_out_min_max_bg(arr_pl_M_T_K_vars_modif_new, 
                              arr_bg_i_nb_repeat_k, 
                              t, k)
    bg_min_i_t_0_to_k = np.array(bg_min_i_t_0_to_k, dtype=float)
    bg_max_i_t_0_to_k = np.array(bg_max_i_t_0_to_k, dtype=float)
    comp_min_max_bg = np.isclose(bg_min_i_t_0_to_k,
                                  bg_max_i_t_0_to_k, 
                                  equal_nan=False,
                                  atol=pow(10,-fct_aux.N_DECIMALS))
            
    indices_non_playing_players = np.argwhere(comp_min_max_bg)\
                                            .reshape(-1)
    indices_non_playing_players = set(indices_non_playing_players)
    
    if comp_min_max_bg.any() == True \
        and nb_repeat_k != fct_aux.NB_REPEAT_K_MAX:
        # print("V1 indices_non_playing_players_old={}".format(indices_non_playing_players))
        # print("V1 bg_i min == max for players {} --->ERROR".format(
        #         np.argwhere(comp_min_max_bg).reshape(-1)))
        bool_bg_i_min_eq_max = True
        
        # for num_pl_i in indices_non_playing_players:
        #     state_i = arr_pl_M_T_K_vars[num_pl_i,t,k,fct_aux.INDEX_ATTRS["state_i"]]
        #     mode_i = arr_pl_M_T_K_vars[num_pl_i,t,k,fct_aux.INDEX_ATTRS["mode_i"]]
            # print("#### 11 num_pl_i={}, state={}, mode={}".format(num_pl_i, state_i, mode_i))
        
        return arr_pl_M_T_K_vars_modif_new, arr_bg_i_nb_repeat_k, \
                bool_bg_i_min_eq_max, list(indices_non_playing_players)
                
    # u_i_t_k on shape (M_PLAYERS,)
    bg_i_t_k = arr_pl_M_T_K_vars_modif_new[:, t, k, 
                                 fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]]
    u_i_t_k = 1 - (bg_max_i_t_0_to_k - bg_i_t_k)\
                        /(bg_max_i_t_0_to_k - bg_min_i_t_0_to_k)
    u_i_t_k[u_i_t_k == np.inf] = 0
    u_i_t_k[u_i_t_k == -np.inf] = 0
    where_is_nan = np.isnan(list(u_i_t_k))
    u_i_t_k[where_is_nan] = 0
    
    p_i_j_k_minus_1 = arr_pl_M_T_K_vars_modif_new[
                        :,
                        t, k,
                        fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]]\
        if k == 0 \
        else arr_pl_M_T_K_vars_modif_new[
                    :,
                    t, k-1,
                    fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]]
        
    p_i_j_k = p_i_j_k_minus_1 + learning_rate * u_i_t_k * (1 - p_i_j_k_minus_1)
    
    u_i_t_k = np.around(np.array(u_i_t_k, dtype=float), fct_aux.N_DECIMALS)
    p_i_j_k = np.around(np.array(p_i_j_k, dtype=float),
                             fct_aux.N_DECIMALS)
    
    arr_pl_M_T_K_vars_modif_new[
            :,
            t, k,
            fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] = p_i_j_k
    arr_pl_M_T_K_vars_modif_new[
            :,
            t, k,
            fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = u_i_t_k
    arr_pl_M_T_K_vars_modif_new[
            list(indices_non_playing_players),
            t,k,
            fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
            = fct_aux.NON_PLAYING_PLAYERS["NOT_PLAY"]
            
    return arr_pl_M_T_K_vars_modif_new, \
            arr_bg_i_nb_repeat_k, \
            bool_bg_i_min_eq_max, \
            list(indices_non_playing_players)          

def update_p_i_j_k_by_defined_utility_funtion(arr_pl_M_T_K_vars_modif_new, 
                                                arr_bg_i_nb_repeat_k,
                                                t, k,
                                                b0_t_k, c0_t_k,
                                                bens_t_k, csts_t_k,
                                                pi_hp_minus,
                                                pi_0_plus_t_k, pi_0_minus_t_k,
                                                nb_repeat_k,
                                                learning_rate, 
                                                utility_function_version, 
                                                dbg=False):
    
    bool_bg_i_min_eq_max = None
    indices_non_playing_players = set()
    if utility_function_version == 1:
        # version 1 of utility function 
        arr_pl_M_T_K_vars_modif_new, \
        arr_bg_i_nb_repeat_k, \
        bool_bg_i_min_eq_max, \
        indices_non_playing_players \
            = utility_function_version1(
                arr_pl_M_T_K_vars_modif_new.copy(), 
                arr_bg_i_nb_repeat_k.copy(), 
                bens_t_k, csts_t_k, 
                t, k, 
                nb_repeat_k,
                learning_rate)
    else:
        # version 2 of utility function 
        arr_pl_M_T_K_vars_modif_new, \
        arr_bg_i_nb_repeat_k, \
        bool_bg_i_min_eq_max, \
        indices_non_playing_players \
            = utility_function_version2(
                arr_pl_M_T_K_vars_modif_new.copy(), 
                arr_bg_i_nb_repeat_k.copy(), 
                b0_t_k, c0_t_k,
                bens_t_k, csts_t_k, 
                pi_hp_minus, pi_0_minus_t_k,
                t, k, 
                nb_repeat_k,
                learning_rate, dbg)
            
    return arr_pl_M_T_K_vars_modif_new, \
            arr_bg_i_nb_repeat_k, \
            bool_bg_i_min_eq_max, \
            indices_non_playing_players

#_______________   definition of learning functions: fin    ___________________

def balanced_player_game_4_random_mode(arr_pl_M_T_K_vars_modif, t, k, dbg):
    m_players = arr_pl_M_T_K_vars_modif.shape[0]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Si']] \
            if k == 0 \
            else arr_pl_M_T_K_vars_modif[num_pl_i, t, k-1, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
        Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i, prod_i, cons_i, r_i = 0, 0, 0, 0
        state_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                             # update R_i_old
        
        # select mode for player num_pl_i
        p_i_t_k = arr_pl_M_T_K_vars_modif[num_pl_i, 
                                    t, k, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] \
            if k == 0 \
            else arr_pl_M_T_K_vars_modif[num_pl_i, 
                                    t, k-1, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]]
        pl_i.select_mode_i(p_i=p_i_t_k)
        
        # compute cons, prod, r_i
        pl_i.update_prod_cons_r_i()

        
        # is pl_i balanced?
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        
        # update variables in arr_pl_M_T_k
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), ("mode_i", pl_i.get_mode_i()), 
                ("balanced_pl_i", boolean), ("formule", formule)]
        for col, val in tup_cols_values:
            arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                    fct_aux.INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_K_vars_modif
        
def compute_gamma_4_players(arr_pl_M_T_K_vars_modif, t, k, 
                            pi_0_plus_t_k, pi_0_minus_t_k,
                            pi_hp_plus, pi_hp_minus,
                            manual_debug=False, dbg=False):
    
    dico_gamma_players_t_k = dict()
    
    m_players = arr_pl_M_T_K_vars_modif.shape[0]
    t_periods = arr_pl_M_T_K_vars_modif.shape[1]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Si']]
        Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        state_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        mode_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['mode_i']]
        prod_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['prod_i']]
        cons_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['cons_i']]
        r_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['r_i']]
        
        gamma_i = 0
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_mode_i(mode_i)
        
        # compute gamma_i, Si_{plus,minus}
        Pi_t_plus_1_k \
            = arr_pl_M_T_K_vars_modif[num_pl_i, t+1, k, 
                                fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]] \
                if t+1 < t_periods \
                else 0
        Ci_t_plus_1_k \
            = arr_pl_M_T_K_vars_modif[num_pl_i, t+1, k, 
                                fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]] \
                if t+1 < t_periods \
                else 0
                     
        pl_i.select_storage_politic(
            Ci_t_plus_1 = Ci_t_plus_1_k, 
            Pi_t_plus_1 = Pi_t_plus_1_k, 
            pi_0_plus = pi_0_plus_t_k,
            pi_0_minus = pi_0_minus_t_k,
            pi_hp_plus = pi_hp_plus, 
            pi_hp_minus = pi_hp_minus)
        
        gamma_i = None
        if manual_debug:
            gamma_i = fct_aux.MANUEL_DBG_GAMMA_I # 5
            pl_i.set_gamma_i(gamma_i)
        else:
            gamma_i = pl_i.get_gamma_i()
        
        dico = dict()
        if gamma_i < min(pi_0_plus_t_k, pi_0_minus_t_k)-1:
            dico["min_pi_0"] = gamma_i
        elif gamma_i > max(pi_hp_minus, pi_hp_plus):
            dico["max_pi_hp"] = gamma_i
            
        dico["state_i"] = state_i; dico["mode_i"] = mode_i
        dico["gamma_i"] = gamma_i
        dico_gamma_players_t_k["player_"+str(num_pl_i)] = dico
    
        # update variables gamma_i, Si_minus, Si_max
        tup_cols_values = [("gamma_i", gamma_i), 
                           ("Si_minus", pl_i.get_Si_minus() ),
                           ("Si_plus", pl_i.get_Si_plus() )]
        for col, val in tup_cols_values:
            arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                              fct_aux.AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_K_vars_modif, dico_gamma_players_t_k
        
def compute_prices_inside_SG(arr_pl_M_T_K_vars_modif, t, k,
                             pi_hp_plus, pi_hp_minus, 
                             manual_debug, dbg):
    """
    compute the prices' and benefits/costs variables: 
        ben_i, cst_i
        pi_sg_plus_t_k, pi_sg_minus_t_k
        pi_0_plus_t_k, pi_0_minus_t_k
    """
    pi_sg_plus_t_k_new, pi_sg_minus_t_k_new \
        = fct_aux.determine_new_pricing_sg(
            arr_pl_M_T_K_vars_modif[:,:,k,:], 
            pi_hp_plus, 
            pi_hp_minus, 
            t, 
            dbg=dbg)
    print("pi_sg_plus_{}_{}_new={}, pi_sg_minus_{}_{}_new={}".format(
        t, k, pi_sg_plus_t_k_new, t, k, pi_sg_minus_t_k_new)) if dbg else None
    
    if t == 0 and k == 0: 
        pi_sg_plus_t_k = pi_hp_plus-1
        pi_sg_minus_t_k = pi_hp_minus-1
    elif t == 0 and k > 0: 
        pi_sg_plus_t_k = pi_sg_plus_t_k_new
        pi_sg_minus_t_k = pi_sg_minus_t_k_new
    else:
        pi_sg_plus_t_k = pi_sg_plus_t_k_new
        pi_sg_minus_t_k = pi_sg_minus_t_k_new
    
    
    pi_0_plus_t_k = round(pi_sg_minus_t_k*pi_hp_plus/pi_hp_minus, 
                          fct_aux.N_DECIMALS)
    pi_0_minus_t_k = pi_sg_minus_t_k
    
    
    if manual_debug:
        pi_sg_plus_t_k = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
        pi_sg_minus_t_k = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
        pi_0_plus_t_k = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #2 
        pi_0_minus_t_k = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
        
    print(", pi_0_plus_t_k={}, pi_0_minus_t_k={},".format(
        pi_0_plus_t_k, pi_0_minus_t_k)) \
        if dbg else None
    print("pi_sg_plus_t_k={}, pi_sg_minus_t_k={} \n".format(pi_sg_plus_t_k, 
                                                            pi_sg_minus_t_k)) \
        if dbg else None
    
    
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                        arr_pl_M_T_K_vars_modif[:,:,k,:], t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t_k, c0_t_k = fct_aux.compute_energy_unit_price(
                        pi_0_plus_t_k, pi_0_minus_t_k, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg) 
    
    # compute ben, cst of shapes (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_K_vars_modif[:, t, k, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["gamma_i"]]
    bens_t_k, csts_t_k = fct_aux.compute_utility_players(
                            arr_pl_M_T_K_vars_modif[:,t,:,:], 
                            gamma_is, 
                            k, 
                            b0_t_k, 
                            c0_t_k)
    print('#### bens_t_k={}, csts_t_k={}'.format(
            bens_t_k.shape, csts_t_k.shape)) \
        if dbg else None
    
    return arr_pl_M_T_K_vars_modif, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k

        

def balanced_player_game_t(arr_pl_M_T_K_vars_modif, t, k, 
                           pi_hp_plus, pi_hp_minus, 
                           m_players, t_periods, 
                           manual_debug=False, dbg=False):
    
    # find mode, prod, cons, r_i
    arr_pl_M_T_K_vars_modif = balanced_player_game_4_random_mode(
                                arr_pl_M_T_K_vars_modif.copy(), t, k, dbg)
    
    # compute pi_sg_{plus,minus}_t_k, pi_0_{plus,minus}_t_k
    arr_pl_M_T_K_vars_modif, \
    b0_t_k, c0_t_k, \
    bens_t_k, csts_t_k, \
    pi_sg_plus_t_k, pi_sg_minus_t_k, \
    pi_0_plus_t_k, pi_0_minus_t_k \
        = compute_prices_inside_SG(arr_pl_M_T_K_vars_modif, t, k,
                                     pi_hp_plus, pi_hp_minus, 
                                     manual_debug, dbg)
    
    # compute gamma_i, Si_{plus,minus}
    arr_pl_M_T_K_vars_modif, dico_gamma_players_t_k \
        = compute_gamma_4_players(
            arr_pl_M_T_K_vars_modif.copy(), t, k, 
            pi_0_plus_t_k, pi_0_minus_t_k,
            pi_hp_plus, pi_hp_minus,
            manual_debug=False, dbg=False)
        
    return arr_pl_M_T_K_vars_modif, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k, \
            dico_gamma_players_t_k
    


# ______________       main function of LRI   ---> debut      _________________
def lri_balanced_player_game(arr_pl_M_T_vars_init,
                             pi_hp_plus=0.10, 
                             pi_hp_minus=0.15,
                             k_steps=5, 
                             learning_rate=0.1,
                             p_i_j_ks=[0.5, 0.5, 0.5],
                             utility_function_version=1,
                             path_to_save="tests", 
                             manual_debug=False, dbg=False):
    
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # _______ variables' initialization --> debut ________________
    
    pi_sg_plus_T_K = np.empty(shape=(t_periods,k_steps)); 
    pi_sg_plus_T_K.fill(np.nan)
    pi_sg_minus_T_K = np.empty(shape=(t_periods,k_steps)); 
    pi_sg_minus_T_K.fill(np.nan)
    pi_0_plus_T_K = np.empty(shape=(t_periods, k_steps)); 
    pi_0_plus_T_K.fill(np.nan)
    pi_0_minus_T_K = np.empty(shape=(t_periods, k_steps)); 
    pi_0_minus_T_K.fill(np.nan)
    b0_s_T_K = np.empty(shape=(t_periods, k_steps)); b0_s_T_K.fill(np.nan)
    c0_s_T_K = np.empty(shape=(t_periods, k_steps)); c0_s_T_K.fill(np.nan)
    BENs_M_T_K = np.empty(shape=(m_players, t_periods, k_steps)); 
    BENs_M_T_K.fill(np.nan)
    CSTs_M_T_K = np.empty(shape=(m_players, t_periods, k_steps)); 
    CSTs_M_T_K.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)); C_is_M.fill(np.nan)
    
    # ____   turn arr_pl_M_T in an array of 4 dimensions   ____
    
    ## good time 21.3 ns for k_steps = 1000
    arrs = []
    for k in range(0, k_steps):
        arrs.append(list(arr_pl_M_T_vars_init))
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
                             arrs.shape[3]), 
                            dtype=object)
    arr_pl_M_T_K_vars[:,:,:,:] = arrs
    arr_pl_M_T_K_vars[:,:,:,fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] = 0.5
    arr_pl_M_T_K_vars[:,:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    # ____      run balanced sg for all num_periods at any k_step     ________
    arr_pl_M_T_K_vars_modif = arr_pl_M_T_K_vars.copy()
    
    
    # pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = None, None
    # if manual_debug:
    #     pi_sg_plus_t_minus_1 = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K # 8 
    #     pi_sg_minus_t_minus_1 = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K # 10
    # else:
    #     pi_sg_plus_t_minus_1 = pi_hp_plus-1
    #     pi_sg_minus_t_minus_1 = pi_hp_minus-1
        
    dico_stats_res = dict()
    for t in range(0, t_periods):
        print("******* t = {} *******".format(t))
        
        # pi_sg_plus_t = pi_sg_plus_t_minus_1 if t == 0 \
        #                                     else pi_sg_plus_T_K[t, k_steps]
        # pi_sg_minus_t = pi_sg_minus_t_minus_1 if t == 0 \
        #                                     else pi_sg_minus_T_K[t, k_steps]
        # pi_0_plus_t = round(pi_sg_plus_t*pi_hp_plus/pi_hp_minus, 
        #                     fct_aux.N_DECIMALS) if t == 0 \
        #                                         else pi_0_plus_T_K[t, k_steps]
        # pi_0_minus_t = pi_sg_minus_t if t == 0 \
        #                              else pi_0_minus_T_K[t, k_steps]
                                            
        
        indices_non_playing_players = []      # indices of non-playing players because bg_min = bg_max
        arr_bg_i_nb_repeat_k = np.empty(
                                shape=(m_players,fct_aux.NB_REPEAT_K_MAX))
        arr_bg_i_nb_repeat_k.fill(np.nan)
        
        dico_gamma_players_t = dict()
        nb_repeat_k = 0
        k = 0
        while k<k_steps:
            # print("------- pi_sg_plus_t_k={}, pi_sg_minus_t_k={} -------".format(
            #         pi_sg_plus_t_k, pi_sg_minus_t_k)) \
            #     if dbg else None
            
            print(" -------  k = {}  ------- ".format(k))
             
            # TODO: a demander a Dominique si a chaque k pi_sg_{plus,minus}_t_k = {8,10}
            # if manual_debug:
            #     pi_sg_plus_t_k = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            #     pi_sg_minus_t_k = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            
            ### balanced_player_game_t
            arr_pl_M_T_K_vars_modif_new, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k, \
            dico_gamma_players_t_k \
                = balanced_player_game_t(arr_pl_M_T_K_vars_modif.copy(), t, k, 
                           pi_hp_plus, pi_hp_minus, 
                           m_players, t_periods, 
                           manual_debug, dbg=False)
            dico_gamma_players_t[t] = dico_gamma_players_t_k    
            
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
            
            
            ## compute players' utility
        
            arr_pl_M_T_K_vars_modif_new, \
            arr_bg_i_nb_repeat_k, \
            bool_bg_i_min_eq_max, \
            indices_non_playing_players \
                = update_p_i_j_k_by_defined_utility_funtion(
                    arr_pl_M_T_K_vars_modif_new.copy(), 
                    arr_bg_i_nb_repeat_k.copy(),
                    t, k,
                    b0_t_k, c0_t_k,
                    bens_t_k, csts_t_k,
                    pi_hp_minus,
                    pi_0_plus_t_k, pi_0_minus_t_k,
                    nb_repeat_k,
                    learning_rate, 
                    utility_function_version)
            
            if bool_bg_i_min_eq_max and nb_repeat_k != fct_aux.NB_REPEAT_K_MAX:
                k = k
                arr_bg_i_nb_repeat_k[:,nb_repeat_k] \
                    = arr_pl_M_T_K_vars_modif_new[
                        :,
                        t, k,
                        fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]]
                nb_repeat_k += 1
                arr_pl_M_T_K_vars_modif[:,t,k,:] \
                    = arr_pl_M_T_K_vars_modif_new[:,t,k,:].copy()
                    
            elif bool_bg_i_min_eq_max and nb_repeat_k == fct_aux.NB_REPEAT_K_MAX:
                arr_pl_M_T_K_vars_modif_new[
                    indices_non_playing_players, t, k,
                    fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] \
                    = arr_pl_M_T_K_vars_modif_new[
                        indices_non_playing_players, t, k-1,
                        fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] \
                        if k > 0 \
                        else arr_pl_M_T_K_vars_modif_new[
                                indices_non_playing_players, t, k,
                                fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]]
                        
                arr_pl_M_T_K_vars_modif[:,t,k,:] \
                    = arr_pl_M_T_K_vars_modif_new[:,t,k,:].copy()
                
                k = k+1
                nb_repeat_k = 0
                arr_bg_i_nb_repeat_k = np.empty(shape=(m_players, 
                                                       fct_aux.NB_REPEAT_K_MAX)
                                                )
                arr_bg_i_nb_repeat_k.fill(np.nan)
            
            else:
                arr_pl_M_T_K_vars_modif[:,t,k,:] \
                    = arr_pl_M_T_K_vars_modif_new[:,t,k,:].copy()
                
                k = k+1
                nb_repeat_k = 0
                arr_bg_i_nb_repeat_k = np.empty(shape=(m_players, 
                                                       fct_aux.NB_REPEAT_K_MAX)
                                                )
                arr_bg_i_nb_repeat_k.fill(np.nan)
       
        
        dico_stats_res[t] = dico_gamma_players_t
    # __________        compute prices variables         ______________________
    ## B_is, C_is of shape (M_PLAYERS, )
    prod_M_T = arr_pl_M_T_K_vars_modif[:,:, k_steps-1, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
    cons_M_T = arr_pl_M_T_K_vars_modif[:,:, k_steps-1, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
    B_is_M = np.sum(b0_s_T_K[:,k_steps-1] * prod_M_T, axis=1)
    C_is_M = np.sum(c0_s_T_K[:,k_steps-1] * cons_M_T, axis=1)
    
    ## BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is = np.sum(arr_pl_M_T_K_vars_modif[
                        :, :,
                        k_steps-1, fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]], 
                     axis=1)
    PROD_is = np.sum(arr_pl_M_T_K_vars_modif[
                        :, :, 
                        k_steps-1, fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]], 
                     axis=1)
    BB_is_M = pi_sg_plus_T_K[-1,-1] * PROD_is #np.sum(PROD_is)
    CC_is_M = pi_sg_minus_T_K[-1,-1] * CONS_is #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * t_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * t_periods, dtype=object)
    
    #__________      save computed variables locally      _____________________
    algo_name = "LRI1" if utility_function_version == 1 else "LRI2"
    fct_aux.save_variables(path_to_save, arr_pl_M_T_K_vars_modif, 
            b0_s_T_K, c0_s_T_K, B_is_M, C_is_M, BENs_M_T_K, CSTs_M_T_K, 
            BB_is_M, CC_is_M, RU_is_M, 
            pi_sg_minus_T_K, pi_sg_plus_T_K, 
            pi_0_minus_T_K, pi_0_plus_T_K,
            pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
            algo=algo_name)
    
    return arr_pl_M_T_K_vars_modif

# ______________       main function of LRI   ---> fin        _________________

###############################################################################
#                   definition  des unittests
#
###############################################################################

def test_lri_balanced_player_game():
    # steps of learning
    k_steps = 5 # 250
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    pi_hp_plus = 0.2*pow(10,-3)
    pi_hp_minus = 0.33
    learning_rate = 0.1
    utility_function_version=2
    
    manual_debug=True
    
    t_periods = 2
    set1_m_players, set2_m_players = 20, 12
    set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    #set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    set1_states, set2_states = None, None
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance(
                                    set1_m_players, set2_m_players, 
                                   t_periods, 
                                   set1_states, 
                                   set2_states,
                                   set1_stateId0_m_players,
                                   set2_stateId0_m_players, 
                                   path_to_arr_pl_M_T, used_instances)
    # return arr_pl_M_T_vars_init
    
    arr_pl_M_T_K_vars_modif = lri_balanced_player_game(arr_pl_M_T_vars_init,
                             pi_hp_plus=pi_hp_plus, 
                             pi_hp_minus=pi_hp_minus,
                             k_steps=k_steps, 
                             learning_rate=learning_rate,
                             p_i_j_ks=p_i_j_ks,
                             utility_function_version=utility_function_version,
                             path_to_save="tests", 
                             manual_debug=manual_debug, 
                             dbg=False)
    return arr_pl_M_T_K_vars_modif
    
    
###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    arr_pl_M_T_K_vars = test_lri_balanced_player_game()
    
    print("runtime = {}".format(time.time() - ti))