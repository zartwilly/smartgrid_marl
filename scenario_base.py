# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:36:18 2020

@author: jwehounou

scenario base

load values of variables for each player and then 
* play determinist algo
* play lri 1, 2
"""
import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd
import itertools as it
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import game_model_period_T as gmpT
import deterministic_game_model as detGameModel
import force_brute_game_model as bfGameModel
import lri_game_model as lriGameModel
import visu_bkh_scenBase as vizScenBase

from datetime import datetime
from pathlib import Path

def create_scenario_base(m_players=4,num_periods=2):
    """
    for each player, give its initial values of variables. 

    Returns
    -------
    None.

    """
    arr_pl_M_T_vars = np.empty(shape=(m_players,num_periods, 
                                      len(fct_aux.INDEX_ATTRS)))
    arr_pl_M_T_vars.fill(np.nan)
    
    Pis = np.array([[2,2,3,3],[3,4,4,3]],dtype=object); 
    Cis = np.array([[4,4,2,2],[5,6,6,5]],dtype=object); 
    Si_maxs = np.array([[3.2,3.2,4,4],[3.2,3.2,4,4]], dtype=object); 
    Sis = np.array([[1,1,2,2],[0.5,2,1,2]], dtype=object); 
    gamma_is = np.array([[2,2,2,2],[2,2,2,2]], dtype=object)
    
    # players: a_0, b_1, c_2, d_3
    arr_pl_M_T_vars[0,0,fct_aux.INDEX_ATTRS["Pi"]]
    
    for t in range(0, num_periods):
        for m in range(0, m_players):
            arr_pl_M_T_vars[m,t,fct_aux.INDEX_ATTRS["Pi"]] = Pis[t,m]
            arr_pl_M_T_vars[m,t,fct_aux.INDEX_ATTRS["Ci"]] = Cis[t,m]
            arr_pl_M_T_vars[m,t,fct_aux.INDEX_ATTRS["Si"]] = Sis[t,m]
            arr_pl_M_T_vars[m,t,fct_aux.INDEX_ATTRS["Si_max"]] = Si_maxs[t,m]
            arr_pl_M_T_vars[m,t,fct_aux.INDEX_ATTRS["gamma_i"]] = gamma_is[t,m]
            
    return arr_pl_M_T_vars
            
#______________________________________________________________________________
#
#                   ALGO DETERMINIST 
#______________________________________________________________________________

def get_mode_i_from_variables(arr_pl_M_T_vars, num_pl_i, t, state_i):
    """
    determine the mode of player pl_i from some conditions written below.

    Parameters
    ----------
    arr_pl_M_T_vars : TYPE
        DESCRIPTION.
    num_pl_i : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    state_i : TYPE
        DESCRIPTION.

    Returns
    -------
    mode_i : TYPE
        DESCRIPTION.

    """
    
    num_periods = arr_pl_M_T_vars.shape[1]
    
    Pi_t_plus_1 = arr_pl_M_T_vars[num_pl_i, 
                                          t+1, 
                                          fct_aux.INDEX_ATTRS["Pi"]] \
                            if t+1 < num_periods \
                            else 0
    Ci_t_plus_1 = arr_pl_M_T_vars[num_pl_i, 
                             t+1, 
                             fct_aux.INDEX_ATTRS["Ci"]] \
                    if t+1 < num_periods \
                    else 0
    Si_t_minus_1_minus = arr_pl_M_T_vars[num_pl_i, 
                                  t-1, 
                                  fct_aux.INDEX_ATTRS["Si_minus"]] \
                    if t-1 > 0 \
                    else 0
    Si_t_minus_1_plus = arr_pl_M_T_vars[num_pl_i, 
                             t-1, 
                             fct_aux.INDEX_ATTRS["Si_plus"]] \
                    if t-1 > 0 \
                    else 0
    
    mode_i = None              
    if state_i == fct_aux.STATES[0] \
        and fct_aux.fct_positive(
                Ci_t_plus_1, 
                Pi_t_plus_1) <= Si_t_minus_1_minus:
        mode_i = fct_aux.STATE1_STRATS[0]           # CONS+, state1
    elif state_i == fct_aux.STATES[0] \
        and fct_aux.fct_positive(
                Ci_t_plus_1, 
                Pi_t_plus_1) > Si_t_minus_1_minus:
        mode_i = fct_aux.STATE1_STRATS[1]           # CONS-, state1
    elif state_i == fct_aux.STATES[1] \
        and fct_aux.fct_positive(
                Ci_t_plus_1, 
                Pi_t_plus_1) <= Si_t_minus_1_minus:
        mode_i = fct_aux.STATE2_STRATS[1]           # CONS-, state2
    elif state_i == fct_aux.STATES[1] \
        and fct_aux.fct_positive(
                Ci_t_plus_1, 
                Pi_t_plus_1) > Si_t_minus_1_minus:
        mode_i = fct_aux.STATE2_STRATS[0]           # DIS, state2
    elif state_i == fct_aux.STATES[2] \
        and fct_aux.fct_positive(
                Ci_t_plus_1, 
                Pi_t_plus_1) <= Si_t_minus_1_minus:
        mode_i = fct_aux.STATE3_STRATS[0]           # DIS, state3
    elif state_i == fct_aux.STATES[2] \
        and fct_aux.fct_positive(
                Ci_t_plus_1, 
                Pi_t_plus_1) > Si_t_minus_1_minus:
        mode_i = fct_aux.STATE3_STRATS[1]           # PROD, state3
    return mode_i

def scenario_base_det(arr_pl_M_T_vars_init, pi_hp_plus=0.0002, 
                             pi_hp_minus=0.33,
                             m_players=4, 
                             num_periods=2, 
                             k_steps=5,
                             prob_Ci=0.3, 
                             learning_rate=0.01,
                             probs_modes_states=[0.5, 0.5, 0.5],
                             scenario="scenario_base",
                             random_determinist=False,
                             utility_function_version=1,
                             path_to_save="tests", 
                             manual_debug=False,
                             dbg=False):
    
    # pi_sg_{plus, minus}, pi_0_{plus, minus}
    #pi_sg_plus_t, pi_sg_minus_t = 5, 4
    #pi_0_plus_t, pi_0_minus_t = 3, 1
    
    pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K
    pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K
    pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K
    pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K
    
    # constances 
    t = 0
    cpt_error_gamma = 0; cpt_balanced = 0;
    dico_state_mode_i = {}; dico_balanced_pl_i = {}
    
    # add new columns to arr_pl_M_T
    fct_aux.INDEX_ATTRS["Si_minus"] = 16
    fct_aux.INDEX_ATTRS["Si_plus"] = 17
    nb_vars_2_add = 2
    arr_pl_M_T_vars = np.zeros((arr_pl_M_T_vars_init.shape[0],
                                arr_pl_M_T_vars_init.shape[1],
                                arr_pl_M_T_vars_init.shape[2]+nb_vars_2_add), 
                               dtype=object)
    arr_pl_M_T_vars[:,:,:-nb_vars_2_add] = arr_pl_M_T_vars_init
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_plus"]] = np.nan
    
    # balanced player game at instant t
    for num_pl_i in range(0, arr_pl_M_T_vars.shape[0]):
        Ci = round(arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Ci"]],
                   fct_aux.N_DECIMALS)
        Pi = round(arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Pi"]],
                   fct_aux.N_DECIMALS)
        Si = round(arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Si"]], 
                   fct_aux.N_DECIMALS)
        Si_max = round(arr_pl_M_T_vars[num_pl_i, t, 
                                  fct_aux.INDEX_ATTRS["Si_max"]],
                       fct_aux.N_DECIMALS)
        gamma_i = round(arr_pl_M_T_vars[num_pl_i, t, 
                                  fct_aux.INDEX_ATTRS["gamma_i"]],
                       fct_aux.N_DECIMALS)
        prod_i, cons_i, r_i, state_i = 0, 0, 0, ""
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                            prod_i, cons_i, r_i, state_i)
        
        # get state_i and update R_i_old
        pl_i.set_R_i_old(Si_max-Si)
        state_i = pl_i.find_out_state_i()
        
        # get mode_i
        if random_determinist:
            pl_i.select_mode_i()
        else:
            mode_i = get_mode_i_from_variables(
                        arr_pl_M_T_vars, num_pl_i, t, state_i)
            
            pl_i.set_mode_i(mode_i)
        
        # update prod, cons and r_i
        pl_i.update_prod_cons_r_i()
        
        # balancing
        print("state_i={}, mode_i={}".format(pl_i.get_state_i(), pl_i.get_mode_i()))
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        
        # select storage politic with gamme_i
        pl_i.set_gamma_i(gamma_i)
        
        # update columns in arr_pl_M_T
        columns = ["prod_i", "cons_i", "gamma_i", "r_i", "Si", "Si_minus", 
                   "Si_plus", "Si_old", "state_i", "mode_i", "balanced_pl_i", 
                   "formule"]
        values = [pl_i.get_prod_i(), pl_i.get_cons_i(), pl_i.get_gamma_i(), 
                  pl_i.get_r_i(), pl_i.get_Si(), pl_i.get_Si_minus(), 
                  pl_i.get_Si_plus(), pl_i.get_Si_old(), pl_i.get_state_i(),
                  pl_i.get_mode_i(), boolean, formule]
        for col,val in zip(columns, values):
            arr_pl_M_T_vars[num_pl_i, t, 
                        fct_aux.INDEX_ATTRS[col]] = val
    
    # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
    # from a pricing model in the document
    pi_sg_plus_t_new, pi_sg_minus_t_new = fct_aux.determine_new_pricing_sg(
                                            arr_pl_M_T_vars, 
                                            pi_hp_plus, 
                                            pi_hp_minus, 
                                            t, dbg=dbg)
    print("t={}, pi_sg_plus_t_new={}, pi_sg_minus_t_new={}".format(
        t, pi_sg_plus_t_new, pi_sg_minus_t_new))
    
    pi_sg_plus_t = pi_sg_plus_t if pi_sg_plus_t_new is np.nan \
                                else pi_sg_plus_t_new
    pi_sg_minus_t = pi_sg_minus_t if pi_sg_minus_t_new is np.nan \
                                else pi_sg_minus_t_new
    pi_0_plus_t = round(pi_sg_minus_t*pi_hp_plus/pi_hp_minus, 
                            fct_aux.N_DECIMALS)
    pi_0_minus_t = pi_sg_minus_t
    
    if manual_debug:
        pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
        pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
        pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #2 
        pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
    
    ## ______ compute prices inside smart grids _______
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_M_T_vars, t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t, c0_t = fct_aux.compute_energy_unit_price(
                    pi_0_plus_t, pi_0_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg)
    
    # compute ben, cst of shape (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_vars[:, t, fct_aux.INDEX_ATTRS["gamma_i"]]
    bens_t, csts_t = fct_aux.compute_utility_players(arr_pl_M_T_vars, 
                                              gamma_is, 
                                              t, 
                                              b0_t, 
                                              c0_t)
    # B_is, C_is of shape (M_PLAYERS, )
    prod_i_t = arr_pl_M_T_vars[:,t, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_t = arr_pl_M_T_vars[:,t, fct_aux.INDEX_ATTRS["cons_i"]]
    print("prod_i={}, cons_t={}, b0_t={}, c0_t={}".format(prod_i_t, cons_i_t, b0_t, c0_t))
    B_is = b0_t * prod_i_t
    C_is = c0_t * cons_i_t
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is = arr_pl_M_T_vars[:,t, fct_aux.INDEX_ATTRS["cons_i"]]
    PROD_is = arr_pl_M_T_vars[:,t, fct_aux.INDEX_ATTRS["prod_i"]]
    BB_is = pi_sg_plus_t * PROD_is #np.sum(PROD_is)
    print("BB_is={}, pi_sg_plus_t={}".format(BB_is, pi_sg_plus_t))
    for num_pl, bb_i in enumerate(BB_is):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    CC_is = pi_sg_minus_t * CONS_is #np.sum(CONS_is)
    RU_is = BB_is - CC_is
    
    # save variables to scenario_base directory
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    algo_name = "RD-DETERMINIST" if random_determinist else "DETERMINIST"
    path_to_save = os.path.join(path_to_save, scenario, str(prob_Ci), 
                                msg,algo_name)
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars, 
                   b0_t, c0_t, B_is, C_is, 
                   bens_t, csts_t, 
                   BB_is, CC_is, RU_is, 
                   pi_sg_minus_t, pi_sg_plus_t, 
                   pi_0_minus_t, pi_0_plus_t,
                   [pi_hp_plus], [pi_hp_minus], 
                   dico_stats_res={}, 
                   algo=algo_name)
    
    print("pi_sg_plus_t_new={}, pi_sg_minus_t_new={}, pi_0_plus_t_new={}, pi_0_minus_t_new={} \n"\
          .format(pi_sg_plus_t, pi_sg_minus_t, pi_0_plus_t, pi_0_minus_t))
    
    print("determinist game: {}, pi_hp_plus={}, pi_hp_minus ={} ---> end \n"\
          .format(scenario, pi_hp_plus, pi_hp_minus))
        
    return arr_pl_M_T_vars


#______________________________________________________________________________
#
#                   ALGO  BRUTE FORCE 
#______________________________________________________________________________

def scenario_base_bf(arr_pl_M_T_vars_init, pi_hp_plus=0.0002, 
                        pi_hp_minus=0.33,
                        m_players=4, 
                        num_periods=2, 
                        k_steps=5,
                        prob_Ci=0.3, 
                        learning_rate=0.01,
                        probs_modes_states=[0.5, 0.5, 0.5],
                        scenario="scenario_base",
                        algo_name="BEST-BRUTE-FORCE",
                        path_to_save="tests", 
                        manual_debug=False, 
                        dbg=False):
    
    # pi_sg_{plus, minus}, pi_0_{plus, minus}
    # pi_sg_plus_t, pi_sg_minus_t = 5, 4
    # pi_0_plus_t, pi_0_minus_t = 3, 1
    pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K
    pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K
    pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K
    pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K
    
    # constances 
    t = 0
    cpt_error_gamma = 0; cpt_balanced = 0;
    dico_state_mode_i = {}; dico_balanced_pl_i = {}
    
    # add new columns to arr_pl_M_T
    fct_aux.INDEX_ATTRS["Si_minus"] = 16
    fct_aux.INDEX_ATTRS["Si_plus"] = 17
    nb_vars_2_add = 2
    arr_pl_M_T_vars = np.zeros((arr_pl_M_T_vars_init.shape[0],
                                arr_pl_M_T_vars_init.shape[1],
                                arr_pl_M_T_vars_init.shape[2]+nb_vars_2_add), 
                               dtype=object)
    arr_pl_M_T_vars[:,:,:-nb_vars_2_add] = arr_pl_M_T_vars_init
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_plus"]] = np.nan
    
    # _____   balanced players at t   _______
    arr_pl_M_T_vars, possibles_modes = bfGameModel.reupdate_state_players(
                                        arr_pl_M_T_vars.copy(), 0, 0)
    
    mode_profiles = it.product(*possibles_modes)
        
    dico_mode_profs = dict()
    cpt_xxx = 0
    
    for mode_profile in mode_profiles:
        dico_balanced_pl_i_mode_prof, cpt_balanced_mode_prof = dict(), 0
        dico_state_mode_i_mode_prof = dict()
        
        arr_pl_M_T_vars_mode_prof, \
        dico_balanced_pl_i_mode_prof, \
        dico_state_mode_i_mode_prof, \
        cpt_balanced_mode_prof \
            = bfGameModel.balanced_player_game_4_mode_profil(
                arr_pl_M_T_vars.copy(),
                mode_profile, t,
                pi_sg_plus_t, pi_sg_minus_t, 
                pi_hp_plus, pi_hp_minus,
                dico_balanced_pl_i_mode_prof, 
                dico_state_mode_i_mode_prof,
                cpt_balanced_mode_prof,
                m_players, num_periods, 
                manual_debug
                )
        
        # compute In_sg, Out_sg
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                            arr_pl_M_T_vars_mode_prof, 
                            t)
        diff_In_Out_sg = In_sg-Out_sg
        if diff_In_Out_sg in dico_mode_profs:
            dico_mode_profs[diff_In_Out_sg].append(mode_profile)
        else:
            dico_mode_profs[diff_In_Out_sg] = [mode_profile]
         
        cpt_xxx += 1
    
    # max_min_moy_bf
    best_key_In_Out_sg = None
    if algo_name == fct_aux.ALGO_NAMES_BF[0]:              # BEST-BRUTE-FORCE
        best_key_In_Out_sg = min(dico_mode_profs.keys())
    elif algo_name == fct_aux.ALGO_NAMES_BF[1]:            # BAD-BRUTE-FORCE
        best_key_In_Out_sg = max(dico_mode_profs.keys())
    elif algo_name == fct_aux.ALGO_NAMES_BF[2]:            # MIDDLE-BRUTE-FORCE
        mean_key_In_Out_sg  = np.mean(list(dico_mode_profs.keys()))
        if mean_key_In_Out_sg in dico_mode_profs.keys():
            best_key_In_Out_sg = mean_key_In_Out_sg
        else:
            sorted_keys = sorted(dico_mode_profs.keys())
            boolean = True; i_key = 1
            while boolean:
                if sorted_keys[i_key] <= mean_key_In_Out_sg:
                    i_key += 1
                else:
                    boolean = False; i_key -= 1
            best_key_In_Out_sg = sorted_keys[i_key]
                
    # find the best, bad, middle key in dico_mode_profs and 
    # the best, bad, middle mode_profile
    best_mode_profiles = dico_mode_profs[best_key_In_Out_sg]
    best_mode_profile = None
    if len(best_mode_profiles) == 1:
        best_mode_profile = best_mode_profiles[0]
    else:
        rd = np.random.randint(0, len(best_mode_profiles))
        best_mode_profile = best_mode_profiles[rd]
    
    print("cpt_xxx={}, best_key_In_Out_sg={}, best_mode_profile={}".format(
            cpt_xxx, best_key_In_Out_sg, best_mode_profile))
    
    arr_pl_M_T_vars_mode_prof_best, \
    dico_balanced_pl_i, dico_state_mode_i, \
    cpt_balanced \
        = bfGameModel.balanced_player_game_4_mode_profil(
            arr_pl_M_T_vars.copy(),
            best_mode_profile, t,
            pi_sg_plus_t, pi_sg_minus_t, 
            pi_hp_plus, pi_hp_minus,
            dico_balanced_pl_i, dico_state_mode_i, 
            cpt_balanced,
            m_players, num_periods, manual_debug
            )
    
    dico_stats_res = dict()
    dico_stats_res[t] = (round(cpt_balanced/m_players, fct_aux.N_DECIMALS),
                     round(cpt_error_gamma/m_players, fct_aux.N_DECIMALS), 
                     dico_state_mode_i)
    dico_stats_res[t] = {"balanced": dico_balanced_pl_i, 
                         "gamma_i": dico_state_mode_i}
    
    # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
    # from a pricing model in the document
    pi_sg_plus_t_new, pi_sg_minus_t_new \
        = fct_aux.determine_new_pricing_sg(
                arr_pl_M_T_vars_mode_prof_best.copy(), 
                pi_hp_plus, 
                pi_hp_minus, 
                t, dbg=dbg)
    print("t={}, pi_sg_plus_t_new={}, pi_sg_minus_t_new={}".format(
        t, pi_sg_plus_t_new, pi_sg_minus_t_new))  if dbg else None
    print("t={}, pi_sg_plus_t_new={}, pi_sg_minus_t_new={}".format(
        t, pi_sg_plus_t_new, pi_sg_minus_t_new))                
    pi_sg_plus_t = pi_sg_plus_t if pi_sg_plus_t_new is np.nan \
                                else pi_sg_plus_t_new
    pi_sg_minus_t = pi_sg_minus_t if pi_sg_minus_t_new is np.nan \
                                else pi_sg_minus_t_new
    pi_0_plus_t = round(pi_sg_minus_t*pi_hp_plus/pi_hp_minus, 
                        fct_aux.N_DECIMALS)
    pi_0_minus_t = pi_sg_minus_t
    
    if manual_debug:
        pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
        pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
        pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #2 
        pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
    
    
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                    arr_pl_M_T_vars_mode_prof_best, t)
    diff = np.abs(best_key_In_Out_sg-(In_sg-Out_sg))
    print("best_key==In_sg-Out_sg --> OK (diff={}) \n".format(diff)) \
        if diff < 0.1 \
        else print("best_key==In_sg-Out_sg --> NOK (diff={}) \n".format(diff))
    # compute prices of an energy unit price for cost and benefit players
    b0_t, c0_t = fct_aux.compute_energy_unit_price(
                    pi_0_plus_t, pi_0_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg)

    # compute ben, cst of shape (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is_t = arr_pl_M_T_vars_mode_prof_best[
                    :, t, fct_aux.INDEX_ATTRS["gamma_i"]]
    bens_t, csts_t = fct_aux.compute_utility_players(
                        arr_pl_M_T_vars_mode_prof_best, 
                        gamma_is_t, 
                        t, 
                        b0_t, 
                        c0_t)
    
    arr_pl_M_T_vars[:,t,:] = arr_pl_M_T_vars_mode_prof_best[:,t,:].copy()
    
    # B_is, C_is of shape (M_PLAYERS, )
    prod_i_T = arr_pl_M_T_vars[:,t, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_T = arr_pl_M_T_vars[:,t, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is = b0_t * prod_i_T
    C_is = c0_t * cons_i_T
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is = np.sum(arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["cons_i"]], axis=1)
    PROD_is = np.sum(arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prod_i"]], axis=1)
    BB_is = pi_sg_plus_t * PROD_is #np.sum(PROD_is)
    for num_pl, bb_i in enumerate(BB_is):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    CC_is = pi_sg_minus_t * CONS_is #np.sum(CONS_is)
    RU_is = BB_is - CC_is
    
    
    # save variables to scenario_base directory
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    path_to_save = os.path.join(path_to_save, scenario, str(prob_Ci), 
                                msg,algo_name)
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars, 
                   b0_t, c0_t, B_is, C_is, 
                   bens_t, csts_t, 
                   BB_is, CC_is, RU_is, 
                   pi_sg_minus_t, pi_sg_plus_t, 
                   pi_0_minus_t, pi_0_plus_t,
                   [pi_hp_plus], [pi_hp_minus], 
                   dico_stats_res={}, 
                   algo=algo_name)
    
    return arr_pl_M_T_vars
#______________________________________________________________________________
#
#                   ALGO LRI 
#______________________________________________________________________________
def scenario_base_LRI_OLD_OLD_OLD(arr_pl_M_T,
                             pi_hp_plus=0.0002, 
                             pi_hp_minus=0.33,
                             m_players=4, 
                             num_periods=2, 
                             k_steps=5,
                             prob_Ci=0.3, 
                             learning_rate=0.01,
                             probs_modes_states=[0.5, 0.5, 0.5],
                             scenario="scenario_base",
                             utility_function_version=1,
                             path_to_save="tests", dbg=False):
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_plus_T_K.fill(np.nan)
    pi_sg_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_minus_T_K.fill(np.nan)
    
    pi_0_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_0_plus_T_K.fill(np.nan)
    pi_0_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_0_minus_T_K.fill(np.nan)
    
    b0_s_T_K = np.empty(shape=(num_periods,k_steps)); b0_s_T_K.fill(np.nan)
    c0_s_T_K = np.empty(shape=(num_periods,k_steps)); c0_s_T_K.fill(np.nan)
    BENs_M_T_K = np.empty(shape=(m_players,num_periods,k_steps)); 
    BENs_M_T_K.fill(np.nan)
    CSTs_M_T_K = np.empty(shape=(m_players,num_periods,k_steps)); 
    CSTs_M_T_K.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)); C_is_M.fill(np.nan)
    
    dico_stats_res = dict()
    
    fct_aux.INDEX_ATTRS["prob_mode_state_i"] = 16
    fct_aux.INDEX_ATTRS["u_i"] = 17
    fct_aux.INDEX_ATTRS["bg_i"] = 18
    nb_vars_2_add = 3
    # _______ variables' initialization --> fin   ________________
    
    # constances, pi_sg_{plus, minus}, pi_0_{plus, minus}
    t = 0
    pi_sg_plus_t, pi_sg_minus_t = 5, 4
    pi_0_plus_t, pi_0_minus_t = 3, 1

    # _______   turn arr_pl_M_T in a array of 4 dimensions   ________
    arrs = []
    for k in range(0, k_steps):
        arrs.append(list(arr_pl_M_T))
    arrs = np.array(arrs, dtype=object)
    arrs = np.transpose(arrs, [1,2,0,3])
    
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
    
    arr_pl_M_T_K_vars_modif = arr_pl_M_T_K_vars.copy()

    # ______      run balanced sg for t=0 at any k_step     ________
    nb_repeat_k = 0
    k = 0
    arr_bg_i_nb_repeat_k = np.empty(
                            shape=(m_players,fct_aux.NB_REPEAT_K_MAX))
    arr_bg_i_nb_repeat_k.fill(np.nan)
    pi_sg_plus_t_k, pi_sg_minus_t_k = pi_sg_plus_t, pi_sg_minus_t
    pi_0_plus_t_k, pi_0_minus_t_k = pi_0_plus_t, pi_0_minus_t
    pi_sg_plus_T_K[t,k] = pi_sg_plus_t_k; pi_sg_minus_T_K[t,k] = pi_sg_minus_t_k
    pi_0_plus_T_K[t,k] = pi_0_plus_t_k; pi_0_minus_T_K[t,k] = pi_0_minus_t_k
    while (k < k_steps):
        print("------- t = {}, k = {}, repeat_k = {}, prob_modes={} -------"\
              .format(
                t, k, nb_repeat_k,
                arr_pl_M_T_K_vars_modif[:,t,k,
                        fct_aux.INDEX_ATTRS["prob_mode_state_i"]]))
        print("pi_sg_plus_t_k = {}, pi_sg_minus_t_k={}".format(
                pi_sg_plus_t_k, pi_sg_minus_t_k ))
        
        ## balanced_player_game_t
        arr_pl_M_T_K_vars_modif, \
        b0_t_k, c0_t_k, \
        bens_t_k, csts_t_k, \
        pi_sg_plus_t_k_plus_1, pi_sg_minus_t_k_plus_1, \
        pi_0_plus_t_k_plus_1, pi_0_minus_t_k_plus_1, \
        dico_stats_res_t_k \
            = lriGameModel.balanced_player_game_t(
                    arr_pl_M_T_K_vars_modif, t, k, 
                    pi_hp_plus, pi_hp_minus, 
                    pi_sg_plus_t_k, pi_sg_minus_t_k,
                    m_players, num_periods, nb_repeat_k, dbg=False)
        
        ## update pi_sg_minus_t_k_minus_1 and pi_sg_plus_t_k_minus_1
        pi_sg_minus_t_k = pi_sg_minus_t_k_plus_1
        pi_sg_plus_t_k = pi_sg_plus_t_k_plus_1
            
        ## update variables at each step because they must have to converge in the best case
        #### update pi_sg_plus, pi_sg_minus of shape (NUM_PERIODS,K_STEPS)
        if k<k_steps-1:
            pi_sg_minus_T_K[t,k+1] = pi_sg_minus_t_k_plus_1
            pi_sg_plus_T_K[t,k+1] = pi_sg_plus_t_k_plus_1
            pi_0_minus_T_K[t,k+1] = pi_0_minus_t_k_plus_1
            pi_0_plus_T_K[t,k+1] = pi_0_plus_t_k_plus_1
        #### update b0_s, c0_s of shape (NUM_PERIODS,K_STEPS) 
        b0_s_T_K[t,k] = b0_t_k
        c0_s_T_K[t,k] = c0_t_k
        #### update BENs, CSTs of shape (M_PLAYERS,NUM_PERIODS,K_STEPS)
        #### shape: bens_t_k: (M_PLAYERS,)
        BENs_M_T_K[:,t,k] = bens_t_k
        CSTs_M_T_K[:,t,k] = csts_t_k
        
        ## compute new strategies probabilities by using utility fonction
        print("bens_t_k={}, csts_t_k={}".format(bens_t_k.shape, csts_t_k.shape))
        arr_pl_M_T_K_vars_modif, arr_bg_i_nb_repeat_k, \
        bool_bg_i_min_eq_max, \
        bg_min_i_t_0_to_k, bg_max_i_t_0_to_k \
            = lriGameModel.update_probs_modes_states_by_defined_utility_funtion(
                arr_pl_M_T_K_vars_modif, 
                arr_bg_i_nb_repeat_k,
                t, k,
                b0_t_k, c0_t_k,
                bens_t_k, csts_t_k,
                pi_hp_minus,
                pi_0_plus_t_k, pi_0_minus_t_k,
                m_players,
                learning_rate, 
                utility_function_version)
           
        if bool_bg_i_min_eq_max:
                k = k
                arr_bg_i_nb_repeat_k[:,nb_repeat_k] \
                    = arr_pl_M_T_K_vars_modif[
                        :,
                        t, k,
                        fct_aux.INDEX_ATTRS["bg_i"]]
                nb_repeat_k += 1
                arr_pl_M_T_K_vars_modif[:,t,k,:] = arr_pl_M_T_K_vars[:,t,k,:]
                print("REPEAT t={}, k={}, repeat_k={}, min(bg_i)==max(bg_i)".format(
                        t, k, nb_repeat_k))
        else:
            arr_pl_M_T_K_vars[:,t,k,:] = arr_pl_M_T_K_vars_modif[:,t,k,:]
            
            k = k+1
            nb_repeat_k = 0
            arr_bg_i_nb_repeat_k \
                = np.empty(
                    shape=(m_players,fct_aux.NB_REPEAT_K_MAX)
                    )
            arr_bg_i_nb_repeat_k.fill(np.nan)
        if nb_repeat_k == fct_aux.NB_REPEAT_K_MAX:
            # TODO: A DELETE A THE END OF CODING
            arr_pl_M_T_K_vars[:,t,k,:] = arr_pl_M_T_K_vars_modif[:,t,k,:]
            
            # B_is, C_is
            B_is_M.fill(2.306)
            C_is_M.fill(2.306)
            # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
            BB_is_M = np.empty(shape=(m_players,)); BB_is_M.fill(2.306)
            CC_is_M = np.empty(shape=(m_players,)); CC_is_M.fill(2.306) 
            RU_is_M = np.empty(shape=(m_players,)); RU_is_M.fill(2.306)
            # pi_hp_plus_s, pi_hp_minus_s of shape (NUM_PERIODS,)
            pi_hp_plus_s = np.empty(shape=(num_periods,)); 
            pi_hp_plus_s.fill(2.306)
            pi_hp_minus_s = np.empty(shape=(num_periods,)); 
            pi_hp_minus_s.fill(2.306)
            
            msg = "pi_hp_plus_"+str(pi_hp_plus)\
                       +"_pi_hp_minus_"+str(pi_hp_minus)
            algo_name = "LRI1" if utility_function_version == 1 else "LRI2"
            path_to_save = os.path.join(path_to_save, scenario, str(prob_Ci), 
                                msg, algo_name, str(learning_rate))
            fct_aux.save_variables(
                path_to_save, arr_pl_M_T_K_vars, 
                b0_s_T_K, c0_s_T_K, 
                B_is_M, C_is_M, 
                BENs_M_T_K, CSTs_M_T_K, 
                BB_is_M, CC_is_M, RU_is_M, 
                pi_sg_minus_T_K, pi_sg_plus_T_K, 
                pi_0_minus_T_K, pi_0_plus_T_K,
                pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                algo="LRI")
            return arr_pl_M_T_K_vars
        
    # update pi_sg_plus_t_minus_1 and pi_sg_minus_t_minus_1
    pi_sg_plus_t = pi_sg_plus_T_K[t,k_steps-1]
    pi_sg_minus_t = pi_sg_minus_T_K[t,k_steps-1]
    
    # __________        compute prices variables         ______________________
    ## B_is, C_is of shape (M_PLAYERS, )
    print("cons_is={}".format(arr_pl_M_T_K_vars[
                        :, t,
                        k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]))
    # CONS_is = np.sum(arr_pl_M_T_K_vars[
    #                     :, t,
    #                     k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]], 
    #                  axis=1)
    # PROD_is = np.sum(arr_pl_M_T_K_vars[
    #                     :, t, 
    #                     k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]], 
    #                  axis=1)
    prod_i_t = arr_pl_M_T_K_vars[:,t, k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_t = arr_pl_M_T_K_vars[:,t, k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is_M = b0_s_T_K[t,k_steps-1] * prod_i_t
    C_is_M = c0_s_T_K[t,k_steps-1] * cons_i_t
    ## BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    # BB_is_M = pi_sg_plus_T_K[t,-1] * PROD_is #np.sum(PROD_is)
    BB_is_M = pi_sg_plus_T_K[t,-1] * prod_i_t
    for num_pl, bb_i in enumerate(BB_is_M):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    # CC_is_M = pi_sg_minus_T_K[t,-1] * CONS_is #np.sum(CONS_is)
    CC_is_M = pi_sg_minus_T_K[t,-1] * cons_i_t #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * num_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * num_periods, dtype=object)
    
    #__________      save computed variables locally      _____________________
    msg = "pi_hp_plus_"+str(pi_hp_plus)\
                       +"_pi_hp_minus_"+str(pi_hp_minus)
    algo_name = "LRI1" if utility_function_version == 1 else "LRI2"
    path_to_save = os.path.join(path_to_save, scenario, str(prob_Ci), 
                                msg, algo_name, str(learning_rate))
    fct_aux.save_variables(path_to_save, arr_pl_M_T_K_vars, 
                   b0_s_T_K, c0_s_T_K, B_is_M, C_is_M, BENs_M_T_K, CSTs_M_T_K, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T_K, pi_sg_plus_T_K, 
                   pi_0_minus_T_K, pi_0_plus_T_K,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo="LRI")
    
    print("VARS shapes: b0_s_T_K={}, B_is_M={}, BENs_M_T_K={}, BB_is_M={}, pi_sg_minus_T_K={}, pi_0_minus_T_K={}".format(
        b0_s_T_K.shape, B_is_M.shape, BENs_M_T_K.shape, BB_is_M.shape, 
        pi_sg_minus_T_K.shape, pi_0_minus_T_K.shape))
    print("b0_s_T_K={}".format(b0_s_T_K))
    print("{}: {}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, probs_mode={} ---> fin \n".format(
            algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus, 
            probs_modes_states))
    
    return arr_pl_M_T_K_vars
         
def scenario_base_LRI_OLD_OLD(arr_pl_M_T,
                        pi_hp_plus=0.0002, 
                        pi_hp_minus=0.33,
                        m_players=4, 
                        num_periods=2, 
                        k_steps=5,
                        prob_Ci=0.3, 
                        learning_rate=0.01,
                        probs_modes_states=[0.5, 0.5, 0.5],
                        scenario="scenario_base",
                        utility_function_version=1,
                        path_to_save="tests", dbg=False):
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_plus_T_K.fill(np.nan)
    pi_sg_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_minus_T_K.fill(np.nan)
    
    pi_0_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_0_plus_T_K.fill(np.nan)
    pi_0_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_0_minus_T_K.fill(np.nan)
    
    b0_s_T_K = np.empty(shape=(num_periods,k_steps)); b0_s_T_K.fill(np.nan)
    c0_s_T_K = np.empty(shape=(num_periods,k_steps)); c0_s_T_K.fill(np.nan)
    BENs_M_T_K = np.empty(shape=(m_players,num_periods,k_steps)); 
    BENs_M_T_K.fill(np.nan)
    CSTs_M_T_K = np.empty(shape=(m_players,num_periods,k_steps)); 
    CSTs_M_T_K.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)); C_is_M.fill(np.nan)
    
    dico_stats_res = dict()
    
    fct_aux.INDEX_ATTRS["prob_mode_state_i"] = 16
    fct_aux.INDEX_ATTRS["u_i"] = 17
    fct_aux.INDEX_ATTRS["bg_i"] = 18
    fct_aux.INDEX_ATTRS["non_playing_players"] = 19
    nb_vars_2_add = 4
    # _______ variables' initialization --> fin   ________________
    
    # constances, pi_sg_{plus, minus}, pi_0_{plus, minus}
    t = 0
    pi_sg_plus_t, pi_sg_minus_t = 5, 4
    pi_0_plus_t, pi_0_minus_t = 3, 1

    # _______   turn arr_pl_M_T in a array of 4 dimensions   ________
    arrs = []
    for k in range(0, k_steps):
        arrs.append(list(arr_pl_M_T))
    arrs = np.array(arrs, dtype=object)
    arrs = np.transpose(arrs, [1,2,0,3])
    
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
        = lriGameModel.NON_PLAYING_PLAYERS["PLAY"]
    
    arr_pl_M_T_K_vars_modif = arr_pl_M_T_K_vars.copy()

    # ______      run balanced sg for t=0 at any k_step     ________
    nb_repeat_k = 0
    k = 0
    indices_non_playing_players = []      # indices of non-playing players because bg_min = bg_max
    arr_bg_i_nb_repeat_k = np.empty(
                            shape=(m_players,fct_aux.NB_REPEAT_K_MAX))
    arr_bg_i_nb_repeat_k.fill(np.nan)
    pi_sg_plus_t_k, pi_sg_minus_t_k = pi_sg_plus_t, pi_sg_minus_t
    pi_0_plus_t_k, pi_0_minus_t_k = pi_0_plus_t, pi_0_minus_t
    pi_sg_plus_T_K[t,k] = pi_sg_plus_t_k; pi_sg_minus_T_K[t,k] = pi_sg_minus_t_k
    pi_0_plus_T_K[t,k] = pi_0_plus_t_k; pi_0_minus_T_K[t,k] = pi_0_minus_t_k
    while (k < k_steps):
        print("------- t = {}, k = {}, repeat_k = {}, prob_modes={} -------"\
              .format(
                t, k, nb_repeat_k,
                arr_pl_M_T_K_vars_modif[:,t,k,
                        fct_aux.INDEX_ATTRS["prob_mode_state_i"]]))
        print("pi_sg_plus_t_k = {}, pi_sg_minus_t_k={}".format(
                pi_sg_plus_t_k, pi_sg_minus_t_k ))
        print("indices_non_playing_players={}".format(indices_non_playing_players))
        
        ## balanced_player_game_t
        arr_pl_M_T_K_vars_modif, \
        b0_t_k, c0_t_k, \
        bens_t_k, csts_t_k, \
        pi_sg_plus_t_k_plus_1, pi_sg_minus_t_k_plus_1, \
        pi_0_plus_t_k_plus_1, pi_0_minus_t_k_plus_1, \
        dico_stats_res_t_k \
            = lriGameModel.balanced_player_game_t(
                    arr_pl_M_T_K_vars_modif, t, k, 
                    pi_hp_plus, pi_hp_minus, 
                    pi_sg_plus_t_k, pi_sg_minus_t_k,
                    m_players, indices_non_playing_players,
                    num_periods, nb_repeat_k, dbg=False)
        
        ## update pi_sg_minus_t_k_minus_1 and pi_sg_plus_t_k_minus_1
        pi_sg_minus_t_k = pi_sg_minus_t_k_plus_1
        pi_sg_plus_t_k = pi_sg_plus_t_k_plus_1
            
        ## update variables at each step because they must have to converge in the best case
        #### update pi_sg_plus, pi_sg_minus of shape (NUM_PERIODS,K_STEPS)
        if k<k_steps-1:
            pi_sg_minus_T_K[t,k+1] = pi_sg_minus_t_k_plus_1
            pi_sg_plus_T_K[t,k+1] = pi_sg_plus_t_k_plus_1
            pi_0_minus_T_K[t,k+1] = pi_0_minus_t_k_plus_1
            pi_0_plus_T_K[t,k+1] = pi_0_plus_t_k_plus_1
        #### update b0_s, c0_s of shape (NUM_PERIODS,K_STEPS) 
        b0_s_T_K[t,k] = b0_t_k
        c0_s_T_K[t,k] = c0_t_k
        #### update BENs, CSTs of shape (M_PLAYERS,NUM_PERIODS,K_STEPS)
        #### shape: bens_t_k: (M_PLAYERS,)
        BENs_M_T_K[:,t,k] = bens_t_k
        CSTs_M_T_K[:,t,k] = csts_t_k
        
        ## compute new strategies probabilities by using utility fonction
        print("bens_t_k={}, csts_t_k={}".format(bens_t_k.shape, csts_t_k.shape))
        arr_pl_M_T_K_vars_modif, arr_bg_i_nb_repeat_k, \
        bool_bg_i_min_eq_max, indices_non_playing_players_new,\
        bg_min_i_t_0_to_k, bg_max_i_t_0_to_k \
            = lriGameModel.update_probs_modes_states_by_defined_utility_funtion(
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
           
        print("__bool_bg_i_min_eq_max={}, nb_repeat_k={}__".format(bool_bg_i_min_eq_max, 
                nb_repeat_k ))
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
                        = lriGameModel.NON_PLAYING_PLAYERS["NOT_PLAY"]
            ### update bg_i, mode_i, u_i_t_k, p_i_t_k for not playing players from k+1 to k_steps 
            for var in ["bg_i", "mode_i", "prod_i", "cons_i",
                        "u_i", "prob_mode_state_i", "r_i", 
                        "gamma_i", "state_i"]:
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
            print("p_i_t_k: arr={},arr_modif={}".format(
                arr_pl_M_T_K_vars[:,t,k,fct_aux.INDEX_ATTRS["prob_mode_state_i"]],
                arr_pl_M_T_K_vars_modif[:,t,k,fct_aux.INDEX_ATTRS["prob_mode_state_i"]]))
            
            k = k+1
            nb_repeat_k = 0
            arr_bg_i_nb_repeat_k \
                = np.empty(
                    shape=(m_players, fct_aux.NB_REPEAT_K_MAX)
                    )
            arr_bg_i_nb_repeat_k.fill(np.nan)
            indices_non_playing_players = indices_non_playing_players_new
        
    arr_pl_M_T_K_vars = arr_pl_M_T_K_vars_modif.copy()
    
    #### recompute BENs, CSTs of shape (M_PLAYERS,NUM_PERIODS,K_STEPS)
    BENs_M_T_K[:,t,:] = 1 + b0_s_T_K[t,:] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["prod_i"]] \
                        + arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["gamma_i"]] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["r_i"]] 
    CSTs_M_T_K[:,t,:] =  c0_s_T_K[t,:] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["cons_i"]]
                        
    # update pi_sg_plus_t_minus_1 and pi_sg_minus_t_minus_1
    pi_sg_plus_t = pi_sg_plus_T_K[t,k_steps-1]
    pi_sg_minus_t = pi_sg_minus_T_K[t,k_steps-1]
    
    # __________        compute prices variables         ______________________
    ## B_is, C_is of shape (M_PLAYERS, )
    print("cons_is={}".format(arr_pl_M_T_K_vars[
                        :, t,
                        k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]))
    # CONS_is = np.sum(arr_pl_M_T_K_vars[
    #                     :, t,
    #                     k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]], 
    #                  axis=1)
    # PROD_is = np.sum(arr_pl_M_T_K_vars[
    #                     :, t, 
    #                     k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]], 
    #                  axis=1)
    prod_i_t = arr_pl_M_T_K_vars[:,t, k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_t = arr_pl_M_T_K_vars[:,t, k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is_M = b0_s_T_K[t,k_steps-1] * prod_i_t
    C_is_M = c0_s_T_K[t,k_steps-1] * cons_i_t
    ## BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    # BB_is_M = pi_sg_plus_T_K[t,-1] * PROD_is #np.sum(PROD_is)
    BB_is_M = pi_sg_plus_T_K[t,-1] * prod_i_t
    for num_pl, bb_i in enumerate(BB_is_M):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    # CC_is_M = pi_sg_minus_T_K[t,-1] * CONS_is #np.sum(CONS_is)
    CC_is_M = pi_sg_minus_T_K[t,-1] * cons_i_t #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * num_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * num_periods, dtype=object)
    
    #__________      save computed variables locally      _____________________
    msg = "pi_hp_plus_"+str(pi_hp_plus)\
                       +"_pi_hp_minus_"+str(pi_hp_minus)
    algo_name = "LRI1" if utility_function_version == 1 else "LRI2"
    path_to_save = os.path.join(path_to_save, scenario, str(prob_Ci), 
                                msg, algo_name, str(learning_rate))
    fct_aux.save_variables(path_to_save, arr_pl_M_T_K_vars, 
                   b0_s_T_K, c0_s_T_K, B_is_M, C_is_M, BENs_M_T_K, CSTs_M_T_K, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T_K, pi_sg_plus_T_K, 
                   pi_0_minus_T_K, pi_0_plus_T_K,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo="LRI")
    
    print("VARS shapes: b0_s_T_K={}, B_is_M={}, BENs_M_T_K={}, BB_is_M={}, pi_sg_minus_T_K={}, pi_0_minus_T_K={}".format(
        b0_s_T_K.shape, B_is_M.shape, BENs_M_T_K.shape, BB_is_M.shape, 
        pi_sg_minus_T_K.shape, pi_0_minus_T_K.shape))
    print("b0_s_T_K={}".format(b0_s_T_K))
    print("{}: {}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, probs_mode={} ---> fin \n".format(
            algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus, 
            probs_modes_states))
    
    return arr_pl_M_T_K_vars

def scenario_base_LRI_OLD(arr_pl_M_T,
                        pi_hp_plus=0.0002, 
                        pi_hp_minus=0.33,
                        m_players=4, 
                        num_periods=2, 
                        k_steps=5,
                        prob_Ci=0.3, 
                        learning_rate=0.01,
                        probs_modes_states=[0.5, 0.5, 0.5],
                        scenario="scenario_base",
                        utility_function_version=1,
                        path_to_save="tests", dbg=False):
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_plus_T_K.fill(np.nan)
    pi_sg_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_minus_T_K.fill(np.nan)
    
    pi_0_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_0_plus_T_K.fill(np.nan)
    pi_0_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_0_minus_T_K.fill(np.nan)
    
    b0_s_T_K = np.empty(shape=(num_periods,k_steps)); b0_s_T_K.fill(np.nan)
    c0_s_T_K = np.empty(shape=(num_periods,k_steps)); c0_s_T_K.fill(np.nan)
    BENs_M_T_K = np.empty(shape=(m_players,num_periods,k_steps)); 
    BENs_M_T_K.fill(np.nan)
    CSTs_M_T_K = np.empty(shape=(m_players,num_periods,k_steps)); 
    CSTs_M_T_K.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)); C_is_M.fill(np.nan)
    
    dico_stats_res = dict()
    
    fct_aux.INDEX_ATTRS["Si_minus"] = 16
    fct_aux.INDEX_ATTRS["Si_plus"] = 17
    fct_aux.INDEX_ATTRS["prob_mode_state_i"] = 18
    fct_aux.INDEX_ATTRS["u_i"] = 19
    fct_aux.INDEX_ATTRS["bg_i"] = 20
    fct_aux.INDEX_ATTRS["non_playing_players"] = 21
    nb_vars_2_add = 6
    # _______ variables' initialization --> fin   ________________
    
    # constances, pi_sg_{plus, minus}, pi_0_{plus, minus}
    t = 0
    pi_sg_plus_t, pi_sg_minus_t = 5, 4
    pi_0_plus_t, pi_0_minus_t = 3, 1

    # _______   turn arr_pl_M_T in a array of 4 dimensions   ________
    arrs = []
    for k in range(0, k_steps):
        arrs.append(list(arr_pl_M_T))
    arrs = np.array(arrs, dtype=object)
    arrs = np.transpose(arrs, [1,2,0,3])
    
    ## add initial values for the new attributs
    arr_pl_M_T_K_vars = np.zeros((arrs.shape[0],
                             arrs.shape[1],
                             arrs.shape[2],
                             arrs.shape[3]+nb_vars_2_add), 
                            dtype=object)
    arr_pl_M_T_K_vars[:,:,:,:-nb_vars_2_add] = arrs
    arr_pl_M_T_K_vars[:,:,:,fct_aux.INDEX_ATTRS["Si_minus"]] = 0
    arr_pl_M_T_K_vars[:,:,:,fct_aux.INDEX_ATTRS["Si_plus"]] = 0
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["u_i"]] = np.nan
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["bg_i"]] = np.nan
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["prob_mode_state_i"]] = 0.5
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["non_playing_players"]] \
        = lriGameModel.NON_PLAYING_PLAYERS["PLAY"]
    
    arr_pl_M_T_K_vars_modif = arr_pl_M_T_K_vars.copy()

    # ______      run balanced sg for t=0 at any k_step     ________
    nb_repeat_k = 0
    k = 0
    indices_non_playing_players = []      # indices of non-playing players because bg_min = bg_max
    arr_bg_i_nb_repeat_k = np.empty(
                            shape=(m_players,fct_aux.NB_REPEAT_K_MAX))
    arr_bg_i_nb_repeat_k.fill(np.nan)
    pi_sg_plus_t_k, pi_sg_minus_t_k = pi_sg_plus_t, pi_sg_minus_t
    pi_0_plus_t_k, pi_0_minus_t_k = pi_0_plus_t, pi_0_minus_t
    pi_sg_plus_T_K[t,k] = pi_sg_plus_t_k; pi_sg_minus_T_K[t,k] = pi_sg_minus_t_k
    pi_0_plus_T_K[t,k] = pi_0_plus_t_k; pi_0_minus_T_K[t,k] = pi_0_minus_t_k
    while (k < k_steps):
        print("------- t = {}, k = {}, repeat_k = {}, prob_modes={} -------"\
              .format(
                t, k, nb_repeat_k,
                arr_pl_M_T_K_vars_modif[:,t,k,
                        fct_aux.INDEX_ATTRS["prob_mode_state_i"]]))
        print("pi_sg_plus_t_k = {}, pi_sg_minus_t_k={}".format(
                pi_sg_plus_t_k, pi_sg_minus_t_k ))
        print("indices_non_playing_players={}".format(indices_non_playing_players))
        
        ## balanced_player_game_t
        arr_pl_M_T_K_vars_modif, \
        b0_t_k, c0_t_k, \
        bens_t_k, csts_t_k, \
        pi_sg_plus_t_k_plus_1, pi_sg_minus_t_k_plus_1, \
        pi_0_plus_t_k_plus_1, pi_0_minus_t_k_plus_1, \
        dico_stats_res_t_k \
            = lriGameModel.balanced_player_game_t(
                    arr_pl_M_T_K_vars_modif, t, k, 
                    pi_hp_plus, pi_hp_minus, 
                    pi_sg_plus_t_k, pi_sg_minus_t_k,
                    m_players, indices_non_playing_players,
                    num_periods, nb_repeat_k, dbg=False)
        
        ## update pi_sg_minus_t_k_minus_1 and pi_sg_plus_t_k_minus_1
        pi_sg_minus_t_k = pi_sg_minus_t_k_plus_1
        pi_sg_plus_t_k = pi_sg_plus_t_k_plus_1
            
        ## update variables at each step because they must have to converge in the best case
        #### update pi_sg_plus, pi_sg_minus of shape (NUM_PERIODS,K_STEPS)
        if k<k_steps-1:
            pi_sg_minus_T_K[t,k+1] = pi_sg_minus_t_k_plus_1
            pi_sg_plus_T_K[t,k+1] = pi_sg_plus_t_k_plus_1
            pi_0_minus_T_K[t,k+1] = pi_0_minus_t_k_plus_1
            pi_0_plus_T_K[t,k+1] = pi_0_plus_t_k_plus_1
        #### update b0_s, c0_s of shape (NUM_PERIODS,K_STEPS) 
        b0_s_T_K[t,k] = b0_t_k
        c0_s_T_K[t,k] = c0_t_k
        #### update BENs, CSTs of shape (M_PLAYERS,NUM_PERIODS,K_STEPS)
        #### shape: bens_t_k: (M_PLAYERS,)
        BENs_M_T_K[:,t,k] = bens_t_k
        CSTs_M_T_K[:,t,k] = csts_t_k
        
        ## compute new strategies probabilities by using utility fonction
        print("bens_t_k={}, csts_t_k={}".format(bens_t_k.shape, csts_t_k.shape))
        arr_pl_M_T_K_vars_modif, arr_bg_i_nb_repeat_k, \
        bool_bg_i_min_eq_max, indices_non_playing_players_new,\
        bg_min_i_t_0_to_k, bg_max_i_t_0_to_k \
            = lriGameModel.update_probs_modes_states_by_defined_utility_funtion(
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
           
        print("__bool_bg_i_min_eq_max={}, nb_repeat_k={}__".format(bool_bg_i_min_eq_max, 
                nb_repeat_k ))
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
                        = lriGameModel.NON_PLAYING_PLAYERS["NOT_PLAY"]
            ### update bg_i, mode_i, u_i_t_k, p_i_t_k for not playing players from k+1 to k_steps 
            for var in ["bg_i", "mode_i", "prod_i", "cons_i",
                        "u_i", "prob_mode_state_i", "r_i", 
                        "gamma_i", "state_i"]:
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
            print("p_i_t_k: arr={},arr_modif={}".format(
                arr_pl_M_T_K_vars[:,t,k,fct_aux.INDEX_ATTRS["prob_mode_state_i"]],
                arr_pl_M_T_K_vars_modif[:,t,k,fct_aux.INDEX_ATTRS["prob_mode_state_i"]]))
            
            k = k+1
            nb_repeat_k = 0
            arr_bg_i_nb_repeat_k \
                = np.empty(
                    shape=(m_players, fct_aux.NB_REPEAT_K_MAX)
                    )
            arr_bg_i_nb_repeat_k.fill(np.nan)
            indices_non_playing_players = indices_non_playing_players_new
        
    arr_pl_M_T_K_vars = arr_pl_M_T_K_vars_modif.copy()
    
    #### recompute BENs, CSTs of shape (M_PLAYERS,NUM_PERIODS,K_STEPS)
    BENs_M_T_K[:,t,:] = 1 + b0_s_T_K[t,:] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["prod_i"]] \
                        + arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["gamma_i"]] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["r_i"]] 
    CSTs_M_T_K[:,t,:] =  c0_s_T_K[t,:] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["cons_i"]]
                        
    # update pi_sg_plus_t_minus_1 and pi_sg_minus_t_minus_1
    pi_sg_plus_t = pi_sg_plus_T_K[t,k_steps-1]
    pi_sg_minus_t = pi_sg_minus_T_K[t,k_steps-1]
    
    # __________        compute prices variables         ______________________
    ## B_is, C_is of shape (M_PLAYERS, )
    print("cons_is={}".format(arr_pl_M_T_K_vars[
                        :, t,
                        k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]))
    # CONS_is = np.sum(arr_pl_M_T_K_vars[
    #                     :, t,
    #                     k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]], 
    #                  axis=1)
    # PROD_is = np.sum(arr_pl_M_T_K_vars[
    #                     :, t, 
    #                     k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]], 
    #                  axis=1)
    prod_i_t = arr_pl_M_T_K_vars[:,t, k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_t = arr_pl_M_T_K_vars[:,t, k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is_M = b0_s_T_K[t,k_steps-1] * prod_i_t
    C_is_M = c0_s_T_K[t,k_steps-1] * cons_i_t
    ## BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    # BB_is_M = pi_sg_plus_T_K[t,-1] * PROD_is #np.sum(PROD_is)
    BB_is_M = pi_sg_plus_T_K[t,-1] * prod_i_t
    for num_pl, bb_i in enumerate(BB_is_M):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    # CC_is_M = pi_sg_minus_T_K[t,-1] * CONS_is #np.sum(CONS_is)
    CC_is_M = pi_sg_minus_T_K[t,-1] * cons_i_t #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * num_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * num_periods, dtype=object)
    
    #__________      save computed variables locally      _____________________
    msg = "pi_hp_plus_"+str(pi_hp_plus)\
                       +"_pi_hp_minus_"+str(pi_hp_minus)
    algo_name = "LRI1" if utility_function_version == 1 else "LRI2"
    path_to_save = os.path.join(path_to_save, scenario, str(prob_Ci), 
                                msg, algo_name, str(learning_rate))
    fct_aux.save_variables(path_to_save, arr_pl_M_T_K_vars, 
                   b0_s_T_K, c0_s_T_K, B_is_M, C_is_M, BENs_M_T_K, CSTs_M_T_K, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T_K, pi_sg_plus_T_K, 
                   pi_0_minus_T_K, pi_0_plus_T_K,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo="LRI")
    
    print("VARS shapes: b0_s_T_K={}, B_is_M={}, BENs_M_T_K={}, BB_is_M={}, pi_sg_minus_T_K={}, pi_0_minus_T_K={}".format(
        b0_s_T_K.shape, B_is_M.shape, BENs_M_T_K.shape, BB_is_M.shape, 
        pi_sg_minus_T_K.shape, pi_0_minus_T_K.shape))
    print("b0_s_T_K={}".format(b0_s_T_K))
    print("{}: {}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, probs_mode={} ---> fin \n".format(
            algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus, 
            probs_modes_states))
    
    return arr_pl_M_T_K_vars

def scenario_base_LRI(arr_pl_M_T,
                        pi_hp_plus=0.0002, 
                        pi_hp_minus=0.33,
                        m_players=4, 
                        num_periods=2, 
                        k_steps=5,
                        prob_Ci=0.3, 
                        learning_rate=0.01,
                        probs_modes_states=[0.5, 0.5, 0.5],
                        scenario="scenario_base",
                        utility_function_version=1,
                        path_to_save="tests", 
                        manual_debug=False, 
                        dbg=False):
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_plus_T_K.fill(np.nan)
    pi_sg_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_sg_minus_T_K.fill(np.nan)
    
    pi_0_plus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_0_plus_T_K.fill(np.nan)
    pi_0_minus_T_K = np.empty(shape=(num_periods,k_steps)); 
    pi_0_minus_T_K.fill(np.nan)
    
    b0_s_T_K = np.empty(shape=(num_periods,k_steps)); b0_s_T_K.fill(np.nan)
    c0_s_T_K = np.empty(shape=(num_periods,k_steps)); c0_s_T_K.fill(np.nan)
    BENs_M_T_K = np.empty(shape=(m_players,num_periods,k_steps)); 
    BENs_M_T_K.fill(np.nan)
    CSTs_M_T_K = np.empty(shape=(m_players,num_periods,k_steps)); 
    CSTs_M_T_K.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)); C_is_M.fill(np.nan)
    
    dico_stats_res = dict()
    
    fct_aux.INDEX_ATTRS["Si_minus"] = 16
    fct_aux.INDEX_ATTRS["Si_plus"] = 17
    fct_aux.INDEX_ATTRS["prob_mode_state_i"] = 18
    fct_aux.INDEX_ATTRS["u_i"] = 19
    fct_aux.INDEX_ATTRS["bg_i"] = 20
    fct_aux.INDEX_ATTRS["non_playing_players"] = 21
    nb_vars_2_add = 6
    # _______ variables' initialization --> fin   ________________
    
    # _______   turn arr_pl_M_T in a array of 4 dimensions   ________
    arrs = []
    for k in range(0, k_steps):
        arrs.append(list(arr_pl_M_T))
    arrs = np.array(arrs, dtype=object)
    arrs = np.transpose(arrs, [1,2,0,3])
    
    ## add initial values for the new attributs
    arr_pl_M_T_K_vars = np.zeros((arrs.shape[0],
                             arrs.shape[1],
                             arrs.shape[2],
                             arrs.shape[3]+nb_vars_2_add), 
                            dtype=object)
    arr_pl_M_T_K_vars[:,:,:,:-nb_vars_2_add] = arrs
    arr_pl_M_T_K_vars[:,:,:,fct_aux.INDEX_ATTRS["Si_minus"]] = 0
    arr_pl_M_T_K_vars[:,:,:,fct_aux.INDEX_ATTRS["Si_plus"]] = 0
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["u_i"]] = np.nan
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["bg_i"]] = np.nan
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["prob_mode_state_i"]] = 0.5
    arr_pl_M_T_K_vars[:,:,:, fct_aux.INDEX_ATTRS["non_playing_players"]] \
        = lriGameModel.NON_PLAYING_PLAYERS["PLAY"]
    
    arr_pl_M_T_K_vars = fct_aux.reupdate_state_players(arr_pl_M_T_K_vars)
    arr_pl_M_T_K_vars_modif = arr_pl_M_T_K_vars.copy()
    
    # ____      run balanced sg for t_period = 0 at any k_step     ________
    
    # constances, pi_sg_{plus, minus}, pi_0_{plus, minus}
    # pi_sg_plus_t, pi_sg_minus_t = 5, 4
    # pi_0_plus_t, pi_0_minus_t = 3, 1
    t = 0
    pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K
    pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K
    pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K
    pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K
    
    # initialization of variables
    nb_repeat_k = 0
    k = 0
    indices_non_playing_players = []      # indices of non-playing players because bg_min = bg_max
    arr_bg_i_nb_repeat_k = np.empty(
                            shape=(m_players,fct_aux.NB_REPEAT_K_MAX))
    arr_bg_i_nb_repeat_k.fill(np.nan)
    pi_sg_plus_t_k, pi_sg_minus_t_k = pi_sg_plus_t, pi_sg_minus_t
    pi_0_plus_t_k, pi_0_minus_t_k = pi_0_plus_t, pi_0_minus_t
    pi_sg_plus_T_K[t,k] = pi_sg_plus_t_k; 
    pi_sg_minus_T_K[t,k] = pi_sg_minus_t_k;
    pi_0_plus_T_K[t,k] = pi_0_plus_t_k; 
    pi_0_minus_T_K[t,k] = pi_0_minus_t_k;
    
    # learning steps
    while (k < k_steps):
        print("------- t = {}, k = {}, repeat_k = {}, prob_modes={} -------"\
              .format(
                t, k, nb_repeat_k,
                arr_pl_M_T_K_vars_modif[:,t,k,
                        fct_aux.INDEX_ATTRS["prob_mode_state_i"]]))
        print("pi_sg_plus_t_k = {}, pi_sg_minus_t_k={}".format(
                pi_sg_plus_t_k, pi_sg_minus_t_k ))
        print("indices_non_playing_players={}".format(indices_non_playing_players))
        
        ## balanced_player_game_t
        arr_pl_M_T_K_vars_modif, \
        b0_t_k, c0_t_k, \
        bens_t_k, csts_t_k, \
        pi_sg_plus_t_k_plus_1, pi_sg_minus_t_k_plus_1, \
        pi_0_plus_t_k_plus_1, pi_0_minus_t_k_plus_1, \
        dico_stats_res_t_k \
            = lriGameModel.balanced_player_game_t(
                    arr_pl_M_T_K_vars_modif, t, k, 
                    pi_hp_plus, pi_hp_minus, 
                    pi_sg_plus_t_k, pi_sg_minus_t_k,
                    m_players, indices_non_playing_players,
                    num_periods, nb_repeat_k, dbg=False)
            
        if manual_debug:
            pi_sg_plus_t_k_plus_1 = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t_k_plus_1 = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            pi_0_plus_t_k_plus_1 = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K #2 
            pi_0_minus_t_k_plus_1 = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
        
        ## update pi_sg_minus_t_k_minus_1 and pi_sg_plus_t_k_minus_1
        pi_sg_minus_t_k = pi_sg_minus_t_k_plus_1
        pi_sg_plus_t_k = pi_sg_plus_t_k_plus_1
            
        ## update variables at each step because they must have to converge in the best case
        #### update pi_sg_plus, pi_sg_minus of shape (NUM_PERIODS,K_STEPS)
        if k<k_steps-1:
            pi_sg_minus_T_K[t,k+1] = pi_sg_minus_t_k_plus_1
            pi_sg_plus_T_K[t,k+1] = pi_sg_plus_t_k_plus_1
            pi_0_minus_T_K[t,k+1] = pi_0_minus_t_k_plus_1
            pi_0_plus_T_K[t,k+1] = pi_0_plus_t_k_plus_1
        #### update b0_s, c0_s of shape (NUM_PERIODS,K_STEPS) 
        b0_s_T_K[t,k] = b0_t_k
        c0_s_T_K[t,k] = c0_t_k
        #### update BENs, CSTs of shape (M_PLAYERS,NUM_PERIODS,K_STEPS)
        #### shape: bens_t_k: (M_PLAYERS,)
        BENs_M_T_K[:,t,k] = bens_t_k
        CSTs_M_T_K[:,t,k] = csts_t_k
        
        ## compute new strategies probabilities by using utility fonction
        print("bens_t_k={}, csts_t_k={}".format(bens_t_k.shape, csts_t_k.shape))
        arr_pl_M_T_K_vars_modif, arr_bg_i_nb_repeat_k, \
        bool_bg_i_min_eq_max, indices_non_playing_players_new,\
        bg_min_i_t_0_to_k, bg_max_i_t_0_to_k \
            = lriGameModel.update_probs_modes_states_by_defined_utility_funtion(
                arr_pl_M_T_K_vars_modif.copy(), 
                arr_bg_i_nb_repeat_k.copy(),
                t, k,
                b0_t_k, c0_t_k,
                bens_t_k, csts_t_k,
                pi_hp_minus,
                pi_0_plus_t_k, pi_0_minus_t_k,
                m_players, 
                indices_non_playing_players, nb_repeat_k,
                learning_rate, 
                utility_function_version)
           
        print("__bool_bg_i_min_eq_max={}, nb_repeat_k={}__".format(bool_bg_i_min_eq_max, 
                nb_repeat_k ))
        
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
                    t,k,
                    fct_aux.INDEX_ATTRS["non_playing_players"]] \
                        = fct_aux.NON_PLAYING_PLAYERS["NOT_PLAY"]
            arr_pl_M_T_K_vars_modif[
                    indices_non_playing_players,
                    t,k,
                    fct_aux.INDEX_ATTRS["non_playing_players"]] \
                        = fct_aux.NON_PLAYING_PLAYERS["NOT_PLAY"]
            ### update bg_i, mode_i, u_i_t_k, p_i_t_k for not playing players from k+1 to k_steps 
            for var in ["prob_mode_state_i"]:
                # TODO change prob_mode_state_i by p_i_t_k
                arr_pl_M_T_K_vars_modif[
                    indices_non_playing_players,
                    t,k,
                    fct_aux.INDEX_ATTRS[var]] \
                    = arr_pl_M_T_K_vars_modif[
                                indices_non_playing_players,
                                t,k-1,
                                fct_aux.INDEX_ATTRS[var]] \
                        if k>0 \
                        else arr_pl_M_T_K_vars_modif[
                                indices_non_playing_players,
                                t,k,
                                fct_aux.INDEX_ATTRS[var]] #.reshape(-1,1)
                arr_pl_M_T_K_vars[
                    indices_non_playing_players,
                    t,k,
                    fct_aux.INDEX_ATTRS[var]] \
                            = arr_pl_M_T_K_vars_modif[
                                indices_non_playing_players,
                                t,k-1,
                                fct_aux.INDEX_ATTRS[var]] \
                        if k>0 \
                        else arr_pl_M_T_K_vars_modif[
                                indices_non_playing_players,
                                t,k,
                                fct_aux.INDEX_ATTRS[var]] #.reshape(-1,1)
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
            indices_non_playing_players = set()
            arr_bg_i_nb_repeat_k \
                = np.empty(
                    shape=(m_players, fct_aux.NB_REPEAT_K_MAX)
                    )
            arr_bg_i_nb_repeat_k.fill(np.nan)
            
        else:
            arr_pl_M_T_K_vars[:,t,k,:] \
                = arr_pl_M_T_K_vars_modif[:,t,k,:].copy()
            print("p_i_t_k: arr={},arr_modif={}".format(
                arr_pl_M_T_K_vars[:,t,k,fct_aux.INDEX_ATTRS["prob_mode_state_i"]],
                arr_pl_M_T_K_vars_modif[:,t,k,fct_aux.INDEX_ATTRS["prob_mode_state_i"]]))
            
            k = k+1
            nb_repeat_k = 0
            arr_bg_i_nb_repeat_k \
                = np.empty(
                    shape=(m_players, fct_aux.NB_REPEAT_K_MAX)
                    )
            arr_bg_i_nb_repeat_k.fill(np.nan)
            indices_non_playing_players = indices_non_playing_players_new
            indices_non_playing_players = set()
            
    arr_pl_M_T_K_vars = arr_pl_M_T_K_vars_modif.copy()
    arr_pl_M_T_K_vars = fct_aux.reupdate_state_players(arr_pl_M_T_K_vars.copy())

    #### recompute BENs, CSTs of shape (M_PLAYERS,NUM_PERIODS,K_STEPS)
    BENs_M_T_K[:,t,:] = 1 + b0_s_T_K[t,:] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["prod_i"]] \
                        + arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["gamma_i"]] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["r_i"]] 
    CSTs_M_T_K[:,t,:] =  c0_s_T_K[t,:] \
                        * arr_pl_M_T_K_vars[:,t,
                                            :,fct_aux.INDEX_ATTRS["cons_i"]]
                        
    # update pi_sg_plus_t_minus_1 and pi_sg_minus_t_minus_1
    pi_sg_plus_t = pi_sg_plus_T_K[t,k_steps-1]
    pi_sg_minus_t = pi_sg_minus_T_K[t,k_steps-1]
    
    # __________        compute prices variables         ______________________
    ## B_is, C_is of shape (M_PLAYERS, )
    print("cons_is={}".format(arr_pl_M_T_K_vars[
                        :, t,
                        k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]))
    # CONS_is = np.sum(arr_pl_M_T_K_vars[
    #                     :, t,
    #                     k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]], 
    #                  axis=1)
    # PROD_is = np.sum(arr_pl_M_T_K_vars[
    #                     :, t, 
    #                     k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]], 
    #                  axis=1)
    prod_i_t = arr_pl_M_T_K_vars[:,t, k_steps-1, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_t = arr_pl_M_T_K_vars[:,t, k_steps-1, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is_M = b0_s_T_K[t,k_steps-1] * prod_i_t
    C_is_M = c0_s_T_K[t,k_steps-1] * cons_i_t
    ## BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    # BB_is_M = pi_sg_plus_T_K[t,-1] * PROD_is #np.sum(PROD_is)
    BB_is_M = pi_sg_plus_T_K[t,-1] * prod_i_t
    for num_pl, bb_i in enumerate(BB_is_M):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    # CC_is_M = pi_sg_minus_T_K[t,-1] * CONS_is #np.sum(CONS_is)
    CC_is_M = pi_sg_minus_T_K[t,-1] * cons_i_t #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * num_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * num_periods, dtype=object)
    
    #__________      save computed variables locally      _____________________
    msg = "pi_hp_plus_"+str(pi_hp_plus)\
                       +"_pi_hp_minus_"+str(pi_hp_minus)
    algo_name = "LRI1" if utility_function_version == 1 else "LRI2"
    path_to_save = os.path.join(path_to_save, scenario, str(prob_Ci), 
                                msg, algo_name, str(learning_rate))
    fct_aux.save_variables(path_to_save, arr_pl_M_T_K_vars, 
                   b0_s_T_K, c0_s_T_K, B_is_M, C_is_M, BENs_M_T_K, CSTs_M_T_K, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T_K, pi_sg_plus_T_K, 
                   pi_0_minus_T_K, pi_0_plus_T_K,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo="LRI")
    
    print("VARS shapes: b0_s_T_K={}, B_is_M={}, BENs_M_T_K={}, BB_is_M={}, pi_sg_minus_T_K={}, pi_0_minus_T_K={}".format(
        b0_s_T_K.shape, B_is_M.shape, BENs_M_T_K.shape, BB_is_M.shape, 
        pi_sg_minus_T_K.shape, pi_0_minus_T_K.shape))
    print("b0_s_T_K={}".format(b0_s_T_K))
    print("{}: {}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, probs_mode={} ---> fin \n".format(
            algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus, 
            probs_modes_states))
    
    return arr_pl_M_T_K_vars

#______________________________________________________________________________
#
#                   REGROUPEMENT ALGOS DETERMINIST, LRI1, LRI2 
#______________________________________________________________________________
def execution_algos(algos, arr_pl_M_T_vars_init, 
                    pi_hp_plus=0.0002, pi_hp_minus=0.33,
                    m_players=4, num_periods=2, k_steps=5,
                    prob_Ci=0.3, learning_rate=0.01,
                    probs_modes_states=[0.5, 0.5, 0.5],
                    scenario="SCENARIO_BASE", random_determinist=False,
                    utility_function_version=1, manual_debug=False, 
                    path_to_save="tests"):
    
    path_to_save = os.path.join(path_to_save, "SCENARIO_BASE")
    
    dico_arrs = dict()
    for algo in algos:
        arr_pl_M_T_vars = None
        if algo == "DETERMINIST":
            arr_pl_M_T_vars = scenario_base_det(arr_pl_M_T_vars_init, 
                                  pi_hp_plus=pi_hp_plus, 
                                  pi_hp_minus=pi_hp_minus,
                                  m_players=m_players, 
                                  num_periods=num_periods, 
                                  k_steps=k_steps,
                                  prob_Ci=prob_Ci, 
                                  learning_rate=learning_rate,
                                  probs_modes_states=probs_modes_states,
                                  scenario=scenario,
                                  random_determinist=random_determinist,
                                  utility_function_version=utility_function_version,
                                  path_to_save=path_to_save, 
                                  manual_debug=manual_debug,
                                  dbg=False)
        elif algo == fct_aux.ALGO_NAMES_BF[0]:              # BEST-BRUTE-FORCE
            arr_pl_M_T_vars = scenario_base_bf(arr_pl_M_T_vars_init, 
                                  pi_hp_plus=pi_hp_plus, 
                                  pi_hp_minus=pi_hp_minus,
                                  m_players=m_players, 
                                  num_periods=num_periods, 
                                  k_steps=k_steps,
                                  prob_Ci=prob_Ci, 
                                  learning_rate=learning_rate,
                                  probs_modes_states=probs_modes_states,
                                  scenario=scenario,
                                  algo_name=algo,
                                  path_to_save=path_to_save, 
                                  manual_debug=manual_debug,
                                  dbg=False)
        elif algo == fct_aux.ALGO_NAMES_BF[1]:            # BAD-BRUTE-FORCE
            arr_pl_M_T_vars = scenario_base_bf(arr_pl_M_T_vars_init, 
                                  pi_hp_plus=pi_hp_plus, 
                                  pi_hp_minus=pi_hp_minus,
                                  m_players=m_players, 
                                  num_periods=num_periods, 
                                  k_steps=k_steps,
                                  prob_Ci=prob_Ci, 
                                  learning_rate=learning_rate,
                                  probs_modes_states=probs_modes_states,
                                  scenario=scenario,
                                  algo_name=algo,
                                  path_to_save=path_to_save, 
                                  manual_debug=manual_debug,
                                  dbg=False)
        elif algo == fct_aux.ALGO_NAMES_BF[2]:            # MIDDLE-BRUTE-FORCE
            arr_pl_M_T_vars = scenario_base_bf(arr_pl_M_T_vars_init, 
                                  pi_hp_plus=pi_hp_plus, 
                                  pi_hp_minus=pi_hp_minus,
                                  m_players=m_players, 
                                  num_periods=num_periods, 
                                  k_steps=k_steps,
                                  prob_Ci=prob_Ci, 
                                  learning_rate=learning_rate,
                                  probs_modes_states=probs_modes_states,
                                  scenario=scenario,
                                  algo_name=algo,
                                  path_to_save=path_to_save, 
                                  manual_debug=manual_debug,
                                  dbg=False)
        elif algo == "LRI1":
            arr_pl_M_T_vars = scenario_base_LRI(arr_pl_M_T_vars_init,
                             pi_hp_plus=pi_hp_plus, 
                             pi_hp_minus=pi_hp_minus,
                             m_players=m_players, 
                             num_periods=num_periods, 
                             k_steps=k_steps,
                             prob_Ci=prob_Ci, 
                             learning_rate=learning_rate,
                             probs_modes_states=probs_modes_states,
                             scenario=scenario,
                             utility_function_version=1,
                             path_to_save=path_to_save, 
                             manual_debug=manual_debug,
                             dbg=False)
        elif algo == "LRI2":
            arr_pl_M_T_vars = scenario_base_LRI(arr_pl_M_T_vars_init,
                             pi_hp_plus=pi_hp_plus, 
                             pi_hp_minus=pi_hp_minus,
                             m_players=m_players, 
                             num_periods=num_periods, 
                             k_steps=k_steps,
                             prob_Ci=prob_Ci, 
                             learning_rate=learning_rate,
                             probs_modes_states=probs_modes_states,
                             scenario=scenario,
                             utility_function_version=2,
                             path_to_save=path_to_save, 
                             manual_debug=manual_debug, 
                             dbg=False)
         
        dico_arrs[algo] = arr_pl_M_T_vars
        
    return dico_arrs
        

if __name__ == "__main__":
    ti = time.time()
    
    t = 0
    
    name_dir = 'tests'
    game_dir = 'INSTANCES_GAMES'
    
    fct_aux.N_DECIMALS = 6
    fct_aux.Ci_LOW = 10
    fct_aux.Ci_HIGH = 60
    fct_aux.NB_REPEAT_K_MAX = 15
    
    fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K = 5
    fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K = 4
    fct_aux.MANUEL_DBG_PI_0_PLUS_T_K = 3 
    fct_aux.MANUEL_DBG_PI_0_MINUS_T_K = 1
    
    manual_debug = True # False #True
    
    scenarios=["scenario1"]
    prob_Cis=0.3
    date_hhmm="1611_1041"
    algos= ["DETERMINIST","LRI1","LRI2"] + fct_aux.ALGO_NAMES_BF
    scenario="SCENARIO_BASE"
    learning_rate = 0.1#0.01 # list(np.arange(0.05, 0.15, step=0.05))
    pi_hp_plus = 0.2*pow(10,-3)
    pi_hp_minus = 0.33
    

    arr_pl_M_T_vars_init = create_scenario_base(m_players=4,num_periods=2)
    
    k_steps = 40 #25 # 15
    execution_algos(algos, arr_pl_M_T_vars_init, k_steps=k_steps, 
                    manual_debug=manual_debug)
    
    # visualisation 
    
    vizScenBase.MULT_WIDTH = 2.25;
    vizScenBase.MULT_HEIGHT = 1.1;
    name_simu = "SCENARIO_BASE"; k_steps_args = k_steps
    ## -- turn_arr4d_2_df()
    tuple_paths, scenarios_new, prob_Cis_new, \
        prices_new, algos_new, learning_rates_new \
            = vizScenBase.get_tuple_paths_of_arrays(name_simu=name_simu)
    df_arr_M_T_Ks, df_ben_cst_M_T_K, df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M \
        = vizScenBase.get_array_turn_df_for_t_scenBase(
            tuple_paths, t, k_steps_args, 
            algos_for_not_learning=["DETERMINIST","RD-DETERMINIST",
                                    "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE", 
                                    "MIDDLE-BRUTE-FORCE"])
    ## -- plot figures    
    name_dir = os.path.join("tests", name_simu)
    vizScenBase.group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, t, name_dir, 
                        vizScenBase.NAME_RESULT_SHOW_VARS)
    
    print("runtime = {}".format(time.time() - ti))  
    