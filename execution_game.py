# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 08:32:09 2020

@author: jwehounou

Execution game
"""
import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import game_model_period_T as gmpT
import visu_bkh as bkh

from datetime import datetime
from pathlib import Path




#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------
def execute_game_onecase(case=fct_aux.CASE3):
    """
    execution of the game with initialisation of constances and variables.

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
    case : tuple, optionnal
        DESCRIPTION.
        min and max values of random variable Pi. 
    Returns
    -------
    None.

    """
    
    str_case = str(case[0]) +"_"+ str(case[1])
    # create test directory tests and put file in the directory simu_date_hhmm
    name_dir = "tests"; date_hhmm = datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, str_case)
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    
    pi_hp_plus, pi_hp_minus = gmpT.generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = gmpT.generate_random_values(zero=1)
    
    gmpT.run_game_model_SG(pi_hp_plus, pi_hp_minus, 
                           pi_0_plus, pi_0_minus, 
                           case, path_to_save)
    

def execute_game_allcases(cases):
    """
    run game for all listed cases 

    Parameters
    ----------
    cases : list of tuples
        DESCRIPTION.

    Returns
    -------
    None.

    """
    name_dir = "tests"; date_hhmm = datetime.now().strftime("%d%m_%H%M")
    for case in cases:
        str_case = str(case[0]) +"_"+ str(case[1])
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, str_case)
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        
        pi_hp_plus, pi_hp_minus = gmpT.generate_random_values(zero=1)
        pi_0_plus, pi_0_minus = gmpT.generate_random_values(zero=1)
        
        gmpT.run_game_model_SG(pi_hp_plus, pi_hp_minus, 
                               pi_0_plus, pi_0_minus, 
                               case, path_to_save)
        
def generate_Pi_Si_by_profil_scenario(n_players=3, num_periods=5, 
                                   scenario="scenario1", prob_Ci=0.3, 
                                   Ci_low=10, Ci_high=60):
    arr_pl_M_T = []
    for num_pl in range(0, n_players):
        Ci = None; profili = None
        prob = np.random.uniform(0, 1)
        if prob <= prob_Ci:
            Ci = Ci_low
            Si_max = 0.8 * Ci
            if scenario == "scenario1":
                profili = fct_aux.PROFIL_L
            elif scenario == "scenario2":
                profili = fct_aux.PROFIL_H
            elif scenario == "scenario3":
                profili = fct_aux.PROFIL_M
        else:
            Ci = Ci_low
            Si_max = 0.5 * Ci
            if scenario == "scenario1":
                profili = fct_aux.PROFIL_H
            elif scenario == "scenario2":
                profili = fct_aux.PROFIL_M
            elif scenario == "scenario3":
                profili = np.random.default_rng().choice(
                            p=[0.5, 0.5],
                            a=[fct_aux.PROFIL_H, fct_aux.PROFIL_M])
                
        # profil_casei = None
        # prob_casei = np.random.uniform(0,1)
        # if prob_casei <= profili[0]:
        #     profil_casei = fct_aux.CASE1
        # elif prob_casei > profili[0] \
        #     and prob_casei <= profili[0]+profili[1]:
        #     profil_casei = fct_aux.CASE2
        # else:
        #     profil_casei = fct_aux.CASE3
        profil_casei = None
        if prob <= profili[0]:
            profil_casei = fct_aux.CASE1
        elif prob > profili[0] \
            and prob <= profili[0]+profili[1]:
            profil_casei = fct_aux.CASE2
        else:
            profil_casei = fct_aux.CASE3
                
        min_val_profil = profil_casei[0]*Ci 
        max_val_profil = profil_casei[1]*Ci
        
        Pi_s = list( np.around(np.random.uniform(low=min_val_profil, high=max_val_profil, 
                                 size=(num_periods,)), decimals=2) )
        Si_s = list( np.around(np.random.uniform(0,1,size=(num_periods,))*Si_max,
                    decimals=2))
        Si_s[0] = 0; 
        str_profili_s = ["_".join(map(str, profili))] * num_periods
        str_casei_s = ["_".join(map(str, profil_casei))] * num_periods
        
        # building list of list 
        """
        "Ci":0, "Pi":1, "Si":2, "Si_max":3, "gamma_i":4, 
        "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9,
        "Profili":10, "Casei":11, "R_i_old":12
        """
        Ci_s = [Ci] * num_periods
        Si_max_s = [Si_max] * num_periods
        gamma_i_s, r_i_s = [0]*num_periods, [0]*num_periods
        prod_i_s, cons_i_s = [0]*num_periods, [0]*num_periods
        state_i_s, mode_i_s = [""]*num_periods, [""]*num_periods
        R_i_old_s = [round(x - y, 2) for x, y in zip(Si_max_s, Si_s)]
        init_values_i_s = list(zip(Ci_s, Pi_s, Si_s, Si_max_s, gamma_i_s, 
                                   prod_i_s, cons_i_s, r_i_s, state_i_s, 
                                   mode_i_s, str_profili_s, str_casei_s, 
                                   R_i_old_s))
        arr_pl_M_T.append(init_values_i_s)
    
    arr_pl_M_T = np.array(arr_pl_M_T, dtype=object)
    
    return arr_pl_M_T


#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------ 
def test_execute_game_onecase(case):
    execute_game_onecase(case=case)
    
def test_execute_game_allcase():
    cases = [fct_aux.CASE1, fct_aux.CASE2, fct_aux.CASE3]
    execute_game_allcases(cases)
    
def test_balanced_player_all_time(thres=0.01):
    name_dir = "tests"
    # choose the last directory
    reps = os.listdir(name_dir)
    rep = reps[-1]
    print("_____ repertoire choisi : {} ______".format(rep))
    # verify the balancing player for all cases
    dico_cases = dict()
    rep_cases = os.listdir(os.path.join(name_dir, rep))
    for rep_case in rep_cases:
        path_to_variable = os.path.join(name_dir, rep, rep_case)
        arr_pls_M_T, RUs, \
        B0s, C0s, \
        BENs, CSTs, \
        pi_sg_plus_s, pi_sg_minus_s = \
            bkh.get_local_storage_variables(path_to_variable)
            
        df_bol = pd.DataFrame(index=range(0, arr_pls_M_T.shape[0]), 
                              columns=range(0, arr_pls_M_T.shape[1]))
        dico_numT = dict()
        for num_period in range(0, arr_pls_M_T.shape[1]):
            dico_pls = dict()
            for num_pl in range(0, arr_pls_M_T.shape[0]):
                state_i = arr_pls_M_T[num_pl, 
                                      num_period, 
                                      fct_aux.INDEX_ATTRS["state_i"]]
                mode_i = arr_pls_M_T[num_pl, 
                                      num_period, 
                                      fct_aux.INDEX_ATTRS["mode_i"]]
                Pi = arr_pls_M_T[num_pl, 
                                 num_period, 
                                 fct_aux.INDEX_ATTRS["Pi"]]
                Ci = arr_pls_M_T[num_pl, 
                                 num_period, 
                                 fct_aux.INDEX_ATTRS["Pi"]]
                Si = arr_pls_M_T[num_pl, 
                                 num_period, 
                                 fct_aux.INDEX_ATTRS["Si"]]
                Si_max = arr_pls_M_T[num_pl, 
                                 num_period, 
                                 fct_aux.INDEX_ATTRS["Si_max"]]
                Ri = Si_max - Si
                R_i_old = arr_pls_M_T[num_pl, 
                                 num_period, 
                                 fct_aux.INDEX_ATTRS["R_i_old"]]
                cons_i = arr_pls_M_T[num_pl, 
                                 num_period, 
                                 fct_aux.INDEX_ATTRS["cons_i"]]
                prod_i = arr_pls_M_T[num_pl, 
                                 num_period, 
                                 fct_aux.INDEX_ATTRS["prod_i"]]
                
                boolean = None; dico = dict()
                if state_i == "state1" and mode_i == "CONS+":
                    boolean = True \
                        if np.abs(Pi+(Si_max-R_i_old)+cons_i - Ci)<thres \
                        else False
                    formule = "Pi+(Si_max-R_i_old)+cons_i - Ci"
                    res = Pi+(Si_max-R_i_old)+cons_i - Ci
                    dico = {'Pi':np.round(Pi,2), 'Ci':np.round(Ci,2),
                            'Si':np.round(Si,2), 'Si_max':np.round(Si_max,2), 
                            'cons_i':np.round(cons_i,2), 'R_i_old': np.round(R_i_old,2),
                            "state_i": state_i, "mode_i": mode_i, 
                            "formule": formule, "res": res, "case":rep_case}
                elif state_i == "state1" and mode_i == "CONS-":
                    boolean = True if np.abs(Pi+cons_i - Ci)<thres else False
                    formule = "Pi+cons_i - Ci"
                    res = Pi+cons_i - Ci
                    dico = {'Pi':np.round(Pi,2), 'Si':np.round(Si,2), 
                            'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                            'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                            "state_i": state_i, "mode_i": mode_i, 
                            "formule": formule, "res": res, "case":rep_case}
                elif state_i == "state2" and mode_i == "DIS":
                    boolean = True if np.abs(Pi+(Si_max-R_i_old-Si) - Ci)<thres else False
                    formule = "Pi+(Si_max-R_i_old-Si) - Ci"
                    res = Pi+(Si_max-R_i_old-Si) - Ci
                    dico = {'Pi':np.round(Pi,2), 'Si':np.round(Si,2), 
                            'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                            'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                            "state_i": state_i, "mode_i": mode_i, 
                            "formule": formule, "res": res, "case":rep_case}
                elif state_i == "state2" and mode_i == "CONS-":
                    boolean = True if np.abs(Pi+cons_i - Ci)<thres else False
                    formule = "Pi+cons_i - Ci"
                    res = Pi+cons_i - Ci
                    dico = {'Pi':np.round(Pi,2), 'Si':np.round(Si,2), 
                            'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                            'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                            "state_i": state_i, "mode_i": mode_i, 
                            "formule": formule, "res": res, "case":rep_case}
                elif state_i == "state3" and mode_i == "PROD":
                    boolean = True if np.abs(Pi - Ci-prod_i)<thres else False
                    formule = "Pi - Ci-Si-prod_i"
                    res = Pi - Ci-prod_i
                    dico = {'Pi':np.round(Pi,2), 'Si':np.round(Si,2), 
                            'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                            "prod_i": np.round(prod_i,2), 
                            'cons_i': np.round(cons_i,2), 
                            'Ci': np.round(Ci,2), "state_i": state_i, 
                            "mode_i": mode_i, "formule": formule, 
                            "res": res, "case":rep_case}
                elif state_i == "state3" and mode_i == "DIS":
                    boolean = True if np.abs(Pi - Ci-(Si_max-Si)-prod_i)<thres else False
                    formule = "Pi - Ci-(Si_max-Si)-prod_i"
                    res = Pi - Ci-(Si_max-Si)-prod_i
                    dico = {'Pi': np.round(Pi,2), 'Si': np.round(Si,2), 
                            'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                            "prod_i": np.round(prod_i,2), 
                            'cons_i': np.round(cons_i,2), 
                            'Ci': np.round(Ci,2), "state_i": state_i, 
                            "mode_i": mode_i, "formule": formule, 
                                "res": res, "case":rep_case}
                
                df_bol.loc[num_pl, num_period] = boolean
                dico_pls[num_pl] = dico
            dico_numT[num_period] = dico_pls
        dico_cases[rep_case] = dico_numT
    df_res = pd.concat({"t_"+str(k): pd.DataFrame(v).T for k, v in dico_numT.items()}, 
                        axis=0)
    # df_res = pd.concat({"t_"+str(k_): pd.DataFrame(v_).T 
    #                     for k, dico_v in dico_cases.items() 
    #                     for k_, v_ in dico_v.items()}, 
    #                    axis=0)
    # convert dataframe to json
    # TODO
    return df_bol, df_res, dico_cases #dico_numT #df_dico


def test_generate_Pi_Si_by_profil_scenario():
    
    arr_pl_M_T = generate_Pi_Si_by_profil_scenario(n_players=3, num_periods=5, 
                                   scenario="scenario1", prob_Ci=0.3, 
                                   Ci_low=10, Ci_high=60)
    
    print("___ arr_pl_M_T : {}".format(arr_pl_M_T.shape))
    return arr_pl_M_T
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    #test_execute_game_onecase(fct_aux.CASE2)
    #test_execute_game_allcase()
    #df_bol, dico = test_balanced_player_all_time()
    # df_bol, df_res, dico_cases = test_balanced_player_all_time(
    
    arrs = test_generate_Pi_Si_by_profil_scenario()
    print("runtime = {}".format(time.time() - ti))  
    