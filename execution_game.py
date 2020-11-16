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
import itertools as it
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import game_model_period_T as gmpT
import deterministic_game_model as detGameModel
import lri_game_model as lriGameModel
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
        
def execute_game_probCis_scenarios(
            pi_hp_plus=[10], pi_hp_minus=[15],
            probCis=[0.3, 0.5, 0.7], 
            scenarios=["scenario1", "scenario2", "scenario3"],
            m_players=3, num_periods=5, 
            Ci_low=10, Ci_high=60, name_dir='tests', dbg=False):
    """
    run the game using the combinaison of probCis and scenarios

    Parameters
    ----------
    pi_hp_plus : float,
        DESCRIPTION.
        the price of exported energy from SG to HP
    pi_hp_minus : float,
        DESCRIPTION.
        the price of imported energy from HP to SG
    probCis : list of float
        DESCRIPTION.
        list of probability for choosing the kind of consommation
    scenarios : list of String
        DESCRIPTION.
        a plan for operating a game of players
    m_players : Integer optional
        DESCRIPTION. The default is 3.
        the number of players
    num_periods : Integer, optional
        DESCRIPTION. The default is 5.
        the number of time intervals 
    Ci_low : float, optional
        DESCRIPTION. The default is 10.
        the min value of the consumption
    Ci_high : float, optional
        DESCRIPTION. The default is 60.
        the max value of the consumption
    name_dir: String,
         DESCRIPTION.
         name of directory for saving variables of players
    Returns
    -------
    None.
    Nothing to return

    """

    date_hhmm = datetime.now().strftime("%d%m_%H%M")
    
    zip_pi_hp = zip(pi_hp_plus, pi_hp_minus)
    
    probCi_scen_piHpPlus_piHpMinus_s = it.product(probCis, 
                                                  scenarios, 
                                                  zip_pi_hp)
    for (prob_Ci, scenario, (pi_hp_plus_elt, pi_hp_minus_elt)) \
        in probCi_scen_piHpPlus_piHpMinus_s:
        msg = "pi_hp_plus_"+str(pi_hp_plus_elt)\
                                        +"_pi_hp_minus_"+str(pi_hp_minus_elt)
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg
                                    )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        
        arr_pl_M_T_old, arr_pl_M_T, \
        b0_s, c0_s, \
        B_is, C_is, \
        BENs, CSTs, \
        BB_is, CC_is, RU_is , \
        pi_sg_plus, pi_sg_minus, \
        pi_0_plus, pi_0_minus, \
        dico_stats_res = \
            detGameModel.balance_player_game(
                            pi_hp_plus = pi_hp_plus_elt, 
                            pi_hp_minus = pi_hp_minus_elt,
                            m_players = m_players, 
                            num_periods = num_periods,
                            Ci_low = Ci_low, 
                            Ci_high = Ci_high,
                            prob_Ci = prob_Ci, 
                            scenario = scenario,
                            path_to_save = path_to_save, 
                            dbg = dbg
                            )
        
# ____  new version with all algorithms (LRI1, LRI2, DETERM, RANDOM) : debut __  
        
def execute_algos():
    
    name_dir = 'tests'
    # constances 
    m_players = 3 # 10 # 100
    num_periods = 5 # 50
    k_steps = 5 # 10 # 50
    # fct_aux.N_DECIMALS = 4
    fct_aux.NB_REPEAT_K_MAX = 3 #10
    # fct_aux.Ci_LOW = 10
    # fct_aux.Ci_HIGH = 60
    probs_modes_states = [0.5, 0.5, 0.5]
    
    # list of algos
    algos = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]
    # list of pi_hp_plus, pi_hp_minus
    pi_hp_plus = [5]
    pi_hp_minus = [15]
    # list of scenario
    scenarios = ["scenario1", "scenario2", "scenario3"] # ["scenario1"] # ["scenario1", "scenario2", "scenario3"]
    # list of prob_Ci
    prob_Cis = [0.3, 0.5, 0.7]
    # learning rate 
    learning_rates = [0.01] # list(np.arange(0.05, 0.15, step=0.05))
    
    # generation arrays 
    date_hhmm = datetime.now().strftime("%d%m_%H%M")
    
    zip_pi_hp = list(zip(pi_hp_plus, pi_hp_minus))
    
    cpt = 0
    for (prob_Ci, scenario) in it.product(prob_Cis, scenarios):
        arr_pl_M_T_probCi_scen \
            = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                m_players=m_players, 
                num_periods=num_periods, 
                scenario=scenario, prob_Ci=prob_Ci, 
                Ci_low=fct_aux.Ci_LOW, Ci_high=fct_aux.Ci_HIGH)
            
        algo_piHpPlusMinus_learning_arrPlMT \
            = it.product(algos, zip_pi_hp, learning_rates, 
                         [arr_pl_M_T_probCi_scen])
        for (algo, (pi_hp_plus_elt, pi_hp_minus_elt), 
             learning_rate, arr_pl_M_T_probCi_scen) \
            in algo_piHpPlusMinus_learning_arrPlMT:
            
            print("______ execution {}: {}, {}:{}______".format(cpt, 
                    algo, scenario, prob_Ci))
            cpt += 1
            msg = "pi_hp_plus_"+str(pi_hp_plus_elt)\
                       +"_pi_hp_minus_"+str(pi_hp_minus_elt)
            if algo == algos[3]:
                # RD-DETERMINIST
                print("*** RD-DETERMINIST *** ")
                random_determinist = True
                
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg, algo
                                    )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_vars = detGameModel.determinist_balanced_player_game(
                                 arr_pl_M_T_probCi_scen.copy(),
                                 pi_hp_plus=pi_hp_plus_elt, 
                                 pi_hp_minus=pi_hp_minus_elt,
                                 m_players=m_players, 
                                 num_periods=num_periods,
                                 prob_Ci=prob_Ci,
                                 scenario=scenario,
                                 random_determinist=random_determinist,
                                 path_to_save=path_to_save, dbg=False)
                
            elif algo == algos[2]:
                # DETERMINIST
                print("*** DETERMINIST *** ")
                random_determinist = False
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg, algo
                                    )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_vars = detGameModel.determinist_balanced_player_game(
                                 arr_pl_M_T_probCi_scen.copy(),
                                 pi_hp_plus=pi_hp_plus_elt, 
                                 pi_hp_minus=pi_hp_minus_elt,
                                 m_players=m_players, 
                                 num_periods=num_periods,
                                 prob_Ci=prob_Ci,
                                 scenario=scenario,
                                 random_determinist=random_determinist,
                                 path_to_save=path_to_save, dbg=False)
                
            elif algo == algos[1]:
                # LRI2
                print("*** LRI 2 *** ")
                utility_function_version = 2
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg, algo, str(learning_rate)
                                    )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_K_vars = lriGameModel.lri_balanced_player_game(
                                arr_pl_M_T_probCi_scen.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                m_players=m_players, 
                                num_periods=num_periods, 
                                k_steps=k_steps, 
                                prob_Ci=prob_Ci, 
                                learning_rate=learning_rate,
                                probs_modes_states=probs_modes_states,
                                scenario=scenario,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, dbg=False)
                
            elif algo == algos[0]:
                # LRI1
                print("*** LRI 1 *** ")
                utility_function_version = 1
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg, algo, str(learning_rate)
                                    )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_K_vars = lriGameModel.lri_balanced_player_game(
                                arr_pl_M_T_probCi_scen.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                m_players=m_players, 
                                num_periods=num_periods, 
                                k_steps=k_steps, 
                                prob_Ci=prob_Ci, 
                                learning_rate=learning_rate,
                                probs_modes_states=probs_modes_states,
                                scenario=scenario,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, dbg=False)
        
    print("NB_EXECUTION cpt={}".format(cpt))

def get_or_create_instance(m_players, num_periods, prob_Ci, 
                          scenario, path_to_arr_pl_M_T, used_instances):
    """
    get instance if it exists else create instance.

    Parameters
    ----------
    m_players : integer
        DESCRIPTION.
    num_periods : integer
        DESCRIPTION.
    prob_Ci : float
        DESCRIPTION.
    scenario : string
        DESCRIPTION.
    path_to_arr_pl_M_T : string
        DESCRIPTION.
        path to array arr_pl_M_T
        example: tests/INSTANCES_GAMES/scenario{1,2,3}/prob_Ci/\
                    arr_pl_M_T_players_{m_players}_periods_{num_periods}.npy
    used_instances : boolean
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_probCi_scen : array of 
        DESCRIPTION.

    """
    arr_pl_M_T_probCi_scen = None
    if os.path.exists(path_to_arr_pl_M_T):
        # read arr_pl_M_T
        if used_instances:
            arr_pl_M_T_probCi_scen \
                = np.load(path_to_arr_pl_M_T,
                          allow_pickle=True)
            print("READ INSTANCE GENERATED")
            
        else:
            # create arr_pl_M_T when used_instances = False
            arr_pl_M_T_probCi_scen \
                = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                    m_players=m_players, 
                    num_periods=num_periods, 
                    scenario=scenario, prob_Ci=prob_Ci, 
                    Ci_low=fct_aux.Ci_LOW, Ci_high=fct_aux.Ci_HIGH)
            print("CREATE INSTANCE used_instance={}".format(used_instances))
    else:
        # create arr_pl_M_T
        arr_pl_M_T_probCi_scen \
            = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                m_players=m_players, 
                num_periods=num_periods, 
                scenario=scenario, prob_Ci=prob_Ci, 
                Ci_low=fct_aux.Ci_LOW, Ci_high=fct_aux.Ci_HIGH)
        print("NO INSTANCE CREATED")
            
    return arr_pl_M_T_probCi_scen

def execute_algos_used_Generated_instances(game_dir='tests', 
                                           name_dir='INSTANCES_GAMES', 
                                           scenarios=None,
                                           prob_Cis=None,
                                           date_hhmm=None,
                                           algos=None,
                                           used_instances=True):
    """
    execute algos by used generated instances if there exists or 
        by generated new instances
    
    scenarios = ["scenario1"]
    prob_Cis=[0.3]
    date_hhmm="1041"
    algos=["LRI1"]
    """
    # constances 
    m_players = 3 # 10 # 100
    num_periods = 5 # 50
    k_steps = 5 # 10 # 50
    fct_aux.NB_REPEAT_K_MAX = 3 #10
    probs_modes_states = [0.5, 0.5, 0.5]
    
    # list of algos
    ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]
    algos = ALGOS if algos is None \
                    else algos
    # list of pi_hp_plus, pi_hp_minus
    pi_hp_plus = [5]
    pi_hp_minus = [15]
    # list of scenario
    scenarios = ["scenario1", "scenario2", "scenario3"] \
            if scenarios is None \
            else scenarios
    # list of prob_Ci
    prob_Cis = [0.3, 0.5, 0.7] \
            if prob_Cis is None \
            else prob_Cis
    # learning rate 
    learning_rates = [0.01] # list(np.arange(0.05, 0.15, step=0.05))
    
    # generation arrays 
    date_hhmm = datetime.now().strftime("%d%m_%H%M") \
            if date_hhmm is None \
            else date_hhmm
    
    zip_pi_hp = list(zip(pi_hp_plus, pi_hp_minus))
    
    cpt = 0
    for (prob_Ci, scenario) in it.product(prob_Cis, scenarios):
        
        name_file_arr_pl = "arr_pl_M_T_players_{}_periods_{}.npy".format(
                                m_players, num_periods)
        path_to_arr_pl_M_T = os.path.join(
                                name_dir, game_dir,
                                scenario, str(prob_Ci), 
                                name_file_arr_pl)
         
        arr_pl_M_T_probCi_scen \
            = get_or_create_instance(m_players, num_periods, 
                                    prob_Cis, scenarios, 
                                    path_to_arr_pl_M_T,
                                    used_instances)
            
        algo_piHpPlusMinus_learning_arrPlMT \
            = it.product(algos, zip_pi_hp, learning_rates, 
                         [arr_pl_M_T_probCi_scen])
        for (algo, (pi_hp_plus_elt, pi_hp_minus_elt), 
             learning_rate, arr_pl_M_T_probCi_scen) \
            in algo_piHpPlusMinus_learning_arrPlMT:
            
            print("______ execution {}: {}, {}:{}______".format(cpt, 
                    algo, scenario, prob_Ci))
            cpt += 1
            msg = "pi_hp_plus_"+str(pi_hp_plus_elt)\
                       +"_pi_hp_minus_"+str(pi_hp_minus_elt)
            if algo == ALGOS[3]:
                # RD-DETERMINIST
                print("*** RD-DETERMINIST *** ")
                random_determinist = True
                
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg, algo
                                    )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_vars = detGameModel.determinist_balanced_player_game(
                                 arr_pl_M_T_probCi_scen.copy(),
                                 pi_hp_plus=pi_hp_plus_elt, 
                                 pi_hp_minus=pi_hp_minus_elt,
                                 m_players=m_players, 
                                 num_periods=num_periods,
                                 prob_Ci=prob_Ci,
                                 scenario=scenario,
                                 random_determinist=random_determinist,
                                 path_to_save=path_to_save, dbg=False)
                
            elif algo == ALGOS[2]:
                # DETERMINIST
                print("*** DETERMINIST *** ")
                random_determinist = False
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg, algo
                                    )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_vars = detGameModel.determinist_balanced_player_game(
                                 arr_pl_M_T_probCi_scen.copy(),
                                 pi_hp_plus=pi_hp_plus_elt, 
                                 pi_hp_minus=pi_hp_minus_elt,
                                 m_players=m_players, 
                                 num_periods=num_periods,
                                 prob_Ci=prob_Ci,
                                 scenario=scenario,
                                 random_determinist=random_determinist,
                                 path_to_save=path_to_save, dbg=False)
                
            elif algo == ALGOS[1]:
                # LRI2
                print("*** LRI 2 *** ")
                utility_function_version = 2
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg, algo, str(learning_rate)
                                    )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_K_vars = lriGameModel.lri_balanced_player_game(
                                arr_pl_M_T_probCi_scen.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                m_players=m_players, 
                                num_periods=num_periods, 
                                k_steps=k_steps, 
                                prob_Ci=prob_Ci, 
                                learning_rate=learning_rate,
                                probs_modes_states=probs_modes_states,
                                scenario=scenario,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, dbg=False)
                
            elif algo == ALGOS[0]:
                # LRI1
                print("*** LRI 1 *** ")
                utility_function_version = 1
                path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg, algo, str(learning_rate)
                                    )
                Path(path_to_save).mkdir(parents=True, exist_ok=True)
                arr_M_T_K_vars = lriGameModel.lri_balanced_player_game(
                                arr_pl_M_T_probCi_scen.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                m_players=m_players, 
                                num_periods=num_periods, 
                                k_steps=k_steps, 
                                prob_Ci=prob_Ci, 
                                learning_rate=learning_rate,
                                probs_modes_states=probs_modes_states,
                                scenario=scenario,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, dbg=False)
        
    print("NB_EXECUTION cpt={}".format(cpt))


# ____  new version with all algorithms (LRI1, LRI2, DETERM, RANDOM) : fin   __  
       
# ____          Generation of instances of players : Debut          _________
def generation_instances(name_dir, game_dir):
    """
    Generation instances of players for various players' numbers and periods'
    numbers

    Returns
    -------
    None.

    """
    name_dir = 'tests'
    game_dir = 'INSTANCES_GAMES'
    
    nb_players = [3, 10, 20, 50, 300]
    nb_periods = [5, 15, 25, 50, 50]
    zip_player_period = zip(nb_players, nb_periods)
    
    # list of scenario
    scenarios = ["scenario1", "scenario2", "scenario3"] # ["scenario1"] # ["scenario1", "scenario2", "scenario3"]
    # list of prob_Ci
    prob_Cis = [0.3, 0.5, 0.7]
    
    for (prob_Ci, scenario, (m_players, num_periods)) in \
        it.product(prob_Cis, scenarios, zip_player_period):
        
        path_to_save = os.path.join(
                            name_dir, game_dir,
                            scenario, str(prob_Ci)
                            )    
        
        name_file_arr_pl = "arr_pl_M_T_players_{}_periods_{}.npy".format(
                                m_players, num_periods)
        if os.path.exists(os.path.join(path_to_save, name_file_arr_pl)):
            print("file {} already EXISTS".format(name_file_arr_pl))
        else:
            arr_pl_M_T_probCi_scen \
                = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                    m_players=m_players, 
                    num_periods=num_periods, 
                    scenario=scenario, prob_Ci=prob_Ci, 
                    Ci_low=fct_aux.Ci_LOW, Ci_high=fct_aux.Ci_HIGH)
                
            fct_aux.save_instances_games(
                        arr_pl_M_T_probCi_scen, 
                        name_file_arr_pl,  
                        path_to_save)
        
            print("Generation instances players={}, periods={}, {}, prob_Ci={} ---> OK".format(
                    m_players, num_periods, scenario, prob_Ci))
        
    print("Generation instances TERMINEE")
    
# ____          Generation of instances of players : Fin            _________
        
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


def test_execute_game_probCis_scenarios():
    pi_hp_plus= [5, 10, 15]
    coef = 3; coefs = [coef]
    for i in range(0,len(pi_hp_plus)-1):
        val = round(coefs[i]/coef,1)
        coefs.append(val)
    pi_hp_minus = [ int(math.floor(pi_hp_plus[i]*coefs[i])) 
                   for i in range(0, len(pi_hp_plus))]
    probCis = [0.3, 0.5, 0.7]
    scenarios = ["scenario1", "scenario2", "scenario3"]
    m_players = 1000
    num_periods = 50
    execute_game_probCis_scenarios(
            pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
            probCis=probCis, 
            scenarios=scenarios,
            m_players=m_players, num_periods=num_periods, 
            Ci_low=10, Ci_high=60, name_dir='tests', dbg=False)

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    
    name_dir = 'tests'
    game_dir = 'INSTANCES_GAMES'
    
    fct_aux.N_DECIMALS = 4
    fct_aux.Ci_LOW = 10
    fct_aux.Ci_HIGH = 60
    
    
    generation_instances(name_dir, game_dir)
    
    scenarios=["scenario1"]
    prob_Cis=[0.3]
    date_hhmm=None # "1041"
    algos=["LRI1","LRI2"]
    execute_algos_used_Generated_instances(game_dir, 
                                           name_dir, 
                                           scenarios=scenarios,
                                           prob_Cis=prob_Cis,
                                           date_hhmm=date_hhmm,
                                           algos=algos,
                                           used_instances=True)
    
    #test_execute_game_onecase(fct_aux.CASE2)
    #test_execute_game_allcase()
    #df_bol, dico = test_balanced_player_all_time()
    # df_bol, df_res, dico_cases = test_balanced_player_all_time()
    #test_execute_game_probCis_scenarios()
    
    #execute_algos()
    
    print("runtime = {}".format(time.time() - ti))  
    