# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 09:10:32 2021

@author: jwehounou
"""
import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd
import itertools as it
import multiprocessing as mp

import fonctions_auxiliaires as fct_aux
import deterministic_game_model_automate_4_all_t as autoDetGameModel
import lri_game_model_automate_4_all_t as autoLriGameModel
import force_brute_game_model_automate_4_all_t as autoBfGameModel
import force_brute_game_model_automate_4_all_t_oneAlgo_V1 as autoBfGameModel_1algoV1
import detection_nash_game_model_automate_4_all_t as autoNashGameModel

from datetime import datetime
from pathlib import Path


ALGOS_LRI = ["LRI1", "LRI2"]
ALGOS_DET = ["DETERMINIST", "RD-DETERMINIST"]
#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------
#_________              create dico of parameters: debut         ______________ 
def define_parameters(dico_params):
    """
    create a list of parameters applied to the running of each algo 
    in the multiprocessing execution 

    """
    
    params = list()
    
    for algo_name, pi_hp_plus_minus, learning_rate \
        in it.product(dico_params["algos"], 
                      dico_params["tuple_pi_hp_plus_minus"],
                      dico_params['learning_rates']):
        params.append( (dico_params["arr_pl_M_T_vars_init"], 
                        algo_name, 
                        pi_hp_plus_minus[0], 
                        pi_hp_plus_minus[1],
                        learning_rate, 
                        dico_params["k_steps"],
                        dico_params["name_dir"], 
                        dico_params["date_hhmm"],
                        dico_params["gamma_version"], 
                        dico_params["used_instances"], 
                        dico_params["used_storage_det"], 
                        dico_params["manual_debug"], 
                        dico_params["criteria_bf"],
                        dico_params["debug"]
                        ) )
    return params

def define_parameters_multi_gammaV(dico_params):
    """
    create a list of parameters applied to the running of each algo 
    in the multiprocessing execution 

    """
    
    params = list()
    
    for algo_name, pi_hp_plus_minus, learning_rate, gamma_version \
        in it.product(dico_params["algos"], 
                      dico_params["tuple_pi_hp_plus_minus"],
                      dico_params['learning_rates'], 
                      dico_params["gamma_versions"]):
        
        date_hhmm_new = "_".join([dico_params["date_hhmm"], dico_params["scenario"], 
                              "".join(["T", str(dico_params["t_periods"]),
                                "".join(["gammaV", str(gamma_version)])])])
            
        params.append( (dico_params["arr_pl_M_T_vars_init"], 
                        algo_name, 
                        pi_hp_plus_minus[0], 
                        pi_hp_plus_minus[1],
                        learning_rate, 
                        dico_params["k_steps"],
                        dico_params["name_dir"], 
                        date_hhmm_new,
                        gamma_version, 
                        dico_params["used_instances"], 
                        dico_params["used_storage_det"], 
                        dico_params["manual_debug"], 
                        dico_params["criteria_bf"],
                        dico_params["debug"]
                        ) )
    return params

#_________              create dico of parameters: fin           ______________ 

#_________                  One algo: debut             _______________________ 
def execute_one_algo_used_Generated_instances(arr_pl_M_T_vars_init,
                                            algo_name,
                                            pi_hp_plus=None,
                                            pi_hp_minus=None,
                                            learning_rate=None,
                                            k_steps=None,
                                            name_dir="",
                                            date_hhmm="",
                                            gamma_version=1,
                                            used_instances=True,
                                            used_storage_det=True,
                                            manual_debug=False, 
                                            criteria_bf="Perf_t", 
                                            debug=False):
    """
    execute algos by using generated instances if there exists or 
        by generating new instances
    
    date_hhmm="1041"
    algos=["LRI1"]
    
    """
    # directory to save  execution algos
    print("______ execution: {}, rate={}______".format( 
                algo_name, learning_rate))
    
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name
                                    )
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    if algo_name == ALGOS_LRI[0]:
        # 0: LRI1
        print("*** ALGO: {} *** ".format(algo_name))
        utility_function_version = 1
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name, str(learning_rate)
                                )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_K_vars = autoLriGameModel\
                            .lri_balanced_player_game_all_pijk_upper_08(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                gamma_version=gamma_version,
                                k_steps=k_steps, 
                                learning_rate=learning_rate,
                                p_i_j_ks=p_i_j_ks,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, 
                                manual_debug=manual_debug, dbg=debug)
    elif algo_name == ALGOS_LRI[1]:
        # LRI2
        print("*** ALGO: {} *** ".format(algo_name))
        utility_function_version = 2
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name, str(learning_rate)
                                )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_K_vars = autoLriGameModel\
                            .lri_balanced_player_game_all_pijk_upper_08(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                gamma_version=gamma_version,
                                k_steps=k_steps, 
                                learning_rate=learning_rate,
                                p_i_j_ks=p_i_j_ks,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, 
                                manual_debug=manual_debug, dbg=debug)
        
    elif algo_name in ALGOS_DET:
        # 2: DETERMINIST, 3: RANDOM DETERMINIST
        print("*** ALGO: {} *** ".format(algo_name))
        random_determinist = False if algo_name == ALGOS_DET[0] else True
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name
                                )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_vars = autoDetGameModel.determinist_balanced_player_game(
                         arr_pl_M_T_vars_init.copy(),
                         pi_hp_plus=pi_hp_plus, 
                         pi_hp_minus=pi_hp_minus,
                         gamma_version=gamma_version,
                         random_determinist=random_determinist,
                         used_storage=used_storage_det,
                         path_to_save=path_to_save, 
                         manual_debug=manual_debug, dbg=debug)
        
    elif algo_name in fct_aux.ALGO_NAMES_BF:
        # 0: BEST_BRUTE_FORCE (BF) , 1:BAD_BF, 2: MIDDLE_BF
        print("*** ALGO: {} *** ".format(algo_name))
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name
                                    )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_vars = autoBfGameModel_1algoV1\
                            .bf_balanced_player_game_ONE_ALGO(
                                arr_pl_M_T_vars_init.copy(),
                                algo_name,
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                gamma_version=gamma_version,
                                path_to_save=path_to_save, 
                                name_dir=name_dir, 
                                date_hhmm=date_hhmm,
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
                       
    elif algo_name == fct_aux.ALGO_NAMES_NASH[0] :
        # 0: "BEST-NASH", 1: "BAD-NASH", 2: "MIDDLE-NASH"
        print("*** ALGO: {} *** ".format(algo_name))
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name
                                    )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_vars = autoNashGameModel\
                        .nash_balanced_player_game_perf_t_USE_DICT_MODE_PROFIL(
                            arr_pl_M_T_vars_init.copy(),
                            pi_hp_plus=pi_hp_plus, 
                            pi_hp_minus=pi_hp_minus,
                            gamma_version=gamma_version,
                            path_to_save=path_to_save, 
                            name_dir=name_dir, 
                            date_hhmm=date_hhmm,
                            manual_debug=manual_debug, 
                            dbg=debug)          
    
#_________                      One algo: fin                   _______________
 
#------------------------------------------------------------------------------
#                   definitions of unittests
#------------------------------------------------------------------------------
def test_debug_procedurale():
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    t_periods = 3
    #setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3               # 12 players
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    gamma_version = 1
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                            setA_m_players, setB_m_players, setC_m_players, 
                            t_periods, 
                            scenario1,
                            path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    
    """ _____ list of parameters with their type
    arr_pl_M_T_vars_init : array of shape (m_players, t_periods, k_steps, len(vars)) 
    name_dir: string,
    date_hhmm: string,
    k_steps: integer,
    NB_REPEAT_K_MAX: integer,
    algos: list of string,
    learning_rates: list of integer,
    "tuple_pi_hp_plus_minus": tuple of a couple of pi_hp_plus, pi_hp_minus
    gamma_version: integer,
    used_instances: boolean,
    used_storage_det: boolean,
    manual_debug: boolean, 
    criteria_bf: string, 
    debug: boolean
    ____ """
    name_dir = "tests"
    date_hhmm = datetime.now().strftime("%d%m_%H%M") 
    
    # steps of learning
    k_steps = 250
    k_steps = 5 if k_steps is None else k_steps
    NB_REPEAT_K_MAX = None
    fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
    
    # list of algos
    algos = None
    ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]\
            + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    algos = ALGOS if algos is None \
                  else algos
    # list of pi_hp_plus, pi_hp_minus
    pi_hp_plus = [10]
    pi_hp_minus = [20]
    tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
    
    # learning rate 
    learning_rates = [0.1]
    learning_rates = [0.01] \
            if learning_rates is None \
            else learning_rates
            
    gamma_version = 1
    utility_function_version = 1
    used_instances = True
    used_storage_det = True
    manual_debug = False 
    criteria_bf = "Perf_t" 
    debug = False
    
    dico_params = {
        "arr_pl_M_T_vars_init" : arr_pl_M_T_vars_init, 
        "name_dir": name_dir,
        "date_hhmm": date_hhmm,
        "k_steps": k_steps,
        "NB_REPEAT_K_MAX": NB_REPEAT_K_MAX,
        "algos": algos,
        "learning_rates": learning_rates,
        "tuple_pi_hp_plus_minus": tuple_pi_hp_plus_minus,
        "utility_function_version": utility_function_version,
        "gamma_version": gamma_version,
        "used_instances": used_instances,
        "used_storage_det": used_storage_det,
        "manual_debug": manual_debug, 
        "criteria_bf": criteria_bf, 
        "debug": debug
        }
    params = define_parameters(dico_params)
    print("define parameters finished")
    
    for param in params:
        print("param {} debut".format(param[1]))
        execute_one_algo_used_Generated_instances(
            arr_pl_M_T_vars_init=param[0],
            algo_name=param[1],
            pi_hp_plus=param[2],
            pi_hp_minus=param[3],
            learning_rate=param[4],
            k_steps=param[5],
            name_dir=param[6],
            date_hhmm=param[7],
            gamma_version=param[8],
            used_instances=param[9],
            used_storage_det=param[10],
            manual_debug=param[11], 
            criteria_bf=param[12], 
            debug=param[13]
            )
        print("param {} FIN".format(param[1]))
    
def test_execute_algos_used_Generated_instances():
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    t_periods = 4
    #setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3               # 12 players
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    gamma_version = 1
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                            setA_m_players, setB_m_players, setC_m_players, 
                            t_periods, 
                            scenario1,
                            path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    
    """ _____ list of parameters with their type
    arr_pl_M_T_vars_init : array of shape (m_players, t_periods, k_steps, len(vars)) 
    name_dir: string,
    date_hhmm: string,
    k_steps: integer,
    NB_REPEAT_K_MAX: integer,
    algos: list of string,
    learning_rates: list of integer,
    "tuple_pi_hp_plus_minus": tuple of a couple of pi_hp_plus, pi_hp_minus
    gamma_version: integer,
    used_instances: boolean,
    used_storage_det: boolean,
    manual_debug: boolean, 
    criteria_bf: string, 
    debug: boolean
    ____ """
    name_dir = "tests"
    date_hhmm = datetime.now().strftime("%d%m_%H%M") 
    
    # steps of learning
    k_steps = 250
    k_steps = 5 if k_steps is None else k_steps
    NB_REPEAT_K_MAX = None
    fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
    
    # list of algos
    algos = None
    ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]\
            + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    algos = ALGOS if algos is None \
                  else algos
    # list of pi_hp_plus, pi_hp_minus
    pi_hp_plus = [10]
    pi_hp_minus = [20]
    tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
    
    # learning rate 
    learning_rates = [0.1]
    learning_rates = [0.01] \
            if learning_rates is None \
            else learning_rates
            
    gamma_version = 1
    utility_function_version = 1
    used_instances = True
    used_storage_det = True
    manual_debug = False 
    criteria_bf = "Perf_t" 
    debug = False
    
    dico_params = {
        "arr_pl_M_T_vars_init" : arr_pl_M_T_vars_init, 
        "name_dir": name_dir,
        "date_hhmm": date_hhmm,
        "k_steps": k_steps,
        "NB_REPEAT_K_MAX": NB_REPEAT_K_MAX,
        "algos": algos,
        "learning_rates": learning_rates,
        "tuple_pi_hp_plus_minus": tuple_pi_hp_plus_minus,
        "gamma_version": gamma_version,
        "used_instances": used_instances,
        "used_storage_det": used_storage_det,
        "manual_debug": manual_debug, 
        "criteria_bf": criteria_bf, 
        "debug": debug
        }
    params = define_parameters(dico_params)
    print("define parameters finished")
    
    # multi processing execution
    p = mp.Pool(mp.cpu_count()-1)
    p.starmap(execute_one_algo_used_Generated_instances, 
              params)
    # multi processing execution
    
    return params

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    params = test_execute_algos_used_Generated_instances()
    #test_debug_procedurale()
    print("runtime = {}".format(time.time() - ti))