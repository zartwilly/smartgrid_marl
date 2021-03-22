# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:18:12 2021

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
import fonctions_auxiliaires as fct_aux
import deterministic_game_model_automate_4_all_t as autoDetGameModel
import lri_game_model_automate_4_all_t as autoLriGameModel
import force_brute_game_model_automate_4_all_t as autoBfGameModel
import force_brute_game_model_automate_4_all_t_V1 as autoBfGameModel_V1
import force_brute_game_model_automate_4_all_t_oneAlgo_V1 as autoBfGameModel_1algoV1
import detection_nash_game_model_automate_4_all_t as autoNashGameModel

from datetime import datetime
from pathlib import Path


#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------

# #_________          USE_DICT_MODE_PROFIL: debut     __________________________ 
# def execute_algos_used_Generated_instances_USE_DICT_MODE_PROFIL(
#                                             arr_pl_M_T_vars_init,
#                                             name_dir=None,
#                                             date_hhmm=None,
#                                             k_steps=None,
#                                             NB_REPEAT_K_MAX=None,
#                                             algos=None,
#                                             learning_rates=None,
#                                             pi_hp_plus=None,
#                                             pi_hp_minus=None,
#                                             gamma_version=1,
#                                             used_instances=True,
#                                             used_storage_det=True,
#                                             manual_debug=False, 
#                                             criteria_bf="Perf_t", 
#                                             debug=False):
#     """
#     execute algos by using generated instances if there exists or 
#         by generating new instances
    
#     date_hhmm="1041"
#     algos=["LRI1"]
    
#     """
#     # directory to save  execution algos
#     name_dir = "tests" if name_dir is None else name_dir
#     date_hhmm = datetime.now().strftime("%d%m_%H%M") \
#             if date_hhmm is None \
#             else date_hhmm
    
#     # steps of learning
#     k_steps = 5 if k_steps is None else k_steps
#     fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
#     p_i_j_ks = [0.5, 0.5, 0.5]
    
#     # list of algos
#     ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]\
#             + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
#     algos = ALGOS if algos is None \
#                   else algos
#     # list of pi_hp_plus, pi_hp_minus
#     pi_hp_plus = [0.2*pow(10,-3)] if pi_hp_plus is None else pi_hp_plus
#     pi_hp_minus = [0.33] if pi_hp_minus is None else pi_hp_minus
#     # learning rate 
#     learning_rates = [0.01] \
#             if learning_rates is None \
#             else learning_rates # list(np.arange(0.05, 0.15, step=0.05))
    
    
    
#     zip_pi_hp = list(zip(pi_hp_plus, pi_hp_minus))
    
#     cpt = 0
#     algo_piHpPlusMinus_learningRate \
#             = it.product(algos, zip_pi_hp, learning_rates)
    
#     for (algo_name, (pi_hp_plus_elt, pi_hp_minus_elt), 
#              learning_rate) in algo_piHpPlusMinus_learningRate:
        
#         print("______ execution {}: {}, rate={}______".format(cpt, 
#                     algo_name, learning_rate))
#         cpt += 1
#         msg = "pi_hp_plus_"+str(pi_hp_plus_elt)\
#                        +"_pi_hp_minus_"+str(pi_hp_minus_elt)
#         path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
#                                     msg, algo_name
#                                     )
#         if algo_name == ALGOS[0]:
#             # 0: LRI1
#             print("*** ALGO: {} *** ".format(algo_name))
#             utility_function_version = 1
#             path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
#                                     msg, algo_name, str(learning_rate)
#                                     )
#             Path(path_to_save).mkdir(parents=True, exist_ok=True)
#             arr_M_T_K_vars = autoLriGameModel\
#                                 .lri_balanced_player_game_all_pijk_upper_08(
#                                     arr_pl_M_T_vars_init.copy(),
#                                     pi_hp_plus=pi_hp_plus_elt, 
#                                     pi_hp_minus=pi_hp_minus_elt,
#                                     gamma_version=gamma_version,
#                                     k_steps=k_steps, 
#                                     learning_rate=learning_rate,
#                                     p_i_j_ks=p_i_j_ks,
#                                     utility_function_version=utility_function_version,
#                                     path_to_save=path_to_save, 
#                                     manual_debug=manual_debug, dbg=debug)
#         elif algo_name == ALGOS[1]:
#             # LRI2
#             print("*** ALGO: {} *** ".format(algo_name))
#             utility_function_version = 2
#             path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
#                                     msg, algo_name, str(learning_rate)
#                                     )
#             Path(path_to_save).mkdir(parents=True, exist_ok=True)
#             arr_M_T_K_vars = autoLriGameModel\
#                                 .lri_balanced_player_game_all_pijk_upper_08(
#                                     arr_pl_M_T_vars_init.copy(),
#                                     pi_hp_plus=pi_hp_plus_elt, 
#                                     pi_hp_minus=pi_hp_minus_elt,
#                                     gamma_version=gamma_version,
#                                     k_steps=k_steps, 
#                                     learning_rate=learning_rate,
#                                     p_i_j_ks=p_i_j_ks,
#                                     utility_function_version=utility_function_version,
#                                     path_to_save=path_to_save, 
#                                     manual_debug=manual_debug, dbg=debug)
            
#         elif algo_name == ALGOS[2] or algo_name == ALGOS[3]:
#             # 2: DETERMINIST, 3: RANDOM DETERMINIST
#             print("*** ALGO: {} *** ".format(algo_name))
#             random_determinist = False if algo_name == ALGOS[2] else True
#             Path(path_to_save).mkdir(parents=True, exist_ok=True)
#             arr_M_T_vars = autoDetGameModel.determinist_balanced_player_game(
#                              arr_pl_M_T_vars_init.copy(),
#                              pi_hp_plus=pi_hp_plus_elt, 
#                              pi_hp_minus=pi_hp_minus_elt,
#                              gamma_version=gamma_version,
#                              random_determinist=random_determinist,
#                              used_storage=used_storage_det,
#                              path_to_save=path_to_save, 
#                              manual_debug=manual_debug, dbg=debug)
            
#         elif algo_name == fct_aux.ALGO_NAMES_BF[0] :
#             # 0: BEST_BRUTE_FORCE (BF) , 1:BAD_BF, 2: MIDDLE_BF
#             print("*** ALGO: {} *** ".format(algo_name))
#             Path(path_to_save).mkdir(parents=True, exist_ok=True)
#             arr_M_T_vars = autoBfGameModel.bf_balanced_player_game_USE_DICT_MODE_PROFIL(
#                                 arr_pl_M_T_vars_init.copy(),
#                                 pi_hp_plus=pi_hp_plus_elt, 
#                                 pi_hp_minus=pi_hp_minus_elt,
#                                 gamma_version=gamma_version,
#                                 path_to_save=path_to_save, 
#                                 name_dir=name_dir, 
#                                 date_hhmm=date_hhmm,
#                                 manual_debug=manual_debug, 
#                                 criteria_bf=criteria_bf, dbg=debug)
            
#             # arr_M_T_vars = autoBfGameModel_V1.bf_balanced_player_game(
#             #                     arr_pl_M_T_vars_init.copy(),
#             #                     pi_hp_plus=pi_hp_plus_elt, 
#             #                     pi_hp_minus=pi_hp_minus_elt,
#             #                     gamma_version=gamma_version,
#             #                     path_to_save=path_to_save, 
#             #                     name_dir=name_dir, 
#             #                     date_hhmm=date_hhmm,
#             #                     manual_debug=manual_debug, 
#             #                     criteria_bf=criteria_bf, dbg=debug)
                           
#         elif algo_name == fct_aux.ALGO_NAMES_NASH[0] :
#             # 0: "BEST-NASH", 1: "BAD-NASH", 2: "MIDDLE-NASH"
#             print("*** ALGO: {} *** ".format(algo_name))
#             Path(path_to_save).mkdir(parents=True, exist_ok=True)
#             arr_M_T_vars = autoNashGameModel\
#                             .nash_balanced_player_game_perf_t_USE_DICT_MODE_PROFIL(
#                                 arr_pl_M_T_vars_init.copy(),
#                                 pi_hp_plus=pi_hp_plus_elt, 
#                                 pi_hp_minus=pi_hp_minus_elt,
#                                 gamma_version=gamma_version,
#                                 path_to_save=path_to_save, 
#                                 name_dir=name_dir, 
#                                 date_hhmm=date_hhmm,
#                                 manual_debug=manual_debug, 
#                                 dbg=debug)          
        
#     print("NB_EXECUTION cpt={}".format(cpt))
# #_________              USE_DICT_MODE_PROFIL: fin       _______________________ 

#_________                  ONE ALGO: debut                     ______________ 
def execute_algos_used_Generated_instances(arr_pl_M_T_vars_init,
                                            name_dir=None,
                                            date_hhmm=None,
                                            k_steps=None,
                                            NB_REPEAT_K_MAX=None,
                                            algos=None,
                                            learning_rates=None,
                                            pi_hp_plus=None,
                                            pi_hp_minus=None,
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
    name_dir = "tests" if name_dir is None else name_dir
    date_hhmm = datetime.now().strftime("%d%m_%H%M") \
            if date_hhmm is None \
            else date_hhmm
    
    # steps of learning
    k_steps = 5 if k_steps is None else k_steps
    fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    # list of algos
    ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]\
            + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    algos = ALGOS if algos is None \
                  else algos
    # list of pi_hp_plus, pi_hp_minus
    pi_hp_plus = [0.2*pow(10,-3)] if pi_hp_plus is None else pi_hp_plus
    pi_hp_minus = [0.33] if pi_hp_minus is None else pi_hp_minus
    # learning rate 
    learning_rates = [0.01] \
            if learning_rates is None \
            else learning_rates # list(np.arange(0.05, 0.15, step=0.05))
    
    
    
    zip_pi_hp = list(zip(pi_hp_plus, pi_hp_minus))
    
    cpt = 0
    algo_piHpPlusMinus_learningRate \
            = it.product(algos, zip_pi_hp, learning_rates)
    
    for (algo_name, (pi_hp_plus_elt, pi_hp_minus_elt), 
             learning_rate) in algo_piHpPlusMinus_learningRate:
        
        print("______ execution {}: {}, rate={}______".format(cpt, 
                    algo_name, learning_rate))
        cpt += 1
        msg = "pi_hp_plus_"+str(pi_hp_plus_elt)\
                       +"_pi_hp_minus_"+str(pi_hp_minus_elt)
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
        if algo_name == ALGOS[0]:
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
                                    pi_hp_plus=pi_hp_plus_elt, 
                                    pi_hp_minus=pi_hp_minus_elt,
                                    gamma_version=gamma_version,
                                    k_steps=k_steps, 
                                    learning_rate=learning_rate,
                                    p_i_j_ks=p_i_j_ks,
                                    utility_function_version=utility_function_version,
                                    path_to_save=path_to_save, 
                                    manual_debug=manual_debug, dbg=debug)
        elif algo_name == ALGOS[1]:
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
                                    pi_hp_plus=pi_hp_plus_elt, 
                                    pi_hp_minus=pi_hp_minus_elt,
                                    gamma_version=gamma_version,
                                    k_steps=k_steps, 
                                    learning_rate=learning_rate,
                                    p_i_j_ks=p_i_j_ks,
                                    utility_function_version=utility_function_version,
                                    path_to_save=path_to_save, 
                                    manual_debug=manual_debug, dbg=debug)
            
        elif algo_name == ALGOS[2] or algo_name == ALGOS[3]:
            # 2: DETERMINIST, 3: RANDOM DETERMINIST
            print("*** ALGO: {} *** ".format(algo_name))
            random_determinist = False if algo_name == ALGOS[2] else True
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoDetGameModel.determinist_balanced_player_game(
                             arr_pl_M_T_vars_init.copy(),
                             pi_hp_plus=pi_hp_plus_elt, 
                             pi_hp_minus=pi_hp_minus_elt,
                             gamma_version=gamma_version,
                             random_determinist=random_determinist,
                             used_storage=used_storage_det,
                             path_to_save=path_to_save, 
                             manual_debug=manual_debug, dbg=debug)
            
        elif algo_name in fct_aux.ALGO_NAMES_BF :
            # 0: BEST_BRUTE_FORCE (BF) , 1:BAD_BF, 2: MIDDLE_BF
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoBfGameModel_1algoV1\
                            .bf_balanced_player_game_ONE_ALGO(
                                arr_pl_M_T_vars_init.copy(),
                                algo_name,
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                gamma_version=gamma_version,
                                path_to_save=path_to_save, 
                                name_dir=name_dir, 
                                date_hhmm=date_hhmm,
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
                           
        elif algo_name == fct_aux.ALGO_NAMES_NASH[0] :
            # 0: "BEST-NASH", 1: "BAD-NASH", 2: "MIDDLE-NASH"
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoNashGameModel\
                            .nash_balanced_player_game_perf_t_USE_DICT_MODE_PROFIL(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                gamma_version=gamma_version,
                                path_to_save=path_to_save, 
                                name_dir=name_dir, 
                                date_hhmm=date_hhmm,
                                manual_debug=manual_debug, 
                                dbg=debug)          
        
    print("NB_EXECUTION cpt={}".format(cpt))
#_________                   ONE ALGO: fin                     ______________ 

#------------------------------------------------------------------------------
#                   definitions of unittests
#------------------------------------------------------------------------------

def test_execute_algos_used_Generated_instances():
    t_periods = 2
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    t_periods = 3
    setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    gamma_version = 1
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                            setA_m_players, setB_m_players, setC_m_players, 
                            t_periods, 
                            scenario1,
                            path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    
    algos = None
    if arr_pl_M_T_vars_init.shape[0] <= 16:
        algos = ["LRI1", "LRI2", "DETERMINIST"]+ fct_aux.ALGO_NAMES_NASH \
                + fct_aux.ALGO_NAMES_BF 
    else:
        algos = ["LRI1", "LRI2", "DETERMINIST"]
    k_steps = 250
    learning_rates = [0.1]
    execute_algos_used_Generated_instances(
        arr_pl_M_T_vars_init, algos=algos, 
        k_steps=k_steps, 
        learning_rates=learning_rates, 
        gamma_version=gamma_version)
 
#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    
    boolean_execute = True #False
    
    if boolean_execute:
        test_execute_algos_used_Generated_instances()
    
    print("runtime = {}".format(time.time() - ti))