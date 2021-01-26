# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:51:09 2021

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
import deterministic_game_model_automate as autoDetGameModel
import lri_game_model_automate as autoLriGameModel
import force_brute_game_model_automate as autoBfGameModel
import detection_nash_game_model_automate as autoNashGameModel

from datetime import datetime
from pathlib import Path


#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------
def execute_algos_used_Generated_instances(arr_pl_M_T_vars_init,
                                           name_dir=None,
                                           date_hhmm=None,
                                           k_steps=None,
                                           NB_REPEAT_K_MAX=None,
                                           algos=None,
                                           learning_rates=None,
                                           pi_hp_plus=None,
                                           pi_hp_minus=None,
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
            arr_M_T_K_vars = autoLriGameModel.lri_balanced_player_game(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
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
            arr_M_T_K_vars = autoLriGameModel.lri_balanced_player_game(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                k_steps=k_steps, 
                                learning_rate=learning_rate,
                                p_i_j_ks=p_i_j_ks,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, 
                                manual_debug=manual_debug, dbg=debug)
            
        elif algo_name == ALGOS[2] or algo_name == ALGOS[3]:
            # 2: DETERMINIST, 3: RANDOM DETERMINIST
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoDetGameModel.determinist_balanced_player_game(
                             arr_pl_M_T_vars_init.copy(),
                             pi_hp_plus=pi_hp_plus_elt, 
                             pi_hp_minus=pi_hp_minus_elt,
                             algo_name=algo_name,
                             used_storage=used_storage_det,
                             path_to_save=path_to_save, 
                             manual_debug=manual_debug, dbg=debug)
            
        elif algo_name == fct_aux.ALGO_NAMES_BF[0] \
            or algo_name == fct_aux.ALGO_NAMES_BF[1] \
            or algo_name == fct_aux.ALGO_NAMES_BF[2]:
            # 0: BEST_BRUTE_FORCE (BF) , 1:BAD_BF, 2: MIDDLE_BF
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoBfGameModel.bf_balanced_player_game(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                algo_name=algo_name,
                                path_to_save=path_to_save, 
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
            
        elif algo_name == fct_aux.ALGO_NAMES_NASH[0] \
            or algo_name == fct_aux.ALGO_NAMES_NASH[1] \
            or algo_name == fct_aux.ALGO_NAMES_NASH[2]:
            # 0: "BEST-NASH", 1: "BAD-NASH", 2: "MIDDLE-NASH"
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoNashGameModel.nash_balanced_player_game_perf_t(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                algo_name=algo_name,
                                path_to_save=path_to_save, 
                                manual_debug=manual_debug, 
                                dbg=debug)
        
    print("NB_EXECUTION cpt={}".format(cpt))

#------------------------------------------------------------------------------
#                   definitions of unittests
#------------------------------------------------------------------------------
def test_get_or_create_instance():
    t_periods = 2
    set1_m_players, set2_m_players = 20, 12
    set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    #set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    set1_states, set2_states = None, None
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True#False #True
    
    arr_pl_M_T_vars = fct_aux.get_or_create_instance(
                            set1_m_players, 
                            set2_m_players, 
                           t_periods, 
                           set1_states, 
                           set2_states,
                           set1_stateId0_m_players,
                           set2_stateId0_m_players, 
                           path_to_arr_pl_M_T, used_instances)
    
    t = 0
    # number of players of set1 and set2
    if type(set1_stateId0_m_players) is int:
        set1_stateId0_m_players = set1_stateId0_m_players
    else:
        set1_stateId0_m_players = int(np.rint(set1_m_players
                                              *set1_stateId0_m_players))
    if type(set2_stateId0_m_players) is int:
        set2_stateId0_m_players = set2_stateId0_m_players
    else:
        set2_stateId0_m_players = int(np.rint(set2_m_players
                                              *set2_stateId0_m_players))
        
    nb_players = 0
    for val in ["set1", "set2"]:
        nb_players_setX = len(arr_pl_M_T_vars[
                            arr_pl_M_T_vars[:,t,
                                        fct_aux.AUTOMATE_INDEX_ATTRS["set"]] \
                                    == val])
        nb_players += nb_players_setX
        # # combien de joueurs de set1 sont a l etat state1 or Deficit
        # setX_states = None
        # if val == "set1":
        #     setX_states = [fct_aux.STATES[0], fct_aux.STATES[1]]
        # else:
        #     setX_states = [fct_aux.STATES[1], fct_aux.STATES[2]]
        # for state_i in setX_states:
        #     mask = (arr_pl_M_T_vars[:,t,
        #                 fct_aux.AUTOMATE_INDEX_ATTRS["set"]] == val) \
        #             & (arr_pl_M_T_vars[:,t,
        #                 fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] == state_i)
        #     # Y = {0,1}
        #     arr_setX_stateIdY = arr_pl_M_T_vars[mask]
        #     boolean = True
        #     if state_i == fct_aux.STATES[0]:
        #         nb_players_setX_stateIdY = arr_setX_stateIdY.shape[0]
        #         if nb_players_setX_stateIdY != set1_stateId0_m_players:
        #             print("PROBLEME nb_players_set1_state1 diff")
        #             boolean = False
        #         elif nb_players_setX_stateIdY != set1_stateId0_m_players
            
        
    if nb_players != set1_m_players+set2_m_players: 
        print("PROBLEME: nb_players != set1_m_players+set2_m_players")
    else:
        print("OK no problem in m_players")
    return arr_pl_M_T_vars

def test_execute_algos_used_Generated_instances():
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
    
    algos = ["LRI1", "LRI2"]
    k_steps = 5
    learning_rates = [0.1]
    execute_algos_used_Generated_instances(arr_pl_M_T_vars_init, algos=algos, 
                                           k_steps=k_steps, 
                                           learning_rates=learning_rates)
 
#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    
    boolean_get_create = True
    boolean_execute = False
    
    if boolean_get_create:
        arr_pl_M_T_vars = test_get_or_create_instance()
    if boolean_execute:
        test_execute_algos_used_Generated_instances()
    
    print("runtime = {}".format(time.time() - ti))