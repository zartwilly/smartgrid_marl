# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:44:45 2021

@author: jwehounou
"""
import os
import time
import math

import numpy as np
import pandas as pd
import itertools as it
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

from pathlib import Path

###############################################################################
#                   definition  des fonctions annexes
#
###############################################################################


# ____________________ checkout LRI profil --> debut _________________________

# ____________________ checkout LRI profil -->  fin  _________________________

# ______________   turn dico stats into df  --> debut   ______________________

# ______________   turn dico stats into df  -->  fin    ______________________


###############################################################################
#                   definition  de l algo LRI
#
###############################################################################

# ______________       main function of LRI   ---> debut      _________________


# ______________       main function of LRI   ---> fin        _________________

###############################################################################
#                   definition  des unittests
#
###############################################################################
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
                                        fct_aux.AUTOMATE_INDEX_ATTRS["set"]]
            Pi = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_vars_init[num_pl_i, t, 
                                      fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
            
            if setX == fct_aux.SET_ABC[0]:                                     # setA
                Pis = [2,8]; Cis = [10]; Sis = [3]
                nb_setA_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == fct_aux.SET_ABC[1]:                                   # setB
                Pis = [12,20]; Cis = [20]; Sis = [4]
                nb_setB_t += 1
                cpt_t_Pi_ok += 1 if Pi >= Pis[0] and Pi <= Pis[1] else 0
                cpt_t_Pi_nok += 1 if Pi < Pis[0] or Pi > Pis[1] else 0
                cpt_t_Ci_ok += 1 if Ci in Cis else 0
                cpt_t_Ci_nok += 1 if Ci not in Cis else 0
                cpt_t_Si_ok += 1 if Si in Sis else 0
                cpt_t_Si_nok += 1 if Si not in Sis else 0
            elif setX == fct_aux.SET_ABC[2]:                                   # setC
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
    
    
def test_lri_balanced_player_game_all_pijk_upper_08_Pi_Ci_NEW_AUTOMATE():
    # steps of learning
    k_steps = 250 # 5,250
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    pi_hp_plus = 10 #0.2*pow(10,-3)
    pi_hp_minus = 20 # 0.33
    learning_rate = 0.1
    utility_function_version=1
    
    manual_debug= False #True
    
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    t_periods = 4
    setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                            setA_m_players, setB_m_players, setC_m_players, 
                            t_periods, 
                            scenario1,
                            path_to_arr_pl_M_T, used_instances)
    checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    return arr_pl_M_T_vars_init
    
    arr_pl_M_T_K_vars_modif = lri_balanced_player_game_all_pijk_upper_08(
                                arr_pl_M_T_vars_init,
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
    
    arr_pl_M_T_K_vars_modif \
        = test_lri_balanced_player_game_all_pijk_upper_08_Pi_Ci_NEW_AUTOMATE()
    
    print("runtime = {}".format(time.time() - ti))