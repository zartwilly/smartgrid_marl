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

# _______      update players p_i_j_k at t and k --> debut       ______________
update_p_i_j_k_by_defined_utility_funtion

# _______      update players p_i_j_k at t and k --> fin         ______________

# _______        balanced players at t and k --> debut          ______________
def balanced_player_game_4_random_mode(arr_pl_M_T_K_vars_modif, t, k, 
                                       pi_0_plus_t, pi_0_minus_t, 
                                       pi_hp_plus, pi_hp_minus, 
                                       random_mode,
                                       manual_debug, dbg):
    
    dico_gamma_players_t_k = dict()
    
    m_players = arr_pl_M_T_K_vars_modif.shape[0]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Si']]
        Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['gamma_i']]
        prod_i, cons_i, r_i = 0, 0, 0
        state_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                             # update R_i_old
        
        # select mode for player num_pl_i
        if random_mode:
            S1_p_i_t_k = arr_pl_M_T_K_vars_modif[num_pl_i, 
                                t, k, 
                                fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]] \
                if k == 0 \
                else arr_pl_M_T_K_vars_modif[num_pl_i, 
                                t, k-1, 
                                fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]]
            pl_i.select_mode_i(p_i=S1_p_i_t_k)
        else:
            mode_i = arr_pl_M_T_K_vars_modif[num_pl_i, 
                                t, k,
                                fct_aux.AUTOMATE_INDEX_ATTRS['mode_i']]
            pl_i.set_mode_i(mode_i)
        
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
                                    fct_aux.AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_K_vars_modif, dico_gamma_players_t_k

def compute_prices_inside_SG(arr_pl_M_T_K_vars_modif, t, k,
                             pi_hp_plus, pi_hp_minus, 
                             pi_0_plus_t, pi_0_minus_t, 
                             manual_debug, dbg):
    """
    compute the prices' and benefits/costs variables: 
        ben_i, cst_i
        b0, c0 
    """
        
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                        arr_pl_M_T_K_vars_modif[:,:,k,:], t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t_k, c0_t_k = fct_aux.compute_energy_unit_price(
                        pi_0_plus_t, pi_0_minus_t, 
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
            bens_t_k, csts_t_k

def balanced_player_game_t(arr_pl_M_T_K_vars_modif, t, k, 
                           pi_hp_plus, pi_hp_minus, 
                           pi_0_plus_t, pi_0_minus_t,
                           m_players, t_periods, 
                           random_mode=True,
                           manual_debug=False, dbg=False):
    
    # find mode, prod, cons, r_i
    arr_pl_M_T_K_vars_modif, dico_gamma_players_t_k \
        = balanced_player_game_4_random_mode(
            arr_pl_M_T_K_vars_modif.copy(), t, k, 
            pi_0_plus_t, pi_0_minus_t, 
            pi_hp_plus, pi_hp_minus,
            random_mode, 
            manual_debug, dbg)
    
    # compute pi_sg_{plus,minus}_t_k, pi_0_{plus,minus}_t_k
    arr_pl_M_T_K_vars_modif, \
    b0_t_k, c0_t_k, \
    bens_t_k, csts_t_k \
        = compute_prices_inside_SG(arr_pl_M_T_K_vars_modif, t, k,
                                     pi_hp_plus, pi_hp_minus, 
                                     pi_0_plus_t, pi_0_minus_t, 
                                     manual_debug, dbg)
        
    return arr_pl_M_T_K_vars_modif, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            dico_gamma_players_t_k
            
# _______        balanced players at t and k --> fin            ______________            


# ____________________ checkout LRI profil --> debut _________________________

# ____________________ checkout LRI profil -->  fin  _________________________

# ______________   turn dico stats into df  --> debut   ______________________

# ______________   turn dico stats into df  -->  fin    ______________________


###############################################################################
#                   definition  de l algo LRI
#
###############################################################################

# ______________       main function of LRI   ---> debut      _________________
def lri_balanced_player_game_all_pijk_upper_08(arr_pl_M_T_vars_init,
                                               pi_hp_plus=0.10, 
                                               pi_hp_minus=0.15,
                                               k_steps=5, 
                                               learning_rate=0.1,
                                               p_i_j_ks=[0.5, 0.5, 0.5],
                                               utility_function_version=1,
                                               path_to_save="tests", 
                                               manual_debug=False, dbg=False):
    """
    algorithm LRI with stopping learning when all players p_i_j_ks are higher 
    than STOP_LEARNING_PROBA = 0.8
    """
    
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_T = np.empty(shape=(t_periods, )); pi_sg_plus_T.fill(np.nan)
    pi_sg_minus_T = np.empty(shape=(t_periods, )); pi_sg_minus_T.fill(np.nan)
    pi_0_plus_T = np.empty(shape=(t_periods, )); pi_0_plus_T.fill(np.nan)
    pi_0_minus_T = np.empty(shape=(t_periods, )); pi_0_minus_T.fill(np.nan)
    b0_s_T_K = np.empty(shape=(t_periods, k_steps)); b0_s_T_K.fill(np.nan)
    c0_s_T_K = np.empty(shape=(t_periods, k_steps)); c0_s_T_K.fill(np.nan)
    BENs_M_T_K = np.empty(shape=(m_players, t_periods, k_steps)); 
    BENs_M_T_K.fill(np.nan)
    CSTs_M_T_K = np.empty(shape=(m_players, t_periods, k_steps)); 
    CSTs_M_T_K.fill(np.nan)
    B_is_M = np.empty(shape=(m_players, )); B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players, )); C_is_M.fill(np.nan)
    
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
    arr_pl_M_T_K_vars[:,:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]] = 0.5
    arr_pl_M_T_K_vars[:,:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S2_p_i_j_k"]] = 0.5
    arr_pl_M_T_K_vars[:,:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
    for num_pl_i in range(0, m_players):
        for t in range(0, t_periods):
            arr_pl_M_T_K_vars[num_pl_i,t,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]] = \
                arr_pl_M_T_K_vars[num_pl_i,t,0,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
            arr_pl_M_T_K_vars[num_pl_i,t,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_max"]] = \
                arr_pl_M_T_K_vars[num_pl_i,t,0,fct_aux.AUTOMATE_INDEX_ATTRS["Si_max"]]
                
    # ____          run balanced sg for all t_periods : debut         ________
    arr_pl_M_T_K_vars_modif = arr_pl_M_T_K_vars.copy()
    
    
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
        
    dico_stats_res = dict()
    dico_k_stop_learnings = dict()
    for t in range(0, t_periods):
        print("******* t = {} BEGIN *******".format(t))
        
        nb_max_reached_repeat_k_per_t = 0
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K #2 
            pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
        else:
            pi_sg_plus_t_minus_1 = pi_sg_plus_t0_minus_1 if t == 0 \
                                                         else pi_sg_plus_t
            pi_sg_minus_t_minus_1 = pi_sg_minus_t0_minus_1 if t == 0 \
                                                            else pi_sg_minus_t
            pi_0_plus_t = round(pi_sg_plus_t_minus_1*pi_hp_plus/pi_hp_minus, 
                                fct_aux.N_DECIMALS)
            pi_0_minus_t = pi_sg_minus_t_minus_1
            if t == 0:
               pi_0_plus_t = 4 
               pi_0_minus_t = 3
               
        arr_pl_M_T_K_vars_modif = fct_aux.compute_gamma_state_4_period_t(
                                arr_pl_M_T_K_vars=arr_pl_M_T_K_vars_modif.copy(), 
                                t=t, 
                                pi_0_plus=pi_0_plus_t, pi_0_minus=pi_0_minus_t,
                                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
                                dbg=dbg)
                
        print("t={}, pi_sg_plus_t={}, pi_sg_minus_t={}, pi_0_plus_t={}, pi_0_minus_t={}".format(
             t, pi_sg_plus_t, pi_sg_minus_t, pi_0_plus_t, pi_0_minus_t))
        
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        
        indices_non_playing_players = []      # indices of non-playing players because bg_min = bg_max
        arr_bg_i_nb_repeat_k = np.empty(
                                    shape=(m_players, fct_aux.NB_REPEAT_K_MAX))
        arr_bg_i_nb_repeat_k.fill(np.nan)
        
        # ____   run balanced sg for one period and all k_steps : debut   _____
        dico_gamma_players_t = dict()
        bool_stop_learning = False
        k_stop_learning = 0
        nb_repeat_k = 0
        k = 0
        while k<k_steps and not bool_stop_learning:
            print(" -------  k = {}, nb_repeat_k = {}  ------- ".format(k, 
                    nb_repeat_k)) if k%50 == 0 else None
            
            ### balanced_player_game_t
            random_mode = True
            arr_pl_M_T_K_vars_modif_new, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            dico_gamma_players_t_k \
                = balanced_player_game_t(
                    arr_pl_M_T_K_vars_modif.copy(), t, k, 
                    pi_hp_plus, pi_hp_minus, 
                    pi_0_plus_t, pi_0_minus_t,
                    m_players, t_periods, 
                    random_mode,
                    manual_debug, dbg=dbg)
            
            ## update variables at each step because they must have to converge in the best case
            #### update b0_s, c0_s of shape (T_PERIODS,K_STEPS) 
            b0_s_T_K[t,k] = b0_t_k
            c0_s_T_K[t,k] = c0_t_k
            #### update BENs, CSTs of shape (M_PLAYERS,T_PERIODS,K_STEPS)
            #### shape: bens_t_k: (M_PLAYERS,)
            BENs_M_T_K[:,t,k] = bens_t_k
            CSTs_M_T_K[:,t,k] = csts_t_k
            
            
            ## compute p_i_j_k of players and compute players' utility
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
                    pi_0_plus_t, pi_0_minus_t,
                    nb_repeat_k,
                    learning_rate, 
                    utility_function_version)
            
            
            
        
        # ____   run balanced sg for one period and all k_steps : fin     _____
        
    # ____          run balanced sg for all t_periods : fin           ________
    
    # _____        save computed variables locally: debut        ______________
    
    # _____        save computed variables locally: fin          ______________
    return arr_pl_M_T_K_vars

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
    # return arr_pl_M_T_vars_init
    
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