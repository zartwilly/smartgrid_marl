# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:12:47 2021

@author: jwehounou
"""
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
    arr_pl_M_T_K_vars[:,:,:,fct_aux.INDEX_ATTRS["p_i_j_k"]] = 0.5
    arr_pl_M_T_K_vars[:,:,:,fct_aux.INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    # ____      run balanced sg for all num_periods at any k_step     ________
    arr_pl_M_T_K_vars_modif = arr_pl_M_T_K_vars.copy()
    
    pi_sg_plus_t, pi_sg_minus_t = None, None
    if manual_debug:
        pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K # 8 
        pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K # 10
    else:
        pi_sg_plus_t, pi_sg_minus_t = 0, 0
        
    for t in range(0, t_periods):
        print("******* t = {} *******".format(t))
        
        # prices at t
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K # 8 
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K # 10
        else:
            pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t
            pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t
        
        pi_sg_plus_t_k, pi_sg_minus_t_k = None, None
        
        indices_non_playing_players = []      # indices of non-playing players because bg_min = bg_max
        arr_bg_i_nb_repeat_k = np.empty(
                                shape=(m_players,fct_aux.NB_REPEAT_K_MAX))
        arr_bg_i_nb_repeat_k.fill(np.nan)
        
        nb_repeat_k = 0
        k = 0
        while k<k_steps:
            print("------- pi_sg_plus_t_k={}, pi_sg_minus_t_k={} -------".format(
                    pi_sg_plus_t_k, pi_sg_minus_t_k)) \
                if dbg else None
            
            pi_sg_plus_t_k = pi_sg_plus_t \
                                if k == 0 \
                                else pi_sg_plus_t_k
            pi_sg_minus_t_k = pi_sg_minus_t \
                                if k == 0 \
                                else pi_sg_minus_t_k
             
            # TODO: a demander a Dominique si a chaque k pi_sg_{plus,minus}_t_k = {8,10}
            # if manual_debug:
            #     pi_sg_plus_t_k = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            #     pi_sg_minus_t_k = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            
            ### balanced_player_game_t
            
            
                                
            
            
    
# ______________       main function of LRI   ---> fin        _________________

###############################################################################
#                   definition  des unittests
#
###############################################################################

###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    arr_pl_M_T_K_vars = test_lri_balanced_player_game()
    
    print("runtime = {}".format(time.time() - ti))