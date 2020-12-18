# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:06:23 2020

@author: jwehounou
"""
import time

import numpy as np
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

#------------------------------------------------------------------------------
#                       definition of functions --> debut
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                       definition of functions --> fin
#
#------------------------------------------------------------------------------

# ______________      main function of brut force   ---> debut      ___________
def bf_balanced_player_game(arr_pl_M_T,
                             pi_hp_plus=0.10, 
                             pi_hp_minus=0.15,
                             m_players=3, 
                             t_periods=4, 
                             k_steps=5,
                             prob_Ci=0.3, 
                             scenario="scenario1",
                             path_to_save="tests", dbg=False):
    """
    brute force algorithm for balanced players' game.
    determine the best solution by enumerating all players' profils.
    The comparison critera is the In_sg-Out_sg value 

    Returns
    -------
    None.

    """
    
    print("determinist game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> debut \n"\
          .format(scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
        
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_sg_plus_T = np.empty(shape=(t_periods,)) #      shape (t_periods,)
    pi_sg_plus_T.fill(np.nan)
    pi_sg_minus_T = np.empty(shape=(t_periods,)) #      shape (t_periods,)
    pi_sg_plus_T.fill(np.nan)
    pi_0_plus_t, pi_0_minus_t = 0, 0
    pi_0_plus_T = np.empty(shape=(t_periods,)) #     shape (t_periods,)
    pi_0_plus_T.fill(np.nan)
    pi_0_minus_T = np.empty(shape=(t_periods,)) #     shape (t_periods,)
    pi_0_minus_T.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    C_is_M.fill(np.nan)
    b0_ts_T = np.empty(shape=(t_periods,)) #   shape (t_periods,)
    b0_ts_T.fil_Tl(np.nan)
    c0_ts_T = np.empty(shape=(t_periods,))
    c0_ts_T.fill(np.nan)
    BENs_T_K = np.empty(shape=(m_players, t_periods)) #   shape (M_PLAYERS, t_periods)
    CSTs_T_K = np.empty(shape=(m_players, t_periods))
    
    
    # _______ variables' initialization --> fin ________________
    
    
    # ____      add initial values for the new attributs ---> debut    _______
    nb_vars_2_add = 6
    fct_aux.INDEX_ATTRS["Si_minus"] = 16
    fct_aux.INDEX_ATTRS["Si_plus"] = 17
    fct_aux.INDEX_ATTRS["prob_mode_state_i"] = 18
    fct_aux.INDEX_ATTRS["u_i"] = 19
    fct_aux.INDEX_ATTRS["bg_i"] = 20
    fct_aux.INDEX_ATTRS["non_playing_players"] = 21
    
    arr_pl_M_T_vars = np.zeros((arr_pl_M_T.shape[0],
                                arr_pl_M_T.shape[1],
                                arr_pl_M_T.shape[2]+nb_vars_2_add), 
                               dtype=object)
    arr_pl_M_T_vars[:,:,:-nb_vars_2_add] = arr_pl_M_T
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_plus"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["u_i"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["bg_i"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prob_mode_state_i"]] = 0.5
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
    
    # ____      add initial values for the new attributs ---> fin    _______
    
    dico_stats_res={}
    
    
# ______________      main function of brut force   ---> fin      ___________
