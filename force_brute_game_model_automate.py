# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 07:49:45 2021

@author: jwehounou
"""
import os
import time

import numpy as np
import pandas as pd
import itertools as it
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

from datetime import datetime
#------------------------------------------------------------------------------
#                       definition of functions
#
#------------------------------------------------------------------------------

def balanced_player_game_4_mode_profil(arr_pl_M_T_vars_modif, 
                                        mode_profile, t, 
                                        pi_hp_plus, pi_hp_minus,
                                        pi_0_plus_t, pi_0_minus_t, 
                                        manual_debug, dbg):
    """
    attribute modes of all players and get players' variables as prod_i, 
    cons_i, r_i, gamma_i saved to  arr_pl_M_T_vars_mode_prof

    Parameters
    ----------
    arr_pl_M_T_vars : TYPE
        DESCRIPTION.
    dico_balanced_pl_i : TYPE
        DESCRIPTION.

    Returns
    -------
    arr_pl_M_T_vars_mode_prof, dico_balanced_pl_i.

    """
    
    dico_gamma_players_t = dict()
    
    arr_pl_M_T_vars_mode_prof = arr_pl_M_T_vars_modif.copy()
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    t_periods = arr_pl_M_T_vars_modif.shape[1]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_vars_mode_prof[num_pl_i, t, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_vars_mode_prof[num_pl_i, t, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_vars_mode_prof[num_pl_i, t, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS['Si']]
        Si_max = arr_pl_M_T_vars_mode_prof[num_pl_i, t, 
                                        fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i, prod_i, cons_i, r_i = 0, 0, 0, 0
        mode_i = mode_profile[num_pl_i]
        state_i = arr_pl_M_T_vars_mode_prof[num_pl_i, t, 
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        
        pl_i.set_R_i_old(Si_max-Si)
        pl_i.set_mode_i(mode_i)
        pl_i.set_state_i(state_i)
        
        # update prod, cons and r_i
        pl_i.update_prod_cons_r_i()
        
        # is pl_i balanced?
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        
        # compute gamma_i
        Pi_t_plus_1 = arr_pl_M_T_vars_mode_prof[num_pl_i, t+1, 
                                            fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]] \
                        if t+1 < t_periods \
                        else 0
        Ci_t_plus_1 = arr_pl_M_T_vars_mode_prof[num_pl_i, 
                                 t+1, 
                                 fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]] \
                        if t+1 < t_periods \
                        else 0
          
        pl_i.select_storage_politic(
            Ci_t_plus_1 = Ci_t_plus_1, 
            Pi_t_plus_1 = Pi_t_plus_1, 
            pi_0_plus = pi_0_plus_t,
            pi_0_minus = pi_0_minus_t,
            pi_hp_plus = pi_hp_plus, 
            pi_hp_minus = pi_hp_minus)
        
        gamma_i = None
        if manual_debug:
            gamma_i = fct_aux.MANUEL_DBG_GAMMA_I # 5
            pl_i.set_gamma_i(gamma_i)
        else:
            gamma_i = pl_i.get_gamma_i()
        
        dico = dict()
        if gamma_i < min(pi_0_plus_t, pi_0_minus_t)-1:
            dico["min_pi_0"] = gamma_i
        elif gamma_i > max(pi_hp_minus, pi_hp_plus):
            dico["max_pi_hp"] = gamma_i
            
        dico["state_i"] = state_i; dico["mode_i"] = mode_i
        dico["gamma_i"] = gamma_i
        dico_gamma_players_t["player_"+str(num_pl_i)] = dico
        
        # update variables in arr_pl_M_T_modif
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), ("mode_i", mode_i), 
                ("balanced_pl_i", boolean), ("formule", formule), 
                ("gamma_i", gamma_i), 
                ("Si_minus", pl_i.get_Si_minus() ),
                ("Si_plus", pl_i.get_Si_plus() )]
        for col, val in tup_cols_values:
            arr_pl_M_T_vars_mode_prof[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_vars_mode_prof, dico_gamma_players_t
    
def balanced_player_game_4_mode_profil_prices_SG(arr_pl_M_T_vars_modif,
                                        mode_profile, t,
                                        pi_hp_plus, pi_hp_minus,
                                        pi_0_plus_t, pi_0_minus_t,
                                        manual_debug, dbg):
    
    # find mode, prod, cons, r_i
    arr_pl_M_T_vars_mode_prof, dico_gamma_t \
        = balanced_player_game_4_mode_profil(
            arr_pl_M_T_vars_modif.copy(), 
            mode_profile, t, 
            pi_hp_plus, pi_hp_minus,
            pi_0_plus_t, pi_0_minus_t, 
            manual_debug, dbg)
        
    # compute pi_sg_{plus,minus}_t_k, pi_0_{plus,minus}_t_k
    b0_t, c0_t, \
    bens_t, csts_t, \
    pi_sg_plus_t, pi_sg_minus_t \
        = fct_aux.compute_prices_inside_SG(arr_pl_M_T_vars_mode_prof, t,
                                             pi_hp_plus, pi_hp_minus,
                                             pi_0_plus_t, pi_0_minus_t,
                                             manual_debug, dbg)
    
    return arr_pl_M_T_vars_mode_prof, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamma_t


# ______________      main function of brut force   ---> debut      ___________
def bf_balanced_player_game(arr_pl_M_T_vars_init,
                            pi_hp_plus=0.02, 
                            pi_hp_minus=0.33,
                            algo_name="BEST_BF",
                            path_to_save="tests", 
                            manual_debug=False, 
                            criteria_bf="Perf_t", dbg=False):
    
    """
    brute force algorithm for balanced players' game.
    determine the best solution by enumerating all players' profils.
    The comparison critera is the Pref_t value with
    Perf_t = \sum\limits_{1\leq i \leq N}ben_i-cst_i

    Returns
    -------
    None.

    """
    
    print("\n \n {} game: {}, pi_hp_minus ={} ---> debut \n"\
          .format(algo_name, pi_hp_plus, pi_hp_minus))
        
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_sg_plus_T = np.empty(shape=(t_periods,)) #      shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_sg_minus_T = np.empty(shape=(t_periods,)) #      shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_0_plus_t, pi_0_minus_t = 0, 0
    pi_0_plus_T = np.empty(shape=(t_periods,)) #     shape (T_PERIODS,)
    pi_0_plus_T.fill(np.nan)
    pi_0_minus_T = np.empty(shape=(t_periods,)) #     shape (T_PERIODS,)
    pi_0_minus_T.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    C_is_M.fill(np.nan)
    b0_ts_T = np.empty(shape=(t_periods,)) #   shape (T_PERIODS,)
    b0_ts_T.fill(np.nan)
    c0_ts_T = np.empty(shape=(t_periods,))
    c0_ts_T.fill(np.nan)
    BENs_M_T = np.empty(shape=(m_players, t_periods)) #   shape (M_PLAYERS, T_PERIODS)
    CSTs_M_T = np.empty(shape=(m_players, t_periods))    
        
    
    arr_pl_M_T_vars_modif = arr_pl_M_T_vars_init.copy()
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S2_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    # ____      game beginning for all t_period ---> debut      _____
    dico_stats_res = dict()
    
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
    for t in range(0, t_periods):
        print("----- t = {} ------ ".format(t))
        possibles_modes = fct_aux.possibles_modes_players_automate(
                                        arr_pl_M_T_vars_modif.copy(), t=t, k=0)
        print("possibles_modes={}".format(len(possibles_modes)))

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
            
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        
        # balanced player game at instant t    
        mode_profiles = it.product(*possibles_modes)
        
        dico_mode_profs = dict()
        cpt_xxx = 0
        for mode_profile in mode_profiles:
            dico_gamme_t = dict()
            arr_pl_M_T_vars_mode_prof, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamme_t \
                = balanced_player_game_4_mode_profil_prices_SG(
                    arr_pl_M_T_vars_modif.copy(),
                    mode_profile, t,
                    pi_hp_plus, pi_hp_minus,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg)
            
            # compute Perf_t
            Perf_t = np.sum(bens_t - csts_t, axis=0)  
            
            if Perf_t in dico_mode_profs:
                dico_mode_profs[Perf_t].append(mode_profile)
            else:
                dico_mode_profs[Perf_t] = [mode_profile]
             
            cpt_xxx += 1
            if cpt_xxx%5000 == 0:
                print("cpt_xxx={}".format(cpt_xxx))
            
        # max_min_moy_bf
        best_key_Perf_t = None
        if algo_name == fct_aux.ALGO_NAMES_BF[0]:              # BEST-BRUTE-FORCE
            best_key_Perf_t = max(dico_mode_profs.keys())
        elif algo_name == fct_aux.ALGO_NAMES_BF[1]:            # BAD-BRUTE-FORCE
            best_key_Perf_t = min(dico_mode_profs.keys())
        elif algo_name == fct_aux.ALGO_NAMES_BF[2]:            # MIDDLE-BRUTE-FORCE
            mean_key_Perf_t  = np.mean(list(dico_mode_profs.keys()))
            if mean_key_Perf_t in dico_mode_profs.keys():
                best_key_Perf_t = mean_key_Perf_t
            else:
                sorted_keys = sorted(dico_mode_profs.keys())
                boolean = True; i_key = 1
                while boolean:
                    if sorted_keys[i_key] <= mean_key_Perf_t:
                        i_key += 1
                    else:
                        boolean = False; i_key -= 1
                best_key_Perf_t = sorted_keys[i_key]
                
        # find the best, bad, middle key in dico_mode_profs and 
        # the best, bad, middle mode_profile
        best_mode_profiles = dico_mode_profs[best_key_Perf_t]
        best_mode_profile = None
        if len(best_mode_profiles) == 1:
            best_mode_profile = best_mode_profiles[0]
        else:
            rd = np.random.randint(0, len(best_mode_profiles))
            best_mode_profile = best_mode_profiles[rd]
            
        print("cpt_xxx={}, best_key_Perf_t={}, best_mode_profile={}".format(
                cpt_xxx, best_key_Perf_t, best_mode_profile))
        
        arr_pl_M_T_vars_mode_prof_best, \
        b0_t, c0_t, \
        bens_t, csts_t, \
        pi_sg_plus_t, pi_sg_minus_t, \
        dico_gamme_t \
            = balanced_player_game_4_mode_profil_prices_SG(
                arr_pl_M_T_vars_modif.copy(),
                best_mode_profile, t,
                pi_hp_plus, pi_hp_minus,
                pi_0_plus_t, pi_0_minus_t,
                manual_debug, dbg)
        dico_stats_res[t] = dico_gamme_t
        suff = algo_name.split("-")[0]
        dico_stats_res[t] = {"gamma_i": dico_gamme_t,
                             suff+"_mode_profiles": best_mode_profiles,
                             "nb_"+suff+"_mode_profiles": len(best_mode_profiles),
                             suff+"_mode_profile": best_mode_profile,
                             suff+"_Perf_t": best_key_Perf_t }
        
        # verification of best key quality 
        diff = np.abs(best_key_Perf_t-( np.sum(bens_t-csts_t) ))
        print("best_key==bens_t-csts_t --> OK (diff={}) \n".format(diff)) \
            if diff < 0.1 \
            else print("best_key==bens_t-csts_t --> NOK (diff={}) \n".format(diff))
        
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        pi_sg_plus_T[t] = pi_sg_plus_t
        pi_sg_minus_T[t] = pi_sg_minus_t
        
        # b0_ts, c0_ts of shape (T_PERIODS,)
        b0_ts_T[t] = b0_t
        c0_ts_T[t] = c0_t
        
        # BENs, CSTs of shape (M_PLAYERS, T_PERIODS)
        BENs_M_T[:,t] = bens_t
        CSTs_M_T[:,t] = csts_t
        
        arr_pl_M_T_vars_modif[:,t,:] = arr_pl_M_T_vars_mode_prof_best[:,t,:].copy()
        # ____      game beginning for all t_period ---> fin      _____
    
    # __________        compute prices variables         ____________________
    # B_is, C_is of shape (M_PLAYERS, )
    prod_i_M_T = arr_pl_M_T_vars_modif[:,:, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
    cons_i_M_T = arr_pl_M_T_vars_modif[:,:, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
    B_is_M = np.sum(b0_ts_T * prod_i_M_T, axis=1)
    C_is_M = np.sum(c0_ts_T * cons_i_M_T, axis=1)
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is_M_T = np.sum(arr_pl_M_T_vars_modif[:,:, 
                                         fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]], 
                         axis=1)
    PROD_is_M_T = np.sum(arr_pl_M_T_vars_modif[:,:, 
                                         fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]], 
                         axis=1)
    BB_is_M = pi_sg_plus_T[-1] * PROD_is_M_T #np.sum(PROD_is)
    for num_pl, bb_i in enumerate(BB_is_M):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    CC_is_M = pi_sg_minus_T[-1] * CONS_is_M_T #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * t_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * t_periods, dtype=object) 
    
    #__________      save computed variables locally      _____________________
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif, 
                   b0_ts_T, c0_ts_T, B_is_M, C_is_M, 
                   BENs_M_T, CSTs_M_T, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T, pi_sg_plus_T, 
                   pi_0_minus_T, pi_0_plus_T,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo=algo_name,
                   dico_best_steps=dict())
    
    return arr_pl_M_T_vars_modif
    
def bf_balanced_player_game_USE_DICT_MODE_PROFIL_OLD(arr_pl_M_T_vars_init,
                            pi_hp_plus=0.02, 
                            pi_hp_minus=0.33,
                            path_to_save="tests", 
                            manual_debug=False, 
                            criteria_bf="Perf_t", dbg=False):
    
    """
    brute force algorithm for balanced players' game.
    determine the best solution by enumerating all players' profils.
    The comparison critera is the Pref_t value with
    Perf_t = \sum\limits_{1\leq i \leq N}ben_i-cst_i

    Returns
    -------
    None.

    """
    
    print("\n \n game: {}, pi_hp_minus ={} ---> debut \n"\
          .format( pi_hp_plus, pi_hp_minus))
        
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]

    arr_pl_M_T_vars_modif = arr_pl_M_T_vars_init.copy()
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    # ____      game beginning for all t_period ---> debut      _____
    dico_stats_res = dict()
    
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
    dico_mode_profs_T = dict()
    for t in range(0, t_periods):
        print("----- t = {} ------ ".format(t))
        possibles_modes = fct_aux.possibles_modes_players_automate(
                                        arr_pl_M_T_vars_modif.copy(), t=t, k=0)
        print("possibles_modes={}".format(len(possibles_modes)))

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
            
        # balanced player game at instant t    
        mode_profiles = it.product(*possibles_modes)
        
        dico_mode_profs = dict()
        cpt_xxx = 0
        for mode_profile in mode_profiles:
            dico_gamme_t = dict()
            arr_pl_M_T_vars_mode_prof, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamme_t \
                = balanced_player_game_4_mode_profil_prices_SG(
                    arr_pl_M_T_vars_modif.copy(),
                    mode_profile, t,
                    pi_hp_plus, pi_hp_minus,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg)
            
            # compute Perf_t
            Perf_t = np.sum(bens_t - csts_t, axis=0)  
            
            if Perf_t in dico_mode_profs:
                dico_mode_profs[Perf_t].append(mode_profile)
            else:
                dico_mode_profs[Perf_t] = [mode_profile]
             
            cpt_xxx += 1
            if cpt_xxx%5000 == 0:
                print("cpt_xxx={}".format(cpt_xxx))
        dico_mode_profs_T[t] = dico_mode_profs
        print(" t={}, cpt_xxx={}".format(t, cpt_xxx()))
                
        for algo_name in fct_aux.ALGO_NAMES_BF:
            # ____    variables' initialization for alga_name --> debut   _____
            pi_sg_plus_t, pi_sg_minus_t = 0, 0
            pi_sg_plus_T = np.empty(shape=(t_periods,)) #      shape (T_PERIODS,)
            pi_sg_plus_T.fill(np.nan)
            pi_sg_minus_T = np.empty(shape=(t_periods,)) #      shape (T_PERIODS,)
            pi_sg_plus_T.fill(np.nan)
            pi_0_plus_t, pi_0_minus_t = 0, 0
            pi_0_plus_T = np.empty(shape=(t_periods,)) #     shape (T_PERIODS,)
            pi_0_plus_T.fill(np.nan)
            pi_0_minus_T = np.empty(shape=(t_periods,)) #     shape (T_PERIODS,)
            pi_0_minus_T.fill(np.nan)
            B_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
            B_is_M.fill(np.nan)
            C_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
            C_is_M.fill(np.nan)
            b0_ts_T = np.empty(shape=(t_periods,)) #   shape (T_PERIODS,)
            b0_ts_T.fill(np.nan)
            c0_ts_T = np.empty(shape=(t_periods,))
            c0_ts_T.fill(np.nan)
            BENs_M_T = np.empty(shape=(m_players, t_periods)) #   shape (M_PLAYERS, T_PERIODS)
            CSTs_M_T = np.empty(shape=(m_players, t_periods))  
                
            pi_0_plus_T[t] = pi_0_plus_t
            pi_0_minus_T[t] = pi_0_minus_t
            
            # max_min_moy_bf
            best_key_Perf_t = None
            if algo_name == fct_aux.ALGO_NAMES_BF[0]:              # BEST-BRUTE-FORCE
                best_key_Perf_t = max(dico_mode_profs.keys())
            elif algo_name == fct_aux.ALGO_NAMES_BF[1]:            # BAD-BRUTE-FORCE
                best_key_Perf_t = min(dico_mode_profs.keys())
            elif algo_name == fct_aux.ALGO_NAMES_BF[2]:            # MIDDLE-BRUTE-FORCE
                mean_key_Perf_t  = np.mean(list(dico_mode_profs.keys()))
                if mean_key_Perf_t in dico_mode_profs.keys():
                    best_key_Perf_t = mean_key_Perf_t
                else:
                    sorted_keys = sorted(dico_mode_profs.keys())
                    boolean = True; i_key = 1
                    while boolean:
                        if sorted_keys[i_key] <= mean_key_Perf_t:
                            i_key += 1
                        else:
                            boolean = False; i_key -= 1
                    best_key_Perf_t = sorted_keys[i_key]
                    
            # find the best, bad, middle key in dico_mode_profs and 
            # the best, bad, middle mode_profile
            best_mode_profiles = dico_mode_profs[best_key_Perf_t]
            best_mode_profile = None
            if len(best_mode_profiles) == 1:
                best_mode_profile = best_mode_profiles[0]
            else:
                rd = np.random.randint(0, len(best_mode_profiles))
                best_mode_profile = best_mode_profiles[rd]
                
            print("cpt_xxx={}, best_key_Perf_t={}, best_mode_profile={}".format(
                    cpt_xxx, best_key_Perf_t, best_mode_profile))
            
            arr_pl_M_T_vars_mode_prof_best, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamme_t \
                = balanced_player_game_4_mode_profil_prices_SG(
                    arr_pl_M_T_vars_modif.copy(),
                    best_mode_profile, t,
                    pi_hp_plus, pi_hp_minus,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg)
            dico_stats_res[t] = dico_gamme_t
            suff = algo_name.split("-")[0]
            dico_stats_res[t] = {"gamma_i": dico_gamme_t,
                                 suff+"_mode_profiles": best_mode_profiles,
                                 "nb_"+suff+"_mode_profiles": len(best_mode_profiles),
                                 suff+"_mode_profile": best_mode_profile,
                                 suff+"_Perf_t": best_key_Perf_t }
            
            # verification of best key quality 
            diff = np.abs(best_key_Perf_t-( np.sum(bens_t-csts_t) ))
            print("best_key==bens_t-csts_t --> OK (diff={}) \n".format(diff)) \
                if diff < 0.1 \
                else print("best_key==bens_t-csts_t --> NOK (diff={}) \n".format(diff))
            
            # pi_sg_{plus,minus} of shape (T_PERIODS,)
            pi_sg_plus_T[t] = pi_sg_plus_t
            pi_sg_minus_T[t] = pi_sg_minus_t
            
            # b0_ts, c0_ts of shape (T_PERIODS,)
            b0_ts_T[t] = b0_t
            c0_ts_T[t] = c0_t
            
            # BENs, CSTs of shape (M_PLAYERS, T_PERIODS)
            BENs_M_T[:,t] = bens_t
            CSTs_M_T[:,t] = csts_t
            
            arr_pl_M_T_vars_modif[:,t,:] = arr_pl_M_T_vars_mode_prof_best[:,t,:].copy()
            # ____      game beginning for all t_period ---> fin      _____
                
            
    



def bf_balanced_player_game_USE_DICT_MODE_PROFIL(arr_pl_M_T_vars_init,
                            pi_hp_plus=0.02, 
                            pi_hp_minus=0.33,
                            path_to_save="tests", 
                            manual_debug=False, 
                            criteria_bf="Perf_t", dbg=False):
    
    """
    brute force algorithm for balanced players' game.
    determine the best solution by enumerating all players' profils.
    The comparison critera is the Pref_t value with
    Perf_t = \sum\limits_{1\leq i \leq N}ben_i-cst_i

    NB : I use the same dico_mode_profs to show the bad, middle best Brute Force
    Returns
    -------
    None.

    """
    
    print("\n \n game: {}, pi_hp_minus ={} ---> debut \n"\
          .format(pi_hp_plus, pi_hp_minus))
        
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_sg_plus_T = np.empty(shape=(t_periods,)) #      shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_sg_minus_T = np.empty(shape=(t_periods,)) #      shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_0_plus_t, pi_0_minus_t = 0, 0
    pi_0_plus_T = np.empty(shape=(t_periods,)) #     shape (T_PERIODS,)
    pi_0_plus_T.fill(np.nan)
    pi_0_minus_T = np.empty(shape=(t_periods,)) #     shape (T_PERIODS,)
    pi_0_minus_T.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    C_is_M.fill(np.nan)
    b0_ts_T = np.empty(shape=(t_periods,)) #   shape (T_PERIODS,)
    b0_ts_T.fill(np.nan)
    c0_ts_T = np.empty(shape=(t_periods,))
    c0_ts_T.fill(np.nan)
    BENs_M_T = np.empty(shape=(m_players, t_periods)) #   shape (M_PLAYERS, T_PERIODS)
    CSTs_M_T = np.empty(shape=(m_players, t_periods))
    CC_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    CC_is_M.fill(np.nan)
    BB_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    BB_is_M.fill(np.nan)
    RU_is_M = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    RU_is_M.fill(np.nan)    
    
    arr_pl_M_T_vars_modif = arr_pl_M_T_vars_init.copy()
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S2_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    # ____      game beginning for all t_period ---> debut      _____
    dico_stats_res = dict()
    
    arr_pl_M_T_vars_modif_BADBF = None
    arr_pl_M_T_vars_modif_BESTBF = None
    arr_pl_M_T_vars_modif_MIDBF = None
    
    pi_sg_plus_T_BESTBF = pi_sg_plus_T.copy()
    pi_sg_minus_T_BESTBF = pi_sg_minus_T.copy()
    pi_0_plus_T_BESTBF = pi_0_plus_T.copy()
    pi_0_minus_T_BESTBF = pi_0_minus_T.copy()
    B_is_M_BESTBF = B_is_M.copy()
    C_is_M_BESTBF = C_is_M.copy()
    b0_ts_T_BESTBF = b0_ts_T.copy()
    c0_ts_T_BESTBF = c0_ts_T.copy()
    BENs_M_T_BESTBF = BENs_M_T.copy()
    CSTs_M_T_BESTBF = CSTs_M_T.copy()
    CC_is_M_BESTBF = CC_is_M.copy()
    BB_is_M_BESTBF = BB_is_M.copy()
    RU_is_M_BESTBF = RU_is_M.copy()
    C_is_M_BESTBF = CC_is_M.copy()
    B_is_M_BESTBF = BB_is_M.copy()
    dico_stats_res_BESTBF = dict()
    
    pi_sg_plus_T_BADBF = pi_sg_plus_T.copy()
    pi_sg_minus_T_BADBF = pi_sg_minus_T.copy()
    pi_0_plus_T_BADBF = pi_0_plus_T.copy()
    pi_0_minus_T_BADBF = pi_0_minus_T.copy()
    B_is_M_BADBF = B_is_M.copy()
    C_is_M_BADBF = C_is_M.copy()
    b0_ts_T_BADBF = b0_ts_T.copy()
    c0_ts_T_BADBF = c0_ts_T.copy()
    BENs_M_T_BADBF = BENs_M_T.copy()
    CSTs_M_T_BADBF = CSTs_M_T.copy()
    CC_is_M_BADBF = CC_is_M.copy()
    BB_is_M_BADBF = BB_is_M.copy()
    RU_is_M_BADBF = RU_is_M.copy()
    C_is_M_BADBF = CC_is_M.copy()
    B_is_M_BADBF = BB_is_M.copy()
    dico_stats_res_BADBF = dict()
    
    pi_sg_plus_T_MIDBF = pi_sg_plus_T.copy()
    pi_sg_minus_T_MIDBF = pi_sg_minus_T.copy()
    pi_0_plus_T_MIDBF = pi_0_plus_T.copy()
    pi_0_minus_T_MIDBF = pi_0_minus_T.copy()
    B_is_M_MIDBF = B_is_M.copy()
    C_is_M_MIDBF = C_is_M.copy()
    b0_ts_T_MIDBF = b0_ts_T.copy()
    c0_ts_T_MIDBF = c0_ts_T.copy()
    BENs_M_T_MIDBF = BENs_M_T.copy()
    CSTs_M_T_MIDBF = CSTs_M_T.copy()
    CC_is_M_MIDBF = CC_is_M.copy()
    BB_is_M_MIDBF = BB_is_M.copy()
    RU_is_M_MIDBF = RU_is_M.copy()
    C_is_M_MIDBF = CC_is_M.copy()
    B_is_M_MIDBF = BB_is_M.copy()
    dico_stats_res_MIDBF = dict()
    
    arr_pl_M_T_vars_modif_BADBF = arr_pl_M_T_vars_modif.copy()
    arr_pl_M_T_vars_modif_BESTBF = arr_pl_M_T_vars_modif.copy()
    arr_pl_M_T_vars_modif_MIDBF = arr_pl_M_T_vars_modif.copy()
    
    dico_mode_prof_by_players_T = dict()
    
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
    pi_hp_plus_s = np.array([pi_hp_plus] * t_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * t_periods, dtype=object)
    for t in range(0, t_periods):
        print("----- t = {} ------ ".format(t))
        possibles_modes = fct_aux.possibles_modes_players_automate(
                                        arr_pl_M_T_vars_modif.copy(), t=t, k=0)
        print("possibles_modes={}".format(len(possibles_modes)))

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
               pi_0_plus_t = 2
               pi_0_minus_t = 2
            
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        
        # balanced player game at instant t    
        mode_profiles = it.product(*possibles_modes)
        
        dico_mode_profs = dict()
        dico_modes_profs_by_players = dict()
        cpt_xxx = 0
        for mode_profile in mode_profiles:
            dico_gamme_t = dict()
            arr_pl_M_T_vars_mode_prof, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamme_t \
                = balanced_player_game_4_mode_profil_prices_SG(
                    arr_pl_M_T_vars_modif.copy(),
                    mode_profile, t,
                    pi_hp_plus, pi_hp_minus,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg)
            
            In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                                arr_pl_M_T_vars_mode_prof, 
                                t)
            
            # compute Perf_t
            bens_csts_t = bens_t - csts_t
            Perf_t = np.sum(bens_csts_t, axis=0)
            dico_mode_prof_by_players = dict()
            for num_pl_i in range(0, m_players):
                Vi = bens_csts_t[num_pl_i]
                state_i = arr_pl_M_T_vars_mode_prof[
                            num_pl_i, t, 
                            fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
                mode_i = arr_pl_M_T_vars_mode_prof[
                            num_pl_i, t, 
                            fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]]
                dico_mode_prof_by_players["PLAYER_"+str(num_pl_i)+"_t_"+str(t)] \
                = (state_i, mode_i, Vi)
                
                dico_vars = dict()
                dico_vars["Vi"] = round(Vi, 2)
                dico_vars["ben_i"] = round(bens_t[num_pl_i], 2)
                dico_vars["cst_i"] = round(csts_t[num_pl_i], 2)
                variables = ["state_i", "mode_i", "prod_i", "cons_i", "r_i", 
                             "gamma_i", "Pi", "Ci", "Si", "Si_old", 
                             "Si_minus", "Si_plus"]
                for variable in variables:
                    dico_vars[variable] = arr_pl_M_T_vars_mode_prof[
                                            num_pl_i, t, 
                                            fct_aux.AUTOMATE_INDEX_ATTRS[variable]]
                    
                dico_mode_prof_by_players_T["PLAYER_"
                                            +str(num_pl_i)
                                            +"_t_"+str(t)
                                            +"_"+str(cpt_xxx)] \
                    = dico_vars
                
            dico_mode_prof_by_players_T["Perf_t_"+str(t)+"_"+str(cpt_xxx)] = round(Perf_t, 2)
            dico_mode_prof_by_players_T["b0_t_"+str(t)+"_"+str(cpt_xxx)] = round(b0_t,2)
            dico_mode_prof_by_players_T["c0_t_"+str(t)+"_"+str(cpt_xxx)] = round(c0_t,2)
            dico_mode_prof_by_players_T["Out_sg_"+str(t)+"_"+str(cpt_xxx)] = round(Out_sg,2)
            dico_mode_prof_by_players_T["In_sg_"+str(t)+"_"+str(cpt_xxx)] = round(In_sg,2)
            dico_mode_prof_by_players_T["pi_sg_plus_t_"+str(t)+"_"+str(cpt_xxx)] = round(pi_sg_plus_t,2)
            dico_mode_prof_by_players_T["pi_sg_minus_t_"+str(t)+"_"+str(cpt_xxx)] = round(pi_sg_minus_t,2)
            dico_mode_prof_by_players_T["pi_0_plus_t_"+str(t)+"_"+str(cpt_xxx)] = round(pi_0_plus_t,2)
            dico_mode_prof_by_players_T["pi_0_minus_t_"+str(t)+"_"+str(cpt_xxx)] = round(pi_0_minus_t,2)
            
            if Perf_t in dico_mode_profs:
                dico_mode_profs[Perf_t].append(mode_profile)
                dico_modes_profs_by_players[Perf_t].append(
                    ("BF"+str(cpt_xxx), dico_mode_prof_by_players))
            else:
                dico_mode_profs[Perf_t] = [mode_profile]
                dico_modes_profs_by_players[Perf_t] \
                    = [ ("BF"+str(cpt_xxx), dico_mode_prof_by_players) ]
             
            cpt_xxx += 1
            if cpt_xxx%5000 == 0:
                print("cpt_xxx={}".format(cpt_xxx))
        
        arr_pl_M_T_vars_modif_algo = None
        b0_ts_T_algo, c0_ts_T_algo = None, None
        BENs_M_T_algo, CSTs_M_T_algo = None, None
        pi_0_plus_T_algo, pi_0_minus_T_algo = None, None
        for algo_name in fct_aux.ALGO_NAMES_BF:
            
            if algo_name == fct_aux.ALGO_NAMES_BF[0]:              # BEST-BRUTE-FORCE
                arr_pl_M_T_vars_modif_algo = arr_pl_M_T_vars_modif_BESTBF.copy()
                pi_sg_plus_T_algo = pi_sg_plus_T_BESTBF.copy()
                pi_sg_minus_T_algo = pi_sg_minus_T_BESTBF.copy()
                b0_ts_T_algo = b0_ts_T_BESTBF.copy()
                c0_ts_T_algo = c0_ts_T_BESTBF.copy()
                BENs_M_T_algo = BENs_M_T_BESTBF.copy()
                CSTs_M_T_algo = CSTs_M_T_BESTBF.copy()
                pi_0_plus_T_algo = pi_0_plus_T_BESTBF.copy() 
                pi_0_minus_T_algo = pi_0_minus_T_BESTBF.copy()
                dico_stats_res_algo = dico_stats_res_BESTBF.copy()
                               
            elif algo_name == fct_aux.ALGO_NAMES_BF[1]:            # BAD-BRUTE-FORCE
                arr_pl_M_T_vars_modif_algo = arr_pl_M_T_vars_modif_BADBF.copy()
                pi_sg_plus_T_algo = pi_sg_plus_T_BADBF.copy()
                pi_sg_minus_T_algo = pi_sg_minus_T_BADBF.copy()
                b0_ts_T_algo = b0_ts_T_BADBF.copy()
                c0_ts_T_algo = c0_ts_T_BADBF.copy()
                BENs_M_T_algo = BENs_M_T_BADBF.copy()
                CSTs_M_T_algo = CSTs_M_T_BADBF.copy()
                pi_0_plus_T_algo = pi_0_plus_T_BADBF.copy()
                pi_0_minus_T_algo = pi_0_minus_T_BADBF.copy()
                dico_stats_res_algo = dico_stats_res_BADBF.copy()
                
            elif algo_name == fct_aux.ALGO_NAMES_BF[2]:            # MIDDLE-BRUTE-FORCE
                arr_pl_M_T_vars_modif_algo = arr_pl_M_T_vars_modif_MIDBF.copy()
                pi_sg_plus_T_algo = pi_sg_plus_T_MIDBF.copy()
                pi_sg_minus_T_algo = pi_sg_minus_T_MIDBF.copy()
                b0_ts_T_algo = b0_ts_T_MIDBF.copy()
                c0_ts_T_algo = c0_ts_T_MIDBF.copy()
                BENs_M_T_algo = BENs_M_T_MIDBF.copy()
                CSTs_M_T_algo = CSTs_M_T_MIDBF.copy()
                pi_0_plus_T_algo = pi_0_plus_T_MIDBF.copy()
                pi_0_minus_T_algo = pi_0_minus_T_MIDBF.copy()
                dico_stats_res_algo = dico_stats_res_MIDBF.copy()
            
            
            # ____      game beginning for one algo for t ---> debut      _____
            # max_min_moy_bf
            best_key_Perf_t = None
            if algo_name == fct_aux.ALGO_NAMES_BF[0]:              # BEST-BRUTE-FORCE
                best_key_Perf_t = max(dico_mode_profs.keys())
            elif algo_name == fct_aux.ALGO_NAMES_BF[1]:            # BAD-BRUTE-FORCE
                best_key_Perf_t = min(dico_mode_profs.keys())
            elif algo_name == fct_aux.ALGO_NAMES_BF[2]:            # MIDDLE-BRUTE-FORCE
                mean_key_Perf_t  = np.mean(list(dico_mode_profs.keys()))
                if mean_key_Perf_t in dico_mode_profs.keys():
                    best_key_Perf_t = mean_key_Perf_t
                else:
                    sorted_keys = sorted(dico_mode_profs.keys())
                    boolean = True; i_key = 1
                    while boolean:
                        if sorted_keys[i_key] <= mean_key_Perf_t:
                            i_key += 1
                        else:
                            boolean = False; i_key -= 1
                    best_key_Perf_t = sorted_keys[i_key]
                
            # find the best, bad, middle key in dico_mode_profs and 
            # the best, bad, middle mode_profile
            best_mode_profiles = dico_mode_profs[best_key_Perf_t]
            list_dico_best_mode_profs = dico_modes_profs_by_players[best_key_Perf_t]
            best_mode_profile = None
            if len(best_mode_profiles) == 1:
                best_mode_profile = best_mode_profiles[0]
            else:
                rd = np.random.randint(0, len(best_mode_profiles))
                best_mode_profile = best_mode_profiles[rd]
                
            print("cpt_xxx={}, best_key_Perf_t={}, {}_mode_profile={}".format(
                    cpt_xxx, best_key_Perf_t, algo_name.split("-")[0], 
                    best_mode_profile))
        
            arr_pl_M_T_vars_mode_prof_best_algo, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamme_t \
                = balanced_player_game_4_mode_profil_prices_SG(
                    arr_pl_M_T_vars_modif_algo.copy(),
                    best_mode_profile, t,
                    pi_hp_plus, pi_hp_minus,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg)
            dico_stats_res[t] = dico_gamme_t
            suff = algo_name.split("-")[0]
            dico_stats_res_algo[t] = {"gamma_i": dico_gamme_t,
                                 suff+"_mode_profiles": best_mode_profiles,
                                 "nb_"+suff+"_mode_profiles": len(best_mode_profiles),
                                 suff+"_mode_profile": best_mode_profile,
                                 suff+"_Perf_t": best_key_Perf_t,
                                 "list_dico_best_mode_profs": 
                                     list_dico_best_mode_profs
                                }
        
            # verification of best key quality 
            diff = np.abs(best_key_Perf_t-( np.sum(bens_t-csts_t) ))
            print("best_key==bens_t-csts_t --> OK (diff={}) ".format(diff)) \
                if diff < 0.1 \
                else print("best_key==bens_t-csts_t --> NOK (diff={}) \n".format(diff))
            In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                                arr_pl_M_T_vars_mode_prof_best_algo, t)
            print("b0_t={}, c0_t={}, Out_sg={},In_sg={} \n".format(b0_t, c0_t, Out_sg, In_sg))
            
            # pi_sg_{plus,minus} of shape (T_PERIODS,)
            pi_sg_plus_T_algo[t] = pi_sg_plus_t
            pi_sg_minus_T_algo[t] = pi_sg_minus_t
            pi_0_plus_T_algo[t] = pi_0_plus_t
            pi_0_minus_T_algo[t] = pi_0_minus_t
            
            # b0_ts, c0_ts of shape (T_PERIODS,)
            b0_ts_T_algo[t] = b0_t
            c0_ts_T_algo[t] = c0_t
            
            # BENs, CSTs of shape (M_PLAYERS, T_PERIODS)
            BENs_M_T_algo[:,t] = bens_t
            CSTs_M_T_algo[:,t] = csts_t
            
            # ____      game beginning for one algo for t ---> debut      _____
    
            # __________        compute prices variables         ____________________
            # B_is, C_is of shape (M_PLAYERS, )
            prod_i_M_T_algo = arr_pl_M_T_vars_modif_algo[:,:, 
                                               fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
            cons_i_M_T_algo = arr_pl_M_T_vars_modif_algo[:,:, 
                                               fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
            B_is_M_algo = np.sum(b0_ts_T_algo * prod_i_M_T_algo, axis=1)
            C_is_M_algo = np.sum(c0_ts_T_algo * cons_i_M_T_algo, axis=1)
            
            # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
            CONS_is_M_T_algo = np.sum(arr_pl_M_T_vars_modif_algo[:,:, 
                                                 fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]], 
                                 axis=1)
            PROD_is_M_T_algo = np.sum(arr_pl_M_T_vars_modif_algo[:,:, 
                                                 fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]], 
                                 axis=1)
            BB_is_M_algo = pi_sg_plus_T_algo[-t] * PROD_is_M_T_algo #np.sum(PROD_is)
            for num_pl, bb_i in enumerate(BB_is_M_algo):
                if bb_i != 0:
                    print("player {}, BB_i={}".format(num_pl, bb_i))
            CC_is_M_algo = pi_sg_minus_T_algo[t] * CONS_is_M_T_algo #np.sum(CONS_is)
            RU_is_M_algo = BB_is_M_algo - CC_is_M_algo
            
            pi_hp_plus_s = np.array([pi_hp_plus] * t_periods, dtype=object)
            pi_hp_minus_s = np.array([pi_hp_minus] * t_periods, dtype=object) 
            # # __________        compute prices variables         ____________________
            
            # __________        maj arrays for all algos: debut    ____________
            if algo_name == fct_aux.ALGO_NAMES_BF[0]:                           # BEST-BRUTE-FORCE
                arr_pl_M_T_vars_modif_BESTBF = arr_pl_M_T_vars_mode_prof_best_algo.copy()
                pi_sg_plus_T_BESTBF[t] = pi_sg_plus_T_algo[t]
                pi_sg_minus_T_BESTBF[t] = pi_sg_minus_T_algo[t]
                pi_0_plus_T_BESTBF[t] = pi_0_plus_T_algo[t]
                pi_0_minus_T_BESTBF[t] = pi_0_minus_T_algo[t]
                B_is_M_BESTBF = B_is_M_algo.copy()
                C_is_M_BESTBF = C_is_M_algo.copy()
                b0_ts_T_BESTBF = b0_ts_T_algo.copy()
                c0_ts_T_BESTBF = c0_ts_T_algo.copy()
                BENs_M_T_BESTBF = BENs_M_T_algo.copy()
                CSTs_M_T_BESTBF = CSTs_M_T_algo.copy()
                CC_is_M_BESTBF = CC_is_M_algo.copy()
                BB_is_M_BESTBF = BB_is_M_algo.copy()
                RU_is_M_BESTBF = RU_is_M_algo.copy()
                dico_stats_res_BESTBF = dico_stats_res_algo.copy()
            elif algo_name == fct_aux.ALGO_NAMES_BF[1]:                         # BAD-BRUTE-FORCE
                arr_pl_M_T_vars_modif_BADBF = arr_pl_M_T_vars_mode_prof_best_algo.copy()
                pi_sg_plus_T_BADBF[t] = pi_sg_plus_T_algo[t]
                pi_sg_minus_T_BADBF[t] = pi_sg_minus_T_algo[t]
                pi_0_plus_T_BADBF[t] = pi_0_plus_T_algo[t]
                pi_0_minus_T_BADBF[t] = pi_0_minus_T_algo[t]
                B_is_M_BADBF = B_is_M_algo.copy()
                C_is_M_BADBF = C_is_M_algo.copy()
                b0_ts_T_BADBF = b0_ts_T_algo.copy()
                c0_ts_T_BADBF = c0_ts_T_algo.copy()
                BENs_M_T_BADBF = BENs_M_T_algo.copy()
                CSTs_M_T_BADBF = CSTs_M_T_algo.copy()
                CC_is_M_BADBF = CC_is_M_algo.copy()
                BB_is_M_BADBF = BB_is_M_algo.copy()
                RU_is_M_BADBF = RU_is_M_algo.copy()
                dico_stats_res_BADBF = dico_stats_res_algo.copy()
            elif algo_name == fct_aux.ALGO_NAMES_BF[2]:                         # MIDDLE-BRUTE-FORCE
                arr_pl_M_T_vars_modif_MIDBF = arr_pl_M_T_vars_mode_prof_best_algo.copy()
                pi_sg_plus_T_MIDBF[t] = pi_sg_plus_T_algo[t]
                pi_sg_minus_T_MIDBF[t] = pi_sg_minus_T_algo[t]
                pi_0_plus_T_MIDBF[t] = pi_0_plus_T_algo[t]
                pi_0_minus_T_MIDBF[t] = pi_0_minus_T_algo[t]
                B_is_M_MIDBF = B_is_M_algo.copy()
                C_is_M_MIDBF = C_is_M_algo.copy()
                b0_ts_T_MIDBF = b0_ts_T_algo.copy()
                c0_ts_T_MIDBF = c0_ts_T_algo.copy()
                BENs_M_T_MIDBF = BENs_M_T_algo.copy()
                CSTs_M_T_MIDBF = CSTs_M_T_algo.copy()
                CC_is_M_MIDBF = CC_is_M_algo.copy()
                BB_is_M_MIDBF = BB_is_M_algo.copy()
                RU_is_M_MIDBF = RU_is_M_algo.copy()
                dico_stats_res_MIDBF = dico_stats_res_algo.copy()
            # __________        maj arrays for all algos: fin      ____________
    
    # ____      game beginning for all t_period ---> fin      _____

    #_______      save computed variables locally from algo_name     __________
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    name_dir = "tests"; date_hhmm = "DDMM_HHMM"
    
    print("path_to_save={}".format(path_to_save))
    algo_name = fct_aux.ALGO_NAMES_BF[0]
    if "simu_DDMM_HHMM" in path_to_save:
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_BESTBF, 
                   b0_ts_T_BESTBF, c0_ts_T_BESTBF, B_is_M_BESTBF, C_is_M_BESTBF, 
                   BENs_M_T_BESTBF, CSTs_M_T_BESTBF, 
                   BB_is_M_BESTBF, CC_is_M_BESTBF, RU_is_M_BESTBF, 
                   pi_sg_minus_T_BESTBF, pi_sg_plus_T_BESTBF, 
                   pi_0_minus_T_BESTBF, pi_0_plus_T_BESTBF,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res_BESTBF, 
                   algo=algo_name,
                   dico_best_steps=dico_mode_prof_by_players_T)
    turn_dico_stats_res_into_df_BF(dico_stats_res_algo= dico_stats_res_BESTBF, 
                                path_to_save = path_to_save, 
                                t_periods = t_periods, 
                                manual_debug = manual_debug, 
                                algo_name = algo_name)
    
    algo_name = fct_aux.ALGO_NAMES_BF[1]
    if "simu_DDMM_HHMM" in path_to_save:
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_BADBF, 
                   b0_ts_T_BADBF, c0_ts_T_BADBF, B_is_M_BADBF, C_is_M_BADBF, 
                   BENs_M_T_BADBF, CSTs_M_T_BADBF, 
                   BB_is_M_BADBF, CC_is_M_BADBF, RU_is_M_BADBF, 
                   pi_sg_minus_T_BADBF, pi_sg_plus_T_BADBF, 
                   pi_0_minus_T_BADBF, pi_0_plus_T_BADBF,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res_BADBF, 
                   algo=algo_name,
                   dico_best_steps=dico_mode_prof_by_players_T)
    turn_dico_stats_res_into_df_BF(dico_stats_res_algo= dico_stats_res_BADBF, 
                                path_to_save = path_to_save, 
                                t_periods = t_periods, 
                                manual_debug = manual_debug, 
                                algo_name = algo_name)
    
    algo_name = fct_aux.ALGO_NAMES_BF[2]
    if "simu_DDMM_HHMM" in path_to_save:
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif_MIDBF, 
                   b0_ts_T_MIDBF, c0_ts_T_MIDBF, B_is_M_MIDBF, C_is_M_MIDBF, 
                   BENs_M_T_MIDBF, CSTs_M_T_MIDBF, 
                   BB_is_M_MIDBF, CC_is_M_MIDBF, RU_is_M_MIDBF, 
                   pi_sg_minus_T_MIDBF, pi_sg_plus_T_MIDBF, 
                   pi_0_minus_T_MIDBF, pi_0_plus_T_MIDBF,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res_MIDBF, 
                   algo=algo_name,
                   dico_best_steps=dico_mode_prof_by_players_T)
    turn_dico_stats_res_into_df_BF(dico_stats_res_algo= dico_stats_res_MIDBF, 
                                path_to_save = path_to_save, 
                                t_periods = t_periods, 
                                manual_debug = manual_debug, 
                                algo_name = algo_name)
    
    return arr_pl_M_T_vars_modif
    
def turn_dico_stats_res_into_df_BF(dico_stats_res_algo, path_to_save, t_periods=2, 
                                manual_debug=True, 
                                algo_name="BEST-BF"):
    """
    transform the dico in the row dico_nash_profils into a DataFrame

    Parameters
    ----------
    path_to_variable : TYPE
        DESCRIPTION.
        

    Returns
    -------
    None.

    """
    df = None
    for t in range(0,t_periods):
        list_dico_best_bf = dico_stats_res_algo[t]["list_dico_best_mode_profs"]
        dico_best_bf = dict()
        for cpt, dico_best_bf_items in list_dico_best_bf:
            dico_best_bf[cpt] = dico_best_bf_items
            
        df_t = pd.DataFrame.from_dict(dico_best_bf, orient='columns')
        if df is None:
            df = df_t.copy()
        else:
            df = pd.concat([df, df_t], axis=0)
            
    # save df to csv
    df.to_csv( os.path.join( *[path_to_save, algo_name+"_"+"dico_BF.csv"] ))
    
# ______________      main function of brut force   ---> final      ___________

#------------------------------------------------------------------------------
#                       definition of unittests 
#
#------------------------------------------------------------------------------
def test_brute_force_game(algo_name, criteria):
    fct_aux.N_DECIMALS = 6
    
    pi_hp_plus = 0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 0.33 #[15, 5]
    
    manual_debug=True
    debug = False
    criteria_bf = "Perf_t"
    
    t_periods = 2
    set1_m_players, set2_m_players = 20, 12
    set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    #set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    
    set1_m_players, set2_m_players = 10, 6
    # set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    
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
    
    criteria_bf = "Perf_t"
    
    arr_pl_M_T_vars = bf_balanced_player_game(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                algo_name=algo_name,
                                path_to_save="tests", 
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
    
    return arr_pl_M_T_vars

def test_brute_force_game_USE_DICT_MODE_PROFIL(criteria):
    fct_aux.N_DECIMALS = 6
    
    pi_hp_plus = 0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 0.33 #[15, 5]
    
    manual_debug=True
    debug = False
    criteria_bf = "Perf_t"
    
    t_periods = 2
    set1_m_players, set2_m_players = 20, 12
    set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    #set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    
    set1_m_players, set2_m_players = 10, 6
    # set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    
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
    
    criteria_bf = "Perf_t"
    
    arr_pl_M_T_vars = bf_balanced_player_game_USE_DICT_MODE_PROFIL(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                path_to_save="tests", 
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)


def test_brute_force_game_USE_DICT_MODE_PROFIL_OLD(criteria):
    fct_aux.N_DECIMALS = 6
    
    pi_hp_plus = 0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 0.33 #[15, 5]
    
    manual_debug=True
    debug = False
    criteria_bf = "Perf_t"
    
    t_periods = 2
    set1_m_players, set2_m_players = 20, 12
    set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    #set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    
    set1_m_players, set2_m_players = 10, 6
    # set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    
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
    
    criteria_bf = "Perf_t"
    
    arr_pl_M_T_vars = bf_balanced_player_game_USE_DICT_MODE_PROFIL_OLD(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                path_to_save="tests", 
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
    
    return arr_pl_M_T_vars

#------------------------------------------------------------------------------
#                       execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    criteria = "Perf_t"
    # for algo_name in fct_aux.ALGO_NAMES_BF:
    #     arr_pl_M_T_vars = test_brute_force_game(algo_name, criteria)
        
    test_brute_force_game_USE_DICT_MODE_PROFIL(criteria)
    
    print("runtime = {}".format(time.time() - ti))