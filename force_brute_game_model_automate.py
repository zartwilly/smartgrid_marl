# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 07:49:45 2021

@author: jwehounou
"""
import os
import time

import numpy as np
import itertools as it
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

from datetime import datetime
#------------------------------------------------------------------------------
#                       definition of functions
#
#------------------------------------------------------------------------------
def reupdate_state_players_automate(arr_pl_M_T_K_vars, t=0, k=0):
    """
    after remarking that some players have 2 states during the game, 
    I decide to write this function to set uniformly the players' state for all
    periods and all learning step

    Parameters
    ----------
    arr_pl_M_T_K_vars : TYPE, optional
        DESCRIPTION. The default is None.
    t : TYPE, optional
        DESCRIPTION. The default is 0.
    k : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """

    m_players = arr_pl_M_T_K_vars.shape[0]
    possibles_modes = list()
    
    arr_pl_vars = None
    if len(arr_pl_M_T_K_vars.shape) == 3:
        arr_pl_vars = arr_pl_M_T_K_vars
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_vars[num_pl_i, t, 
                                  fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] 
            
            # get mode_i
            if state_i == "state1":
                possibles_modes.append(fct_aux.STATE1_STRATS)
            elif state_i == "state2":
                possibles_modes.append(fct_aux.STATE2_STRATS)
            elif state_i == "state3":
                possibles_modes.append(fct_aux.STATE3_STRATS)
            # print("3: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
                
    elif len(arr_pl_M_T_K_vars.shape) == 4:
        arr_pl_vars = arr_pl_M_T_K_vars
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_vars[num_pl_i, t, k, 
                                  fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
            
            # get mode_i
            if state_i == "state1":
                possibles_modes.append(fct_aux.STATE1_STRATS)
            elif state_i == "state2":
                possibles_modes.append(fct_aux.STATE2_STRATS)
            elif state_i == "state3":
                possibles_modes.append(fct_aux.STATE3_STRATS)
            # print("4: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
    else:
        print("STATE_i: NOTHING TO UPDATE.")
        
    return possibles_modes

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
    
def compute_prices_inside_SG(arr_pl_M_T_vars_modif, t,
                                pi_hp_plus, pi_hp_minus,
                                pi_0_plus_t, pi_0_minus_t,
                                manual_debug, dbg):
    # TODO : A deplacer dans fonctions_auxiliaires
    
    # compute the new prices pi_sg_plus_t, pi_sg_minus_t
    # from a pricing model in the document
    pi_sg_plus_t, pi_sg_minus_t = fct_aux.determine_new_pricing_sg(
                                            arr_pl_M_T_vars_modif, 
                                            pi_hp_plus, 
                                            pi_hp_minus, 
                                            t, dbg=dbg)
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_M_T_vars_modif.copy(), t)
    print("In_sg={}, Out_sg={}".format(In_sg, Out_sg ))
    # compute prices of an energy unit price for cost and benefit players
    b0_t, c0_t = fct_aux.compute_energy_unit_price(
                    pi_0_plus_t, pi_0_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg)
    
    # compute ben, cst of shape (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_vars_modif[:, t, fct_aux.AUTOMATE_INDEX_ATTRS["gamma_i"]]
    bens_t, csts_t = fct_aux.compute_utility_players(arr_pl_M_T_vars_modif, 
                                              gamma_is, 
                                              t, 
                                              b0_t, 
                                              c0_t)
    
    return b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t

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
        = compute_prices_inside_SG(arr_pl_M_T_vars_mode_prof, t,
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
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    print("\n \n {} game: {}, pi_hp_minus ={} ---> fin \n"\
          .format(algo_name, pi_hp_plus, pi_hp_minus))
        
    # ____      game beginning for all t_period ---> debut      _____
    dico_stats_res = dict()
    
    possibles_modes = reupdate_state_players_automate(
                                        arr_pl_M_T_vars.copy(), 0, 0)
    
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
    for t in range(0, t_periods):
        print("----- t = {} ------ ".format(t))
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
    prod_i_M_T = arr_pl_M_T_vars_mode_prof_best[:,:, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
    cons_i_M_T = arr_pl_M_T_vars_mode_prof_best[:,:, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
    B_is_M = np.sum(b0_ts_T * prod_i_M_T, axis=1)
    C_is_M = np.sum(c0_ts_T * cons_i_M_T, axis=1)
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is_M_T = np.sum(arr_pl_M_T_vars_mode_prof_best[:,:, 
                                         fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]], 
                         axis=1)
    PROD_is_M_T = np.sum(arr_pl_M_T_vars_mode_prof_best[:,:, 
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
                   algo=algo_name)
    
    return arr_pl_M_T_vars_modif
    
    
    
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

#------------------------------------------------------------------------------
#                       execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    criteria = "Perf_t"
    for algo_name in fct_aux.ALGO_NAMES_BF:
        arr_pl_M_T_vars = test_brute_force_game(algo_name, criteria)
    
    print("runtime = {}".format(time.time() - ti))