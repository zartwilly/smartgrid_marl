# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:06:23 2020

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
#                       definition of functions --> debut
#
#------------------------------------------------------------------------------
def reupdate_state_players(arr_pl_M_T_K_vars, t=0, k=0):
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
            Ci = round(arr_pl_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Ci"]], 
                       fct_aux.N_DECIMALS)
            Pi = round(arr_pl_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Pi"]], 
                       fct_aux.N_DECIMALS)
            Si = round(arr_pl_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Si"]], 
                       fct_aux.N_DECIMALS)
            Si_max = round(arr_pl_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Si_max"]],
                        fct_aux.N_DECIMALS)
            gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
            pl_i = None
            pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                prod_i, cons_i, r_i, state_i)
            
            # get mode_i, state_i and update R_i_old
            state_i = pl_i.find_out_state_i()
            col = "state_i"
            arr_pl_vars[num_pl_i,:,fct_aux.INDEX_ATTRS[col]] = state_i
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
            Ci = round(arr_pl_vars[num_pl_i, t, k, fct_aux.INDEX_ATTRS["Ci"]], 
                       fct_aux.N_DECIMALS)
            Pi = round(arr_pl_vars[num_pl_i, t, k, fct_aux.INDEX_ATTRS["Pi"]], 
                       fct_aux.N_DECIMALS)
            Si = round(arr_pl_vars[num_pl_i, t, k, fct_aux.INDEX_ATTRS["Si"]], 
                       fct_aux.N_DECIMALS)
            Si_max = round(arr_pl_vars[num_pl_i, t, k, fct_aux.INDEX_ATTRS["Si_max"]],
                        fct_aux.N_DECIMALS)
            gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
            pl_i = None
            pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                prod_i, cons_i, r_i, state_i)
            
            # get mode_i, state_i and update R_i_old
            state_i = pl_i.find_out_state_i()
            col = "state_i"
            arr_pl_vars[num_pl_i,:,:,fct_aux.INDEX_ATTRS[col]] = state_i
            if state_i == "state1":
                possibles_modes.append(fct_aux.STATE1_STRATS)
            elif state_i == "state2":
                possibles_modes.append(fct_aux.STATE2_STRATS)
            elif state_i == "state3":
                possibles_modes.append(fct_aux.STATE3_STRATS)
            # print("4: num_pl_i={}, state_i = {}".format(num_pl_i, state_i))
    else:
        print("STATE_i: NOTHING TO UPDATE.")
        
    return arr_pl_vars, possibles_modes

def compute_prices_bf(arr_pl_M_T_vars, t,
                      pi_sg_plus_t, pi_sg_minus_t,
                      pi_hp_plus, pi_hp_minus, manual_debug, dbg):
    """
    compute the prices' and benefits/costs variables: 
        ben_i, cst_i
        pi_sg_plus_t_k, pi_sg_minus_t_k
        pi_0_plus_t_k, pi_0_minus_t_k
    """
    pi_sg_plus_t_new, pi_sg_minus_t_new \
        = fct_aux.determine_new_pricing_sg(
            arr_pl_M_T_vars, 
            pi_hp_plus, 
            pi_hp_minus, 
            t, 
            dbg=dbg)
    print("pi_sg_plus_{}_new={}, pi_sg_minus_{}_new={}".format(
        t, pi_sg_plus_t_new, t, pi_sg_minus_t_new)) if dbg else None
          
    pi_sg_plus_t = pi_sg_plus_t \
                            if pi_sg_plus_t_new is np.nan \
                            else pi_sg_plus_t_new
    pi_sg_minus_t = pi_sg_minus_t \
                            if pi_sg_minus_t_new is np.nan \
                            else pi_sg_minus_t_new
    pi_0_plus_t = round(pi_sg_minus_t*pi_hp_plus/pi_hp_minus, 
                          fct_aux.N_DECIMALS)
    pi_0_minus_t = pi_sg_minus_t
    
    if manual_debug:
        pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
        pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
        pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #2 
        pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
        
    print("pi_sg_minus_t={}, pi_0_plus_t={}, pi_0_minus_t={},"\
          .format(pi_sg_minus_t, pi_0_plus_t, pi_0_minus_t)) \
        if dbg else None
    print("pi_sg_plus_t={}, pi_sg_minus_t={} \n"\
          .format(pi_sg_plus_t, pi_sg_minus_t)) \
        if dbg else None
    
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_M_T_vars, t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t, c0_t = fct_aux.compute_energy_unit_price(
                        pi_0_plus_t, pi_0_minus_t, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg) 
    
    # compute ben, cst of shapes (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_vars[:, t, fct_aux.INDEX_ATTRS["gamma_i"]]
    bens_t, csts_t = fct_aux.compute_utility_players(
                            arr_pl_M_T_vars, 
                            gamma_is, 
                            t, 
                            b0_t, 
                            c0_t)
    
    return pi_sg_plus_t, pi_sg_minus_t, \
            pi_0_plus_t, pi_0_minus_t, \
            b0_t, c0_t, \
            bens_t, csts_t 


def balanced_player_game_4_mode_profil(arr_pl_M_T_vars, 
                                       mode_profile, t,
                                       pi_sg_plus_t, pi_sg_minus_t, 
                                       pi_hp_plus, pi_hp_minus,
                                       dico_balanced_pl_i, 
                                       dico_state_mode_i,
                                       cpt_balanced,
                                       m_players, t_periods, 
                                       manual_debug):
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
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS['Si']]
        Si_max = arr_pl_M_T_vars[num_pl_i, t, 
                                 fct_aux.INDEX_ATTRS['Si_max']]
        gamma_i, prod_i, cons_i, r_i = 0, 0, 0, 0
        mode_i = mode_profile[num_pl_i]
        state_i = arr_pl_M_T_vars[num_pl_i, t, 
                                 fct_aux.INDEX_ATTRS['state_i']]
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                    prod_i, cons_i, r_i, state_i)
        
        pl_i.set_R_i_old(Si_max-Si)
        pl_i.set_mode_i(mode_i)
        pl_i.set_state_i(state_i)
        
        # update prod, cons and r_i
        pl_i.update_prod_cons_r_i()
        
        # is pl_i balanced?
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        
        cpt_balanced += round(1/m_players, fct_aux.N_DECIMALS) \
                        if boolean else 0
        dico_balanced_pl_i["cpt"] = cpt_balanced
        if "player" in dico_balanced_pl_i and boolean is False:
            dico_balanced_pl_i['player'].append(num_pl_i)
        elif boolean is False:
            dico_balanced_pl_i['player'] = [num_pl_i]
            
        # compute gamma_i
        Pi_t_plus_1 = arr_pl_M_T_vars[num_pl_i, t+1, 
                                      fct_aux.INDEX_ATTRS["Pi"]] \
                        if t+1 < t_periods \
                        else 0
        Ci_t_plus_1 = arr_pl_M_T_vars[num_pl_i, 
                                 t+1, 
                                 fct_aux.INDEX_ATTRS["Ci"]] \
                        if t+1 < t_periods \
                        else 0
                     
        pl_i.select_storage_politic(
            Ci_t_plus_1 = Ci_t_plus_1, 
            Pi_t_plus_1 = Pi_t_plus_1, 
            pi_0_plus = pi_sg_plus_t, 
            pi_0_minus = pi_sg_minus_t, 
            pi_hp_plus = pi_hp_plus, 
            pi_hp_minus = pi_hp_minus)
        
        gamma_i = None
        if manual_debug:
            gamma_i = fct_aux.MANUEL_DBG_GAMMA_I # 5
            pl_i.set_gamma_i(gamma_i)
        else:
            gamma_i = pl_i.get_gamma_i()
            
        if gamma_i >= min(pi_sg_minus_t, pi_sg_plus_t) -1 \
            and gamma_i <= max(pi_hp_minus, pi_hp_plus):
            pass
        else :
            cpt_error_gamma = round(1/m_players, 2)
            dico_state_mode_i["cpt"] = \
                dico_state_mode_i["cpt"] + cpt_error_gamma \
                if "cpt" in dico_state_mode_i \
                else cpt_error_gamma
            dico_state_mode_i[(pl_i.state_i, pl_i.mode_i)] \
                = dico_state_mode_i[(pl_i.state_i, pl_i.mode_i)] + 1 \
                if (pl_i.state_i, pl_i.mode_i) in dico_state_mode_i \
                else 1
        
        # update variables in arr_pl_M_T_k
        tup_cols_values = [
            ("prod_i", pl_i.get_prod_i()),("cons_i", pl_i.get_cons_i()), 
            ("gamma_i", pl_i.get_gamma_i()), ("r_i", pl_i.get_r_i()),
            ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
            ("Si_old", pl_i.get_Si_old()), ("state_i", pl_i.get_state_i()),
            ("mode_i", mode_i), ("balanced_pl_i", boolean), 
            ("formule", formule), ("Si_minus", 0),("Si_plus", 0)]
        for col,val in tup_cols_values:
            arr_pl_M_T_vars[num_pl_i, t,
                            fct_aux.INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_vars, dico_balanced_pl_i, dico_state_mode_i, cpt_balanced
#------------------------------------------------------------------------------
#                       definition of functions --> fin
#
#------------------------------------------------------------------------------

# ______________      main function of brut force   ---> debut      ___________
def bf_balanced_player_game_In_sg_Out_sg(arr_pl_M_T,
                                         pi_hp_plus=0.10, 
                                         pi_hp_minus=0.15,
                                         m_players=3, 
                                         t_periods=4,
                                         prob_Ci=0.3, 
                                         scenario="scenario1",
                                         algo_name="BEST_BF",
                                         path_to_save="tests", 
                                         manual_debug=False, dbg=False):
    """
    brute force algorithm for balanced players' game.
    determine the best solution by enumerating all players' profils.
    The comparison critera is the In_sg-Out_sg value 

    Returns
    -------
    None.

    """
    
    print("\n \n {} game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> debut \n"\
          .format(algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
        
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
    b0_ts_T.fill(np.nan)
    c0_ts_T = np.empty(shape=(t_periods,))
    c0_ts_T.fill(np.nan)
    BENs_M_T = np.empty(shape=(m_players, t_periods)) #   shape (M_PLAYERS, t_periods)
    CSTs_M_T = np.empty(shape=(m_players, t_periods))
    
    
    # _______ variables' initialization --> fin ________________
    
    
    # ____      add initial values for the new attributs ---> debut    _______
    # nb_vars_2_add = 6
    # fct_aux.INDEX_ATTRS["Si_minus"] = 16
    # fct_aux.INDEX_ATTRS["Si_plus"] = 17
    # fct_aux.INDEX_ATTRS["prob_mode_state_i"] = 18
    # fct_aux.INDEX_ATTRS["u_i"] = 19
    # fct_aux.INDEX_ATTRS["bg_i"] = 20
    # fct_aux.INDEX_ATTRS["non_playing_players"] = 21
    
    # arr_pl_M_T_vars = np.zeros((arr_pl_M_T.shape[0],
    #                             arr_pl_M_T.shape[1],
    #                             arr_pl_M_T.shape[2]+nb_vars_2_add), 
    #                            dtype=object)
    # arr_pl_M_T_vars[:,:,:-nb_vars_2_add] = arr_pl_M_T
    # arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_minus"]] = np.nan
    # arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_plus"]] = np.nan
    # arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["u_i"]] = np.nan
    # arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["bg_i"]] = np.nan
    # arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prob_mode_state_i"]] = 0.5
    # arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["non_playing_players"]] \
    #     = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    # print("SHAPE: arr_pl_M_T={}, arr_pl_M_T_vars={}".format(
    #         arr_pl_M_T.shape, arr_pl_M_T_vars.shape))
    
    nb_vars_2_add = 2
    fct_aux.INDEX_ATTRS["Si_minus"] = 16
    fct_aux.INDEX_ATTRS["Si_plus"] = 17
    
    arr_pl_M_T_vars = np.zeros((arr_pl_M_T.shape[0],
                                arr_pl_M_T.shape[1],
                                arr_pl_M_T.shape[2]+nb_vars_2_add), 
                               dtype=object)
    arr_pl_M_T_vars[:,:,:-nb_vars_2_add] = arr_pl_M_T
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_plus"]] = np.nan
        
    print("SHAPE: arr_pl_M_T={}, arr_pl_M_T_vars={}".format(
            arr_pl_M_T.shape, arr_pl_M_T_vars.shape))
    
    # ____      add initial values for the new attributs ---> fin    _______
    
    # ____      game beginning for all t_period ---> debut      _____ 
    dico_stats_res={}
    
    arr_pl_M_T_vars, possibles_modes = reupdate_state_players(
                                        arr_pl_M_T_vars.copy(), 0, 0)
    
    print("m_players={}, possibles_modes={}".format(m_players, 
                                                   len(possibles_modes)))
    
    for t in range(0, t_periods):
        print("******* t = {} *******".format(t)) if dbg else None
        print("___t = {}, pi_sg_plus_t={}, pi_sg_minus_t={}".format(
                t, pi_sg_plus_t, pi_sg_minus_t)) \
            if t%20 == 0 \
            else None
            
        # compute pi_0_plus, pi_0_minus, pi_sg_plus, pi_sg_minus
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
        else:
            pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t
            pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t
        
        cpt_error_gamma = 0; cpt_balanced = 0;
        dico_state_mode_i = {}; dico_balanced_pl_i = {}
        
        mode_profiles = it.product(*possibles_modes)
        
        dico_mode_profs = dict()
        cpt_xxx = 0
        for mode_profile in mode_profiles:
            dico_balanced_pl_i_mode_prof, cpt_balanced_mode_prof = dict(), 0
            dico_state_mode_i_mode_prof = dict()
            
            arr_pl_M_T_vars_mode_prof, \
            dico_balanced_pl_i_mode_prof, \
            dico_state_mode_i_mode_prof, \
            cpt_balanced_mode_prof \
                = balanced_player_game_4_mode_profil(
                    arr_pl_M_T_vars.copy(),
                    mode_profile, t,
                    pi_sg_plus_t, pi_sg_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    dico_balanced_pl_i_mode_prof, 
                    dico_state_mode_i_mode_prof,
                    cpt_balanced_mode_prof,
                    m_players, t_periods,
                    manual_debug
                    )
            
            # compute In_sg, Out_sg
            In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                                arr_pl_M_T_vars_mode_prof, 
                                t)
            diff_In_Out_sg = In_sg-Out_sg
            if diff_In_Out_sg in dico_mode_profs:
                dico_mode_profs[diff_In_Out_sg].append(mode_profile)
            else:
                dico_mode_profs[diff_In_Out_sg] = [mode_profile]
             
            cpt_xxx += 1
        
        # max_min_moy_bf
        best_key_In_Out_sg = None
        if algo_name == fct_aux.ALGO_NAMES_BF[0]:              # BEST-BRUTE-FORCE
            best_key_In_Out_sg = max(dico_mode_profs.keys())
        elif algo_name == fct_aux.ALGO_NAMES_BF[1]:            # BAD-BRUTE-FORCE
            best_key_In_Out_sg = min(dico_mode_profs.keys())
        elif algo_name == fct_aux.ALGO_NAMES_BF[2]:            # MIDDLE-BRUTE-FORCE
            mean_key_In_Out_sg  = np.mean(list(dico_mode_profs.keys()))
            if mean_key_In_Out_sg in dico_mode_profs.keys():
                best_key_In_Out_sg = mean_key_In_Out_sg
            else:
                sorted_keys = sorted(dico_mode_profs.keys())
                boolean = True; i_key = 1
                while boolean:
                    if sorted_keys[i_key] <= mean_key_In_Out_sg:
                        i_key += 1
                    else:
                        boolean = False; i_key -= 1
                best_key_In_Out_sg = sorted_keys[i_key]
                    
        # find the best, bad, middle key in dico_mode_profs and 
        # the best, bad, middle mode_profile
        best_mode_profiles = dico_mode_profs[best_key_In_Out_sg]
        best_mode_profile = None
        if len(best_mode_profiles) == 1:
            best_mode_profile = best_mode_profiles[0]
        else:
            rd = np.random.randint(0, len(best_mode_profiles))
            best_mode_profile = best_mode_profiles[rd]
        
        ### ____ best 5 keys and values: debut ____
        # import collections
        # od = collections.OrderedDict(sorted(dico_mode_profs.items()))
        # cpt_od = 0
        # for k, profil in od.items(): 
        #     print('*** key={}, profil={}'.format(k, profil))
        #     cpt_od += 1
        #     if cpt_od > 5:
        #         break
        ### ____ best 5 keys and values: fin ____
        
        print("cpt_xxx={}, best_key_In_Out_sg={}, best_mode_profile={}".format(
                cpt_xxx, best_key_In_Out_sg, best_mode_profile))
        
        arr_pl_M_T_vars_mode_prof_best, \
        dico_balanced_pl_i, dico_state_mode_i, \
        cpt_balanced \
            = balanced_player_game_4_mode_profil(
                arr_pl_M_T_vars.copy(),
                best_mode_profile, t,
                pi_sg_plus_t, pi_sg_minus_t, 
                pi_hp_plus, pi_hp_minus,
                dico_balanced_pl_i, dico_state_mode_i, 
                cpt_balanced,
                m_players, t_periods, manual_debug
                )
        
        dico_stats_res[t] = (round(cpt_balanced/m_players, fct_aux.N_DECIMALS),
                         round(cpt_error_gamma/m_players, fct_aux.N_DECIMALS), 
                         dico_state_mode_i)
        dico_stats_res[t] = {"balanced": dico_balanced_pl_i, 
                             "gamma_i": dico_state_mode_i}
        
        # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
        # from a pricing model in the document
        pi_sg_plus_t_new, pi_sg_minus_t_new \
            = fct_aux.determine_new_pricing_sg(
                    arr_pl_M_T_vars_mode_prof_best, 
                    pi_hp_plus, 
                    pi_hp_minus, 
                    t, dbg=dbg)
        print("t={}, pi_sg_plus_t_new={}, pi_sg_minus_t_new={}".format(
            t, pi_sg_plus_t_new, pi_sg_minus_t_new))  if dbg else None
        print("t={}, pi_sg_plus_t_new={}, pi_sg_minus_t_new={}".format(
            t, pi_sg_plus_t_new, pi_sg_minus_t_new))                
        pi_sg_plus_t = pi_sg_plus_t if pi_sg_plus_t_new is np.nan \
                                    else pi_sg_plus_t_new
        pi_sg_minus_t = pi_sg_minus_t if pi_sg_minus_t_new is np.nan \
                                    else pi_sg_minus_t_new
        pi_0_plus_t = round(pi_sg_minus_t*pi_hp_plus/pi_hp_minus, 
                            fct_aux.N_DECIMALS)
        pi_0_minus_t = pi_sg_minus_t
        
        print("-- Avant: pi_sg_plus_t={}, pi_sg_minus_t={}, pi_0_plus_t={}, pi_0_minus_t={},".format(
                pi_sg_plus_t, pi_sg_minus_t, pi_0_plus_t, pi_0_minus_t))
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #2 
            pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
        print("-- Apres: pi_sg_plus_t={}, pi_sg_minus_t={}, pi_0_plus_t={}, pi_0_minus_t={},".format(
                pi_sg_plus_t, pi_sg_minus_t, pi_0_plus_t, pi_0_minus_t))
        
        ## compute prices inside smart grids
        # compute In_sg, Out_sg
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                        arr_pl_M_T_vars_mode_prof_best, t)
        diff = np.abs(best_key_In_Out_sg-(In_sg-Out_sg))
        print("best_key==In_sg-Out_sg --> OK (diff={}) \n".format(diff)) \
            if diff < 0.1 \
            else print("best_key==In_sg-Out_sg --> NOK (diff={}) \n".format(diff))
        # compute prices of an energy unit price for cost and benefit players
        b0_t, c0_t = fct_aux.compute_energy_unit_price(
                        pi_0_plus_t, pi_0_minus_t, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg)
    
        # compute ben, cst of shape (M_PLAYERS,) 
        # compute cost (csts) and benefit (bens) players by energy exchanged.
        gamma_is_t = arr_pl_M_T_vars_mode_prof_best[
                        :, t, fct_aux.INDEX_ATTRS["gamma_i"]]
        bens_t, csts_t = fct_aux.compute_utility_players(
                            arr_pl_M_T_vars_mode_prof_best, 
                            gamma_is_t, 
                            t, 
                            b0_t, 
                            c0_t)
        
        # pi_sg_plus_T, pi_sg_minus_T of shape (t_periods,)
        pi_sg_plus_T[t] = pi_sg_plus_t
        pi_sg_minus_T[t] = pi_sg_minus_t
        
        # pi_0_plus, pi_0_minus of shape (t_periods,)
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        
        # b0_ts, c0_ts of shape (t_periods,)
        b0_ts_T[t] = b0_t
        c0_ts_T[t] = c0_t 
        
        # BENs, CSTs of shape (M_PLAYERS, t_periods)
        BENs_M_T[:,t] = bens_t
        CSTs_M_T[:,t] = csts_t
        
        arr_pl_M_T_vars[:,t,:] = arr_pl_M_T_vars_mode_prof_best[:,t,:].copy()
        
    # ____      game beginning for all t_period ---> fin      _____
        
    # B_is, C_is of shape (M_PLAYERS, )
    prod_i_T = arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_T = arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is_M = np.sum(b0_ts_T * prod_i_T, axis=1)
    C_is_M = np.sum(c0_ts_T * cons_i_T, axis=1)
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is = np.sum(arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["cons_i"]], axis=1)
    PROD_is = np.sum(arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prod_i"]], axis=1)
    BB_is_M = pi_sg_plus_T[-1] * PROD_is #np.sum(PROD_is)
    for num_pl, bb_i in enumerate(BB_is_M):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    CC_is_M = pi_sg_minus_T[-1] * CONS_is #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * t_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * t_periods, dtype=object)
    
    # save computed variables
    #algo_name = "BRUTE-FORCE"
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars.copy(), 
                   b0_ts_T, c0_ts_T, B_is_M, C_is_M, 
                   BENs_M_T, CSTs_M_T, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T, pi_sg_plus_T, 
                   pi_0_minus_T, pi_0_plus_T,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo=algo_name)
    
    print("{} game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> end \n"\
          .format(algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
        
    return arr_pl_M_T_vars
    

def bf_balanced_player_game_perf_t(arr_pl_M_T,
                                   pi_hp_plus=0.10, 
                                   pi_hp_minus=0.15,
                                   m_players=3, 
                                   t_periods=4,
                                   prob_Ci=0.3, 
                                   scenario="scenario1",
                                   algo_name="BEST_BF",
                                   path_to_save="tests", 
                                   manual_debug=False, dbg=False):
    """
    brute force algorithm for balanced players' game.
    determine the best solution by enumerating all players' profils.
    The comparison critera is the Pref_t value with
    Perf_t = \sum\limits_{1\leq i \leq N}ben_i-cst_i

    Returns
    -------
    None.

    """
    
    print("\n \n {} game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> debut \n"\
          .format(algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
        
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
    b0_ts_T.fill(np.nan)
    c0_ts_T = np.empty(shape=(t_periods,))
    c0_ts_T.fill(np.nan)
    BENs_M_T = np.empty(shape=(m_players, t_periods)) #   shape (M_PLAYERS, t_periods)
    CSTs_M_T = np.empty(shape=(m_players, t_periods))
    
    
    # _______ variables' initialization --> fin ________________
    
    
    # ____      add initial values for the new attributs ---> debut    _______
    nb_vars_2_add = 2
    fct_aux.INDEX_ATTRS["Si_minus"] = 16
    fct_aux.INDEX_ATTRS["Si_plus"] = 17
    
    arr_pl_M_T_vars = np.zeros((arr_pl_M_T.shape[0],
                                arr_pl_M_T.shape[1],
                                arr_pl_M_T.shape[2]+nb_vars_2_add), 
                               dtype=object)
    arr_pl_M_T_vars[:,:,:-nb_vars_2_add] = arr_pl_M_T
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_plus"]] = np.nan
        
    print("SHAPE: arr_pl_M_T={}, arr_pl_M_T_vars={}".format(
            arr_pl_M_T.shape, arr_pl_M_T_vars.shape))
    
    # ____      add initial values for the new attributs ---> fin    _______
    
    # ____      game beginning for all t_period ---> debut      _____ 
    dico_stats_res={}
    
    arr_pl_M_T_vars, possibles_modes = reupdate_state_players(
                                        arr_pl_M_T_vars.copy(), 0, 0)
    
    print("m_players={}, possibles_modes={}".format(m_players, 
                                                   len(possibles_modes)))
    
    for t in range(0, t_periods):
        print("******* t = {} *******".format(t)) if dbg else None
        print("___t = {}, pi_sg_plus_t={}, pi_sg_minus_t={}".format(
                t, pi_sg_plus_t, pi_sg_minus_t)) \
            if t%20 == 0 \
            else None
            
        # compute pi_0_plus, pi_0_minus, pi_sg_plus, pi_sg_minus
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
        else:
            pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t
            pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t
        
        cpt_error_gamma = 0; cpt_balanced = 0;
        dico_state_mode_i = {}; dico_balanced_pl_i = {}
        
        mode_profiles = it.product(*possibles_modes)
        
        dico_mode_profs = dict()
        cpt_xxx = 0
        for mode_profile in mode_profiles:
            dico_balanced_pl_i_mode_prof, cpt_balanced_mode_prof = dict(), 0
            dico_state_mode_i_mode_prof = dict()
            
            arr_pl_M_T_vars_mode_prof, \
            dico_balanced_pl_i_mode_prof, \
            dico_state_mode_i_mode_prof, \
            cpt_balanced_mode_prof \
                = balanced_player_game_4_mode_profil(
                    arr_pl_M_T_vars.copy(),
                    mode_profile, t,
                    pi_sg_plus_t, pi_sg_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    dico_balanced_pl_i_mode_prof, 
                    dico_state_mode_i_mode_prof,
                    cpt_balanced_mode_prof,
                    m_players, t_periods,
                    manual_debug
                    )
            
            # compute Perf_t 
            pi_sg_plus_t, pi_sg_minus_t, \
            pi_0_plus_t, pi_0_minus_t, \
            b0_t, c0_t, \
            bens_t, csts_t \
                = compute_prices_bf(arr_pl_M_T_vars_mode_prof.copy(), t, 
                            pi_sg_plus_t, pi_sg_minus_t,
                            pi_hp_plus, pi_hp_minus, manual_debug, dbg)
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
        
        ### ____ best 5 keys and values: debut ____
        # import collections
        # od = collections.OrderedDict(sorted(dico_mode_profs.items()))
        # cpt_od = 0
        # for k, profil in od.items(): 
        #     print('*** key={}, profil={}'.format(k, profil))
        #     cpt_od += 1
        #     if cpt_od > 5:
        #         break
        ### ____ best 5 keys and values: fin ____
        
        print("cpt_xxx={}, best_key_In_Out_sg={}, best_mode_profile={}".format(
                cpt_xxx, best_key_Perf_t, best_mode_profile))
        
        arr_pl_M_T_vars_mode_prof_best, \
        dico_balanced_pl_i, dico_state_mode_i, \
        cpt_balanced \
            = balanced_player_game_4_mode_profil(
                arr_pl_M_T_vars.copy(),
                best_mode_profile, t,
                pi_sg_plus_t, pi_sg_minus_t, 
                pi_hp_plus, pi_hp_minus,
                dico_balanced_pl_i, dico_state_mode_i, 
                cpt_balanced,
                m_players, t_periods, manual_debug
                )
        
        dico_stats_res[t] = (round(cpt_balanced/m_players, fct_aux.N_DECIMALS),
                         round(cpt_error_gamma/m_players, fct_aux.N_DECIMALS), 
                         dico_state_mode_i)
        dico_stats_res[t] = {"balanced": dico_balanced_pl_i, 
                             "gamma_i": dico_state_mode_i}
        
        # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
        # from a pricing model in the document
        pi_sg_plus_t_new, pi_sg_minus_t_new \
            = fct_aux.determine_new_pricing_sg(
                    arr_pl_M_T_vars_mode_prof_best, 
                    pi_hp_plus, 
                    pi_hp_minus, 
                    t, dbg=dbg)
        print("t={}, pi_sg_plus_t_new={}, pi_sg_minus_t_new={}".format(
            t, pi_sg_plus_t_new, pi_sg_minus_t_new))  if dbg else None
        print("t={}, pi_sg_plus_t_new={}, pi_sg_minus_t_new={}".format(
            t, pi_sg_plus_t_new, pi_sg_minus_t_new))                
        pi_sg_plus_t = pi_sg_plus_t if pi_sg_plus_t_new is np.nan \
                                    else pi_sg_plus_t_new
        pi_sg_minus_t = pi_sg_minus_t if pi_sg_minus_t_new is np.nan \
                                    else pi_sg_minus_t_new
        pi_0_plus_t = round(pi_sg_minus_t*pi_hp_plus/pi_hp_minus, 
                            fct_aux.N_DECIMALS)
        pi_0_minus_t = pi_sg_minus_t
        
        print("-- Avant: pi_sg_plus_t={}, pi_sg_minus_t={}, pi_0_plus_t={}, pi_0_minus_t={},".format(
                pi_sg_plus_t, pi_sg_minus_t, pi_0_plus_t, pi_0_minus_t))
        if manual_debug:
            pi_sg_plus_t = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            pi_sg_minus_t = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #2 
            pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
        print("-- Apres: pi_sg_plus_t={}, pi_sg_minus_t={}, pi_0_plus_t={}, pi_0_minus_t={},".format(
                pi_sg_plus_t, pi_sg_minus_t, pi_0_plus_t, pi_0_minus_t))
        
        ## compute prices inside smart grids
        # compute In_sg, Out_sg
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                            arr_pl_M_T_vars_mode_prof_best, t)
        
        # compute prices of an energy unit price for cost and benefit players
        b0_t, c0_t = fct_aux.compute_energy_unit_price(
                        pi_0_plus_t, pi_0_minus_t, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg)
    
        # compute ben, cst of shape (M_PLAYERS,) 
        # compute cost (csts) and benefit (bens) players by energy exchanged.
        gamma_is_t = arr_pl_M_T_vars_mode_prof_best[
                        :, t, fct_aux.INDEX_ATTRS["gamma_i"]]
        bens_t, csts_t = fct_aux.compute_utility_players(
                            arr_pl_M_T_vars_mode_prof_best, 
                            gamma_is_t, 
                            t, 
                            b0_t, 
                            c0_t)
        diff = np.abs(best_key_Perf_t-( np.sum(bens_t-csts_t) ))
        print("best_key==bens_t-csts_t --> OK (diff={}) \n".format(diff)) \
            if diff < 0.1 \
            else print("best_key==bens_t-csts_t --> NOK (diff={}) \n".format(diff))
        
        # pi_sg_plus_T, pi_sg_minus_T of shape (t_periods,)
        pi_sg_plus_T[t] = pi_sg_plus_t
        pi_sg_minus_T[t] = pi_sg_minus_t
        
        # pi_0_plus, pi_0_minus of shape (t_periods,)
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        
        # b0_ts, c0_ts of shape (t_periods,)
        b0_ts_T[t] = b0_t
        c0_ts_T[t] = c0_t 
        
        # BENs, CSTs of shape (M_PLAYERS, t_periods)
        BENs_M_T[:,t] = bens_t
        CSTs_M_T[:,t] = csts_t
        
        arr_pl_M_T_vars[:,t,:] = arr_pl_M_T_vars_mode_prof_best[:,t,:].copy()
        
    # ____      game beginning for all t_period ---> fin      _____
        
    # B_is, C_is of shape (M_PLAYERS, )
    prod_i_T = arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_T = arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is_M = np.sum(b0_ts_T * prod_i_T, axis=1)
    C_is_M = np.sum(c0_ts_T * cons_i_T, axis=1)
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is = np.sum(arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["cons_i"]], axis=1)
    PROD_is = np.sum(arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prod_i"]], axis=1)
    BB_is_M = pi_sg_plus_T[-1] * PROD_is #np.sum(PROD_is)
    for num_pl, bb_i in enumerate(BB_is_M):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    CC_is_M = pi_sg_minus_T[-1] * CONS_is #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * t_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * t_periods, dtype=object)
    
    # save computed variables
    #algo_name = "BRUTE-FORCE"
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars.copy(), 
                   b0_ts_T, c0_ts_T, B_is_M, C_is_M, 
                   BENs_M_T, CSTs_M_T, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T, pi_sg_plus_T, 
                   pi_0_minus_T, pi_0_plus_T,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo=algo_name)
    
    print("{} game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> end \n"\
          .format(algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
        
    return arr_pl_M_T_vars
    

def bf_balanced_player_game(arr_pl_M_T,
                            pi_hp_plus=0.10, 
                            pi_hp_minus=0.15,
                            m_players=3, 
                            t_periods=4,
                            prob_Ci=0.3, 
                            scenario="scenario1",
                            algo_name="BEST_BF",
                            path_to_save="tests", 
                            manual_debug=False, criteria_bf="Perf_t", dbg=False):
    
    if criteria_bf == "In_sg_Out_sg":
        arr_pl_M_T = bf_balanced_player_game_In_sg_Out_sg(
                                 arr_pl_M_T.copy(),
                                 pi_hp_plus, 
                                 pi_hp_minus,
                                 m_players, 
                                 t_periods,
                                 prob_Ci, 
                                 scenario,
                                 algo_name,
                                 path_to_save, manual_debug, dbg=False)
    elif criteria_bf == "Perf_t":
        arr_pl_M_T = bf_balanced_player_game_perf_t(
                                 arr_pl_M_T.copy(),
                                 pi_hp_plus, 
                                 pi_hp_minus,
                                 m_players, 
                                 t_periods,
                                 prob_Ci, 
                                 scenario,
                                 algo_name,
                                 path_to_save, manual_debug, dbg=False)
        
    return arr_pl_M_T
    
# ______________      main function of brut force   ---> fin      ___________


#------------------------------------------------------------------------------
#                       definition of unittests 
#
#------------------------------------------------------------------------------
def test_brute_force_game(algo_name="BEST-BRUTE-FORCE", criteria_bf="Perf_t"):
    fct_aux.N_DECIMALS = 6
    
    pi_hp_plus = 0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 0.33 #[15, 5]
    m_players = 10            #100 # 10 # 1001000
    t_periods = 5           # 50
     
    name_dir = 'tests'
    game_dir = 'INSTANCES_GAMES'
        
    date_hhmm=None
    Visualisation = False
    date_hhmm="2306_2206" if Visualisation \
                            else datetime.now().strftime("%d%m_%H%M") 
    
    scenario = "scenario1"
    prob_Ci = 0.3
    
    path_to_save = os.path.join(
                            name_dir, game_dir,
                            scenario, str(prob_Ci)
                            )    
    
    arr_pl_M_T_probCi_scen = None
    
    name_file_arr_pl = "arr_pl_M_T_players_{}_periods_{}.npy".format(
                            m_players, t_periods)
    path_to_arr_pl_M_T = os.path.join(path_to_save, name_file_arr_pl)
    if os.path.exists(path_to_arr_pl_M_T):
        print("file {} already EXISTS".format(name_file_arr_pl))
        arr_pl_M_T_probCi_scen = np.load(path_to_arr_pl_M_T, allow_pickle=True)
        print("READ INSTANCE --> OK")
    else:
        arr_pl_M_T_probCi_scen \
            = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                m_players=m_players, 
                num_periods=t_periods, 
                scenario=scenario, prob_Ci=prob_Ci, 
                Ci_low=fct_aux.Ci_LOW, Ci_high=fct_aux.Ci_HIGH)
            
        fct_aux.save_instances_games(
                    arr_pl_M_T_probCi_scen, 
                    name_file_arr_pl,  
                    path_to_save)
    
        print("Generation instances players={}, periods={}, {}, prob_Ci={} ---> OK".format(
                m_players, t_periods, scenario, prob_Ci))
    
    print("Generation/Recuperation instances TERMINEE")
    
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    path_to_save = os.path.join(name_dir, algo_name.split('-')[0]+"_bf_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg)
    
    manual_debug = True #False #True #False
    criteria_bf = "In_sg_Out_sg"
    #criteria_bf = "Perf_t"
    arr_pl_M_T_probCi_scen = bf_balanced_player_game(
                                 arr_pl_M_T_probCi_scen.copy(),
                                 pi_hp_plus, 
                                 pi_hp_minus,
                                 m_players, 
                                 t_periods,
                                 prob_Ci, 
                                 scenario,
                                 algo_name,
                                 path_to_save, manual_debug, 
                                 criteria_bf, dbg=False)
    
    return arr_pl_M_T_probCi_scen

#------------------------------------------------------------------------------
#                       execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    criteria = "Perf_t"
    for algo_name in fct_aux.ALGO_NAMES_BF:
        arr_pl_M_T_vars = test_brute_force_game(algo_name, criteria)
    
    print("runtime = {}".format(time.time() - ti))
    
    
