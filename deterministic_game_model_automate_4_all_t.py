# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:04:57 2021

@author: jwehounou
"""
import os
import time

import numpy as np
import pandas as pd
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

from pathlib import Path

###############################################################################
#                   definition  des fonctions annexes
#
###############################################################################
# ____________________ checkout LRI profil --> debut _________________________
def checkout_nash_4_profils_by_periods(arr_pl_M_T_vars_modif,
                                        arr_pl_M_T_vars,
                                        pi_hp_plus, pi_hp_minus, 
                                        pi_0_minus_t, pi_0_plus_t, 
                                        ben_csts_M_t,
                                        t,
                                        manual_debug):
    """
    verify if the profil at time t and k_stop_learning is a Nash equilibrium.
    """
    # create a result dataframe of checking players' stability and nash equilibrium
    cols = ["players", "nash_modes_t{}".format(t), 'states_t{}'.format(t), 
            'Vis_t{}'.format(t), 'Vis_bar_t{}'.format(t), 
               'res_t{}'.format(t)] 
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    id_players = list(range(0, m_players))
    df_nash_t = pd.DataFrame(index=id_players, columns=cols)
    
    # revert Si to the initial value ie at t and k=0
    Sis = arr_pl_M_T_vars[:, t,
                          fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
    arr_pl_M_T_vars_modif[:, t,
                            fct_aux.AUTOMATE_INDEX_ATTRS["Si"]] = Sis
    
    # stability of each player
    modes_profil = list(arr_pl_M_T_vars_modif[
                            :, t,
                            fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]] )
    for num_pl_i in range(0, m_players):
        state_i = arr_pl_M_T_vars_modif[
                        num_pl_i, t,
                        fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]] 
        mode_i = modes_profil[num_pl_i]
        mode_i_bar = fct_aux.find_out_opposite_mode(state_i, mode_i)
        
        opposite_modes_profil = modes_profil.copy()
        opposite_modes_profil[num_pl_i] = mode_i_bar
        opposite_modes_profil = tuple(opposite_modes_profil)
        
        df_nash_t.loc[num_pl_i, "players"] = fct_aux.RACINE_PLAYER+"_"+str(num_pl_i)
        df_nash_t.loc[num_pl_i, "nash_modes_t{}".format(t)] = mode_i
        df_nash_t.loc[num_pl_i, "states_t{}".format(t)] = state_i
        
        random_mode = False
        arr_pl_M_T_K_vars_modif_mode_prof_BAR, \
        b0_t_k_bar, c0_t_k_bar, \
        bens_t_k_bar, csts_t_k_bar, \
        dico_gamma_players_t_k \
            = fct_aux.balanced_player_game_t_4_mode_profil_prices_SG(
                    arr_pl_M_T_vars_modif.copy(), 
                    opposite_modes_profil,
                    t, 
                    pi_hp_plus, pi_hp_minus, 
                    pi_0_plus_t, pi_0_minus_t,
                    random_mode,
                    manual_debug, dbg=False)
        
        Vi = ben_csts_M_t[num_pl_i]
        bens_csts_t_k_bar = bens_t_k_bar - csts_t_k_bar
        Vi_bar = bens_csts_t_k_bar[num_pl_i]
    
        df_nash_t.loc[num_pl_i, 'Vis_t{}'.format(t)] = Vi
        df_nash_t.loc[num_pl_i, 'Vis_bar_t{}'.format(t)] = Vi_bar
        res = None
        if Vi >= Vi_bar:
            res = "STABLE"
            df_nash_t.loc[num_pl_i, 'res_t{}'.format(t)] = res
        else:
            res = "INSTABLE"
            df_nash_t.loc[num_pl_i, 'res_t{}'.format(t)] = res   
            
    return df_nash_t
    
# ____________________ checkout LRI profil -->  fin  _________________________


# _______        balanced players at t and k --> debut          ______________
def balanced_player_game_4_random_mode(arr_pl_M_T_vars_modif, t, 
                                       pi_hp_plus, pi_hp_minus, 
                                       pi_0_plus_t, pi_0_minus_t, 
                                       random_determinist, 
                                       used_storage, 
                                       manual_debug, dbg):
    
    dico_gamma_players_t = dict()
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    t_periods = arr_pl_M_T_vars_modif.shape[1]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Si']] 
        Si_max = arr_pl_M_T_vars_modif[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['gamma_i']]
        prod_i, cons_i, r_i, state_i = 0, 0, 0, ""
        state_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                  fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                             # update R_i_old
        
        # get mode_i
        mode_i = None
        # if t == 0 or random_determinist:
        #     pl_i.select_mode_i(p_i = 0.5)
        #     mode_i = pl_i.get_mode_i()
        if random_determinist:
            pl_i.select_mode_i(p_i = 0.5)
            mode_i = pl_i.get_mode_i()
        else:
            # t in [1,num_periods]
            Pi_t_plus_1 = arr_pl_M_T_vars_modif[num_pl_i, 
                                          t+1, 
                                          fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]] \
                            if t+1 < t_periods \
                            else 0
            Ci_t_plus_1 = arr_pl_M_T_vars_modif[num_pl_i, 
                                     t+1, 
                                     fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]] \
                            if t+1 < t_periods \
                            else 0
            Si_t_minus_1_minus = arr_pl_M_T_vars_modif[num_pl_i, 
                                    t-1, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] \
                                if t-1 > 0 \
                                else 0
            Si_t_minus = arr_pl_M_T_vars_modif[num_pl_i, 
                                    t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]]
            Si_t_minus_1_plus = arr_pl_M_T_vars_modif[num_pl_i, 
                                     t-1, 
                                     fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] \
                                if t-1 > 0 \
                                else 0
            
            if used_storage:
                if state_i == fct_aux.STATES[0] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_t_minus:
                    mode_i = fct_aux.STATE1_STRATS[0]                          # CONS+, state1
                elif state_i == fct_aux.STATES[0] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_t_minus:
                    mode_i = fct_aux.STATE1_STRATS[1]                          # CONS-, state1
                elif state_i == fct_aux.STATES[1] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_t_minus:
                    mode_i = fct_aux.STATE2_STRATS[1]                          # CONS-, state2
                elif state_i == fct_aux.STATES[1] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_t_minus:
                    mode_i = fct_aux.STATE2_STRATS[0]                          # DIS, state2
                elif state_i == fct_aux.STATES[2] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) >= Si_t_minus:
                    mode_i = fct_aux.STATE3_STRATS[0]                          # DIS, state3
                elif state_i == fct_aux.STATES[2] and \
                    fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1) < Si_t_minus:
                    mode_i = fct_aux.STATE3_STRATS[1]                          # PROD, state3
            else:
                if state_i == fct_aux.STATES[0]:
                    mode_i = fct_aux.STATE1_STRATS[1]           # CONS-, state1
                elif state_i == fct_aux.STATES[1]:
                    mode_i = fct_aux.STATE2_STRATS[0]           # DIS, state2
                elif state_i == fct_aux.STATES[2]:
                    mode_i = fct_aux.STATE3_STRATS[1]           # PROD, state3
                    
            pl_i.set_mode_i(mode_i)
                
        # update prod, cons and r_i
        pl_i.update_prod_cons_r_i()
    
        # print("T={}, player_{}: state_i={}, mode_i={}, prod_i={}, cons_i={}".format(
        #     t, num_pl_i, state_i, mode_i, pl_i.get_prod_i(), pl_i.get_cons_i()))
    
        # is pl_i balanced?
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        
        # update variables in arr_pl_M_T_modif
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), 
                ("mode_i", pl_i.get_mode_i()), ("state_i", pl_i.get_state_i()), 
                ("balanced_pl_i", boolean), ("formule", formule)]
        for col, val in tup_cols_values:
            arr_pl_M_T_vars_modif[num_pl_i, t, 
                                  fct_aux.AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_vars_modif, dico_gamma_players_t


def balanced_player_game_t(arr_pl_M_T_vars_modif, t, 
                            pi_hp_plus, pi_hp_minus,
                            pi_0_plus_t, pi_0_minus_t,
                            random_determinist, used_storage,
                            manual_debug, dbg):
    # find mode, prod, cons, r_i
    arr_pl_M_T_vars_modif, dico_gamma_players_t \
        = balanced_player_game_4_random_mode(
            arr_pl_M_T_vars_modif.copy(), t, 
            pi_hp_plus, pi_hp_minus,
            pi_0_plus_t, pi_0_minus_t, 
            random_determinist, 
            used_storage, 
            manual_debug, dbg)
    
    # compute pi_sg_{plus,minus}_t_k, pi_0_{plus,minus}_t_k
    b0_t, c0_t, \
    bens_t, csts_t, \
    pi_sg_plus_t, pi_sg_minus_t, \
        = fct_aux.compute_prices_inside_SG(arr_pl_M_T_vars_modif, t,
                                             pi_hp_plus, pi_hp_minus,
                                             pi_0_plus_t, pi_0_minus_t,
                                             manual_debug, dbg)
        
    return arr_pl_M_T_vars_modif, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamma_players_t
# _______        balanced players at t and k --> fin            ______________

# __________       main function of DETERMINIST   ---> debut      ____________
def determinist_balanced_player_game(arr_pl_M_T_vars_init,
                                     pi_hp_plus=0.2, 
                                     pi_hp_minus=0.33,
                                     gamma_version=1,
                                     random_determinist=False,
                                     used_storage=False,
                                     path_to_save="tests", 
                                     manual_debug=False, dbg=False):
    
    """
    create a game for balancing all players at all periods of time T_PERIODS = [0..T-1]

    Parameters
    ----------
    arr_pl_M_T: array of shape (M_PLAYERS, T_PERIODS, len(AUTOMATE_INDEX_ATTRS))
        DESCRIPTION.
    pi_hp_plus : float, optional
        DESCRIPTION. The default is 0.10.
        the price of exported energy from SG to HP
    pi_hp_minus : float, optional
        DESCRIPTION. The default is 0.15.
        the price of imported energy from HP to SG
    random_determinist: boolean, optional
        DESCRIPTION. The default is False
        decide if the mode of player a_i is randomly chosen (True) or 
        deterministly chosen (False) 
    path_to_save : String, optional
        DESCRIPTION. The default is "tests".
        name of directory for saving variables of players
    dbg : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    
    """
    print("determinist game: pi_hp_plus={}, pi_hp_minus ={} ---> debut \n"\
          .format( pi_hp_plus, pi_hp_minus))
        
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
        
    # ____          run balanced sg for all t_periods : debut         ________
    dico_stats_res = dict()
    dico_mode_prof_by_players_T = dict()
    
    dico_id_players = {"players":[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i) 
                                  for num_pl_i in range(0, m_players)]}
    df_nash = pd.DataFrame.from_dict(dico_id_players)
    
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
            pi_0_plus_t = round(pi_sg_minus_t_minus_1*pi_hp_plus/pi_hp_minus, 
                                fct_aux.N_DECIMALS)
            pi_0_minus_t = pi_sg_minus_t_minus_1
            if t == 0:
               pi_0_plus_t = fct_aux.PI_0_PLUS_INIT #4
               pi_0_minus_t = fct_aux.PI_0_MINUS_INIT #3
               
        if t == 0:
            print("before compute gamma state 4 t={}, modes={}, Sis={},  Si_old={}, ris={}".format(
            t,
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
            )) if dbg else None
        else:
            print("before compute gamma state 4 t={}, modes={}, Sis={},  Si_old={}, ris={}".format(
                t-1,
                arr_pl_M_T_vars_modif[:,t-1, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]],
                arr_pl_M_T_vars_modif[:,t-1, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
                arr_pl_M_T_vars_modif[:,t-1, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]],
                arr_pl_M_T_vars_modif[:,t-1, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
                )) if dbg else None
            print("before compute gamma state 4 t={}, modes={}, Sis={},  Si_old={}, ris={}".format(
                t,
                arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]],
                arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
                arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]],
                arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
                )) if dbg else None
            
        arr_pl_M_T_vars_modif = fct_aux.compute_gamma_state_4_period_t(
                                arr_pl_M_T_K_vars=arr_pl_M_T_vars_modif.copy(), 
                                t=t, 
                                pi_0_plus=pi_0_plus_t, pi_0_minus=pi_0_minus_t,
                                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
                                gamma_version=gamma_version,
                                manual_debug=manual_debug,
                                dbg=dbg)
        
        print("after compute gamma state 4 t={}, modes={}, Sis={},  Si_old={}, ris={}".format(
            t,
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_old"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["r_i"]]
            )) if dbg else None
            
        pi_0_plus_T[t] = pi_0_plus_t
        pi_0_minus_T[t] = pi_0_minus_t
        
        # balanced player game at instant t
        dico_gamme_t = dict()
        arr_pl_M_T_vars_modif, \
        b0_t, c0_t, \
        bens_t, csts_t, \
        pi_sg_plus_t, pi_sg_minus_t, \
        dico_gamme_t \
            = balanced_player_game_t(
                arr_pl_M_T_vars_modif.copy(), t, 
                pi_hp_plus, pi_hp_minus,
                pi_0_plus_t, pi_0_minus_t,
                random_determinist, used_storage,
                manual_debug, dbg)
        Sis = arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]].copy()
        arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]] = Sis
        print("after Sis update 4 t={}, Sis={}, modes={}".format(
            t,
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]],
            arr_pl_M_T_vars_modif[:,t, fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]]
            )) if dbg else None
        dico_stats_res[t] = dico_gamme_t
        
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        if np.isnan(pi_sg_plus_t):
            pi_sg_plus_t = 0
        if np.isnan(pi_sg_minus_t):
            pi_sg_minus_t = 0
        pi_sg_plus_T[t] = pi_sg_plus_t
        pi_sg_minus_T[t] = pi_sg_minus_t
        
        # b0_ts, c0_ts of shape (T_PERIODS,)
        b0_ts_T[t] = b0_t
        c0_ts_T[t] = c0_t
        
        # BENs, CSTs of shape (M_PLAYERS, T_PERIODS)
        BENs_M_T[:,t] = bens_t
        CSTs_M_T[:,t] = csts_t
        
        # compute Perf_t
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                                arr_pl_M_T_vars_modif, 
                                t)
        
        bens_csts_M_t = bens_t - csts_t
        Perf_t = np.sum(bens_csts_M_t, axis=0)
        dico_players = dict()
        for num_pl_i in range(0, m_players):
            Vi = bens_csts_M_t[num_pl_i]
            
            dico_vars = dict()
            dico_vars["Vi"] = round(Vi, 2)
            dico_vars["ben_i"] = round(bens_t[num_pl_i], 2)
            dico_vars["cst_i"] = round(csts_t[num_pl_i], 2)
            variables = ["set", "state_i", "mode_i", "Pi", "Ci", "Si_max", 
                         "Si_old", "Si", "prod_i", "cons_i", "r_i", 
                         "Si_minus", "Si_plus", "gamma_i"]
            for variable in variables:
                dico_vars[variable] = arr_pl_M_T_vars_modif[
                                        num_pl_i, t, 
                                        fct_aux.AUTOMATE_INDEX_ATTRS[variable]]
                
            dico_players[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i)] = dico_vars
        
        dico_players["Perf_t"] = round(Perf_t, 2)
        dico_players["b0_t"] = round(b0_t, 2)
        dico_players["c0_t"] = round(c0_t, 2)
        dico_players["Out_sg"] = round(Out_sg, 2)
        dico_players["In_sg"] = round(In_sg, 2)
        dico_players["pi_sg_plus_t"] = round(pi_sg_plus_t, 2)
        dico_players["pi_sg_minus_t"] = round(pi_sg_minus_t, 2)
        dico_players["pi_0_plus_t"] = round(pi_0_plus_t, 2)
        dico_players["pi_0_minus_t"] = round(pi_0_minus_t, 2)
        
        dico_mode_prof_by_players_T["t_"+str(t)] = dico_players
        
        ## checkout NASH equilibrium
        df_nash_t = None
        df_nash_t = checkout_nash_4_profils_by_periods(
                        arr_pl_M_T_vars_modif.copy(),
                        arr_pl_M_T_vars_init,
                        pi_hp_plus, pi_hp_minus, 
                        pi_0_minus_t, pi_0_plus_t, 
                        bens_csts_M_t,
                        t,
                        manual_debug)
        df_nash = pd.merge(df_nash, df_nash_t, on='players', how='outer')
    
    # ____          run balanced sg for all t_periods : fin           ________
    
    # __________        compute prices variables         ____________________
    # B_is, C_is of shape (M_PLAYERS, )
    prod_i_M_T = arr_pl_M_T_vars_modif[:,:t_periods, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
    cons_i_M_T = arr_pl_M_T_vars_modif[:,:t_periods, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
    
    B_is_M = np.sum(b0_ts_T * prod_i_M_T, axis=1)
    C_is_M = np.sum(c0_ts_T * cons_i_M_T, axis=1)
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is_M = np.sum(cons_i_M_T, axis=1)
    PROD_is_M = np.sum(prod_i_M_T, axis=1)
    
    BB_is_M = pi_sg_plus_T[t_periods-1] * PROD_is_M #np.sum(PROD_is)
    for num_pl, bb_i in enumerate(BB_is_M):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    CC_is_M = pi_sg_minus_T[t_periods-1] * CONS_is_M #np.sum(CONS_is)
    RU_is_M = BB_is_M - CC_is_M
    
    pi_hp_plus_s = np.array([pi_hp_plus] * t_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * t_periods, dtype=object) 
    
    #__________      save computed variables locally      _____________________ 
    algo_name = "RD-DETERMINIST" if random_determinist else "DETERMINIST"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df_nash.to_excel(os.path.join(
                *[path_to_save,
                  "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
                index=False)
    
    if m_players<=22:
        fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif, 
                       b0_ts_T, c0_ts_T, B_is_M, C_is_M, 
                       BENs_M_T, CSTs_M_T, 
                       BB_is_M, CC_is_M, RU_is_M, 
                       pi_sg_minus_T, pi_sg_plus_T, 
                       pi_0_minus_T, pi_0_plus_T,
                       pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                       algo=algo_name, 
                       dico_best_steps=dico_mode_prof_by_players_T)
    else:
        fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif, 
                    b0_ts_T, c0_ts_T, B_is_M, C_is_M, 
                    BENs_M_T, CSTs_M_T, 
                    BB_is_M, CC_is_M, RU_is_M, 
                    pi_sg_minus_T, pi_sg_plus_T, 
                    pi_0_minus_T, pi_0_plus_T,
                    pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                    algo=algo_name, 
                    dico_best_steps=dico_mode_prof_by_players_T)
        
    print("DETERMINIST GAME: pi_hp_plus={}, pi_hp_minus ={} ---> FIN \n"\
          .format( pi_hp_plus, pi_hp_minus))
    
    return arr_pl_M_T_vars_modif
    
    
# __________       main function of DETERMINIST   ---> fin        ____________

###############################################################################
#                   definition  des unittests
#
###############################################################################
def test_DETERMINIST_balanced_player_game_Pi_Ci_NEW_AUTOMATE():
    pi_hp_plus = 10 #0.2*pow(10,-3)
    pi_hp_minus = 20
    random_determinist = False #True #False
    used_storage = True #False
    
    manual_debug = False
    gamma_version = 4 #2 #1 #3: gamma_i_min #4: square_root
    
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
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    
    arr_pl_M_T_vars = \
        determinist_balanced_player_game(
                                 arr_pl_M_T_vars_init.copy(),
                                 pi_hp_plus=pi_hp_plus, 
                                 pi_hp_minus=pi_hp_minus,
                                 gamma_version=gamma_version,
                                 random_determinist=random_determinist,
                                 used_storage=used_storage,
                                 path_to_save="tests", 
                                 manual_debug=manual_debug, dbg=False)
        
    return arr_pl_M_T_vars
    
###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    arr_pl_M_T_K_vars_modif \
        = test_DETERMINIST_balanced_player_game_Pi_Ci_NEW_AUTOMATE()
    
    print("runtime = {}".format(time.time() - ti))