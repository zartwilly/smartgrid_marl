# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:00:30 2021

@author: jwehounou
"""
import os
import time

import numpy as np
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux


#------------------------------------------------------------------------------
#                       definition of functions 
#
#------------------------------------------------------------------------------
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
        gamma_i, prod_i, cons_i, r_i = 0, 0, 0, 0
        state_i = arr_pl_M_T_vars_modif[num_pl_i, t,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                             # update R_i_old
        
        # get mode_i
        mode_i = None
        if t == 0 or random_determinist:
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
            Si_t_minus_1_plus = arr_pl_M_T_vars_modif[num_pl_i, 
                                     t-1, 
                                     fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] \
                                if t-1 > 0 \
                                else 0
            
            if used_storage:
                if state_i == fct_aux.STATES[0] \
                    and fct_aux.fct_positive(
                            Ci_t_plus_1, 
                            Pi_t_plus_1) <= Si_t_minus_1_minus:
                    mode_i = fct_aux.STATE1_STRATS[0]           # CONS+, state1
                elif state_i == fct_aux.STATES[0] \
                    and fct_aux.fct_positive(
                            Ci_t_plus_1, 
                            Pi_t_plus_1) > Si_t_minus_1_minus:
                    mode_i = fct_aux.STATE1_STRATS[1]           # CONS-, state1
                elif state_i == fct_aux.STATES[1] \
                    and fct_aux.fct_positive(
                            Ci_t_plus_1, 
                            Pi_t_plus_1) <= Si_t_minus_1_minus:
                    mode_i = fct_aux.STATE2_STRATS[1]           # CONS-, state2
                elif state_i == fct_aux.STATES[1] \
                    and fct_aux.fct_positive(
                            Ci_t_plus_1, 
                            Pi_t_plus_1) > Si_t_minus_1_minus:
                    mode_i = fct_aux.STATE2_STRATS[0]           # DIS, state2
                elif state_i == fct_aux.STATES[2] \
                    and fct_aux.fct_positive(
                            Ci_t_plus_1, 
                            Pi_t_plus_1) <= Si_t_minus_1_minus:
                    mode_i = fct_aux.STATE3_STRATS[0]           # DIS, state3
                elif state_i == fct_aux.STATES[2] \
                    and fct_aux.fct_positive(
                            Ci_t_plus_1, 
                            Pi_t_plus_1) > Si_t_minus_1_minus:
                    mode_i = fct_aux.STATE3_STRATS[1]           # PROD, state3
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
        
        # compute gamma_i
        Pi_t_plus_1 = arr_pl_M_T_vars_modif[num_pl_i, t+1, 
                                            fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]] \
                        if t+1 < t_periods \
                        else 0
        Ci_t_plus_1 = arr_pl_M_T_vars_modif[num_pl_i, 
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
                ("Si_old", pl_i.get_Si_old()), ("mode_i", pl_i.get_mode_i()), 
                ("balanced_pl_i", boolean), ("formule", formule), 
                ("gamma_i", gamma_i), 
                ("Si_minus", pl_i.get_Si_minus() ),
                ("Si_plus", pl_i.get_Si_plus() )]
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

# ________       main function of DETERMINIST   ---> debut      _______________

def determinist_balanced_player_game(arr_pl_M_T_vars_init,
                                     pi_hp_plus=0.2, 
                                     pi_hp_minus=0.33,
                                     random_determinist=False,
                                     used_storage=False,
                                     path_to_save="tests", 
                                     manual_debug=False, dbg=False):
    
    """
    create a game for balancing all players at all periods of time T_PERIODS = [0..T-1]

    Parameters
    ----------
    arr_pl_M_T: array of shape (M_PLAYERS, T_PERIODS, len(INDEX_ATTRS))
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
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] = 0.5
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
        dico_stats_res[t] = dico_gamme_t
        
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        pi_sg_plus_T[t] = pi_sg_plus_t
        pi_sg_minus_T[t] = pi_sg_minus_t
        
        # b0_ts, c0_ts of shape (T_PERIODS,)
        b0_ts_T[t] = b0_t
        c0_ts_T[t] = c0_t
        
        # BENs, CSTs of shape (M_PLAYERS, T_PERIODS)
        BENs_M_T[:,t] = bens_t
        CSTs_M_T[:,t] = csts_t
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
    algo_name = "RD-DETERMINIST" if random_determinist else "DETERMINIST"
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif, 
                   b0_ts_T, c0_ts_T, B_is_M, C_is_M, 
                   BENs_M_T, CSTs_M_T, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T, pi_sg_plus_T, 
                   pi_0_minus_T, pi_0_plus_T,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo=algo_name, 
                   dico_best_steps=dict())
        
    print("determinist game: pi_hp_plus={}, pi_hp_minus ={} ---> FIN \n"\
          .format( pi_hp_plus, pi_hp_minus))
    
    return arr_pl_M_T_vars_modif
    
    
# ________       main function of DETERMINIST   ---> fin        _______________

#------------------------------------------------------------------------------
#                       definition of unittests 
#
#------------------------------------------------------------------------------
def test_DETERMINIST_balanced_player_game():
    
    pi_hp_plus = 0.2*pow(10,-3)
    pi_hp_minus = 0.33
    random_determinist = False #True #False
    used_storage = False #True #False
    
    manual_debug=True
    
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
    
    arr_pl_M_T_vars = \
        determinist_balanced_player_game(
                                 arr_pl_M_T_vars_init.copy(),
                                 pi_hp_plus=pi_hp_plus, 
                                 pi_hp_minus=pi_hp_minus,
                                 random_determinist=random_determinist,
                                 used_storage=used_storage,
                                 path_to_save="tests", 
                                 manual_debug=manual_debug, dbg=False)
        
    return arr_pl_M_T_vars

#------------------------------------------------------------------------------
#                       execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    arr_M_T_vars = test_DETERMINIST_balanced_player_game()
    print("runtime = {}".format(time.time() - ti))