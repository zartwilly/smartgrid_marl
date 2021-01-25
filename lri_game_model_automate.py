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
def balanced_player_game_4_random_mode(arr_pl_M_T_K_vars_modif, t, k, dbg):
    m_players = arr_pl_M_T_K_vars_modif.shape[0]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Si']] \
            if k == 0 \
            else arr_pl_M_T_K_vars_modif[num_pl_i, t, k-1, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
        Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        gamma_i, prod_i, cons_i, r_i = 0, 0, 0, 0
        state_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                             # update R_i_old
        
        # select mode for player num_pl_i
        p_i_t_k = arr_pl_M_T_K_vars[num_pl_i, 
                                    t, k, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]] \
            if k == 0 \
            else arr_pl_M_T_K_vars[num_pl_i, 
                                    t, k-1, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["p_i_j_k"]]
        pl_i.select_mode_i(p_i=p_i_t_k)
        
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
                                    fct_aux.INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_K_vars_modif
        
def compute_gamma_4_players(arr_pl_M_T_K_vars_modif, t, k, 
                            pi_0_plus_t_k, pi_0_minus_t_k,
                            pi_hp_plus, pi_hp_minus,
                            manual_debug=False, dbg=False):
    
    dico_gamma_players = dict()
    
    m_players = arr_pl_M_T_K_vars_modif.shape[0]
    t_periods = arr_pl_M_T_K_vars_modif.shape[1]
    for num_pl_i in range(0, m_players):
        Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Pi']]
        Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Ci']]
        Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, k, 
                                   fct_aux.AUTOMATE_INDEX_ATTRS['Si']]
        Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['Si_max']]
        state_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['state_i']]
        mode_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['mode_i']]
        prod_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['prod_i']]
        cons_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['cons_i']]
        r_i = arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                                 fct_aux.AUTOMATE_INDEX_ATTRS['r_i']]
        
        gamma_i = 0
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_mode_i(mode_i)
        
        # compute gamma_i, Si_{plus,minus}
        Pi_t_plus_1_k \
            = arr_pl_M_T_K_vars[num_pl_i, t+1, k, 
                                fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]] \
                if t+1 < t_periods \
                else 0
        Ci_t_plus_1_k \
            = arr_pl_M_T_K_vars[num_pl_i, t+1, k, 
                                fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]] \
                if t+1 < t_periods \
                else 0
                     
        pl_i.select_storage_politic(
            Ci_t_plus_1 = Ci_t_plus_1_k, 
            Pi_t_plus_1 = Pi_t_plus_1_k, 
            pi_0_plus = pi_0_plus_t_k,
            pi_0_minus = pi_0_minus_t_k,
            pi_hp_plus = pi_hp_plus, 
            pi_hp_minus = pi_hp_minus)
        
        gamma_i = None
        if manual_debug:
            gamma_i = fct_aux.MANUEL_DBG_GAMMA_I # 5
            pl_i.set_gamma_i(gamma_i)
        else:
            gamma_i = pl_i.get_gamma_i()
        
        dico = dict()
        if gamma_i < min(pi_0_plus_t_k, pi_0_minus_t_k)-1:
            dico["min_pi_0"] = gamma_i
        elif gamma_i > max(pi_hp_minus, pi_hp_plus):
            dico["max_pi_hp"] = gamma_i
            
        dico["state_i"] = state_i; dico["mode_i"] = mode_i 
        dico_gamma_players["player_"+str(num_pl_i)] = dico
    
        # update variables gamma_i, Si_minus, Si_max
        tup_cols_values = [("gamma_i", gamma_i), 
                           ("Si_minus", pl_i.get_Si_minus() ),
                           ("Si_plus", pl_i.get_Si_plus() )]
        for col, val in tup_cols_values:
            arr_pl_M_T_K_vars_modif[num_pl_i, t, k,
                              fct_aux.AUTOMATE_INDEX_ATTRS[col]] = val
            
    return arr_pl_M_T_K_vars_modif, dico_gamma_players
        
def compute_prices_inside_SG(arr_pl_M_T_K_vars_modif, t, k,
                             pi_hp_plus, pi_hp_minus, 
                             manual_debug, dbg):
    """
    compute the prices' and benefits/costs variables: 
        ben_i, cst_i
        pi_sg_plus_t_k, pi_sg_minus_t_k
        pi_0_plus_t_k, pi_0_minus_t_k
    """
    pi_sg_plus_t_k_new, pi_sg_minus_t_k_new \
        = fct_aux.determine_new_pricing_sg(
            arr_pl_M_T_K_vars_modif[:,:,k,:], 
            pi_hp_plus, 
            pi_hp_minus, 
            t, 
            dbg=dbg)
    print("pi_sg_plus_{}_{}_new={}, pi_sg_minus_{}_{}_new={}".format(
        t, k, pi_sg_plus_t_k_new, t, k, pi_sg_minus_t_k_new)) if dbg else None
    
    if t == 0 and k == 0: 
        pi_sg_plus_t_k = pi_hp_plus-1
        pi_sg_minus_t_k = pi_hp_minus-1
    elif t == 0 and k > 0: 
        pi_sg_plus_t_k = pi_sg_plus_t_k_new
        pi_sg_minus_t_k = pi_sg_minus_t_k_new
    else:
        pi_sg_plus_t_k = pi_sg_plus_t_k_new
        pi_sg_minus_t_k = pi_sg_minus_t_k_new
    
    
    pi_0_plus_t_k = round(pi_sg_minus_t_k*pi_hp_plus/pi_hp_minus, 
                          fct_aux.N_DECIMALS)
    pi_0_minus_t_k = pi_sg_minus_t_k
    
    
    if manual_debug:
        pi_sg_plus_t_k = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
        pi_sg_minus_t_k = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
        pi_0_plus_t_k = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #2 
        pi_0_minus_t_k = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
        
    print(", pi_0_plus_t_k={}, pi_0_minus_t_k={},".format(
        pi_0_plus_t_k, pi_0_minus_t_k)) \
        if dbg else None
    print("pi_sg_plus_t_k={}, pi_sg_minus_t_k={} \n".format(pi_sg_plus_t_k, 
                                                            pi_sg_minus_t_k)) \
        if dbg else None
    
    
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(
                        arr_pl_M_T_K_vars_modif[:,:,k,:], t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t_k, c0_t_k = fct_aux.compute_energy_unit_price(
                        pi_0_plus_t_k, pi_0_minus_t_k, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg) 
    
    # compute ben, cst of shapes (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_K_vars_modif[:, t, k, 
                                       fct_aux.AUTOMATE_INDEX_ATTRS["gamma_i"]]
    bens_t_k, csts_t_k = fct_aux.compute_utility_players(
                            arr_pl_M_T_K_vars[:,t,:,:], 
                            gamma_is, 
                            k, 
                            b0_t_k, 
                            c0_t_k)
    print('#### bens_t_k={}, csts_t_k={}'.format(
            bens_t_k.shape, csts_t_k.shape)) \
        if dbg else None
    
    return arr_pl_M_T_K_vars_modif, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k

        

def balanced_player_game_t(arr_pl_M_T_K_vars_modif, t, k, 
                           pi_hp_plus, pi_hp_minus, 
                           m_players, indices_non_playing_players, t_periods, 
                           manual_debug=False, dbg=False):
    
    # find mode, prod, cons, r_i
    arr_pl_M_T_K_vars_modif = balanced_player_game_4_random_mode(
                                arr_pl_M_T_K_vars_modif.copy(), t, k, dbg)
    
    # compute pi_sg_{plus,minus}_t_k, pi_0_{plus,minus}_t_k
    arr_pl_M_T_K_vars_modif, \
    b0_t_k, c0_t_k, \
    bens_t_k, csts_t_k, \
    pi_sg_plus_t_k, pi_sg_minus_t_k, \
    pi_0_plus_t_k, pi_0_minus_t_k \
        = compute_prices_inside_SG(arr_pl_M_T_K_vars_modif, t, k,
                             pi_hp_plus, pi_hp_minus, 
                             manual_debug, dbg)
    
    # compute gamma_i, Si_{plus,minus}
    arr_pl_M_T_K_vars_modif, dico_gamma_players \
        = compute_gamma_4_players(
            arr_pl_M_T_K_vars_modif.copy(), t, k, 
            pi_0_plus_t_k, pi_0_minus_t_k,
            pi_hp_plus, pi_hp_minus,
            manual_debug=False, dbg=False)
        
    return arr_pl_M_T_K_vars_modif, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k, \
            dico_gamma_players
    


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
    
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = None, None
    if manual_debug:
        pi_sg_plus_t_minus_1 = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K # 8 
        pi_sg_minus_t_minus_1 = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K # 10
    else:
        pi_sg_plus_t_minus_1 = pi_hp_plus-1
        pi_sg_minus_t_minus_1 = pi_hp_minus-1
        
    for t in range(0, t_periods):
        print("******* t = {} *******".format(t))
        
        # pi_sg_plus_t = pi_sg_plus_t_minus_1 if t == 0 \
        #                                     else pi_sg_plus_T_K[t, k_steps]
        # pi_sg_minus_t = pi_sg_minus_t_minus_1 if t == 0 \
        #                                     else pi_sg_minus_T_K[t, k_steps]
        # pi_0_plus_t = round(pi_sg_plus_t*pi_hp_plus/pi_hp_minus, 
        #                     fct_aux.N_DECIMALS) if t == 0 \
        #                                         else pi_0_plus_T_K[t, k_steps]
        # pi_0_minus_t = pi_sg_minus_t if t == 0 \
        #                              else pi_0_minus_T_K[t, k_steps]
                                            
        
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
            
             
            # TODO: a demander a Dominique si a chaque k pi_sg_{plus,minus}_t_k = {8,10}
            # if manual_debug:
            #     pi_sg_plus_t_k = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
            #     pi_sg_minus_t_k = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
            
            ### balanced_player_game_t
            arr_pl_M_T_K_vars_modif, \
            b0_t_k, c0_t_k, \
            bens_t_k, csts_t_k, \
            pi_sg_plus_t_k, pi_sg_minus_t_k, \
            pi_0_plus_t_k, pi_0_minus_t_k, \
            dico_gamma_players \
                = balanced_player_game_t(arr_pl_M_T_K_vars_modif.copy(), t, k, 
                           pi_hp_plus, pi_hp_minus, 
                           m_players, indices_non_playing_players, t_periods, 
                           manual_debug, dbg=False)
                
            # compute utility
            
            # if bg_min == bg_max et nb_max_repeat = True on reprend 
            # if bg_min == bg_max et nb_max_repeat = False on maj variables et on passe
            # k+1 ou t+1
            pi_sg_plus_T_K[t,k] = pi_sg_plus_t_k
            pi_sg_minus_T_K[t,k] = pi_sg_minus_t_k
            pi_0_plus_T_K[t,k] = pi_0_plus_t_k
            pi_0_minus_T_K[t,k] = pi_0_minus_t_k
                
            
                                
            
            
    
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