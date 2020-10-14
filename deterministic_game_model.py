
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 08:31:40 2020

@author: jwehounou
"""
import os
import time

import numpy as np
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import game_model_period_T as gmpT
import visu_bkh as bkh

from datetime import datetime
from pathlib import Path

def determine_new_pricing_sg(arr_pl_M_T, pi_hp_plus, pi_hp_minus, t):
    diff_energy_cons_t = 0
    diff_energy_prod_t = 0
    for k in range(0, t+1):
        energ_k_prod = \
            fct_aux.fct_positive(
            sum_list1=sum(arr_pl_M_T[:, k, fct_aux.INDEX_ATTRS["prod_i"]]),
            sum_list2=sum(arr_pl_M_T[:, k, fct_aux.INDEX_ATTRS["cons_i"]])
                    )
        energ_k_cons = \
            fct_aux.fct_positive(
            sum_list1=sum(arr_pl_M_T[:, k, fct_aux.INDEX_ATTRS["cons_i"]]),
            sum_list2=sum(arr_pl_M_T[:, k, fct_aux.INDEX_ATTRS["prod_i"]])
                    )
            
        diff_energy_cons_t += energ_k_cons
        diff_energy_prod_t += energ_k_prod
        print("k={}, energ_k_prod={}, energ_k_cons={}".format(t, energ_k_prod, energ_k_cons))
    sum_cons = sum(sum(arr_pl_M_T[:, :t+1, fct_aux.INDEX_ATTRS["cons_i"]]))
    sum_prod = sum(sum(arr_pl_M_T[:, :t+1, fct_aux.INDEX_ATTRS["prod_i"]]))
    
    new_pi_sg_minus_t = pi_hp_minus*diff_energy_cons_t / sum_cons  \
                    if sum_cons != 0 else np.nan
    new_pi_sg_plus_t = pi_hp_plus*diff_energy_prod_t / sum_prod \
                        if sum_prod != 0 else np.nan
                            
    return new_pi_sg_plus_t, new_pi_sg_minus_t

###############################################################################
#           Nouvelle version 
#
###############################################################################
def balance_player_game(pi_hp_plus = 0.10, pi_hp_minus = 0.15,
                        m_players=3, num_periods=5, path_to_save="tests", 
                        dbg=False):
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_sg_plus, pi_sg_minus = [], []
    B_is, C_is = [], []
    b0_s, c0_s = [], []
    BENs, CSTs = np.array([]), np.array([])
    # _______ variables' initialization --> fin   ________________
    
    arr_pl_M_T = fct_aux.generate_Cis_Pis_Sis_allplayer_alltime(
                    m_players=m_players, num_periods=num_periods, 
                    low_Ci=10, high_Ci=30
                    )
    
    arr_pl_M_T_old = arr_pl_M_T.copy()
    for t in range(0, num_periods):
        print("******* t = {} *******".format(t)) if dbg else None
        
        # compute pi_0_plus, pi_0_minus, pi_sg_plus, pi_sg_minus
        pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t
        pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t
        
        for num_pl_i in range(0, m_players):
            Ci = round(arr_pl_M_T[num_pl_i, t, fct_aux.INDEX_ATTRS["Ci"]],2)
            Pi = round(arr_pl_M_T[num_pl_i, t, fct_aux.INDEX_ATTRS["Pi"]],2)
            Si = round(arr_pl_M_T[num_pl_i, t, fct_aux.INDEX_ATTRS["Si"]],2)
            Si_max = round(arr_pl_M_T[num_pl_i, t, 
                                      fct_aux.INDEX_ATTRS["Si_max"]],2)
            gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
            pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                                prod_i, cons_i, r_i, state_i)
            
            # get mode_i, state_i and update R_i_old
            pl_i.set_R_i_old(Si_max-Si)
            state_i = pl_i.find_out_state_i()
            pl_i.select_mode_i()
            
            pl_i.update_prod_cons_r_i()
        
            # balancing
            boolean = fct_aux.balanced_player(pl_i, thres=0.1)
            print("_____ pl_{} _____".format(num_pl_i)) if dbg else None
            print("Pi={}, Ci={}, Si={}, Si_max={}, state_i={}, mode_i={}".format(
                   pl_i.get_Pi(),pl_i.get_Ci(),pl_i.get_Si(),pl_i.get_Si_max(), 
                   pl_i.get_state_i(), pl_i.get_mode_i() 
                )) if dbg else None
            print("====> prod_i={}, cons_i={}, new_S_i={}, R_i_old={}, r_i={}".format(
                round(pl_i.get_prod_i(),2), round(pl_i.get_cons_i(),2),
                round(pl_i.get_Si(),2), round(pl_i.get_R_i_old(),2), 
                round(pl_i.get_r_i(),2) )) if dbg else None
            print("====> balanced: {}  ".format(boolean)) if dbg else None
            
            # compute gamma_i
            Pi_t_plus_1 = arr_pl_M_T[num_pl_i, t+1, fct_aux.INDEX_ATTRS["Pi"]]
            Ci_t_plus_1 = arr_pl_M_T[num_pl_i, t+1, fct_aux.INDEX_ATTRS["Ci"]]
            pl_i.select_storage_politic(
                Ci_t_plus_1 = Ci_t_plus_1, 
                Pi_t_plus_1 = Pi_t_plus_1, 
                pi_0_plus = pi_sg_plus_t, 
                pi_0_minus = pi_sg_minus_t, 
                pi_hp_plus = pi_hp_plus, 
                pi_hp_minus = pi_hp_minus)
            
            # update variables in arr_pl_M_T
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["prod_i"]] = pl_i.get_prod_i()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["cons_i"]] = pl_i.get_cons_i()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["gamma_i"]] = pl_i.get_gamma_i()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["r_i"]] = pl_i.get_r_i()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["Si"]] = pl_i.get_Si()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["state_i"]] = pl_i.get_state_i()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["mode_i"]] = pl_i.get_mode_i()
            arr_pl_M_T_old[num_pl_i, 
                           t, 
                           fct_aux.INDEX_ATTRS["state_i"]] = pl_i.get_state_i()
            arr_pl_M_T_old[num_pl_i, 
                           t, 
                           fct_aux.INDEX_ATTRS["mode_i"]] = pl_i.get_mode_i()
            
        # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
        # from a pricing model in the document
        pi_sg_plus_t_new, pi_sg_minus_t_new = determine_new_pricing_sg(
                                                arr_pl_M_T, 
                                                pi_hp_plus, 
                                                pi_hp_minus, 
                                                t)
                            
        pi_sg_plus_t = pi_sg_plus_t if pi_sg_plus_t_new is np.nan \
                                    else pi_sg_plus_t_new
        pi_sg_minus_t = pi_sg_minus_t if pi_sg_minus_t_new is np.nan \
                                    else pi_sg_minus_t_new
        pi_sg_plus.append(pi_sg_plus_t)
        pi_sg_minus.append(pi_sg_minus_t)
        
        ## compute prices inside smart grids
        # compute In_sg, Out_sg
        In_sg, Out_sg = gmpT.compute_prod_cons_SG(arr_pl_M_T, t)
        # compute prices of an energy unit price for cost and benefit players
        b0_t, c0_t = gmpT.compute_energy_unit_price(
                        pi_sg_plus_t, pi_sg_minus_t, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg)
        b0_s.append(b0_t)
        c0_s.append(c0_t) 
        
        # compute ben, cst of shape (M_PLAYERS,) 
        # compute cost (csts) and benefit (bens) players by energy exchanged.
        gamma_is = arr_pl_M_T[:, t, fct_aux.INDEX_ATTRS["gamma_i"]]
        bens, csts = gmpT.compute_utility_players(arr_pl_M_T, 
                                                  gamma_is, 
                                                  t, 
                                                  b0_t, 
                                                  c0_t)
        print('#### bens={}'.format(bens.shape)) if dbg else None
        
        # BENs, CSTs of shape (NUM_PERIODS*M_PLAYERS,)
        BENs = np.append(BENs, bens)
        CSTs = np.append(CSTs, csts)
    
    # pi_sg_plus, pi_sg_minus of shape (NUM_PERIODS,)
    pi_sg_plus = np.array(pi_sg_plus, dtype=object).reshape((len(pi_sg_plus),))
    pi_sg_minus = np.array(pi_sg_minus, dtype=object).reshape((len(pi_sg_minus),))
    
    # BENs, CSTs of shape (M_PLAYERS, NUM_PERIODS)
    BENs = BENs.reshape(num_periods, m_players).T
    CSTs = CSTs.reshape(num_periods, m_players).T
    
    # b0_s, c0_s of shape (NUM_PERIODS,)
    b0_s = np.array(b0_s, dtype=object)
    c0_s = np.array(c0_s, dtype=object)
    
    # B_is, C_is of shape (M_PLAYERS, )
    CONS_is = np.sum(arr_pl_M_T[:,:, fct_aux.INDEX_ATTRS["cons_i"]], axis=1)
    PROD_is = np.sum(arr_pl_M_T[:,:, fct_aux.INDEX_ATTRS["prod_i"]], axis=1)
    prod_i_T = arr_pl_M_T[:,1:, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_T = arr_pl_M_T[:,1:, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is = np.sum(b0_s * prod_i_T, axis=1)
    C_is = np.sum(c0_s * cons_i_T, axis=1)
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    BB_is = pi_sg_plus[-1] * PROD_is #np.sum(PROD_is)
    CC_is = pi_sg_minus[-1] * CONS_is #np.sum(CONS_is)
    RU_is = BB_is - CC_is
    
    pi_hp_plus_s = np.array([pi_hp_plus] * num_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * num_periods, dtype=object)
    
    # save computed variables
    path_to_save = path_to_save \
                    if path_to_save != "tests" \
                    else os.path.join(
                                path_to_save, 
                                "simu_"+datetime.now().strftime("%d%m_%H%M"))
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    
    np.save(os.path.join(path_to_save, "arr_pls_M_T_old.npy"), arr_pl_M_T_old)
    np.save(os.path.join(path_to_save, "arr_pls_M_T.npy"), arr_pl_M_T)
    np.save(os.path.join(path_to_save, "b0_s.npy"), b0_s)
    np.save(os.path.join(path_to_save, "c0_s.npy"), c0_s)
    np.save(os.path.join(path_to_save, "B_is.npy"), B_is)
    np.save(os.path.join(path_to_save, "C_is.npy"), C_is)
    np.save(os.path.join(path_to_save, "BENs.npy"), BENs)
    np.save(os.path.join(path_to_save, "CSTs.npy"), CSTs)
    np.save(os.path.join(path_to_save, "BB_is.npy"), BB_is)
    np.save(os.path.join(path_to_save, "CC_is.npy"), CC_is)
    np.save(os.path.join(path_to_save, "RU_is.npy"), RU_is)
    np.save(os.path.join(path_to_save, "pi_sg_minus_s.npy"), pi_sg_minus)
    np.save(os.path.join(path_to_save, "pi_sg_plus_s.npy"), pi_sg_plus)
    np.save(os.path.join(path_to_save, "pi_hp_plus_s.npy"), pi_hp_plus_s)
    np.save(os.path.join(path_to_save, "pi_hp_minus_s.npy"), pi_hp_minus_s)
    
    return arr_pl_M_T_old, arr_pl_M_T, \
            b0_s, c0_s, \
            B_is, C_is, \
            BENs, CSTs, \
            BB_is, CC_is, RU_is, \
            pi_sg_plus, pi_sg_minus
            
            
#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------
def test_balance_player_game():
    pi_hp_plus = 0.10; pi_hp_minus = 0.15
    pi_hp_plus = 10; pi_hp_minus = 15
    m_players = 3; num_periods = 5
    
    arr_pl_M_T_old, arr_pl_M_T, \
    b0_s, c0_s, \
    B_is, C_is, \
    BENs, CSTs, \
    BB_is, CC_is, RU_is , \
    pi_sg_plus, pi_sg_minus = \
        balance_player_game(pi_hp_plus = pi_hp_plus, 
                            pi_hp_minus = pi_hp_minus,
                            m_players = m_players, 
                            num_periods = num_periods)
        
    print("_____ shape _____")
    print("arr_pl_M_T={}".format(arr_pl_M_T.shape))
    print("b0_s={}, c0_s={}".format(b0_s.shape, c0_s.shape))
    print("B_is={}, C_is={}".format(B_is.shape, C_is.shape))
    print("BENs={}, CSTs={}".format(BENs.shape, CSTs.shape))
    print("BB_is={}, CC_is={}".format(BB_is.shape, CC_is.shape))
    print("RU_is={}".format(RU_is.shape))
    
    print("BENs={}".format(BENs))
    print("CSTs={}".format(CSTs))
    print("pi_sg_plus_s={}, pi_sg_minus_s={}".format(
            pi_sg_plus.shape, pi_sg_minus.shape))
def test_determine_new_pricing_sg_and_new():
    
    arrs = [[[0,0,0,0,0,15,1], [0,0,0,0,0,18,20], [0,0,0,0,0,6,30]], 
            [[0,0,0,0,0,2,10], [0,0,0,0,0,12,17], [0,0,0,0,0,5,32]], 
            [[0,0,0,0,0,10,12], [0,0,0,0,0,10,5], [0,0,0,0,0,1,23]]]
    arrs = np.array(arrs, dtype=object)
    
    t = 2
    prod_is_0_t = gmpT.extract_values_to_array(
                    arrs, range(0,t+1), 
                    attribut_position = fct_aux.INDEX_ATTRS["prod_i"])
    cons_is_0_t = gmpT.extract_values_to_array(
                    arrs, range(0,t+1), 
                    attribut_position = fct_aux.INDEX_ATTRS["cons_i"])
    
    pi_hp_plus, pi_hp_minus = 10, 20
    new_pi_sg_plus, new_pi_sg_minus = \
       determine_new_pricing_sg_old(prod_is_0_t, cons_is_0_t, 
                             pi_hp_plus, pi_hp_minus, t, dbg=False)
    print("OLD: new_pi_sg_plus={}, new_pi_sg_minus={}".format(new_pi_sg_plus, new_pi_sg_minus))
 
    new_new_pi_sg_plus, new_new_pi_sg_minus = \
        determine_new_pricing_sg(arrs, pi_hp_plus, pi_hp_minus, t)
    print("NEW: new_pi_sg_plus={}, new_pi_sg_minus={}".format(
            new_new_pi_sg_plus, new_new_pi_sg_minus))

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_determine_new_pricing_sg_and_new()
    test_balance_player_game()
    bkh.test_plot_variables_allplayers()
    print("runtime = {}".format(time.time() - ti))