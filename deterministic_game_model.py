
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 08:31:40 2020

@author: jwehounou
"""
import os
import time

import numpy as np
import pandas as pd
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import game_model_period_T as gmpT
import visu_bkh as bkh

from datetime import datetime
from pathlib import Path

def determine_new_pricing_sg(arr_pl_M_T, pi_hp_plus, pi_hp_minus, t, dbg=False):
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
        print("k={}, energ_k_prod={}, energ_k_cons={}".format(
            t, energ_k_prod, energ_k_cons)) if dbg else None
        # print("k={}, energ_k_prod={}, energ_k_cons={}".format(
        #     t, energ_k_prod, energ_k_cons))
        ## debug
        bool_ = arr_pl_M_T[:, k, fct_aux.INDEX_ATTRS["prod_i"]]>0
        unique,counts=np.unique(bool_,return_counts=True)
        sum_prod_k = round(np.sum(arr_pl_M_T[:, k, fct_aux.INDEX_ATTRS["prod_i"]]),2)
        sum_cons_k = round(np.sum(arr_pl_M_T[:, k, fct_aux.INDEX_ATTRS["cons_i"]]),2)
        diff_sum_prod_cons_k = sum_prod_k - sum_cons_k
        print("t={}, k={}, unique:{}, counts={}, sum_prod_k={}, sum_cons_k={}, diff_sum_k={}".format(
                t,k,unique, counts, sum_prod_k, sum_cons_k, diff_sum_prod_cons_k))
        ## debug
    
    sum_cons = sum(sum(arr_pl_M_T[:, :t+1, fct_aux.INDEX_ATTRS["cons_i"]].astype(np.float64)))
    sum_prod = sum(sum(arr_pl_M_T[:, :t+1, fct_aux.INDEX_ATTRS["prod_i"]].astype(np.float64)))
    
    
    # for k in range(0, t+1):
    #     bool_ = arr_pl_M_T[:, k, fct_aux.INDEX_ATTRS["prod_i"]]>0
    #     unique,counts=np.unique(bool_,return_counts=True)
    #     print("t={}, k={}, unique:{}, counts={}".format(t,k,unique, counts))
    # print("t={}, sum_diff_energy_cons_t={}, sum_diff_energy_prod_t={}, sum_cons={}, sum_prod={}".format(
    #     t, round(diff_energy_cons_t,2), round(diff_energy_prod_t,2), 
    #         round(sum_cons,2), round(sum_prod,2) ))
    # # print("sum_cons={}, sum_prod={}".format(
    # #         round(sum_cons,2), round(sum_prod,2))) \
    # #         if t%20 == 0 \
    # #         else None
    print("NAN: cons={}, prod={}".format(
            np.isnan(arr_pl_M_T[:, :t+1, fct_aux.INDEX_ATTRS["cons_i"]].astype(np.float64)).any(),
            np.isnan(arr_pl_M_T[:, :t+1, fct_aux.INDEX_ATTRS["prod_i"]].astype(np.float64)).any())
        ) if dbg else None
    arr_cons = np.argwhere(np.isnan(arr_pl_M_T[:, :t+1, fct_aux.INDEX_ATTRS["cons_i"]].astype(np.float64)))
    arr_prod = np.argwhere(np.isnan(arr_pl_M_T[:, :t+1, fct_aux.INDEX_ATTRS["prod_i"]].astype(np.float64)))
    # print("positions of nan cons:{}, prod={}".format(arr_cons, arr_prod))
    # print("state")
    if arr_cons.size != 0:
        for arr in arr_cons:
            print("{}-->state:{}, Pi={}, Ci={}, Si={}".format(
                arr, arr_pl_M_T[arr[0], arr[1], fct_aux.INDEX_ATTRS["state_i"]],
                arr_pl_M_T[arr[0], arr[1], fct_aux.INDEX_ATTRS["Pi"]],
                arr_pl_M_T[arr[0], arr[1], fct_aux.INDEX_ATTRS["Ci"]],
                arr_pl_M_T[arr[0], arr[1], fct_aux.INDEX_ATTRS["Si"]]))
    
    new_pi_sg_minus_t = round(pi_hp_minus*diff_energy_cons_t / sum_cons,2)  \
                    if sum_cons != 0 else np.nan
    new_pi_sg_plus_t = round(pi_hp_plus*diff_energy_prod_t / sum_prod,2) \
                        if sum_prod != 0 else np.nan
                            
    return new_pi_sg_plus_t, new_pi_sg_minus_t

###############################################################################
#           Nouvelle version 
#
###############################################################################
def balance_player_game(pi_hp_plus = 0.10, pi_hp_minus = 0.15,
                        m_players=3, num_periods=5, 
                        Ci_low=10, Ci_high=30,
                        prob_Ci=0.3, scenario="scenario1", 
                        path_to_save="tests", dbg=False):
    """
    create a game for balancing all players at all periods of time NUM_PERIODS = [1..T]

    Parameters
    ----------
    pi_hp_plus : float, optional
        DESCRIPTION. The default is 0.10.
        the price of exported energy from SG to HP
    pi_hp_minus : float, optional
        DESCRIPTION. The default is 0.15.
        the price of imported energy from HP to SG
    m_players : Integer, optional
        DESCRIPTION. The default is 3.
        the number of players
    num_periods : Integer, optional
        DESCRIPTION. The default is 5.
        the number of time intervals 
    Ci_low : float, optional
        DESCRIPTION. The default is 10.
        the min value of the consumption
    Ci_high : float, optional
        DESCRIPTION. The default is 30.
        the max value of the consumption
    prob_Ci : float, optional
        DESCRIPTION. The default is 0.3.
        probability for choosing the kind of consommation
    scenario : String, optional
        DESCRIPTION. The default is "scenario1".
        a plan for operating a game of players
    path_to_save : String, optional
        DESCRIPTION. The default is "tests".
        name of directory for saving variables of players
    dbg : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    arr_pl_M_T_old : array of shape (M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))
        DESCRIPTION.
        initial array of variables for all players. it contains initial values 
        before starting a game 
    arr_pl_M_T : array of shape (M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))
        DESCRIPTION.
        array of variables for all players. it contains final values 
        at time NUM_PERIODS
    b0_s : array of shape(NUM_PERIODS,) 
        DESCRIPTION.
        array of unit price of benefit for all periods
    c0_s : array of shape(NUM_PERIODS,) 
        DESCRIPTION.
        array of unit price of cost for all periods
    B_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        array of global benefit for all players
    C_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        array of global cost for all players
    BENs : array of shape (M_PLAYERS, NUM_PERIODS)
        DESCRIPTION.
        array of benefits of a player at for all times t
    CSTs : array of shape (M_PLAYERS, NUM_PERIODS)
        DESCRIPTION.
        array of costs of a player at for all times t
    BB_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        array of real money a player need to pay if it depends on SG
    CC_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        array of real money a player need to pay if it depends on HP 
    RU_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        the difference between BB_i and CC_i for all players
    pi_sg_plus : array of shape (NUM_PERIODS,)
        DESCRIPTION.
        array of exported unit price from player to SG at all time  
    pi_sg_minus : array of shape (NUM_PERIODS,)
        DESCRIPTION.
        array of imported unit price from player to SG at all time 

    """
    print("{}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> debut \n".format(
            scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_sg_plus, pi_sg_minus = [], []
    pi_0_plus, pi_0_minus = [], []
    B_is, C_is = [], []
    b0_s, c0_s = [], []
    BENs, CSTs = np.array([]), np.array([])
    # _______ variables' initialization --> fin   ________________
    
    arr_pl_M_T = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                                    m_players=m_players, 
                                    num_periods=num_periods, 
                                    scenario=scenario, prob_Ci=prob_Ci, 
                                    Ci_low=Ci_low, Ci_high=Ci_high)
    
    arr_pl_M_T_old = arr_pl_M_T.copy()
    dico_stats_res={}
    for t in range(0, num_periods):
        # print("******* t = {} *******".format(t)) if dbg else None
        # print("___t = {}, pi_sg_plus_t={}, pi_sg_minus_t={}".format(
        #         t, pi_sg_plus_t, pi_sg_minus_t)) \
        #     if t%20 == 0 \
        #     else None
        
        # compute pi_0_plus, pi_0_minus, pi_sg_plus, pi_sg_minus
        pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t
        pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t
        
        cpt_error_gamma = 0; cpt_balanced = 0;
        dico_state_mode_i = {}; dico_balanced_pl_i = {}
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
            boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
            cpt_balanced += round(1/m_players, 2) if boolean else 0
            dico_balanced_pl_i["cpt"] = cpt_balanced
            if "player" in dico_balanced_pl_i and boolean is False:
                dico_balanced_pl_i['player'].append(num_pl_i)
            elif boolean is False:
                dico_balanced_pl_i['player'] = [num_pl_i]
            
            print("_____ pl_{} _____".format(num_pl_i)) if dbg else None
            print("Pi={}, Ci={}, Si_old={}, Si={}, Si_max={}, state_i={}, mode_i={}"\
                  .format(
                   pl_i.get_Pi(), pl_i.get_Ci(), pl_i.get_Si_old(), pl_i.get_Si(),
                   pl_i.get_Si_max(), pl_i.get_state_i(), pl_i.get_mode_i() 
                )) if dbg else None
            print("====> prod_i={}, cons_i={}, new_S_i={}, new_Si_old={}, R_i_old={}, r_i={}".format(
                round(pl_i.get_prod_i(),2), round(pl_i.get_cons_i(),2),
                round(pl_i.get_Si(),2), round(pl_i.get_Si_old(),2), 
                round(pl_i.get_R_i_old(),2), round(pl_i.get_r_i(),2) )) \
                if dbg else None
            print("====> balanced: {}  ".format(boolean)) if dbg else None
            
            # compute gamma_i
            Pi_t_plus_1 = arr_pl_M_T[num_pl_i, t+1, fct_aux.INDEX_ATTRS["Pi"]] \
                            if t+1 < num_periods \
                            else 0
            Ci_t_plus_1 = arr_pl_M_T[num_pl_i, 
                                     t+1, 
                                     fct_aux.INDEX_ATTRS["Ci"]] \
                            if t+1 < num_periods \
                            else 0
                         
            pl_i.select_storage_politic(
                Ci_t_plus_1 = Ci_t_plus_1, 
                Pi_t_plus_1 = Pi_t_plus_1, 
                pi_0_plus = pi_sg_plus_t, 
                pi_0_minus = pi_sg_minus_t, 
                pi_hp_plus = pi_hp_plus, 
                pi_hp_minus = pi_hp_minus)
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
                # print(" *** error gamma_i: state_i={}, mode_i={}".format(
                #     pl_i.state_i, pl_i.mode_i))
            
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
                       fct_aux.INDEX_ATTRS["Si_old"]] = pl_i.get_Si_old()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["state_i"]] = pl_i.get_state_i()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["mode_i"]] = pl_i.get_mode_i()
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["balanced_pl_i"]] = boolean
            arr_pl_M_T[num_pl_i, 
                       t, 
                       fct_aux.INDEX_ATTRS["formule"]] = formule
            arr_pl_M_T_old[num_pl_i, 
                           t, 
                           fct_aux.INDEX_ATTRS["state_i"]] = pl_i.get_state_i()
            arr_pl_M_T_old[num_pl_i, 
                           t, 
                           fct_aux.INDEX_ATTRS["mode_i"]] = pl_i.get_mode_i()
        
        dico_stats_res[t] = (round(cpt_balanced/m_players,2),
                             round(cpt_error_gamma/m_players,2), 
                             dico_state_mode_i)
        dico_stats_res[t] = {"balanced": dico_balanced_pl_i, 
                             "gamma_i": dico_state_mode_i}    
            
        # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
        # from a pricing model in the document
        pi_sg_plus_t_new, pi_sg_minus_t_new = determine_new_pricing_sg(
                                                arr_pl_M_T, 
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
        pi_0_plus_t = round(pi_sg_minus_t*pi_hp_plus/pi_hp_minus, 2)
        pi_0_minus_t = pi_sg_minus_t
        
        pi_sg_plus.append(pi_sg_plus_t)
        pi_sg_minus.append(pi_sg_minus_t)
        pi_0_plus.append(pi_0_plus_t)
        pi_0_minus.append(pi_0_minus_t)
        
        ## compute prices inside smart grids
        # compute In_sg, Out_sg
        In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_M_T, t)
        # compute prices of an energy unit price for cost and benefit players
        b0_t, c0_t = fct_aux.compute_energy_unit_price(
                        pi_0_plus_t, pi_0_minus_t, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg)
        b0_s.append(b0_t)
        c0_s.append(c0_t) 
        
        # compute ben, cst of shape (M_PLAYERS,) 
        # compute cost (csts) and benefit (bens) players by energy exchanged.
        gamma_is = arr_pl_M_T[:, t, fct_aux.INDEX_ATTRS["gamma_i"]]
        bens, csts = fct_aux.compute_utility_players(arr_pl_M_T, 
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
    pi_0_plus = np.array(pi_0_plus, dtype=object).reshape((len(pi_0_plus),))
    pi_0_minus = np.array(pi_0_minus, dtype=object).reshape((len(pi_0_minus),))

    
    # BENs, CSTs of shape (M_PLAYERS, NUM_PERIODS)
    BENs = BENs.reshape(num_periods, m_players).T
    CSTs = CSTs.reshape(num_periods, m_players).T
    
    # b0_s, c0_s of shape (NUM_PERIODS,)
    b0_s = np.array(b0_s, dtype=object)
    c0_s = np.array(c0_s, dtype=object)
    
    # B_is, C_is of shape (M_PLAYERS, )
    CONS_is = np.sum(arr_pl_M_T[:,:, fct_aux.INDEX_ATTRS["cons_i"]], axis=1)
    PROD_is = np.sum(arr_pl_M_T[:,:, fct_aux.INDEX_ATTRS["prod_i"]], axis=1)
    prod_i_T = arr_pl_M_T[:,:, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_T = arr_pl_M_T[:,:, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is = np.sum(b0_s * prod_i_T, axis=1)
    C_is = np.sum(c0_s * cons_i_T, axis=1)
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    BB_is = pi_sg_plus[-1] * PROD_is #np.sum(PROD_is)
    for num_pl, bb_i in enumerate(BB_is):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
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
    np.save(os.path.join(path_to_save, "pi_0_minus_s.npy"), pi_0_minus)
    np.save(os.path.join(path_to_save, "pi_0_plus_s.npy"), pi_0_plus)
    np.save(os.path.join(path_to_save, "pi_hp_plus_s.npy"), pi_hp_plus_s)
    np.save(os.path.join(path_to_save, "pi_hp_minus_s.npy"), pi_hp_minus_s)
    pd.DataFrame.from_dict(dico_stats_res)\
        .to_csv(os.path.join(path_to_save, "stats_res.csv"))
        
    print("{}, probCi={}, dico_stats_res[t=0]={} ---> fin \n".format( scenario, prob_Ci,
            len(dico_stats_res[0]["gamma_i"])))
    # print("dico_stats_res={}".format(dico_stats_res))
    
    return arr_pl_M_T_old, arr_pl_M_T, \
            b0_s, c0_s, \
            B_is, C_is, \
            BENs, CSTs, \
            BB_is, CC_is, RU_is, \
            pi_0_plus, pi_0_minus, \
            pi_sg_plus, pi_sg_minus, \
            dico_stats_res
            
            
#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------
def test_balance_player_game():
    pi_hp_plus = 0.10; pi_hp_minus = 0.15
    pi_hp_plus = 10; pi_hp_minus = 15
    m_players = 3; num_periods = 5
    Ci_low = 10; Ci_high = 30
    prob_Ci = 0.3; scenario = "scenario1"
    path_to_save = "tests"
    
    arr_pl_M_T_old, arr_pl_M_T, \
    b0_s, c0_s, \
    B_is, C_is, \
    BENs, CSTs, \
    BB_is, CC_is, RU_is , \
    pi_sg_plus, pi_sg_minus, \
    pi_0_plus, pi_0_minus, \
    dico_stats_res = \
        balance_player_game(pi_hp_plus = pi_hp_plus, 
                            pi_hp_minus = pi_hp_minus,
                            m_players = m_players, 
                            num_periods = num_periods,
                            Ci_low = Ci_low, Ci_high = Ci_high,
                            prob_Ci = prob_Ci, scenario = scenario,
                            path_to_save = path_to_save, dbg = True
                            )
        
        
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
    print("pi_0_plus_s={}, pi_0_minus_s={}".format(
            pi_0_plus.shape, pi_0_minus.shape))
    
    return dico_stats_res
    
def test_determine_new_pricing_sg_and_new():
    
    arrs = [[[0,0,0,0,0,15,1], [0,0,0,0,0,18,20], [0,0,0,0,0,6,30]], 
            [[0,0,0,0,0,2,10], [0,0,0,0,0,12,17], [0,0,0,0,0,5,32]], 
            [[0,0,0,0,0,10,12], [0,0,0,0,0,10,5], [0,0,0,0,0,1,23]]]
    arrs = np.array(arrs, dtype=object)
    
    t = 2
    prod_is_0_t = arrs[:, range(0,t+1), fct_aux.INDEX_ATTRS["prod_i"]]
    cons_is_0_t = arrs[:, range(0,t+1), fct_aux.INDEX_ATTRS["cons_i"]]
    pi_hp_plus, pi_hp_minus = 10, 20
    new_pi_sg_plus, new_pi_sg_minus = \
       gmpT.determine_new_pricing_sg(prod_is_0_t, cons_is_0_t, 
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
    dico_stats_res = test_balance_player_game()
    #bkh.test_plot_variables_allplayers()
    print("runtime = {}".format(time.time() - ti))