# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:42:13 2020

@author: jwehounou
"""
import os
import time

import numpy as np
import pandas as pd
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import deterministic_game_model as detGameModel

import visu_bkh as bkh

from datetime import datetime
from pathlib import Path


###############################################################################
#                   definition  des fonctions
#
###############################################################################
def save_variables(path_to_save, arr_pl_M_T_old, arr_pl_M_T, 
                   b0_s, c0_s, B_is, C_is, BENs, CSTs, BB_is, CC_is, 
                   RU_is, pi_sg_minus, pi_sg_plus, pi_0_minus, pi_0_plus,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res):
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
        
    print("saved computed variables *****")


def find_optimal_vars_on_nsteps(vars_nsteps, b0_s, c0_s, t, m_players, thres):
    """
    implementation a max value of vars_nsteps. I write a glouton algorithm with 
    a threshold 0.10

    Parameters
    ----------
    vars_nsteps : list of variables of nsteps items
        DESCRIPTION.
        variables is a list of 11 items such as 
            arr_pl_M_T_old_t_nstep, arr_pl_M_T_t_nstep, \
            b0_t_nstep, c0_t_nstep, bens_nstep, csts_nstep, \
            pi_sg_plus_t_minus_1_nstep, pi_sg_minus_t_minus_1_nstep, \
            pi_0_plus_t_nstep, pi_0_minus_t_nstep, \
            dico_stats_res_t_nstep
    Returns
    -------
    nstep : integer
        DESCRIPTION.
        number of nsteps chosen
    B_i_t : array of shape (n_player, t+1)
        DESCRIPTION
        list of n_players. Each player has t+1 items
    vars_nstep : one variable being an 11 items' list

    """
    # rd_num = np.random.randint(0, len(vars_nsteps))
    # nstep = rd_num
    # b0_t_s = vars_nsteps[nstep][2]
    # c0_t_s = vars_nsteps[nstep][3]
    # b0_s.append(b0_t_s)
    # c0_s.append(c0_t_s) 
    # prod_i_t_s = vars_nsteps[nstep][1][:,:t+1, fct_aux.INDEX_ATTRS["prod_i"]]
    # cons_i_t_s = vars_nsteps[nstep][1][:,:t+1, fct_aux.INDEX_ATTRS["cons_i"]]
    # B_i_t = prod_i_t_s * b0_s - cons_i_t_s * c0_s
    # print("shapes: b0_t_s={}, c0_t_s={}, prod_i_t_s={}".format( len(b0_s),
    #         len(c0_s),  prod_i_t_s.shape))
    
    nstep_max = 0
    B_i_t_max = np.array([-np.inf]*m_players)
    for nstep in range(0, len(vars_nsteps)):
        b0_t = vars_nsteps[nstep][2]
        c0_t = vars_nsteps[nstep][3]
        # b0_s.append(b0_t_s)
        # c0_s.append(c0_t_s) 
        # prod_i_t_s = vars_nsteps[nstep][1][:,:t+1, fct_aux.INDEX_ATTRS["prod_i"]]
        # cons_i_t_s = vars_nsteps[nstep][1][:,:t+1, fct_aux.INDEX_ATTRS["cons_i"]]
        # B_i_t_nstep = prod_i_t_s * b0_s - cons_i_t_s * c0_s
        
        prod_i_t = vars_nsteps[nstep][1][:, t, fct_aux.INDEX_ATTRS["prod_i"]]
        cons_i_t = vars_nsteps[nstep][1][:, t, fct_aux.INDEX_ATTRS["cons_i"]]
        B_i_t_nstep = prod_i_t * b0_t - cons_i_t * c0_t
        
        if round(np.sum(np.greater_equal(B_i_t_nstep,B_i_t_max))/m_players,2)>0.80:
            nstep_max = nstep
            B_i_t_max = B_i_t_nstep
        
    return nstep_max, B_i_t_max, vars_nsteps[nstep_max]

def balanced_player_game_t(arr_pl_M_T_old, arr_pl_M_T, t, 
                           pi_hp_plus, pi_hp_minus, 
                           pi_sg_plus_t, pi_sg_minus_t,
                           probs_mode,
                           m_players, num_periods, 
                           dbg):
    # pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t_minus_1
    # pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t_minus_1
    
    cpt_error_gamma = 0; cpt_balanced = 0;
    dico_state_mode_i = {}; dico_balanced_pl_i = {}
    for num_pl_i in range(0, m_players):
        Ci = round(arr_pl_M_T[num_pl_i, t, fct_aux.INDEX_ATTRS["Ci"]],2)
        Pi = round(arr_pl_M_T[num_pl_i, t, fct_aux.INDEX_ATTRS["Pi"]],2)
        Si = round(arr_pl_M_T[num_pl_i, t, fct_aux.INDEX_ATTRS["Si"]],2)
        Si_max = round(arr_pl_M_T[num_pl_i, t, 
                                  fct_aux.INDEX_ATTRS["Si_max"]],2)
        gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                            prod_i, cons_i, r_i, state_i)
        
        # get mode_i, state_i and update R_i_old
        pl_i.set_R_i_old(Si_max-Si)
        state_i = pl_i.find_out_state_i()
        p_i = None
        if state_i == fct_aux.STATES[0]:            # state1 
            p_i = probs_mode[0]
        elif state_i == fct_aux.STATES[1]:          # state_2
            p_i = probs_mode[1]
        elif state_i == fct_aux.STATES[2]:          # state_3
            p_i = probs_mode[2]
        else: 
            p_i = None#0.5
        pl_i.select_mode_i(p_i=p_i)
        
        print("mode_i={}".format(pl_i.get_mode_i())) if state_i == "state3" else None
        
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
    
    dico_stats_res_t = (round(cpt_balanced/m_players,2),
                         round(cpt_error_gamma/m_players,2), 
                         dico_state_mode_i)
    dico_stats_res_t = {"balanced": dico_balanced_pl_i, 
                         "gamma_i": dico_state_mode_i}    
        
    # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
    # from a pricing model in the document
    pi_sg_plus_t_new, pi_sg_minus_t_new = \
        detGameModel.determine_new_pricing_sg(
            arr_pl_M_T, 
            pi_hp_plus, 
            pi_hp_minus, 
            t, 
            dbg=dbg)
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
    
    
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_M_T, t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t, c0_t = fct_aux.compute_energy_unit_price(
                    pi_0_plus_t, pi_0_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg) 
    
    # compute ben, cst of shape (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T[:, t, fct_aux.INDEX_ATTRS["gamma_i"]]
    bens, csts = fct_aux.compute_utility_players(arr_pl_M_T, 
                                              gamma_is, 
                                              t, 
                                              b0_t, 
                                              c0_t)
    print('#### bens={}'.format(bens.shape)) if dbg else None
    
    return arr_pl_M_T_old, arr_pl_M_T, \
            b0_t, c0_t, bens, csts, \
            pi_sg_plus_t, pi_sg_minus_t, pi_0_plus_t, pi_0_minus_t, \
            dico_stats_res_t


def lri_balanced_player_game_old(pi_hp_plus = 0.10, pi_hp_minus = 0.15,
                            m_players=3, num_periods=5, 
                            Ci_low=10, Ci_high=30,
                            prob_Ci=0.3, probs_mode=[0.5, 0.5, 0.5],
                            scenario="scenario1", n_steps = 10,
                            path_to_save="tests", dbg=False):
    """
    create a game using LRI learning method for balancing all players 
    at all periods of time NUM_PERIODS = [1..T]

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
    probs_mode: list of float, optional
        DESCRIPTION. The default is [0.5,0.5, 0.5].
        probability for choosing for each state, one mode. 
        exple: if state1, they are 50% to select CONS+ mode and 50% to CONS- mode
    scenario : String, optional
        DESCRIPTION. The default is "scenario1".
        a plan for operating a game of players
    n_steps : integer, optional
        DESCRIPTION. The default is 10.
        number of steps for learning 
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
    print("{}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, probs_mode={} ---> debut \n".format(
            scenario, prob_Ci, pi_hp_plus, pi_hp_minus, probs_mode))
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus, pi_sg_minus = [], []
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_0_plus, pi_0_minus = [], []
    B_is, C_is = [], []
    b0_s, c0_s = [], []
    BENs, CSTs = np.array([]), np.array([])
    dico_stats_res = dict()
    # _______ variables' initialization --> fin   ________________
    
    # _______   generation initial variables for all players at any time   ____
    arr_pl_M_T = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                                    m_players=m_players, 
                                    num_periods=num_periods, 
                                    scenario=scenario, prob_Ci=prob_Ci, 
                                    Ci_low=Ci_low, Ci_high=Ci_high)
    
    # _________     run balanced sg for all num_periods     __________________
    arr_pl_M_T_old = arr_pl_M_T.copy()
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    for t in range(0, num_periods):
        print("******* t = {} *******".format(t)) if dbg else None
        
        pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t_minus_1
        pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t_minus_1
        
        arr_pl_M_T_old, arr_pl_M_T, \
        b0_t, c0_t, bens, csts, \
        pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1, \
        pi_0_plus_t, pi_0_minus_t, \
        dico_stats_res_t = \
            balanced_player_game_t(
            arr_pl_M_T_old, arr_pl_M_T, t, 
            pi_hp_plus, pi_hp_minus, 
            pi_sg_plus_t = pi_sg_plus_t, 
            pi_sg_minus_t = pi_sg_minus_t,
            probs_mode = probs_mode,
            m_players = m_players, num_periods = num_periods, 
            dbg = dbg)
            
        # update pi_sg_plus, pi_sg_minus of shape (NUM_PERIODS,)
        pi_sg_plus.append(pi_sg_plus_t_minus_1)
        pi_sg_minus.append(pi_sg_minus_t_minus_1)
        pi_0_plus.append(pi_0_plus_t)
        pi_0_minus.append(pi_0_minus_t)
        
        # update b0_s, c0_s of shape (NUM_PERIODS,) 
        b0_s.append(b0_t)
        c0_s.append(c0_t) 
        
        # update BENs, CSTs of shape (NUM_PERIODS*M_PLAYERS,)
        BENs = np.append(BENs, bens)
        CSTs = np.append(CSTs, csts)
        
        # update dico_stats_res
        dico_stats_res[t] = dico_stats_res_t
    
    #______________     turn list in numpy array    __________________________ 
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
    
    #__________      save computed variables locally      _____________________
    
    save_variables(path_to_save, arr_pl_M_T_old, arr_pl_M_T, 
                   b0_s, c0_s, B_is, C_is, BENs, CSTs, BB_is, CC_is, 
                   RU_is, pi_sg_minus, pi_sg_plus, pi_0_minus, pi_0_plus,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res)
    
    print("{}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, probs_mode={} ---> fin \n".format(
            scenario, prob_Ci, pi_hp_plus, pi_hp_minus, probs_mode))
    
    return 

def lri_balanced_player_game(pi_hp_plus = 0.10, pi_hp_minus = 0.15,
                            m_players=3, num_periods=5, 
                            Ci_low=10, Ci_high=30,
                            prob_Ci=0.3, probs_mode=[0.5, 0.5, 0.5],
                            scenario="scenario1", n_steps = 10,
                            path_to_save="tests", dbg=False):
    """
    create a game using LRI learning method for balancing all players 
    at all periods of time NUM_PERIODS = [1..T]

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
    probs_mode: list of float, optional
        DESCRIPTION. The default is [0.5,0.5, 0.5].
        probability for choosing for each state, one mode. 
        exple: if state1, they are 50% to select CONS+ mode and 50% to CONS- mode
    scenario : String, optional
        DESCRIPTION. The default is "scenario1".
        a plan for operating a game of players
    n_steps : integer, optional
        DESCRIPTION. The default is 10.
        number of steps for learning 
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
    print("{}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, probs_mode={} ---> debut \n".format(
            scenario, prob_Ci, pi_hp_plus, pi_hp_minus, probs_mode))
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus, pi_sg_minus = [], []
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_0_plus, pi_0_minus = [], []
    B_is, C_is = [], []
    b0_s, c0_s = [], []
    BENs, CSTs = np.array([]), np.array([])
    dico_stats_res = dict()
    # _______ variables' initialization --> fin   ________________
    
    # _______   generation initial variables for all players at any time   ____
    arr_pl_M_T = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                                    m_players=m_players, 
                                    num_periods=num_periods, 
                                    scenario=scenario, prob_Ci=prob_Ci, 
                                    Ci_low=Ci_low, Ci_high=Ci_high)
    
    # _________     run balanced sg for all num_periods     __________________
    arr_pl_M_T_old = arr_pl_M_T.copy()
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    for t in range(0, num_periods):
        print("******* t = {} *******".format(t)) if dbg else None
        arr_pl_M_T_old_t = arr_pl_M_T_old.copy()
        arr_pl_M_T_t = arr_pl_M_T.copy()
        vars_nsteps = []
        for nstep in range(0, n_steps):
            pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t_minus_1
            pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t_minus_1
            
            arr_pl_M_T_old_t_nstep, arr_pl_M_T_t_nstep = None, None
            
            arr_pl_M_T_old_t_nstep, arr_pl_M_T_t_nstep, \
            b0_t_nstep, c0_t_nstep, bens_nstep, csts_nstep, \
            pi_sg_plus_t_minus_1_nstep, pi_sg_minus_t_minus_1_nstep, \
            pi_0_plus_t_nstep, pi_0_minus_t_nstep, \
            dico_stats_res_t_nstep = \
                balanced_player_game_t(
                arr_pl_M_T_old_t, arr_pl_M_T_t, t, 
                pi_hp_plus, pi_hp_minus, 
                pi_sg_plus_t = pi_sg_plus_t, 
                pi_sg_minus_t = pi_sg_minus_t,
                probs_mode = probs_mode,
                m_players = m_players, num_periods = num_periods, 
                dbg = dbg)
                
            vars_nstep = [arr_pl_M_T_old_t_nstep, arr_pl_M_T_t_nstep, \
                        b0_t_nstep, c0_t_nstep, bens_nstep, csts_nstep, \
                        pi_sg_plus_t_minus_1_nstep, pi_sg_minus_t_minus_1_nstep, \
                        pi_0_plus_t_nstep, pi_0_minus_t_nstep, \
                        dico_stats_res_t_nstep]
            vars_nsteps.append(vars_nstep)
            
            # print log
            print("nstep={}, b0_t={}, c0_t={}".format(nstep, b0_t_nstep, c0_t_nstep))
            print("---> pi_sg_plus_t_minus_1={}, pi_sg_minus_t_minus_1={},pi_0_plus_t={}, pi_0_minus_t={}".format(
            pi_sg_plus_t_minus_1_nstep, pi_sg_minus_t_minus_1_nstep, \
            pi_0_plus_t_nstep, pi_0_minus_t_nstep))
            for num_pl in range(0, arr_pl_M_T_t_nstep.shape[0]):
                print("----->pl_{}, state_i={}, mode_i={}, prod_i={}, cons_i={}".format(
                    num_pl, 
                    arr_pl_M_T_t_nstep[num_pl, t, fct_aux.INDEX_ATTRS["state_i"]], 
                    arr_pl_M_T_t_nstep[num_pl, t, fct_aux.INDEX_ATTRS["mode_i"]],
                    round(arr_pl_M_T_t_nstep[num_pl, t, fct_aux.INDEX_ATTRS["prod_i"]],2),
                    round(arr_pl_M_T_t_nstep[num_pl, t, fct_aux.INDEX_ATTRS["cons_i"]],2),
                        ))
            
        nstep, B_i_t, vars_opt = find_optimal_vars_on_nsteps(vars_nsteps, 
                                                             b0_s.copy(), 
                                                             c0_s.copy(),
                                                             t, 
                                                             m_players, 
                                                             thres = 0.80
                                                             )
        arr_pl_M_T_old_t = vars_opt[0]
        arr_pl_M_T_t = vars_opt[1]
        b0_t, c0_t = vars_opt[2], vars_opt[3] 
        bens, csts = vars_opt[4], vars_opt[5]
        pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = vars_opt[6], vars_opt[7]
        pi_0_plus_t, pi_0_minus_t = vars_opt[8], vars_opt[9]
        dico_stats_res_t = vars_opt[10]
        
        print("t={}, pi_sg_plus_t_new={}, pi_sg_minus_t_new={}, nstep={}, B_i_t={} \n".format(
        t, pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1, nstep, B_i_t))
        
        # update pi_sg_plus, pi_sg_minus of shape (NUM_PERIODS,)
        pi_sg_plus.append(pi_sg_plus_t_minus_1)
        pi_sg_minus.append(pi_sg_minus_t_minus_1)
        pi_0_plus.append(pi_0_plus_t)
        pi_0_minus.append(pi_0_minus_t)
        
        # update b0_s, c0_s of shape (NUM_PERIODS,) 
        b0_s.append(b0_t)
        c0_s.append(c0_t) 
        
        # update BENs, CSTs of shape (NUM_PERIODS*M_PLAYERS,)
        BENs = np.append(BENs, bens)
        CSTs = np.append(CSTs, csts)
        
        # update dico_stats_res
        dico_stats_res[t] = dico_stats_res_t
    
    #______________     turn list in numpy array    __________________________ 
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
    
    #__________      save computed variables locally      _____________________
    
    save_variables(path_to_save, arr_pl_M_T_old, arr_pl_M_T, 
                   b0_s, c0_s, B_is, C_is, BENs, CSTs, BB_is, CC_is, 
                   RU_is, pi_sg_minus, pi_sg_plus, pi_0_minus, pi_0_plus,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res)
    
    print("{}, probCi={}, pi_hp_plus={}, pi_hp_minus ={}, probs_mode={} ---> fin \n".format(
            scenario, prob_Ci, pi_hp_plus, pi_hp_minus, probs_mode))
    
    return 

###############################################################################
#                   definition  des unittests
#
###############################################################################
def test_lri_balanced_player_game():
    pi_hp_plus = 0.10; pi_hp_minus = 0.15
    pi_hp_plus = 10; pi_hp_minus = 15
    m_players = 3; num_periods = 5
    Ci_low = 10; Ci_high = 30
    prob_Ci = 0.3; probs_mode = [0.5, 0.5, 0.5]
    n_steps = 5
    scenario = "scenario1"; path_to_save = "tests"
    
    lri_balanced_player_game(pi_hp_plus=pi_hp_plus, 
                             pi_hp_minus=pi_hp_minus,
                             m_players=m_players, 
                             num_periods=num_periods, 
                             Ci_low=Ci_low, 
                             Ci_high=Ci_high,
                             prob_Ci=prob_Ci, 
                             probs_mode=probs_mode,
                             scenario=scenario, n_steps=n_steps, 
                             path_to_save=path_to_save, dbg=False)
    


###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    test_lri_balanced_player_game()
    
    print("runtime = {}".format(time.time() - ti))