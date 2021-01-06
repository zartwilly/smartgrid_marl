# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:55:14 2020

@author: jwehounou
"""
import time

import numpy as np
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux


#------------------------------------------------------------------------------
#                       definition of functions 
#
#------------------------------------------------------------------------------
def balanced_player_game_t(arr_pl_M_T_vars, t, 
                           pi_hp_plus, pi_hp_minus,
                           pi_sg_plus_t, pi_sg_minus_t, 
                           m_players, num_periods, 
                           random_determinist, used_storage,
                           dico_stats_res, dbg):
    """
    balance the player game at time t

    Parameters
    ----------
    arr_pl_M_T_vars : array of shape (M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))
        DESCRIPTION.
    t : integer
        DESCRIPTION.
        instant of time
    pi_hp_plus : float
        DESCRIPTION.
        the price of exported (sold) energy from SG to HP
    pi_hp_minus : float
        DESCRIPTION.
        the price of imported (purchased) energy from HP to SG
    pi_sg_plus_t : float
        DESCRIPTION.
        the price of exported (sold) energy from player to SG
    pi_sg_minus_t : float
        DESCRIPTION.
        the price of imported (purchased) energy from SG to player
    m_players : integer
        DESCRIPTION.
        number of players
    num_periods : integer
        DESCRIPTION.
        number of time instants 
    random_determinist : boolean
        DESCRIPTION.
        decide if the mode of player a_i is randomly chosen (True) or 
        deterministly chosen (False) 
    dico_stats_res : TYPE
        DESCRIPTION.
    dbg : boolean
        DESCRIPTION.
        debug
        
    Returns
    -------
    arr_pl_M_T_vars : array of shape (M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))
        DESCRIPTION.
    b0_t : array of shape(NUM_PERIODS,) 
        DESCRIPTION.
        array of unit price of benefit for all periods
    c0_t : array of shape(NUM_PERIODS,) 
        DESCRIPTION.
        array of unit price of cost for all periods
    bens_t : array of shape (M_PLAYERS, )
        DESCRIPTION.
        array of benefits of all players at the instant time t
    csts_t : array of shape (M_PLAYERS, )
        DESCRIPTION.
        array of benefits of a player at the instant time t
    pi_sg_plus_t : float
        DESCRIPTION.
        exported (sold) unit price from player to SG at the instant time t 
    pi_sg_minus_t : float
        DESCRIPTION.
        imported (purchased) unit price from player to SG at the instant time t 
    pi_0_plus_t : float
        DESCRIPTION.
    pi_0_minus_t : float
        DESCRIPTION.
    dico_stats_res : TYPE
        DESCRIPTION.

    """
    
    cpt_error_gamma = 0; cpt_balanced = 0; 
    dico_balanced_pl_i = {}; dico_state_mode_i = {}
    for num_pl_i in range(0, m_players):
        Ci = round(arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Ci"]],
                   fct_aux.N_DECIMALS)
        Pi = round(arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Pi"]],
                   fct_aux.N_DECIMALS)
        Si = round(arr_pl_M_T_vars[num_pl_i, t, fct_aux.INDEX_ATTRS["Si"]], 
                   fct_aux.N_DECIMALS)
        Si_max = round(arr_pl_M_T_vars[num_pl_i, t, 
                                  fct_aux.INDEX_ATTRS["Si_max"]],
                       fct_aux.N_DECIMALS)
        gamma_i, prod_i, cons_i, r_i, state_i = 0, 0, 0, 0, ""
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                            prod_i, cons_i, r_i, state_i)
        
        # get state_i and update R_i_old
        pl_i.set_R_i_old(Si_max-Si)
        state_i = arr_pl_M_T_vars[
                        num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["state_i"]]
        pl_i.set_state_i(state_i)
        
        # get mode_i
        if t == 0 or random_determinist:
            pl_i.select_mode_i()
        else:
            # t in [1,num_periods]
            Pi_t_plus_1 = arr_pl_M_T_vars[num_pl_i, 
                                          t+1, 
                                          fct_aux.INDEX_ATTRS["Pi"]] \
                            if t+1 < num_periods \
                            else 0
            Ci_t_plus_1 = arr_pl_M_T_vars[num_pl_i, 
                                     t+1, 
                                     fct_aux.INDEX_ATTRS["Ci"]] \
                            if t+1 < num_periods \
                            else 0
            Si_t_minus_1_minus = arr_pl_M_T_vars[num_pl_i, 
                                          t-1, 
                                          fct_aux.INDEX_ATTRS["Si_minus"]] \
                            if t-1 > 0 \
                            else 0
            Si_t_minus_1_plus = arr_pl_M_T_vars[num_pl_i, 
                                     t-1, 
                                     fct_aux.INDEX_ATTRS["Si_plus"]] \
                            if t-1 > 0 \
                            else 0
            mode_i = None
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
    
        # balancing
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        cpt_balanced += round(1/m_players, fct_aux.N_DECIMALS) if boolean else 0
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
        Pi_t_plus_1 = arr_pl_M_T_vars[num_pl_i, t+1, fct_aux.INDEX_ATTRS["Pi"]] \
                        if t+1 < num_periods \
                        else 0
        Ci_t_plus_1 = arr_pl_M_T_vars[num_pl_i, 
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
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["prod_i"]] = pl_i.get_prod_i()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["cons_i"]] = pl_i.get_cons_i()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["gamma_i"]] = pl_i.get_gamma_i()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["r_i"]] = pl_i.get_r_i()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["Si"]] = pl_i.get_Si()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["Si_minus"]] = pl_i.get_Si_minus()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["Si_plus"]] = pl_i.get_Si_plus()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["Si_old"]] = pl_i.get_Si_old()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["mode_i"]] = pl_i.get_mode_i()
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["balanced_pl_i"]] = boolean
        arr_pl_M_T_vars[num_pl_i, 
                        t, 
                        fct_aux.INDEX_ATTRS["formule"]] = formule
    
    dico_stats_res[t] = (round(cpt_balanced/m_players, fct_aux.N_DECIMALS),
                         round(cpt_error_gamma/m_players, fct_aux.N_DECIMALS), 
                         dico_state_mode_i)
    dico_stats_res[t] = {"balanced": dico_balanced_pl_i, 
                         "gamma_i": dico_state_mode_i}    
        
    # compute the new prices pi_sg_plus_t+1, pi_sg_minus_t+1 
    # from a pricing model in the document
    pi_sg_plus_t_new, pi_sg_minus_t_new = fct_aux.determine_new_pricing_sg(
                                            arr_pl_M_T_vars, 
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
    
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = fct_aux.compute_prod_cons_SG(arr_pl_M_T_vars, t)
    # compute prices of an energy unit price for cost and benefit players
    b0_t, c0_t = fct_aux.compute_energy_unit_price(
                    pi_0_plus_t, pi_0_minus_t, 
                    pi_hp_plus, pi_hp_minus,
                    In_sg, Out_sg)
    
    # compute ben, cst of shape (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_pl_M_T_vars[:, t, fct_aux.INDEX_ATTRS["gamma_i"]]
    bens_t, csts_t = fct_aux.compute_utility_players(arr_pl_M_T_vars, 
                                              gamma_is, 
                                              t, 
                                              b0_t, 
                                              c0_t)
    # print('#### bens={}'.format(bens_t.shape)) if dbg else None
    
    
    return arr_pl_M_T_vars, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            pi_0_plus_t, pi_0_minus_t, \
            dico_stats_res

    

def determinist_balanced_player_game(arr_pl_M_T, 
                                     pi_hp_plus = 0.10, 
                                     pi_hp_minus = 0.15,
                                     m_players=3, 
                                     num_periods=5,
                                     prob_Ci=0.3, 
                                     scenario="scenario1", 
                                     random_determinist=False,
                                     used_storage=True,
                                     path_to_save="tests", 
                                     dbg=False):
    """
    create a game for balancing all players at all periods of time NUM_PERIODS = [1..T]

    Parameters
    ----------
    arr_pl_M_T: array of shape (M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))
        DESCRIPTION.
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
    
    print("determinist game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> debut \n"\
          .format(scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_sg_plus = np.empty(shape=(num_periods,)) #      shape (NUM_PERIODS,)
    pi_sg_plus.fill(np.nan)
    pi_sg_minus = np.empty(shape=(num_periods,)) #      shape (NUM_PERIODS,)
    pi_sg_plus.fill(np.nan)
    pi_0_plus_t, pi_0_minus_t = 0, 0
    pi_0_plus = np.empty(shape=(num_periods,)) #     shape (NUM_PERIODS,)
    pi_0_plus.fill(np.nan)
    pi_0_minus = np.empty(shape=(num_periods,)) #     shape (NUM_PERIODS,)
    pi_0_minus.fill(np.nan)
    B_is = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    B_is.fill(np.nan)
    C_is = np.empty(shape=(m_players,)) #   shape (M_PLAYERS, )
    C_is.fill(np.nan)
    b0_ts = np.empty(shape=(num_periods,)) #   shape (NUM_PERIODS,)
    b0_ts.fill(np.nan)
    c0_ts = np.empty(shape=(num_periods,))
    c0_ts.fill(np.nan)
    BENs = np.empty(shape=(m_players, num_periods)) #   shape (M_PLAYERS, NUM_PERIODS)
    CSTs = np.empty(shape=(m_players, num_periods))
    
    fct_aux.INDEX_ATTRS["Si_minus"] = 16
    fct_aux.INDEX_ATTRS["Si_plus"] = 17
    # fct_aux.INDEX_ATTRS[""] = 19
    # _______ variables' initialization --> fin   ________________
        
    
    # ____      add initial values for the new attributs     _______
    nb_vars_2_add = 2
    arr_pl_M_T_vars = np.zeros((arr_pl_M_T.shape[0],
                                arr_pl_M_T.shape[1],
                                arr_pl_M_T.shape[2]+nb_vars_2_add), 
                               dtype=object)
    arr_pl_M_T_vars[:,:,:-nb_vars_2_add] = arr_pl_M_T
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["Si_plus"]] = np.nan
    
    # ____      game beginning for all t_period ---> debut      _____
    dico_stats_res={}
    
    import force_brute_game_model as bfGameModel
    arr_pl_M_T_vars, possibles_modes = bfGameModel.reupdate_state_players(
                                        arr_pl_M_T_vars.copy(), 0, 0)
    
    for t in range(0, num_periods):
        print("******* t = {} *******".format(t)) if dbg else None
        print("___t = {}, pi_sg_plus_t={}, pi_sg_minus_t={}".format(
                t, pi_sg_plus_t, pi_sg_minus_t)) \
            if t%20 == 0 \
            else None
            
        # compute pi_0_plus, pi_0_minus, pi_sg_plus, pi_sg_minus
        pi_sg_plus_t = pi_hp_plus-1 if t == 0 else pi_sg_plus_t
        pi_sg_minus_t = pi_hp_minus-1 if t == 0 else pi_sg_minus_t
        
        cpt_error_gamma = 0; cpt_balanced = 0;
        dico_state_mode_i = {}; dico_balanced_pl_i = {}
        
        # balanced player game at instant t
        arr_pl_M_T_vars, \
        b0_t, c0_t, \
        bens_t, csts_t, \
        pi_sg_plus_t, pi_sg_minus_t, \
        pi_0_plus_t, pi_0_minus_t, \
        dico_stats_res \
            = balanced_player_game_t(arr_pl_M_T_vars.copy(), t, 
                                       pi_hp_plus, pi_hp_minus,
                                       pi_sg_plus_t, pi_sg_minus_t, 
                                       m_players, num_periods, 
                                       random_determinist, used_storage,
                                       dico_stats_res, dbg)
        
        dico_stats_res[t] = (round(cpt_balanced/m_players,2),
                         round(cpt_error_gamma/m_players,2), 
                         dico_state_mode_i)
        dico_stats_res[t] = {"balanced": dico_balanced_pl_i, 
                             "gamma_i": dico_state_mode_i}    
        
        # b0_ts, c0_ts of shape (NUM_PERIODS,)
        pi_sg_plus[t] = pi_sg_plus_t
        pi_sg_minus[t] = pi_sg_minus_t
        
        # pi_0_plus, pi_0_minus of shape (NUM_PERIODS,)
        pi_0_plus[t] = pi_0_plus_t
        pi_0_minus[t] = pi_0_minus_t
        
        # b0_ts, c0_ts of shape (NUM_PERIODS,)
        b0_ts[t] = b0_t
        c0_ts[t] = c0_t 
        
        # BENs, CSTs of shape (NUM_PERIODS,M_PLAYERS)
        BENs[:,t] = bens_t
        CSTs[:,t] = csts_t
        
    # ____      game beginning for all t_period ---> debut      _____    
        
    # B_is, C_is of shape (M_PLAYERS, )
    prod_i_T = arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prod_i"]]
    cons_i_T = arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["cons_i"]]
    B_is = np.sum(b0_ts * prod_i_T, axis=1)
    C_is = np.sum(c0_ts * cons_i_T, axis=1)
    
    # BB_is, CC_is, RU_is of shape (M_PLAYERS, )
    CONS_is = np.sum(arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["cons_i"]], axis=1)
    PROD_is = np.sum(arr_pl_M_T_vars[:,:, fct_aux.INDEX_ATTRS["prod_i"]], axis=1)
    BB_is = pi_sg_plus[-1] * PROD_is #np.sum(PROD_is)
    for num_pl, bb_i in enumerate(BB_is):
        if bb_i != 0:
            print("player {}, BB_i={}".format(num_pl, bb_i))
    CC_is = pi_sg_minus[-1] * CONS_is #np.sum(CONS_is)
    RU_is = BB_is - CC_is
    
    pi_hp_plus_s = np.array([pi_hp_plus] * num_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * num_periods, dtype=object)
    
    # save computed variables
    algo_name = ""
    if random_determinist:
        algo_name = "RD-DETERMINIST"
    else:
        algo_name = "DETERMINIST"
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars, 
                   b0_ts, c0_ts, B_is, C_is, 
                   BENs, CSTs, 
                   BB_is, CC_is, RU_is, 
                   pi_sg_minus, pi_sg_plus, 
                   pi_0_minus, pi_0_plus,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo=algo_name)
    
    print("determinist game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> end \n"\
          .format(scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
        
    return arr_pl_M_T_vars
        
#------------------------------------------------------------------------------
#                       definition of unittests 
#
#------------------------------------------------------------------------------
def test_DETERMINIST_balanced_player_game():
    pi_hp_plus = 0.10; pi_hp_minus = 0.15
    pi_hp_plus = 10; pi_hp_minus = 15
    m_players = 3; num_periods = 5;
    Ci_low = fct_aux.Ci_LOW; Ci_high = fct_aux.Ci_HIGH
    prob_Ci = 0.3;
    scenario = "scenario1"; 
    random_determinist = False ; 
    used_storage = True #False #True;
    path_to_save = "tests"
    
    fct_aux.N_DECIMALS = 3
    
    # ____   generation initial variables for all players at any time   ____
    arr_pl_M_T = fct_aux.generate_Pi_Ci_Si_Simax_by_profil_scenario(
                                    m_players=m_players, 
                                    num_periods=num_periods, 
                                    scenario=scenario, prob_Ci=prob_Ci, 
                                    Ci_low=Ci_low, Ci_high=Ci_high)
    
    arr_M_T_vars = \
    determinist_balanced_player_game(
                             arr_pl_M_T,
                             pi_hp_plus=pi_hp_plus, 
                             pi_hp_minus=pi_hp_minus,
                             m_players=m_players, 
                             num_periods=num_periods,
                             prob_Ci=prob_Ci,
                             scenario=scenario,
                             random_determinist=random_determinist,
                             used_storage=used_storage,
                             path_to_save=path_to_save, dbg=False)
    
    return arr_M_T_vars

#------------------------------------------------------------------------------
#                       execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    arr_M_T_vars = test_DETERMINIST_balanced_player_game()
    print("runtime = {}".format(time.time() - ti))