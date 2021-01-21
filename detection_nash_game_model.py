# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:33:20 2021

@author: jwehounou
"""
import os
import time

import numpy as np
import itertools as it

import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

import force_brute_game_model as bf_model

from datetime import datetime

#------------------------------------------------------------------------------
#                       definition of CONSTANCES --> debut
#
#------------------------------------------------------------------------------
RACINE_PLAYER = "player"
#------------------------------------------------------------------------------
#                       definition of CONSTANCES --> fin
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                       definition of functions --> debut
#
#------------------------------------------------------------------------------
def detect_nash_balancing_profil(dico_profs_Vis_Perf_t, arr_pl_M_T_vars, t):
    """
    detect the profil driving to nash equilibrium
    
    dico_profs_Vis_Perf_t[tuple_prof] = dico_Vis_Pref_t with
        * tuple_prof = (S1, ...., Sm), Si is the strategie of player i
        * dico_Vis_Pref_t has keys "Pref_t" and RACINE_PLAYER+"_"+i
            * the value of "Pref_t" is \sum\limits_{1\leq i \leq N}ben_i-cst_i
            * the value of RACINE_PLAYER+"_"+i is Vi = ben_i - cst_i
            * NB : 0 <= i < m_players or  i \in [0, m_player[
    """
    
    nash_profils = list()
    
    for key_modes_prof, dico_Vi_Pref_t in dico_profs_Vis_Perf_t.items():
        for num_pl_i, mode_i in enumerate(key_modes_prof):                      # 0 <= num_pl_i < m_player            
            Vi = dico_Vi_Pref_t[RACINE_PLAYER+"_"+str(num_pl_i)]
            state_i = arr_pl_M_T_vars[num_pl_i, t, 
                                      fct_aux.INDEX_ATTRS["state_i"]]
            mode_i_bar = None
            if state_i == fct_aux.STATES[0] \
                and mode_i == fct_aux.STATE1_STRATS[0]:                         # state1, CONS+
                mode_i_bar = fct_aux.STATE1_STRATS[1]                           # CONS-
            elif state_i == fct_aux.STATES[0] \
                and mode_i == fct_aux.STATE1_STRATS[1]:                         # state1, CONS-
                mode_i_bar = fct_aux.STATE1_STRATS[0]                           # CONS+
            elif state_i == fct_aux.STATES[1] \
                and mode_i == fct_aux.STATE2_STRATS[0]:                         # state2, DIS
                mode_i_bar = fct_aux.STATE2_STRATS[1]                           # CONS-
            elif state_i == fct_aux.STATES[1] \
                and mode_i == fct_aux.STATE2_STRATS[1]:                         # state2, CONS-
                mode_i_bar = fct_aux.STATE2_STRATS[0]                           # DIS
            elif state_i == fct_aux.STATES[2] \
                and mode_i == fct_aux.STATE3_STRATS[0]:                         # state3, DIS
                mode_i_bar = fct_aux.STATE3_STRATS[1]                           # PROD
            elif state_i == fct_aux.STATES[2] \
                and mode_i == fct_aux.STATE3_STRATS[1]:                         # state3, PROD
                mode_i_bar = fct_aux.STATE3_STRATS[0]                           # DIS
            
            new_key_modes_prof = list(key_modes_prof)
            new_key_modes_prof[num_pl_i] = mode_i_bar
            new_key_modes_prof = tuple(new_key_modes_prof)
            
            Vi_bar = None
            Vi_bar = dico_profs_Vis_Perf_t[new_key_modes_prof]\
                                          [RACINE_PLAYER+"_"+str(num_pl_i)]
            if Vi >= Vi_bar:
                nash_profils.append(key_modes_prof)
                
    return nash_profils

def delete_duplicate_profils(nash_profils):
    """
    delete all occurences of the profiles 
    """
    return nash_profils
#------------------------------------------------------------------------------
#                       definition of functions --> fin
#
#------------------------------------------------------------------------------

# ______      main function of detection nash balancing   ---> debut      _____
def nash_balanced_player_game_perf_t(arr_pl_M_T,
                                     pi_hp_plus=0.10, 
                                     pi_hp_minus=0.15,
                                     m_players=3, 
                                     t_periods=4,
                                     prob_Ci=0.3, 
                                     scenario="scenario1",
                                     algo_name="BEST-NASH",
                                     path_to_save="tests", 
                                     manual_debug=False, dbg=False):
    """
    detect the nash balancing profils by the utility of game Perf_t and select
    the best, bad and middle profils by comparing the value of Perf_t

    Returns
    -------
    None.

    """
    print("\n \n {} game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> BEGIN \n"\
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
    
    arr_pl_M_T_vars, possibles_modes = fct_aux.reupdate_state_players(
                                        arr_pl_M_T_vars.copy(), 0, 0)
    
    print("m_players={}, possibles_modes={}".format(m_players, 
                                                   len(possibles_modes)))
    
    dico_stats_res = dict()
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
            
        
        mode_profiles = it.product(*possibles_modes)
        
        # Perf_t by players' profil modes
        dico_profs_Vis_Perf_t = dict()
        cpt_profs = 0
        for mode_profile in mode_profiles:
            dico_balanced_pl_i_mode_prof, cpt_balanced_mode_prof = dict(), 0
            dico_state_mode_i_mode_prof = dict()
            
            # balanced players
            arr_pl_M_T_vars_mode_prof, \
            dico_balanced_pl_i_mode_prof, \
            dico_state_mode_i_mode_prof, \
            cpt_balanced_mode_prof \
                = bf_model.balanced_player_game_4_mode_profil(
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
                = bf_model.compute_prices_bf(
                    arr_pl_M_T_vars_mode_prof.copy(), t, 
                    pi_sg_plus_t, pi_sg_minus_t,
                    pi_hp_plus, pi_hp_minus, manual_debug, dbg)
            
            bens_csts_t = bens_t - csts_t
            Perf_t = np.sum(bens_csts_t, axis=0)
            dico_Vis_Pref_t = dict()
            for num_pl_i in range(bens_csts_t.shape[0]):            # bens_csts_t.shape[0] = m_players
                dico_Vis_Pref_t[RACINE_PLAYER+"_"+str(num_pl_i)] \
                    = bens_csts_t[num_pl_i]
            dico_Vis_Pref_t["Perf_t"] = Perf_t
            
            dico_profs_Vis_Perf_t[mode_profile] = dico_Vis_Pref_t
            cpt_profs += 1
         
        
        # detection of NASH profils
        nash_profils = list()
        nash_profils = detect_nash_balancing_profil(
                        dico_profs_Vis_Perf_t,
                        arr_pl_M_T_vars, 
                        t)
        nash_profils = delete_duplicate_profils(nash_profils)
        
        # create dico of nash profils with key is Pref_t and value is profil
        dico_Perft_nashProfil = dict()
        for nash_mode_profil in nash_profils:
            Perf_t = dico_profs_Vis_Perf_t[nash_mode_profil]["Perf_t"]
            if Perf_t in dico_Perft_nashProfil:
                dico_Perft_nashProfil[Perf_t].append(nash_mode_profil)
            else:
                dico_Perft_nashProfil[Perf_t] = [nash_mode_profil]
                    
        # min, max, mean of Perf_t
        best_key_Perf_t = None
        if algo_name == fct_aux.ALGO_NAMES_NASH[0]:              # BEST-NASH
            best_key_Perf_t = max(dico_Perft_nashProfil.keys())
        elif algo_name == fct_aux.ALGO_NAMES_NASH[1]:            # BAD-NASH
            best_key_Perf_t = min(dico_Perft_nashProfil.keys())
        elif algo_name == fct_aux.ALGO_NAMES_NASH[2]:            # MIDDLE-NASH
            mean_key_Perf_t  = np.mean(list(dico_Perft_nashProfil.keys()))
            if mean_key_Perf_t in dico_Perft_nashProfil.keys():
                best_key_Perf_t = mean_key_Perf_t
            else:
                sorted_keys = sorted(dico_Perft_nashProfil.keys())
                boolean = True; i_key = 1
                while boolean:
                    if sorted_keys[i_key] <= mean_key_Perf_t:
                        i_key += 1
                    else:
                        boolean = False; i_key -= 1
                best_key_Perf_t = sorted_keys[i_key]
                
        # find the best, bad, middle key in dico_Perft_nashProfil and 
        # the best, bad, middle nash_mode_profile
        best_nash_mode_profiles = dico_Perft_nashProfil[best_key_Perf_t]
        best_nash_mode_profile = None
        if len(best_nash_mode_profiles) == 1:
            best_nash_mode_profile = best_nash_mode_profiles[0]
        else:
            rd = np.random.randint(0, len(best_nash_mode_profiles))
            best_nash_mode_profile = best_nash_mode_profiles[rd]
        
        print("** Running at t={}: numbers of -> cpt_profils={}, best_nash_mode_profiles={}, nash_profils={}"\
              .format(t, cpt_profs, len(best_nash_mode_profiles), 
                      len(nash_profils) ))
        print("best_key_Perf_t={}, {}_nash_mode_profiles={}".format(
                best_key_Perf_t, algo_name, best_nash_mode_profile))
        
        # balanced nash profil and compute prices
        dico_balanced_pl_i_nash_mode_prof = dict() 
        dico_state_mode_i_nash_mode_prof = dict()
        cpt_balanced_nash_mode_prof = 0
        
        arr_pl_M_T_vars_nash_mode_prof, \
        dico_balanced_pl_i_nash_mode_prof, \
        dico_state_mode_i_nash_mode_prof, \
        cpt_balanced_nash_mode_prof, \
            = bf_model.balanced_player_game_4_mode_profil(
                arr_pl_M_T_vars.copy(),
                best_nash_mode_profile, t,
                pi_sg_plus_t, pi_sg_minus_t, 
                pi_hp_plus, pi_hp_minus,
                dico_balanced_pl_i_nash_mode_prof, 
                dico_state_mode_i_nash_mode_prof,
                cpt_balanced_nash_mode_prof,
                m_players, t_periods,
                manual_debug
                )
        # compute Perf_t 
        pi_sg_plus_t, pi_sg_minus_t, \
        pi_0_plus_t, pi_0_minus_t, \
        b0_t, c0_t, \
        bens_t, csts_t \
            = bf_model.compute_prices_bf(
                arr_pl_M_T_vars_nash_mode_prof.copy(), t, 
                pi_sg_plus_t, pi_sg_minus_t,
                pi_hp_plus, pi_hp_minus, manual_debug, dbg)
            
        dico_stats_res[t] = {"balanced": dico_balanced_pl_i_nash_mode_prof, 
                             "gamma_i": dico_state_mode_i_nash_mode_prof,
                             "nash_profils": nash_profils,
                             "best_nash_profil": best_nash_mode_profile,
                             "best_Perf_t": best_key_Perf_t,
                             "nb_nash_profil_byPerf_t": 
                                 dico_Perft_nashProfil[best_key_Perf_t]}
            
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
        
        arr_pl_M_T_vars[:,t,:] = arr_pl_M_T_vars_nash_mode_prof[:,t,:].copy()
    # ____      game beginning for all t_period ---> fin        _____ 
    
    
    # ___       compute global benefits and costs ---> debut        ____
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
    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars.copy(), 
                   b0_ts_T, c0_ts_T, B_is_M, C_is_M, 
                   BENs_M_T, CSTs_M_T, 
                   BB_is_M, CC_is_M, RU_is_M, 
                   pi_sg_minus_T, pi_sg_plus_T, 
                   pi_0_minus_T, pi_0_plus_T,
                   pi_hp_plus_s, pi_hp_minus_s, dico_stats_res, 
                   algo=algo_name)
    # ___       compute global benefits and costs ---> fin          ____
    
    print("{} game: {}, probCi={}, pi_hp_plus={} , pi_hp_minus ={} ---> END \n"\
          .format(algo_name, scenario, prob_Ci, pi_hp_plus, pi_hp_minus))
        
    return arr_pl_M_T_vars
    
# ______      main function of detection nash balancing   ---> fin        _____

#------------------------------------------------------------------------------
#                       definition of unittests --> debut 
#
#------------------------------------------------------------------------------
def test_nash_balanced_player_game_perf_t(algo_name="BEST-NASH"):
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
    path_to_save = os.path.join(name_dir, algo_name+"_"+date_hhmm, 
                                    scenario, str(prob_Ci), 
                                    msg)
    
    manual_debug = True #False #True #False
    arr_pl_M_T_probCi_scen_nashProfil \
        = nash_balanced_player_game_perf_t(
            arr_pl_M_T_probCi_scen.copy(),
            pi_hp_plus, 
            pi_hp_minus,
            m_players, 
            t_periods,
            prob_Ci, 
            scenario,
            algo_name,
            path_to_save, manual_debug, dbg=False
            )
    
    
    return arr_pl_M_T_probCi_scen_nashProfil
    

#------------------------------------------------------------------------------
#                       definition of unittests --> fin
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                       execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    for algo_name in fct_aux.ALGO_NAMES_NASH:
        arr_pl_M_T_vars = test_nash_balanced_player_game_perf_t(algo_name)
    
    print("runtime = {}".format(time.time() - ti))
