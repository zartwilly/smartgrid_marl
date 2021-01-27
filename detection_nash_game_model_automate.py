# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:44:58 2021

@author: jwehounou
"""

import os
import time

import numpy as np
import pandas as pd
import itertools as it

import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

import force_brute_game_model_automate as autoBfGameModel

from datetime import datetime
from pathlib import Path

#------------------------------------------------------------------------------
#                       definition of CONSTANCES
#
#------------------------------------------------------------------------------
RACINE_PLAYER = "player"

#------------------------------------------------------------------------------
#                       definition of functions
#
#------------------------------------------------------------------------------
def find_out_opposite_mode(state_i, mode_i):
    """
    look for the opposite mode of the player.
    for example, 
    if state_i = state1, the possible modes are CONS+ and CONS-
    the opposite mode of CONS+ is CONS- and this of CONS- is CONS+
    """
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

    return mode_i_bar

def detect_nash_balancing_profil(dico_profs_Vis_Perf_t, 
                                 arr_pl_M_T_vars_modif, t):
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
        cpt_players_stables = 0
        for num_pl_i, mode_i in enumerate(key_modes_prof):                      # 0 <= num_pl_i < m_player            
            Vi = dico_Vi_Pref_t[RACINE_PLAYER+"_"+str(num_pl_i)]
            state_i = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                      fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
            mode_i_bar = None
            mode_i_bar = find_out_opposite_mode(state_i, mode_i)
            new_key_modes_prof = list(key_modes_prof)
            new_key_modes_prof[num_pl_i] = mode_i_bar
            new_key_modes_prof = tuple(new_key_modes_prof)
            
            Vi_bar = None
            Vi_bar = dico_profs_Vis_Perf_t[new_key_modes_prof]\
                                          [RACINE_PLAYER+"_"+str(num_pl_i)]
            if Vi >= Vi_bar:
                cpt_players_stables += 1
        
        if cpt_players_stables == len(key_modes_prof):
            nash_profils.append(key_modes_prof)
                
    return nash_profils

# ______________      main function of brut force   ---> debut      ___________
def nash_balanced_player_game_perf_t(arr_pl_M_T_vars_init,
                                     pi_hp_plus=0.2, 
                                     pi_hp_minus=0.33,
                                     algo_name="BEST-NASH",
                                     path_to_save="tests", 
                                     manual_debug=False, 
                                     dbg=False):
    
    """
    detect the nash balancing profils by the utility of game Perf_t and select
    the best, bad and middle profils by comparing the value of Perf_t

    Returns
    -------
    None.

    """
    print("\n \n {} game: pi_hp_plus={} , pi_hp_minus ={} ---> BEGIN \n"\
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
        
    # _______ variables' initialization --> fin ________________
        
    # ____      game beginning for all t_period ---> debut      _____
    dico_stats_res = dict()
    
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
    for t in range(0, t_periods):
        print("----- t = {} ------ ".format(t))
        possibles_modes = fct_aux.possibles_modes_players_automate(
                                        arr_pl_M_T_vars_modif.copy(), t=t, k=0)
        print("possibles_modes={}".format(len(possibles_modes)))

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
        dico_profs_Vis_Perf_t = dict()
        cpt_profs = 0
        
        mode_profiles = it.product(*possibles_modes)
        for mode_profile in mode_profiles:
            dico_gamme_t = dict()
            arr_pl_M_T_vars_mode_prof, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamme_t \
                = autoBfGameModel.balanced_player_game_4_mode_profil_prices_SG(
                    arr_pl_M_T_vars_modif.copy(),
                    mode_profile, t,
                    pi_hp_plus, pi_hp_minus,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg)
                
            bens_csts_t = bens_t - csts_t
            Perf_t = np.sum(bens_csts_t, axis=0)
            dico_Vis_Pref_t = dict()
            for num_pl_i in range(bens_csts_t.shape[0]):            # bens_csts_t.shape[0] = m_players
                dico_Vis_Pref_t[RACINE_PLAYER+"_"+str(num_pl_i)] \
                    = bens_csts_t[num_pl_i]
            dico_Vis_Pref_t["Perf_t"] = Perf_t
            
            dico_profs_Vis_Perf_t[mode_profile] = dico_Vis_Pref_t
            cpt_profs += 1
            
            if cpt_profs%5000 == 0:
                print("cpt_prof={}".format(cpt_profs))
                
        # detection of NASH profils
        nash_profils = list()
        nash_profils = detect_nash_balancing_profil(
                        dico_profs_Vis_Perf_t,
                        arr_pl_M_T_vars_modif, 
                        t)
        
        # delete all occurences of the profiles 
        print("----> avant supp doublons nash_profils={}".format(len(nash_profils)))
        nash_profils = set(nash_profils)
        print("----> apres supp doublons nash_profils={}".format(len(nash_profils)))
        
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
        
        
        arr_pl_M_T_vars_nash_mode_prof, \
        b0_t, c0_t, \
        bens_t, csts_t, \
        pi_sg_plus_t, pi_sg_minus_t, \
        dico_gamme_t_nash_mode_prof \
            = autoBfGameModel.balanced_player_game_4_mode_profil_prices_SG(
                arr_pl_M_T_vars_modif.copy(),
                best_nash_mode_profile, t,
                pi_hp_plus, pi_hp_minus,
                pi_0_plus_t, pi_0_minus_t,
                manual_debug, dbg)
        dico_stats_res[t] = {"gamma_i": dico_gamme_t_nash_mode_prof,
                             "nash_profils": nash_profils,
                             "best_nash_profil": best_nash_mode_profile,
                             "best_Perf_t": best_key_Perf_t,
                             "nb_nash_profil_byPerf_t": 
                                 dico_Perft_nashProfil[best_key_Perf_t]}
            
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        pi_sg_plus_T[t] = pi_sg_plus_t
        pi_sg_minus_T[t] = pi_sg_minus_t
        
        # b0_ts, c0_ts of shape (T_PERIODS,)
        b0_ts_T[t] = b0_t
        c0_ts_T[t] = c0_t
        
        # BENs, CSTs of shape (M_PLAYERS, T_PERIODS)
        BENs_M_T[:,t] = bens_t
        CSTs_M_T[:,t] = csts_t
        
        arr_pl_M_T_vars_modif[:,t,:] = arr_pl_M_T_vars_nash_mode_prof[:,t,:].copy()
    # ____      game beginning for all t_period ---> fin        _____
    
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

def checkout_nash_equilibrium(arr_pl_M_T_vars_modif, path_to_variable,
                              pi_hp_plus, pi_hp_minus, manual_debug,
                              algo_name="BEST-NASH"):
    """
    pour tout instant de temps t:
    verifier la stabilité de chaque joueur
    mettre le resultat dans un fichier excel dont
        * les lignes sont les joueurs J1, ... ,Jm
        * les colonnes sont le temps et les valeurs la stabilité sous la forme 
            d'un booleen
    """
    
    # read pi_sg_{plus,minus}_T
    pi_sg_plus_T = np.load(os.path.join(path_to_variable, "pi_sg_plus_T_K.npy"),
                          allow_pickle=True)
    pi_sg_minus_T = np.load(os.path.join(path_to_variable, "pi_sg_minus_T_K.npy"),
                          allow_pickle=True)
    pi_0_plus_T = np.load(os.path.join(path_to_variable, "pi_0_plus_T_K.npy"),
                          allow_pickle=True)
    pi_0_minus_T = np.load(os.path.join(path_to_variable, "pi_0_minus_T_K.npy"),
                          allow_pickle=True)
    
    print("**** CHECKOUT STABILITY PLAYERS ****")
    print("SHAPE: arr_pl_M_T_vars={}, pi_sg_plus_T={}, pi_sg_minus_T={},".format(
            arr_pl_M_T_vars_modif.shape, pi_sg_plus_T.shape, pi_sg_minus_T.shape))
    
    # create a result dataframe of checking players' stability and nash equilibrium
    cols = [["players", "states", "nash_modes"]]\
            +[['Vis_t{}'.format(str(t)), 'Vis_bar_t{}'.format(str(t)), 
               'res_t{}'.format(str(t))] 
              for t in range(0, arr_pl_M_T_vars_modif.shape[1])]
    cols = [col for subcol in cols for col in subcol]
    
    id_players = list(range(0, arr_pl_M_T_vars_modif.shape[0]))
    df_res = pd.DataFrame(index=id_players, columns=cols)
    
    m_players = arr_pl_M_T_vars_modif.shape[0]
    t_periods = arr_pl_M_T_vars_modif.shape[1]
    
    for t in range(0, t_periods):
        # calcul de l'utilité pour tous les joueurs selon le profil
        pi_sg_plus_t, pi_sg_minus_t = pi_sg_plus_T[t], pi_sg_minus_T[t]
        pi_0_plus_t, pi_0_minus_t = pi_0_plus_T[t], pi_0_minus_T[t]
        possibles_modes = fct_aux.possibles_modes_players_automate(
                                arr_pl_M_T_vars_modif.copy(), t=t, k=0)
        ## ___to delete________________________________________________________
        # mode_profiles = it.product(*possibles_modes)
        # dico_profs_Vis_Perf_t = dict()
        # dico_profs_Vis_Perf_t, cpt_profs = compute_perf_t_by_mode_profiles(
        #                                     arr_pl_M_T_vars, t,
        #                                     mode_profiles, 
        #                                     pi_sg_plus_t, pi_sg_minus_t,
        #                                     pi_hp_plus, pi_hp_minus, 
        #                                     m_players, t_periods,
        #                                     manual_debug, dbg=False)
        ## __to delete_________________________________________________________
        dico_profs_Vis_Perf_t = dict()
        cpt_profs = 0
        
        mode_profiles = it.product(*possibles_modes)
        for mode_profile in mode_profiles:
            dico_gamme_t = dict()
            arr_pl_M_T_vars_mode_prof, \
            b0_t, c0_t, \
            bens_t, csts_t, \
            pi_sg_plus_t, pi_sg_minus_t, \
            dico_gamme_t \
                = autoBfGameModel.balanced_player_game_4_mode_profil_prices_SG(
                    arr_pl_M_T_vars_modif.copy(),
                    mode_profile, t,
                    pi_hp_plus, pi_hp_minus,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg=False)
                
            bens_csts_t = bens_t - csts_t
            Perf_t = np.sum(bens_csts_t, axis=0)
            dico_Vis_Pref_t = dict()
            for num_pl_i in range(bens_csts_t.shape[0]):            # bens_csts_t.shape[0] = m_players
                dico_Vis_Pref_t[RACINE_PLAYER+"_"+str(num_pl_i)] \
                    = bens_csts_t[num_pl_i]
            dico_Vis_Pref_t["Perf_t"] = Perf_t
            
            dico_profs_Vis_Perf_t[mode_profile] = dico_Vis_Pref_t
            cpt_profs += 1
            
            if cpt_profs%5000 == 0:
                print("cpt_prof={}".format(cpt_profs))
        
        # stabilité de chaque joueur
        nash_modes_profil = list(arr_pl_M_T_vars_modif[:, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]])
        for num_pl_i in range(0, m_players):
            state_i = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
            mode_i = arr_pl_M_T_vars_modif[num_pl_i, t, 
                                    fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]]
            df_res.loc[num_pl_i, "players"] = "player_"+str(num_pl_i)
            df_res.loc[num_pl_i, "nash_modes"] = mode_i
            df_res.loc[num_pl_i, "states"] = state_i
            
            mode_i_bar = find_out_opposite_mode(state_i, mode_i)
            opposite_modes_profil = nash_modes_profil.copy()
            opposite_modes_profil[num_pl_i] = mode_i_bar
            opposite_modes_profil= tuple(opposite_modes_profil)
                
            Vi = None
            Vi = dico_profs_Vis_Perf_t[tuple(nash_modes_profil)]\
                                        [RACINE_PLAYER+"_"+str(num_pl_i)]
            Vi_bar = None
            Vi_bar = dico_profs_Vis_Perf_t[opposite_modes_profil]\
                                              [RACINE_PLAYER+"_"+str(num_pl_i)]
            df_res.loc[num_pl_i, 'Vis_t{}'.format(t)] = Vi
            df_res.loc[num_pl_i, 'Vis_bar_t{}'.format(t)] = Vi_bar
            if Vi >= Vi_bar:
                df_res.loc[num_pl_i, 'res_t{}'.format(t)] = "STABLE"
            else:
                df_res.loc[num_pl_i, 'res_t{}'.format(t)] = "INSTABLE"
            
            # print("-> pl_i = {}, Vi={}, Vi_bar={}".format(num_pl_i, Vi, Vi_bar))
            # print("pl_i = {}, mode_i={}, mode_i_bar={}".format(num_pl_i, mode_i, mode_i_bar))
            # print("nash_prof={},\n bar_prof={} \n".format(nash_modes_profil, 
            #                                             opposite_modes_profil))
                
    # save to excel file
    path_to_save = os.path.join(*["files_debug"])
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df_res.to_excel(os.path.join(
                *[path_to_save,
                  "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
                index=False )

#------------------------------------------------------------------------------
#                       definition of unittests 
#
#------------------------------------------------------------------------------
def test_nash_balanced_player_game_perf_t(algo_name="BEST-NASH"):
    
    fct_aux.N_DECIMALS = 6
    
    pi_hp_plus = 0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 0.33 #[15, 5]
    
    manual_debug=True
    debug = False
    
    t_periods = 2
    set1_m_players, set2_m_players = 20, 12
    set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    #set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    
    set1_m_players, set2_m_players = 10, 6
    # set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
    set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    
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
    
    path_to_save = "tests"
    arr_pl_M_T_vars_nashProfil = nash_balanced_player_game_perf_t(
                                    arr_pl_M_T_vars_init.copy(),
                                    pi_hp_plus=pi_hp_plus, 
                                    pi_hp_minus=pi_hp_minus,
                                    algo_name=algo_name,
                                    path_to_save=path_to_save, 
                                    manual_debug=manual_debug, 
                                    dbg=debug)
    
    # checkout_nash_equilibrium(arr_pl_M_T_vars_nashProfil, path_to_save,
    #                           pi_hp_plus, pi_hp_minus, manual_debug,
    #                           algo_name)
    return arr_pl_M_T_vars_nashProfil
    
#------------------------------------------------------------------------------
#                       execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    for algo_name in fct_aux.ALGO_NAMES_NASH:
        arr_pl_M_T_vars = test_nash_balanced_player_game_perf_t(algo_name)
        #checkout_nash_equilibrium() #(algo_name)
    
    print("runtime = {}".format(time.time() - ti))
