# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:35:46 2021

@author: jwehounou
"""
import os
import time
import psutil

import numpy as np
import pandas as pd
import itertools as it

import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import force_brute_game_model_automate_4_all_t_V1 as bf_V1


from pathlib import Path
from openpyxl import load_workbook


# __________       main function of One BF Algo   ---> debut         __________
def bf_balanced_player_game_ONE_ALGO(arr_pl_M_T_vars_init, algo_name,
                            pi_hp_plus=0.02, 
                            pi_hp_minus=0.33,
                            gamma_version=1,
                            path_to_save="tests", 
                            name_dir="tests", 
                            date_hhmm="DDMM_HHMM",
                            manual_debug=False, 
                            criteria_bf="Perf_t", dbg=False):
    """
    """
    print("\n \n game: pi_hp_plus={}, pi_hp_minus={} ---> debut \n"\
          .format(pi_hp_plus, pi_hp_minus))
        
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    
    # _______ variables' initialization --> debut ________________
    pi_sg_plus_t, pi_sg_minus_t = 0, 0
    pi_sg_plus_T = np.empty(shape=(t_periods,))                                 # shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_sg_minus_T = np.empty(shape=(t_periods,))                                # shape (T_PERIODS,)
    pi_sg_plus_T.fill(np.nan)
    pi_0_plus_t, pi_0_minus_t = 0, 0
    pi_0_plus_T = np.empty(shape=(t_periods,))                                  # shape (T_PERIODS,)
    pi_0_plus_T.fill(np.nan)
    pi_0_minus_T = np.empty(shape=(t_periods,))                                 # shape (T_PERIODS,)
    pi_0_minus_T.fill(np.nan)
    B_is_M = np.empty(shape=(m_players,))                                       # shape (M_PLAYERS, )
    B_is_M.fill(np.nan)
    C_is_M = np.empty(shape=(m_players,))                                       # shape (M_PLAYERS, )
    C_is_M.fill(np.nan)
    B_is_M_T = np.empty(shape=(m_players, t_periods))                                       # shape (M_PLAYERS, )
    B_is_M_T.fill(np.nan)
    C_is_M_T = np.empty(shape=(m_players, t_periods))                                       # shape (M_PLAYERS, )
    C_is_M_T.fill(np.nan)
    b0_ts_T = np.empty(shape=(t_periods,))                                      # shape (T_PERIODS,)
    b0_ts_T.fill(np.nan)
    c0_ts_T = np.empty(shape=(t_periods,))
    c0_ts_T.fill(np.nan)
    BENs_M_T = np.empty(shape=(m_players, t_periods))                           # shape (M_PLAYERS, T_PERIODS)
    CSTs_M_T = np.empty(shape=(m_players, t_periods))
    CC_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    CC_is_M.fill(np.nan)
    BB_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    BB_is_M.fill(np.nan)
    RU_is_M = np.empty(shape=(m_players,))                                      # shape (M_PLAYERS, )
    RU_is_M.fill(np.nan)
    CC_is_M_T = np.empty(shape=(m_players, t_periods))                                      # shape (M_PLAYERS, )
    CC_is_M_T.fill(np.nan)
    BB_is_M_T = np.empty(shape=(m_players, t_periods))                                      # shape (M_PLAYERS, )
    BB_is_M_T.fill(np.nan)
    RU_is_M_T = np.empty(shape=(m_players, t_periods))                                      # shape (M_PLAYERS, )
    RU_is_M_T.fill(np.nan)
    
    arr_pl_M_T_vars_modif = arr_pl_M_T_vars_init.copy()
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_minus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["Si_plus"]] = np.nan
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["bg_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["u_i"]] = 0
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S1_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["S2_p_i_j_k"]] = 0.5
    arr_pl_M_T_vars_modif[:,:,fct_aux.AUTOMATE_INDEX_ATTRS["non_playing_players"]] \
        = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
    dico_id_players = {"players":[fct_aux.RACINE_PLAYER+"_"+str(num_pl_i) 
                                  for num_pl_i in range(0, m_players)]}
    df_nash = pd.DataFrame.from_dict(dico_id_players)
    
    # ____      game beginning for all t_period ---> debut      _____
    pi_sg_plus_t0_minus_1 = pi_hp_plus-1
    pi_sg_minus_t0_minus_1 = pi_hp_minus-1
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = 0, 0
    pi_sg_plus_t, pi_sg_minus_t = None, None
    pi_hp_plus_s = np.array([pi_hp_plus] * t_periods, dtype=object)
    pi_hp_minus_s = np.array([pi_hp_minus] * t_periods, dtype=object)
    dico_modes_profs_by_players_t = dict()
    
    for t in range(0, t_periods):
        print("----- t = {} , free memory={}% ------ ".format(
            t, list(psutil.virtual_memory())[2]))
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
               
        arr_pl_M_t_vars_init = arr_pl_M_T_vars_modif[:,t,:].copy()
        arr_pl_M_t_plus_1_vars_init = arr_pl_M_T_vars_modif[:,t+1,:].copy() \
                                        if t+1 < t_periods \
                                        else arr_pl_M_T_vars_modif[:,t,:].copy()
        arr_pl_M_t_minus_1_vars_init = arr_pl_M_T_vars_modif[:,t-1,:].copy() \
                                        if t-1 >= 0 \
                                        else arr_pl_M_T_vars_modif[:,t,:].copy()                        
        
        print("Sis_init = {}".format(arr_pl_M_T_vars_modif[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))
        print("Sis_+1 = {}".format(arr_pl_M_t_plus_1_vars_init[:,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))
        print("Sis_-1 = {}".format(arr_pl_M_t_minus_1_vars_init[:,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))
        
        arr_pl_M_t_vars_modif = bf_V1.compute_gamma_state_4_period_t(
                                arr_pl_M_t_K_vars=arr_pl_M_t_vars_init,
                                arr_pl_M_t_minus_1_K_vars=arr_pl_M_t_minus_1_vars_init,
                                arr_pl_M_t_plus_1_K_vars=arr_pl_M_t_plus_1_vars_init,
                                t=t,
                                pi_0_plus=pi_0_plus_t, pi_0_minus=pi_0_minus_t,
                                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
                                m_players=m_players,
                                t_periods=t_periods,
                                gamma_version=gamma_version,
                                manual_debug=manual_debug,
                                dbg=dbg)
        print("Sis = {} \n".format(arr_pl_M_t_vars_modif[:,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))

        
        # quantity of energies (prod_is, cons_is) from 0 to t-2 to get values 
        # for t-1 periods
        sum_diff_pos_minus_0_t_minus_2 = None                                  # sum of the positive difference btw cons_is and prod_is from 0 to t-2 
        sum_diff_pos_plus_0_t_minus_2 = None                                   # sum of the positive difference btw prod_is and cons_is from 0 to t-2
        sum_cons_is_0_t_minus_2 = None                                         # sum of the cons of all players from 0 to t-2
        sum_prod_is_0_t_minus_2 = None                                         # sum of the prod of all players from 0 to t-2
        
        sum_diff_pos_minus_0_t_minus_2, \
        sum_diff_pos_plus_0_t_minus_2, \
        sum_cons_is_0_t_minus_2, \
        sum_prod_is_0_t_minus_2 \
            = bf_V1.get_sum_cons_prod_from_0_t_minus_2(arr_pl_M_T_vars_modif,t)
        print("t={}, sum_diff_pos_minus_0_t_minus_2={}, sum_diff_pos_plus_0_t_minus_2={}, sum_cons_is_0_t_minus_2={}, sum_prod_is_0_t_minus_2={}".format(t,sum_diff_pos_minus_0_t_minus_2,
                        sum_diff_pos_plus_0_t_minus_2,
                        sum_cons_is_0_t_minus_2, 
                        sum_prod_is_0_t_minus_2))
        
            
        # balanced player game at instant t    
        list_dico_modes_profs_by_players_t_best = list()
        list_dico_modes_profs_by_players_t_bad = list()
        list_dico_modes_profs_by_players_t_mid = list()
        
        list_dico_modes_profs_by_players_t_best, \
        list_dico_modes_profs_by_players_t_bad, \
        list_dico_modes_profs_by_players_t_mid\
            = bf_V1.generer_balanced_players_4_modes_profils(
                arr_pl_M_t_vars_modif, 
                m_players, t,
                sum_diff_pos_minus_0_t_minus_2,
                sum_diff_pos_plus_0_t_minus_2,
                sum_cons_is_0_t_minus_2,                             
                sum_prod_is_0_t_minus_2,
                pi_hp_plus, pi_hp_minus,
                pi_0_plus_t, pi_0_minus_t,
                manual_debug, dbg)
          
        list_dico_modes_profs_by_players_t = dict()
        if algo_name == fct_aux.ALGO_NAMES_BF[0]:                              # BEST-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_best
            
        elif algo_name == fct_aux.ALGO_NAMES_BF[1]:                            # BAD-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_bad
            
        elif algo_name == fct_aux.ALGO_NAMES_BF[2]:                            # MIDDLE-BRUTE-FORCE
            list_dico_modes_profs_by_players_t \
                = list_dico_modes_profs_by_players_t_mid
                
        rd_key = None
        if len(list_dico_modes_profs_by_players_t) == 1:
            rd_key = 0
        else:
            rd_key = np.random.randint(
                        0, 
                        len(list_dico_modes_profs_by_players_t))
        
        id_cpt_xxx, dico_mode_prof_by_players \
            = list_dico_modes_profs_by_players_t[rd_key] 
        print("rd_key={}, cpt_xxx={}".format(rd_key, id_cpt_xxx))
        
        bens_t = dico_mode_prof_by_players["bens_t"]
        csts_t = dico_mode_prof_by_players["csts_t"]
        Perf_t = dico_mode_prof_by_players["Perf_t"]
        b0_t = dico_mode_prof_by_players["b0_t"]
        c0_t = dico_mode_prof_by_players["c0_t"]
        Out_sg = dico_mode_prof_by_players["Out_sg"]
        In_sg = dico_mode_prof_by_players["In_sg"]
        pi_sg_plus_t = dico_mode_prof_by_players["pi_sg_plus_t"]
        pi_sg_minus_t = dico_mode_prof_by_players["pi_sg_minus_t"]
        pi_0_plus_t = dico_mode_prof_by_players["pi_0_plus_t"]
        pi_0_minus_t = dico_mode_prof_by_players["pi_0_minus_t"]
        mode_profile = dico_mode_prof_by_players["mode_profile"]
        
        arr_pl_M_t_vars_modif[:, 
                              fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]] \
            = mode_profile
        
        print("mode_profile={}, mode_is={}".format(mode_profile, 
                arr_pl_M_t_vars_modif[:,fct_aux.AUTOMATE_INDEX_ATTRS["mode_i"]]))
        print("state_is={} ".format( 
                arr_pl_M_t_vars_modif[:,fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]))
        
        arr_pl_M_t_vars_modif = bf_V1.balanced_player_game_4_mode_profil(
                                         arr_pl_M_t_vars_modif, 
                                         m_players,
                                         dbg)
        ## test if there are the same values like these in dico_mode_prof_by_players
        In_sg_new, Out_sg_new, b0_t_new, c0_t_new = None, None, None, None
        bens_t_new, csts_t_new = None, None
        pi_sg_plus_t_new, pi_sg_minus_t_new = None, None
        In_sg_new, Out_sg_new, \
        b0_t_new, c0_t_new, \
        bens_t_new, csts_t_new, \
        pi_sg_plus_t_new, pi_sg_minus_t_new \
            = bf_V1.compute_prices_inside_SG(
                    arr_pl_M_t_vars_modif, 
                    sum_diff_pos_minus_0_t_minus_2,
                    sum_diff_pos_plus_0_t_minus_2,
                    sum_cons_is_0_t_minus_2,                             
                    sum_prod_is_0_t_minus_2,
                    pi_hp_plus, pi_hp_minus,
                    pi_0_plus_t, pi_0_minus_t,
                    manual_debug, dbg)
        bens_csts_t_new = bens_t_new - csts_t_new
        Perf_t_new = np.sum(bens_csts_t_new, axis=0)
        ##### verification of best key quality 
        diff = np.abs(Perf_t_new - Perf_t)
        print(" Perf_t_algo == Perf_t_new --> OK (diff={}) ".format(diff)) \
            if diff < 0.1 \
            else print("Perf_t_algo != Perf_t_new --> NOK (diff={}) \n"\
                       .format(diff))     
        print("b0_t={}, c0_t={}, Out_sg={},In_sg={} \n".format(
                b0_t, c0_t, Out_sg, In_sg))
        
        # pi_sg_{plus,minus} of shape (T_PERIODS,)
        if np.isnan(pi_sg_plus_t):
            pi_sg_plus_t = 0
        if np.isnan(pi_sg_minus_t):
            pi_sg_minus_t = 0
                
        # checkout NASH equilibrium
        bens_csts_M_t = bens_t - csts_t
        df_nash_t = None
        df_nash_t = bf_V1.checkout_nash_4_profils_by_periods(
                        arr_pl_M_t_vars_modif.copy(),
                        arr_pl_M_T_vars_init[:,t,:],
                        sum_diff_pos_minus_0_t_minus_2,
                        sum_diff_pos_plus_0_t_minus_2,
                        sum_cons_is_0_t_minus_2,                             
                        sum_prod_is_0_t_minus_2,
                        pi_hp_plus, pi_hp_minus, 
                        pi_0_minus_t, pi_0_plus_t, 
                        bens_csts_M_t,
                        m_players,
                        t,
                        manual_debug,
                        dbg)
        df_nash = pd.merge(df_nash, df_nash_t, on='players', how='outer')
        
        #_______     save arr_M_t_vars at t in dataframe : debut    _______
        df_arr_M_t_vars_modif \
            = pd.DataFrame(arr_pl_M_t_vars_modif, 
                            columns=fct_aux.AUTOMATE_INDEX_ATTRS.keys(),
                            index=dico_id_players["players"])
        path_to_save_M_t_vars_modif = path_to_save
        msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
        if "simu_DDMM_HHMM" in path_to_save:
            path_to_save_M_t_vars_modif \
                = os.path.join(name_dir, "simu_"+date_hhmm,
                               msg, algo_name, "intermediate_t"
                               )
        Path(path_to_save_M_t_vars_modif).mkdir(parents=True, 
                                                     exist_ok=True)
            
        path_2_xls_df_arr_M_t_vars_modif \
            = os.path.join(
                path_to_save_M_t_vars_modif,
                  "arr_M_T_vars_{}.xlsx".format(algo_name)
                )
        if not os.path.isfile(path_2_xls_df_arr_M_t_vars_modif):
            df_arr_M_t_vars_modif.to_excel(
                path_2_xls_df_arr_M_t_vars_modif,
                sheet_name="t{}".format(t),
                index=True)
        else:
            book = load_workbook(filename=path_2_xls_df_arr_M_t_vars_modif)
            with pd.ExcelWriter(path_2_xls_df_arr_M_t_vars_modif, 
                                engine='openpyxl') as writer:
                writer.book = book
                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)    
            
                ## Your dataframe to append. 
                df_arr_M_t_vars_modif.to_excel(writer, "t{}".format(t))  
            
                writer.save() 
        #_______     save arr_M_t_vars at t in dataframe : fin     _______
        
        #___________ update saving variables : debut ______________________
        arr_pl_M_T_vars_modif[:,t,:] = arr_pl_M_t_vars_modif
        b0_ts_T, c0_ts_T, \
        BENs_M_T, CSTs_M_T, \
        pi_sg_plus_T, pi_sg_minus_T, \
        pi_0_plus_T, pi_0_minus_T, \
        df_nash \
            = bf_V1.update_saving_variables(t, 
                b0_ts_T, b0_t,
                c0_ts_T, c0_t,
                BENs_M_T, bens_t,
                CSTs_M_T, csts_t,
                pi_sg_plus_T, pi_sg_plus_t,
                pi_sg_minus_T, pi_sg_minus_t,
                pi_0_plus_T, pi_0_plus_t,
                pi_0_minus_T, pi_0_minus_t,
                df_nash, df_nash
                )
        BB_is_M_T, CC_is_M_T, RU_is_M_T, \
        B_is_M_T, C_is_M_T \
            = bf_V1.compute_prices_variables(
                arr_pl_M_T_vars_modif, t,
                b0_ts_T, c0_ts_T, 
                pi_sg_plus_T, pi_sg_minus_T,
                pi_0_plus_T, pi_0_minus_T
                )
        dico_modes_profs_by_players_t[t] = dico_mode_prof_by_players
        #___________ update saving variables : fin   ______________________
        print("Sis = {}".format(arr_pl_M_T_vars_modif[:,t,fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]))

        print("----- t={} After running free memory={}% ------ ".format(
            t, list(psutil.virtual_memory())[2]))
    
    # __________        compute prices variables         ____________________
    B_is_M = np.sum(B_is_M_T, axis=1)
    C_is_M = np.sum(C_is_M_T, axis=1)
    BB_is_M = np.sum(BB_is_M_T, axis=1) 
    CC_is_M = np.sum(CC_is_M_T, axis=1) 
    RU_is_M = np.sum(RU_is_M_T, axis=1)
    # __________        compute prices variables         ____________________
    
    #_______      save computed variables locally from algo_name     __________
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    
    print("path_to_save={}".format(path_to_save))
    if "simu_DDMM_HHMM" in path_to_save:
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    df_nash.to_excel(os.path.join(
                *[path_to_save,
                  "resume_verify_Nash_equilibrium_{}.xlsx".format(algo_name)]), 
                index=False)

    fct_aux.save_variables(path_to_save, arr_pl_M_T_vars_modif, 
               b0_ts_T, c0_ts_T, B_is_M, C_is_M, 
               BENs_M_T, CSTs_M_T, 
               BB_is_M, CC_is_M, RU_is_M, 
               pi_sg_minus_T, pi_sg_plus_T, 
               pi_0_minus_T, pi_0_plus_T,
               pi_hp_plus_s, pi_hp_minus_s, 
               dico_modes_profs_by_players_t, 
               algo=algo_name,
               dico_best_steps=dict())
    bf_V1.turn_dico_stats_res_into_df_BF(
          dico_modes_profs_players_algo = dico_modes_profs_by_players_t, 
          path_to_save = path_to_save, 
          t_periods = t_periods, 
          manual_debug = manual_debug, 
          algo_name = algo_name)
    
    return arr_pl_M_T_vars_modif
    
# __________       main function of One BF Algo   ---> fin        ____________

###############################################################################
#                   definition  des unittests
#
###############################################################################
def test_BRUTE_FORCE_balanced_player_game_Pi_Ci_NEW_AUTOMATE():
    fct_aux.N_DECIMALS = 8
    
    pi_hp_plus = 10 #0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 20 #0.33 #[15, 5]
    
    manual_debug = True
    gamma_version = 2 #1,2
    debug = False
    criteria_bf = "Perf_t"
    
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    t_periods = 4 #2
    setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    setA_m_players, setB_m_players, setC_m_players = 8, 3, 3                   # 14 players
    # t_periods = 4
    # setA_m_players, setB_m_players, setC_m_players = 10, 6, 5                  # 21 players
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                            setA_m_players, setB_m_players, setC_m_players, 
                            t_periods, 
                            scenario1,
                            path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    
    algo_name = fct_aux.ALGO_NAMES_BF[0]
    arr_pl_M_T_vars = bf_balanced_player_game_ONE_ALGO(
                                arr_pl_M_T_vars_init.copy(),
                                algo_name,
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                gamma_version = gamma_version,
                                path_to_save="tests", 
                                name_dir="tests", 
                                date_hhmm="DDMM_HHMM",
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
    
    return arr_pl_M_T_vars
    
def test_BRUTE_FORCE_balanced_player_game_Pi_Ci_one_period():
    
    fct_aux.N_DECIMALS = 8
    
    pi_hp_plus = 10 #0.2*pow(10,-3) #[5, 15]
    pi_hp_minus = 20 #0.33 #[15, 5]
    
    manual_debug=True
    gamma_version = 2 #1,2
    debug = False
    criteria_bf = "Perf_t"
    
    setA_m_players = 15; setB_m_players = 10; setC_m_players = 10
    setA_m_players, setB_m_players, setC_m_players = 8, 3, 3
    setA_m_players, setB_m_players, setC_m_players = 6, 2, 2
    t_periods = 1 
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False#True
    
    prob_A_A = 0.7; prob_A_B = 0.3; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario = [(prob_A_A, prob_A_B, prob_A_C), (prob_B_A, prob_B_B, prob_B_C),
                (prob_C_A, prob_C_B, prob_C_C)]
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_one_period(
                        setA_m_players, setB_m_players, setC_m_players, 
                        t_periods, 
                        scenario,
                        path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl_one_period(arr_pl_M_T_vars_init)
    
    algo_name = fct_aux.ALGO_NAMES_BF[0]
    arr_pl_M_T_vars = bf_balanced_player_game_ONE_ALGO(
                                arr_pl_M_T_vars_init.copy(),
                                algo_name,
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                gamma_version = gamma_version,
                                path_to_save="tests", 
                                name_dir="tests", 
                                date_hhmm="DDMM_HHMM",
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
    
    return arr_pl_M_T_vars
    

###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    arr_pl_M_T_vars_modif \
        = test_BRUTE_FORCE_balanced_player_game_Pi_Ci_NEW_AUTOMATE()
        
    # arr_pl_M_T_vars_modif \
    #     = test_BRUTE_FORCE_balanced_player_game_Pi_Ci_one_period()
    
    print("runtime = {}".format(time.time() - ti))