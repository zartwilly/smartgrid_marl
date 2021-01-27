# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:56:10 2021

@author: jwehounou
"""

import os
import time
import execution_game_automate as autoExeGame
import fonctions_auxiliaires as fct_aux
import visu_bkh_automate as autoVizGame


if __name__ == "__main__":
    ti = time.time()
    
    debug = True #False #False #True
    
    name_dir="tests"
    
    pi_hp_plus, pi_hp_minus = None, None
    set1_m_players, set2_m_players = None, None
    t_periods, k_steps, NB_REPEAT_K_MAX = None, None, None
    learning_rates = None, None, None
    date_hhmm, Visualisation = None, None
    used_storage_det=True
    criteria_bf="Perf_t" # "In_sg_Out_sg"
    
    
    if debug:
        # ---- new constances simu_DDMM_HHMM --- **** debug *****
        date_hhmm="DDMM_HHMM"
        t_periods = 2
        k_steps = 250
        NB_REPEAT_K_MAX= 3 #15 #30
        learning_rates = [0.1] #[0.01] #[0.0001]
        
        pi_hp_plus = [0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [0.33] #[15, 5]
        algos = ["LRI1", "LRI2", "DETERMINIST"] \
                + fct_aux.ALGO_NAMES_NASH \
                + fct_aux.ALGO_NAMES_BF
        
        used_storage_det= False #True
        manual_debug = True
        Visualisation = True #False, True
        
        # ---- initialization of variables for generating instances ----
        set1_m_players, set2_m_players = 10, 6
        # set1_stateId0_m_players, set2_stateId0_m_players = 15, 5
        set1_stateId0_m_players, set2_stateId0_m_players = 0.75, 0.42 #0.42
    else:
       # ---- new constances simu_2306_2206 --- **** debug ***** 
       date_hhmm="2306_2206"
       t_periods = 110
       k_steps = 1000
       NB_REPEAT_K_MAX = 15 #30
       learning_rates = [0.1] #[0.01] #[0.0001]
       
       pi_hp_plus = [0.2*pow(10,-3)] #[5, 15]
       pi_hp_minus = [0.33] #[15, 5]
       
       used_storage_det= True #False #True
       manual_debug = False #True
       Visualisation = True #False, True
       
       # ---- initialization of variables for generating instances ----
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
    
    algos= None #["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST", "BRUTE-FORCE"] 
    if set1_m_players + set2_m_players <= 20:
        algos = ["LRI1", "LRI2", "DETERMINIST"] \
                + fct_aux.ALGO_NAMES_NASH \
                + fct_aux.ALGO_NAMES_BF
    else:
        algos=["LRI1", "LRI2", "DETERMINIST"] 
        
    
    autoExeGame.execute_algos_used_Generated_instances(
                arr_pl_M_T_vars_init, 
                name_dir = name_dir,
                date_hhmm = date_hhmm,
                k_steps = k_steps,
                NB_REPEAT_K_MAX = NB_REPEAT_K_MAX,
                algos = algos,
                learning_rates = learning_rates,
                pi_hp_plus = pi_hp_plus,
                pi_hp_minus = pi_hp_minus,
                used_instances = used_instances,
                used_storage_det = used_storage_det,
                manual_debug = manual_debug, 
                criteria_bf = criteria_bf, 
                debug = False
                )
    
    if Visualisation: 
        autoVizGame.MULT_WIDTH = 2.25;
        autoVizGame.MULT_HEIGHT = 1.1;
        
        name_simu = "simu_"+date_hhmm; k_steps_args = k_steps
        t = 1
        
        autoVizGame.NAME_RESULT_SHOW_VARS \
            = autoVizGame.NAME_RESULT_SHOW_VARS.format(pi_hp_plus[0], 
                                                       pi_hp_minus[0])
        ## ____________          turn_arr4d_2_df() ---> debut          ____________
        algos_4_no_learning = ["DETERMINIST","RD-DETERMINIST"] \
                                + fct_aux.ALGO_NAMES_BF \
                                + fct_aux.ALGO_NAMES_NASH
        algos_4_learning = ["LRI1", "LRI2"]
        tuple_paths, prices, algos, learning_rates \
            = autoVizGame.get_tuple_paths_of_arrays(name_simu=name_simu, 
                                        algos_4_no_learning=algos_4_no_learning)
            
        df_arr_M_T_Ks, \
        df_ben_cst_M_T_K, \
        df_b0_c0_pisg_pi0_T_K, \
        df_B_C_BB_CC_RU_M \
            = autoVizGame.get_array_turn_df_for_t(
                tuple_paths, t, k_steps_args, 
                algos_4_no_learning=algos_4_no_learning, 
                algos_4_learning=algos_4_learning)
        print("df_arr_M_T_Ks: {}, df_ben_cst_M_T_K={}, df_b0_c0_pisg_pi0_T_K={}, df_B_C_BB_CC_RU_M={}".format( 
            df_arr_M_T_Ks.shape, df_ben_cst_M_T_K.shape, df_b0_c0_pisg_pi0_T_K.shape, 
            df_B_C_BB_CC_RU_M.shape ))
        print("size t={}, df_arr_M_T_Ks={} Mo, df_ben_cst_M_T_K={} Mo, df_b0_c0_pisg_pi0_T_K={} Mo, df_B_C_BB_CC_RU_M={} Mo".format(
                t, 
              round(df_arr_M_T_Ks.memory_usage().sum()/(1024*1024), 2),  
              round(df_ben_cst_M_T_K.memory_usage().sum()/(1024*1024), 2),
              round(df_b0_c0_pisg_pi0_T_K.memory_usage().sum()/(1024*1024), 2),
              round(df_B_C_BB_CC_RU_M.memory_usage().sum()/(1024*1024), 2)
              ))
        ## ____________          turn_arr4d_2_df() ---> fin            ____________
        
        ## ____________          plot figures ---> debut            ____________
        name_dir = os.path.join("tests", name_simu)
        autoVizGame.group_plot_on_panel(
            df_arr_M_T_Ks, df_ben_cst_M_T_K, t, name_dir, 
            autoVizGame.NAME_RESULT_SHOW_VARS)
        
        ## ____________          plot figures ---> fin            ____________
           
        # fct_aux.resume_game_on_excel_file_automate(
        #         df_arr_M_T_Ks, 
        #         df_ben_cst_M_T_K,
        #         t = t, 
        #         m_players=m_players, 
        #         t_periods=num_periods, 
        #         k_steps=k_steps,
        #         scenario=scenarios[0], 
        #         learning_rate=learning_rates[-1], 
        #         prob_Ci=prob_Ci)
        fct_aux.resume_game_on_excel_file_automate(
                df_arr_M_T_Ks, df_ben_cst_M_T_K, t = t, 
                set1_m_players = set1_m_players, 
                set1_stateId0_m_players = set1_stateId0_m_players, 
                set2_m_players = set2_m_players, 
                set2_stateId0_m_players = set2_stateId0_m_players, 
                t_periods = t_periods, k_steps = k_steps_args, 
                learning_rate=learning_rates[-1])
        
    print("runtime = {}".format(time.time() - ti))