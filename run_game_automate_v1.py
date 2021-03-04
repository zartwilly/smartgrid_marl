# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:32:57 2021

@author: jwehounou
"""

import os
import time
import execution_game_automate as autoExeGame
import fonctions_auxiliaires as fct_aux
import visu_bkh_automate as autoVizGame
import visu_bkh_automate_v1 as autoVizGameV1


if __name__ == "__main__":
    ti = time.time()
    
    # _____                     scenarios --> debut                 __________
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    prob_A_A = 0.9; prob_A_B = 0.1; prob_A_C = 0.0;
    prob_B_A = 0.1; prob_B_B = 0.8; prob_B_C = 0.1;
    prob_C_A = 0.0; prob_C_B = 0.1; prob_C_C = 0.9;
    scenario2 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    prob_A_A = 0.3; prob_A_B = 0.4; prob_A_C = 0.3;
    prob_B_A = 0.4; prob_B_B = 0.2; prob_B_C = 0.4;
    prob_C_A = 0.3; prob_C_B = 0.4; prob_C_C = 0.3;
    scenario3 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    # _____                     scenarios --> fin                   __________
    
    
    debug = True #False #False #True
    
    name_dir="tests"
    
    pi_hp_plus, pi_hp_minus = None, None
    setA_m_players, setB_m_players, setC_m_players = None, None, None
    t_periods, k_steps, NB_REPEAT_K_MAX = None, None, None
    learning_rates = None, None, None
    date_hhmm, Visualisation = None, None
    used_storage_det=True
    criteria_bf="Perf_t" # "In_sg_Out_sg"
    dbg_234_players = None
    
    if debug:
        # ---- new constances simu_DDMM_HHMM --- **** debug *****
        date_hhmm="DDMM_HHMM"
        t_periods = 3#30 #35 #55 #117 #15 #3
        k_steps = 250 #5000 #2000 #50 #250
        NB_REPEAT_K_MAX= 10 #3 #15 #30
        learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
        fct_aux.N_DECIMALS = 8
        
        pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
        
        algos = ["LRI1", "LRI2", "DETERMINIST"] \
                + fct_aux.ALGO_NAMES_NASH \
                + fct_aux.ALGO_NAMES_BF
        
        dbg_234_players = False #True #False
        used_storage_det= False #True
        manual_debug = False #True #False #True
        Visualisation = True #False, True
        
        scenario = scenario1
        
        # ---- initialization of variables for generating instances ----
        setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
        if dbg_234_players:
            t_periods = 2
            setA_m_players, setB_m_players, setC_m_players = 1, 1, 1
    else:
       # ---- new constances simu_2306_2206 --- **** debug ***** 
       date_hhmm="2306_2206"
       t_periods = 110
       k_steps = 1000
       NB_REPEAT_K_MAX = 15 #30
       learning_rates = [0.1] #[0.01] #[0.0001]
       
       pi_hp_plus = [0.2*pow(10,-3)] #[5, 15]
       pi_hp_minus = [0.33] #[15, 5]
       
       dbg_234_players = False
       used_storage_det= True #False #True
       manual_debug = False #True
       Visualisation = True #False, True
       
       # ---- initialization of variables for generating instances ----
       setA_m_players, setB_m_players, setC_m_players = 20, 12, 6
       
       
    set1_states, set2_states = None, None
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    algos= None
    arr_pl_M_T_vars_init = None 
    if not dbg_234_players:
        arr_pl_M_T_vars_init \
            = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                setA_m_players, setB_m_players, setC_m_players, 
                t_periods, 
                scenario,
                path_to_arr_pl_M_T, used_instances)
        algos= None #["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST", "BRUTE-FORCE"] 
        if setA_m_players + setB_m_players + setC_m_players <= 20:
            algos = ["LRI1", "LRI2", "DETERMINIST"] \
                    + fct_aux.ALGO_NAMES_BF
                    # + fct_aux.ALGO_NAMES_NASH 
        else:
            algos= ["LRI1", "DETERMINIST"]  #["LRI1", "LRI2", "DETERMINIST"] 
    else:
        # ---- DEBUG A EFFACER apres debug ----
        arr_pl_M_T_vars_init \
            = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                setA_m_players, setB_m_players, setC_m_players, 
                t_periods, 
                scenario,
                path_to_arr_pl_M_T, used_instances)
        algos = ["LRI1", "LRI2", "DETERMINIST"] \
                    + fct_aux.ALGO_NAMES_BF \
                    + fct_aux.ALGO_NAMES_NASH 
        
    
    # autoExeGame.execute_algos_used_Generated_instances(
    #             arr_pl_M_T_vars_init, 
    #             name_dir = name_dir,
    #             date_hhmm = date_hhmm,
    #             k_steps = k_steps,
    #             NB_REPEAT_K_MAX = NB_REPEAT_K_MAX,
    #             algos = algos,
    #             learning_rates = learning_rates,
    #             pi_hp_plus = pi_hp_plus,
    #             pi_hp_minus = pi_hp_minus,
    #             used_instances = used_instances,
    #             used_storage_det = used_storage_det,
    #             manual_debug = manual_debug, 
    #             criteria_bf = criteria_bf, 
    #             debug = False
    #             )
    autoExeGame.execute_algos_used_Generated_instances_USE_DICT_MODE_PROFIL(
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
        autoVizGameV1.MULT_WIDTH = 2.0 #2.25;
        autoVizGameV1.MULT_HEIGHT = 0.7 #1.0 #1.1;
        
        name_simu = "simu_"+date_hhmm; k_steps_args = k_steps
        t = 0 #1
        
        autoVizGameV1.NAME_RESULT_SHOW_VARS \
            = autoVizGameV1.NAME_RESULT_SHOW_VARS.format(pi_hp_plus[0], 
                                                       pi_hp_minus[0])
            
        ## _________          turn_arr4d_2_df() ---> debut           __________
        algos_4_no_learning = ["DETERMINIST","RD-DETERMINIST"] \
                                + fct_aux.ALGO_NAMES_BF \
                                + fct_aux.ALGO_NAMES_NASH
        algos_4_learning = ["LRI1", "LRI2"]
        algos_4_showing = ["DETERMINIST", "LRI1", "LRI2"] \
                        + [fct_aux.ALGO_NAMES_BF[0], fct_aux.ALGO_NAMES_BF[1]]
                        
        tuple_paths, prices, \
        algos, learning_rates, \
        path_2_best_learning_steps \
        = autoVizGameV1.get_tuple_paths_of_arrays(
            name_simu=name_simu, 
            algos_4_no_learning=algos_4_no_learning, 
            algos_4_showing = algos_4_showing
            )
        
        print("get_tuple_paths_of_arrays: TERMINE")    
        
        dico_k_stop = dict()
        df_LRI_12, df_k_stop = autoVizGameV1.get_k_stop_4_periods(
                                                path_2_best_learning_steps)
        print("get_k_stop_4_periods: TERMINE") 
    
        
        df_arr_M_T_Ks, \
        df_ben_cst_M_T_K, \
        df_b0_c0_pisg_pi0_T_K, \
        df_B_C_BB_CC_RU_M \
            = autoVizGameV1.get_array_turn_df_for_t(
                tuple_paths, t=None, k_steps_args=k_steps_args, 
                algos_4_no_learning=algos_4_no_learning, 
                algos_4_learning=algos_4_learning)
        print("df_arr_M_T_Ks: {}, df_ben_cst_M_T_K={}, df_b0_c0_pisg_pi0_T_K={}, df_B_C_BB_CC_RU_M={}".format( 
            df_arr_M_T_Ks.shape, df_ben_cst_M_T_K.shape, 
            df_b0_c0_pisg_pi0_T_K.shape, df_B_C_BB_CC_RU_M.shape ))
        print("size t={}, df_arr_M_T_Ks={} Mo, df_ben_cst_M_T_K={} Mo, df_b0_c0_pisg_pi0_T_K={} Mo, df_B_C_BB_CC_RU_M={} Mo".format(
                t, 
              round(df_arr_M_T_Ks.memory_usage().sum()/(1024*1024), 2),  
              round(df_ben_cst_M_T_K.memory_usage().sum()/(1024*1024), 2),
              round(df_b0_c0_pisg_pi0_T_K.memory_usage().sum()/(1024*1024), 2),
              round(df_B_C_BB_CC_RU_M.memory_usage().sum()/(1024*1024), 2)
              ))
        print("get_array_turn_df_for_t: TERMINE")
        ## __________          turn_arr4d_2_df() ---> fin         ____________
    
        ## ____________          plot figures ---> debut           ____________
        name_dir = os.path.join("tests", name_simu)
        autoVizGameV1.group_plot_on_panel(
                        df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                        df_B_C_BB_CC_RU_M,
                        df_b0_c0_pisg_pi0_T_K,
                        t, k_steps_args, name_dir, 
                        df_LRI_12, df_k_stop,
                        path_2_best_learning_steps, 
                        autoVizGameV1.NAME_RESULT_SHOW_VARS)
        ## ____________          plot figures ---> fin            ____________
        
        
    
    # if Visualisation: 
    #     autoVizGame.MULT_WIDTH = 2.25;
    #     autoVizGame.MULT_HEIGHT = 1.1;
        
    #     name_simu = "simu_"+date_hhmm; k_steps_args = k_steps
    #     t = 0 #1
        
    #     autoVizGame.NAME_RESULT_SHOW_VARS \
    #         = autoVizGame.NAME_RESULT_SHOW_VARS.format(pi_hp_plus[0], 
    #                                                    pi_hp_minus[0])
    #     ## ____________          turn_arr4d_2_df() ---> debut          ____________
    #     algos_4_no_learning = ["DETERMINIST","RD-DETERMINIST"] \
    #                             + fct_aux.ALGO_NAMES_BF \
    #                             + fct_aux.ALGO_NAMES_NASH
    #     algos_4_learning = ["LRI1", "LRI2"]
    #     tuple_paths, prices, algos, learning_rates, path_2_best_learning_steps \
    #         = autoVizGame.get_tuple_paths_of_arrays(name_simu=name_simu, 
    #                                     algos_4_no_learning=algos_4_no_learning)
            
    #     df_arr_M_T_Ks, \
    #     df_ben_cst_M_T_K, \
    #     df_b0_c0_pisg_pi0_T_K, \
    #     df_B_C_BB_CC_RU_M \
    #         = autoVizGame.get_array_turn_df_for_t(
    #             tuple_paths, t, k_steps_args, 
    #             algos_4_no_learning=algos_4_no_learning, 
    #             algos_4_learning=algos_4_learning)
    #     print("df_arr_M_T_Ks: {}, df_ben_cst_M_T_K={}, df_b0_c0_pisg_pi0_T_K={}, df_B_C_BB_CC_RU_M={}".format( 
    #         df_arr_M_T_Ks.shape, df_ben_cst_M_T_K.shape, df_b0_c0_pisg_pi0_T_K.shape, 
    #         df_B_C_BB_CC_RU_M.shape ))
    #     print("size t={}, df_arr_M_T_Ks={} Mo, df_ben_cst_M_T_K={} Mo, df_b0_c0_pisg_pi0_T_K={} Mo, df_B_C_BB_CC_RU_M={} Mo".format(
    #             t, 
    #           round(df_arr_M_T_Ks.memory_usage().sum()/(1024*1024), 2),  
    #           round(df_ben_cst_M_T_K.memory_usage().sum()/(1024*1024), 2),
    #           round(df_b0_c0_pisg_pi0_T_K.memory_usage().sum()/(1024*1024), 2),
    #           round(df_B_C_BB_CC_RU_M.memory_usage().sum()/(1024*1024), 2)
    #           ))
    #     ## ____________          turn_arr4d_2_df() ---> fin            ____________
        
    #     ## ____________          plot figures ---> debut            ____________
    #     name_dir = os.path.join("tests", name_simu)
    #     autoVizGame.group_plot_on_panel(
    #         df_arr_M_T_Ks, df_ben_cst_M_T_K, t, name_dir, 
    #         path_2_best_learning_steps,
    #         autoVizGame.NAME_RESULT_SHOW_VARS)
        
    #     ## ____________          plot figures ---> fin            ____________
           
    #     # fct_aux.resume_game_on_excel_file_automate(
    #     #         df_arr_M_T_Ks, 
    #     #         df_ben_cst_M_T_K,
    #     #         t = t, 
    #     #         m_players=m_players, 
    #     #         t_periods=num_periods, 
    #     #         k_steps=k_steps,
    #     #         scenario=scenarios[0], 
    #     #         learning_rate=learning_rates[-1], 
    #     #         prob_Ci=prob_Ci)
    #     # fct_aux.resume_game_on_excel_file_automate(
    #     #         df_arr_M_T_Ks, df_ben_cst_M_T_K, 
    #     #         df_b0_c0_pisg_pi0_T_K = df_b0_c0_pisg_pi0_T_K,
    #     #         t = t, 
    #     #         set1_m_players = set1_m_players, 
    #     #         set1_stateId0_m_players = set1_stateId0_m_players, 
    #     #         set2_m_players = set2_m_players, 
    #     #         set2_stateId0_m_players = set2_stateId0_m_players, 
    #     #         t_periods = t_periods, k_steps = k_steps_args, 
    #     #         learning_rate=learning_rates[-1], 
    #     #         price = str(pi_hp_plus[0]) +"_"+ str(pi_hp_minus[0]))
        
    
        
    print("runtime = {}".format(time.time() - ti))