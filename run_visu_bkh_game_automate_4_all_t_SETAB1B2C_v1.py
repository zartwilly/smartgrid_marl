# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:43:50 2021

@author: jwehounou
"""
import os
import time
import visu_bkh_automate_v1 as autoVizGameV1
import fonctions_auxiliaires as fct_aux


if __name__ == "__main__":
    ti = time.time()
    nb_periods = None 
    debug_one_period = False 
    if debug_one_period:
        nb_periods = 0
    else:
        nb_periods = None
        
    autoVizGameV1.MULT_WIDTH = 2.0 #2.25;
    autoVizGameV1.MULT_HEIGHT = 0.7 
    # fct_aux.N_DECIMALS = 7;
    pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
    pi_hp_minus = [20]
    
    
    name_simu = "simu_DDMM_HHMM_scenario1_T10gammaV3"; k_steps_args = 250 #250
    name_simu = "simu_DDMM_HHMM_scenario2_T10gammaV4"; k_steps_args = 50 #250
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
            tuple_paths, t=nb_periods, k_steps_args=k_steps_args, 
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
                    algos_4_learning,
                    path_2_best_learning_steps, 
                    autoVizGameV1.NAME_RESULT_SHOW_VARS)
    ## ____________          plot figures ---> fin            ____________
    
    
    print("runtime = {}".format(time.time() - ti))