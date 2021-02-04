# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:48:40 2021

@author: jwehounou
"""

import os
import time
import visu_bkh_automate as viz

import fonctions_auxiliaires as fct_aux



if __name__ == "__main__":
    ti = time.time()
    
    viz.MULT_WIDTH = 2.5;
    viz.MULT_HEIGHT = 1.2;
    
    pi_hp_plus = 0.2*pow(10,-3); pi_hp_minus = 0.33
    viz.NAME_RESULT_SHOW_VARS = viz.NAME_RESULT_SHOW_VARS.format(pi_hp_plus, pi_hp_minus)
    
    debug = True #False#True
    
    t = 1
    
    ##____________  name simulation and k_steps ---> debut  ___________________
    if debug:
        name_simu = "simu_DDMM_HHMM"; k_steps_args = 250 #50
        name_simu = "simu_DDMM_HHMM"; k_steps_args = 50
    else:
        name_simu = "simu_2306_2206"; k_steps_args = 1000
    ## ____________  name simulation and k_steps ---> fin  ___________________
    
    ## ____________          turn_arr4d_2_df() ---> debut          ____________
    algos_4_no_learning = ["DETERMINIST","RD-DETERMINIST"] \
                            + fct_aux.ALGO_NAMES_BF \
                            + fct_aux.ALGO_NAMES_NASH
    algos_4_learning = ["LRI1", "LRI2"]
    tuple_paths, prices, algos, learning_rates \
        = viz.get_tuple_paths_of_arrays(name_simu=name_simu, 
                                    algos_4_no_learning=algos_4_no_learning)
        
    df_arr_M_T_Ks, \
    df_ben_cst_M_T_K, \
    df_b0_c0_pisg_pi0_T_K, \
    df_B_C_BB_CC_RU_M \
        = viz.get_array_turn_df_for_t(tuple_paths, t, k_steps_args, 
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
    viz.group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, t, name_dir, 
                        viz.NAME_RESULT_SHOW_VARS)
    
     ## ____________          plot figures ---> fin            ____________
    
    print("runtime = {}".format(time.time() - ti))