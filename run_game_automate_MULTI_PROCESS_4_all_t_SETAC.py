# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:56:29 2021

@author: jwehounou
"""

import os
import time
import multiprocessing as mp
import execution_game_automate_MULTI_PROCESS_4_all_t as autoExeGameMulProc4T
import fonctions_auxiliaires as fct_aux
import visu_bkh_automate_v1 as autoVizGameV1


if __name__ == "__main__":
    ti = time.time()
    debug = False
    
    # _____                     scenarios --> debut                 __________
    prob_A_A = 0.6; prob_A_C = 0.4;
    prob_C_A = 0.4; prob_C_C = 0.6;
    scenario3 = [(prob_A_A, prob_A_C), 
                (prob_C_A, prob_C_C)]
    
    
    dico_scenario = {"scenario3": scenario3}
    # _____                     scenarios --> fin                   __________
    
    
    debug_all_periods = True #False #True #False #False #True
    debug_one_period = not debug_all_periods
    
    name_dir="tests"
    
    pi_hp_plus, pi_hp_minus, tuple_pi_hp_plus_minus = None, None, None
    setA_m_players, setC_m_players = None, None
    t_periods, k_steps, NB_REPEAT_K_MAX = None, None, None
    learning_rates = None, None, None
    date_hhmm, Visualisation = None, None
    used_storage_det=True
    gamma_versions = [0,1,2,3,4] #0, 1, 2, 3, 4
    criteria_bf="Perf_t" # "In_sg_Out_sg"
    dbg_234_players = None
    arr_pl_M_T_vars_init = None
    path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
    
    if debug_all_periods:
        nb_periods = None
        # ---- new constances simu_DDMM_HHMM --- **** debug *****
        date_hhmm = "DDMM_HHMM"
        t_periods = 50 #10 #30 #50 #30 #35 #55 #117 #15 #3
        k_steps = 250 #5 #100 #250 #5000 #2000 #50 #250
        NB_REPEAT_K_MAX= 10 #3 #15 #30
        learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
        fct_aux.N_DECIMALS = 8
        
        pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
        fct_aux.PI_0_PLUS_INIT = 10 #4
        fct_aux.PI_0_MINUS_INIT = 20 #3
        tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
        
        # algos = ["LRI1", "LRI2", "DETERMINIST"] \
        #         + fct_aux.ALGO_NAMES_BF
        algos = ["LRI1", "LRI2", "DETERMINIST"]
        
        dbg_234_players = False #True #False
        used_storage_det= True #False #True
        manual_debug = False #True #False #True
        gamma_versions = [0,1,2,3,4]
        Visualisation = True #False, True
        
        scenario = "scenario3"

        # ---- initialization of variables for generating instances ----
        setA_m_players = 10; setC_m_players = 10;                                  # 20 players
        
        if dbg_234_players:
            t_periods = 2
            setA_m_players, setB_m_players, setC_m_players = 1, 1, 1
         
        used_instances = True
        
    elif debug_one_period:
        nb_periods = 0
        # ---- new constances simu_DDMM_HHMM  ONE PERIOD t = 0 --- **** debug *****
        date_hhmm="DDMM_HHMM"
        t_periods = 1 #50 #30 #35 #55 #117 #15 #3
        k_steps = 250 #5000 #2000 #50 #250
        NB_REPEAT_K_MAX= 10 #3 #15 #30
        learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
        fct_aux.N_DECIMALS = 8
        
        pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
        fct_aux.PI_0_PLUS_INIT = 10 #4
        fct_aux.PI_0_MINUS_INIT = 20 #3
        tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
        
        algos = ["LRI1", "LRI2", "DETERMINIST"] \
                + fct_aux.ALGO_NAMES_NASH \
                + fct_aux.ALGO_NAMES_BF
        algos = ["LRI1", "LRI2", "DETERMINIST"] 
        
        dbg_234_players = False #True #False
        used_storage_det= True #False #True
        manual_debug = True#False #True #False #True
        gamma_version = [3] # 2
        Visualisation = True #False, True
        
        scenario = "scenario3"
        
        # ---- initialization of variables for generating instances ----
        setA_m_players, setC_m_players = 15, 10,
        
        used_instances = True
    
        
    else:
        nb_periods = None
        # ---- new constances simu_2306_2206 --- **** debug ***** 
        date_hhmm="2306_2206"
        t_periods = 110
        k_steps = 1000
        NB_REPEAT_K_MAX = 15 #30
        learning_rates = [0.1] #[0.01] #[0.0001]
       
        pi_hp_plus = [0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [0.33] #[15, 5]
        fct_aux.PI_0_PLUS_INIT = 10 #4
        fct_aux.PI_0_MINUS_INIT = 20 #3
        tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
       
        dbg_234_players = False
        used_storage_det= True #False #True
        manual_debug = False #True
        gamma_version = 1 # 2
        Visualisation = True #False, True
        
        scenario = "scenario3"
       
        # ---- initialization of variables for generating instances ----
        setA_m_players, setB_m_players, setC_m_players = 20, 12, 6
     
    # ---- generation of data into array ----
    arr_pl_M_T_vars_init \
            = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC(
                        setA_m_players, setC_m_players, 
                        t_periods, 
                        dico_scenario[scenario],
                        path_to_arr_pl_M_T, used_instances)
       
    dico_params = {
        "arr_pl_M_T_vars_init": arr_pl_M_T_vars_init, 
        "scenario": scenario,
        "name_dir": name_dir,
        "date_hhmm": date_hhmm,
        "t_periods": t_periods,
        "k_steps": k_steps,
        "NB_REPEAT_K_MAX": NB_REPEAT_K_MAX,
        "algos": algos,
        "learning_rates": learning_rates,
        "tuple_pi_hp_plus_minus": tuple_pi_hp_plus_minus,
        "gamma_versions": gamma_versions,
        "used_instances": used_instances,
        "used_storage_det": used_storage_det,
        "manual_debug": manual_debug, 
        "criteria_bf": criteria_bf, 
        "debug": debug
        }
    params = autoExeGameMulProc4T.define_parameters_multi_gammaV(dico_params)
    print("define parameters finished")
    
    # multi processing execution
    p = mp.Pool(mp.cpu_count()-1)
    p.starmap(
        autoExeGameMulProc4T\
            .execute_one_algo_used_Generated_instances, 
        params)
    # multi processing execution
    
    print("Multi process running time ={}".format(time.time()-ti))
    
    
    print("runtime = {}".format(time.time() - ti))