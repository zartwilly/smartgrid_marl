# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:41:27 2021

@author: jwehounou
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:15:18 2021

@author: jwehounou
"""


import os
import time
import execution_game_automate_4_all_t as autoExeGame4T
import fonctions_auxiliaires as fct_aux
import itertools as it


if __name__ == "__main__":
    ti = time.time()
    
    # _____                     scenarios --> debut                 __________
    prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
    prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
    prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
    prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
    scenario1 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                 (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                 (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                 (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
    
    prob_A_A = 0.8; prob_A_B1 = 0.2; prob_A_B2 = 0.0; prob_A_C = 0.0;
    prob_B1_A = 0.8; prob_B1_B1 = 0.2; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
    prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.2; prob_B2_C = 0.8;
    prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.2; prob_C_C = 0.8
    scenario2 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                 (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                 (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                 (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
    
    
    dico_scenario = {"scenario1": scenario1, 
                     "scenario2": scenario2}
    # _____                     scenarios --> fin                   __________
    
    # _____                     gamma_version --> debut             __________
    #2 #1 #3: gamma_i_min #4: square_root
    gamma_versions = [0,1,2,3,4]
    # _____                     gamma_version --> fin               __________
    
    # _____                    players by sets --> debut             __________
    # ---- initialization of variables for generating instances ----
    setA_m_players = 10; setB1_m_players = 3; 
    setB2_m_players = 5; setC_m_players = 8;                                    # 26 players
    
    # setA_m_players, setB1_m_players = 15, 5, 
    # setB1_m_players, setC_m_players = 5, 10                                  # debug_one_period
    # setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    # setA_m_players, setB1_m_players = 10, 3, 
    # setB1_m_players, setC_m_players = 3, 5 
    # _____                    players by sets --> fin               __________
    
    debug_all_periods = True #False #True #False #False #True
    debug_one_period = not debug_all_periods
    
    name_dir="tests"
    
    pi_hp_plus, pi_hp_minus = None, None
    t_periods, k_steps, NB_REPEAT_K_MAX = None, None, None
    learning_rates = None, None, None
    date_hhmm, Visualisation = None, None
    used_storage_det = True
    criteria_bf="Perf_t" # "In_sg_Out_sg"
    dbg_234_players = None
    arr_pl_M_T_vars_init = None
    path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
    
    if debug_all_periods:
        nb_periods = None
        # ---- new constances simu_DDMM_HHMM --- **** debug *****
        date_hhmm = "DDMM_HHMM"
        t_periods = 10 #4 #10 #30 #50 #30 #35 #55 #117 #15 #3
        k_steps = 250 #250 #250 #100 #250 #5000 #2000 #50 #250
        NB_REPEAT_K_MAX= 10 #3 #15 #30
        learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
        fct_aux.N_DECIMALS = 8
        
        pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
        fct_aux.PI_0_PLUS_INIT = 20 #4
        fct_aux.PI_0_MINUS_INIT = 10 #3
        
        algos = ["LRI1", "LRI2", "DETERMINIST"] \
                + fct_aux.ALGO_NAMES_BF
        algos = ["LRI1", "LRI2", "DETERMINIST"]
        
        dbg_234_players = False #True #False
        used_storage_det= True #False #True
        manual_debug = False #True #False #True
        Visualisation = True #False, True
    
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
        
        algos = ["LRI1", "LRI2", "DETERMINIST"] \
                + fct_aux.ALGO_NAMES_NASH \
                + fct_aux.ALGO_NAMES_BF
        algos = ["LRI1", "LRI2", "DETERMINIST"] 
        
        dbg_234_players = False #True #False
        used_storage_det= True #False #True
        manual_debug = True#False #True #False #True
        gamma_version = 1 # 2
        Visualisation = True #False, True
        
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
       
        dbg_234_players = False
        used_storage_det= True #False #True
        manual_debug = False #True
        Visualisation = True #False, True
       
       
    for scenario, gamma_version in it.product( list(dico_scenario.keys()), 
                                              gamma_versions):
        date_hhmm_new = "_".join([date_hhmm, scenario, 
                              "".join(["T", str(t_periods),
                                "".join(["gammaV", str(gamma_version)])])])
        
        used_instances = True
        arr_pl_M_T_vars_init \
            = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C(
                        setA_m_players, setB1_m_players, 
                        setB2_m_players, setC_m_players, 
                        t_periods, 
                        dico_scenario[scenario],
                        path_to_arr_pl_M_T, used_instances)
    
        autoExeGame4T.execute_algos_used_Generated_instances(
                    arr_pl_M_T_vars_init, 
                    name_dir = name_dir,
                    date_hhmm = date_hhmm_new,
                    k_steps = k_steps,
                    NB_REPEAT_K_MAX = NB_REPEAT_K_MAX,
                    algos = algos,
                    learning_rates = learning_rates,
                    pi_hp_plus = pi_hp_plus,
                    pi_hp_minus = pi_hp_minus,
                    gamma_version = gamma_version,
                    used_instances = used_instances,
                    used_storage_det = used_storage_det,
                    manual_debug = manual_debug, 
                    criteria_bf = criteria_bf, 
                    debug = False
                    )
    print("Procedural running time ={}".format(time.time()-ti))
    
        
    print("runtime = {}".format(time.time() - ti))