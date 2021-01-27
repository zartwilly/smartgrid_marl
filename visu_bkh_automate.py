# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:59:50 2021

@author: jwehounou
"""

import os
import time
import numpy as np
import pandas as pd
import itertools as it

import fonctions_auxiliaires as fct_aux

from bokeh.models.tools import HoverTool, PanTool, BoxZoomTool, WheelZoomTool 
from bokeh.models.tools import RedoTool, ResetTool, SaveTool, UndoTool
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row, column
from bokeh.models import Panel, Tabs, Legend
from bokeh.transform import factor_cmap
# from bokeh.models import Select
# from bokeh.io import curdoc
# from bokeh.plotting import reset_output
# from bokeh.models.widgets import Slider


# Importing a pallette
from bokeh.palettes import Category20
#from bokeh.palettes import Spectral5 
#from bokeh.palettes import Viridis256


from bokeh.models.annotations import Title

#------------------------------------------------------------------------------
#                   definitions of constants
#------------------------------------------------------------------------------
WIDTH = 500;
HEIGHT = 500;
MULT_WIDTH = 2.5;
MULT_HEIGHT = 3.5;

MARKERS = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", 
               "P", "*", "h", "H", "+", "x", "X", "D", "d"]
COLORS = Category20[19] #["red", "yellow", "blue", "green", "rosybrown","darkorange", "fuchsia", "grey", ]

TOOLS = [
            PanTool(),
            BoxZoomTool(),
            WheelZoomTool(),
            UndoTool(),
            RedoTool(),
            ResetTool(),
            SaveTool(),
            HoverTool(tooltips=[
                ("Price", "$y"),
                ("Time", "$x")
                ])
            ]

NAME_RESULT_SHOW_VARS = "resultat_show_variables_pi_plus_{}_pi_minus_{}.html"

#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------

# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> debut
# _____________________________________________________________________________
def get_local_storage_variables(path_to_variable):
    """
    obtain the content of variables stored locally .

    Returns
    -------
     arr_pls_M_T, RUs, B0s, C0s, BENs, CSTs, pi_sg_plus_s, pi_sg_minus_s.
    
    arr_pls_M_T: array of players with a shape M_PLAYERS*T_PERIODS*INDEX_ATTRS
    arr_T_nsteps_vars : array of players with a shape 
                        M_PLAYERS*T_PERIODS*NSTEPS*vars_nstep
                        avec len(vars_nstep)=20
    RUs: array of (M_PLAYERS,)
    BENs: array of M_PLAYERS*T_PERIODS
    CSTs: array of M_PLAYERS*T_PERIODS
    B0s: array of (T_PERIODS,)
    C0s: array of (T_PERIODS,)
    pi_sg_plus_s: array of (T_PERIODS,)
    pi_sg_minus_s: array of (T_PERIODS,)

    pi_hp_plus_s: array of (T_PERIODS,)
    pi_hp_minus_s: array of (T_PERIODS,)
    """

    arr_pl_M_T_K_vars = np.load(os.path.join(path_to_variable, 
                                             "arr_pl_M_T_K_vars.npy"),
                          allow_pickle=True)
    b0_s_T_K = np.load(os.path.join(path_to_variable, "b0_s_T_K.npy"),
                          allow_pickle=True)
    c0_s_T_K = np.load(os.path.join(path_to_variable, "c0_s_T_K.npy"),
                          allow_pickle=True)
    B_is_M = np.load(os.path.join(path_to_variable, "B_is_M.npy"),
                          allow_pickle=True)
    C_is_M = np.load(os.path.join(path_to_variable, "C_is_M.npy"),
                          allow_pickle=True)
    BENs_M_T_K = np.load(os.path.join(path_to_variable, "BENs_M_T_K.npy"),
                          allow_pickle=True)
    CSTs_M_T_K = np.load(os.path.join(path_to_variable, "CSTs_M_T_K.npy"),
                          allow_pickle=True)
    BB_is_M = np.load(os.path.join(path_to_variable, "BB_is_M.npy"),
                          allow_pickle=True)
    CC_is_M = np.load(os.path.join(path_to_variable, "CC_is_M.npy"),
                          allow_pickle=True)
    RU_is_M = np.load(os.path.join(path_to_variable, "RU_is_M.npy"),
                          allow_pickle=True)
    pi_sg_plus_T_K = np.load(os.path.join(path_to_variable, "pi_sg_plus_T_K.npy"),
                          allow_pickle=True)
    pi_sg_minus_T_K = np.load(os.path.join(path_to_variable, "pi_sg_minus_T_K.npy"),
                          allow_pickle=True)
    pi_0_plus_T_K = np.load(os.path.join(path_to_variable, "pi_0_plus_T_K.npy"),
                          allow_pickle=True)
    pi_0_minus_T_K = np.load(os.path.join(path_to_variable, "pi_0_minus_T_K.npy"),
                          allow_pickle=True)
    pi_hp_plus_s = np.load(os.path.join(path_to_variable, "pi_hp_plus_s.npy"),
                          allow_pickle=True)
    pi_hp_minus_s = np.load(os.path.join(path_to_variable, "pi_hp_minus_s.npy"),
                          allow_pickle=True)
    
    return arr_pl_M_T_K_vars, \
            b0_s_T_K, c0_s_T_K, \
            B_is_M, C_is_M, \
            BENs_M_T_K, CSTs_M_T_K, \
            BB_is_M, CC_is_M, RU_is_M, \
            pi_sg_plus_T_K, pi_sg_minus_T_K, \
            pi_0_plus_T_K, pi_0_minus_T_K, \
            pi_hp_plus_s, pi_hp_minus_s
            
 
def get_tuple_paths_of_arrays(name_dir="tests", name_simu="simu_1811_1754",
                prices=None, algos=None, learning_rates=None, 
                algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                     "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE", 
                                       "MIDDLE-BRUTE-FORCE"], 
                ext=".npy", 
                exclude_html_files=[NAME_RESULT_SHOW_VARS,"html"]):
    
    tuple_paths = []
    rep_dir_simu = os.path.join(name_dir, name_simu)
    
    prices_new = None
    if prices is None:
        prices_new = os.listdir(rep_dir_simu)
    else:
        prices_new = prices
    prices_new = [x for x in prices_new 
                     if x.split('.')[-1] not in exclude_html_files]
    for price in prices_new:
        path_price = os.path.join(name_dir, name_simu, price)
        algos_new = None
        if algos is None:
            algos_new = os.listdir(path_price)
        else:
            algos_new = algos
        for algo in algos_new:
            path_price_algo = os.path.join(name_dir, name_simu, price, algo)
            if algo not in algos_4_no_learning:
                learning_rates_new = None
                if learning_rates is None:
                    learning_rates_new = os.listdir(path_price_algo)
                else:
                    learning_rates_new = learning_rates
                for learning_rate in learning_rates_new:
                    path_price_algo_learning = \
                        os.path.join(name_dir, name_simu, price, 
                                     algo, learning_rate)
                    tuple_paths.append( (name_dir, name_simu, price, 
                                         algo, learning_rate) )
            else:
                tuple_paths.append( (name_dir, name_simu, price, algo) )
                
    return tuple_paths, prices_new, algos_new, learning_rates_new

def get_array_turn_df_for_t(tuple_paths, t=1, k_steps_args=250, 
                            algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                                 "BEST-BRUTE-FORCE",
                                                 "BAD-BRUTE-FORCE", 
                                                 "MIDDLE-BRUTE-FORCE"], 
                            algos_4_learning=["LRI1", "LRI2"]):
    df_arr_M_T_Ks = []
    df_b0_c0_pisg_pi0_T_K = []
    df_B_C_BB_CC_RU_M = []
    df_ben_cst_M_T_K = []
    for tuple_path in tuple_paths:
        path_to_variable = os.path.join(*tuple_path)
        
        arr_pl_M_T_K_vars, \
        b0_s_T_K, c0_s_T_K, \
        B_is_M, C_is_M, \
        BENs_M_T_K, CSTs_M_T_K, \
        BB_is_M, CC_is_M, RU_is_M, \
        pi_sg_plus_T, pi_sg_minus_T, \
        pi_0_plus_T, pi_0_minus_T, \
        pi_hp_plus_s, pi_hp_minus_s \
            = get_local_storage_variables(path_to_variable)
        
        price = tuple_path[2].split("_")[3]+"_"+tuple_path[2].split("_")[-1]
        algo = tuple_path[3];
        rate = tuple_path[4] if algo in algos_4_learning else 0
        
        m_players = arr_pl_M_T_K_vars.shape[0]
        t_periods = arr_pl_M_T_K_vars.shape[1]
        k_steps = arr_pl_M_T_K_vars.shape[2] if arr_pl_M_T_K_vars.shape == 4 \
                                             else k_steps_args
                                             
        t_periods = None; tu_mtk = None; tu_tk = None; tu_m = None
        if t is None:
            t_periods = arr_pl_M_T_K_vars.shape[1]
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     range(0, t_periods), 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    range(0, t_periods), 
                                    range(0, k_steps)))
        elif type(t) is list:
            t_periods = t
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    t_periods, 
                                    range(0, k_steps)))
        elif type(t) is int:
            t_periods = [t]
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    t_periods, 
                                    range(0, k_steps)))
                      
        tu_m = list(it.product([algo], [rate], [price], range(0, m_players)))
                    
        variables = list(fct_aux.AUTOMATE_INDEX_ATTRS.keys())
        
        if algo in algos_4_learning:
            arr_pl_M_T_K_vars_t = arr_pl_M_T_K_vars[:, t_periods, :, :]
            ## process of arr_pl_M_T_K_vars 
            arr_pl_M_T_K_vars_2D = arr_pl_M_T_K_vars_t.reshape(
                                        -1, 
                                        arr_pl_M_T_K_vars.shape[3])
            df_lri_x = pd.DataFrame(data=arr_pl_M_T_K_vars_2D, 
                                    index=tu_mtk, 
                                    columns=variables)
            
            df_arr_M_T_Ks.append(df_lri_x)
            
            ## process of df_b0_c0_pisg_pi0_T_K
            b0_s_T_K_1D = b0_s_T_K[t_periods,:].reshape(-1)
            c0_s_T_K_1D = c0_s_T_K[t_periods,:].reshape(-1)
            pi_0_minus_T_K_1D = pi_0_minus_T[t_periods].reshape(-1)
            pi_0_plus_T_K_1D = pi_0_plus_T[t_periods].reshape(-1)
            pi_sg_minus_T_K_1D = pi_sg_minus_T[t_periods].reshape(-1)
            pi_sg_plus_T_K_1D = pi_sg_plus_T[t_periods].reshape(-1)
            df_b0_c0_pisg_pi0_T_K_lri \
                = pd.DataFrame({
                        "b0":b0_s_T_K_1D, "c0":c0_s_T_K_1D, 
                        "pi_0_minus":pi_0_minus_T_K_1D, 
                        "pi_0_plus":pi_0_plus_T_K_1D, 
                        "pi_sg_minus":pi_sg_minus_T_K_1D, 
                        "pi_sg_plus":pi_sg_plus_T_K_1D}, 
                    index=tu_tk)
            df_b0_c0_pisg_pi0_T_K.append(df_b0_c0_pisg_pi0_T_K_lri)
            
            ## process of df_ben_cst_M_T_K
            BENs_M_T_K_1D = BENs_M_T_K[:,t_periods,:].reshape(-1)
            CSTs_M_T_K_1D = CSTs_M_T_K[:,t_periods,:].reshape(-1)
            df_ben_cst_M_T_K_lri = pd.DataFrame({
                'ben':BENs_M_T_K_1D, 'cst':CSTs_M_T_K_1D}, index=tu_mtk)
            df_ben_cst_M_T_K.append(df_ben_cst_M_T_K_lri)
            ## process of df_B_C_BB_CC_RU_M
            df_B_C_BB_CC_RU_M_lri \
                = pd.DataFrame({
                        "B":B_is_M, "C":C_is_M, 
                        "BB":BB_is_M, "CC":CC_is_M, "RU":RU_is_M}, 
                    index=tu_m)
            df_B_C_BB_CC_RU_M.append(df_B_C_BB_CC_RU_M_lri)
            ## process of 
            ## process of
            
        elif algo in algos_4_no_learning:
            arr_pl_M_T_K_vars_t = arr_pl_M_T_K_vars[:, t_periods, :]
            ## process of arr_pl_M_T_K_vars 
            # turn array from 3D to 4D
            arrs = []
            for k in range(0, k_steps):
                arrs.append(list(arr_pl_M_T_K_vars_t))
            arrs = np.array(arrs, dtype=object)
            arrs = np.transpose(arrs, [1,2,0,3])
            arr_pl_M_T_K_vars_4D = np.zeros((arrs.shape[0],
                                              arrs.shape[1],
                                              arrs.shape[2],
                                              arrs.shape[3]), 
                                            dtype=object)
            #print("len: variables={}, {}".format(len(variables),variables))
            # arr_pl_M_T_K_vars_4D[:,:,:,:-1] = arrs
            # arr_pl_M_T_K_vars_4D[:,:,:, len(variables)-1] = 0.5
            
            # arr_pl_M_T_K_vars_4D[:,:,:,:-nb_vars_2_add+2] = arrs
            # ind_prob_mode_state_i = 16
            # ind_u_i = 17
            # ind_bg_i = 18
            # ind_non_playing_players = 19
            # arr_pl_M_T_K_vars_4D[:,:,:, ind_prob_mode_state_i] = 0.5
            # arr_pl_M_T_K_vars_4D[:,:,:, ind_u_i] = 0
            # arr_pl_M_T_K_vars_4D[:,:,:, ind_bg_i] = 0
            # arr_pl_M_T_K_vars_4D[:,:,:, ind_non_playing_players] = 1
            
            # # turn in 2D
            arr_pl_M_T_K_vars_2D = arr_pl_M_T_K_vars_4D.reshape(
                                        -1, 
                                        arr_pl_M_T_K_vars_4D.shape[3])
            # turn arr_2D to df_{RD}DET 
            # variables[:-3] = ["Si_minus","Si_plus",
            #        "added column so that columns df_lri and df_det are identicals"]
            df_rd_det = pd.DataFrame(data=arr_pl_M_T_K_vars_2D, 
                                     index=tu_mtk, columns=variables)
            
            df_arr_M_T_Ks.append(df_rd_det)
            
            ## process of df_b0_c0_pisg_pi0_T_K
            # turn array from 1D to 2D
            arrs_b0_2D, arrs_c0_2D = [], []
            arrs_pi_0_plus_2D, arrs_pi_0_minus_2D = [], []
            arrs_pi_sg_plus_2D, arrs_pi_sg_minus_2D = [], []
            # print("shape: b0_s_T_K={}, pi_0_minus_T_K={}".format(
            #     b0_s_T_K.shape, pi_0_minus_T_K.shape))
            for k in range(0, k_steps):
                # print("type: b0_s_T_K={}, b0_s_T_K={}; bool={}".format(type(b0_s_T_K), 
                #      b0_s_T_K.shape, b0_s_T_K.shape == ()))
                if b0_s_T_K.shape == ():
                    arrs_b0_2D.append([b0_s_T_K])
                else:
                    arrs_b0_2D.append(list(b0_s_T_K[t_periods]))
                if c0_s_T_K.shape == ():
                    arrs_c0_2D.append([c0_s_T_K])
                else:
                    arrs_c0_2D.append(list(c0_s_T_K[t_periods]))
                if pi_0_plus_T.shape == ():
                    arrs_pi_0_plus_2D.append([pi_0_plus_T])
                else:
                    arrs_pi_0_plus_2D.append(list(pi_0_plus_T[t_periods]))
                if pi_0_minus_T.shape == ():
                    arrs_pi_0_minus_2D.append([pi_0_minus_T])
                else:
                    arrs_pi_0_minus_2D.append(list(pi_0_minus_T[t_periods]))
                if pi_sg_plus_T.shape == ():
                    arrs_pi_sg_plus_2D.append([pi_sg_plus_T])
                else:
                    arrs_pi_sg_plus_2D.append(list(pi_sg_plus_T[t_periods]))
                if pi_sg_minus_T.shape == ():
                    arrs_pi_sg_minus_2D.append([pi_sg_minus_T])
                else:
                    arrs_pi_sg_minus_2D.append(list(pi_sg_minus_T[t_periods]))
                 #arrs_c0_2D.append(list(c0_s_T_K))
                 #arrs_pi_0_plus_2D.append(list(pi_0_plus_T_K))
                 #arrs_pi_0_minus_2D.append(list(pi_0_minus_T_K))
                 #arrs_pi_sg_plus_2D.append(list(pi_sg_plus_T_K))
                 #arrs_pi_sg_minus_2D.append(list(pi_sg_minus_T_K))
            arrs_b0_2D = np.array(arrs_b0_2D, dtype=object)
            arrs_c0_2D = np.array(arrs_c0_2D, dtype=object)
            arrs_pi_0_plus_2D = np.array(arrs_pi_0_plus_2D, dtype=object)
            arrs_pi_0_minus_2D = np.array(arrs_pi_0_minus_2D, dtype=object)
            arrs_pi_sg_plus_2D = np.array(arrs_pi_sg_plus_2D, dtype=object)
            arrs_pi_sg_minus_2D = np.array(arrs_pi_sg_minus_2D, dtype=object)
            arrs_b0_2D = np.transpose(arrs_b0_2D, [1,0])
            arrs_c0_2D = np.transpose(arrs_c0_2D, [1,0])
            arrs_pi_0_plus_2D = np.transpose(arrs_pi_0_plus_2D, [1,0])
            arrs_pi_0_minus_2D = np.transpose(arrs_pi_0_minus_2D, [1,0])
            arrs_pi_sg_plus_2D = np.transpose(arrs_pi_sg_plus_2D, [1,0])
            arrs_pi_sg_minus_2D = np.transpose(arrs_pi_sg_minus_2D, [1,0])
            # turn array from 2D to 1D
            arrs_b0_1D = arrs_b0_2D.reshape(-1)
            arrs_c0_1D = arrs_c0_2D.reshape(-1)
            arrs_pi_0_minus_1D = arrs_pi_0_minus_2D.reshape(-1)
            arrs_pi_0_plus_1D = arrs_pi_0_plus_2D.reshape(-1)
            arrs_pi_sg_minus_1D = arrs_pi_sg_minus_2D.reshape(-1)
            arrs_pi_sg_plus_1D = arrs_pi_sg_plus_2D.reshape(-1)
            # create dataframe
            df_b0_c0_pisg_pi0_T_K_det \
                = pd.DataFrame({
                    "b0":arrs_b0_1D, 
                    "c0":arrs_c0_1D, 
                    "pi_0_minus":arrs_pi_0_minus_1D, 
                    "pi_0_plus":arrs_pi_0_plus_1D, 
                    "pi_sg_minus":arrs_pi_sg_minus_1D, 
                    "pi_sg_plus":arrs_pi_sg_plus_1D}, index=tu_tk)
            df_b0_c0_pisg_pi0_T_K.append(df_b0_c0_pisg_pi0_T_K_det) 

            ## process of df_ben_cst_M_T_K
            # turn array from 2D to 3D
            arrs_ben_3D, arrs_cst_3D = [], []
            for k in range(0, k_steps):
                 arrs_ben_3D.append(list(BENs_M_T_K[:,t_periods]))
                 arrs_cst_3D.append(list(CSTs_M_T_K[:,t_periods]))
            arrs_ben_3D = np.array(arrs_ben_3D, dtype=object)
            arrs_cst_3D = np.array(arrs_cst_3D, dtype=object)
            arrs_ben_3D = np.transpose(arrs_ben_3D, [1,2,0])
            arrs_cst_3D = np.transpose(arrs_cst_3D, [1,2,0])
    
            # turn array from 3D to 1D
            BENs_M_T_K_1D = arrs_ben_3D.reshape(-1)
            CSTs_M_T_K_1D = arrs_cst_3D.reshape(-1)
            #create dataframe
            df_ben = pd.DataFrame(data=BENs_M_T_K_1D, 
                              index=tu_mtk, columns=['ben'])
            df_cst = pd.DataFrame(data=CSTs_M_T_K_1D, 
                              index=tu_mtk, columns=['cst'])
            df_ben_cst_M_T_K_det = pd.concat([df_ben, df_cst], axis=1)

            df_ben_cst_M_T_K.append(df_ben_cst_M_T_K_det)
            
            
            ## process of df_B_C_BB_CC_RU_M
            df_B_C_BB_CC_RU_M_det = pd.DataFrame({
                "B":B_is_M, "C":C_is_M, 
                "BB":BB_is_M,"CC":CC_is_M,"RU":RU_is_M,}, index=tu_m)
            df_B_C_BB_CC_RU_M.append(df_B_C_BB_CC_RU_M_det)
            ## process of 
            ## process of 
            
    df_arr_M_T_Ks = pd.concat(df_arr_M_T_Ks, axis=0)
    df_ben_cst_M_T_K = pd.concat(df_ben_cst_M_T_K, axis=0)
    df_b0_c0_pisg_pi0_T_K = pd.concat(df_b0_c0_pisg_pi0_T_K, axis=0)
    df_B_C_BB_CC_RU_M = pd.concat(df_B_C_BB_CC_RU_M, axis=0)
    
    # insert index as columns of dataframes
    ###  df_arr_M_T_Ks
    columns_df = df_arr_M_T_Ks.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i","t","k"]
    indices = list(df_arr_M_T_Ks.index)
    df_ind = pd.DataFrame(indices,columns=columns_ind)
    df_arr_M_T_Ks = pd.concat([df_ind.reset_index(), 
                                df_arr_M_T_Ks.reset_index()],
                              axis=1, ignore_index=True)
    df_arr_M_T_Ks.drop(df_arr_M_T_Ks.columns[[0]], axis=1, inplace=True)
    df_arr_M_T_Ks.columns = columns_ind+["old_index"]+columns_df
    df_arr_M_T_Ks.pop("old_index")
    ###  df_ben_cst_M_T_K
    columns_df = df_ben_cst_M_T_K.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i","t","k"]
    indices = list(df_ben_cst_M_T_K.index)
    df_ind = pd.DataFrame(indices,columns=columns_ind)
    df_ben_cst_M_T_K = pd.concat([df_ind.reset_index(), 
                                df_ben_cst_M_T_K.reset_index()],
                              axis=1, ignore_index=True)
    df_ben_cst_M_T_K.drop(df_ben_cst_M_T_K.columns[[0]], axis=1, inplace=True)
    df_ben_cst_M_T_K.columns = columns_ind+["old_index"]+columns_df
    df_ben_cst_M_T_K.pop("old_index")
    df_ben_cst_M_T_K["state_i"] = df_arr_M_T_Ks["state_i"]
    ###  df_b0_c0_pisg_pi0_T_K
    columns_df = df_b0_c0_pisg_pi0_T_K.columns.to_list()
    columns_ind = ["algo","rate","prices","t","k"]
    indices = list(df_b0_c0_pisg_pi0_T_K.index)
    df_ind = pd.DataFrame(indices, columns=columns_ind)
    df_b0_c0_pisg_pi0_T_K = pd.concat([df_ind.reset_index(), 
                                        df_b0_c0_pisg_pi0_T_K.reset_index()],
                                        axis=1, ignore_index=True)
    df_b0_c0_pisg_pi0_T_K.drop(df_b0_c0_pisg_pi0_T_K.columns[[0]], 
                               axis=1, inplace=True)
    df_b0_c0_pisg_pi0_T_K.columns = columns_ind+["old_index"]+columns_df
    df_b0_c0_pisg_pi0_T_K.pop("old_index")
    ###  df_B_C_BB_CC_RU_M
    columns_df = df_B_C_BB_CC_RU_M.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i"]
    indices = list(df_B_C_BB_CC_RU_M.index)
    df_ind = pd.DataFrame(indices, columns=columns_ind)
    df_B_C_BB_CC_RU_M = pd.concat([df_ind.reset_index(), 
                                        df_B_C_BB_CC_RU_M.reset_index()],
                                        axis=1, ignore_index=True)
    df_B_C_BB_CC_RU_M.drop(df_B_C_BB_CC_RU_M.columns[[0]], 
                               axis=1, inplace=True)
    df_B_C_BB_CC_RU_M.columns = columns_ind+["old_index"]+columns_df
    df_B_C_BB_CC_RU_M.pop("old_index")
    
    return df_arr_M_T_Ks, df_ben_cst_M_T_K, \
            df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M
# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> fin
# _____________________________________________________________________________ 


# _____________________________________________________________________________ 
#
#                       histogramme de strategies  ---> debut
# _____________________________________________________________________________ 
def plot_histo_strategy_onestate_for_scenario(df_ra_pri_al, 
                                 rate, price, state_i, 
                                 algo, t):
    """
    INUTILE car suppression des scenarios
    """
    
    df_ra_pri_al["mode_i"][(df_ra_pri_al.mode_i == '')] = 'UNDEF'
    
    cols = ["mode_i","pl_i"]
    df_mode = df_ra_pri_al\
                .groupby(cols)[["scenario"]].count()
    df_mode.rename(columns={"scenario":"nb_players"}, inplace=True)
    df_mode = df_mode.reset_index()
    df_mode["pl_i"] = df_mode["pl_i"].astype(str)
    
    x = list(map(tuple,list(df_mode[cols].values)))
    nb_players = list(df_mode["nb_players"])
                     
    TOOLS[7] = HoverTool(tooltips=[
                            ("nb_players", "@nb_players")
                            ]
                        )
    px= figure(x_range=FactorRange(*x), plot_height=250, 
               title="number of players by mode, t={}, {} ({}, rate={}, price={})".format(
                  t, state_i, algo, rate, price),
                toolbar_location=None, tools=TOOLS)

    data = dict(x=x, nb_players=nb_players)
    
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='nb_players', width=0.9, source=source, 
            fill_color=factor_cmap('x', palette=Category20[10], 
                                   factors=list(df_mode["pl_i"].unique()), 
                                   start=1, end=2))
    
    px.y_range.start = 0
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    
    return px

def plot_histo_strategies(df_arr_M_T_Ks, t):
    """
    plot the strategies' histogram  for each state

    x-axis : strategies
    y-axis : number of cases having one strategy
    
    Returns
    -------
    None.

    """
    rates = df_arr_M_T_Ks["rate"].unique(); rates = rates[rates!=0]
    prices = df_arr_M_T_Ks["prices"].unique()
    states = df_arr_M_T_Ks["state_i"].unique()
    algos = df_arr_M_T_Ks["algo"].unique()
    
    dico_pxs = dict()
    cpt = 0
    for rate, price, state_i, algo\
        in it.product(rates, prices, states, algos):
        
        mask_ra_pri = ((df_arr_M_T_Ks.rate == rate) 
                                 | (df_arr_M_T_Ks.rate == 0)) \
                            & (df_arr_M_T_Ks.prices == price) \
                            & (df_arr_M_T_Ks.t == t) \
                            & (df_arr_M_T_Ks.state_i == state_i) \
                            & (df_arr_M_T_Ks.algo == algo)    
        df_ra_pri_al = df_arr_M_T_Ks[mask_ra_pri].copy()
        
        if df_ra_pri_al.shape[0] != 0:
            px_st = plot_histo_strategy_onestate_for_scenario(
                                    df_ra_pri_al, 
                                    rate, price, state_i, 
                                    algo, t)
            px_st.legend.click_policy="hide"
    
            if (rate, price, state_i, algo) \
                not in dico_pxs.keys():
                dico_pxs[(rate, price, state_i, algo)] \
                    = [px_st]
            else:
                dico_pxs[(rate, price, state_i, algo)] \
                    .append(px_st)
            cpt += 1
        
    # aggregate by state_i i.e each state_i is on new column.
    col_mode_S1S2_nbplayers = []
    for key, px_st in dico_pxs.items():
        row_pxs = row(px_st)
        col_mode_S1S2_nbplayers.append(row_pxs)
    col_mode_S1S2_nbplayers=column(children=col_mode_S1S2_nbplayers, 
                                   sizing_mode='stretch_both')  
    #show(col_mode_S1S2_nbplayers)
    
    return col_mode_S1S2_nbplayers

# _____________________________________________________________________________ 
#
#                       histogramme de strategies  ---> fin
# _____________________________________________________________________________ 

# _____________________________________________________________________________ 
#
#         moyenne de la grande proba dune mode pour chaque state  ---> debut
# _____________________________________________________________________________ 
def plot_max_proba_mode_onestate(df_ra_pri_st, 
                                rate, price, state_i, t, 
                                algos):
    
    # algos = ["LRI1","LRI2"]
    
    tup_legends = [] 
    
    px = figure(plot_height = int(HEIGHT*MULT_HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    
    algo_df = set(df_ra_pri_st["algo"].unique())
    algos = set(algos).intersection(algo_df)
    
    for algo in algos:
        df_al = df_ra_pri_st[(df_ra_pri_st.algo == algo)]
        
        title = "mean of max proba for each mode at {}, t={} (rate={}, price={})".format(
                state_i, t, rate, price)
        xlabel = "k step of learning" 
        ylabel_moyS1 = "moy_S1"; 
        ylabel_moyS2 = "moy_S2";
        ylabel_moyMaxS12 = "moy_max_S12"
        label_moyS1 = "moy_S1_{}".format(algo)
        label_moyS2 = "moy_S2_{}".format(algo)
        label_moyMaxS12 = "moy_max_S12_{}".format(algo)
        
        px.title.text = title
        px.xaxis.axis_label = xlabel
        px.yaxis.axis_label = "moy mode"
        TOOLS[7] = HoverTool(tooltips=[
                            ("algo", "@algo"),
                            ("k", "@k"),
                            ("moy_S1", "@moy_S1"),
                            ("moy_S2", "@moy_S2"),
                            ("moy_max_S12", "@moy_max_S12"),
                            ("S1", "@S1"),
                            ("S2", "@S2"),
                            ]
                        )
        px.tools = TOOLS
        
        S1, S2 = None, None
        if state_i == "state1":
            S1 = fct_aux.STATE1_STRATS[0]
            S2 = fct_aux.STATE1_STRATS[1]
        elif state_i == "state2":
            S1 = fct_aux.STATE2_STRATS[0]
            S2 = fct_aux.STATE2_STRATS[1]
        elif state_i == "state3":
            S1 = fct_aux.STATE3_STRATS[0]
            S2 = fct_aux.STATE3_STRATS[1]
        
        df_al["p_i_j_k"] = df_al["p_i_j_k"]\
                                .apply(pd.to_numeric,
                                       downcast='float',
                                       errors='coerce')
        df_al.loc[:,"S1"] = df_al["p_i_j_k"].copy()
        df_al.loc[:,"S2"] = 1 - df_al["p_i_j_k"].copy()
        df_al.loc[:,"S1"] = df_al["S1"].apply(pd.to_numeric,
                                        downcast='float',
                                        errors='coerce').copy()
        df_al.loc[:,"S2"] = df_al["S2"].apply(pd.to_numeric,
                                        downcast='float',
                                        errors='coerce').copy()
        print("S1={}".format(df_al["S1"]))
        print("S1={}, df_al={}, rate={}, price={}, state_i={}, t={}, algo={}".format(
                df_al["S1"].shape, df_al.shape, rate, price, state_i, t, algo))
        df_al_k = df_al.groupby("k")[["S1","S2"]]\
                    .aggregate(np.mean).reset_index()
        df_al_k.rename(columns={"k":"k","S1":ylabel_moyS1,"S2":ylabel_moyS2}, 
                       inplace=True)
        df_al_k_moyMaxS12 = df_al.groupby("k")[["S1","S2"]]\
                                .aggregate(np.max)\
                                .apply(lambda x: (x[0]-x[1])/2, axis=1)\
                                .reset_index()
        df_al_k_moyMaxS12.rename(columns={0:ylabel_moyMaxS12}, inplace=True)
        df_al_k_merge = pd.merge(df_al_k, df_al_k_moyMaxS12, on='k')
        
        df_al_k_merge.loc[:,"algo"] = algo
        df_al_k_merge.loc[:,"t"] = t
        df_al_k_merge.loc[:,"S1"] = S1
        df_al_k_merge.loc[:,"S2"] = S2
        source = ColumnDataSource(data = df_al_k_merge)
        
        ind_color = 0
        r1 = px.line(x="k", y=ylabel_moyS1, source=source, 
                     legend_label=label_moyS1,
                     line_width=2, color=COLORS[ind_color], 
                     line_dash=[0,0])
        ind_color = 1
        r2 = px.line(x="k", y=ylabel_moyS2, source=source, 
                     legend_label=label_moyS2,
                     line_width=2, color=COLORS[ind_color], 
                     line_dash=[0,0])
        ind_color = 2
        # r3 = px.line(x="k", y=ylabel_moyMaxS12, source=source, 
        #              legend_label=label_moyMaxS12,
        #              line_width=2, color=COLORS[ind_color], 
        #              line_dash=[0,0])
        
        nb_k_steps = len(list(df_al_k['k'].unique()))
        if int(nb_k_steps*10/250) > 0:
            ls = range(0,nb_k_steps,int(nb_k_steps*10/250))
        else:
            ls = range(0,nb_k_steps,1)
        df_al_slice = df_al[df_al.index.isin(ls)]
        source_slice = ColumnDataSource(data = df_al_slice)
        
        if algo == "LRI1":
            ind_color = 3
            r4 = px.asterisk(x="k", y=ylabel_moyS1, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label_moyS1)
            r5 = px.asterisk(x="k", y=ylabel_moyS2, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label_moyS2)
            # r6 = px.asterisk(x="k", y=ylabel_moyMaxS12, size=7, source=source, 
            #             color=COLORS[ind_color], legend_label=label_moyMaxS12)
            tup_legends.append((algo, [r1,r2,r4,r5] ))
            # tup_legends.append((algo, [r1,r2,r3,r4,r5,r6] ))
        elif algo == "LRI2":
            ind_color = 4
            r4 = px.asterisk(x="k", y=ylabel_moyS1, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label_moyS1)
            r5 = px.asterisk(x="k", y=ylabel_moyS2, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label_moyS2)
            # r6 = px.asterisk(x="k", y=ylabel_moyMaxS12, size=7, source=source, 
            #             color=COLORS[ind_color], legend_label=label_moyMaxS12)
            tup_legends.append((algo, [r1,r2,r4,r5] ))
            # tup_legends.append((algo, [r1,r2,r3,r4,r5,r6] ))
        
    legend = Legend(items= tup_legends, location="center")
    px.legend.label_text_font_size = "8px"
    px.legend.click_policy="hide"
    px.add_layout(legend, 'right') 

    return px                
        
def plot_max_proba_mode(df_arr_M_T_Ks, t, algos=['LRI1','LRI2']):
    """
    plot the mean of the max probability of one mode. 
    The steps of process are:
        * select all players having state_i = stateX, X in {1,2,3}
        * compute the mean of each mode for each k_step moy_S_1_X, moy_S_2_X
        * select the max of each mode for each k_step max_S_1_X, max_S_2_X
        * compute the mean of max of each mode for each k_step ie
            moy_max_S_12_X = (max_S_1_X - max_S_2_X)/2
        * plot the curves of moy_S_1_X, moy_S_2_X and moy_max_S_12_X
        
    x-axis : k_step
    y-axis : moy

    """
    rates = df_arr_M_T_Ks["rate"].unique(); rates = rates[rates!=0]
    prices = df_arr_M_T_Ks["prices"].unique()
    states = df_arr_M_T_Ks["state_i"].unique()
    
    dico_pxs = dict()
    cpt = 0
    for rate, price, state_i\
        in it.product(rates, prices, states):
        
        mask_ra_pri = ((df_arr_M_T_Ks.rate == rate) 
                                         | (df_arr_M_T_Ks.rate == 0)) \
                            & (df_arr_M_T_Ks.prices == price) \
                            & (df_arr_M_T_Ks.t == t) \
                            & (df_arr_M_T_Ks.state_i == state_i)    
        df_ra_pri = df_arr_M_T_Ks[mask_ra_pri].copy()
        
        if df_ra_pri.shape[0] != 0:
            px_st_mode = plot_max_proba_mode_onestate(
                            df_ra_pri, 
                            rate, price, state_i, 
                            t, algos)
            px_st_mode.legend.click_policy="hide"
    
            if (rate, price, state_i) not in dico_pxs.keys():
                dico_pxs[(rate, price, state_i)] \
                    = [px_st_mode]
            else:
                dico_pxs[(rate, price, state_i)].append(px_st_mode)
            cpt += 1
        
    # aggregate by state_i i.e each state_i is on new column.
    col_px_st_S1S2s = []
    for key, px_st_mode in dico_pxs.items():
        row_pxs = row(px_st_mode)
        col_px_st_S1S2s.append(row_pxs)
    col_px_st_S1S2s=column(children=col_px_st_S1S2s, 
                           sizing_mode='stretch_both')  
    # show(col_px_scen_st_S1S2s)
    
    return col_px_st_S1S2s
     
# _____________________________________________________________________________
#
#        moyenne de la grande proba dune mode pour chaque state  ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#           affichage Perf_t pour chaque state  ---> debut
# _____________________________________________________________________________
def plot_Perf_t_all_states_for_scenarios(df_ra_pri_st, 
                                rate, price, state_i, t):
                                
    """
    plot the Perf_t at each learning step for all states 
    considering all scenarios and all algorithms.
    each figure is for one state, one scenario, one prob_Ci, one learning_rate 
    and one price.
    
    Perf_t = \sum\limits_{1\leq i \leq N} ben_i-cst_i
    
    x-axis : one time t
    y-axis : Perf_t
    """
    algos = df_ra_pri_st["algo"].unique()
    
    tup_legends = [] 
    
    px = figure(plot_height = int(HEIGHT*MULT_HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    
    for algo in algos:
        df_al = df_ra_pri_st[(df_ra_pri_st.algo == algo)]
        
        title = "Perf_t for {} at t={} (rate={}, price={})".format(
                state_i, t, rate, price)
        xlabel = "k step of learning" 
        ylabel = "Perf_t" #"ben_i-cst_i"
        label = "{}".format(algo)
        
        px.title.text = title
        px.xaxis.axis_label = xlabel
        px.yaxis.axis_label = ylabel
        TOOLS[7] = HoverTool(tooltips=[
                            ("algo", "@algo"),
                            ("k", "@k"),
                            (ylabel, "$y")
                            ]
                        )
        px.tools = TOOLS
        
        cols = ['ben','cst']
        # TODO lauch warning See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy self[k1] = value[k2]
        df_al[cols] = df_al[cols].apply(pd.to_numeric, 
                                        downcast='float', 
                                        errors='coerce')
                
        df_al_k = df_al.groupby(by=["k","pl_i"])[cols]\
                    .aggregate(np.sum)\
                    .apply(lambda x: x[0]-x[1], axis=1).reset_index()
        df_al_k.rename(columns={0:ylabel}, inplace=True)
        df_al_k = df_al_k.groupby("k")[ylabel].aggregate(np.sum).reset_index()  
        
        df_al_k.loc[:,"algo"] = algo
        df_al_k.loc[:,"t"] = t
        source = ColumnDataSource(data = df_al_k)

        ind_color = 0
        if algo == "LRI1":
            ind_color = 1 #10
        elif algo == "LRI2":
            ind_color = 2 #10
        elif algo == "DETERMINIST":
            ind_color = 3 #10
        elif algo == "RD-DETERMINIST":
            ind_color = 4 #10
        elif algo == "BEST-BRUTE-FORCE":
            ind_color = 5 #10
        elif algo == "BAD-BRUTE-FORCE":
            ind_color = 6 #10
        elif algo == "MIDDLE-BRUTE-FORCE":
            ind_color = 7 #10
        elif algo == fct_aux.ALGO_NAMES_NASH[0]:                                # "BEST-NASH"
            ind_color = 8 #10
        elif algo == fct_aux.ALGO_NAMES_NASH[1]:                                # "BAD-NASH"
            ind_color = 9 #10
        elif algo == fct_aux.ALGO_NAMES_NASH[2]:                                # "MIDDLE-NASH"
            ind_color = 10 #10
            
        r1 = px.line(x="k", y=ylabel, source=source, legend_label=label,
                line_width=2, color=COLORS[ind_color], 
                line_dash=[0,0])
        
        nb_k_steps = len(list(df_al_k['k'].unique()))
        interval = int(nb_k_steps*10/250)
        print(".... nb_k_steps={}, interval={} .... ".format(nb_k_steps, interval))
        ls = None
        if nb_k_steps > interval and interval > 0:
            ls = range(0,nb_k_steps,interval)
        if nb_k_steps > interval and interval <= 0:
            ls = range(0, nb_k_steps)
        else:
            ls = range(0, nb_k_steps)
        # ls = range(0,nb_k_steps,interval) \
        #             if nb_k_steps < interval \
        #             else range(0, nb_k_steps) 
        # ls = range(0,nb_k_steps,int(nb_k_steps*10/250))
        if int(nb_k_steps*10/250) > 0:
            ls = range(0,nb_k_steps,int(nb_k_steps*10/250))
        else:
            ls = range(0,nb_k_steps,1)
        df_al_slice = df_al[df_al.index.isin(ls)]
        source_slice = ColumnDataSource(data = df_al_slice)
        
        if algo == "LRI1":
            ind_color = 1
            r2 = px.asterisk(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
            # tup_legends.append((label, [r2] ))
        elif algo == "LRI2":
            ind_color = 2
            r2 = px.circle(x="k", y=ylabel, size=7, source=source, 
                      color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
            # tup_legends.append((label, [r2] ))
        elif algo == "DETERMINIST":
            ind_color = 3
            r2 = px.triangle_dot(x="k", y=ylabel, size=7, source=source, 
                      color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
            # tup_legends.append((label, [r2] ))
        elif algo == "RD-DETERMINIST":
            ind_color = 4
            r2 = px.triangle(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
            # tup_legends.append((label, [r2] ))
        elif algo == "BEST-BRUTE-FORCE":
            r2 = px.diamond(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == "BAD-BRUTE-FORCE":
            r2 = px.diamond_cross(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == "MIDDLE-BRUTE-FORCE":
            r2 = px.diamond_dot(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == fct_aux.ALGO_NAMES_NASH[0]:                                # "BEST-NASH"
            r2 = px.square_cross(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == fct_aux.ALGO_NAMES_NASH[1]:                                # "BAD-NASH"
            r2 = px.square_pin(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == fct_aux.ALGO_NAMES_NASH[2]:                                # "MIDDLE-NASH"
            r2 = px.square_x(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        
    legend = Legend(items= tup_legends, location="center")
    px.legend.label_text_font_size = "8px"
    px.legend.click_policy="hide"
    px.add_layout(legend, 'right') 

    return px               

def plot_Perf_t_players_all_states_for_scenarios(df_ben_cst_M_T_K, t):
    """
    plot the Perf_t of players at each learning step k for all prob_Ci, price, 
    learning_rate for any state
    
    Perf_t = \sum\limits_{1\leq i \leq N} ben_i-cst_i
    """
    rates = df_ben_cst_M_T_K["rate"].unique(); rates = rates[rates!=0]
    prices = df_ben_cst_M_T_K["prices"].unique()
    states = df_ben_cst_M_T_K["state_i"].unique()
    
    dico_pxs = dict()
    cpt = 0
    for rate, price, state_i \
        in it.product(rates, prices, states):
        
        mask_ra_pri_st = ((df_ben_cst_M_T_K.rate == rate) 
                                 | (df_ben_cst_M_T_K.rate == 0)) \
                            & (df_ben_cst_M_T_K.prices == price) \
                            & (df_ben_cst_M_T_K.t == t) \
                            & (df_ben_cst_M_T_K.state_i == state_i)
        df_ra_pri_st = df_ben_cst_M_T_K[mask_ra_pri_st].copy()
        
        px_st = plot_Perf_t_all_states_for_scenarios(
                                df_ra_pri_st, 
                                rate, price, state_i, t)
        # return px_scen_st
        px_st.legend.click_policy="hide"
        
        if (rate, price, state_i) not in dico_pxs.keys():
            dico_pxs[(rate, price, state_i)] \
                = [px_st]
        else:
            dico_pxs[(rate, price, state_i)].append(px_st)
        cpt += 1                            
        
    # aggregate by state_i i.e each state_i is on new column.
    col_px_sts = []
    for key, px_sts in dico_pxs.items():
        row_px_sts = row(px_sts)
        col_px_sts.append(row_px_sts)
    col_px_sts=column(children=col_px_sts, 
                      sizing_mode='stretch_both')
    return col_px_sts

    # # aggregate by state_i i.e each state_i is on new column.
    # row_px_sts = []
    # for key, px_sts in dico_pxs.items():
    #     col_px_sts = column(px_sts)
    #     row_px_sts.append(col_px_sts)
    # row_px_sts = row(children=row_px_sts, sizing_mode='stretch_both')
        
    # return  row_px_sts 
# _____________________________________________________________________________
#
#           affichage Perf_t pour chaque state  ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> debut
# _____________________________________________________________________________
def group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, t, name_dir, 
                        NAME_RESULT_SHOW_VARS):
    
    col_pxs_Pref_t = plot_Perf_t_players_all_states_for_scenarios(
                        df_ben_cst_M_T_K, t)
    col_px_scen_st_S1S2s = plot_max_proba_mode(df_arr_M_T_Ks, t, 
                                                algos=['LRI1','LRI2'])
    # col_pxs_in_out = plot_in_out_sg_ksteps_for_scenarios(df_arr_M_T_Ks, t)
    # col_pxs_ben_cst = plot_mean_ben_cst_players_all_states_for_scenarios(
    #                     df_ben_cst_M_T_K, t)
    # col_px_scen_sts = histo_states(df_arr_M_T_Ks, t)
    # col_px_scen_mode_S1S2_nbplayers = plot_histo_strategies(df_arr_M_T_Ks, t)
    # col_playing_players = plot_histo_playing(df_arr_M_T_Ks, t)
    
    tab_Pref_t=Panel(child=col_pxs_Pref_t, title="Pref_t")
    tab_S1S2=Panel(child=col_px_scen_st_S1S2s, title="mean_S1_S2")
    # tab_inout=Panel(child=col_pxs_in_out, title="In_sg-Out_sg")
    # tab_bencst=Panel(child=col_pxs_ben_cst, title="mean(ben-cst)")
    # tab_sts=Panel(child=col_px_scen_sts, title="number players")
    # tab_mode_S1S2_nbplayers=Panel(child=col_px_scen_mode_S1S2_nbplayers, 
    #                               title="number players by strategies")
    # tab_playing=Panel(child=col_playing_players, 
    #              title="number players playing/not_playing")
    
    tabs = Tabs(tabs= [ tab_Pref_t, 
                        tab_S1S2,
                        #tab_inout, 
                        #tab_bencst,
                        #tab_sts, 
                        #tab_mode_S1S2_nbplayers, 
                        #tab_playing
                        ])
    
    output_file( os.path.join(name_dir, NAME_RESULT_SHOW_VARS)  )
    save(tabs)
    show(tabs)
    
# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> fin
# _____________________________________________________________________________


#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    MULT_WIDTH = 2.5;
    MULT_HEIGHT = 1.2;
    
    pi_hp_plus = 0.2*pow(10,-3); pi_hp_minus = 0.33
    NAME_RESULT_SHOW_VARS = NAME_RESULT_SHOW_VARS.format(pi_hp_plus, pi_hp_minus)
    
    debug = True #False#True
    
    t = 1
    
    name_simu =  "simu_2701_1259"; k_steps_args = 250
    
    algos_4_no_learning = ["DETERMINIST","RD-DETERMINIST"] \
                            + fct_aux.ALGO_NAMES_BF \
                            + fct_aux.ALGO_NAMES_NASH
    algos_4_learning = ["LRI1", "LRI2"]
    tuple_paths, prices, algos, learning_rates \
        = get_tuple_paths_of_arrays(name_simu=name_simu, 
                                    algos_4_no_learning=algos_4_no_learning)
        
    df_arr_M_T_Ks, \
    df_ben_cst_M_T_K, \
    df_b0_c0_pisg_pi0_T_K, \
    df_B_C_BB_CC_RU_M \
        = get_array_turn_df_for_t(tuple_paths, t, k_steps_args, 
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
    
    ## -- plot figures
    name_dir = os.path.join("tests", name_simu)
    group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, t, name_dir, 
                        NAME_RESULT_SHOW_VARS)
    
    print('runtime = {}'.format(time.time()-ti))
