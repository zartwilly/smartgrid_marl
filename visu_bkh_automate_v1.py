# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 08:21:09 2021

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
from bokeh.models import Band
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row, column
from bokeh.models import Panel, Tabs, Legend
from bokeh.transform import factor_cmap
from bokeh.transform import dodge

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
                algos_4_showing=["DETERMINIST", "LRI1", "LRI2",
                                 "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE"],
                ext=".npy", 
                exclude_html_files=[NAME_RESULT_SHOW_VARS,"html"]):
    
    tuple_paths = []
    path_2_best_learning_steps = []
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
            if algo in algos_4_showing:
                path_price_algo = os.path.join(name_dir, name_simu, price, algo)
                if algo not in algos_4_no_learning:
                    learning_rates_new = None
                    if learning_rates is None:
                        learning_rates_new = os.listdir(path_price_algo)
                    else:
                        learning_rates_new = learning_rates
                    for learning_rate in learning_rates_new:
                        tuple_paths.append( (name_dir, name_simu, price, 
                                             algo, learning_rate) )
                        if algo is not algos_4_no_learning:
                            path_2_best_learning_steps.append(
                                (name_dir, name_simu, price, 
                                 algo, learning_rate))
                else:
                    tuple_paths.append( (name_dir, name_simu, price, algo) )
                
    return tuple_paths, prices_new, algos_new, \
            learning_rates_new, path_2_best_learning_steps
            
def get_k_stop_4_periods(path_2_best_learning_steps):
    """
     determine the upper k_stop from algos LRI1 and LRI2 for each period

    Parameters
    ----------
    path_2_best_learning_steps : Tuple
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df_tmp = None #pd.DataFrame()
    for tuple_path_2_algo in path_2_best_learning_steps:
        path_2_algo = os.path.join(*tuple_path_2_algo)
        algo = tuple_path_2_algo[3]
        df_al = pd.read_csv(
                    os.path.join(path_2_algo, "best_learning_steps.csv"),
                    index_col=0)
        index_mapper = {"k_stop":algo+"_k_stop"}
        df_al.rename(index=index_mapper, inplace=True)
        if df_tmp is None:
            df_tmp = df_al
        else:
            df_tmp = pd.concat([df_tmp, df_al], axis=0)
            
    cols = df_tmp.columns.tolist()
    indices = df_tmp.index.tolist()
    df_k_stop = pd.DataFrame(columns=cols, index=["k_stop"])
    for col in cols:
        best_index = None
        for index in indices:
            if best_index is None:
                best_index = index
            elif df_tmp.loc[best_index, col] < df_tmp.loc[index, col]:
                best_index = index
        df_k_stop.loc["k_stop", col] = df_tmp.loc[best_index, col]
        
    return df_k_stop

def get_array_turn_df_for_t_BON(tuple_paths, t=1, k_steps_args=250, 
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
        #for t in range(0, t_periods):                                     
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
            t_periods = list(range(0, t_periods))
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
                      
        print('t_periods = {}'.format(t_periods))
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
            b0_s_T_K_2D = []
            c0_s_T_K_2D = []
            pi_0_minus_T_K_2D = []
            pi_0_plus_T_K_2D = []
            pi_sg_minus_T_K_2D = []
            pi_sg_plus_T_K_2D = []
            for tx in t_periods:
                b0_s_T_K_2D.append(list( b0_s_T_K[tx,:].reshape(-1)))
                c0_s_T_K_2D.append(list( c0_s_T_K[tx,:].reshape(-1)))
                pi_0_minus_T_K_2D.append([ pi_0_minus_T[tx] ]*k_steps_args)
                pi_0_plus_T_K_2D.append([ pi_0_plus_T[tx] ]*k_steps_args)
                pi_sg_minus_T_K_2D.append([ pi_sg_minus_T[tx] ]*k_steps_args)
                pi_sg_plus_T_K_2D.append([ pi_sg_plus_T[tx] ]*k_steps_args)
            b0_s_T_K_2D = np.array(b0_s_T_K_2D, dtype=object)
            c0_s_T_K_2D = np.array(c0_s_T_K_2D, dtype=object)
            pi_0_minus_T_K_2D = np.array(pi_0_minus_T_K_2D, dtype=object)
            pi_0_plus_T_K_2D = np.array(pi_0_plus_T_K_2D, dtype=object)
            pi_sg_minus_T_K_2D = np.array(pi_sg_minus_T_K_2D, dtype=object)
            pi_sg_plus_T_K_2D = np.array(pi_sg_plus_T_K_2D, dtype=object)
            
            b0_s_T_K_1D = b0_s_T_K_2D.reshape(-1)
            c0_s_T_K_1D = c0_s_T_K_2D.reshape(-1)
            pi_0_minus_T_K_1D = pi_0_minus_T_K_2D.reshape(-1)
            pi_0_plus_T_K_1D = pi_0_plus_T_K_2D.reshape(-1)
            pi_sg_minus_T_K_1D = pi_sg_minus_T_K_2D.reshape(-1)
            pi_sg_plus_T_K_1D = pi_sg_plus_T_K_2D.reshape(-1)
            
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
            
            arr_pl_M_T_K_vars_4D[:,:,:,:] = arrs.copy()
            # turn in 2D
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
        #for t in range(0, t_periods):                                     
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
            t_periods = list(range(0, t_periods))
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
                      
        print('t_periods = {}'.format(t_periods))
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
            b0_s_T_K_2D = []
            c0_s_T_K_2D = []
            pi_0_minus_T_K_2D = []
            pi_0_plus_T_K_2D = []
            pi_sg_minus_T_K_2D = []
            pi_sg_plus_T_K_2D = []
            for tx in t_periods:
                b0_s_T_K_2D.append(list( b0_s_T_K[tx,:].reshape(-1)))
                c0_s_T_K_2D.append(list( c0_s_T_K[tx,:].reshape(-1)))
                pi_0_minus_T_K_2D.append([ pi_0_minus_T[tx] ]*k_steps_args)
                pi_0_plus_T_K_2D.append([ pi_0_plus_T[tx] ]*k_steps_args)
                pi_sg_minus_T_K_2D.append([ pi_sg_minus_T[tx] ]*k_steps_args)
                pi_sg_plus_T_K_2D.append([ pi_sg_plus_T[tx] ]*k_steps_args)
            b0_s_T_K_2D = np.array(b0_s_T_K_2D, dtype=object)
            c0_s_T_K_2D = np.array(c0_s_T_K_2D, dtype=object)
            pi_0_minus_T_K_2D = np.array(pi_0_minus_T_K_2D, dtype=object)
            pi_0_plus_T_K_2D = np.array(pi_0_plus_T_K_2D, dtype=object)
            pi_sg_minus_T_K_2D = np.array(pi_sg_minus_T_K_2D, dtype=object)
            pi_sg_plus_T_K_2D = np.array(pi_sg_plus_T_K_2D, dtype=object)
            
            b0_s_T_K_1D = b0_s_T_K_2D.reshape(-1)
            c0_s_T_K_1D = c0_s_T_K_2D.reshape(-1)
            pi_0_minus_T_K_1D = pi_0_minus_T_K_2D.reshape(-1)
            pi_0_plus_T_K_1D = pi_0_plus_T_K_2D.reshape(-1)
            pi_sg_minus_T_K_1D = pi_sg_minus_T_K_2D.reshape(-1)
            pi_sg_plus_T_K_1D = pi_sg_plus_T_K_2D.reshape(-1)
            
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
            
            arr_pl_M_T_K_vars_4D[:,:,:,:] = arrs.copy()
            # turn in 2D
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
#                   distribution by states for periods ---> debut
# _____________________________________________________________________________
def plot_distribution(df_al_pr_ra, algo, rate, price,
                      path_2_best_learning_steps):
    """
    plot the bar plot with key is (t, stateX) (X={1,2,3})
    """
    cols = ["t", "state_i"]
    df_state = df_al_pr_ra.groupby(cols)[["state_i"]].count()
    df_state.rename(columns={"state_i":"nb_players"}, inplace=True)
    df_state = df_state.reset_index()
    df_state["t"] = df_state["t"].astype(str)
    
    x = list(map(tuple,list(df_state[cols].values)))
    nb_players = list(df_state["nb_players"])
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("nb_players", "@nb_players")
                            ]
                        )
    px= figure(x_range=FactorRange(*x), plot_height=350, 
               title="number of players, ({}, rate={}, price={})".format(
                  algo, rate, price),
                toolbar_location=None, tools=TOOLS)

    data = dict(x=x, nb_players=nb_players)
    
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='nb_players', width=0.9, source=source, 
            fill_color=factor_cmap('x', palette=Category20[20], 
                                   factors=list(df_state["t"].unique()), 
                                   start=0, end=1))
    
    px.y_range.start = 0
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    
    return px
    
def plot_distribution_by_states_4_periods(df_arr_M_T_Ks, k_steps_args,
                                          path_2_best_learning_steps):
    """
    plot the distribution by state for each period
    plot is the bar plot with key is (t, stateX) (X={1,2,3})
    
    """
    
    rates = df_arr_M_T_Ks["rate"].unique().tolist(); rate = rates[rates!=0]
    prices = df_arr_M_T_Ks["prices"].unique().tolist()
    algos = df_arr_M_T_Ks["algo"].unique().tolist()
    
    dico_pxs = dict()
    for algo, price in it.product(algos, prices):
        mask_al_pr_ra = ((df_arr_M_T_Ks.rate == str(rate)) 
                                 | (df_arr_M_T_Ks.rate == 0)) \
                            & (df_arr_M_T_Ks.prices == price) \
                            & (df_arr_M_T_Ks.algo == algo) \
                            & (df_arr_M_T_Ks.k == k_steps_args-1)    
        df_al_pr_ra = df_arr_M_T_Ks[mask_al_pr_ra].copy()
        
        pxs_al_pr_ra = plot_distribution(df_al_pr_ra, algo, rate, price,
                                         path_2_best_learning_steps)
        
        if (algo, price, rate) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate)] \
                = [pxs_al_pr_ra]
        else:
            dico_pxs[(algo, price, rate)].append(pxs_al_pr_ra)
        
    rows_dists_ts = list()
    for key, pxs_al_pr_ra in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra)
        rows_dists_ts.append(col_px_sts)
    rows_dists_ts=column(children=rows_dists_ts, 
                            sizing_mode='stretch_both')
    return rows_dists_ts

# _____________________________________________________________________________
#
#                   distribution by states for periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#                   utility of players for periods ---> debut
# _____________________________________________________________________________
def compute_CONS_PROD(df_prod_cons, algo, path_2_best_learning_steps):
    """
    compute CONS, PROD 
    """
    k_stop = None;
    path_2_best_learning = None
    if algo in ["LRI1", "LRI2"]:
        for path_2_best in path_2_best_learning_steps:
            if algo == path_2_best[3]:
                path_2_best_learning = os.path.join(*path_2_best)        
        df_tmp = pd.read_csv(os.path.join(path_2_best_learning,
                                           "best_learning_steps.csv"), 
                             sep=',', index_col=0)
    else:
        k_step_max=df_prod_cons.k.unique().max()
        t_periods = df_prod_cons.t.unique().tolist()
        dico = dict()
        for t in t_periods:
            dico[str(t)] = {"k_stop":k_step_max}
        df_tmp = pd.DataFrame.from_dict(dico)
        #k_stop = df_prod_cons.k.unique().max()
    
    # print("df_tmp: index={}, cols={}".format(df_tmp.index, df_tmp.columns))
    
    list_of_players = df_prod_cons.pl_i.unique().tolist()
    cols = ['pl_i', 'PROD_i', 'CONS_i']
    df_PROD_CONS = pd.DataFrame(columns=cols, 
                                index=list_of_players)
    for num_pl_i in list_of_players:
        sum_prod_pl_i = 0; sum_cons_pl_i = 0
        for t in df_prod_cons.t.unique().tolist():
            k_stop = df_tmp.loc["k_stop",str(t)]
            mask_pli_kstop = (df_prod_cons.t == t) \
                             & (df_prod_cons.k == k_stop) \
                             & (df_prod_cons.pl_i == num_pl_i)
            # print("df_prod_cons[mask_pli_kstop]={}".format(df_prod_cons[mask_pli_kstop].shape))
            # df_prod_cons_mask = df_prod_cons[mask_pli_kstop]
            # print("pl_{}, prod_i={}, value={}".format(num_pl_i, 
            #       df_prod_cons_mask["prod_i"], df_prod_cons_mask["prod_i"].values[0] ))
            # sum_prod_pl_i += df_prod_cons[mask_pli_kstop].loc[num_pl_i,"prod_i"]
            # sum_cons_pl_i += df_prod_cons[mask_pli_kstop].loc[num_pl_i,"cons_i"]
            sum_prod_pl_i += df_prod_cons[mask_pli_kstop]["prod_i"].values[0]
            sum_cons_pl_i += df_prod_cons[mask_pli_kstop]["cons_i"].values[0]
        # print('pl_{}, sum_prod_pl_i={}, sum_cons_pl_i={}'.format(
        #         num_pl_i, sum_prod_pl_i, sum_cons_pl_i))
        df_PROD_CONS.loc[num_pl_i, "PROD_i"] = sum_prod_pl_i
        df_PROD_CONS.loc[num_pl_i, "CONS_i"] = sum_cons_pl_i
        df_PROD_CONS.loc[num_pl_i, "pl_i"] = num_pl_i
        
    return df_PROD_CONS

def plot_CONS_PROD(df_prod_cons, algo, rate, price,
                   path_2_best_learning_steps):
    """
    plot CONS, PROD for each player
    """
    k_stop = None;
    path_2_best_learning = None
    if algo in ["LRI1", "LRI2"]:
        for path_2_best in path_2_best_learning_steps:
            if algo == path_2_best[3]:
                path_2_best_learning = os.path.join(*path_2_best)
    else:
        k_stop = df_prod_cons.k.unique().max()
            
    print("path_2_best_learning={}".format(path_2_best_learning))
    df_tmp = pd.read_csv(os.path.join(path_2_best_learning,
                                       "best_learning_steps.csv"), 
                         sep=',', index_col=0)
    
    list_of_players = df_prod_cons.pl_i.unique().tolist()
    cols = ['pl_i', 'PROD_i', 'CONS_i']
    df_PROD_CONS = pd.DataFrame(columns=cols, 
                                index=list_of_players)
    for num_pl_i in list_of_players:
        sum_prod_pl_i = 0; sum_cons_pl_i = 0
        for t in df_prod_cons.t.unique().tolist():
            k_stop = df_tmp.loc["k_stop",str(t)]
            mask_pli_kstop = (df_prod_cons.t == t) \
                             & (df_prod_cons.k == k_stop) \
                             & (df_prod_cons.pl_i == num_pl_i)
            sum_prod_pl_i += df_prod_cons[mask_pli_kstop]["prod_i"]
            sum_cons_pl_i += df_prod_cons[mask_pli_kstop]["cons_i"]
        df_PROD_CONS.loc[num_pl_i, "PROD_i"] = sum_prod_pl_i
        df_PROD_CONS.loc[num_pl_i, "CONS_i"] = sum_cons_pl_i
        df_PROD_CONS.loc[num_pl_i, "pl_i"] = num_pl_i
        
    # plot
    df_PROD_CONS["pl_i"] = df_PROD_CONS["pl_i"].astype(str)
    idx = df_PROD_CONS["pl_i"].unique().tolist()
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("pl_i", "@pl_i"),
                            ("PROD_i", "@PROD_i"),
                            ("CONS_i", "@CONS_i"),
                            ]
                        )
    
    px = figure(x_range=idx, 
                y_range=(0, df_PROD_CONS[cols[1:]].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "{}: PROD/CONS of players (rate:{}, price={})"\
                .format(algo, rate, price)
    px.title.text = title
           
    source = ColumnDataSource(data = df_PROD_CONS)
    
    width= 0.2 #0.5
    px.vbar(x=dodge('pl_i', -0.3, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#c9d9d3", legend_label=cols[1])
    px.vbar(x=dodge('pl_i', -0.3+width, range=px.x_range), top=cols[2], 
                    width=width, source=source,
                    color="#718dbf", legend_label=cols[2])
    
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "players"
    px.yaxis.axis_label = "values"
    
    return px, df_PROD_CONS
            
def plot_utility_OLD(df_al_pr_ra, algo, rate, price,
                 path_2_best_learning_steps):
    """
    plot the bar plot of each player relying on the real utility RU
    """
    
    df_al_pr_ra["pl_i"] = df_al_pr_ra["pl_i"].astype(str)
    idx = df_al_pr_ra["pl_i"].unique().tolist()
    cols = ['pl_i', 'BB', 'CC', 'RU']
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("RU", "@RU"),
                            ("BB", "@BB"),
                            ("CC", "@CC"),
                            ]
                        )
    
    px = figure(x_range=idx, 
                y_range=(0, df_al_pr_ra[cols[1:]].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "{}: utility of players (rate:{}, price={})".format(algo, rate, price)
    px.title.text = title
           
    source = ColumnDataSource(data = df_al_pr_ra)
    
    width= 0.2 #0.5
    px.vbar(x=dodge('pl_i', -0.3, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#c9d9d3", legend_label=cols[1])
    px.vbar(x=dodge('pl_i', -0.3+width, range=px.x_range), top=cols[2], 
                    width=width, source=source,
                    color="#718dbf", legend_label=cols[2])
    px.vbar(x=dodge('pl_i', -0.3+2*width, range=px.x_range), top=cols[3], 
                   width=width, source=source,
                   color="#e84d60", legend_label=cols[3])
    
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "players"
    px.yaxis.axis_label = "values"
    
    return px
    
def plot_utility(df_res, algo, rate, price,
                 path_2_best_learning_steps):
    """
    plot the bar plot of each player relying on the real utility RU
    """
    
    df_res["pl_i"] = df_res["pl_i"].astype(str)
    idx = df_res["pl_i"].unique().tolist()
    cols = ['pl_i', 'BB', 'CC', 'RU', "CONS_i", "PROD_i"]
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("pl_i", "@pl_i"),
                            ("RU", "@RU"),
                            ("BB", "@BB"),
                            ("CC", "@CC"),
                            ("CONS_i", "@CONS_i"),
                            ("PROD_i", "@PROD_i")
                            ]
                        )
    
    px = figure(x_range=idx, 
                y_range=(0, df_res[cols[1:]].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "{}: utility of players (rate:{}, price={})".format(algo, rate, price)
    px.title.text = title
           
    source = ColumnDataSource(data = df_res)
    
    width= 0.2 #0.5
    px.vbar(x=dodge('pl_i', -0.3, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#2E8B57", legend_label=cols[1])
    px.vbar(x=dodge('pl_i', -0.3+width, range=px.x_range), top=cols[2], 
                    width=width, source=source,
                    color="#718dbf", legend_label=cols[2])
    px.vbar(x=dodge('pl_i', -0.3+2*width, range=px.x_range), top=cols[3], 
                   width=width, source=source,
                   color="#e84d60", legend_label=cols[3])
    px.vbar(x=dodge('pl_i', -0.3+3*width, range=px.x_range), top=cols[4], 
                   width=width, source=source,
                   color="#ddb7b1", legend_label=cols[4])
    px.vbar(x=dodge('pl_i', -0.3+5*width, range=px.x_range), top=cols[5], 
                   width=width, source=source,
                   color="#FFD700", legend_label=cols[5])
    
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "players"
    px.yaxis.axis_label = "values"
    
    return px
    
def plot_utilities_by_player_4_periods(df_arr_M_T_Ks, 
                                       df_b0_c0_pisg_pi0_T_K,
                                       df_B_C_BB_CC_RU_M, 
                                       path_2_best_learning_steps):
    """
    plot the utility RU, CONS and PROD of players.
    for each algorithm, plot the utility of each player 
    """
    rates = df_arr_M_T_Ks.rate.unique().tolist(); rate = rates[rates!=0]
    prices = df_arr_M_T_Ks.prices.unique().tolist()
    algos = df_B_C_BB_CC_RU_M.algo.unique().tolist()
    
    dico_pxs = dict()
    for algo, price in it.product(algos, prices):
        # CONS_is, PROD_is = compute_CONS_PROD(df_arr_M_T_Ks, 
        #                                      algo, 
        #                                      path_2_best_learning_steps)
        # mask_al_pr_ra = ((df_B_C_BB_CC_RU_M.rate == str(rate)) 
        #                          | (df_B_C_BB_CC_RU_M.rate == 0)) \
        #                     & (df_B_C_BB_CC_RU_M.prices == price) \
        #                     & (df_B_C_BB_CC_RU_M.algo == algo)     
        
        # mask_al_pr_ra_prod_cons = ((df_arr_M_T_Ks.rate == str(rate)) 
        #                            | (df_arr_M_T_Ks.rate == 0)) \
        #                             & (df_arr_M_T_Ks.prices == price) \
        #                             & (df_arr_M_T_Ks.algo == algo)  
        
        # df_al_pr_ra = df_B_C_BB_CC_RU_M[mask_al_pr_ra].copy()
        # pxs_al_pr_ra = plot_utility(
        #                     df_al_pr_ra, algo, rate, price,
        #                     path_2_best_learning_steps)
        
        # df_prod_cons = df_arr_M_T_Ks[mask_al_pr_ra_prod_cons].copy()
        # df_PROD_CONS = compute_CONS_PROD(df_prod_cons, algo, 
        #                                  path_2_best_learning_steps)
        
        # pxs_prod_cons, df_PROD_CONS = plot_CONS_PROD(
        #                                 df_prod_cons, algo, rate, price,
        #                                 path_2_best_learning_steps)
        
        ######################################################################
        print("ALGO={}".format(algo))
        mask_al_pr_ra = ((df_B_C_BB_CC_RU_M.rate == str(rate)) 
                                 | (df_B_C_BB_CC_RU_M.rate == 0)) \
                            & (df_B_C_BB_CC_RU_M.prices == price) \
                            & (df_B_C_BB_CC_RU_M.algo == algo)     
        
        mask_al_pr_ra_prod_cons = ((df_arr_M_T_Ks.rate == str(rate)) 
                                   | (df_arr_M_T_Ks.rate == 0)) \
                                    & (df_arr_M_T_Ks.prices == price) \
                                    & (df_arr_M_T_Ks.algo == algo)  
        
        df_al_pr_ra = df_B_C_BB_CC_RU_M[mask_al_pr_ra].copy()
        
        df_prod_cons = df_arr_M_T_Ks[mask_al_pr_ra_prod_cons].copy()
        df_PROD_CONS = compute_CONS_PROD(df_prod_cons, algo, 
                                         path_2_best_learning_steps)
        print("{}: df_PROD_CONS={}, df_al_pr_ra={}".format(
            algo, df_PROD_CONS.shape, df_al_pr_ra.shape))
        # merge on column pl_i
        df_res = pd.merge(df_al_pr_ra, df_PROD_CONS, on="pl_i")
        pxs_al_pr_ra = plot_utility(
                            df_res, algo, rate, price,
                            path_2_best_learning_steps)
        
        ######################################################################
        
        if (algo, price, rate) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate)] \
                = [pxs_al_pr_ra]
        else:
            dico_pxs[(algo, price, rate)].append(pxs_al_pr_ra)
        
    rows_RU_CONS_PROD_ts = list()
    for key, pxs_al_pr_ra in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra)
        rows_RU_CONS_PROD_ts.append(col_px_sts)
    rows_RU_CONS_PROD_ts=column(children=rows_RU_CONS_PROD_ts, 
                                sizing_mode='stretch_both')
    return rows_RU_CONS_PROD_ts

# _____________________________________________________________________________
#
#                   utility of players for periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> debut
# _____________________________________________________________________________
def group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                        t, k_steps_args, name_dir,
                        path_2_best_learning_steps, 
                        NAME_RESULT_SHOW_VARS):
    
    rows_dists_ts = plot_distribution_by_states_4_periods(
                        df_arr_M_T_Ks, k_steps_args,
                        path_2_best_learning_steps)
    
    rows_RU_CONS_PROD_ts = plot_utilities_by_player_4_periods(
                            df_arr_M_T_Ks, 
                            df_b0_c0_pisg_pi0_T_K,
                            df_B_C_BB_CC_RU_M, 
                            path_2_best_learning_steps)
    
    # col_pxs_Pref_t = plot_Perf_t_players_all_states_for_scenarios(
    #                     df_ben_cst_M_T_K, t)
    # col_pxs_Pref_algo_t = plot_Perf_t_players_all_algos(df_ben_cst_M_T_K, t)
    # col_px_scen_st_S1S2s = plot_max_proba_mode(df_arr_M_T_Ks, t, 
    #                                            path_2_best_learning_steps, 
    #                                            algos=['LRI1','LRI2'])
    # col_pxs_in_out = plot_in_out_sg_ksteps_for_scenarios(df_arr_M_T_Ks, t)
    # col_pxs_ben_cst = plot_mean_ben_cst_players_all_states_for_scenarios(
    #                     df_ben_cst_M_T_K, t)
    # col_px_scen_sts = histo_states(df_arr_M_T_Ks, t)
    # col_px_scen_mode_S1S2_nbplayers = plot_histo_strategies(df_arr_M_T_Ks, t)
    # col_playing_players = plot_histo_playing(df_arr_M_T_Ks, t)
    
    tab_dists_ts = Panel(child=rows_dists_ts, title="distribution by state")
    tab_RU_CONS_PROD_ts = Panel(child=rows_RU_CONS_PROD_ts, 
                                title="utility of players")
    # tab_Pref_t=Panel(child=col_pxs_Pref_t, title="Pref_t by state")
    # tab_Pref_algo_t=Panel(child=col_pxs_Pref_algo_t, title="Pref_t")
    # tab_S1S2=Panel(child=col_px_scen_st_S1S2s, title="mean_S1_S2")
    # tab_inout=Panel(child=col_pxs_in_out, title="In_sg-Out_sg")
    # tab_bencst=Panel(child=col_pxs_ben_cst, title="mean(ben-cst)")
    # tab_sts=Panel(child=col_px_scen_sts, title="number players")
    # tab_mode_S1S2_nbplayers=Panel(child=col_px_scen_mode_S1S2_nbplayers, 
    #                               title="number players by strategies")
    # tab_playing=Panel(child=col_playing_players, 
    #              title="number players playing/not_playing")
    
    tabs = Tabs(tabs= [ 
                        tab_dists_ts,
                        tab_RU_CONS_PROD_ts
                        #tab_Pref_t, 
                        #tab_Pref_algo_t,
                        #tab_S1S2,
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
    # fct_aux.N_DECIMALS = 7;
    
    pi_hp_plus = 0.2*pow(10,-3); pi_hp_minus = 0.33
    NAME_RESULT_SHOW_VARS = NAME_RESULT_SHOW_VARS.format(pi_hp_plus, pi_hp_minus)
    
    debug = True #False#True
    
    t = 1
    
    name_simu =  "simu_2701_1259"; k_steps_args = 250
    name_simu =  "simu_DDMM_HHMM"; k_steps_args = 50#2000#250
    name_simu = "simu_DDMM_HHMM_T2_scenario8_set1_10_repSet1_0.95_set2_6_repSet2_0.95"
    k_steps_args = 350 #2000#250
    
    
    algos_4_no_learning = ["DETERMINIST","RD-DETERMINIST"] \
                            + fct_aux.ALGO_NAMES_BF \
                            + fct_aux.ALGO_NAMES_NASH
    algos_4_learning = ["LRI1", "LRI2"]
    algos_4_showing = ["DETERMINIST", "LRI1", "LRI2"] \
                        + [fct_aux.ALGO_NAMES_BF[0], fct_aux.ALGO_NAMES_BF[1]]
                        
    tuple_paths, prices, algos, learning_rates, path_2_best_learning_steps \
        = get_tuple_paths_of_arrays(
            name_simu=name_simu, 
            algos_4_no_learning=algos_4_no_learning, 
            algos_4_showing = algos_4_showing
            )
    #print("tuple_paths:{}".format(tuple_paths))
    #print("path_2_best_learning_steps:{}".format(path_2_best_learning_steps))
    print("get_tuple_paths_of_arrays: TERMINE")    
        
    
    dico_k_stop = dict()
    df_k_stop = get_k_stop_4_periods(path_2_best_learning_steps)
    print("get_k_stop_4_periods: TERMINE") 
    
    
    df_arr_M_T_Ks, \
    df_ben_cst_M_T_K, \
    df_b0_c0_pisg_pi0_T_K, \
    df_B_C_BB_CC_RU_M \
        = get_array_turn_df_for_t(tuple_paths, t=None, k_steps_args=k_steps_args, 
                                  algos_4_no_learning=algos_4_no_learning, 
                                  algos_4_learning=algos_4_learning)
    print("get_array_turn_df_for_t: TERMINE")
    
    ## -- plot figures
    name_dir = os.path.join("tests", name_simu)
    group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                        t, k_steps_args, name_dir, 
                        path_2_best_learning_steps, 
                        NAME_RESULT_SHOW_VARS)
    
    
    




