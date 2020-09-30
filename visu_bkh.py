# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:38:02 2020

@author: jwehounou

visualization of the prices or production/ consumption for each/all agents. 
"""
import os
import time

import numpy as np
import pandas as pd
import fonctions_auxiliaires as fct_aux
import game_model_period_T as gmT

from pathlib import Path

from bokeh.plotting import *
from bokeh.layouts import grid, column;
from bokeh.models import ColumnDataSource, FactorRange;
from bokeh.plotting import figure, output_file, show, gridplot;
from bokeh.core.properties import value
from bokeh.palettes import Spectral5
from bokeh.models.tools import HoverTool
from bokeh.models.tickers import FixedTicker
from bokeh.models import FuncTickFormatter


#------------------------------------------------------------------------------
#                   definitions of constants
#------------------------------------------------------------------------------
WIDTH = 500;
HEIGHT = 500;
#MUL_WIDTH = 2.5;
#MUL_HEIGHT = 3.5;

MARKERS = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", 
               "P", "*", "h", "H", "+", "x", "X", "D", "d"]
COLORS = ["red", "yellow", "blue", "green", "rosybrown", 
              "darkorange", "fuchsia", "grey"]

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------
def get_local_storage_variables(path_to_variable):
    """
    obtain the content of variables stored locally .

    Returns
    -------
     arr_pls_M_T, RUs, B0s, C0s, BENs, CSTs, pi_sg_plus_s, pi_sg_minus_s.
    
    arr_pls_M_T: array of players with a shape M_PLAYERS*NUM_PERIODS*INDEX_ATTRS
    RUs: array of (M_PLAYERS,)
    BENs: array of M_PLAYERS*NUM_PERIODS
    CSTs: array of M_PLAYERS*NUM_PERIODS
    B0s: array of (NUM_PERIODS,)
    C0s: array of (NUM_PERIODS,)
    pi_sg_plus_s: array of (NUM_PERIODS,)
    pi_sg_minus_s: array of (NUM_PERIODS,)

    """
    arr_pls_M_T = np.load(os.path.join(path_to_variable, "arr_pls_M_T.npy"),
                          allow_pickle=True)
    RUs = np.load(os.path.join(path_to_variable, "RUs.npy"),
                          allow_pickle=True)
    B0s = np.load(os.path.join(path_to_variable, "B0s.npy"),
                          allow_pickle=True)
    C0s = np.load(os.path.join(path_to_variable, "C0s.npy"),
                          allow_pickle=True)
    BENs = np.load(os.path.join(path_to_variable, "BENs.npy"),
                          allow_pickle=True)
    CSTs = np.load(os.path.join(path_to_variable, "CSTs.npy"),
                          allow_pickle=True)
    pi_sg_plus_s = np.load(os.path.join(path_to_variable, "pi_sg_plus_s.npy"),
                          allow_pickle=True)
    pi_sg_minus_s = np.load(os.path.join(path_to_variable, "pi_sg_minus_s.npy"),
                          allow_pickle=True)
    
    return arr_pls_M_T, RUs, B0s, C0s, BENs, CSTs, pi_sg_plus_s, pi_sg_minus_s

def plot_pi_sg(pi_sg_plus_s, pi_sg_minus_s, path_to_variable):
    """
    it is also pi_0

    Parameters
    ----------
    pi_sg_plus_s : array of (NUM_PERIODS,), price of exported energy from SG to HP
        DESCRIPTION.
    pi_sg_minus_s : array of (NUM_PERIODS,), price of imported energy from HP to SG
        DESCRIPTION.

    Returns
    -------
    None.

    """
    title = "price of exported/imported energy inside SG"
    xlabel = "periods of time" 
    ylabel = "price SG"
    
    dico_pi_sg = {"t":range(0,pi_sg_plus_s.shape[0]), 
                  "pi_sg_plus": pi_sg_plus_s,
                  "pi_sg_minus": pi_sg_minus_s}
    df_pi_sg = pd.DataFrame(dico_pi_sg)
    src = ColumnDataSource(df_pi_sg)
    p_sg = figure(plot_height = HEIGHT, 
                     plot_width = WIDTH, 
                     title = title,
                     x_axis_label = xlabel, 
                     y_axis_label = ylabel, 
                     x_axis_type = "linear",
                     tools = TOOLS)
    p_sg.line(x="t", y="pi_sg_plus", source=src, legend_label="pi_sg_plus",
                   line_width=2,color=COLORS[0])
    p_sg.line(x="t", y="pi_sg_minus", source=src, legend_label="pi_sg_minus",
                   line_width=2,color=COLORS[1])
    p_sg.legend.location = "top_right"
    
    #show(p_sg)
    
    # configuration figure
    rep_visu = os.path.join(path_to_variable, "visu")
    Path(rep_visu).mkdir(parents=True, exist_ok=True)
    output_file(os.path.join(rep_visu,"pi_sg_dashboard.html"))
    
def plot_pi_X(args):
    """
    plot data depending only on time like 
    pi_sg_plus_s, pi_sg_minus_s, B0s and C0s 

    Parameters
    ----------
    args : dict. 
        DESCRIPTION.
        Keys are "title", "xlabel", "ylabel",
          "pi_sg_plus_s", "pi_sg_minus_s", "key_plus", 
          "key_minus".
         key_plus is a first string name for data array we want to show.
             exple : "key_plus" = "pi_sg_plus_s"
         key_minus is a second string name for data array we want to show.
             exple : "key_minus" = "pi_sg_minus_s"
         pi_sg_plus_s is the first data we want to show.
             exple : "pi_sg_plus_s" = pi_sg_plus_s
        idem for pi_sg_minus_s
    Returns
    -------
    p_X : TYPE
        DESCRIPTION.

    """
    title = args["title"]
    xlabel = args["xlabel"]
    ylabel = args["ylabel"]
    
    dico_pi_X = {"t": range(0, args["pi_sg_plus_s"].shape[0]),
                 args["key_plus"]: args["pi_sg_plus_s"],
                 args["key_minus"]: args["pi_sg_minus_s"]}
    
    df_pi_X = pd.DataFrame(dico_pi_X)
    src = ColumnDataSource(df_pi_X)
    p_X = figure(plot_height = HEIGHT, 
                     plot_width = WIDTH, 
                     title = title,
                     x_axis_label = xlabel, 
                     y_axis_label = ylabel, 
                     x_axis_type = "linear",
                     tools = TOOLS)
    
    p_X.line(x="t", y=args["key_plus"], source=src, 
              legend_label= args["key_plus"],
              line_width=2, color=COLORS[0])
    p_X.line(x="t", y=args["key_minus"], source=src, 
              legend_label= args["key_minus"],
              line_width=2, color=COLORS[1])
    
    p_X.legend.location = "top_right"
    
    return p_X

def plot_more_prices(path_to_variable):
    # get datas
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
    
    # _______ plot pi_sg_plus, pi_sg_minus _______
    title = "price of exported/imported energy inside SG"
    xlabel = "periods of time" 
    ylabel = "price SG"
    key_plus = "pi_sg_plus"
    key_minus = "pi_sg_minus"
    
    args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
          "pi_sg_plus_s": pi_sg_plus_s,"pi_sg_minus_s":pi_sg_minus_s, 
          "key_plus":key_plus, "key_minus":key_minus}
    
    p_sg = plot_pi_X(args)
    
    # _______ plot B0s, C0s _______
    title = "benefit/cost price for an unit of energy inside SG over the time"
    xlabel = "periods of time" 
    ylabel = "prices"
    key_plus = "B0s"
    key_minus = "C0s"
    
    args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
          "pi_sg_plus_s": B0s,"pi_sg_minus_s":C0s, 
          "key_plus":key_plus, "key_minus":key_minus}
    
    p_B0_C0s = plot_pi_X(args)
    
    # show p_sg and p_B0_C0s on the same html
    p = gridplot([[p_sg, p_B0_C0s]], 
                 toolbar_location='above')
    show(p)
    
    # configuration figure
    rep_visu = os.path.join(path_to_variable, "visu")
    Path(rep_visu).mkdir(parents=True, exist_ok=True)
    output_file(os.path.join(rep_visu,"pi_sg_dashboard.html"))
#------------------------------------------------------------------------------
#                   definitions of unit test of defined functions
#------------------------------------------------------------------------------
def test_get_local_storage_variables():
    name_dir = "tests"
    reps = os.listdir(name_dir)
    rep = reps[np.random.randint(0,len(reps))]
    path_to_variable = os.path.join(name_dir, rep)
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    print("____test_get_local_storage_variables____")
    print("arr_pls_M_T: {},RUs={},B0s={},pi_sg_plus_s={}".format(arr_pls_M_T.shape,RUs.shape,B0s.shape,pi_sg_plus_s.shape)) 
    print("arr_pls_M_T: OK") \
        if arr_pls_M_T.shape == (gmT.M_PLAYERS,gmT.NUM_PERIODS+1,len(gmT.INDEX_ATTRS)) \
        else print("arr_pls_M_T: NOK")
    print("RUs: OK") \
        if RUs.shape == (gmT.M_PLAYERS,) \
        else print("RUs: NOK")
    print("B0s: OK") \
        if B0s.shape == (gmT.NUM_PERIODS,) \
        else print("B0s: NOK")
    print("C0s: OK") \
        if C0s.shape == (gmT.NUM_PERIODS,) \
        else print("C0s: NOK")
    print("pi_sg_plus_s: OK") \
        if pi_sg_plus_s.shape == (gmT.NUM_PERIODS,) \
        else print("pi_sg_plus_s: NOK")
    print("pi_sg_minus_s: OK") \
        if pi_sg_minus_s.shape == (gmT.NUM_PERIODS,) \
        else print("pi_sg_minus_s: NOK")
    
def test_plot_pi_sg():
    name_dir = "tests"
    reps = os.listdir(name_dir)
    rep = reps[np.random.randint(0,len(reps))]
    path_to_variable = os.path.join(name_dir, rep)
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    plot_pi_sg(pi_sg_plus_s, pi_sg_minus_s, path_to_variable)
    
def test_plot_more_prices():
    name_dir = "tests"
    reps = os.listdir(name_dir)
    rep = reps[np.random.randint(0,len(reps))]
    path_to_variable = os.path.join(name_dir, rep)
    plot_more_prices(path_to_variable)
    
#------------------------------------------------------------------------------
#                   execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_get_local_storage_variables()
    test_plot_pi_sg()
    test_plot_more_prices()
    print("runtime {}".format(time.time()-ti))
    

