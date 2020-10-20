# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:38:02 2020

@author: jwehounou

visualization of the prices or production/ consumption for each/all agents. 
"""
import os
import time
import random

import numpy as np
import pandas as pd
import fonctions_auxiliaires as fct_aux
import game_model_period_T as gmT
import execution_game as exec_game

from pathlib import Path

from bokeh.plotting import *
from bokeh.layouts import grid, column;
from bokeh.models import ColumnDataSource, FactorRange, Range1d;
from bokeh.plotting import figure, output_file, show, gridplot;
from bokeh.core.properties import value
from bokeh.palettes import Spectral5
from bokeh.models.tools import HoverTool, PanTool, BoxZoomTool, WheelZoomTool, UndoTool
from bokeh.models.tools import RedoTool, ResetTool, SaveTool
from bokeh.models.tickers import FixedTicker
from bokeh.models import FuncTickFormatter
from bokeh.io import curdoc
from bokeh.plotting import reset_output
#Importing a pallette
from bokeh.palettes import Spectral5, Viridis256, Colorblind, Magma256, Turbo256
from bokeh.palettes import Category20
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

    pi_hp_plus_s: array of (NUM_PERIODS,)
    pi_hp_minus_s: array of (NUM_PERIODS,)
    """
    arr_pl_M_T = np.load(os.path.join(path_to_variable, "arr_pls_M_T.npy"),
                          allow_pickle=True)
    arr_pl_M_T_old = np.load(os.path.join(path_to_variable, "arr_pls_M_T_old.npy"),
                          allow_pickle=True)
    RU_is = np.load(os.path.join(path_to_variable, "RU_is.npy"),
                          allow_pickle=True)
    b0_s = np.load(os.path.join(path_to_variable, "b0_s.npy"),
                          allow_pickle=True)
    c0_s = np.load(os.path.join(path_to_variable, "c0_s.npy"),
                          allow_pickle=True)
    B_is = np.load(os.path.join(path_to_variable, "B_is.npy"),
                          allow_pickle=True)
    C_is = np.load(os.path.join(path_to_variable, "C_is.npy"),
                          allow_pickle=True)
    BENs = np.load(os.path.join(path_to_variable, "BENs.npy"),
                          allow_pickle=True)
    CSTs = np.load(os.path.join(path_to_variable, "CSTs.npy"),
                          allow_pickle=True)
    BB_is = np.load(os.path.join(path_to_variable, "BB_is.npy"),
                          allow_pickle=True)
    CC_is = np.load(os.path.join(path_to_variable, "CC_is.npy"),
                          allow_pickle=True)
    pi_sg_plus_s = np.load(os.path.join(path_to_variable, "pi_sg_plus_s.npy"),
                          allow_pickle=True)
    pi_sg_minus_s = np.load(os.path.join(path_to_variable, "pi_sg_minus_s.npy"),
                          allow_pickle=True)
    pi_hp_plus_s = np.load(os.path.join(path_to_variable, "pi_hp_plus_s.npy"),
                          allow_pickle=True)
    pi_hp_minus_s = np.load(os.path.join(path_to_variable, "pi_hp_minus_s.npy"),
                          allow_pickle=True)
    
    return arr_pl_M_T_old, arr_pl_M_T, \
            b0_s, c0_s, \
            B_is, C_is, \
            BENs, CSTs, \
            BB_is, CC_is, RU_is, \
            pi_sg_plus_s, pi_sg_minus_s, \
            pi_hp_plus_s, pi_hp_minus_s

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
    
def plot_pi_X(args, p_X=None, styles={"line":"solid"}):
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
    p_X : Figure
        DESCRIPTION.
        contain a chart of variable(s)
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
    if p_X is None:
        p_X = figure(plot_height = HEIGHT, 
                         plot_width = WIDTH, 
                         title = title,
                         x_axis_label = xlabel, 
                         y_axis_label = ylabel, 
                         x_axis_type = "linear",
                         tools = TOOLS)
    
    p_X.line(x="t", y=args["key_plus"], source=src, 
              legend_label= args["key_plus"],
              line_width=2, color=Category20[20][6], line_dash=styles["line"])
    p_X.circle(x="t", y=args["key_plus"], source=src, 
               size=4, fill_color='white')
    p_X.line(x="t", y=args["key_minus"], source=src, 
              legend_label= args["key_minus"],
              line_width=2, color=Category20[20][8], line_dash=styles["line"])
    p_X.circle(x="t", y=args["key_minus"], source=src, 
               size=4, fill_color='white')
    
    p_X.legend.location = "top_right"
    
    return p_X

def plot_more_prices(path_to_variable, dbg=True):
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
    
    if dbg:
        # # show p_sg and p_B0_C0s on the same html
        p = gridplot([p_sg, p_B0_C0s], ncols = 2,
                      toolbar_location='above')
        
        # configuration figure
        rep_visu = os.path.join(path_to_variable, "visu")
        Path(rep_visu).mkdir(parents=True, exist_ok=True)
        output_file(os.path.join(rep_visu,"pi_sg_dashboard.html"))
        
        show(p)
        return None, None
    else:
        return p_sg, p_B0_C0s
    
def plot_player(arr_pls_M_T, RUs, BENs, CSTs, id_pls,
                path_to_variable, dbg=True):
    """
    plot the benefit and the cost for some players over the time as well as 
    a real utility of the selected players.

    Parameters
    ----------
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*INDEX_ATTRS
        DESCRIPTION.
    RUs : array of (M_PLAYERS,)
        DESCRIPTION.
    id_pls : number_of_players items
        index of selected players
        DESCRIPTION.
         it's the numbers of players you will selected to arr_pls_M_T. we obtain 
         an array called arr_pls_M_T_nop
    Returns
    -------
    None.

    """ 
    #id_pls = np.random.choice(arr_pls_M_T.shape[0], number_of_players)
    
    arr_pls_M_T_nop = arr_pls_M_T[id_pls,:,:]
    
    ps_pls = []
    for num_pl in range(0,arr_pls_M_T_nop.shape[0]):
        #prod_i = arr_pls_M_T_nop[num_pl,:,gmT.INDEX_ATTRS['prod_i']]
        ben_is = BENs[num_pl]
        cst_is = CSTs[num_pl]
                
        #_____ plot ben_i, cst_i ______
        title = "benefit/cost  for a player pl_"+str(id_pls[num_pl])+" inside SG"
        xlabel = "periods of time" 
        ylabel = "price"
        key_plus = "ben_i"
        key_minus = "cst_i"
    
        args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
              "pi_sg_plus_s": ben_is,"pi_sg_minus_s":cst_is, 
              "key_plus":key_plus, "key_minus":key_minus}
        p_ben_cst = plot_pi_X(args, p_X=None)
        
        ps_pls.append(p_ben_cst)
    
    if dbg:
    
        print("ps_pls = {}".format(ps_pls)) if dbg else None
        # show p_ben_cst for all selected players
        ps = gridplot(ps_pls, ncols = 2,
                      toolbar_location='above')
        
        ## configuration figure
        rep_visu = os.path.join(path_to_variable, "visu")
        Path(rep_visu).mkdir(parents=True, exist_ok=True)
        output_file(os.path.join(rep_visu,"player_benef_costs_dashboard.html"))
        show(ps)
        return None
    else: 
        return ps_pls
    
def plot_prod_cons_player(arr_pls_M_T, id_pls, path_to_variable, dbg=True):
    """
    plot production and consumption for "number_of_players" players.

    Parameters
    ----------
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*INDEX_ATTRS
        DESCRIPTION.
    path_to_variable : TYPE
        DESCRIPTION.
    id_pls : number_of_players items
        index of selected players
        DESCRIPTION. 
        it's the numbers of players you will selected to arr_pls_M_T. We obtain 
         an array called arr_pls_M_T_nop
    dbg : Boolean, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    arr_pls_M_T_nop = arr_pls_M_T[id_pls,:,:]
    
    ps_pls = []
    for num_pl in range(0,arr_pls_M_T_nop.shape[0]):
        prod_is = arr_pls_M_T_nop[num_pl,:, gmT.INDEX_ATTRS["prod_i"]]
        cons_is = arr_pls_M_T_nop[num_pl,:, gmT.INDEX_ATTRS["cons_i"]]
        Pis = arr_pls_M_T_nop[num_pl,:, gmT.INDEX_ATTRS["Pi"]]
        Cis = arr_pls_M_T_nop[num_pl,:, gmT.INDEX_ATTRS["Ci"]]
        
        #_____ plot prod_i, cons_i ______
        title = "production/consumption of a player pl_"+str(id_pls[num_pl])\
                +" exchanged with SG"
        xlabel = "periods of time" 
        ylabel = "quantity(kwh)"
        key_plus = "prod_i"
        key_minus = "cons_i"
    
        args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
              "pi_sg_plus_s": prod_is,"pi_sg_minus_s":cons_is, 
              "key_plus":key_plus, "key_minus":key_minus}
        p_prod_cons = plot_pi_X(args, p_X=None)
        
        
        #_____ plot Pi, Ci on the same figure than prod_i, cons_i
        key_plus = "Pi"
        key_minus = "Ci"
    
        args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
              "pi_sg_plus_s": Pis,"pi_sg_minus_s": Cis, 
              "key_plus":key_plus, "key_minus":key_minus}
        p_prod_cons_Pi_Ci = plot_pi_X(args, p_X=p_prod_cons, 
                                      styles={"line":"dashed"})
        
        ps_pls.append(p_prod_cons_Pi_Ci)
    return ps_pls
     
####################        plot -----> debut 
from bokeh.transform import factor_cmap
from bokeh.transform import dodge
def plot_state_mode(state_i_s, mode_i_s):
    """
    represent barplot of states and modes for a player i

    Parameters
    ----------
    state_i_s : array of (NUM_PERIODs, ) item of states
        DESCRIPTION.
    mode_i_s : array of (NUM_PERIODs, ) item of modes
        DESCRIPTION.

    Returns
    -------
    list of plots containing plots of state and mode.

    """
    # _____ state_i _____
    
    STATES = ["state1","state2","state3"]
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]
    # x = [(str(t), state) for t in range(0, exec_game.NUM_PERIODS+1) 
    #                   for state in STATES ]
    data = {"t": list(map(str,range(0, exec_game.NUM_PERIODS+1))),
            "state1": [0]*(exec_game.NUM_PERIODS+1), 
            "state2": [0]*(exec_game.NUM_PERIODS+1), 
            "state3": [0]*(exec_game.NUM_PERIODS+1)}
    for num, state in enumerate(state_i_s):
        if state == "state1":
            data["state1"][num] = 1
        elif state == "state2":
            data["state2"][num] = 1
        elif state == "state3":
            data["state3"][num] = 1
    p_state = figure(x_range= list(map(str,range(0, exec_game.NUM_PERIODS+1))), 
                      plot_height=350, plot_width=550,
                      title="States over the time",
                      tools = TOOLS)        
    source = ColumnDataSource(data=data)
    
    p_state.vbar(x=dodge('t', -0.25, range=p_state.x_range), 
                top='state1', width=0.2, 
                source=source, color="#c9d9d3", legend_label="state1")
    
    p_state.vbar(x=dodge('t', -0.25, range=p_state.x_range), 
                top='state2', width=0.2, 
                source=source, color="#718dbf", legend_label="state2")
    
    p_state.vbar(x=dodge('t', -0.25, range=p_state.x_range), 
                top='state3', width=0.2, 
                source=source, color="#e84d60", legend_label="state3")
    p_state.xaxis.major_label_orientation = "vertical"       
            
    #   ____ mode_i ____
    MODES = ["CONS+", "CONS-", "DIS", "PROD"]
    # x = [(str(t), mode) for t in range(0, exec_game.NUM_PERIODS+1) 
    #                  for mode in MODES ]
    data = {"t": list(map(str,range(0, exec_game.NUM_PERIODS+1))),
            "CONS+": [0]*(exec_game.NUM_PERIODS+1), 
            "CONS-": [0]*(exec_game.NUM_PERIODS+1), 
            "DIS": [0]*(exec_game.NUM_PERIODS+1),
            "PROD": [0]*(exec_game.NUM_PERIODS+1),}
    for num, mode in enumerate(mode_i_s):
        if mode == "CONS+":
            data[mode][num] = 1
        elif mode == "CONS-":
            data[mode][num] = 1
        elif mode == "DIS":
            data[mode][num] = 1
        elif mode == "PROD":
            data[mode][num] = 1
    
    p_mode = figure(x_range= list(map(str,range(0, exec_game.NUM_PERIODS+1))), 
                      plot_height=350, plot_width=550,
                      title="Modes over the time",
                      tools = TOOLS)
    
    source = ColumnDataSource(data=data)
    
    p_mode.vbar(x=dodge('t', -0.25, range=p_mode.x_range), 
                top='CONS+', width=0.2, 
                source=source, color="#c9d9d3", legend_label="CONS+")
    
    p_mode.vbar(x=dodge('t', -0.25, range=p_mode.x_range), 
                top='CONS-', width=0.2, 
                source=source, color="#718dbf", legend_label="CONS-")
    
    p_mode.vbar(x=dodge('t', -0.25, range=p_mode.x_range), 
                top='DIS', width=0.2, 
                source=source, color="#e84d60", legend_label="DIS")
    
    p_mode.vbar(x=dodge('t', -0.25, range=p_mode.x_range), 
                top='PROD', width=0.2, 
                source=source, color="#E1DD8F", legend_label="PROD")
    p_mode.xaxis.major_label_orientation = "vertical"
    
    return [p_state, p_mode]

####################           plot -----> fin

def plot_variables_onehtml(list_of_plt, path_to_variable, ncols = 2, 
                name_file_html="prices_sg_player_attributs_dashboard.html"):
    """
    plot the variables in one html

    Parameters
    ----------
    list_of_plt : list of Figure()
        DESCRIPTION.

    Returns
    -------
    None.

    """
    gp = gridplot(list_of_plt, ncols = ncols, toolbar_location='above')
    # configuration figure
    rep_visu = os.path.join(path_to_variable, "visu")
    Path(rep_visu).mkdir(parents=True, exist_ok=True)   
    output_file(os.path.join(rep_visu, name_file_html))
    
    show(gp)
    
    
def plot_variables_players_game_old(arr_pls_M_T, RUs, pi_sg_plus_s, pi_sg_minus_s):
    """
    A EFFACER
    representation of players' variables and game variables (pi_sg)

    Parameters
    ----------
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*INDEX_ATTRS
        DESCRIPTION.
    RUs : TYPE
        DESCRIPTION.
    pi_sg_plus_s : TYPE
        DESCRIPTION.
    pi_sg_minus_s : TYPE
        DESCRIPTION.

    Returns
    -------
    list of figures

    """
    ps = []
    # ______ plot pi_sg_plus, pi_sg_minus _______
    title = "prices pi_sg_plus, pi_sg_minus" #"price of exported/imported energy inside SG"
    xlabel = "periods of time" 
    ylabel = "price SG"
    key_plus = "pi_sg_plus"
    key_minus = "pi_sg_minus"
    
    args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
          "pi_sg_plus_s": pi_sg_plus_s,"pi_sg_minus_s":pi_sg_minus_s, 
          "key_plus":key_plus, "key_minus":key_minus}
    
    p_pi_sg = plot_pi_X(args)
    ps.append(p_pi_sg)
    
    
    # ______ plot B0s, C0s, RU _______
    title = "real utility of players"
    ylabel = "price"
    BB_i, CC_i, RU_i = fct_aux.compute_real_money_SG(
                            arr_pls_M_T, 
                            pi_sg_plus_s, 
                            pi_sg_minus_s, gmT.INDEX_ATTRS)
    pls = range(0, arr_pls_M_T.shape[0])
    data_pl_ru_bb_cc = {"pl":pls, "RU":RU_i, "BB":BB_i, "CC":CC_i}
    source = ColumnDataSource(data=data_pl_ru_bb_cc)
    TOOLS[7] = HoverTool(tooltips=[
                            ("price", "$y"),
                            ("player", "$x")
                            ]
                        ) 
    p_ru = figure(plot_height = int(HEIGHT*1.0), 
                        plot_width = int(WIDTH*1.5), 
                        title = title,
                        x_axis_label = xlabel, 
                        y_axis_label = ylabel, 
                        x_axis_type = "linear",
                        tools = TOOLS)
    p_ru.line(x="pl", y="RU", source=source, legend_label="RU",
              line_width=2, color=Category20[20][0], line_dash=[1,1])
    p_ru.line(x="pl", y="BB", source=source, legend_label="BB",
              line_width=2, color=Category20[20][2], line_dash=[1,1])
    p_ru.line(x="pl", y="CC", source=source, legend_label="CC",
              line_width=2, color=Category20[20][4], line_dash=[1,1])
    p_ru.circle(x="pl", y="RU", source=source, size=4, fill_color='white')
    p_ru.circle(x="pl", y="BB", source=source, size=4, fill_color='white')
    p_ru.circle(x="pl", y="CC", source=source, size=4, fill_color='white')
    ps.append(p_ru)
    
    # ______ plot CONS_i, PROD_i variables
    CONS_i_s = np.sum(arr_pls_M_T[:,:, fct_aux.INDEX_ATTRIBUTS["cons_i"]],
                      axis=1)
    PROD_i_s = np.sum(arr_pls_M_T[:,:, fct_aux.INDEX_ATTRIBUTS["prod_i"]],
                      axis=1)
    title = "production/consumption of players over periods"
    ylabel = "quantity"
    xlabel = "player"
    pls = range(0, arr_pls_M_T.shape[0])
    data_pl_cons_prod = {"pl":pls, "CONS":CONS_i_s, "PROD":PROD_i_s}
    source = ColumnDataSource(data=data_pl_cons_prod)
    TOOLS[7] = HoverTool(tooltips=[
                            ("quantity", "$y"),
                            ("player", "$x")
                            ]
                        ) 
    p_cons_prod = figure(plot_height = int(HEIGHT*1.0), 
                        plot_width = int(WIDTH*1.5), 
                        title = title,
                        x_axis_label = xlabel, 
                        y_axis_label = ylabel, 
                        x_axis_type = "linear",
                        tools = TOOLS)
    p_cons_prod.line(x="pl", y="CONS", source=source, legend_label="CONS_i",
              line_width=2, color=Category20[20][0], line_dash=[1,1])
    p_cons_prod.line(x="pl", y="PROD", source=source, legend_label="PROD_i",
              line_width=2, color=Category20[20][2], line_dash=[1,1])
    
    ps.append(p_cons_prod)
    
    # ______ plot players' variables _______
    TOOLS[7] = HoverTool(tooltips=[
                            ("quantity", "$y"),
                            ("Time", "$x")
                            ]
                        ) 
    ylabel = "energy quantity"
    ts = list(map(str,range(0, exec_game.NUM_PERIODS+1)))
    for num_pl in range(0, arr_pls_M_T.shape[0]):
        title = "attributs of player pl_"+str(num_pl)
        Pi_s = arr_pls_M_T[num_pl, :, gmT.INDEX_ATTRS["Pi"]]
        Ci_s = arr_pls_M_T[num_pl, :, gmT.INDEX_ATTRS["Ci"]]
        diff_Pi_Ci_s = Pi_s - Ci_s 
        Si_max_s = arr_pls_M_T[num_pl, :, gmT.INDEX_ATTRS["Si_max"]]
        Si_s = arr_pls_M_T[num_pl, :, gmT.INDEX_ATTRS["Si"]]
        Ri_s = Si_max_s - Si_s
        r_i_s = arr_pls_M_T[num_pl, :, gmT.INDEX_ATTRS["r_i"]]
        prod_i_s = arr_pls_M_T[num_pl, :, gmT.INDEX_ATTRS["prod_i"]]
        cons_i_s = arr_pls_M_T[num_pl, :, gmT.INDEX_ATTRS["cons_i"]]
        
        data_vars_pl = {"t":ts, "diff_Pi_Ci": diff_Pi_Ci_s, 
                        "Ri":Ri_s, "r_i":r_i_s, "prod_i":prod_i_s, 
                        "cons_i":cons_i_s}
        source = ColumnDataSource(data=data_vars_pl)
        arr_reshape = np.concatenate((diff_Pi_Ci_s, Ri_s, r_i_s, prod_i_s, cons_i_s))
        min_val = arr_reshape.min()
        max_val = arr_reshape.max()
        p_pl = figure(plot_height = int(HEIGHT*1.0), 
                        plot_width = int(WIDTH*1.5), 
                        title = title,
                        x_axis_label = xlabel, 
                        y_axis_label = ylabel, 
                        x_axis_type = "linear",
                        y_range = Range1d(int(min_val), int(max_val)),
                        tools = TOOLS)
        p_pl.y_range = Range1d(int(min_val), int(max_val))
        # diff_Pi_Ci
        p_pl.line(x="t", y="diff_Pi_Ci", source=source, 
                  legend_label="diff_Pi_Ci",
                  line_width=2, color=Category20[20][0], line_dash=[1,1])
        p_pl.circle(x="t", y="diff_Pi_Ci", source=source, 
                    size=4, fill_color='white')
        
        # prod_i_s
        p_pl.line(x="t", y="prod_i", source=source, 
                  legend_label="prod_i",
                  line_width=2, color=Category20[20][2], line_dash=[1,1])
        p_pl.circle(x="t", y="prod_i", source=source, 
                    size=4, fill_color='white')
        # cons_i_s
        p_pl.line(x="t", y="cons_i", source=source, 
                  legend_label="cons_i",
                  line_width=2, color=Category20[20][4], line_dash=[1,1])
        p_pl.circle(x="t", y="cons_i", source=source, 
                    size=4, fill_color='white')
        # Ri_s
        p_pl.line(x="t", y="Ri", source=source, 
                  legend_label="Ri",
                  line_width=2, color=Category20[20][6], line_dash=[1,1])
        p_pl.circle(x="t", y="Ri", source=source, 
                    size=4, fill_color='white')
        # r_i_s
        p_pl.line(x="t", y="r_i", source=source, 
                  legend_label="r_i",
                  line_width=2, color=Category20[20][8], line_dash=[1,1])
        p_pl.circle(x="t", y="r_i", source=source, 
                    size=4, fill_color='white')
        
        ps.append(p_pl)
        
    return ps

def plot_variables_players_game(arr_pl_M_T, 
                                b0_s, c0_s, 
                                B_is, C_is, 
                                BENs, CSTs, 
                                BB_is, CC_is, RU_is, 
                                pi_sg_plus_s, pi_sg_minus_s, 
                                pi_hp_plus_s, pi_hp_minus_s
                                ):
    """
    representation of players' variables and game variables

    Parameters
    ----------
    arr_pl_M_T : array of shape (M_PLAYERS, NUM_PERIODS+1, len(INDEX_ATTRIBUTS))
        DESCRIPTION.
    b0_s : array of shape (NUM_PERIODS,)
        DESCRIPTION.
        benefit price of one energy unit.
    c0_s : array of shape (NUM_PERIODS,)
        DESCRIPTION.
        cost price of one energy unit.
    B_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        Global benefit of each player
    C_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        Global cost of each player
    BENs : array of shape (M_PLAYERS, NUM_PERIODS)
        DESCRIPTION.
        the benefit of each player at time
    CSTs : array of shape (M_PLAYERS, NUM_PERIODS)
        DESCRIPTION.
        the cost of each player at time
    BB_is : array of shape (M_PLAYERS, ) 
        DESCRIPTION.
        real benefit money of each player inside SG
    CC_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        real cost money of each player inside SG
    RU_is : array of shape (M_PLAYERS, )
        DESCRIPTION.
        the real utility of a player 
    pi_sg_plus_s : array of shape (NUM_PERIODS,)
        DESCRIPTION.
        the price pf exchanged energy inside SG (selling from player to SG)
    pi_sg_minus_s : array of shape (NUM_PERIODS,)
        DESCRIPTION.
        the price pf exchanged energy inside SG (purchasing from SG to player)
    
    Returns
    -------
    ps: list of figures

    """
    
    ps = []
    # ______ plot pi_sg_plus, pi_sg_minus _______
    title = "prices pi_sg_plus, pi_sg_minus" #"price of exported/imported energy inside SG"
    xlabel = "periods of time" 
    ylabel = "price SG"
    key_plus = "pi_sg_plus"
    key_minus = "pi_sg_minus"
    
    args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
          "pi_sg_plus_s": pi_sg_plus_s,
          "pi_sg_minus_s":pi_sg_minus_s, 
          "key_plus":key_plus, "key_minus":key_minus}
    
    p_pi_sg_hp = plot_pi_X(args)
    
    key_plus = "pi_hp_plus"
    key_minus = "pi_hp_minus"
    
    args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
          "pi_sg_plus_s": pi_hp_plus_s,"pi_sg_minus_s":pi_hp_minus_s, 
          "key_plus":key_plus, "key_minus":key_minus}
    p_pi_sg_hp = plot_pi_X(args, p_X=p_pi_sg_hp, styles={"line":"dotted"})
    ps.append(p_pi_sg_hp)
    
    # ______ plot b0_s, c0_s _______
    
    # ______ plot BB_is, CC_is, RU _______
    title = "real utility of players"
    xlabel = "player" 
    ylabel = "price"
    pls = range(0, arr_pl_M_T.shape[0])
    data_pl_ru_bb_cc = {"pl":pls, "RU_i":RU_is, "BB_i":BB_is, "CC_i":CC_is}
    source = ColumnDataSource(data=data_pl_ru_bb_cc)
    TOOLS[7] = HoverTool(tooltips=[
                            ("price", "$y"),
                            ("player", "$x")
                            ]
                        ) 
    p_ru = figure(plot_height = int(HEIGHT*1.0), 
                        plot_width = int(WIDTH*1.5), 
                        title = title,
                        x_axis_label = xlabel, 
                        y_axis_label = ylabel, 
                        x_axis_type = "linear",
                        tools = TOOLS)
    p_ru.line(x="pl", y="RU_i", source=source, legend_label="RU_i",
              line_width=2, color=Category20[20][0], line_dash=[1,1])
    p_ru.line(x="pl", y="BB_i", source=source, legend_label="BB_i",
              line_width=2, color=Category20[20][2], line_dash=[1,1])
    p_ru.line(x="pl", y="CC_i", source=source, legend_label="CC_i",
              line_width=2, color=Category20[20][4], line_dash=[1,1])
    p_ru.circle(x="pl", y="RU_i", source=source, size=4, fill_color='white')
    p_ru.circle(x="pl", y="BB_i", source=source, size=4, fill_color='white')
    p_ru.circle(x="pl", y="CC_i", source=source, size=4, fill_color='white')
    ps.append(p_ru)
    
    # ______ plot BB_is, CC_is pour les forts et faibles consommations
    ind_rows, ind_cols = np.where(arr_pl_M_T[:,:,fct_aux.INDEX_ATTRS["Ci"]] == 10)
    ind_rows = list(set(ind_rows))
    BB_is_weak = BB_is[ind_rows]
    ind_rows, ind_cols = np.where(arr_pl_M_T[:,:,fct_aux.INDEX_ATTRS["Ci"]] == 60)
    ind_rows = list(set(ind_rows))
    BB_is_strong = BB_is[ind_rows] 
    
    ind_rows, ind_cols = np.where(arr_pl_M_T[:,:,fct_aux.INDEX_ATTRS["Ci"]] == 10)
    ind_rows = list(set(ind_rows))
    CC_is_weak = CC_is[ind_rows]
    ind_rows, ind_cols = np.where(arr_pl_M_T[:,:,fct_aux.INDEX_ATTRS["Ci"]] == 60)
    ind_rows = list(set(ind_rows))
    CC_is_strong = CC_is[ind_rows]
    print("ind_rows:{}, BB_is_strong={}, CC_is_strong={}".format(
        ind_rows, BB_is_strong, CC_is_strong)) 
    print("ind_rows:{}, BB_is_weak={}, CC_is_weak={}".format(
        ind_rows, BB_is_weak.shape, CC_is_weak.shape))                      
    
    
    ## BB_is, CC_is WEAK
    title = "real utility of weak players"
    xlabel = "player" 
    ylabel = "price"
    pls = range(0, BB_is_weak.shape[0])
    data_pl_bb_cc_weak = {"pl":pls, "BB_is_weak": BB_is_weak, 
                        "CC_is_weak": CC_is_weak}
    source = ColumnDataSource(data=data_pl_bb_cc_weak)
    TOOLS[7] = HoverTool(tooltips=[
                            ("price", "$y"),
                            ("player", "$x")
                            ]
                        ) 
    p_bb_cc_weak = figure(plot_height = int(HEIGHT*1.0), 
                        plot_width = int(WIDTH*1.5), 
                        title = title,
                        x_axis_label = xlabel, 
                        y_axis_label = ylabel, 
                        x_axis_type = "linear",
                        tools = TOOLS)
    p_bb_cc_weak.line(x="pl", y="BB_is_weak", source=source, 
                      legend_label="BB_is_weak", line_width=2, 
                      color=Category20[20][0], line_dash=[1,1])
    p_bb_cc_weak.line(x="pl", y="CC_is_weak", source=source, 
                      legend_label="CC_is_weak", line_width=2, 
                      color=Category20[20][2], line_dash=[1,1])
    
    
    ## BB_is, CC_is STRONG
    title = "real utility of strong players"
    xlabel = "player" 
    ylabel = "price"
    pls = range(0, BB_is_strong.shape[0])
    data_pl_bb_cc_strong = {"pl":pls, "BB_is_strong": BB_is_strong, 
                        "CC_is_strong": CC_is_strong}
    source = ColumnDataSource(data=data_pl_bb_cc_strong)
    TOOLS[7] = HoverTool(tooltips=[
                            ("price", "$y"),
                            ("player", "$x")
                            ]
                        ) 
    p_bb_cc_strong = figure(plot_height = int(HEIGHT*1.0), 
                        plot_width = int(WIDTH*1.5), 
                        title = title,
                        x_axis_label = xlabel, 
                        y_axis_label = ylabel, 
                        x_axis_type = "linear",
                        tools = TOOLS)
    p_bb_cc_strong.line(x="pl", y="BB_is_strong", source=source, 
                      legend_label="BB_is_strong", line_width=2, 
                      color=Category20[20][0], line_dash=[1,1])
    p_bb_cc_strong.line(x="pl", y="CC_is_strong", source=source, 
                      legend_label="CC_is_strong", line_width=2, 
                      color=Category20[20][2], line_dash=[1,1])
    
    ps.append(p_bb_cc_weak)
    ps.append(p_bb_cc_strong)
    
    # ______ plot CONS_i, PROD_i variables
    CONS_i_s = np.sum(arr_pl_M_T[:,:, fct_aux.INDEX_ATTRS["cons_i"]],
                      axis=1)
    PROD_i_s = np.sum(arr_pl_M_T[:,:, fct_aux.INDEX_ATTRS["prod_i"]],
                      axis=1)
    title = "production/consumption of players over periods"
    ylabel = "quantity"
    xlabel = "player"
    pls = range(0, arr_pl_M_T.shape[0])
    data_pl_cons_prod = {"pl":pls, "CONS":CONS_i_s, "PROD":PROD_i_s}
    source = ColumnDataSource(data=data_pl_cons_prod)
    TOOLS[7] = HoverTool(tooltips=[
                            ("quantity", "$y"),
                            ("player", "$x")
                            ]
                        ) 
    p_cons_prod = figure(plot_height = int(HEIGHT*1.0), 
                        plot_width = int(WIDTH*1.5), 
                        title = title,
                        x_axis_label = xlabel, 
                        y_axis_label = ylabel, 
                        x_axis_type = "linear",
                        tools = TOOLS)
    p_cons_prod.line(x="pl", y="CONS", source=source, legend_label="CONS_i",
              line_width=2, color=Category20[20][0], line_dash=[1,1])
    p_cons_prod.line(x="pl", y="PROD", source=source, legend_label="PROD_i",
              line_width=2, color=Category20[20][2], line_dash=[1,1])
    
    ps.append(p_cons_prod)
    
    # ______ plot players' variables _______
    TOOLS[7] = HoverTool(tooltips=[
                            ("quantity", "$y"),
                            ("Time", "$x"),
                            ("mode", "@mode_i"),
                            ("state", "@state_i"), 
                            ("formule", "@formule"),
                            ("balanced", "@balanced"),
                            ("profile", "@Profile"), 
                            ("case","@Case")
                            ]
                        ) 
    ylabel = "energy quantity"
    xlabel = "time" 
    ts = list(map(str,range(0, arr_pl_M_T.shape[1])))
    for num_pl in range(0, arr_pl_M_T.shape[0]):
        title = "attributs of player pl_"+str(num_pl)+" : profile = " \
                + arr_pl_M_T[num_pl, 0, fct_aux.INDEX_ATTRS["Profili"]] \
                + ", case = " \
                + arr_pl_M_T[num_pl, 0, fct_aux.INDEX_ATTRS["Casei"]]
        Pi_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["Pi"]]
        Ci_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["Ci"]]
        diff_Pi_Ci_s = Pi_s - Ci_s 
        Si_max_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["Si_max"]]
        Si_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["Si"]]
        Ri_s = Si_max_s - Si_s
        R_i_old = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["R_i_old"]]
        Si_old = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["Si_old"]]
        r_i_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["r_i"]]
        prod_i_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["prod_i"]]
        cons_i_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["cons_i"]]
        mode_i_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["mode_i"]]
        state_i_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["state_i"]]
        formule_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["formule"]]
        Profili_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["Profili"]]
        Casei_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["Casei"]]
        balanced_s = arr_pl_M_T[num_pl, :, fct_aux.INDEX_ATTRS["balanced_pl_i"]]
    
        data_vars_pl = {"t":ts, "diff_Pi_Ci": diff_Pi_Ci_s, 
                        "Ri":Ri_s, "r_i":r_i_s, "prod_i":prod_i_s, 
                        "cons_i": cons_i_s, "R_i_old": R_i_old, 
                        "Si_old": Si_old, "mode_i": mode_i_s, 
                        "state_i": state_i_s, "formule":formule_s,
                        "Profile": Profili_s, "Case":Casei_s, 
                        "balanced": balanced_s}
        source = ColumnDataSource(data=data_vars_pl)
        arr_reshape = np.concatenate((diff_Pi_Ci_s, Ri_s, r_i_s, prod_i_s, cons_i_s))
        min_val = arr_reshape.min()
        max_val = arr_reshape.max()
        p_pl = figure(plot_height = int(HEIGHT*1.0), 
                        plot_width = int(WIDTH*1.5), 
                        title = title,
                        x_axis_label = xlabel, 
                        y_axis_label = ylabel, 
                        x_axis_type = "linear",
                        y_range = Range1d(int(min_val), int(max_val)),
                        tools = TOOLS)
        p_pl.y_range = Range1d(int(min_val), int(max_val))
        # diff_Pi_Ci
        p_pl.line(x="t", y="diff_Pi_Ci", source=source, 
                  legend_label="diff_Pi_Ci",
                  line_width=2, color=Category20[20][0], line_dash=[1,1])
        p_pl.circle(x="t", y="diff_Pi_Ci", source=source, 
                    size=4, fill_color='white')
        
        # prod_i_s
        p_pl.line(x="t", y="prod_i", source=source, 
                  legend_label="prod_i",
                  line_width=2, color=Category20[20][2], line_dash=[1,1])
        p_pl.circle(x="t", y="prod_i", source=source, 
                    size=4, fill_color='white')
        # cons_i_s
        p_pl.line(x="t", y="cons_i", source=source, 
                  legend_label="cons_i",
                  line_width=2, color=Category20[20][4], line_dash=[1,1])
        p_pl.circle(x="t", y="cons_i", source=source, 
                    size=4, fill_color='white')
        # Ri_s
        p_pl.line(x="t", y="Ri", source=source, 
                  legend_label="Ri",
                  line_width=2, color=Category20[20][6], line_dash=[1,1])
        p_pl.circle(x="t", y="Ri", source=source, 
                    size=4, fill_color='white')
        # r_i_s
        p_pl.line(x="t", y="r_i", source=source, 
                  legend_label="r_i",
                  line_width=2, color=Category20[20][8], line_dash=[1,1])
        p_pl.circle(x="t", y="r_i", source=source, 
                    size=4, fill_color='white')
        # R_i_old
        p_pl.line(x="t", y="R_i_old", source=source, 
                  legend_label="R_i_old",
                  line_width=2, color=Category20[20][10], line_dash=[1,1])
        p_pl.circle(x="t", y="R_i_old", source=source, 
                    size=4, fill_color='white')
        
        ps.append(p_pl)
        
    return ps
    
    
#------------------------------------------------------------------------------
#                   definitions of unit test of defined functions
#------------------------------------------------------------------------------
def test_get_local_storage_variables():
    name_dir = "tests"
    path_to_variable = fct_aux.find_path_to_variables(
                            name_dir=name_dir, 
                            ext=".npy",
                            threshold=0.89,
                            n_depth=2)
    
    arr_pl_M_T_old, arr_pl_M_T, \
    b0_s, c0_s, \
    B_is, C_is, \
    BENs, CSTs, \
    BB_is, CC_is, RU_is, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    print("____test_get_local_storage_variables____")
    print("arr_pls_M_T: {},RUs={},B0s={},pi_sg_plus_s={}".format(arr_pls_M_T.shape,RUs.shape,B0s.shape,pi_sg_plus_s.shape)) 
    print("arr_pls_M_T: OK") \
        if arr_pls_M_T.shape == (exec_game.M_PLAYERS,
                                 exec_game.NUM_PERIODS+1,
                                 len(gmT.INDEX_ATTRS)) \
        else print("arr_pls_M_T: NOK")
    print("RUs: OK") \
        if RUs.shape == (exec_game.M_PLAYERS,) \
        else print("RUs: NOK")
    print("B0s: OK") \
        if B0s.shape == (exec_game.NUM_PERIODS,) \
        else print("B0s: NOK")
    print("C0s: OK") \
        if C0s.shape == (exec_game.NUM_PERIODS,) \
        else print("C0s: NOK")
    print("pi_sg_plus_s: OK") \
        if pi_sg_plus_s.shape == (exec_game.NUM_PERIODS,) \
        else print("pi_sg_plus_s: NOK")
    print("pi_sg_minus_s: OK") \
        if pi_sg_minus_s.shape == (exec_game.NUM_PERIODS,) \
        else print("pi_sg_minus_s: NOK")
    
def test_plot_pi_sg():
    name_dir = "tests"
    path_to_variable = fct_aux.find_path_to_variables(
                            name_dir=name_dir, 
                            ext=".npy",
                            threshold=0.89,
                            n_depth=2)
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    plot_pi_sg(pi_sg_plus_s, pi_sg_minus_s, path_to_variable)
    
def test_plot_more_prices():
    name_dir = "tests"
    path_to_variable = fct_aux.find_path_to_variables(
                            name_dir=name_dir, 
                            ext=".npy",
                            threshold=0.89,
                            n_depth=2)
    p_sg, p_B0_C0s = plot_more_prices(path_to_variable)
    
def test_plot_player():    
    name_dir = "tests"
    path_to_variable = fct_aux.find_path_to_variables(
                            name_dir=name_dir, 
                            ext=".npy",
                            threshold=0.89,
                            n_depth=2)
    
    
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    ps_pls = plot_player(arr_pls_M_T, RUs, BENs, CSTs, 
                         path_to_variable, 5)
    
def test_plot_variables_onehtml(number_of_players=5):
    """

    Parameters
    ----------
    number_of_players : integer, optinal   
        DESCRIPTION.
         it's number of players you will selected to arr_pls_M_T. we obtain 
         an array called arr_pls_M_T_nop

    Returns
    -------
    None.

    """

    name_dir = "tests"
    path_to_variable = fct_aux.find_path_to_variables(
                            name_dir=name_dir, 
                            ext=".npy",
                            threshold=0.89,
                            n_depth=2)
    
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    id_pls = np.random.choice(arr_pls_M_T.shape[0], number_of_players, 
                              replace=False)
    
    p_sg, p_B0_C0s = plot_more_prices(path_to_variable, dbg=False)
    ps_pls = plot_player(arr_pls_M_T, RUs, BENs, CSTs, id_pls,
                         path_to_variable, dbg=False)
    ps_pls_prod_conso = plot_prod_cons_player(arr_pls_M_T, id_pls, 
                          path_to_variable, dbg=True)
    
    flat_list = [item 
                 for sublist in [[p_sg],[p_B0_C0s],ps_pls, ps_pls_prod_conso] 
                 for item in sublist]
    
    plot_variables_onehtml(list_of_plt=flat_list, 
                           path_to_variable=path_to_variable)
    
def test_plot_variables_onehtml_allcases(number_of_players=5):
    """
    TODO : RETURN THE ERROR MESSAGE 
    RuntimeError: Models must be owned by only a single document, BoxAnnotation(id='220268', ...) is already in a doc
    
    Parameters
    ----------
    number_of_players : integer, optinal   
        DESCRIPTION.
         it's number of players you will selected to arr_pls_M_T. we obtain 
         an array called arr_pls_M_T_nop

    Returns
    -------
    None.

    """
    name_dir = "tests"
    reps = os.listdir(name_dir)
    rep = reps[np.random.randint(0, len(reps))]
    cases = [exec_game.CASE3, exec_game.CASE2, exec_game.CASE1]
    for case in cases:
        str_case = str(case[0]) +"_"+ str(case[1])
        path_to_variable = os.path.join(name_dir, rep, str_case)
        
        arr_pls_M_T, RUs, \
        B0s, C0s, \
        BENs, CSTs, \
        pi_sg_plus_s, pi_sg_minus_s = \
            get_local_storage_variables(path_to_variable)
            
        id_pls = np.random.choice(arr_pls_M_T.shape[0], number_of_players, 
                                  replace=False)
        
        p_sg, p_B0_C0s = plot_more_prices(path_to_variable, dbg=False)
        ps_pls = plot_player(arr_pls_M_T, RUs, BENs, CSTs, id_pls,
                             path_to_variable, dbg=False)
        ps_pls_prod_conso = plot_prod_cons_player(arr_pls_M_T, id_pls, 
                              path_to_variable, dbg=True)
        
        flat_list = [item 
                     for sublist in [[p_sg],[p_B0_C0s],ps_pls, ps_pls_prod_conso] 
                     for item in sublist]
        plot_variables_onehtml(list_of_plt=flat_list, 
                               path_to_variable=path_to_variable,
                               name_file_html="representation_attributs_players_allcases.html")
        
def test_plot_variables_onecase(rep,case):
    name_dir = "tests"
    if not os.path.isdir(os.path.join(name_dir, rep)):
        reps = os.listdir(name_dir)
        rep = reps[np.random.randint(0, len(reps))]
    str_case = str(case[0]) +"_"+ str(case[1])
    if not os.path.isdir(os.path.join(name_dir, rep, str_case)):
        reps = os.listdir(rep)
        str_case = reps[np.random.randint(0, len(reps))]
    path_to_variable = os.path.join(name_dir, rep, str_case)
    
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    id_pls = np.random.choice(arr_pls_M_T.shape[0], number_of_players, 
                              replace=False)
    
    p_sg, p_B0_C0s = plot_more_prices(path_to_variable, dbg=False)
    ps_pls = plot_player(arr_pls_M_T, RUs, BENs, CSTs, id_pls,
                         path_to_variable, dbg=False)
    ps_pls_prod_conso = plot_prod_cons_player(arr_pls_M_T, id_pls, 
                          path_to_variable, dbg=True)
    
    flat_list = [item 
                 for sublist in [[p_sg],[p_B0_C0s],ps_pls, ps_pls_prod_conso] 
                 for item in sublist]
    plot_variables_onehtml(list_of_plt=flat_list, 
                           path_to_variable=path_to_variable,
                           name_file_html="representation_attributs_player_onecase.html")
    
def test_plot_variables_oneplayer(rep, case):
    """
    plot variable figures for one random player 
    """      
    name_dir = "tests"
    if not os.path.isdir(os.path.join(name_dir, rep)):
        reps = os.listdir(name_dir)
        rep = reps[np.random.randint(0, len(reps))]
    str_case = str(case[0]) +"_"+ str(case[1])
    if not os.path.isdir(os.path.join(name_dir, rep, str_case)):
        reps = os.listdir(rep)
        str_case = reps[np.random.randint(0, len(reps))]
    path_to_variable = os.path.join(name_dir, rep, str_case)
    
    # import arr_pls_M_T
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    # selected player
    id_pl = np.random.choice(arr_pls_M_T.shape[0], size=1, replace=False)
    
    # select values for all time
    arr_pls_i_T_nop = arr_pls_M_T[id_pl,:,:]
    ben_i_T = BENs[id_pl].T.reshape(-1)
    cst_i_T = CSTs[id_pl].T.reshape(-1)
    gamma_i_T = arr_pls_M_T[id_pl,:,gmT.INDEX_ATTRS["gamma_i"]].T.reshape(-1)
    r_i_T = arr_pls_M_T[id_pl,:,gmT.INDEX_ATTRS["r_i"]].T.reshape(-1)
    state_i_T = arr_pls_M_T[id_pl,:,gmT.INDEX_ATTRS["state_i"]].T.reshape(-1)
    mode_i_T = arr_pls_M_T[id_pl,:,gmT.INDEX_ATTRS["mode_i"]].T.reshape(-1)
               
    #_____ plot ben_i, cst_i ______
    title = "benefit/cost for a player pl_"+str(id_pl)+" inside SG"
    xlabel = "periods of time" 
    ylabel = "price"
    key_plus = "ben_i"
    key_minus = "cst_i"
    
    args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
          "pi_sg_plus_s": ben_i_T,"pi_sg_minus_s": cst_i_T, 
          "key_plus": key_plus, "key_minus": key_minus}
    p_ben_cst = plot_pi_X(args, p_X=None)
     
    #_____ plot prod_i, cst_i _____
    ps_pls_prod_conso = plot_prod_cons_player(arr_pls_M_T, id_pl, 
                          path_to_variable, dbg=True)
    
    #_____ plot gamma_i, r_i    ______
    title = "storage decision and free place for a player pl_"+str(id_pl)+" inside SG"
    xlabel = "periods of time" 
    ylabel = "quantity"
    key_plus = "gamma_i"
    key_minus = "r_i"

    args={"title": title, "xlabel": xlabel, "ylabel": ylabel,
          "pi_sg_plus_s": gamma_i_T,"pi_sg_minus_s": r_i_T, 
          "key_plus": key_plus, "key_minus": key_minus}
    p_gamma_r = plot_pi_X(args, p_X=None)
    
    #_____ plot state_i, mode_i ______
    p_state_mode = plot_state_mode(state_i_T, mode_i_T)
    
    #_____ plot gridplot _____
    list_of_plt = [[p_ben_cst], ps_pls_prod_conso, [p_gamma_r],
                   p_state_mode]
    flat_list = [item 
                 for sublist in list_of_plt
                 for item in sublist]
    plot_variables_onehtml(list_of_plt=flat_list, 
                           path_to_variable=path_to_variable,
                           name_file_html="representation_attributs_oneplayer_onecase.html")
    
def test_plot_variables_allplayers_old(rep, case):
    """
    IL NEXISTE PLUS DE REPERTOIRES "CASE"  
    plot variable figures for one random player 
    """      
    name_dir = "tests"
    if not os.path.isdir(os.path.join(name_dir, rep)):
        reps = os.listdir(name_dir)
        rep = reps[np.random.randint(0, len(reps))]
    str_case = str(case[0]) +"_"+ str(case[1])
    if not os.path.isdir(os.path.join(name_dir, rep, str_case)):
        reps = os.listdir(rep)
        str_case = reps[np.random.randint(0, len(reps))]
    path_to_variable = os.path.join(name_dir, rep, str_case)
    path_to_variable = "tests/simu_1510_1736/scenario1/0.3/"
    
    # import arr_pls_M_T
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
    pi_sg_plus_s, pi_sg_minus_s = \
        get_local_storage_variables(path_to_variable)
        
    ps = plot_variables_players_game(arr_pls_M_T, RUs, 
                                     pi_sg_plus_s, pi_sg_minus_s)
    
    plot_variables_onehtml(ps, path_to_variable, ncols = 1, 
                name_file_html="player_attributs_game_variables_dashboard.html")
    
def test_plot_variables_allplayers(rep="debug", 
                                   num_period_to_show=50,
                                   num_player_to_show=50, 
                                   name_dir="tests", 
                                   date_hhmm="2010_1258",
                                   probCi=0.3,
                                   scenario="scenario1",
                                   pi_hp_plus=15,
                                   pi_hp_minus=4):
    """
    Plot the variables of games and also the attributs of players

    Parameters
    ----------
    rep : name of simulation directory
        DESCRIPTION.
    num_period_to_show : integer,
        DESCRIPTION.
        Number of periods to display
    num_player_to_show : integer,
        DESCRIPTION.
        Number of players to display

    Returns
    -------
    None.

    """    
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    path_to_variable = os.path.join(name_dir, "simu_"+date_hhmm, scenario, 
                                   str(probCi), msg)
    
    # import variables for file
    arr_pl_M_T_old, arr_pl_M_T, \
    b0_s, c0_s, \
    B_is, C_is, \
    BENs, CSTs, \
    BB_is, CC_is, RU_is, \
    pi_sg_plus_s, pi_sg_minus_s, \
    pi_hp_plus_s, pi_hp_minus_s= \
        get_local_storage_variables(path_to_variable)
    
    ps = plot_variables_players_game(
                arr_pl_M_T[:num_player_to_show,:num_period_to_show], 
                b0_s[:num_period_to_show], 
                c0_s[:num_period_to_show], 
                B_is[:num_player_to_show], 
                C_is[:num_player_to_show], 
                BENs[:num_player_to_show,:num_period_to_show], 
                CSTs[:num_player_to_show,:num_period_to_show], 
                BB_is[:num_player_to_show], 
                CC_is[:num_player_to_show], 
                RU_is[:num_player_to_show], 
                pi_sg_plus_s[:num_period_to_show], 
                pi_sg_minus_s[:num_period_to_show],
                pi_hp_plus_s[:num_period_to_show], 
                pi_hp_minus_s[:num_period_to_show])
    plot_variables_onehtml(ps, path_to_variable, ncols = 1, 
                name_file_html="player_attributs_game_variables_dashboard.html")
    
#------------------------------------------------------------------------------
#                   execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    number_of_players = 5
    #test_get_local_storage_variables()
    #test_plot_pi_sg()
    #test_plot_more_prices()
    #test_plot_player()
    
    #test_plot_variables_onehtml(number_of_players)
    #test_plot_variables_onehtml_allcases(number_of_players=5)
    # test_plot_variables_onecase(rep="simu_0510_1817",case=exec_game.CASE1)
    #test_plot_variables_oneplayer(rep="simu_0510_1817",case=exec_game.CASE3)
    
    #test_plot_variables_allplayers(rep="simu_0510_1817",case=exec_game.CASE1)
    
    #test_plot_variables_allplayers(rep="simu_1410_0859")
    #test_plot_variables_allplayers(rep="simu_1410_0931")
    name_dir="tests"
    date_hhmm="2010_1258"
    probCi=0.3
    scenario="scenario1"
    pi_hp_plus=15
    pi_hp_minus=4
    test_plot_variables_allplayers(num_period_to_show=50,
                                   num_player_to_show=50,
                                   name_dir=name_dir, 
                                   date_hhmm=date_hhmm,
                                   probCi=probCi,
                                   scenario=scenario,
                                   pi_hp_plus=pi_hp_plus,
                                   pi_hp_minus=pi_hp_minus)
    print("runtime {}".format(time.time()-ti))
    

