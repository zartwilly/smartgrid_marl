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
from bokeh.models import ColumnDataSource, FactorRange;
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
              line_width=2, color=COLORS[0], line_dash=styles["line"])
    p_X.circle(x="t", y=args["key_plus"], source=src, 
               size=4, fill_color='white')
    p_X.line(x="t", y=args["key_minus"], source=src, 
              legend_label= args["key_minus"],
              line_width=2, color=COLORS[1], line_dash=styles["line"])
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

def plot_state_mode_old(state_i_s, mode_i_s):
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
    x = [(str(t), state) for t in range(0, exec_game.NUM_PERIODS+1) 
                     for state in STATES ]
    data = {"state1": [0]*(exec_game.NUM_PERIODS+1), 
            "state2": [0]*(exec_game.NUM_PERIODS+1), 
            "state3": [0]*(exec_game.NUM_PERIODS+1)}
    for num, state in enumerate(state_i_s):
        if state == "state1":
            data["state1"][num] = 1
        elif state == "state2":
            data["state2"][num] = 1
        elif state == "state3":
            data["state3"][num] = 1
    counts = sum(zip(data['state1'], data['state2'], data['state3']), ())
    source = ColumnDataSource(
                data=dict(x=x,
                          counts=counts))
    
    #Initializing our plot
    p_state = figure(x_range=FactorRange(*x), 
                     plot_height=350, plot_width=1950,
                     title="States over the time",
                     tools = TOOLS)
    #Plotting our vertical bar chart
    p_state.vbar(x='x', top='counts', width=0.9, source=source,
           # use the palette to colormap based on the the x[1:2] values
           fill_color=factor_cmap('x', palette=colors, factors=STATES, 
                              start=1, end=2))
    #Enhancing our graph
    p_state.y_range.start = 0
    p_state.x_range.range_padding = 0.1
    p_state.xaxis.major_label_orientation = .9
    p_state.xgrid.grid_line_color = None
    p_state.legend.location = "top_right"
    
    #   ____ mode_i ____
    MODES = ["CONS+", "CONS-", "DIS", "PROD"]
    x = [(str(t), mode) for t in range(0, exec_game.NUM_PERIODS+1) 
                     for mode in MODES ]
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
    counts = sum(zip(data['CONS+'], data['CONS-'], data['DIS'], data['PROD']), 
                  ())
    source = ColumnDataSource(
                data=dict(x=x,
                          counts=counts))
    
    #Initializing our plot
    p_mode = figure(x_range=FactorRange(*x), 
                      plot_height=350, plot_width=1950,
                      title="Modes over the time",
                      tools = TOOLS)
    #Plotting our vertical bar chart
    p_mode.vbar(x='x', top='counts', width=0.9, source=source,
            # use the palette to colormap based on the the x[1:2] values
            fill_color=factor_cmap('x', palette=colors, factors=MODES, 
                              start=1, end=2))
    #Enhancing our graph
    p_mode.y_range.start = 0
    p_mode.x_range.range_padding = 0.1
    p_mode.xaxis.major_label_orientation = .9
    p_mode.xgrid.grid_line_color = None
    p_mode.legend.location = "top_right"
    p_mode.legend.orientation = "horizontal"
    # # Initializing our plot
    # p_mode = figure(x_range= list(map(str,range(0, exec_game.NUM_PERIODS+1))), 
    #                   plot_height=350, plot_width=550,
    #                   title="Modes over the time",
    #                   tools = TOOLS)
    
    # source = ColumnDataSource(data=data)
    
    # p_mode.vbar(x=dodge('t', -0.25, range=p_mode.x_range), 
    #             top='CONS+', width=0.2, 
    #             source=source, color="#c9d9d3", legend_label="CONS+")
    
    # p_mode.vbar(x=dodge('t', -0.25, range=p_mode.x_range), 
    #             top='CONS-', width=0.2, 
    #             source=source, color="#718dbf", legend_label="CONS-")
    
    # p_mode.vbar(x=dodge('t', -0.25, range=p_mode.x_range), 
    #             top='DIS', width=0.2, 
    #             source=source, color="#e84d60", legend_label="DIS")
    
    # p_mode.vbar(x=dodge('t', -0.25, range=p_mode.x_range), 
    #             top='PROD', width=0.2, 
    #             source=source, color="#E1DD8F", legend_label="PROD")
    
    return [p_state, p_mode]
        
def plot_state_mode_old_old(state_i_s, mode_i_s):
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
    T = range(0, exec_game.NUM_PERIODS+1)
    count_states = sum(zip([int(state[-1]) for state in state_i_s]),())
    x = list(zip(map(str,T), state_i_s))
    source = ColumnDataSource(
                data=dict(x=x,
                          counts=count_states,
                          color=random.sample(Turbo256, 
                                              exec_game.NUM_PERIODS+1 )))
    
    #Initializing our plot
    p = figure(x_range=FactorRange(*x), 
               plot_height=350, 
               title="States over the time")
    #Plotting our vertical bar chart
    p.vbar(x='x', top='counts', width=0.9 ,fill_color='color', source=source)
    #Enhancing our graph
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = .9
    p.xgrid.grid_line_color = None
    return p
####################           plot -----> fin

def plot_variables_onehtml(list_of_plt, path_to_variable):
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
    gp = gridplot(list_of_plt, ncols = 2, toolbar_location='above')
    # configuration figure
    rep_visu = os.path.join(path_to_variable, "visu")
    Path(rep_visu).mkdir(parents=True, exist_ok=True)   
    output_file(os.path.join(rep_visu,
                             "prices_sg_player_attributs_dashboard.html"))
    
    show(gp)
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
    
    arr_pls_M_T, RUs, \
    B0s, C0s, \
    BENs, CSTs, \
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
                               path_to_variable=path_to_variable)
        
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
                           path_to_variable=path_to_variable)
    
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
                           path_to_variable=path_to_variable)
#------------------------------------------------------------------------------
#                   execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    number_of_players = 5
    test_get_local_storage_variables()
    #test_plot_pi_sg()
    #test_plot_more_prices()
    #test_plot_player()
    
    #test_plot_variables_onehtml(number_of_players)
    #test_plot_variables_onehtml_allcases(number_of_players=5)
    #test_plot_variables_onecase(rep="simu_0510_1817",case=exec_game.CASE3)
    test_plot_variables_oneplayer(rep="simu_0510_1817",case=exec_game.CASE1)
    print("runtime {}".format(time.time()-ti))
    

