# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:45:38 2020

@author: jwehounou
"""
import os
import sys
import time
import json
import string
import random
import numpy as np
import itertools as it

#------------------------------------------------------------------------------
#                       definition of constantes
#------------------------------------------------------------------------------
M_PLAYERS = 10
NUM_PERIODS = 50

CHOICE_RU = 1

LOW_VAL_Ci = 100 
HIGH_VAL_Ci = 300

STATE1_STRATS = ("CONS+", "CONS-")                                             # strategies possibles pour l'etat 1 de a_i
STATE2_STRATS = ("DIS", "CONS-")                                               # strategies possibles pour l'etat 2 de a_i
STATE3_STRATS = ("DIS", "PROD")

CASE1 = (0.75, 1.5)
CASE2 = (0.4, 0.75)
CASE3 = (0, 0.3)

PROFIL_H = (0.6, 0.2, 0.2)
PROFIL_M = (0.2, 0.6, 0.2)
PROFIL_L = (0.2, 0.2, 0.6)

INDEX_ATTRS = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, "gamma_i":4, 
               "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9,
               "Profili":10, "Casei":11, "R_i_old":12}

#------------------------------------------------------------------------------
#           definitions of class
#------------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
#------------------------------------------------------------------------------
#           definitions of functions
#------------------------------------------------------------------------------

def fct_positive(sum_list1, sum_list2):
    """
    sum_list1 : sum of items in the list1
    sum_list2 : sum of items in the list2
    
    difference between sum of list1 et sum of list2 such as :
         diff = 0 if sum_list1 - sum_list2 <= 0
         diff = sum_list1 - sum_list2 if sum_list1 - sum_list2 > 0

        diff = 0 if sum_list1 - sum_list2 <= 0 else sum_list1 - sum_list2
    Returns
    -------
    return 0 or sum_list1 - sum_list2
    
    """
    
    # boolean = sum_list1 - sum_list2 > 0
    # diff = boolean * (sum_list1 - sum_list2)
    diff = 0 if sum_list1 - sum_list2 <= 0 else sum_list1 - sum_list2
    return diff

def generate_energy_unit_price_SG(pi_hp_plus, pi_hp_minus):
    """
    generate intern cost and intern price of one unit of energy inside SG

    Returns
    -------
    pi_0_plus, pi_0_minus

    """
    rd_num = np.random.random()
    pi_0_plus = pi_hp_plus * rd_num
    pi_0_minus = pi_hp_minus * rd_num
    return pi_0_plus, pi_0_minus

def compute_utility_players(arr_pls_M_T, gamma_is, t, b0, c0):
    """
    calculate the benefit and the cost of each player at time t

    Parameters
    ----------
    arr_pls_M_T : array of shape M_PLAYERS*NUM_PERIODS*9
        DESCRIPTION.
    gamma_is :  array of shape (M_PLAYERS,)
        DESCRIPTION.
    t : integer
        DESCRIPTION.
    b0 : float
        benefit per unit.
    c0 : float
        cost per unit.

    Returns
    -------
    bens: benefits of M_PLAYERS, shape (M_PLAYERS,).
    csts: costs of M_PLAYERS, shape (M_PLAYERS,)
    """
    bens = b0 * arr_pls_M_T[:, t, INDEX_ATTRS["prod_i"]] \
            + gamma_is * arr_pls_M_T[:, t, INDEX_ATTRS["r_i"]]
    csts = c0 * arr_pls_M_T[:, t, INDEX_ATTRS["cons_i"]]
    return bens, csts     


# def generate_Cis_Pis_Sis(n_items, low_1, high_1, low_2, high_2):
#     """
#     generate Cis, Pis, Sis and Si_maxs.
#     each variable has 1*n_items shape
#     low_1 and high_1 are the limit of Ci items generated
#     low_2 and high_2 are the limit of Pi, Si, Si_max items generated
    
#     return:
#         Cis, Pis, Si_maxs, Sis
#     """
#     Cis = np.random.uniform(low=low_1, high=high_1, 
#                             size=(1, n_items))
    
#     low = low_2; high = high_2
#     # Pi
#     inters = map(lambda x: (low*x, high*x), Cis.reshape(-1))
#     Pis = np.array([np.random.uniform(low=low_item, high=high_item) 
#                     for (low_item,high_item) in inters]).reshape((1,-1))
#     # Si
#     inters = map(lambda x: (low*x, high*x), Pis.reshape(-1))
#     Si_maxs = np.array([np.random.uniform(low=low_item, high=high_item) 
#                     for (low_item,high_item) in inters]).reshape((1,-1))
#     inters = map(lambda x: (low*x, high*x), Si_maxs.reshape(-1))
#     Sis = np.array([np.random.uniform(low=low_item, high=high_item) 
#                     for (low_item,high_item) in inters]).reshape((1,-1))
    
#     ## code initial 
#     # Cis = np.random.uniform(low=LOW_VAL_Ci, high=HIGH_VAL_Ci, 
#     #                         size=(1, M_PLAYERS))
    
#     # low = sys_inputs['case'][0]; high = sys_inputs['case'][1]
#     # # Pi
#     # inters = map(lambda x: (low*x, high*x), Cis.reshape(-1))
#     # Pis = np.array([np.random.uniform(low=low_item, high=high_item) 
#     #                 for (low_item,high_item) in inters]).reshape((1,-1))
#     # # Si
#     # inters = map(lambda x: (low*x, high*x), Pis.reshape(-1))
#     # Si_maxs = np.array([np.random.uniform(low=low_item, high=high_item) 
#     #                 for (low_item,high_item) in inters]).reshape((1,-1))
#     # inters = map(lambda x: (low*x, high*x), Si_maxs.reshape(-1))
#     # Sis = np.array([np.random.uniform(low=low_item, high=high_item) 
#     #                 for (low_item,high_item) in inters]).reshape((1,-1))
#     return Cis, Pis, Si_maxs, Sis

# def generate_Cis_Pis_Sis_oneplayer_alltime(num_player, num_periods, 
#                                            low_Ci, high_Ci):
#     """
#     create initial values for a player attributs

#     Parameters
#     ----------
#     num_player : integer
#         DESCRIPTION.
#         number of player
#     n_periods : integer
#         DESCRIPTION.
#         number of periods in the time
#     low_Ci : integer
#         DESCRIPTION.
#         low value of Ci, Ci constante all the periods
#     high_Ci : integer
#         DESCRIPTION.
#         high value of Ci, Ci constante all the periods
        
#     Returns
#     -------

#     arr_pl_i_T : list of (num_periods+1, len(init_values_i_t))
#         DESCRIPTION.
#         avec 
#         init_values_i_t = [Ci_t, Pi_t, Si_t, Si_max_t, str_profil_t, str_case_t]
        
#         avec forall t, Ci_t=Ci_t+1, Si_max_t = Si_max_t+1
#     """
#     arr_pl_i_T = []
    
#     # Ci = np.random.uniform(low=low_Ci, high=high_Ci)
#     # prob = np.random.uniform(0,1)
#     # Si_max = Ci * 0.8 if prob <= 0.3 else Ci * 0.5
    
    
#     for t in range(0, num_periods+1):
#         profil_t= None
#         Ci = 0; Si_max = 0
#         prob = np.random.uniform(0,1)
#         if prob <= 0.3:
#             profil_t = np.random.default_rng().choice(
#                             p=[0.5, 0.5],
#                             a=[PROFIL_L, PROFIL_H])
#             Ci = 10
#             Si_max = Ci * 0.8
#         else:
#             profil_t = np.random.default_rng().choice(
#                             p=[0.5, 0.5],
#                             a=[PROFIL_L, PROFIL_M])
#             Ci = 60
#             Si_max = Ci * 0.5
            
#         profil_case_t = None
#         prob_case_t = np.random.uniform(0,1)
#         if prob_case_t <= profil_t[0]:
#             profil_case_t = CASE1
#         elif prob_case_t > profil_t[0] \
#             and prob_case_t <= profil_t[0]+profil_t[1]:
#             profil_case_t = CASE2
#         else:
#             profil_case_t = CASE3
        
#         min_val_profil = profil_case_t[0]*Ci 
#         max_val_profil = profil_case_t[1]*Ci
#         Pi_t = np.random.uniform(low=min_val_profil, high=max_val_profil) 
                            
#         Si_t = 0 if t == 0 else np.random.uniform(0,1) * Si_max
        
#         str_profil_t = "_".join(map(str, profil_t))
#         str_case_t = "_".join(map(str, profil_case_t))
        
#         init_values_i_t = [Ci, Pi_t, Si_t, Si_max,
#                            0, 0, 0, 0, "", "", 
#                            str_profil_t, str_case_t, Si_max - Si_t]
        
#         arr_pl_i_T.append(init_values_i_t)
        
#     return arr_pl_i_T
      
# def generate_Cis_Pis_Sis_allplayer_alltime(m_players, num_periods, 
#                                             low_Ci, high_Ci):
#     """
#     create initial values for all player attributs all the time


#     Parameters
#     ----------
#     m_players : integer
#         DESCRIPTION.
#         number of players in the game
#     n_periods : integer
#         DESCRIPTION.
#         number of periods in the time
#     low_Ci : integer
#         DESCRIPTION.
#         low value of Ci, Ci constante all the periods
#     high_Ci : integer
#         DESCRIPTION.
#         high value of Ci, Ci constante all the periods

#     Returns
#     -------
#     arr_pl_M_T : array of (num_players, num_periods+1, len(init_values_i_t))
#         DESCRIPTION.
#         avec 
#         init_values_i_t = [Ci_t, Pi_t, Si_t, Si_max_t, str_profil_t, str_case_t]
        
#         avec forall t, Ci_t = Ci_t+1, Si_max_t = Si_max_t+1
#     """
    
#     arr_pl_M_T = []
#     for num_player in range(0, m_players):
#         arr_pl_i_T = generate_Cis_Pis_Sis_oneplayer_alltime(
#                         num_player, num_periods, 
#                         low_Ci, high_Ci)
#         arr_pl_M_T.append(arr_pl_i_T)
    
#     arr_pl_M_T = np.array( arr_pl_M_T, dtype=object)
#     return arr_pl_M_T

# __________    generate Cis, Pis, Si_maxs and Sis --> debut   ________________
def generate_Pi_Ci_Si_Simax_by_profil_scenario(
                                    m_players=3, num_periods=5, 
                                    scenario="scenario1", prob_Ci=0.3, 
                                    Ci_low=10, Ci_high=60):
    """
    create the initial values of all players at all time intervals

    Parameters
    ----------
    m_players : Integer optional
        DESCRIPTION. The default is 3.
        the number of players
    num_periods : Integer, optional
        DESCRIPTION. The default is 5.
        the number of time intervals 
    scenario : String, optional
        DESCRIPTION. The default is "scenario1".
        indicate the scenario to play
    prob_Ci : float, optional
        DESCRIPTION. The default is 0.3.
        the probability of choosing the type of players' consumption
    Ci_low : float, optional
        DESCRIPTION. The default is 10.
        the min value of the consumption
    Ci_high : float, optional
        DESCRIPTION. The default is 60.
        the max value of the consumption
    Returns
    -------
    arr_pl_M_T : array of shape (M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))
        DESCRIPTION.

    """
    arr_pl_M_T = []
    for num_pl, prob in enumerate(np.random.uniform(0, 1, size=m_players)):
        Ci = None; profili = None
        if prob < prob_Ci:
            Ci = Ci_low
            Si_max = 0.8 * Ci
            if scenario == "scenario1":
                profili = PROFIL_L
            elif scenario == "scenario2":
                profili = PROFIL_H
            elif scenario == "scenario3":
                profili = PROFIL_M
        else:
            Ci = Ci_high
            Si_max = 0.5 * Ci
            if scenario == "scenario1":
                profili = PROFIL_H
            elif scenario == "scenario2":
                profili = PROFIL_M
            elif scenario == "scenario3":
                profili = np.random.default_rng().choice(
                            p=[0.5, 0.5],
                            a=[PROFIL_H, PROFIL_M])
            
        profil_casei = None
        prob_casei = np.random.uniform(0,1)
        if prob_casei < profili[0]:
            profil_casei = CASE1
        elif prob_casei >= profili[0] \
            and prob_casei < profili[0]+profili[1]:
            profil_casei = CASE2
        else:
            profil_casei = CASE3
        # profil_casei = None
        # if prob < profili[0]:
        #     profil_casei = CASE1
        # elif prob >= profili[0] \
        #     and prob < profili[0]+profili[1]:
        #     profil_casei = CASE2
        # else:
        #     profil_casei = CASE3
                
        min_val_profil = profil_casei[0]*Ci 
        max_val_profil = profil_casei[1]*Ci
        
        Pi_s = list( np.around(np.random.uniform(
                                low=min_val_profil, 
                                high=max_val_profil, 
                                size=(num_periods,)
                                ), decimals=2) )
        Si_s = list( np.around(np.random.uniform(0,1,size=(num_periods,))*Si_max,
                               decimals=2))
        Si_s[0] = 0; 
        str_profili_s = ["_".join(map(str, profili))] * num_periods
        str_casei_s = ["_".join(map(str, profil_casei))] * num_periods
        
        # building list of list 
        Ci_s = [Ci] * num_periods
        Si_max_s = [Si_max] * num_periods
        gamma_i_s, r_i_s = [0]*num_periods, [0]*num_periods
        prod_i_s, cons_i_s = [0]*num_periods, [0]*num_periods
        state_i_s, mode_i_s = [""]*num_periods, [""]*num_periods
        R_i_old_s = [round(x - y, 2) for x, y in zip(Si_max_s, Si_s)]
        init_values_i_s = list(zip(Ci_s, Pi_s, Si_s, Si_max_s, gamma_i_s, 
                                   prod_i_s, cons_i_s, r_i_s, state_i_s, 
                                   mode_i_s, str_profili_s, str_casei_s, 
                                   R_i_old_s))
        arr_pl_M_T.append(init_values_i_s)
    
    arr_pl_M_T = np.array(arr_pl_M_T, dtype=object)
    
    return arr_pl_M_T

def generer_Pi_Ci_Si_Simax_for_all_scenarios(scenarios=["scenario1"], 
                                    m_players=3, num_periods=5, 
                                    prob_Ci=0.3, Ci_low=10, Ci_high=60):
    """
    create the variables for all scenarios

    Parameters
    ----------
    scenarios : list of String, optional
        DESCRIPTION. The default is [].
    m_players : Integer optional
        DESCRIPTION. The default is 3.
        the number of players
    num_periods : Integer, optional
        DESCRIPTION. The default is 5.
        the number of time intervals 
    prob_Ci : float, optional
        DESCRIPTION. The default is 0.3.
        the probability of choosing the type of players' consumption
    Ci_low : float, optional
        DESCRIPTION. The default is 10.
        the min value of the consumption
    Ci_high : float, optional
        DESCRIPTION. The default is 60.
        the max value of the consumption

    Returns
    -------
    list of arrays. 
    Each array has a shape ((M_PLAYERS, NUM_PERIODS, len(INDEX_ATTRS))).

    """
    l_arr_pl_M_T = []
    for scenario in scenarios:
        arr_pl_M_T = generate_Pi_Ci_Si_Simax_by_profil_scenario(
                            m_players=m_players, num_periods=num_periods, 
                            scenario=scenario, prob_Ci=prob_Ci, 
                            Ci_low=Ci_low, Ci_high=Ci_high)
        l_arr_pl_M_T.append(arr_pl_M_T)
        
    return l_arr_pl_M_T

# __________    generate Cis, Pis, Si_maxs and Sis --> fin   ________________

# __________    look for whether pli is balanced or not --> debut  ____________
def balanced_player(pl_i, thres=0.1, dbg=False):
    """
    verify if pl_i is whether balanced or unbalanced

    Parameters
    ----------
    pl_i : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    Pi = pl_i.get_Pi(); Ci = pl_i.get_Ci(); Si = pl_i.get_Si(); 
    Si_max = pl_i.get_Si_max(); R_i_old = pl_i.get_R_i_old()
    state_i = pl_i.get_state_i(); 
    mode_i = pl_i.get_mode_i()
    cons_i = pl_i.get_cons_i(); prod_i = pl_i.get_prod_i()
    
    if dbg:
        print("_____ balanced_player Pi={}, Ci={}, Si={}, Si_max={}, state_i={}, mode_i={}"\
              .format(pl_i.get_Pi(), pl_i.get_Ci(), pl_i.get_Si(), 
                      pl_i.get_Si_max(), pl_i.get_state_i(), 
                      pl_i.get_mode_i())) 
    boolean = None
    if state_i == "state1" and mode_i == "CONS+":
        boolean = True if np.abs(Pi+(Si_max-R_i_old)+cons_i - Ci)<thres else False
        formule = "Pi+(Si_max-R_i_old)+cons_i - Ci"
        res = Pi+(Si_max-R_i_old)+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Ci':np.round(Ci,2),
                'Si':np.round(Si,2), 'Si_max':np.round(Si_max,2), 
                'cons_i':np.round(cons_i,2), 'R_i_old': np.round(R_i_old,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "state1" and mode_i == "CONS-":
        boolean = True if np.abs(Pi+cons_i - Ci)<thres else False
        formule = "Pi+cons_i - Ci"
        res = Pi+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Si':np.round(Si,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "state2" and mode_i == "DIS":
        boolean = True if np.abs(Pi+(Si_max-R_i_old-Si) - Ci)<thres else False
        formule = "Pi+(Si_max-R_i_old-Si) - Ci"
        res = Pi+(Si_max-R_i_old-Si) - Ci
        dico = {'Pi':np.round(Pi,2), 'Si':np.round(Si,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "state2" and mode_i == "CONS-":
        boolean = True if np.abs(Pi+cons_i - Ci)<thres else False
        formule = "Pi+cons_i - Ci"
        res = Pi+cons_i - Ci
        dico = {'Pi':np.round(Pi,2), 'Si':np.round(Si,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                'cons_i':np.round(cons_i,2), 'Ci':np.round(Ci,2),
                "state_i": state_i, "mode_i": mode_i, 
                "formule": formule, "res": res}
    elif state_i == "state3" and mode_i == "PROD":
        boolean = True if np.abs(Pi - Ci-prod_i)<thres else False
        formule = "Pi - Ci-Si-prod_i"
        res = Pi - Ci-prod_i
        dico = {'Pi':np.round(Pi,2), 'Si':np.round(Si,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                "prod_i": np.round(prod_i,2), 
                'cons_i': np.round(cons_i,2), 
                'Ci': np.round(Ci,2), "state_i": state_i, 
                "mode_i": mode_i, "formule": formule, 
                "res": res}
    elif state_i == "state3" and mode_i == "DIS":
        boolean = True if np.abs(Pi - Ci-(Si_max-Si)-prod_i)<thres else False
        formule = "Pi - Ci-(Si_max-Si)-prod_i"
        res = Pi - Ci-(Si_max-Si)-prod_i
        dico = {'Pi': np.round(Pi,2), 'Si': np.round(Si,2), 
                'Si_max':np.round(Si_max,2), 'R_i_old': np.round(R_i_old,2),
                "prod_i": np.round(prod_i,2), 
                'cons_i': np.round(cons_i,2), 
                'Ci': np.round(Ci,2), "state_i": state_i, 
                "mode_i": mode_i, "formule": formule, 
                    "res": res, }
    return boolean

# __________    look for whether pli is balanced or not --> fin  ____________


def compute_real_money_SG(arr_pls_M_T, pi_sg_plus_s, pi_sg_minus_s, 
                          INDEX_ATTRS):
    """
    compute real cost (CC)/benefit (BB) and real money (RU) inside the SG

    Parameters
    ----------
    arr_pls_M_T : array of players with a shape M_PLAYERS*NUM_PERIODS*len(INDEX_ATTRS)
        DESCRIPTION.
    pi_sg_plus_s : list of energy price exported to HP. NUM_PERIODS items
        DESCRIPTION.
    pi_sg_minus_s : list of energy price imported from HP. NUM_PERIODS items
        DESCRIPTION.

    Returns
    -------
    BB_i: real benefits' array of M_PLAYERS, 
    CC_i: real costs' array of M_PLAYERS, 
    RU_i: real money's array of M_PLAYERS.

    """
    BB, CC, RU = [], [], []
    for num_pl in range(0, arr_pls_M_T.shape[0]):
        CONS_pl = arr_pls_M_T[num_pl, :, INDEX_ATTRS["cons_i"]]
        PROD_pl = arr_pls_M_T[num_pl, :, INDEX_ATTRS["prod_i"]]
        BB_pl = pi_sg_plus_s[-1] * sum(PROD_pl)
        CC_pl = pi_sg_minus_s[-1] * sum(CONS_pl)
        ru_pl = BB_pl - CC_pl
        
        BB.append(BB_pl); CC.append(CC_pl);RU.append(ru_pl)
        
    return BB, CC, RU
    
def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def find_path_to_variables(name_dir, ext=".npy", threshold= 0.89, n_depth=2):
    """
    create the complet path to variables of extensions .npy

    Parameters
    ----------
    name_dir : TYPE
        DESCRIPTION.
    ext : String, Optional
        DESCRIPTION.
        extension of variables
    threshold: float, Optional
        DESCRIPTION.
        percent of specified files in a directory 
    depth: integer, Optional
        DESCRIPTION.
        number of subdirectories we have to open
        
    Returns
    -------
    path_to_variables: String.

    """
    dirs = []
    dirs.append(name_dir)
    boolean = True
    depth = 0
    while boolean:
        depth += 1
        reps = os.listdir(name_dir)
        rep = reps[np.random.randint(0,len(reps))]
        dirs.append(rep)
        #print("dirs = {}, rep={}".format(dirs, os.path.join(*dirs) ))
        files = os.listdir(os.path.join(*dirs))
        located_files = [fn for fn in files if fn.endswith(ext)]
        if round(len(located_files)/len(files)) >= threshold \
            or depth == n_depth:
            boolean = False
        else:
            name_dir = os.path.join(*dirs)
            
    path_to_variables = os.path.join(*dirs)
    #print('dirs = {}, path_to_variables={}, type={}'.format(dirs, path_to_variables, type( path_to_variables)))
    
    return path_to_variables

def one_hot_string_without_set_classe(array):
    """
    convert an array of string assuming that all items in array are 
    inside the set classe

    Parameters
    ----------
    array : (n_items,)
        DESCRIPTION.

    Returns
    -------
    onehot : array of (array.shape[0], set(array))
        DESCRIPTION.

    """
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot

def one_hot_string_with_set_classe(array, classes):
    """
    convert an array of string assuming that all items in array are not
    inside the set classe

    Parameters
    ----------
    array : (n_items,)
        DESCRIPTION.

    Returns
    -------
    onehot : array of (array.shape[0], set(classes))
        DESCRIPTION.

    """
    # define a mapping of chars to integers
    string_to_int = dict((c, i) for i, c in enumerate(classes))
    int_to_string = dict((i, c) for i, c in enumerate(classes))
    # integer encode input data
    integer_encoded = [string_to_int[string_] for string_ in array]
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        string_ = [0 for _ in range(len(classes))]
        string_[value] = 1
        onehot_encoded.append(string_)
    
    return onehot_encoded

#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------    
def test_fct_positive():
    N = 100
    OK, NOK = 0, 0
    for n in range(N):
        list1 = np.random.randint(1, 10, 10)
        list2 =  np.random.randint(1, 10, 10)
        diff = fct_positive(sum_list1=sum(list1), sum_list2=sum(list2))
        
        if sum(list1)>sum(list2) and diff != 0:
            OK += 1
            # print("OK1, n={} {}>{} => diff={}"
            #       .format(n, sum(list1), sum(list2), diff))
        elif sum(list1)<sum(list2) and diff == 0:
            OK += 1
            # print("OK2, n={} {}<{} => diff={}"\
            #       .format(n, sum(list1), sum(list2), diff))
        elif sum(list1)<sum(list2) and diff != 0:
            NOK += 1
            # print("NOK1, n={} {}<{} => diff={}"\
            #       .format(n, sum(list1), sum(list2), diff))
        elif sum(list1)>sum(list2) and diff == 0:
            NOK += 1
            # print("NOK2, n={} {}>{} => diff={}"\
            #       .format(n, sum(list1), sum(list2), diff))
                
    print("fct_positive: %OK={}, %NOK={}".format(OK/(OK+NOK), NOK/(OK+NOK)))
      
def test_generate_energy_unit_price_SG():
    N = 10
    pi_hp_plus = np.random.random_sample(N) * 20
    pi_hp_minus = np.random.random_sample(N) * 20
    res = np.array(list(map(generate_energy_unit_price_SG, 
                                     *(pi_hp_plus, pi_hp_minus))))
    
    pi_0_plus, pi_0_minus = res[:,0], res[:,1]
    if (pi_0_plus<pi_hp_plus).all() and (pi_0_minus<pi_hp_minus).all():
        print("generate_energy_unit_price_SG: OK")
    else:
        print("generate_energy_unit_price_SG: NOK")
    
def test_compute_real_money_SG():  
    m_players = 5
    num_periods = 5
    INDEX_ATTRS = {"Ci":0, "Pi":1, "Si":2, "Si_max":3, "gamma_i":4, 
                   "prod_i":5, "cons_i":6, "r_i":7, "state_i":8, "mode_i":9}
    arr_pls_M_T = [] #np.ones(shape=(m_players, num_periods), dtype=object)
    for (num_pl, t) in it.product(range(m_players), range(num_periods)):
        arr_pls_M_T.append([t]*10)
    arr_pls_M_T = [arr_pls_M_T[i:i+num_periods] for i in range(0, len(arr_pls_M_T), num_periods)]
    arr_pls_M_T = np.array(arr_pls_M_T, dtype=object)
    #print("arr_pls_M_T = {} \n {}".format(arr_pls_M_T.shape, arr_pls_M_T))
    pi_sg_minus_s = [2]*num_periods
    pi_sg_plus_s = [3]*num_periods
    BB, CC, RU = compute_real_money_SG(arr_pls_M_T, 
                                       pi_sg_plus_s, pi_sg_minus_s, 
                                       INDEX_ATTRS)
    print("BB={},CC={},RU={}".format(BB, CC, RU))
    if sum(BB)/m_players == BB[0] \
        and sum(CC)/m_players == CC[0] \
        and sum(RU)/m_players == RU[0]:
        print("True")
    else:
        print("False")
    
def test_compute_utility_players():
    
    arr_pls_M_T = generate_Pi_Ci_Si_Simax_by_profil_scenario(
                            m_players=M_PLAYERS, num_periods=NUM_PERIODS, 
                            scenario="scenario1", prob_Ci=0.3, 
                            Ci_low=10, Ci_high=60)
    
    OK = 0
    for t in range(0, NUM_PERIODS):
        b0, c0 = np.random.randn(), np.random.randn()
        gamma_is = arr_pls_M_T[:,t, INDEX_ATTRS["gamma_i"]]
            
        bens, csts = compute_utility_players(arr_pls_M_T, gamma_is, t, b0, c0)
        
        if bens.shape == (M_PLAYERS,) \
            and csts.shape == (M_PLAYERS,):
            print("bens={}, csts={}, gamma_is={}".format(
                    bens.shape, csts.shape, gamma_is.shape))
            OK += 1
    print("test_compute_utility_players: rp={}".format(
            round(OK/NUM_PERIODS,2)))
    
# def test_find_path_to_variables():
#     name_dir = "tests"
#     depth = 2
#     ext = "npy"
#     find_path_to_variables(name_dir, ext, depth)
    
# def test_generate_Cis_Pis_Sis_oneplayer_alltime():
#     num_player = 1; 
#     num_periods = 5;
#     low_Ci, high_Ci = 1, 30
#     init_values_i_t = ["Ci","Pi_t","Si_max","Si_t",
#                        0, 0, 0, 0, 0, 0, 
#                        "str_profil_t","str_case_t"]
#     arr_pl_i_T = generate_Cis_Pis_Sis_oneplayer_alltime(
#                     num_player, num_periods, low_Ci, high_Ci)
#     arr_pl_i_T = np.array(arr_pl_i_T, dtype=object)
#     if arr_pl_i_T.shape == (num_periods+1, len(init_values_i_t)):
#         print("test_generate_Cis_Pis_Sis_oneplayer_alltime OK")
#     else:
#         print("test_generate_Cis_Pis_Sis_oneplayer_alltime NOK")
#     print("arr_pl_i_T shape: {}".format( arr_pl_i_T.shape ))
    
# def test_generate_Cis_Pis_Sis_allplayer_alltime():
    
#     m_players = 50; 
#     num_periods = 50;
#     low_Ci, high_Ci = 1, 30
#     init_values_i_t = ["Ci","Pi_t","Si_t","Si_max", 0, 0, 0, 0, 0, 0,
#                        "str_profil_t","str_case_t"]
    
#     arr_pl_M_T = generate_Cis_Pis_Sis_allplayer_alltime(
#                     m_players, num_periods, 
#                     low_Ci, high_Ci)
    
#     if arr_pl_M_T.shape == (m_players, num_periods+1, len(init_values_i_t)):
#         print("test_generate_Cis_Pis_Sis_allplayer_alltime OK")
#     else: 
#         print("test_generate_Cis_Pis_Sis_allplayer_alltime NOK")
#     print("arr_pl_M_T shape={}, size={} Mo".format(arr_pl_M_T.shape, 
#             round(sys.getsizeof(arr_pl_M_T)/(1024*1024),3)))

def test_generate_Pi_Ci_Si_Simax_by_profil_scenario():
    
    arr_pl_M_T = generate_Pi_Ci_Si_Simax_by_profil_scenario(
                            m_players=30, num_periods=5, 
                            scenario="scenario1", prob_Ci=0.3, 
                            Ci_low=10, Ci_high=60)
    # compter le nombre players ayant Ci = 10 et Ci = 60
    cis_weak = arr_pl_M_T[arr_pl_M_T[:, 1, INDEX_ATTRS["Ci"]] == 10].shape[0]
    cis_strong = arr_pl_M_T[arr_pl_M_T[:, 1, INDEX_ATTRS["Ci"]] == 60].shape[0]
    
    print("___ arr_pl_M_T : {}, Ci_weak={}, Ci_strong={}".format(
            arr_pl_M_T.shape, round(cis_weak/arr_pl_M_T.shape[0],2), 
            round(cis_strong/arr_pl_M_T.shape[0],2)))
    return arr_pl_M_T

def test_generer_Pi_Ci_Si_Simax_for_all_scenarios():
    m_players=100; num_periods=250
    l_arr_pl_M_T = []
    l_arr_pl_M_T = generer_Pi_Ci_Si_Simax_for_all_scenarios(
                        scenarios=["scenario1", "scenario2", "scenario3"], 
                        m_players=m_players, num_periods=num_periods, 
                        prob_Ci=0.3, Ci_low=10, Ci_high=60)
    cpt_true = 0
    for arr_pl_M_T in l_arr_pl_M_T:
        if arr_pl_M_T.shape == (m_players, num_periods, len(INDEX_ATTRS)):
            cpt_true += 1
            
    if cpt_true == len(l_arr_pl_M_T):
        print("___ generer_Pi_Ci_Si_Simax_for_all_scenarios ___ OK")
        print(" m_players={}, num_periods={}".format(m_players, num_periods))
        print(" memsize \n scenario1:{} Mo, scenario2:{} Mo, scenario3:{} Mo".format(
                sys.getsizeof(l_arr_pl_M_T[0])/(1024*1024), 
                sys.getsizeof(l_arr_pl_M_T[1])/(1024*1024),
                sys.getsizeof(l_arr_pl_M_T[2])/(1024*1024)))
    else:
        print("___ generer_Pi_Ci_Si_Simax_for_all_scenarios ___ NOK")

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------    
if __name__ == "__main__":
    ti = time.time()
    test_fct_positive()
    test_generate_energy_unit_price_SG()
    
    # path_file = test_find_path_to_variables()
    test_compute_utility_players()
    test_compute_real_money_SG()
    
    # test_generate_Cis_Pis_Sis_oneplayer_alltime()
    # test_generate_Cis_Pis_Sis_allplayer_alltime()
    
    arrs = test_generate_Pi_Ci_Si_Simax_by_profil_scenario()
    
    #test_generer_Pi_Ci_Si_Simax_for_all_scenarios()
    
    print("runtime = {}".format(time.time() - ti))