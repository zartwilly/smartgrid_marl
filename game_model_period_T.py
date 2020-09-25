# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:33:06 2020

@author: jwehounou
"""
import time
import math
import numpy as np
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

#------------------------------------------------------------------------------
#                       definition of constantes
#------------------------------------------------------------------------------
N_INSTANCE = 10
M_PLAYERS = 10
CASE1 = (0.75, 1.5)
CASE2 = (0.4, 0.75)
CASE3 = (0, 0.3)
LOW_VAL_Ci = 1 
HIGH_VAL_Ci = 30
NUM_PERIODS = 5

#------------------------------------------------------------------------------
# ebouche de solution generale
#                   definitions of functions
#------------------------------------------------------------------------------
def generate_random_values(zero=0):
    """
    at time t=0, the values of C_T_plus_1, P_T_plus_1 are null. 
    see decription of smartgrid system model document.

    if zero=0 then this is zero array else not null array
    Returns
    -------
    array of shape (2,).

    """
    C_T_plus_1, P_T_plus_1 = np.array((0,0))
    if zero != 0:
        C_T_plus_1 = np.random.random_sample()
        P_T_plus_1 = np.random.random_sample()
    return  C_T_plus_1, P_T_plus_1

def initialize_game_create_agents_t0(sys_inputs):
    """
    

    Parameters
    ----------
    sys_inputs : dict
        contain variables like hp prices, sg prices, storage capacity 
        at t=0 and t=1, the future production and consumption, 
        the case of the game (more production, more consumption,).

    Returns
    -------
     arr_pls:  array of players of shape 1*M M*T
     arr_pls_M_T: array of players of shape M*T*9
     pl_m^t contains a list of 
             Pi, Ci, Si, Si_max, gamma_i, prod_i, cons_i, r_i, state_i
     m \in [0,M-1] and t \in [0,T-1] 
    """
    # declaration variables
    arr_pls = np.array([]) 
    arr_pls_M_T = [] # np.array([])
    Ci_t_plus_1, Pi_t_plus_1 = sys_inputs["Ci_t_plus_1"], sys_inputs["Pi_t_plus_1"] 
    pi_0_plus, pi_0_minus = sys_inputs["pi_0_plus"], sys_inputs["pi_0_minus"] 
    pi_hp_plus, pi_hp_minus = sys_inputs["pi_hp_plus"], sys_inputs["pi_hp_minus"]
    
    # create the M players
    Cis, Pis, Si_maxs, Sis = fct_aux.generate_Cis_Pis_Sis(
                                n_items = M_PLAYERS, 
                                low_1 = LOW_VAL_Ci, 
                                high_1 = HIGH_VAL_Ci,
                                low_2 = sys_inputs['case'][0],
                                high_2 = sys_inputs['case'][1]
                                )
    
    gamma_is = np.zeros(shape=(1, M_PLAYERS))
    prod_is = np.zeros(shape=(1, M_PLAYERS))
    cons_is = np.zeros(shape=(1, M_PLAYERS))
    r_is = np.zeros(shape=(1, M_PLAYERS))
    state_is = np.array([None]*M_PLAYERS).reshape((1,-1))
    for ag in np.concatenate((Pis, Cis, Sis, Si_maxs, gamma_is, prod_is, 
                              cons_is, r_is, state_is)).T:
        pl = players.Player(*ag)
        pl.get_state_i()
        pl.select_mode_i()
        pl.update_prod_cons_r_i()
        pl.select_storage_politic(Ci_t_plus_1, Pi_t_plus_1, 
                                  pi_0_plus, pi_0_minus, 
                                  pi_hp_plus, pi_hp_minus)
        arr_pls = np.append(arr_pls, pl) #arr_pls.append(pl)
        
        a_i_t_s = []
        Ci = pl.get_Ci(); Pi = pl.get_Pi(); Si = pl.get_Si(); 
        Si_max = pl.get_Si_max(); gamma_i = pl.get_gamma_i(); 
        prod_i = pl.get_prod_i(); cons_i = pl.get_cons_i(); 
        r_i = pl.get_r_i(); state_i = pl.get_state_i()
        a_i_t_s.append([Ci, Pi, Si, Si_max, gamma_i, prod_i, 
                        cons_i, r_i, state_i])
        
        # for each player, generate the attribut values of players a each time.
        Cis, Pis, Si_maxs, Sis = fct_aux.generate_Cis_Pis_Sis(
                                    n_items = NUM_PERIODS, 
                                    low_1 = LOW_VAL_Ci, 
                                    high_1 = HIGH_VAL_Ci,
                                    low_2 = sys_inputs['case'][0],
                                    high_2 = sys_inputs['case'][1]
                                    )
        for (Ci, Pi, Si, Si_max) in np.concatenate((Cis, Pis, Sis, Si_maxs)).T:
            pl.set_Ci(Ci, update=False)
            pl.set_Pi(Pi, update=False)
            pl.set_Si(Si, update=False)
            pl.set_Si_max(Si_max, update=False)
            pl.get_state_i()
            pl.select_mode_i()
            pl.update_prod_cons_r_i()
            # TODO IMPORTANT
            # est il necessaire to update S_i^t pdt la phase d'initialisation des Pi^t, Ci^t, Si^t, Si^t_max,...
            pl.select_storage_politic(Ci_t_plus_1, Pi_t_plus_1, 
                                  pi_0_plus, pi_0_minus, 
                                  pi_hp_plus, pi_hp_minus)
            
            gamma_i = pl.get_gamma_i(); prod_i = pl.get_prod_i() 
            cons_i = pl.get_cons_i(); r_i = pl.get_r_i(); 
            state_i = pl.get_state_i()
            a_i_t_s.append([Ci, Pi, Si, Si_max, gamma_i, prod_i, 
                            cons_i, r_i, state_i])
            
        # TODO a resoudre cela
        # arr_pls_M_T = np.array(a_i_t_s) \
        #     if len(arr_pls_M_T)==0 \
        #     else np.concatenate((arr_pls_M_T, np.array(a_i_t_s)),axis=0) #arr_pls_M_T.append(a_i_t_s)
        # # arr_pls_M_T.reshape(M_PLAYERS,NUM_PERIODS+1,-1) #--> FALSE
        arr_pls_M_T.append(a_i_t_s)
        
    arr_pls_M_T = np.array(arr_pls_M_T)
    return arr_pls, arr_pls_M_T
    """
    initialize a game by
        * create N+1 agents
        * attribute characteristic values of agents such as agents become players
            characteristic values are :
                - name_ai, state_ai, mode_i, prod_i, cons_i, r_i

    Parameters
    ----------
    sys_inputs : TYPE
        DESCRIPTION.

    Returns
    -------
    arr_pls, arr_val_pls.

    """
    return None

def compute_prod_cons_SG(arr_val_pls):
    """
    compute the production In_sg and the consumption Out_sg in the SG.

    Parameters
    ----------
    arr_val_pls : numpy array (N+1, 6)
        DESCRIPTION.

    Returns
    -------
    In_sg, Out_sg: Integer.

    """
    return None

def determine_new_pricing_sg(prod_is, cons_is, pi_hp_plus, pi_hp_minus):
    """
    determine the new price of energy in the SG from the amounts produced 
    (prod_is), consumpted (cons_is) and the price of one unit of energy 
    exchanged with the HP

    Parameters
    ----------
    prod_is : array of 1 x N
        DESCRIPTION.
    cons_is : array of 1 x N
        DESCRIPTION.
    pi_hp_plus : a float 
        DESCRIPTION.
    pi_hp_minus : float
        DESCRIPTION.

    Returns
    -------
    new_pi_0_plus, new_pi_0_minus.

    """
    return None

def update_player(arr_pls, list_valeurs_by_variable):
    """
    

    Parameters
    ----------
    arr_pls : TYPE
        DESCRIPTION.
    list_valeurs_by_variable : list of tuples
        DESCRIPTION.
        EXEMPLE
            [(1,new_gamma_i_t_plus_1_s)])
            1 denotes update gamma_i and generate/update prod_i or cons_i from gamma_i
            [(2, state_ais)]
            2 denotes update variable state_ai without generate/update prod_i or cons_i
    Returns
    -------
    arr_pls.

    """

def game_model_SG_T(T, pi_hp_plus, pi_hp_minus, pi_0_plus, pi_0_minus, case):
    """
    create the game model divised in T periods such as players maximize their 
    profits on each period t

    diff between agent and player:
        an agent becomes a player iff he has state_ai, mode_i, prod_i, cons_i, 
        r_is
        
    form of arr_pls : N x 6
        one row: [AG, state_ai, mode_i, prod_i, cons_i, r_i]
    Returns
    -------
    None.

    """
    S_0, S_1, = 0, 0;
    # TODO:how to determine Ci_t_plus_1_s, Pi_t_plus_1_s?
    ## generate random values of Ci_t_plus_1_s, Pi_t_plus_1_s
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_T_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls = initialize_game_create_agents_t0(sys_inputs)
    B0s, C0s, BENs, CSTs = [], [], [], []
    pi_sg_plus, pi_sg_minus = 0, 0
    for t in range(0, T):
        # compute In_sg, Out_sg
        In_sg, Out_sg = compute_prod_cons_SG(arr_pls)
        # compute price of an energy unit price for cost and benefit players
        b0, c0 = compute_energy_unit_price(
                        pi_0_plus, pi_0_minus, 
                        pi_hp_plus, pi_hp_minus,
                        In_sg, Out_sg)
        B0s.append(b0);
        C0s.append(C0);
        # compute cost and benefit players by energy exchanged.
        gamma_is = extract_values_to_array(arr_pls, attribut_position=0)        
        bens, csts = compute_utility_players(arr_pls, gamma_is)
        BENs.append(bens), CSTs.append(csts)
        
        #compute the new prices pi_0_plus and pi_0_minus from a pricing model in the document
        cons_is = extract_values_to_array(arr_pls, attribut_position=4)
        prod_is = extract_values_to_array(arr_pls, attribut_position=3)
        pi_0_plus, pi_0_minus = determine_new_pricing_sg(
                                    prod_is, cons_is,
                                    pi_hp_plus, pi_hp_minus)
        # update CONS_i for a each player
        arr_pls = update_player(arr_pls, [(2,cons_is), (2,prod_is)])
        
        
        # definition of the new storage politic
        new_gamma_i_t_plus_1_s = select_storage_politic(
                                    arr_pls, state_ais, mode_is, 
                                    Ci_t_plus_1_s[t], Pi_t_plus_1_s[t], 
                                    pi_0_plus, pi_0_minus)
        arr_pls = update_player(arr_pls, [(1,new_gamma_i_t_plus_1_s)])
        # choose a new state, new mode at t+1
        state_ais, mode_is = select_mode_compute_r_i(arr_pls)
        arr_pls = update_player(arr_pls, [(2,state_ais), (2,mode_is)])
        
        # update prices of the SG
        pi_sg_plus, pi_sg_minus = pi_0_plus, pi_0_minus
        
    # compute utility of 
    RUs = compute_real_utility(arr_pls, pi_sg_plus, pi_sg_minus)
    
    return None

def game_model_SG():
    """
    create a game for one period T = [1..T]

    Returns
    -------
    None.

    """
    return None
    
#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------    
def test_initialize_game_create_agents_t0():
    """
    Si OK alors arr_pls.shape = (M_PLAYERS,)
                et arr_pls_M_T.shape = (M_PLAYERS, NUM_PERIODS, 10)

    Returns
    -------
    None.

    """
    S_0, S_1, = 0, 0; case = CASE3
    # generate pi_hp_plus, pi_hp_minus
    pi_hp_plus, pi_hp_minus = generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = generate_random_values(zero=1)
    Ci_t_plus_1, Pi_t_plus_1 = generate_random_values(zero=0)
    
    sys_inputs = {"S_0":S_0, "S_1":S_1, "case":case,
                    "Ci_t_plus_1":Ci_t_plus_1, "Pi_t_plus_1":Pi_t_plus_1,
                    "pi_hp_plus":pi_hp_plus, "pi_hp_minus":pi_hp_minus, 
                    "pi_0_plus":pi_0_plus, "pi_0_minus":pi_0_minus}
    arr_pls, arr_pls_M_T = initialize_game_create_agents_t0(sys_inputs)
    
    print("shape: arr_pls={}, arr_pls_M_T={}".format(
            arr_pls.shape, arr_pls_M_T.shape))
    if arr_pls.shape == (M_PLAYERS,) \
        and arr_pls_M_T.shape == (M_PLAYERS, NUM_PERIODS+1, 9):
            print("test_initialize_game_create_agents_t0: OK")
            print("shape: arr_pls={}, arr_pls_M_T={}".format(
            arr_pls.shape, arr_pls_M_T.shape))
    else:
        print("test_initialize_game_create_agents_t0: NOK")
        print("shape: arr_pls={}, arr_pls_M_T={}".format(
            arr_pls.shape, arr_pls_M_T.shape))
    
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_initialize_game_create_agents_t0()
    print("runtime = {}".format(time.time() - ti))    
    