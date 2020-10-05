# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 08:32:09 2020

@author: jwehounou

Execution game
"""
import os
import sys
import time
import math
import json
import numpy as np
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux
import game_model_period_T as gmpT

from datetime import datetime
from pathlib import Path


#------------------------------------------------------------------------------
#                       definition of constantes
#------------------------------------------------------------------------------
N_INSTANCE = 10
M_PLAYERS = 10
CHOICE_RU = 1
CASE1 = (0.75, 1.5)
CASE2 = (0.4, 0.75)
CASE3 = (0, 0.3)
LOW_VAL_Ci = 100 
HIGH_VAL_Ci = 300

NUM_PERIODS = 50

#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------
def execute_game_onecase(case=CASE3):
    """
    execution of the game with initialisation of constances and variables.

    Parameters
    ----------
    pi_hp_plus : integer
        DESCRIPTION.
    pi_hp_minus : integer
        DESCRIPTION.
    pi_0_plus : integer
        DESCRIPTION.
    pi_0_minus : integer
        DESCRIPTION.
    case : tuple, optionnal
        DESCRIPTION.
        min and max values of random variable Pi. 
    Returns
    -------
    None.

    """
    
    str_case = str(case[0]) +"_"+ str(case[1])
    # create test directory tests and put file in the directory simu_date_hhmm
    name_dir = "tests"; date_hhmm = datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, str_case)
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    
    pi_hp_plus, pi_hp_minus = gmpT.generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = gmpT.generate_random_values(zero=1)
    
    gmpT.run_game_model_SG(pi_hp_plus, pi_hp_minus, 
                           pi_0_plus, pi_0_minus, 
                           case, path_to_save)
    

def execute_game_allcases(cases):
    """
    run game for all listed cases 

    Parameters
    ----------
    cases : list of tuples
        DESCRIPTION.

    Returns
    -------
    None.

    """
    name_dir = "tests"; date_hhmm = datetime.now().strftime("%d%m_%H%M")
    for case in cases:
        str_case = str(case[0]) +"_"+ str(case[1])
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm, str_case)
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        
        pi_hp_plus, pi_hp_minus = gmpT.generate_random_values(zero=1)
        pi_0_plus, pi_0_minus = gmpT.generate_random_values(zero=1)
        
        gmpT.run_game_model_SG(pi_hp_plus, pi_hp_minus, 
                               pi_0_plus, pi_0_minus, 
                               case, path_to_save)
        
#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------ 
def test_execute_game_onecase(case):
    execute_game_onecase(case=case)
    
def test_execute_game_allcase():
    cases = [CASE1, CASE2, CASE3]
    execute_game_allcases(cases)
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_execute_game_onecase(CASE2)
    test_execute_game_allcase()
    print("runtime = {}".format(time.time() - ti))  
    