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
LOW_VAL_Ci = 1 
HIGH_VAL_Ci = 30

NUM_PERIODS = 5

#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------
def execution_game(case=CASE3):
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
    
    # create test directory tests and put file in the directory simu_date_hhmm
    name_dir = "tests"; date_hhmm = datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join(name_dir, "simu_"+date_hhmm)
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    
    case = CASE3
    pi_hp_plus, pi_hp_minus = gmpT.generate_random_values(zero=1)
    pi_0_plus, pi_0_minus = gmpT.generate_random_values(zero=1)
    
    gmpT.run_game_model_SG(pi_hp_plus, pi_hp_minus, 
                           pi_0_plus, pi_0_minus, 
                           case, path_to_save)
    

#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------ 
def test_execution_game(case):
    execution_game(case=case)
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_execution_game(CASE2)
    print("runtime = {}".format(time.time() - ti))  
    