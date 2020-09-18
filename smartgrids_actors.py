# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:51:47 2020

@author: jwehounou
"""
import time
import numpy as np

import fonctions_auxiliaires as fct_aux

N_INSTANCE = 10

MODES = ("CONS","PROD","DIS")

STATE1_STRATS = ("CONS+", "CONS-")                                             # strategies possibles pour l'etat 1 de a_i
STATE2_STRATS = ("CONS-", "DIS")                                               # strategies possibles pour l'etat 2 de a_i
STATE3_STRATS = ("PROD", "DIS")                                                # strategies possibles pour l'etat 3 de a_i

class Agent:
    
    cpt_agent =  0
    
    def __init__(self, Pi, Ci, Si, Si_max, gamma_i):
        self.name = ("").join(["a",str(self.cpt_agent)])
        self.Pi = Pi
        self.Ci = Ci
        self.Si = Si
        self.Si_max = Si_max
        self.gamma_i = gamma_i
        Agent.cpt_agent += 1
       
        
    #--------------------------------------------------------------------------
    #           definition of caracteristics of an agent
    #--------------------------------------------------------------------------
    def get_Pi(self):
        """
        return the value of quantity of production
        """
        return self.Pi
    
    def set_Pi(self, new_Pi, update=False):
        """
        return the new quantity of production or the energy quantity 
        to add from the last quantity of production.
        
        self.Pi = new_Pi if update==True else self.Pi + new_Pi
        """
        self.Pi = (update==True)*new_Pi + (update==False)*(self.Pi + new_Pi)
            
    def get_Ci(self):
        """
        return the quantity of consumption 
        """
        return self.Ci
    
    def set_Ci(self, new_Ci, update=False):
        """
        return the new quantity of consumption or the energy quantity 
        to add from the last quantity of production.
        
        self.Ci = new_Ci if update==True else self.Ci + new_Ci
        """
        self.Ci = (update==False)*new_Ci + (update==True)*(self.Ci + new_Ci)
        
    def get_Si(self):
        """
        return the value of quantity of battery storage
        """
        return self.Si
    
    def set_Si(self, new_Si, update=False):
        """
        return the new quantity of battery storage or the energy quantity 
        to add from the last quantity of storage.
        
        self.Si = new_Si if update==True else self.Si + new_Si
        """
        self.Si = (update==True)*new_Si + (update==False)*(self.Si + new_Si)
        
    def get_Si_max(self):
        """
        return the value of quantity of production
        """
        return self.Si_max
    
    def set_Si_max(self, new_Si_max, update=False):
        """
        return the new quantity of the maximum battery storage or the energy 
        quantity to add from the last quantity of teh maximum storage.
        
        self.Si = new_Pi if update==True else self.Pi + new_Pi
        """
        self.Si_max = (update==True)*new_Si_max \
                        + (update==False)*(self.Si_max + new_Si_max)
                        
    def get_gamma_i(self):
        """
        gamma denotes the behaviour of the agent to store energy or not. 
        the value implies the price of purchase/sell energy.
        return the value of the behaviour  
        """
        return self.gamma_i
    
    def set_gamma_i(self, new_gamma_i):
        """
        return the new value of the behaviour or the energy 
        quantity to add from the last quantity of teh maximum storage.
        
        self.Si = new_Pi if update==True else self.Pi + new_Pi
        """
        self.gamma_i = new_gamma_i 
                        
    #--------------------------------------------------------------------------
    #           definition of functions of an agent
    #--------------------------------------------------------------------------
    def identify_state(self):
        """
        determine the state of an agent following its characteristics

        Returns
        -------
        state_ai = {"state1", "state2", "state3", None}

        """
        if self.Pi + self.Si <= self.Ci:
            return "state1"
        elif self.Pi + self.Si > self.Ci and self.Pi < self.Ci:
            return "state2"
        elif self.Pi > self.Ci:
            return "state3"

        return None
    
    def select_mode(self, state_ai):
        """
        choose a mode for an agent under its state and its characteristics

        Returns
        -------
        mode_i, prod_i, cons_i

        """
        res = (None, np.inf, np.inf)
        rd_num =  np.random.choice([0,1])
        if state_ai == None:
            print("Conditions non respectees pour state1,2,3")
        elif state_ai == "state1":
            mode_i = STATE1_STRATS[rd_num]
            prod_i = 0
            cons_i = (1-rd_num)*( self.Ci - (self.Pi - self.Si) ) \
                        + rd_num*(self.Ci - self.Pi) 
            self.Si = (1-rd_num)*0 + rd_num * self.Si
            res = (mode_i, prod_i, cons_i)
        elif state_ai == "state2":
            mode_i = STATE2_STRATS[rd_num]
            prod_i = 0
            cons_i = (1-rd_num)*(self.Ci - self.Pi) + rd_num*0
            self.Si = (1-rd_num)*0 + rd_num*(self.Si - (self.Ci - self.Pi))
            res = (mode_i, prod_i, cons_i)
        elif state_ai == "state3":
            mode_i = STATE3_STRATS[rd_num]
            cons_i = 0
            self.Si = (1-rd_num)*self.Si \
                        + rd_num*(max(self.Si_max, 
                                       self.Si + (self.Pi - self.Ci))) 
            Ri = self.Si_max - self.Si
            prod_i = (1-rd_num)*(self.Pi - self.Ci) \
                        + rd_num*fct_aux.fct_positive(sum([self.Pi]), 
                                                      sum([self.Ci, Ri]))
            res = (mode_i, prod_i, cons_i)
        return res
    
#------------------------------------------------------------------------------
#           unit test of functions
#------------------------------------------------------------------------------
def test_classe_agent():
    Pis = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    Cis = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    Sis = np.random.randint(1, 15, N_INSTANCE) + np.random.randn(1, N_INSTANCE) 
    Si_maxs = np.random.randint(1, 30, N_INSTANCE) + np.random.randn(1, N_INSTANCE) 
    gamma_is = np.random.randint(1, 3, N_INSTANCE) + np.random.randn(1, N_INSTANCE)
    
    # utiliser itertools pour des tuples de pi, ci, si, ...
    # creer des instances d'agents
    # faire les fonctions. 
        # pour chaque fonction, calculer le rapport de ri = OK/(NOK+OK)
    # les afflicher ces indicateurs ri
    
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    test_classe_agent()
    print("classe agent runtime = {}".format(time.time() - ti))
