# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:51:47 2020

@author: jwehounou
"""
cpt_agents = 0
class Agent:
    
    def __init__(self, Pi, Ci, Si, Si_max, gamma_i):
        self.name = ().join(["a",str(cpt_agents)])
        self.Pi = Pi
        self.Ci = Ci
        self.Si = Si
        self.Si_max = Si_max
        self.gamma_i = gamma_i
        cpt_agents += 1
        
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
                        
    