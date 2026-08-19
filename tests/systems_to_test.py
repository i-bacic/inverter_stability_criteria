#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from inverter_stability_criteria.system_classes import System

from typing import Literal
from numpy.typing import NDArray

__all__ = ["setup_system_2_nodes", 
           "setup_system_open_triangle_middle_producing"]


def test_node_edge_incidence_matrix():
    
    return

def setup_box_with_diagonal():
    
    return


def setup_system_2_nodes(Pi: float,
                   BB_line: float,
                   volt_val: float,
                   Qi: float = .1,
                   time_constant_factor: float = .05,
                   beta_gain_factor: float = .1):
    """Setup a system with two nodes with the same voltage at 
    both buses.
    """
    
    nr_nodes = 2
    admittance_matrix = np.array([[BB_line, - BB_line],
                                  [- BB_line, BB_line]],
                                 dtype=float) * 1j
    
    power_setpoints_active = np.array([Pi, -Pi])
    power_setpoints_reactive = np.ones(nr_nodes) * Qi
    
    volt_arr = np.ones(nr_nodes) * volt_val
    
    # Rest are dummy variables
    (time_constants_active, 
     time_constants_reactive) = [np.ones(nr_nodes) * time_constant_factor] * 2
    (beta_gain_active, 
     beta_gain_reactive) = [np.ones(nr_nodes) * beta_gain_factor] * 2
    
    (disturbance_active, disturbance_reactive) = [np.zeros(nr_nodes)] * 2
    
    sys_cls = System(admittance_matrix=admittance_matrix,
                     volt_operating_point=volt_arr,
                     time_constants_active=time_constants_active,
                     time_constants_reactive=time_constants_reactive,
                     beta_gain_active=beta_gain_active,
                     beta_gain_reactive=beta_gain_reactive,
                     power_setpoints_active=power_setpoints_active,
                     power_setpoints_reactive=power_setpoints_reactive,
                     disturbance_active=disturbance_active,
                     disturbance_reactive=disturbance_reactive)
    
    return sys_cls
    
    
def setup_system_open_triangle_middle_producing(Pi: float,
                                                Qi: float,
                     BB_dd_vec: NDArray[np.float64, Literal[2]],
                     volt_vec: NDArray[np.float64, Literal[3]],
                     time_constant_factor: float = .1,
                     beta_gain_factor: float = .1):
    """Setup a system consisting of a central nodes with two connected 
    nodes. The middle nodes produces Pi, while the two others consume 1/2
    """
    
    nr_nodes = 3
    
    BB_12 = BB_dd_vec[0]
    BB_13 = BB_dd_vec[1]
    
    admittance_matrix = np.array([[BB_12 + BB_13, - BB_12, - BB_13],
                                  [- BB_12, BB_12, 0],
                                  [- BB_13, 0, BB_13]]) *1j
    
    # Rest are dummy variables
    (time_constants_active, 
     time_constants_reactive) =  [np.ones(nr_nodes)] * 2
    
    (beta_gain_active, 
      beta_gain_reactive) = [np.ones(nr_nodes)] * 2
    
    (disturbance_active,
     disturbance_reactive) = [np.ones(nr_nodes)] * 2
    
    power_vec = np.array([Pi, -.5*Pi, -.5*Pi])
    power_set_react = np.ones([nr_nodes]) * Qi
    
    sys = System(admittance_matrix=admittance_matrix,
                 volt_operating_point=volt_vec,
                 time_constants_active=time_constants_active,
                 time_constants_reactive=time_constants_reactive,
                 beta_gain_active=beta_gain_active,
                 beta_gain_reactive=beta_gain_reactive,
                 power_setpoints_active=power_vec,
                 power_setpoints_reactive=power_set_react,
                 disturbance_active=disturbance_active,
                 disturbance_reactive=disturbance_reactive)

    return sys
