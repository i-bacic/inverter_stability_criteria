#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tests import systems_to_test
from inverter_stability_criteria import stationary_state

from inverter_stability_criteria.powerflow_solution import active_power_flow, reactive_power_flow


def test_linear_power_flow_calculation():
    """Test if the linear power flow calculation works as expected."""

    # Two nodes
    
    # Tripplet


def test_two_nodes(beta_gain_factor=.1):
    """Test the stationary state evaluation for two nodes"""
    
    Pi_two = 1.
    BB_line_two = 2.
    
    volt_val_two = 1.
    Qi = .1
    
    
    two_node_sys = systems_to_test.setup_system_2_nodes(Pi=Pi_two, BB_line=BB_line_two, 
                                                        volt_val=volt_val_two, Qi=Qi,
                                                        beta_gain_factor=beta_gain_factor)


    sol_state = stationary_state.solve_stationary_state(two_node_sys)
    
    print(sol_state)
    
    act_pf_two = active_power_flow(two_node_sys.BB_matrix, 
                                   sol_state[0], sol_state[1])
    react_pf_two = reactive_power_flow(two_node_sys.BB_matrix, 
                                       sol_state[0], sol_state[1])
    
    diff_power_active = act_pf_two - two_node_sys.power_setpoints_active
    
    assert (abs(diff_power_active)/two_node_sys.power_setpoints_active < 1.2).all()
    
    return


def test_open_triangle():
    
    
    return
    