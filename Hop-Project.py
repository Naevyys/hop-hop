# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:41:45 2022

@author: tahat
"""
import matplotlib.pyplot as plt
import numpy as np
from neurodynex.hopfield_network import pattern_tools, plot_tools, network

pattern_size = 5
nr_neurons=300
# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(nr_neurons= nr_neurons)

