# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:28:22 2022

@author: tahat
"""

import matplotlib.pyplot as plt
import numpy as np
from neurodynex.hopfield_network import pattern_tools, plot_tools, network

N = 300
n_patterns = 5
a = 0.5
b = 0.5
theta=0


def generate_and_store_random_patterns(net, n_patterns):
    """
    Resets weights of the network passed and learn new weights for n_patterns random patterns.
    :param net: hopfield network
    :param n_patterns: number of patterns
    :return: updated hopfield network, generated random patterns
    """
    random_patterns=random_pattern(N, a, n_patterns)
    random_patterns=[(element + 1)/2 for element in random_patterns]
    new_weights = weight_matrix(n_patterns, N, a, b, random_patterns)
    net.weights=new_weights
    return net, random_patterns

def random_pattern(N, a,n_patterns):
    """ generate random patterns """
    pattern_generator = pattern_tools.PatternFactory(N, pattern_width=1)
    return pattern_generator.create_random_pattern_list(n_patterns,a)


def weight_matrix(n_patterns, N, a, b, patterns):
    """
    compute the weight matrix
    w: weights
    """
    w=np.zeros([N, N])
    cprime=1 / a / (1-a) / N
    for i in range(n_patterns):
        w=w+np.dot(patterns[i]-b, np.transpose(patterns[i]-a))*cprime
    return w


def hamming_distance(x,y):
    """
    Hamming distance
    """
    return (sum(x)+sum(y)-2*np.dot(x, y)) / np.size(x)


def map_function(theta):
    def map_function_theta(sigma_t0, w):
        """
        changing the dynamic of the network
        """
        sigma_t1=np.zeros(np.size(sigma_t0))
        for i in range(np.size(sigma_t0)):
            sigma_t1[i]=0.5 * (1 + np.sign(np.dot(w[i], sigma_t0) - theta))      
        return sigma_t1
    return map_function_theta


def hop_network(N,n_patterns,a,b,theta):
    """
    Execute exercise 1.
    :return: hopfield network, generated random patterns
    """
    #Creating the random patterns of 1 & 0    
    random_patterns=random_pattern(N, a, n_patterns)
    random_patterns=[(element + 1)/2 for element in random_patterns]
    
    #Creating the new weights from the random patterns
    new_weights = weight_matrix(n_patterns, N, a, b, random_patterns)
    net = network.HopfieldNetwork(N)
    net.weights=new_weights
    net.set_dynamics_to_user_function(map_function(theta))
    return net, random_patterns

net , random_patterns = hop_network(N,n_patterns,a,b,theta)


def hop_run(net, stored_patterns, hamming_distance, selected_pattern_id=None):
    """
    Execute exercise 3.
    :param net: Hopfield network
    :param stored_patterns: Pattern that the network has stored in memory.
    :param distance_function: Function used to compute the distance between two patterns.
    :param selected_pattern_id: Pattern from which to create the initial state
    :return: Distances between last state and all patterns, boolean flag indicating whether the pattern was correctly
             retrieved, selected_pattern_id
    """

    # Create and set initial state
    if selected_pattern_id is None:
        selected_pattern_id = np.random.randint(0, len(stored_patterns))
    else:
        assert 0 <= selected_pattern_id < len(stored_patterns), "Selected pattern is not in the range of patterns list"
    selected_pattern = stored_patterns[selected_pattern_id]
    initial_state = selected_pattern  # Flip 15 randomly chosen neurons TODO\
    initial_state = initial_state * 2 -1
    net.set_state_from_pattern(initial_state)
    theta = 0
    # Run the network
    n_steps = 6
    net.run(nr_steps=n_steps)
    final_state = net.state

    # Calculate distance between the final state and all the patterns
    distances = [hamming_distance(final_state, np.squeeze(pattern)) for pattern in stored_patterns]
    correctly_retrieved = True if distances[selected_pattern_id] <= 0.05 else False

    return distances, correctly_retrieved, selected_pattern_id

x1,x2,x3 = hop_run(net, random_patterns, hamming_distance, selected_pattern_id=None)
print(x1)

net , random_patterns = hop_network(N,n_patterns,a,b,theta)
def ex4(hamming_distance, m_vals, n_runs, N, a, b, theta):
    """
    Computes for each value of m the mean distance between the final state of the network and the target pattern, and
    plot the results.

    Algorithm:
    - For 5 to 8 values M from 5 to 100:
       - Generate list of size M with random patterns
       - Reset weights of network
       - Store patterns of the list
       - For each pattern:
           - Randomly flip 5% of that pattern, set this as initial state, run for 3 to 10 steps
           - Compute distance between final state and target pattern, store that value
       - Compute the mean distance between final state and target pattern, store it
    - Plot the means
    :param net: hopfield network
    :param distance_function: distance function to use
    :param m_vals: List of different values for m
    :param n_runs: Number of runs to reach the state to compare with the target pattern
    :return: mean distances
    """
    
    def compute_mean_distance(n_patterns):
        network, random_patterns = generate_and_store_random_patterns(net, n_patterns)
        distances = np.zeros(n_patterns)
        n_flips = 15  # 5% of 300
        for i, pattern in enumerate(random_patterns):
            new_pattern=pattern*2-1
            initial_state = pattern_tools.flip_n(new_pattern, n_flips)
            initial_state=(initial_state + 1)/2
            network.set_state_from_pattern(initial_state)
            network.run(n_runs)
            final_state = network.state
            distances[i] = hamming_distance(final_state, np.squeeze(pattern))
        return np.mean(distances)

    means = np.array([compute_mean_distance(m) for m in m_vals])
    plt.plot(m_vals, means)
    plt.title("Ex2-5: Means distances between final state and target pattern.")
    plt.xlabel("Number of patterns stored in the network")
    plt.ylabel("Error (measured using Hamming distance)")
    plt.savefig("plots/ex2-5.png")
    plt.show()

    return means

x4=ex4(hamming_distance, m_vals=(5, 20, 40, 60, 80, 100), n_runs=6, N=300, a=0.5, b=0.5, theta=0)
print(x4)