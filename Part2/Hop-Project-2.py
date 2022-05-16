# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:28:22 2022

@author: tahat
"""

import matplotlib.pyplot as plt
import numpy as np
from neurodynex.hopfield_network import pattern_tools, network



def random_pattern(N, a, n_patterns):
    pattern_generator = pattern_tools.PatternFactory(N, pattern_width=1)
    return pattern_generator.create_random_pattern_list(n_patterns, on_probability=a)


def weight_matrix(n_patterns, N, a, b, patterns):
    w=np.zeros([N, N])
    cprime=1 / a / (1-a) / N
    for i in range(n_patterns):
        w=w+np.dot(patterns[i]-b, np.transpose(patterns[i]-a))*cprime
    for i in range(N):
        w[i,i]=0
    return w


def hamming_distance(x,y):
    return (sum(x)+sum(y)-2*np.dot(x, y)) / N


def map_function(theta_0):
    def map_function_theta(sigma_t0, w):
        sigma_t1=[0.5 * (1 + np.sign(np.dot(wi,sigma_t0) - theta_0)) for wi in w]
        return sigma_t1
    return map_function_theta


def hop_network(N,n_patterns,a,b,theta_0):
    #Creating the random patterns of 1 & 0    
    random_patterns = random_pattern(N, a, n_patterns)
    random_patterns = [(element + 1)/2 for element in random_patterns]  
    #Creating the new weights from the random patterns
    new_weights = weight_matrix(n_patterns, N, a, b, random_patterns)
    net = network.HopfieldNetwork(N)
    #Changing the weights of the network
    net.weights=new_weights
    net.set_dynamics_to_user_function(map_function(theta_0))
    return net, random_patterns


def pattern_mean_error(hamming_distance, m_vals, n_runs, flip_rate, N, 
                       a, b, opt_theta, theta_0 ):
    n_flips = int(flip_rate*N)
    no_patterns=len(m_vals)
    means=np.zeros(no_patterns)
    for i in range(no_patterns):
        theta = (1-opt_theta)*theta_0
        net, random_patterns = hop_network(N,m_vals[i],a,b,theta)
        mean_error=0
        for pattern in random_patterns:
            new_pattern=pattern*2-1
            initial_state = pattern_tools.flip_n(new_pattern, n_flips)
            initial_state=(initial_state + 1)/2
            net.set_state_from_pattern(initial_state)
            net.run(n_runs)
            mean_error+=(hamming_distance(net.state,pattern)/m_vals[i])
        means[i]=mean_error
    return means

def retrievness(hamming_distance, m_vals, n_runs, flip_rate, N, 
                       a, b, opt_theta, theta_0 ):
    n_flips = int(flip_rate*N)
    no_patterns=len(m_vals)
    means=np.zeros(no_patterns)
    no_retrieved=np.zeros(no_patterns)
    for i in range(no_patterns):
        theta = (1-opt_theta)*theta_0
        net, random_patterns = hop_network(N,m_vals[i],a,b,theta)
        mean_error=0
        correctly_retrieved=0
        for pattern in random_patterns:
            new_pattern=pattern*2-1
            initial_state = pattern_tools.flip_n(new_pattern, n_flips)
            initial_state=(initial_state + 1)/2
            net.set_state_from_pattern(initial_state)
            net.run(n_runs)
            correctly_retrieved+=(hamming_distance(net.state,pattern)<0.05)
        no_retrieved[i] = sum(correctly_retrieved)/m_vals[i]
    Capacity=int(np.interp(0.5, np.flip(no_retrieved), np.flip(m_vals)))
    return Capacity



N = 300
a = 0.5
b = 0.5
theta = 0
n_runs = 6
m_vals = (5, 10, 20, 40, 60, 80, 100)
flip_rate = 0.05


def ex2_5():
    mean_error = 0
    number_runs = 3
    for i in range(number_runs):
        mean_error=pattern_mean_error(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                       a=a, b=b, opt_theta=1, theta_0=theta)/number_runs
    print(mean_error)
    plt.plot(m_vals, mean_error)
    plt.title("Ex2-5: Means distances between the final state and the target pattern.")
    plt.xlabel("Number of patterns stored in the network")
    plt.ylabel("Error (measured using Hamming distance)")
    plt.savefig("plots/ex2-5.png")
    plt.show()
    print(np.interp(0.05, mean_error, m_vals))

#x,y=retrievness(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
#                              a=a, b=b, opt_theta=1, theta_0=theta)

def ex2_5_2():
    theta_list=np.linspace(-5, 5, num = 10)
    capacity=np.zeros(len(theta_list))
    

    for i, theta in enumerate(theta_list):
        capacity[i]=retrievness(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                      a=a, b=b, opt_theta=1, theta_0=theta)/N

        
    plt.plot(theta_list, capacity)
    plt.title("Ex2-6: Capacity of the netwrok for different values of theta.")
    plt.xlabel("theta")
    plt.ylabel("Maximum capacity")
    plt.savefig("plots/ex2-6.png")
    plt.show()

def ex2_6():
    theta_list=np.linspace(-0.7, 0.7, num = 10)
    capacity=np.zeros(len(theta_list))
    mean_error=0
    

    for i, theta in enumerate(theta_list):
        capacity[i]=retrievness(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                      a=a, b=b, opt_theta=0, theta_0=theta)/N

        
    plt.plot(theta_list, capacity)
    plt.title("Ex2-6: Capacity of the netwrok for different values of theta.")
    plt.xlabel("theta")
    plt.ylabel("Maximum capacity")
    plt.savefig("plots/ex2-6.png")
    plt.show()


def ex2_7():
    a = 0.1
    b = 0.1
    theta_list=np.linspace(0.3, 1, num = 10)
    capacity=np.zeros(len(theta_list))
    mean_error=0
    
    
    for i, theta in enumerate(theta_list):
        capacity[i]=retrievness(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                      a=a, b=b, opt_theta=0, theta_0=theta)/N
        
    plt.plot(theta_list, capacity)
    plt.title("Ex2-7: Capacity of the netwrok for different values of theta for a=b=0.1.")
    plt.xlabel("theta")
    plt.ylabel("Maximum capacity")
    plt.savefig("plots/ex2-7.png")
    plt.show()


def ex2_7_2():
    a = 0.05
    b = 0.05
    theta_list=np.linspace(0.3, 1, num = 10)
    capacity=np.zeros(len(theta_list))
    mean_error=0
    
    
    for i, theta in enumerate(theta_list):
        capacity[i]=retrievness(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                      a=a, b=b, opt_theta=0, theta_0=theta)/N
    
    plt.plot(theta_list, capacity)
    plt.title("Ex2-7: Capacity of the netwrok for different values of theta for a=b=0.05.")
    plt.xlabel("theta")
    plt.ylabel("Maximum capacity")
    plt.savefig("plots/ex2-7-2.png")
    plt.show()


def ex2_8():
    a = 0.1
    b_list = np.linspace(0.1, 0.6, num=4)
    theta_list=np.linspace(-0.7, 0.7, num=15)
    capacity=np.zeros([len(b_list),len(theta_list)])
    
    for j, b in enumerate(b_list):
        for i, theta in enumerate(theta_list):
            capacity[j,i] = retrievness(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                          a=a, b=b, opt_theta=0, theta_0=theta)/N
    
    plt.figure('Figure 2.8')
    for j, b in enumerate(b_list):
        plt.plot(theta_list, capacity[j])
    lines=["b={}".format(b) for b in b_list]
    plt.legend(lines)
    plt.title("Ex2-8: Capacity of the netwrok for different values of theta and b for a=0.1.")
    plt.xlabel("theta")
    plt.ylabel("Maximum capacity")
    plt.savefig("plots/ex2-8.png")
    plt.show()



#ex2_5()
#ex2_5_2()
ex2_6()
ex2_7()
ex2_7_2()
ex2_8()

