# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:28:22 2022

@author: tahat
"""

import matplotlib.pyplot as plt
import numpy as np
from neurodynex.hopfield_network import pattern_tools, network



def random_pattern_generator(N, a, n_patterns):
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
    random_patterns = random_pattern_generator(N, a, n_patterns)
    random_patterns = [(element + 1)/2 for element in random_patterns]  
    #Creating the new weights from the random patterns
    new_weights = weight_matrix(n_patterns, N, a, b, random_patterns)
    net = network.HopfieldNetwork(N)
    net.set_dynamics_sign_async()
    net.set_dynamics_to_user_function(map_function(theta_0))
    #Changing the weights of the network
    net.weights=new_weights
    return net, random_patterns



def Capacity(hamming_distance, m_vals, n_runs, flip_rate, N, 
                       a, b, opt_theta, theta_0 ):
    n_flips = int(flip_rate*N)
    no_patterns=len(m_vals)
    no_retrieved=np.zeros(no_patterns)
    for i in range(no_patterns):
        #If opt_theta=1, theta would be equal to 0.
        #Otherwise the value of theta would be assigned by the inputs
        theta = (1-opt_theta)*theta_0
        #Creating the network and patterns of 1 & 0
        net, random_patterns = hop_network(N,m_vals[i],a,b,theta)
        correctly_retrieved=0
        for pattern in random_patterns:
            #Creating an initial state of 1 & 0 with 15 flipped bits
            initial_state = pattern_tools.flip_n(pattern*2-1, n_flips)
            initial_state=(initial_state + 1)/2
            #Running the network
            net.set_state_from_pattern(initial_state)
            net.run(n_runs)
            #Measuring the distance between the final state and the pattern
            correctly_retrieved+=(hamming_distance(net.state,pattern)<0.05)
        #Calculating the percentage of the retrieved patterns.
        no_retrieved[i] = sum(correctly_retrieved)/m_vals[i]
    Capacity=int(np.interp(0.95, np.flip(no_retrieved), np.flip(m_vals)))
    return Capacity, no_retrieved



N = 300
a = 0.5
b = 0.5
theta = 0
n_runs = 6
m_vals = (5, 20, 30, 40, 50, 60, 80, 100)
flip_rate = 0.05



def ex2_5():

    capacity=Capacity(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                      a=a, b=b, opt_theta=1, theta_0=theta)[1]*100

        
    plt.plot(m_vals, capacity)
    plt.title("Ex2-5: Percentage of the correctly retrieved patterns.")
    plt.xlabel("Dictionary size")
    plt.ylabel("Percentage of the correctly retrieved patterns")
    plt.savefig("plots/ex2-5.png")
    plt.show()

def ex2_6():
    theta_list=np.linspace(0.1, 0.9, num = 14)
    capacity=np.zeros(len(theta_list))

    for i, theta in enumerate(theta_list):
        capacity[i]=Capacity(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                      a=a, b=b, opt_theta=0, theta_0=theta)[0]/N

        
    plt.plot(theta_list, capacity)
    plt.title("Ex2-6: Capacity of the netwrok for different values of theta.")
    plt.xlabel("theta")
    plt.ylabel("Maximum capacity")
    plt.savefig("plots/ex2-6.png")
    plt.show()


def ex2_7():
    a = 0.1
    b = 0.1
    theta_list=np.linspace(0.1, 0.9, num = 14)
    capacity=np.zeros(len(theta_list))
    
    for i, theta in enumerate(theta_list):
        capacity[i]=Capacity(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                      a=a, b=b, opt_theta=0, theta_0=theta)[0]/N
        
    plt.plot(theta_list, capacity)
    plt.title("Ex2-7: Capacity of the netwrok for different values of theta for a=b=0.1.")
    plt.xlabel("theta")
    plt.ylabel("Maximum capacity")
    plt.savefig("plots/ex2-7.png")
    plt.show()


def ex2_7_2():
    a = 0.05
    b = 0.05
    theta_list=np.linspace(0.1, 0.9, num = 14)
    capacity=np.zeros(len(theta_list))
    
    for i, theta in enumerate(theta_list):
        capacity[i]=Capacity(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                      a=a, b=b, opt_theta=0, theta_0=theta)[0]/N
    
    plt.plot(theta_list, capacity)
    plt.title("Ex2-7: Capacity of the netwrok for different values of theta for a=b=0.05.")
    plt.xlabel("theta")
    plt.ylabel("Maximum capacity")
    plt.savefig("plots/ex2-7-2.png")
    plt.show()


def ex2_8():
    a = 0.1
    b_list = np.linspace(0.05, 0.2, num=4)
    theta_list=np.linspace(0, 0.7, num=15)
    capacity=np.zeros([len(b_list),len(theta_list)])
    
    for j, b in enumerate(b_list):
        for i, theta in enumerate(theta_list):
            capacity[j,i] = Capacity(hamming_distance, m_vals=m_vals, n_runs=n_runs, N=N, flip_rate=flip_rate, 
                                          a=a, b=b, opt_theta=0, theta_0=theta)[0]/N
    
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
#ex2_6()
#ex2_7()
#ex2_7_2()
#ex2_8()
