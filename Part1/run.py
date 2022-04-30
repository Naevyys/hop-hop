import numpy as np
import matplotlib.pyplot as plt
from neurodynex.hopfield_network import network, pattern_tools


# ----- Helper functions ----- #


def generate_and_store_random_patterns(net, n_patterns):
    """
    Resets weights of the network passed and learn new weights for n_patterns random patterns.
    :param net: hopfield network
    :param n_patterns: number of patterns
    :return: updated hopfield network, generated random patterns
    """

    pattern_generator = pattern_tools.PatternFactory(300, pattern_width=1)
    random_patterns = pattern_generator.create_random_pattern_list(n_patterns)
    net.reset_weights()
    net.store_patterns(random_patterns)

    return net, random_patterns


# ----- Exercises code ----- #


def ex1():
    """
    Execute exercise 1.
    :return: hopfield network, generated random patterns
    """
    N = 300
    n_patterns = 5
    net = network.HopfieldNetwork(N)
    net, random_patterns = generate_and_store_random_patterns(net, n_patterns)
    return net, random_patterns


def ex2():
    """
    Execute exercise 2.
    :return: Function to compute the hamming distance between two patterns
    """
    def compute_hamming_distance(pattern1, pattern2):
        """
        Computes the hamming distance between two patterns.
        Hamming distance is defined in our case as (N - dot(pattern1, pattern2)) / 2N, where N is the number of neurons.
        Shapes of the patterns must match, otherwise an exception is raised.
        :param pattern1: First pattern
        :param pattern2: Second pattern
        :return:
        """

        assert pattern1.shape == pattern2.shape, "Shapes of the patterns do not match!"

        N = np.prod(pattern1.shape)
        return (N - np.dot(pattern1.flatten(), pattern2.flatten())) / 2*N

    return compute_hamming_distance


def ex3(net, stored_patterns, distance_function, selected_pattern_id=None):
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
    n_flips = 15
    initial_state = pattern_tools.flip_n(selected_pattern, n_flips)  # Flip 15 randomly chosen neurons
    net.set_state_from_pattern(initial_state)

    # Run the network
    n_steps = 6
    net.run(nr_steps=n_steps)
    final_state = net.state  # TODO: let TAs know about the shape mistake in the documentation of the framework

    # Calculate distance between the final state and all the patterns
    distances = [distance_function(final_state, np.squeeze(pattern)) for pattern in stored_patterns]
    correctly_retrieved = True if distances[selected_pattern_id] <= 0.05 else False

    return distances, correctly_retrieved, selected_pattern_id


def ex4(net, distance_function, m_vals=(5, 20, 40, 60, 80, 100), n_runs=6, plot=True):
    """
    Computes for each value of m the mean distance between the final state of the network and the target pattern, and
    plot the results if plot == True.

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
    :param plot: Whether to plot and save the plot or not
    :return: mean distances
    """

    def compute_mean_distance(n_patterns):
        network, random_patterns = generate_and_store_random_patterns(net, n_patterns)
        distances = np.zeros(n_patterns)
        n_flips = 15  # 5% of 300
        for i, pattern in enumerate(random_patterns):
            initial_state = pattern_tools.flip_n(pattern, n_flips)
            network.set_state_from_pattern(initial_state)
            network.run(n_runs)
            final_state = network.state
            distances[i] = distance_function(final_state, np.squeeze(pattern))
        return np.mean(distances)

    means = np.array([compute_mean_distance(m) for m in m_vals])

    if plot:
        plt.plot(m_vals, means)
        plt.title("Ex4: Means distances between final state and target pattern.")
        plt.xlabel("Number of patterns stored in the network")
        plt.ylabel("Error (measured using Hamming distance)")
        plt.savefig("plots/ex4.png")
        plt.show()

    return means


def ex5(net, distance_function, n_trials=8):
    """
    Repeats experiment of ex4 several times to obtain error bars on the means, and plots the results.
    :param net: hopfield network
    :param distance_function: function used to compute the distance
    :param n_trials: number of times the experiment should be repeated to get the error bars.
    :return: mean of means, std of means
    """

    m_vals = (5, 20, 40, 60, 80, 100)
    means = np.array([ex4(net, distance_function, m_vals=m_vals, plot=False) for _ in range(n_trials)])
    mean_of_means = np.mean(means, axis=0)
    std_of_means = np.std(means, axis=0)  # TODO: change if needed according to TA answers

    plt.errorbar(m_vals, mean_of_means, yerr=std_of_means)
    plt.title("Ex5: Means distances with errorbars between final state and target pattern.")
    plt.xlabel("Number of patterns stored in the network")
    plt.ylabel("Error (measured using Hamming distance)")
    plt.savefig("plots/ex5.png")
    plt.show()

    return mean_of_means, std_of_means


if __name__ == "__main__":
    net, stored_patterns = ex1()
    compute_hamming_distance = ex2()  # TODO: answer theory question of ex2
    distances, correctly_retrieved, selected_pattern_id = ex3(net, stored_patterns, compute_hamming_distance)
    means = ex4(net, compute_hamming_distance)  # TODO: answer theory question of ex4
    mean_of_means, std_of_means = ex5(net, compute_hamming_distance)


# Questions for the TAs:
# - Ex4: "what is the maximum number of patterns that can be retrieved?" What kind of answer is expected from us there?
#   A theory answer or a computational answer?
# - Ex5: Which error bars are you expecting? min/max? std? 95% confidence interval?

