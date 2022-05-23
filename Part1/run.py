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

    pattern_generator = pattern_tools.PatternFactory(net.nrOfNeurons, pattern_width=1)
    random_patterns = pattern_generator.create_random_pattern_list(n_patterns)
    net.reset_weights()
    net.store_patterns(random_patterns)

    return net, random_patterns


def compute_distances(net, n_patterns, n_runs, distance_function, percentage=0.15):
    """
    Computes the distance between final state and target pattern for each pattern stored in the network.
    :param net: Hopfield network
    :param n_patterns: Number of patterns to store in the network
    :param n_runs: Number of runs to perform to reach the final state
    :param distance_function: Function used to compute the distance
    :param percentage: Percentage of flipped neurons in the initial state
    :return: Computed distances
    """

    network, random_patterns = generate_and_store_random_patterns(net, n_patterns)
    distances = np.zeros(n_patterns)
    n_flips = int(network.nrOfNeurons*percentage)
    for i, pattern in enumerate(random_patterns):
        initial_state = pattern_tools.flip_n(pattern, n_flips)
        network.set_state_from_pattern(initial_state)
        network.run(n_runs)
        final_state = network.state
        distances[i] = distance_function(final_state, np.squeeze(pattern))
    return distances


def compute_m_max(m_vals, distances, tol_distance=0.05, tol_error_percentage=0.05):
    """
    Compute the maximum number of patterns that a network can store to reach an error rate of at most 1 - tol_error_percentage
    :param m_vals: M values for which the distances were computed
    :param distances: List of distance lists, the order must match the order of m_vals
    :param tol_distance: Distance tolerance for considering a pattern as correctly retrieved
    :param tol_error_percentage: Tolerance for the error rate
    :return: maximum number of patterns that can be stored, percentages of correctly retrieved patterns
    """

    correctly_retrieved = [distance <= tol_distance for distance in distances]
    correctly_retrieved_percentage = np.array([np.count_nonzero(item) / len(item) for item in correctly_retrieved])
    m_max = np.array(m_vals)[correctly_retrieved_percentage > (1 - tol_error_percentage)][-1]

    return m_max, correctly_retrieved_percentage*100


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
        :return: computed distance
        """

        assert pattern1.shape == pattern2.shape, "Shapes of the patterns do not match!"

        N = np.prod(pattern1.shape)
        return (N - np.dot(pattern1.flatten(), pattern2.flatten())) / (2*N)

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


def ex4(net, distance_function, m_vals=(5, 20, 30, 40, 60, 80, 100), n_runs=6, tol_distance=0.05, tol_error=0.05):
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
    :param tol_distance: Max distance considered as correctly classified.
    :param tol_error: Percentage of error tolerance for retrieving patterns correctly
    :return: mean distances, m max if is_ex4, means otherwise
    """

    distances = [compute_distances(net, m, n_runs, distance_function) for m in m_vals]
    means = np.array([np.mean(distance) for distance in distances])
    m_max, correctly_retrieved_percentage = compute_m_max(m_vals, distances, tol_distance=tol_distance, tol_error_percentage=tol_error)

    plt.plot(m_vals, means)
    plt.title("Ex4: Means distance between final state and target pattern.")
    plt.xlabel("Number of patterns stored in the network")
    plt.ylabel("Error (measured using Hamming distance)")
    plt.savefig("plots/ex4.png")
    plt.show()

    plt.plot(m_vals, correctly_retrieved_percentage)
    plt.title("Ex4: Percentage of correctly retrieved patterns.")
    plt.xlabel("Number of patterns stored in the network")
    plt.ylabel("Percentage of correctly retrieved patterns")
    plt.savefig("plots/ex4_percentage.png")
    plt.show()

    return means, m_max, m_vals


def ex5(net, distance_function, n_trials=8, m_vals=(5, 20, 30, 40, 60, 80, 100), n_runs=6):
    """
    Repeats experiment of ex4 several times to obtain error bars on the means, and plots the results.
    :param net: hopfield network
    :param distance_function: function used to compute the distance
    :param n_trials: number of times the experiment should be repeated to get the error bars.
    :param m_vals: List of different values for m
    :param n_runs: Number of runs to reach the state to compare with the target pattern
    :return: mean of means, std of means
    """

    all_means = []
    for _ in range(n_trials):
        distances = [compute_distances(net, m, n_runs, distance_function) for m in m_vals]
        means = np.array([np.mean(distance) for distance in distances])
        all_means.append(means)

    all_means = np.stack(all_means)
    mean_of_means = np.mean(all_means, axis=0)
    std_of_means = np.std(all_means, axis=0)

    plt.errorbar(m_vals, mean_of_means, yerr=std_of_means)
    plt.title("Ex5: Means distance with errorbars between final state and target pattern.")
    plt.xlabel("Number of patterns stored in the network")
    plt.ylabel("Error (measured using Hamming distance)")
    plt.savefig("plots/ex5.png")
    plt.show()

    return mean_of_means, std_of_means


def ex7(distance_function, net_sizes=(50, 250, 500, 750, 1000), n_runs=6, tol_distance=0.05, tol_error=0.05):
    """"""

    capacity = np.zeros(len(net_sizes))

    for i, net_size in enumerate(net_sizes):
        net = network.HopfieldNetwork(net_size)
        m_vals = (int(net_size*0.05), int(net_size*0.1), int(net_size*0.2), int(net_size*0.3))  # We take 5, 10, 20 and 30% of the network size

        distances = [compute_distances(net, m, n_runs, distance_function, percentage=0) for m in m_vals]  # We take the target pattern as original pattern
        m_max, percentage_correctly_retrieved = compute_m_max(m_vals, distances, tol_distance=tol_distance, tol_error_percentage=tol_error)
        capacity[i] = m_max / net_size

        plt.plot(m_vals, percentage_correctly_retrieved)
        plt.title("Ex7: % of retrieved patterns for net size {} starting from pattern".format(net_size))
        plt.xlabel("Number of patterns stored")
        plt.ylabel("Percentage of correctly retrieved patterns")
        plt.savefig("plots/ex7_percentage_net_size_{}.png".format(net_size))
        plt.show()

    plt.plot(net_sizes, capacity)
    plt.title("Ex7: Capacity of the network by network size")
    plt.xlabel("Size of the network")
    plt.ylabel("Capacity")
    plt.savefig("plots/ex7.png")
    plt.show()

    return capacity


if __name__ == "__main__":
    # TODO: Format and print results nicely for when they run the code
    # Exercise 1
    net, stored_patterns = ex1()
    # Exercise 2
    compute_hamming_distance = ex2()
    # Exercise 3
    distances, correctly_retrieved, selected_pattern_id = ex3(net, stored_patterns, compute_hamming_distance)
    print("Exercise 3 results:")
    print(f"We selected pattern number {selected_pattern_id} for our experiments")
    print(f"Distances between final state of the network and each pattern: {distances}")
    print(f"The network retrieved the selected pattern correctly: {correctly_retrieved}")
    # Exercise 4
    means, m_max, m_vals = ex4(net, compute_hamming_distance)
    print("Exercise 4 results:")
    print(f"Dictionary sizes: {m_vals}")
    print(f"Mean error of pattern retrieval: {means}")
    print(f"Maximum number of patterns that can be retrieved: {m_max}")
    # Exercise 5
    mean_of_means, std_of_means = ex5(net, compute_hamming_distance)  # This takes a while to run
    # Exercise 6 - all in the report
    # Exercise 7
    capacity = ex7(compute_hamming_distance)  # This takes an even longer while to run
