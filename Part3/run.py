import numpy as np
import matplotlib.pyplot as plt
from neurodynex.hopfield_network import network, pattern_tools


# ----- Helper functions ----- #


def create_random_patterns(n_exc, n_inh, a, n_patterns):
    pattern_generator = pattern_tools.PatternFactory(n_exc, pattern_width=1)
    patterns = pattern_generator.create_random_pattern_list(n_patterns, on_probability=a)
    patterns_final = []
    for p in patterns:
        zeros = np.zeros((n_exc+n_inh, 1))
        zeros[:n_exc] = p
        patterns_final.append(zeros)
    return patterns_final


def convert_patterns_to_0_and_1(patterns):
    return [(pattern + 1) / 2 for pattern in patterns]


def generate_weight_matrix(patterns, a, K, n_exc, n_inh):
    N = n_exc + n_inh
    weights = np.zeros((N, N))

    for pattern in patterns:
        # Set weights exc -> exc
        weights[:n_exc, :n_exc] += (1/n_exc) * np.outer(pattern[:n_exc], pattern[:n_exc])
        # Set weights inh -> exc
        weights[n_exc:, :n_exc] += - (a/n_inh) * pattern[n_exc:]
    # Set weights exc -> inh
    for i in range(n_inh):
        tmp = np.zeros(n_exc)
        random_exc_neurons = np.random.choice(np.arange(n_exc), size=K)
        tmp[random_exc_neurons] = 1/K
        weights[:n_exc, n_exc+i] = tmp
    # Weights inh -> inh are all 0 anyway, nothing to change there

    np.fill_diagonal(weights, 0)
    return weights


def get_update_function(n_exc):
    def update_dynamics(sigma_t0, w):
        # Update each excitatory neuron according to equation (6)
        sigma_t1_exc = 0.5 * (1 + np.sign(np.dot(w[:, :n_exc].T, sigma_t0)))
        sigma_t1_exc = np.where(sigma_t1_exc == 0.5, 0, sigma_t1_exc)  # Set inconsistent states to 0
        # Update each inhibitory neuron according to equation (10)
        h = np.sum(np.dot(w[:n_exc, :n_exc], w[:n_exc, n_exc:]), axis=0)
        sigma_t1_inh = (np.random.random(size=h.shape) < h).astype(int)
        return np.concatenate((sigma_t1_exc, sigma_t1_inh))
    return update_dynamics


def compute_hamming_distance(pattern1, pattern2):
    """
    Computes the hamming distance between two patterns.
    Shapes of the patterns must match, otherwise an exception is raised.
    :param pattern1: First pattern
    :param pattern2: Second pattern
    :return: computed distance
    """

    assert pattern1.shape == pattern2.shape, "Shapes of the patterns do not match!"

    N = np.prod(pattern1.shape)
    return (np.sum(pattern1) + np.sum(pattern2) - 2*np.dot(pattern1.flatten(), pattern2.flatten())) / N


def compute_distances(net, patterns, n_runs, distance_function, n_exc, percentage=0.15):
    """
    Computes the distance between final state and target pattern for each pattern stored in the network.
    :param net: Hopfield network
    :param patterns: Patterns stored in the network, but as -1 and 1
    :param n_runs: Number of runs to perform to reach the final state
    :param distance_function: Function used to compute the distance
    :param percentage: Percentage of flipped neurons in the initial state
    :return: Computed distances
    """

    distances = np.zeros(len(patterns))
    n_flips = int(net.nrOfNeurons * percentage)
    for i, pattern in enumerate(patterns):
        initial_state = pattern_tools.flip_n(pattern[:n_exc], n_flips)  # Only flip excitatory neurons
        initial_state = convert_patterns_to_0_and_1([initial_state])[0]  # Convert them to 0 and 1
        initial_state_final = np.zeros((len(pattern), 1))
        initial_state_final[:n_exc] = initial_state
        net.set_state_from_pattern(initial_state_final)
        net.run(n_runs)
        final_state = net.state
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


def ex3(n_patterns, a, sync=None, n_exc=300, n_inh=80):
    N = n_exc + n_inh
    K = 60

    # Create network with excitatory + inhibitory neurons
    net = network.HopfieldNetwork(N)
    # Generate random patterns, map {-1, 1} to {0, 1}
    patterns = create_random_patterns(n_exc, n_inh, a, n_patterns)
    patterns_0_and_1 = convert_patterns_to_0_and_1(patterns)
    # Set update sync
    if sync == True:
        net.set_dynamics_sign_sync()
    elif sync == False:
        net.set_dynamics_sign_async()
    # Else, leave default, whatever it is
    # Generate weight matrix
    weights = generate_weight_matrix(patterns_0_and_1, a, K, n_exc, n_inh)
    # Save matrix to network
    net.weights = weights
    # Create dynamics function
    update_dynamics = get_update_function(n_exc)
    # Set dynamics function to network
    net.set_dynamics_to_user_function(update_dynamics)

    return net, patterns  # Careful, patterns returned here are between -1 and 1!


def ex4(n_steps, n_trials, sync=None, ex_number="4", return_mean_percentages=False):

    n_exc = 300
    n_inh = 80

    a = 0.1
    dict_sizes = range(1, 10, 1)
    all_percentages = []
    all_capacities = []
    distance_function = compute_hamming_distance

    for i in range(n_trials):
        distances = []
        for n_patterns in dict_sizes:
            net, patterns = ex3(n_patterns, a, sync=sync, n_exc=n_exc, n_inh=n_inh)
            dist = compute_distances(net, patterns, n_steps, distance_function, n_exc, percentage=0.05)  # c = 5% when not specified
            distances.append(dist)
        capacity, correctly_retrieved_percentage = compute_m_max(dict_sizes, distances, tol_distance=0.1)
        all_percentages.append(correctly_retrieved_percentage)
        all_capacities.append(capacity)

    mean_of_percentages = np.mean(all_percentages, axis=0)
    std_of_percentages = np.std(all_percentages, axis=0)

    plt.errorbar(dict_sizes, mean_of_percentages, yerr=std_of_percentages)
    plt.title("Ex{}: Means of percentage of correctly retrieved patterns with errorbars.".format(ex_number))
    plt.xlabel("Number of patterns stored in the network")
    plt.ylabel("Percentage of correctly retrieved patterns")
    plt.savefig("plots/ex{}.png".format(ex_number))
    plt.show()

    if return_mean_percentages:
        return dict_sizes, mean_of_percentages
    else:
        return np.mean(all_capacities)


def ex5(n_steps, n_runs):
    sync = [False, True]
    suffixes = ["sync", "async"]
    mean_percentages = []
    dict_sizes = None
    for s, suffix in zip(sync, suffixes):
        dict, mean = ex4(n_steps, n_runs, sync=s, ex_number="5_{}".format(suffix), return_mean_percentages=True)
        dict_sizes = dict  # Both have the same dictionary sizes
        mean_percentages.append(mean)

    for i in range(len(sync)):
        plt.plot(dict_sizes, mean_percentages[i], label=suffixes[i])
    plt.title("Ex5: Means of percentage of correctly retrieved patterns comparison")
    plt.xlabel("Number of patterns stored in the network")
    plt.ylabel("Percentage of correctly retrieved patterns")
    plt.legend()
    plt.savefig("plots/ex5_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("Exercise 4 (and 3) results:")
    mean_capacity = ex4(6, 6)
    print("Mean m_max value of the network for sparseness a = 0.1:", mean_capacity)
    ex5(15, 15)
