import numpy as np
from neurodynex.hopfield_network import network, pattern_tools


def ex1():
    """
    Execute exercise 1.
    :return: hopfield network, generated random patterns
    """
    N = 300
    n_patterns = 5
    net = network.HopfieldNetwork(N)
    pattern_generator = pattern_tools.PatternFactory(300, pattern_width=1)
    random_patterns = pattern_generator.create_random_pattern_list(n_patterns)
    net.store_patterns(random_patterns)
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
    initial_state = selected_pattern  # Flip 15 randomly chosen neurons TODO
    net.set_state_from_pattern(initial_state)

    # Run the network
    n_steps = 6
    net.run(nr_steps=n_steps)
    final_state = net.state  # TODO: let TAs know about the shape mistake in the documentation of the framework

    # Calculate distance between the final state and all the patterns
    distances = [distance_function(final_state, np.squeeze(pattern)) for pattern in stored_patterns]
    correctly_retrieved = True if distances[selected_pattern_id] <= 0.05 else False

    return distances, correctly_retrieved, selected_pattern_id


def ex4():
    """"""
    raise NotImplementedError


if __name__ == "__main__":
    net, stored_patterns = ex1()
    compute_hamming_distance = ex2()  # TODO: answer theory question of ex2
    distances, correctly_retrieved, selected_pattern_id = ex3(net, stored_patterns, compute_hamming_distance)

