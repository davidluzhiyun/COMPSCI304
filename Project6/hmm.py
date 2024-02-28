import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

import feature_extraction



# initialization of segmentation for one template
# a template is a numpy array with axis0 for frames and axis1 for features
def uniform_segmentation(template, k):
    num_frames, num_features = template.shape
    segment_length = num_frames // k
    remainder = num_frames % k

    # Create labels based on uniform segmentation with equal distribution of remaining frames
    labels = np.concatenate([np.full(segment_length + (1 if i < remainder else 0), i) for i in range(k)])

    return labels


# calculate the means and cov for each segment
def calculate_segments_means_covariances(templates, labels, segment_count):
    num_features = templates[0].shape[1]

    means = np.zeros((segment_count, num_features))
    covariances = np.zeros((segment_count, num_features, num_features))

    for segment in range(segment_count):
        segment_frames = np.concatenate(
            [template[np.where(label == segment)[0]] for template, label in zip(templates, labels)])

        # Calculate mean across templates for the segment
        means[segment] = np.mean(segment_frames, axis=0)

        # Calculate covariance across templates for the segment
        covariances[segment] = np.cov(segment_frames, rowvar=False)

    return means, covariances


# template function for calculating cost
def calculate_node_cost(feature_vector, gmm_parameters):
    total_cost = 0

    cluster_centers, variances, weights = gmm_parameters
    for i in range(len(cluster_centers)):
        mean = cluster_centers[i]
        variance = variances[i]
        weight = weights[i]
        # Calculate the multivariate normal log probability
        prob = multivariate_normal.pdf(feature_vector, mean=mean, cov=variance)

        # Accumulate the cost with weights
        total_cost += weight * prob

    return -np.log(total_cost)


# Default calculation for transition and entry score
# Using negative log
def calculate_state_transition_entrance_exit_costs(labels, k):
    num_templates = len(labels)

    # Initialize count matrices
    transition_counts = np.zeros((k, k))
    entrance_counts = np.zeros(k)
    exit_counts = np.zeros(k)

    # Count transitions and entrances
    for label_sequence in labels:
        for i in range(len(label_sequence) - 1):
            transition_counts[label_sequence[i], label_sequence[i + 1]] += 1
        entrance_counts[label_sequence[0]] += 1
        exit_counts[label_sequence[-1]] += 1

    # Calculate probabilities
    # (np.sum(transition_counts, axis=1, keepdims=True) + exit_counts.reshape([-1,1]) gives the number of each state in [-1,1] shape
    state_counts = np.sum(transition_counts, axis=1, keepdims=True) + exit_counts.reshape([-1,1])
    state_transition_probs = transition_counts / state_counts
    entrance_probs = entrance_counts / np.sum(entrance_counts)
    exit_probs = np.divide(exit_counts, state_counts.reshape([-1,]))

    # Convert probabilities to minus log probabilities
    state_transition_scores = -np.log(state_transition_probs)
    entrance_scores = -np.log(entrance_probs)
    exit_scores = -np.log(exit_probs)

    return state_transition_scores, entrance_scores, exit_scores


# create the node cost functions for the segments
# Uses the mahalanobis distance
# swap out for mixed gaussian
def create_node_cost_functions_mahalanobis(means, covariances):
    num_segments, num_features = means.shape[0], means.shape[1]

    alignment_cost_functions = []

    for segment in range(num_segments):
        mean = means[segment]
        covariance_inv = np.linalg.inv(covariances[segment])

        # written as default value to prevent late binding
        def alignment_cost(feature_vector, mean=mean, covariance_inv=covariance_inv):
            diff = feature_vector - mean
            mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, covariance_inv), diff.T))
            return mahalanobis_distance

        alignment_cost_functions.append(alignment_cost)

    return alignment_cost_functions


# perform alignment for one template
def viterbi_alignment(template, alignment_cost_functions, state_transition_scores, entrance_scores, exit_scores):
    num_states = len(alignment_cost_functions)
    num_frames, num_features = template.shape

    # Initialize matrices
    viterbi_matrix = np.full((num_frames, num_states), np.inf)
    backtrack_matrix = np.zeros((num_frames, num_states), dtype=int)

    # Initialize the first column of the Viterbi matrix
    for state in range(num_states):
        viterbi_matrix[0, state] = entrance_scores[state] + alignment_cost_functions[state](template[0])

    # Forward pass: Fill in the Viterbi matrix
    for i in range(1, num_frames):
        for j in range(num_states):
            c = alignment_cost_functions[j](template[i])
            # Alows skipping 1 state
            # Special cases for j = 0 and 1
            if j == 0:
                p = viterbi_matrix[i - 1, j] + state_transition_scores[j, j] + c
                backtrack_matrix[i, j] = 0
            elif j == 1:
                potential_prev = np.array([viterbi_matrix[i - 1, j] + state_transition_scores[j, j],
                                           viterbi_matrix[i - 1, j - 1] + state_transition_scores[j - 1, j]])
                prev_path = np.argmin(potential_prev)
                backtrack_matrix[i, j] = j - prev_path
                p = potential_prev[prev_path] + c

            else:
                potential_prev = np.array([viterbi_matrix[i - 1, j] + state_transition_scores[j, j],
                                           viterbi_matrix[i - 1, j - 1] + state_transition_scores[j - 1, j]
                                              , viterbi_matrix[i - 1, j - 2] + state_transition_scores[j - 2, j]])
                prev_path = np.argmin(potential_prev)
                backtrack_matrix[i, j] = j - prev_path
                p = potential_prev[prev_path] + c
            viterbi_matrix[i, j] = p

    # modification
    # Add the exit costs
    # it only contains two number, but we retain a whole array for convenience in calculation and compatibility
    viterbi_matrix[-1, num_states - 1] += exit_scores[-1]
    viterbi_matrix[-1, num_states - 2] += exit_scores[-2]

    # Backward pass: Find the best path
    # Made changes such that alignment could end with final state or the state before the finale state
    # assert num_states>=2
    if viterbi_matrix[-1, num_states-1] <= viterbi_matrix[-1, num_states-2]:
        total_cost = viterbi_matrix[-1, -1]
        best_path = [num_states - 1]
    else:
        total_cost = viterbi_matrix[-1, -1]
        best_path = [num_states - 2]

    for frame in range(num_frames - 1, 0, -1):
        best_path.append(backtrack_matrix[frame, best_path[-1]])

    best_path.reverse()

    return np.array(best_path), total_cost


def segmental_k_means(templates, num_segments, max_iterations=100, epsilon=0.001):
    # assertion to make sure the new exit costs feature works properly
    assert num_segments >= 2
    # Initialize with uniform segmentation
    segmentations = [uniform_segmentation(template, num_segments) for template in templates]
    # initialize total cost as inf
    total_cost = np.inf
    for i in range(max_iterations):

        means, covariances = calculate_segments_means_covariances(templates, segmentations, num_segments)
        state_transition_scores, entrance_scores, exit_scores = calculate_state_transition_entrance_exit_costs(segmentations,
                                                                                                               num_segments)
        node_cost_funtions = create_node_cost_functions_mahalanobis(means, covariances)
        labels_and_costs = [viterbi_alignment(template, node_cost_funtions, state_transition_scores, entrance_scores, exit_scores)
                            for template in templates]
        new_total_cost = 0
        new_labels = []
        for label, cost in labels_and_costs:
            new_labels.append(label)
            new_total_cost += cost
        improvement = total_cost - new_total_cost
        total_cost = new_total_cost
        segmentations = new_labels
        if improvement >= 0 and improvement < epsilon:
            break
    means, covariances = calculate_segments_means_covariances(templates, segmentations, num_segments)
    node_cost_functions = create_node_cost_functions_mahalanobis(means, covariances)
    state_transition_scores, entrance_scores, exit_scores = calculate_state_transition_entrance_exit_costs(segmentations, num_segments)
    return node_cost_functions, state_transition_scores, entrance_scores, exit_scores, segmentations

