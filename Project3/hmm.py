import numpy as np
from sklearn.cluster import KMeans

# initialization of segmentation for one template
# a template is a numpy array with axis0 for frames and axis1 for features
def uniform_segmentation(template, k):
    num_frames, num_features = template.shape
    segment_length = num_frames // k

    # Create labels based on uniform segmentation
    labels = np.concatenate([np.full(segment_length, i) for i in range(k)])
    return labels


# calculate the means and cov for each segment
def calculate_segments_means_covariances(templates, labels, k):
    num_features = templates[0].shape[1]

    means = np.zeros((k, num_features))
    covariances = np.zeros((k, num_features, num_features))

    for segment in range(k):
        segment_frames = np.concatenate(
            [template[np.where(label == segment)[0]] for template, label in zip(templates, labels)])

        # Calculate mean across templates for the segment
        means[segment] = np.mean(segment_frames, axis=0)

        # Calculate covariance across templates for the segment
        covariances[segment] = np.cov(segment_frames, rowvar=False)

    return means, covariances

# Default calculation for transition and entry score
# Using negative log
def calculate_state_transition_entrance_costs(labels, k):
    num_templates = len(labels)

    # Initialize count matrices
    transition_counts = np.zeros((k, k))
    entrance_counts = np.zeros(k)

    # Count transitions and entrances
    for label_sequence in labels:
        for i in range(len(label_sequence) - 1):
            transition_counts[label_sequence[i], label_sequence[i + 1]] += 1
        entrance_counts[label_sequence[0]] += 1

    # Calculate probabilities
    state_transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
    entrance_probs = entrance_counts / np.sum(entrance_counts)

    # Convert probabilities to minus log probabilities
    state_transition_scores = -np.log(state_transition_probs)
    entrance_scores = -np.log(entrance_probs)

    return state_transition_scores, entrance_scores

# create the node cost functions for the segments
# Uses the mahalanobis distance
# swap out for mixed gaussian
def create_node_cost_functions_mahalanobis(means, covariances):
    num_segments, num_features = means.shape[0], means.shape[1]

    alignment_cost_functions = []

    for segment in range(num_segments):
        mean = means[segment]
        covariance_inv = np.linalg.inv(covariances[segment])

        def alignment_cost(feature_vector):
            diff = feature_vector - mean
            mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, covariance_inv), diff.T))
            return mahalanobis_distance

        alignment_cost_functions.append(alignment_cost)

    return alignment_cost_functions

def segmental_k_means(templates, num_segments, max_iterations=100):
    # Initialize with uniform segmentation
    segmentations = [uniform_segmentation(template, num_segments) for template in templates]
    means, covariances = calculate_segments_means_covariances(templates,segmentations, num_segments)

