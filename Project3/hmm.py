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
def calculate_global_means_covariances(templates, labels, k):
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

def segmental_k_means(templates, num_segments, max_iterations=100):
    # Initialize with uniform segmentation
    segmentations = [uniform_segmentation(template, num_segments) for template in templates]
    means, covariances = calculate_global_means_covariances(templates,segmentations, num_segments)
