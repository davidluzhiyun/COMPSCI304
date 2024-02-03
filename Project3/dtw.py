import numpy as np
from scipy.spatial.distance import cdist
import feature_extraction
from sklearn.metrics.pairwise import euclidean_distances
from queue import PriorityQueue


def dist(x, y):
    return euclidean_distances(x.reshape(1, -1), y.reshape(1, -1))


# The basic dtw routine
# Both input and template have shape (frame count, feature count)
# Uses Euclidean distance, allows Horizontal and Diagonal transition up to 1/2 shrinking
# returns cost
def dtw(input_audio, template):
    # get the length of the input and template
    n, m = len(input_audio), len(template)

    # Initialize p matrix
    # The padded version
    p0 = np.full((n + 1, m + 2), np.inf)
    # The unpadded p1 = p0[1:,2:]
    for i in range(n):
        # initialize (0,0)
        if i == 0:
            p0[1, 2] = dist(input_audio[0], template[0])
            continue
        # filter out the nodes we don't go through
        for j in range(max(0, m - 2 * (n - i - 1) - 1), min(m, 2 * i + 1)):
            # transform to padded coordinates
            i0 = i + 1
            j0 = j + 2
            c = dist(input_audio[i], template[j])
            p0[i0, j0] = min(p0[i0 - 1, j0] + c, p0[i0 - 1, j0 - 1] + c, p0[i0 - 1, j0 - 2] + c)
    # feature_extraction.plot_spectrogram(p0[1:, 2:],name="p")
    return p0[-1, -1]


# print(dtw(feature_extraction.extract_feature("three2.wav"), feature_extraction.extract_feature("three.wav")))


# templates is a list of templates
# output cost of input vs all templates
def dtw_synchronous(input_audio, templates):
    # initialize the padded matrices
    n = len(input_audio)
    p = [np.full((n + 1, len(template) + 2), np.inf) for template in templates]
    jobs = len(templates)
    for i in range(n):
        # initialize (0,0)
        if i == 0:
            for k in range(jobs):
                p[k][1, 2] = dist(input_audio[0], templates[k][0])
            continue
        # filter out the nodes we don't go through
        for k in range(jobs):
            m = len(templates[k])
            for j in range(max(0, m - 2 * (n - i - 1) - 1), min(m, 2 * i + 1)):
                # transform to padded coordinates
                i0 = i + 1
                j0 = j + 2
                c = dist(input_audio[i], templates[k][j])
                p[k][i0, j0] = min(p[k][i0 - 1, j0] + c, p[k][i0 - 1, j0 - 1] + c, p[k][i0 - 1, j0 - 2] + c)

    return [path[-1][-1] for path in p]


# print(dtw_synchronous(feature_extraction.extract_feature("three2.wav"), [feature_extraction.extract_feature("three.wav"), feature_extraction.extract_feature("two.wav")]))

# templates is a list of templates
# output cost of input vs all templates, if template rejected half way, distance would be inf

def dtw_synchronous_pruning(input_audio, templates, beam_width):
    # initialize the padded matrices
    n = len(input_audio)
    p = [np.full((n + 1, len(template) + 2), np.inf) for template in templates]
    jobs = len(templates)
    # initialize the set of points to explore
    b = [set() for template in templates]
    for i in range(n):
        # priority queue for extracting min
        pq = PriorityQueue()
        # initialize (0,0)
        if i == 0:
            for k in range(jobs):
                p[k][1, 2] = dist(input_audio[0], templates[k][0])
                pq.put(p[k][1, 2])
                # add to the beam
                b[k].add(0)

            threshold = pq.get() + beam_width

            b = update_pruning(p, 0, n, b, threshold)

            continue
        for k in range(jobs):
            m = len(templates[k])
            for j in b[k]:
                # transform to padded coordinates
                i0 = i + 1
                j0 = j + 2
                c = dist(input_audio[i], templates[k][j])
                p[k][i0, j0] = min(p[k][i0 - 1, j0] + c, p[k][i0 - 1, j0 - 1] + c, p[k][i0 - 1, j0 - 2] + c)
                pq.put(p[k][i0, j0])

            # Update pruning unless it is the last frame
            if i < n - 1:

                threshold = pq.get() + beam_width

                b = update_pruning(p, i, n, b, threshold)
    # feature_extraction.plot_spectrogram(p[0][1:, 2:], name="beam")
    return [path[-1][-1] for path in p]


# pruning after having all the costs at the ith frame
def update_pruning(p, i, n, b, threshold):
    jobs = len(b)
    i0 = i + 1
    new_b = [set() for k in range(jobs)]
    for k in range(jobs):
        if len(b[k]) == 0:
            continue
        for item in b[k]:
            j0 = item + 2
            if p[k][i0, j0] > threshold:
                continue
            for j in range(item, item + 3):
                if in_trellis(j, i + 1, len(p[k][0]) - 2, n):
                    new_b[k].add(j)

    return new_b


# see if item is a valid j for frame i
def in_trellis(item, i, m, n):
    return max(0, m - 2 * (n - i - 1) - 1) <= item < min(m, 2 * i + 1)

# print(dtw_synchronous_pruning(feature_extraction.extract_feature("three2.wav"), [feature_extraction.extract_feature("three.wav"), feature_extraction.extract_feature("two.wav")], 10))

