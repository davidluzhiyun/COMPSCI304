import numpy as np
from scipy.spatial.distance import cdist

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# The basic dtw routine
# Both input and template have shape (frame count, feature count)
# Uses Euclidean distance, allows Horizontal and Diagonal transition up to 1/2 shrinking
def dtw(input, template):
    # get the length of the input and template
    n, m = len(input), len(template)

    # Initialize p matrix
    # The padded version
    p0 = np.full((n+1, m+2), np.inf)
    # The unpadded p1 = p0[1:,2:]
    for i in range(0, n):
        # initialize (0,0)
        if i == 0:
            p0[1,2] = 0
            continue
        # filter out the nodes we don't go through
        for j in range(max(0, m - 2*(n-i-1)+1), min(m, 2*i+1)):
            # transform to padded coordinates
            i0 = i + 1
            j0 = j + 2
            c = euclidean_distance(input[i], template[j])
            p0[i0, j0] = min(p0[i0 - 1, j0] + c, p0[i0 - 1, j0 - 1] + c, p0[i0, j0] + c, p0[i0-1,j0 - 2] + c)

    return p0[-1,-1]