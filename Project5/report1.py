import feature_extraction
import countinuous_recognition
from hmm import segmental_k_means

DATA_PATH = 'data/'
DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def train_digit(digit):
    templates = [feature_extraction.extract_feature(DATA_PATH + digit + str(i) + '.wav') for i in range(10)]
    node_cost_functions, state_transition_scores, entrance_scores, exit_scores, segmentations = segmental_k_means(
        templates, 5)
    return node_cost_functions, state_transition_scores, entrance_scores, exit_scores


# the hmm parameters from the segmental kmeans for the digits
digit_parameters = [train_digit(digit) for digit in DIGITS]


# create the list of nodes
nodes = [countinuous_recognition.NonEmissionNode() for i in range(8)]

# Configure the root node
nodes[0].active = True
# add to the candidate so that the update does the work automatically
nodes[0].candidate = (None, None, 0, None, None)

# add first digit
print(range(2, 10))


