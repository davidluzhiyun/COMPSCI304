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

# the hmm parameters for silence
silence_node_cost_functions, silence_state_transition_scores, silence_entrance_scores, silence_exit_scores, silence_segmentations = segmental_k_means(
    [feature_extraction.extract_feature(DATA_PATH + 'silence.wav')], 2)

silence_parameters = silence_node_cost_functions, silence_state_transition_scores, silence_entrance_scores, silence_exit_scores

# create the list of nodes
nodes = [countinuous_recognition.NonEmissionNode() for i in range(8)]

# Configure the root node
nodes[0].active = True
# add to the candidate so that the update does the work automatically
nodes[0].candidate = (None, None, 0, None, None)

# add first digit
for i in range(2, 10):
    countinuous_recognition.HMMEdge(digit_parameters[i][0], digit_parameters[i][1], digit_parameters[i][2],
                                    digit_parameters[i][3], nodes[0], nodes[1], str(i))

# add the other digits
for n in range(1, 7):
    for i in range(10):
        countinuous_recognition.HMMEdge(digit_parameters[i][0], digit_parameters[i][1], digit_parameters[i][2],
                                        digit_parameters[i][3], nodes[n], nodes[n + 1], str(i))

# add silence 3,3
countinuous_recognition.HMMEdge(silence_parameters[0], silence_parameters[1], silence_parameters[2],
                                silence_parameters[3], nodes[3], nodes[3], '')
# add jump from 0 to 3
countinuous_recognition.NonHMMEdge(nodes[0], nodes[1], 0)

# Graph construction completed
cost, result = countinuous_recognition.recognize(nodes, feature_extraction.extract_feature('1234567.wav'))
print(cost)
print(result)
