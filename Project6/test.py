import numpy as np

import feature_extraction
import hmm
import continuous_speech_training

DATA_PATH = 'data/'

templates = [feature_extraction.extract_feature(DATA_PATH+'seven'+str(i)+'.wav') for i in range(10)]
node_cost_functions, state_transition_scores, entrance_scores, exit_scores, labels = hmm.segmental_k_means(templates, 5)

print(node_cost_functions)
print(state_transition_scores)
print(entrance_scores)
print(exit_scores)

node_cost_functions, state_transition_scores, entrance_scores, exit_scores = continuous_speech_training.connect_hmms([(node_cost_functions, state_transition_scores, entrance_scores, exit_scores), (node_cost_functions, state_transition_scores, entrance_scores, exit_scores)])

print(node_cost_functions)
print(state_transition_scores)
print(entrance_scores)
print(exit_scores)

components = continuous_speech_training.disconnect_hmms((node_cost_functions, state_transition_scores, entrance_scores, exit_scores), [5, 5])

print(components[0][0])
print(components[0][1])
print(components[0][2])
print(components[0][3])
print(components[1][0])
print(components[1][1])
print(components[1][2])
print(components[1][3])

# res = np.full((5, ), np.inf)
# print(res.shape)
# print(labels)
# print(hmm.viterbi_alignment(templates[0], node_cost_functions, state_transition_scores, entrance_scores, exit_scores)[1])
# a = np.array([[1,1,1,1,1,1,1],
#               [1,4,4,4,1,1,1],
#               [1,4,4,4,1,1,1],
#               [1,4,4,4,1,1,1],
#               [1,4,4,4,1,1,1],
#               [1,4,4,4,1,1,1],
#               [1,1,1,1,1,1,1]])
#
# a_copy = a.copy()
#
# b = np.array([[8,5,5,8],
#               [8,5,5,8]])
#
# print("Array a:")
# print(a)
# print("\nCopy of array a:")
# print(a_copy)
# print("\nArray b:")
# print(b)
#
# a[3:5,2:6] = b
#
# print("Array a after being partially replaced with array b:")
# print(a)
#
# b[0,0] = 0
#
# print(a)