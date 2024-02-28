import numpy as np
# This implementation modifies segmental-kmeans with single gaussians
# because I don't have a BW algorithm from project 3 and the hmm with multiple gaussian has bugs and performance issues

def connect_hmms(hmms):
    # hmms is a list of tuples
    # (node_cost_functions, state_transition_scores, entrance_scores, exit_scores)
    # but we are ignoring entrance_scores and set it as (0,inf,inf....) to force entrance at initial state
    # the viterbi alignment will keep it as such if it is initialized as such
    # exit_scores only actually have two values in it, I will use it to manage transition between hmms

    # calculate total number of states
    total_length = 0

    for hmm in hmms:
        total_length += len(hmm[0])

    # create combined_state_transition_scores
    combined_state_transition_scores = np.full((total_length,), np.inf)
    combined_state_transition_scores[0] = 0

    # create template for transition matrix
    combined_state_transition_scores = np.full((total_length, total_length), np.inf)

    counter = 0
    for hmm in hmms:
        # copy transition matrix
        combined_state_transition_scores[counter: counter + len(hmm[0]),counter: counter + len(hmm[0])] = hmm[1]
        # incorporate the exit scores for word transition
        combined_state_transition_scores[counter + len(hmm[0])-1, counter + len(hmm[0])] =


