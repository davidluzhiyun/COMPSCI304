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

    # create combined_entrance_scores
    combined_entrance_scores = np.full((total_length,), np.inf)
    combined_entrance_scores[0] = 0

    # create combined_exit_scores, which is just the exit scores of the final state
    combined_exit_scores = np.full((total_length,), np.inf)
    combined_exit_scores[-1] = hmms[-1][3][-1]
    combined_exit_scores[-2] = hmms[-1][3][-2]

    # allocate space for combined transition matrix
    combined_state_transition_scores = np.full((total_length, total_length), np.inf)

    counter = 0
    combined_node_cost_functions = []
    for hmm in hmms:
        # add to combined_node_cost_functions
        combined_node_cost_functions += hmm[0]

        # copy transition matrix
        combined_state_transition_scores[counter: counter + len(hmm[0]),counter: counter + len(hmm[0])] = hmm[1]

        # incorporate the exit scores for word transition
        # don't do this for final word
        if counter + len(hmm[0]) < total_length:
            combined_state_transition_scores[counter + len(hmm[0]) - 1, counter + len(hmm[0])] = hmm[3][-1]
            combined_state_transition_scores[counter + len(hmm[0]) - 2, counter + len(hmm[0])] = hmm[3][-2]
        counter += len(hmm[0])
    return combined_node_cost_functions, combined_state_transition_scores, combined_entrance_scores,combined_exit_scores

def disconnect_hmms(combined_hmm, state_counts):
    hmms = []
    counter = 0
    for i in range(len(state_counts)):
        assert state_counts[i] >= 2
        node_cost_functions = combined_hmm[0][counter: counter + state_counts[i]]

        state_transition_scores = combined_hmm[1][counter: counter + state_counts[i], counter: counter + state_counts[i]]

        # ignoring entrance_scores and set it as (0,inf,inf....)
        entrance_scores = np.full((state_counts[i],), np.inf)
        entrance_scores[0] = 0

        exit_scores = np.full((state_counts[i],), np.inf)
        if i == len(state_counts) - 1:
            exit_scores[-1] = combined_hmm[3][-1]
            exit_scores[-2] = combined_hmm[3][-2]
        else:
            exit_scores[-1] = combined_hmm[1][counter + state_counts[i] - 1, counter + state_counts[i]]
            exit_scores[-2] = combined_hmm[1][counter + state_counts[i] - 2, counter + state_counts[i]]

        hmms.append((node_cost_functions, state_transition_scores, entrance_scores, exit_scores))

    return hmms




