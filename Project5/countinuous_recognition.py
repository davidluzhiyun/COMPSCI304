import math
import numpy as np


# this version implements the best hypothesis version of
class NonEmissionNode:
    def __init__(self):
        self.bp = []
        # candidate for dp entry, initialize to None each round. Track record with smallest cost
        # if it is none, indicates that current bp entry has expired
        self.candidate = None
        # edges that goes out, include hmm and non hmm ones
        self.out_edges = []
        # if active, update for all out_edges
        self.active = False
        return

    # update_bp after receiving data
    def update_bp(self):
        # add entry to bp if received valid update
        if self.candidate is not None:
            self.bp.append(self.candidate)
        return

    # reset candidate after using data
    def reset_candidate(self):
        self.candidate = None
        return

    def get_current_score(self):
        # return the latest score if bp valid, else return inf
        if self.candidate is None:
            return math.inf
        else:
            return self.bp[-1][2]

    def get_current_backtrack(self):
        # return the index of the latest entry in bp table
        # since this is only called when the latest entry is valid, no need to make distinction
        return len(self.bp) - 1

    def recieve_data(self, t, word, cost, parent, previous_node):
        if cost != math.inf:
            # activated node when reached
            self.active = True
            # update when record has cost smaller than candidate
            if self.candidate is None or self.candidate[2] > cost:
                self.candidate = (t, word, cost, parent, previous_node)
        return


class HMMEdge:
    def __init__(self, node_cost_functions, state_transition_scores, entrance_scores, exit_scores, begin, end, word):
        assert isinstance(begin, NonEmissionNode)
        assert isinstance(end, NonEmissionNode)
        # the hmm parameters
        self.node_cost_functions = node_cost_functions
        self.state_transition_scores = state_transition_scores
        self.entrance_scores = entrance_scores
        self.exit_scores = exit_scores
        # number of nodes in the hmm
        # need to be at least 2
        self.num_nodes = len(node_cost_functions)
        # the viterbi matrix
        self.costs = [math.inf for i in range(self.num_nodes)]
        # store the parent in the backpointer table in self.begin
        self.backtrack = [0 for i in range(self.num_nodes)]
        # the beginning and ending non-emitting nodes
        self.begin = begin
        # record out_edge
        self.begin.out_edges.append(self)
        self.end = end
        # the word
        self.word = word

    # initialize the scores of the viterbi matrix
    # t = 0 during first comparison
    def initialize_scores(self, input_sequence, t):
        # Initialize the scores the Viterbi matrix
        # consists of entrance scores of hmm(typically infinity), the node alignment score and the score before entrance(from the bp table)
        # this implementation considers entry score
        for state in range(self.num_nodes):
            self.costs[state] = self.entrance_scores[state] + self.node_cost_functions[state](
                input_sequence[t]) + self.begin.get_current_score()
            self.backtrack[state] = self.begin.get_current_backtrack()

    # update the scores and send score to the ending non-emission node if needed
    def update_scores(self, input_sequence, t):
        new_costs = [math.inf for i in range(self.num_nodes)]
        new_backtracks = [0 for i in range(self.num_nodes)]
        for i in range(self.num_nodes):
            # c is the node cost
            c = self.node_cost_functions[i](input_sequence[t])
            # Allows skipping 1 state including to state transition
            # Special cases for i = 0 and 1, incorporate info from beginning node(bp table)
            if i == 0:
                # despite having an entrance cost, this is the only state that allowing transition into from non-emission during updates
                # p0 is the cost without being transitioned into
                p0 = self.costs[i] + self.state_transition_scores[i, i] + c
                # p1 is the score from word transition
                p1 = self.begin.get_current_score() + self.entrance_scores[i] + c
                if p0 <= p1:
                    new_backtrack = self.backtrack[i]
                    p = p0
                else:
                    new_backtrack = self.begin.get_current_backtrack()
                    p = p1
            elif i == 1:
                potential_prev = np.array([self.costs[i] + self.state_transition_scores[i, i],
                                           self.costs[i - 1] + self.state_transition_scores[i - 1, i]])
                prev_path = np.argmin(potential_prev)
                new_backtrack = self.backtrack[i - prev_path]
                p = potential_prev[prev_path] + c

            else:
                potential_prev = np.array([self.costs[i] + self.state_transition_scores[i, i],
                                           self.costs[i - 1] + self.state_transition_scores[i - 1, i],
                                           self.costs[i - 2] + self.state_transition_scores[i - 2, i]])
                prev_path = np.argmin(potential_prev)
                new_backtrack = self.backtrack[i - prev_path]
                p = potential_prev[prev_path] + c
            new_costs[i] = p
            new_backtracks[i] = new_backtrack
        self.costs = new_costs
        self.backtrack = new_backtracks
        # send the score of the last two states to the non-emission state
        self.end.recieve_data(t, self.word, self.costs[-1] + self.exit_scores[-1], self.backtrack[-1], self.begin)
        self.end.recieve_data(t, self.word, self.costs[-2] + self.exit_scores[-2], self.backtrack[-2], self.begin)
        return