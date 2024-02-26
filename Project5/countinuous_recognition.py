import math
import numpy as np


# this version implements the best hypothesis version of
class NonEmissionNode:
    def __init__(self):
        self.bp = []
        # candidate for bp entry, initialize to None each round. Track record with smallest cost
        # if it is none, indicates that current bp entry has expired
        self.candidate = None
        # edges that goes out, include hmm and non hmm ones
        self.out_edges = []
        # if active, update for all out_edges
        self.active = False
        # show if the latest entry of the bp is valid
        self.bp_valid = False
        return

    # update bp, reset candidate and test validity after receiving data
    def update_bp(self):
        # add entry to bp if received valid update
        if self.candidate is not None:
            self.bp.append(self.candidate)
            # reset candidate
            self.candidate = None
            self.bp_valid = True
        # indicate if no valid update recieved
        else:
            self.bp_valid = False
        return


    def get_current_score(self):
        # return the latest score if bp valid, else return inf
        if not self.bp_valid:
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

class NonHMMEdge:
    def __init__(self, begin, end, penalty):
        self.initialized = False
        self.begin = begin
        self.begin.out_edges.append(self)
        self.end = end
        # the penalty score of the edge
        self.penalty = penalty

    def initialize_scores(self, input_sequence, t):
        self.initialized = True
        # send scored even in initialization
        self.update_scores(input_sequence,t)

    def update_scores(self, input_sequence, t):
        # when the current bp entry from parent is valid, penalize and send the bp
        if self.begin.candidate is not None:
            self.end.recieve_data(self.begin.bp[0],self.begin.bp[1],self.begin.bp[2] + self.penalty, self.begin.bp[3], self.begin.bp[4])
        # else just send something that will be fitered out
        else:
            self.end.recieve_data(None, None, math.inf, None, None)


class HMMEdge:
    def __init__(self, node_cost_functions, state_transition_scores, entrance_scores, exit_scores, begin, end, word):
        assert isinstance(begin, NonEmissionNode)
        assert isinstance(end, NonEmissionNode)
        # initialized shows if the edge is activated
        self.initialized = False
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
        self.initialized = True

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


def recognize(nodes, input_features):
    # nodes is a list of nodes in the model
    # input_features is the extracted result of
    # if contain non hmm edge, need to adjust the order of nodes to ensure correct activation
    # assume last node in nodes is end node
    for t in range(len(input_features)):
        for node in nodes:
            assert isinstance(node, NonEmissionNode)
            if node.active:
                # first update the bp table
                node.update_bp()
                # update the edges
                for edge in node.out_edges:
                    assert (isinstance(edge, NonHMMEdge) or isinstance(edge,HMMEdge))
                    if not edge.initialized:
                        edge.initialize_scores(input_features, t)
                    else:
                        edge.update_scores(input_features, t)
    # back tracking
    # start with final entry of the bp of the last node
    current_entry = nodes[-1].bp[-1]
    cost = current_entry[2]
    backtrack = ''
    # end when encounter the root bp
    while current_entry[1] is not None:
        # append the word
        backtrack += current_entry[1]
        # update current entry
        current_entry = current_entry[4].bp[current_entry[3]]
    return cost, ''.join(reversed(backtrack))



