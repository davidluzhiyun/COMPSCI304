import math


class Node:
    def __init__(self, char):
        self.id = None
        self.char = char
        # Children is a dictionary using a tuple of (char, is_end_of_word) to find nodes
        self.children = dict()
        self.parent = None
        self.is_end_of_word = False


class LexicalTree:
    def __init__(self):
        self.root = Node('*')
        # keep a list of its elements in the traversal order so that we don't need to do dfs later
        self.traversal_list = [self.root]
        # each node remembers their position
        self.root.id = 0
        self.leaf_list = []

    def add_word(self, word):
        current_node = self.root
        # for none ending characters
        for char in word[:-1]:
            # add node if it doesn't exist as non-ending
            if (char, False) not in current_node.children:
                new_node = Node(char)
                current_node.children[(char, False)] = new_node
                # add parent
                new_node.parent = current_node
                # add new_node to the list
                new_node.id = len(self.traversal_list)
                self.traversal_list.append(new_node)
            # change current_node to the non-ending char you created or found
            # Never goes to ending char here
            current_node = current_node.children[(char, False)]
        # add ending node
        if (word[-1], True) not in current_node.children:
            new_node = Node(word[-1])
            current_node.children[(word[-1], True)] = new_node
            new_node.is_end_of_word = True
            # add parent
            new_node.parent = current_node
            # add new_node to the list
            new_node.id = len(self.traversal_list)
            self.traversal_list.append(new_node)
            # add also to the leaf_list to assist building of loopy list
            self.leaf_list.append(new_node)

    def construct_lexical_tree(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                word = line.strip()
                self.add_word(word)


    def get_children_indices(self, node):
        return [child.id for child in node.children.values()]

class TreeSearcher:
    def __init__(self, tree, beamWidth=3, trans_penalty=0):
        assert isinstance(tree, LexicalTree)
        self.tree = tree
        self.beamWidth = beamWidth
        # current indices of nodes to work on
        # initialized with all nodes (simulating after comparing '*' with root)
        self.workList = [i for i in range(len(tree.traversal_list))]
        # simulating after comparing '*' with root
        self.resultDict = dict()
        self.resultDict[self.tree.root] = 0
        self.trans_penalty = trans_penalty
        self.backtrack_list = []

    # auxiliary function for handling lookup of previous path cost if the node isn't in the resultDict
    def retrieve_path_cost(self, node, my_result):
        if node is None:
            return math.inf
        if node not in my_result:
            return math.inf
        else:
            return my_result[node]

    # auxiliary function that gets the cost of a char comparing to a node in template
    def get_node_cost(self, node, char):
        if node.char == char:
            return 0
        else:
            return 1

    # deal with root in worklist
    def search_cost_root(self):
        non_transiton_cost = self.retrieve_path_cost(self.tree.root, self.resultDict) + 1
        # transition costs
        leaf_list = [node for node in self.resultDict if node.is_end_of_word]
        cost_list = [self.trans_penalty + self.retrieve_path_cost(node, self.resultDict) for node in leaf_list]
        cost_list.append(non_transiton_cost)
        cost = min(cost_list)
        item = cost_list.index(cost)
        # append backtrack list if indeed transition
        if item < (len(cost_list) - 1):
            self.backtrack_list.append(self.tree.leaf_list[item])
        return cost

    # Update the worklist, add all ids of nodes in resultDict that aren't prune away and their children
    def update_workList(self):
        # initialize new set
        target_set = set()
        # calculate threshold
        threshold = min(self.resultDict.values()) + self.beamWidth
        # add the indices while pruning
        for node in self.resultDict.keys():
            if self.resultDict[node] <= threshold:
                target_set.update(self.tree.get_children_indices(node))
                target_set.add(node.id)
            # if contains leaf, add in the root
                if node.is_end_of_word:
                    target_set.add(0)
        # convert set to sorted list and update
        self.workList = sorted(target_set)


    # Update resultDict based on a new char
    def search_char(self, char):
        # have a new resultDict to track what is calculated this round
        new_resultDict = dict()
        for index in self.workList:
            # update costs if not root
            if index != 0:
                current_node = self.tree.traversal_list[index]
                cost = min(self.get_node_cost(current_node, char)+self.retrieve_path_cost(current_node.parent, self.resultDict),
                           self.retrieve_path_cost(current_node.parent, new_resultDict) + 1,
                           self.retrieve_path_cost(current_node, self.resultDict) + 1)
                new_resultDict[current_node] = cost
            if index == 0:
                cost = self.search_cost_root()
                new_resultDict[self.tree.root] = cost
        # update resultDict
        self.resultDict = new_resultDict

    def line_lookup(self, line):
        # Search
        # no need to prepend since we already initialized with *
        for char in line:
            self.search_char(char)
            self.update_workList()
        # Looking up the node for backtracking
        min_cost = math.inf
        min_node = None
        for node in self.resultDict.keys():
            if node.is_end_of_word and self.resultDict[node] <= min_cost:
                min_cost = self.resultDict[node]
                min_node = node
        # back tracking
        if min_node is None:
            return None
        current_node = min_node
        found_string = ''
        last_word = len(self.backtrack_list) - 1
        while last_word >= 0:
            while current_node != self.tree.root:
                found_string += current_node.char
                current_node = current_node.parent
            found_string += ' '
            current_node = self.backtrack_list[last_word]
            last_word -= 1
        # reset
        # initialized with all nodes (simulating after comparing '*' with root)
        self.workList = [i for i in range(len(self.tree.traversal_list))]
        # simulating after comparing '*' with root
        self.resultDict = dict()
        self.resultDict[self.tree.root] = 0
        self.backtrack_list = []
        return found_string[::-1]





