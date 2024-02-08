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

    def construct_lexical_tree(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                word = line.strip()
                self.add_word(word)

    def get_children_indices(self, node):
        return [child.id for child in node.children.values()]

# # Test for the tree
# tree = LexicalTree()
# tree.add_word("a")
# tree.add_word("apple")
# tree.add_word("app")
# for node in tree.traversal_list:
#     print(node.char)
#     print(node.id)
#     if node.parent is not None:
#         print(node.parent.char)
#     else:
#         print()
#     print(node.is_end_of_word)
#     print(node.children)
#     print()
# print(tree.traversal_list[0].children[('a', False)].char)
# print()
# print(tree.get_children_indices(tree.root))
# print(tree.get_children_indices(tree.root.children[('a', True)]))


class TreeSearcher:
    def __init__(self, tree, beamWidth=3):
        assert isinstance(tree, LexicalTree)
        self.tree = tree
        self.beamWidth = beamWidth
        # current indices of nodes to work on
        # initialized with all nodes (simulating after comparing '*' with root)
        self.workList = [i for i in range(len(tree.traversal_list))]
        # simulating after comparing '*' with root
        self.resultDict = dict()
        self.resultDict[self.tree.root] = 0

    # auxiliary function for handling lookup of previous path cost if the node isn't in the resultDict
    def retrieve_path_cost(self, node):
        if node is None:
            return math.inf
        if node not in self.resultDict:
            return math.inf
        else:
            return self.resultDict[node]

    # auxiliary function that gets the cost of a char comparing to a node in template
    def get_node_cost(self, node, char):
        if node.char == char:
            return 0
        else:
            return 1
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
        # convert set to sorted list and update
        self.workList = sorted(target_set)


    # Update resultDict based on a new char
    def search_char(self, char):
        new_resultDict = dict()
        for index in self.workList:
            # update costs
            current_node = self.tree.traversal_list[index]
            cost = min(self.get_node_cost(current_node, char)+self.retrieve_path_cost(current_node.parent),
                       self.retrieve_path_cost(current_node.parent) + 1,
                       self.retrieve_path_cost(current_node) + 1)
            new_resultDict[current_node] = cost
        # update resultDict
        self.resultDict = new_resultDict

    def word_lookup(self, word):
        # Search
        # no need to prepend since we already initialized with *
        for char in word:
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
        found_word = ''
        while current_node != self.tree.root:
            found_word += current_node.char
            current_node = current_node.parent
        # reset
        # initialized with all nodes (simulating after comparing '*' with root)
        self.workList = [i for i in range(len(self.tree.traversal_list))]
        # simulating after comparing '*' with root
        self.resultDict = dict()
        self.resultDict[self.tree.root] = 0
        return found_word[::-1]

# my_tree = LexicalTree()
# my_tree.construct_lexical_tree('dict_1.txt')
# my_searcher = TreeSearcher(my_tree)
# print(my_searcher.word_lookup('ccological'))
# print(my_searcher.word_lookup('ecologica'))




