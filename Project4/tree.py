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

# Test for the tree
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
# print(tree.traversal_list[0].children[('a',False)].char)

class TreeSearcher:
    def __init__(self, tree, beamWidth=3):
        assert isinstance(tree, LexicalTree)
        self.tree = tree
        self.beamWidth = beamWidth
        self.workList = [tree.root]
        self.resultDict = dict()

    # auxiliary function for handling lookup of previous path cost if the node isn't in the resultDict
    def retrieve_path_cost(self, node):
        if node is None:
            return math.inf
        if node not in self.resultDict:
            return  math.inf
        else:
            return self.resultDict[node]

    # auxiliary function that gets the cost of a char comparing to a node in template
    def get_node_cost(node, char):
        if node.char == char:
            return 0
        else:
            return 1

    # def search_char(self, char):
    #     threshold = math.inf
    #     new_resultDict = dict()
    #     traverse_stack = st
    #     for node in self.workList:
