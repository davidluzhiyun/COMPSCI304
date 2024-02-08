import math
import numpy as np
import matplotlib.pyplot as plt

import tree2
import tree
my_tree = tree2.LexicalTree()
my_tree.construct_lexical_tree('dict_1.txt')
my_searcher = tree2.TreeSearcher(my_tree)
accuracy = []

# calculate missspell by aligning
def misspell_distance(split1, split2):
    m = np.full((len(split1)+1, len(split2)+1), np.inf)
    for i in range(len(split1)+1):
        m[i,0] = i
    for j in range(len(split2)+1):
        m[0, j] = j
    for i in range(1, len(split1) + 1):
        for j in range(1, len(split2) + 1):
            m[i, j] = min(m[i-1, j-1] + int(split1[i-1] != split2[j-1]),
                          m[i-1, j] + 1,
                          m[i, j-1] + 1)
    return m[-1,-1]

with open('unsegmented.txt', 'r') as file:
    fh = open('segmented.txt', 'r')
    original_file = fh.read()
    original_split = original_file.split()

    for width in [5, 10, 15]:
        my_searcher.beamWidth = width
        my_file = ''
        for line in file:
            processed_line = my_searcher.line_lookup(line)
            my_file += processed_line
        my_file_split = my_file.split()
        count_difference = abs(len(my_file_split) - len(original_file))
        # misspelled_words
        misspelled_words = misspell_distance(my_file_split, original_split)

        accuracy.append(count_difference + misspelled_words)

beam_width = [5, 10, 15]

# Plotting
plt.plot(beam_width, accuracy, marker='o', linestyle='-')
plt.title('Accuracy vs. Beam Width')
plt.xlabel('Beam Width')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(beam_width)  # Set x-axis ticks to the beam width values
plt.show()

