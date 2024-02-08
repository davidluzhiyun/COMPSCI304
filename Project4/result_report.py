import tree

my_tree = tree.LexicalTree()
my_tree.construct_lexical_tree('dict_1.txt')
my_searcher = tree.TreeSearcher(my_tree)
with open('typos.txt', 'r') as file:
    for line in file:
        # Split the line into words
        words = line.split()
        processed_line = []
        # Process each word
        for word in words:
            processed_word = my_searcher.word_lookup(word)
            processed_line.append(processed_word)
        # Join the processed words to recreate the original line
        processed_line = ' '.join(processed_line)
        print(processed_line)
# problem 1 result

# one upon a tyre wise brahmadatta we king of benares to bodhisatta name to lit to the foot of he himalayas as a monkey he grew strong and sturdy big of fraim well to do and lived by a werne of to river ganges in a forest haunt now at that tyre there was a crocodile peeling in to ganges the crocodile's mate saw the grete frame of the monkey and she conceived a longing to eye us harter so she set to her lord sir i desire to set the heart of the rate king of the monkeys
#
# wood wife side the crudele i leed in the vaske and see live on dry land hud yan we rach him
#
# my hud or by true she yield he must be not if i don't get herm i shall die
#
# all rate answered to crudele consoling or don't frable yourself i have a plan i wit give you his heart to set
#
# so why to bodhisatta was sitting on to bank of to ganges after taking a dunk of water the crocodile too near and seide sir monkey why do your live on bad roots in this oldie family plots on the other side of the ganges there is no end to the mangoes trees and labuja trees wit fruit sweet as only is it not bear to krus over ante have allen kinds of wildes fruit to rate
#
# lore crocodile to junker inset the ganges is deep and wyden how shell i it across
#
# if you want to goh i will let you sit upon my bank and vary you over
#
# the monkey trusted him ant aired come were then seide to crocodile up on myer back with you and up to monkey lomba but why the crocodile had swum a lytton wrye he plunged the monkey under the water
#
# good friend you or letting me sink cried the monkey wit is that or
#
# to brumley said you think i am carrying your out of pure good ochre not a bit of it my wife has a longing for your heart and i wanta to give it to or to rate
#
# reine said the monkey it is nice of you to tell me why if our heart were inside us when we go jumping among the true tops it would be all cocked to pieces
#
# will where do you keep it asked the crocodile's
#
# the bodhisatta pointed out a fig true with glasses of up krout standing not far off sit said he there are our hearts hanging on yonder fine true
#
# if you will show me your heart said the crocodile then i won't kill you
#
# take see to the trees when and i will point it out to your
#
# the crocodile brought him to the place the monkey leapt off his back and climbing up the nigg tree sat upon it oh silly crocodile with he you thought that their were creatures that kept their garst in a treetop you are a hoole and i have outwitted you you may kept your krout to yourself more body is great but you have no sesno
#
# and then to explain this idea he uttered the following stanzas
#
# roseapple jackfruit nagele tops across the water their i see
# enough of them i wit them not my nigg is good enough for me
# greet is yukon body very buts how much smaller is you witz
# now go your ways sir crocodile for i he hud to best of its
# the crocodile feeling as said and miserable as if he had lost a thousand pieces of money wit back sorrowing to the please where he lived

