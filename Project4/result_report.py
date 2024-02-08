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

# one upon a tyre wise yamagata we ing of benares to shasta name to lit o the foot of he hillas as a monkey he grew strong and sturdy big of fraim well to do and live by a werne of to river ganges in a forest haunt now at that two there was a crocodile peeling in to glues the rooney's mate saw the grete frame of the monkey and she conceived a longing to eye us rate so she set to her lord sir i desire to set the heart of the rate king of the monkeys
#
# wood wife side the crudele i leed in the vaske and see live on dry land hud yan we rach him
#
# my hud or by true she yield he met be not if i don't get herm i shall die
#
# all rate used to crudele consoling or don't frable yield i has a plan i wit give you his haro to set
#
# so why to bodhisatta was sitting on to bank of to glues for taken a dunk of witz the nobody too near and said sir monkey why do your lit on bad roots in this old family plots on the other side of the ganges there is no end to the mango trees and labuja trees wit fruit sweet as only is it not bear to krus over ante has all kinds of wild fruit to rate
#
# lore rodin to junker inset the ganges is deep and wrye how yell i it across
#
# if you want to goh i will let nu sit upon my bank and vary you over
#
# the monkey truth he ant aired come were the said to crocodile up on my back with you and up to monkey lomba but why the crocodile had swum a yell wrye he pfund the monkey uber the water
#
# good fend you or living me sink cried the monkey wit is that or
#
# to brumley said you think i am trying your out of pure good ochre not a bit of it my wife has a longing for your heart and i wanta to eve it to or to rate
#
# reine said the monkey it is nice of you to zeh me why if our haro were inside us when we go jumping among the true tops it wild be all cocked to pieces
#
# all were do you keep it ask the crocodile's
#
# the tunicate pointed out a x. true with glasses of up krout standing not far off sit said he there are our harms hanging on your fine true
#
# if you will show me your heart said the crocodile then i won't kill you
#
# take see to the trees when and i all point it out to your
#
# the crocodile brought him to the place the monkey leapt off his back and climbing up the nigg tree sat upon it oh silly crocodile with he you ought that their were creatures that kept their garst in a treetop you are a hoole and i has putted you you may kea your krout to yourself more body is great but you has no sesno
#
# and then to explain the idea he uttered the followed stanzas
#
# roseapple jackfruit nagele tops across the witz their i see
# enough of the i wit the not my nigg is good enough for me
# greet is yuck body very buts how much smaller is you witz
# now go your ways sir crocodile for i he hud to best of its
# the crocodile feeling as said and miserable as if he had lost a hofland pieces of money wit back sorrowing to the place when he live