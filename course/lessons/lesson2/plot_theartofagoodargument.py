"""
The art of a good argument
==========================

In the book, I describe the way Charlie and Aisha interact in terms of a model. 
We determine whether Charlie will start shouting is using 
the following table.

.. image:: ../../images/lesson2/GoodArgument.png

On this page we implement the model in Python. 

Setting up rules
----------------

We start by setting up these rules.

"""

import numpy as np
import pandas as pd

Charlie_shout =np.array([[0.1, 0.5],[0.7,0.95]])
Charlie_rules = pd.DataFrame(data= Charlie_shout,index=['Aisha calm', 'Aisha shouting'], columns=['Charlie calm', 'Charlie shoting'])
print('Probability of Charlie shouting:')
print(Charlie_rules)

Aisha_shout =np.array([[0.1, 0.7],[0.5,0.95]])
Aisha_rules = pd.DataFrame(data= Aisha_shout,index=['Aisha calm', 'Aisha shouting'], columns=['Charlie calm', 'Charlie shoting'])
print('Probability of Aisha shouting:')
print(Aisha_rules)

# Initially, neither of them are shouting

Charlie=[0]
Aisha=[0]

# Number of time steps
T = 25

###################################################################
# Simulating arguments
# --------------------
#
# We now set up a computer simulation and look at how the two interact.
# To do this, we loop over 25 steps of discussion according to the rules above. 
#


def argument(T,Charlie_rules,Aisha_rules,print_out=1):
    for i in range(T):

        Prob_Charlie = Charlie_rules.iloc[Charlie[i]][Aisha[i]]
        if np.random.rand()<Prob_Charlie:
            describe_Charlie = 'Charlie shouts. '
            Charlie.append(1)
        else:
            describe_Charlie = "Charlie doesn't shout. "
            Charlie.append(0)
                    
        Prob_Aisha = Aisha_rules.iloc[Charlie[i]][Aisha[i]]
        if np.random.rand()<Prob_Aisha:
            describe_Aisha = 'Aisha shouts.'
            Aisha.append(1)
        else:
            describe_Aisha = "Aisha doesn't shout."
            Aisha.append(0)
            
        if print_out:
            print('Time step %d:' % (i+1) + describe_Charlie +describe_Aisha)
    
    return Aisha,Charlie

Aisha,Charlie = argument(T,Charlie_rules,Aisha_rules)
        
###################################################################
#
# We can represent the argument as a binary string, as we do in the book.
#

print("Charlie's shouting as a string of zeros (clam) and ones (shouting):")
print(' '.join(map(str, Charlie)))
print("Aisha's shouting as a string of zeros (clam) and ones (shouting):")
print(' '.join(map(str, Aisha)))


###################################################################
#
# Now we can make them argue lots of times!
#

for j in range(5):
    
    # Both start calm.
    Charlie=[0]
    Aisha=[0]

    # Call the argument function
    print('Argument %d' % (j+1))
    print('----------')
    Aisha,Charlie = argument(T,Charlie_rules,Aisha_rules,0)
    print("Charlie's shouting as a string of zeros (clam) and ones (shouting):")
    print(' '.join(map(str, Charlie)))
    print("Aisha's shouting as a string of zeros (clam) and ones (shouting):")
    print(' '.join(map(str, Aisha)))
    print('\n')




##############################################################################
#
# Charlie makes a change
# ----------------------
#
# Charlie decides to update his rules
#
# .. image:: ../../images/lesson2/CharlieUpdate.png
#

Charlie_shout =np.array([[0.1, 0.1],[0.1,0.95]])
Charlie_rules = pd.DataFrame(data= Charlie_shout,index=['Aisha calm', 'Aisha shouting'], columns=['Charlie calm', 'Charlie shoting'])
print('Probability of Charlie shouting:')
print(Charlie_rules)


###################################################################
#
# Now let's look at five arguments under his new rules.
#


for j in range(5):
    
    Charlie=[0]
    Aisha=[0]
    print('Argument %d' % (j+1))
    print('----------')
    Aisha,Charlie = argument(T,Charlie_rules,Aisha_rules,0)
    print("Charlie's shouting as a string of zeros (clam) and ones (shouting):")
    print(' '.join(map(str, Charlie)))
    print("Aisha's shouting as a string of zeros (clam) and ones (shouting):")
    print(' '.join(map(str, Aisha)))
    print('\n')


###################################################################
#
# Now Charlie shouts less and (as a result) so to does Aisha.
# 
# .. admonition:: Try yourself!
#   
#       Download the code and run it yourself in a Python notebook. Adjust the probabilities
#       so that Aisha adopts the same rules as Charlie. Notice how much less they argue with
#       one another!


##############################################################################
#
# Integrative behavioural couple therapy
# --------------------------------------
#
# An introduction to integrative behavioural couple therapy (IBCT) can be watched below:
#
#  ..  youtube:: 8jidWO_Q8Do
#   :width: 640
#   :height: 349
#
# The model described in this section is inspired by this approach.
#
# To learn more about this approach see:
# 
# `Andrew Christensen and Brian D. Doss, Integrative behavioral couple therapy, 
# Current Opinion in Psychology 13 (2017): 111‒14. <https://www.sciencedirect.com/science/article/pii/S2352250X1630046X?casa_token=aXp8NydeJ_UAAAAA:JEIj5f7kLOoqJqDy-fd6MG73WMNOi8L9_Vel8AtIQ1ZUEYQ7_lVl5GjEpKoADYsFl8KS3628>`_