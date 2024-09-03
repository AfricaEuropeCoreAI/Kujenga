"""
Cellular automata
=================

There are several 


In this section we run some of the cellular automata rules from the book using Python code. 
Download the code and run it yourself to investigate how these models work.


The domino rule
---------------

In Santa Fe, Chris describes the domino rule as follows.

.. image:: ../../images/lesson2/Domino.png

We start by writing code to apply the rule to a binary string.

"""

def apply_domino(string):

    # Convert the strong to cells 
    cells=list(map(int, [*string]))
    new_cells=cells.copy()
    N=len(cells)
    
    # Loop over every cell and update its state based on the neighboring cells
    for j in range(N):
        # Get the neighbors left, above and right.
        # The %N is to take mod, to deal with the edge cases
        left = cells[(j-1)%N]
        above = cells[j]
        right= cells[(j+1)%N]
        
        
        # If the cell to the left is a 1 then the cell becomes a 1
        # Otherwise it remains the same.
        if left==1:
            new_cells[j] = 1
        else:
            new_cells[j]=cells[j]
        
    # Convert back to a new string    
    string=''.join([str(item) for item in new_cells])
    
    return string



##############################################################################
# Let's now apply the rule


string='11101100001111000011111111101'
print('Before applying rule: ' + string)
string=apply_domino(string)
print('After applying rule : ' + string)


##############################################################################
# We can repeadtedly apply it as follows:

steps = 5
string='11101100001111000011111111101'

for i in range(steps):
    print('Time step %d:'%i + string)
    string=apply_domino(string)
    

# .. admonition:: Think yourself!
#   
#       Look at a starting string with all zeros, apart from one one. Run the 
#       cellular automata for enough steps so that all the zeros turn to one.
#       Notice how the domino rally travels from one side to the other.

##############################################################################
# The politics rule
# -----------------
#
# Chris goes on to describe a new set of the following three rules.
#
# .. image:: ../../images/lesson2/Politics.png
#
# Let's implement these in Python
#

def apply_political(string):

    # Convert the strong to cells 
    cells=list(map(int, [*string]))
    new_cells=cells.copy()
    N=len(cells)
    
    # Loop over every cell and update its state based on the neighboring cells
    for j in range(N):
        # Get the neighbors left, above and right.
        # The %N is to take mod, to deal with the edge cases
        left = cells[(j-1)%N]
        above = cells[j]
        right= cells[(j+1)%N]
        
        # If the cell to the left is a 1 then the cell becomes a 1
        # Otherwise it remains the same.
        if ((left==0) & (right==0)):
            new_cells[j] = 0
        elif ((left==1) & (right==1)):
            new_cells[j] = 1
        else:
            new_cells[j]=cells[j]
        
    # Convert back to a new string    
    string=''.join([str(item) for item in new_cells])
    
    return string

##############################################################################
# Let's now apply the rule

string='0100011011110101011010'
print('Before applying rule: ' + string)
string=apply_political(string)
print('After applying rule : ' + string)


##############################################################################
# We can repeadtedly apply the rule as follows:

steps = 5
string='0100011011110101011010'

for i in range(steps):
    print('Time step %d:'%i + string)
    string=apply_political(string)
    
# All the isolated 0's or 1's are replaced 
# by their majority neighbours (we assume that the bits form a loop, so 
# the 1 on the end of the string has neighbour 0 to its left and adopts its 
# right neighbour as the 0 at the start of the string and thus becomes a 0).



##############################################################################
# The alternate rule
# ------------------
#
# In the book, Esther and I find rules which create alternating lines like these.
#
# .. image:: ../../images/lesson2/TransitionAlternate.png
#
# Let's implement these rules in Python. 


def apply_alternate(string):

    # Convert the strong to cells 
    cells=list(map(int, [*string]))
    new_cells=cells.copy()
    N=len(cells)
    
    # Loop over every cell and update its state based on the neighboring cells
    for j in range(N):
        # Get the previous value

        above = cells[j]
        
        # Switch the value of the cell on every time step
        if (above==1):
            new_cells[j] = 0
        else:
            new_cells[j] = 1
        
    # Convert back to a new string    
    string=''.join([str(item) for item in new_cells])
    
    return string


##############################################################################
# We can repeadtedly apply the rule as follows:

steps = 20
string='01110111001100110110'


for i in range(steps):
    print(string)
    string=apply_alternate(string)
    

##############################################################################
# The checquerboard rule
# ----------------------
#
# Esther and I also found a rule which makes the following pattern.
#
# .. image:: ../../images/lesson2/TransitionCheck.png
#
# Let's look at a Python implementation.

def apply_checquerboard(string):

    # Convert the strong to cells 
    cells=list(map(int, [*string]))
    new_cells=cells.copy()
    N=len(cells)
    
    # Loop over every cell and update its state based on the neighboring cells
    for j in range(N):
        # Get the previous value

        left = cells[(j-1)%N]
        above = cells[j]
        right= cells[(j+1)%N]
         
        # Switch the value of the cell on every time step
        if (above==1):
            if ((left==1) & (right==1)):
                new_cells[j] = 1
            else:
                new_cells[j] = 0
        else:
            if ((left==0) & (right==0)):
                new_cells[j] = 0
            else:
                new_cells[j] = 1
                
                
    # Convert back to a new string    
    string=''.join([str(item) for item in new_cells])
    
    return string


##############################################################################
# We can repeadtedly apply the rule as follows:

steps = 20
string='10000001110000000011'

for i in range(steps):
    print(string)
    string=apply_checquerboard(string)
    

##############################################################################
# We will look again at these last two rules in the section on 
# :ref:`cellular chaos<cellularchaos>` where we implement a model which takes
# a set of rules as input and produces a cellular automata as output.

