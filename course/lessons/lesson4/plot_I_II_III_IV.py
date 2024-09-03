"""
I, II, III, IV
==============

Throughout the book, we have used one dimensional cellular automata to illustrate ideas. 
In this section we look at the final class of cellular automata (CA), complex, and investigate 
some of their properties.

In Santa Fe
-----------

Here Chris presents a new set of rules,

.. image:: ../../images/lesson4/ChrisToldUs.png

The output of this looks as follows,

.. image:: ../../images/lesson4/Rule 110.png

We can simulate this cellular automata using the same code as we used in the section
:ref:`cellular chaos<cellularchaos>`, as follows,

"""

import numpy as np

def generate_ca(rule, steps, input_string='', size=50,print_output=1):
    # Initialize the list of cells with a single "on" cell in the middle
    if input_string=='':
        cells = [0] * size
        cells[size // 2] = 1
    else:
        cells=list(map(int, [*input_string]))
        size=len(input_string)
    
    #Store the middle column
    middle_column=[cells[size // 2]]
    
    
    # Create a dictionary for mapping inputs to outputs 
    patterns = {
        (1, 1, 1): rule[0],
        (1, 1, 0): rule[1],
        (1, 0, 1): rule[2],
        (1, 0, 0): rule[3],
        (0, 1, 1): rule[4],
        (0, 1, 0): rule[5],
        (0, 0, 1): rule[6],
        (0, 0, 0): rule[7]
    }
    
    # Generate the specified number of steps
    for i in range(steps):
        # Copy the current state of the cells
        new_cells = [0] * size
        
        # Loop over every cell and update its state based on the neighboring cells
        for j in range(size):
            # Get the pattern of the cell and its neighbors
            if j == 0:
                pattern = (cells[size-1], cells[0], cells[1])
            elif j == size-1:
                pattern = (cells[size-2], cells[size-1], cells[0])
            else:
                pattern = (cells[j-1], cells[j], cells[j+1])
            
            # Update the cell's state based on the pattern
            new_cells[j] = patterns[pattern]
        
        string=''.join([str(item) for item in cells])
        if print_output:
            print(string)
        
        # Update the cells with the new state
        cells = new_cells
    
#Set up the rule
rule = [1, 0, 1, 1, 0, 0, 1, 0] 
size = 70
steps = 70

generate_ca(rule, steps,size=size,print_output=1)

##############################################################################
# Four classes
# ------------
#
# The observation of four different classes of elementary cellular automata
# was first made in the article by Wolfram, which I discuss in the introduction to 
# the book. Here is what `he wrote <https://idp.nature.com/authorize/casa?redirect_uri=https://www.nature.com/articles/311419a0.pdf&casa_token=VLEcK7spHHYAAAAA:4mCoCK3pErHdZUL1Ch3C1AdAyTjcOptyVU6kyKYxwfKkaS8bhJa0hF-BjOM77eWflU8nUYMBgh_SbP8>`_.
#
# .. image:: ../../images/lesson4/WolframFour.png
# 


