"""

.. _google:

What you will learn
===================

The problem
-----------

How can we measure the influence of a person in a social network? Or how important a webpage is in the internet? This was the question 
that Google founders Larry Page and Sergey Brin asked themselves when they were developing their search engine. 
In this lesson we will learn the maths behing solution, now known as the PageRank, which is the basis of the algorithm 
that powers Google search.

The methods
-----------------

Here we will learn about matrices and linear algebra. We will learn about the concept of an eigenvector and eigenvalue of a matrix.

How to use this material
------------------------

This material is taught as part of a 6 hour learning session. Your Juenga instructor will have booked 
a time for an in-person or online two hour session. This means you have two hours to work to do either side of the
session. Here is what you should do:

*Before coming to the class*: You should read through this entire page. At the section on matrix multiplication, try to solve the example both by hand
and using Python. If you get stuck look `here <LINK NEEDED>`_), but otherwise you should 
simply read through and try to understand what we are doing. Once you have read through, you should 
download this page as a Jupyter notebook or as Python code by clicking the links at the bottom of this page.
You will need to have a Python environment set up on your computer or access via Google Colab (see here for
 info on how to set that up). ...

 *During class*: Your teacher will start by going through the theory for `Finding the best fit line`_. 
 Please ask them questions and actively engage! 

Working with matrices
=====================


Defining a matrix
-----------------
"""


import random
import pickle
import math
import numpy as np

AT=np.array([[0,1/4,1/4,1/4,1/4,0,0],
             [1/4,0,1/4,1/4,1/4,0,0],
             [1/4,1/4,0,1/4,1/4,0,0],
             [1/2,0,0,0,1/2,0,0],
             [0,0,0,1/2,0,1/2,0],
             [0,0,0,1/3,1/3,0,1/3],
             [0,0,0,1/3,1/3,1/3,0]])

A=AT.transpose()
print(A)

#############################################################
#Multiplying a matrix and a vector
#---------------------------------

p1=np.zeros(7)
p1[0]=1
print(p1)
p2=np.matmul(A,p1)
print(p2)
p3=np.matmul(A,p2)
print(p3)
p4=np.matmul(A,p3)
print(p4)
p5=np.matmul(A,p4)
print(p5)



#############################################################
#The largest eigenvalue
#----------------------


eigenValues, eigenVectors = np.linalg.eig(A)

idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]

print(eigenVectors)

pieig=eigenVectors[:,0]

pieig=pieig/sum(pieig)
pieig2=np.matmul(A,pieig)
