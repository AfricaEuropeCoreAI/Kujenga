"""

.. _google:

What you will learn
===================

.. youtube:: D8OxGNeXNIY
    :width: 100%
    :align: center

|

The problem
-----------

How can we measure the influence of a person in a social network? Or how important a webpage is in the internet? This was the question
that Google founders Larry Page and Sergey Brin asked themselves when they were developing their search engine.
In this lesson we will learn the maths behind solution, now known as the PageRank, which is the basis of the algorithm
that powers Google search. As Otema explained in the video above the PageRank algorithm is the additional ingredient that allows search engines
to rank the most relevant pages first. The PageRank algorithm is based on the idea that a page is important if it is linked to
by other important pages. Although it was originally intended to rank web pages, the PageRank algorithm can be used to analysis
any kind of network, for example social interactions, biological systems and transportation infrastructure to name a few.


In this lesson we will work with a small network built from Wikipedia pages of `Ethiopia <https://en.wikipedia.org/wiki/Ethiopia>`_, `Ghana <https://en.wikipedia.org/wiki/Ghana>`_, `Kenya <https://en.wikipedia.org/wiki/Kenya>`_, `Nigeria <https://en.wikipedia.org/wiki/Nigeria>`_, `Rwanda <https://en.wikipedia.org/wiki/Rwanda>`_, `South Africa <https://en.wikipedia.org/wiki/South Africa>`_ and `Uganda <https://en.wikipedia.org/wiki/Uganda>`_ -- chosen for being the home countries of the African members of the Kujenga team. These pages represent a tiny part of the internet but the principles we will learn can be applied to networks of any size. See the video below where Amy introduces our tiny internet.

.. image:: ../../images/lesson4/network.png

We represent our network of pages as a directed graph where each node represents a country, shown as labelled country flags in the video below. In order to construct the connections between nodes in the graph we check on each country's page for links to the other 6 countries. For example, in the first paragraph of the page for `Ethiopia <https://en.wikipedia.org/wiki/Ethiopia>`_ we read:

*Ethiopia, officially the Federal Democratic Republic of Ethiopia, is a landlocked country located in the Horn of Africa region of East Africa. It shares borders with Eritrea to the north, Djibouti to the northeast, Somalia to the east,* `Kenya <https://en.wikipedia.org/wiki/Kenya>`_ *to the south, South Sudan to the west, and Sudan to the northwest.*

Here we see a link to Kenya, meaning we should count this as a connection from Ethiopia to Kenya. We represent this connection as a directed edge in the graph, shown by an arrow pointing from Ethiopia to Kenya. The weight of each edge (shown as a number next to the edge) counts the number of links, in this case 4 in total.

.. note::
    Wikipedia is a dynamic website and the links between pages can change over time, the network we are using is a snapshot of the Wikipedia pages taken in November 2024. You may want to update this network or maybe add your home country if is not already included. You can do this by going to the Wikipedia page of your country and checking for links to the other countries in the network. If you find any, you can add them to the network by editing the code in `Working with matrices`_.



.. youtube:: HO-I-6vHEY4
    :width: 100%
    :align: center

.. admonition:: Food for Thought
:class: food-for-thought

As Amy mentions in the video, use your intuitive understanding of the problem to predict which countries you think will have a high (or low) ranking.

The methods
-----------------

We have seen a graphical representation of a network in `The problem`_ but in order to use the PageRank algorithm we will need to represent the network mathematical as a matrix. In the following sections we will learn how to work with matrices as well as some basic concepts of linear algebra. The most useful concepts for understanding PageRank will be the notions of an eigenvector and eigenvalue of a matrix which we cover in `Defining a matrix`_.

How to use this material
------------------------

This material is taught as part of a 6 hour learning session. Your Kujenga instructor will have booked
a time for an in-person or online two hour session. This means you have two hours of work to do either side of the in-person or online
session. Here is what you should do:

*Before coming to the class*: You should read through this entire page. At the section on multiplying a matrix and a vector,
try to solve the example both by hand and using Python. If you get stuck look `here <LINK NEEDED>`_, but otherwise you should
simply read through and try to understand what we are doing. Once you have read through, you should
download this page as a Jupyter notebook or as Python code by clicking the links at the bottom of this page.
You will need to have a Python environment set up on your computer or access via Google Colab (see here for info on how to set that up).

*During class*: Your teacher will start by going through the theory for `Working with matrices`_.
Please ask them questions and actively engage!

Working with matrices
=====================


Defining a matrix
-----------------

"""

# %%

import random
import pickle
import math
import numpy as np

AT = np.array(
    [
        [0, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 0, 0],
        [1 / 4, 0, 1 / 4, 1 / 4, 1 / 4, 0, 0],
        [1 / 4, 1 / 4, 0, 1 / 4, 1 / 4, 0, 0],
        [1 / 2, 0, 0, 0, 1 / 2, 0, 0],
        [0, 0, 0, 1 / 2, 0, 1 / 2, 0],
        [0, 0, 0, 1 / 3, 1 / 3, 0, 1 / 3],
        [0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0],
    ]
)

A = AT.transpose()
print(A)

#############################################################
# Multiplying a matrix and a vector
# ---------------------------------

p1 = np.zeros(7)
p1[0] = 1
print(p1)
p2 = np.matmul(A, p1)
print(p2)
p3 = np.matmul(A, p2)
print(p3)
p4 = np.matmul(A, p3)
print(p4)
p5 = np.matmul(A, p4)
print(p5)


#############################################################
# The largest eigenvalue
# ----------------------


eigenValues, eigenVectors = np.linalg.eig(A)

idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:, idx]

print(eigenVectors)

pieig = eigenVectors[:, 0]

pieig = pieig / sum(pieig)
pieig2 = np.matmul(A, pieig)

# %%
# The Mathematics of the PageRank algorithm
# =====================
#
# .. youtube:: guf36O9rBXs
#     :width: 100%
#     :align: center
# |
# .. admonition:: Food for Thought
#     :class: food-for-thought
#
#     It turned out that ... had the highest PageRank and ... the lowest. Did this match your expectations? If not, what do you think the reason is? If you are still not sure we will prove the answer in this section.
#
# .. list-table:: Number of outgoing links for each country
#     :widths: 50 50
#     :header-rows: 1
#
#     * - Country
#       - Number of outgoing links
#     * - ZA
#       - 7
#     * - GH
#       - 10
#     * - NG
#       - 6
#     * - RW
#       - 25
#     * - UG
#       - 21
#     * - KE
#       - 20
#     * - ET
#       - 18
#
# .. list-table:: PageRank matrix elements
#     :widths: 15 15 15 15 15 15 15 15
#     :header-rows: 1
#
#     * -
#       - ZA
#       - GH
#       - NG
#       - RW
#       - UG
#       - KE
#       - ET
#     * - ZA
#       - 0
#       - 1/10
#       - 1/6
#       - 1/25
#       - 1/21
#       - 1/20
#       - 0
#     * - GH
#       - 0
#       - 0
#       - 1/6
#       - 0
#       - 0
#       - 0
#       - 0
#     * - NG
#       - 1/7
#       - 1/10
#       - 0
#       - 1/25
#       - 1/21
#       - 1/20
#       - 0
#     * - RW
#       - 1/7
#       - 0
#       - 0
#       - 0
#       - 1/21
#       - 0
#       - 1/18
#     * - UG
#       - 0
#       - 0
#       - 0
#       - 1/25
#       - 0
#       - 1/20
#       - 1/18
#     * - KE
#       - 0
#       - 0
#       - 0
#       - 1/25
#       - 1/21
#       - 0
#       - 1/18
#     * - ET
#       - 1/7
#       - 1/10
#       - 0
#       - 1/25
#       - 0
#       - 1/20
#       - 0
#
# .. math::
#     \mathbf{M} =
#     \begin{pmatrix}
#     0 & \frac{1}{10} & \frac{1}{6} & \frac{1}{25} & \frac{1}{21} & \frac{1}{20} & 0 \\
#     0 & 0 & \frac{1}{6} & 0 & 0 & 0 & 0 \\
#     \frac{1}{7} & \frac{1}{10} & 0 & \frac{1}{25} & \frac{1}{21} & \frac{1}{20} & 0 \\
#     \frac{1}{7} & 0 & 0 & 0 & \frac{1}{21} & 0 & \frac{1}{18} \\
#     0 & 0 & 0 & \frac{1}{25} & 0 & \frac{1}{20} & \frac{1}{18} \\
#     0 & 0 & 0 & \frac{1}{25} & \frac{1}{21} & 0 & \frac{1}{18} \\
#     \frac{1}{7} & \frac{1}{10} & 0 & \frac{1}{25} & 0 & \frac{1}{20} & 0
#     \end{pmatrix}


M = np.array(
    [
        [0, 1 / 10, 1 / 6, 1 / 25, 1 / 21, 1 / 20, 0],
        [0, 0, 1 / 6, 0, 0, 0, 0],
        [1 / 7, 1 / 10, 0, 1 / 25, 1 / 21, 1 / 20, 0],
        [1 / 7, 0, 0, 0, 1 / 21, 0, 1 / 18],
        [0, 0, 0, 1 / 25, 0, 1 / 20, 1 / 18],
        [0, 0, 0, 1 / 25, 1 / 21, 0, 1 / 18],
        [1 / 7, 1 / 10, 0, 1 / 25, 0, 1 / 20, 0],
    ]
)

print(AT)

# %%
# .. math::
#     \mathbf{R} = \begin{pmatrix} r_1 \\ r_2 \\ \vdots \\ r_N \end{pmatrix}
# .. math::
#     \mathbf{R}(t + 1) = d\mathbf{M}\mathbf{R}(t) + \frac{1-d}{N} \mathbf{1}
# .. math::
#     \mathbf{R}(t + 1) \approx \mathbf{R}(t)
# .. math::
#     M_{ij} = \begin{cases} 1/L_j^{\text{out}}, & \text{if } j \text{ links to } i \\ 0, & \text{otherwise} \end{cases}
# .. math::
#     \mathbf{R}(t + 1) = \frac{1}{\lambda} \mathbf{M}\mathbf{R}(t)
