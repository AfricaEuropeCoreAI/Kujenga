
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "gallery/lesson3/plot_runners.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_gallery_lesson3_plot_runners.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_gallery_lesson3_plot_runners.py:


.. _google:

What you will learn
===================

The problem
-----------

Are Ethipian or Kenyan runners the best in the world? This is a question that has been asked many times.
In this lesson we will learn how to test a hypothesis using data. 

The methods
-----------------

We will learn how to plot a histogram and how to use a statistical test (t-test) to compare two means. 
For more information on the type of test we will look at see `here https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample`_. 


How to use this material
------------------------

This material is taught as part of a 6 hour learning session. Your Kujenga instructor will have booked 
a time for an in-person or online two hour session. This means you have two hours to work to do either side of the
session. Here is what you should do:

*Before coming to the class*: You should read through this entire page. At the section on calculating a confidence interval
for a mean. If you get stuck look 
`here <https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample/estimating-population-mean/v/introduction-to-t-statistics>`_. 
After that section you should simply read through and try to understand what we are doing. 

Once you have read through, you should 
download this page as a Jupyter notebook or as Python code by clicking the links at the bottom of this page.
You will need to have a Python environment set up on your computer or access via Google Colab (see here for
 info on how to set that up). Please make sure you have the notebook and a Python environment set up before the class.

 *During class*: Your teacher will start by going through the theory for `the t-test`_. 
 Please ask them questions and actively engage! 

Working with data
=================

Loading in the data set
-----------------------

.. GENERATED FROM PYTHON SOURCE LINES 47-61

.. code-block:: default



    import random
    import pickle
    import math
    import numpy as np

    # Load in times of runners

    #with open('course/lessons/data/runners.pkl', 'rb') as f:


    # Plot a histogram of the data








.. GENERATED FROM PYTHON SOURCE LINES 62-78

Calculating a confidence interval for a mean
============================================

 Calculation by hand
 -------------------

 Calculate by hand the confidence interval for the mean of the following times for 10000m runners from Kenya:

 U




 Calculatation using Python
 --------------------------


.. GENERATED FROM PYTHON SOURCE LINES 78-89

.. code-block:: default


    a=1

    # Now for Ethiopian runners
    # -------------------------
    #
    # Here the person should do themselves.
    #










.. GENERATED FROM PYTHON SOURCE LINES 90-92

A statistical test
=================

.. GENERATED FROM PYTHON SOURCE LINES 92-101

.. code-block:: default


    b=2

    #
    #  We will follow test `here https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample`_. 












.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_gallery_lesson3_plot_runners.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example


    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_runners.py <plot_runners.py>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_runners.ipynb <plot_runners.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
