"""
.. _chaosbutterfly:

The butterfly effect
====================

The three differential equations proposed by Edward Lorenz in his 
`1963 paper <https://journals.ametsoc.org/downloadpdf/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.pdf>`_ 
can be thought of (roughly)in terms of weather on a tropical island. We can think of the variables as 
strength of breeze (:math:´X´), which increases with :math:´Y´ 
but decreases as a function of itself

.. math::

    \\frac{dX}{dt} = s(Y-X)

Temperature difference between currents (:math:´Y´)) which increases with 
:math:´X´ but decreases as a function of itself and of :math:´Z´,

.. math::

    \\frac{dY}{dt} = rX - Y - XZ 


And land/air temperature distortion (:math:´Z´) which increases with :math:´X´ but decreases 
as a function of itself,

.. math::

    \\frac{dZ}{dt} = XY - bZ 

  
There are three parameter values, which are set to :math:´s=10´, :math:´r=28´ 
and :math:´b=8/3´ to create the butterfly (th Lorenz attractor). 

Simulating chaos
----------------

Let's first define a function which gives the derivatives at each point

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pylab import rcParams
matplotlib.font_manager.FontProperties(family='Helvetica',size=11)

from scipy import integrate
from mpl_toolkits import mplot3d

def lorenz(XYZ,t=0):
    
    
    dXdt = s*(XYZ[1] - XYZ[0])                  # X 
    dYdt = r*XYZ[0] - XYZ[1] - XYZ[0]*XYZ[2]    # Y 
    dZdt = XYZ[0]*XYZ[1] - b*XYZ[2]             # Z 
    
    return np.array([dXdt, dYdt, dZdt])


# Parameter values
s=10
r=28
b=8/3

##############################################################################
# Now we solve the equations numerically. By plotting them in three dimensions
# we can see how the weather never repeats. 

endtime=300
dt = 0.01 

t = np.linspace(0,endtime, int(endtime/dt))               # time
XYZ0 = np.array([0.3, 1.2, 5.05])          # initial conditions
XYZ = integrate.odeint(lorenz, XYZ0, t)

start=int(200/dt)
finish=int(start+40/dt)

rcParams['figure.figsize'] = 14/2.54, 14/2.54
ax=plt.subplot(projection='3d')
ax.set_xticks(np.arange(-20,21,step=10))
ax.set_yticks(np.arange(-20,21,step=10))
ax.set_zticks(np.arange(0,40,step=10))
ax.yaxis.labelpad=10
ax.plot3D(XYZ[start:finish,0], XYZ[start:finish,1], XYZ[start:finish,2], lw=1,color='k')
ax.set_xlabel("Strength of breeze ($X$)")
ax.set_ylabel("Temperature difference \n between currents ($Y$)")
ax.set_zlabel("Land/air temperature \n distortion ($Z$) ")

# Format the axes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.set_xlim(-21,21)
ax.set_zlim(0,40) 
ax.set_ylim(-21,21)


plt.show()

##############################################################################
# This is the butterfly of chaos in all its glory.
#
#
#
#
#
# How variables change over time
# ------------------------------
#
# In order to better understand chaos, Fetter and Lorenz measured the height of 
# consecutive peaks in the temperature distortion Z. Before we do this, let's 
# start by plotting the three variables over time.


rcParams['figure.figsize'] = 14/2.54, 7/2.54

finish=int(start+20/dt)

fig,ax=plt.subplots(num=1)
ax.plot(XYZ[start:finish,0], color='black')
ax.plot(XYZ[start:finish,1], color='green')
ax.plot(XYZ[start:finish,2], color='red')
plt.show()

##############################################################################
#
# To identify the peaks we note each time Z reaches a maximum. This is done using
# a function argrelextrema from the Scipy package. 
#

from scipy.signal import argrelextrema

# for local maxima
zm= XYZ[:,2]
maxz=argrelextrema(zm, np.greater)
zm=zm[maxz]

zm=zm[20:]
fig,ax=plt.subplots(num=1)
ax.plot(np.arange(len(zm)),zm, color='black')
ax.set_ylabel('Maximum value of $Z$')
ax.set_xlabel('Iteration round the butterfly')
ax.set_ylim((27,52))
ax.set_xlim((0,50))
ax.set_xticks(range(0,60,10))
ax.set_yticks(range(30,60,10))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()

##############################################################################
# To be clear, this is not a plot of Z itself, but rather a plot of the maximum values it 
# takes on each loop round the butterfly shape. 
#
# .. admonition:: Think yourself!
#   
#       Does anything strike you as familiar in this plot?  
#       Compare it to the time series for the :ref:`doubling rule<doublingmap>`. 
#       The rise and fall of the 
#       maxima is not enirely disimilar to the map we saw there. Run the code above for
#       different intitial conditions and look at how the sequence of maxima changes. 
#
# The chaos in Lorenz equation can be teased out further by plotting consecutive maxima of Z to 
# create a map of maxima from one time to the next. So, each time Z reaches a 
# maximum the size of that value is noted and then consecutive values of Z are 
# plotted against each other. 


rcParams['figure.figsize'] = 10/2.54, 10/2.54
fig,ax=plt.subplots(num=1)
ax.plot(zm[:-1],zm[1:],linestyle='none',marker='.',color='k')
ax.set_ylabel('Next Maximum of $Z$')
ax.set_xlabel('Previous Maximum of $Z$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(20,51,step=10))
ax.set_yticks(np.arange(20,51,step=10))
ax.set_xlim(27,52)
ax.set_ylim(27,52) 


##############################################################################
# The same tent-like shape is seen here as we saw :ref:`earlier<doublingmap>`.
# For Lorenz article, Ellen Fetter made similar plots for the height of consecutive peaks. 
# This tells us that in the model, when temperature distortions are small, 
# the temperature distortion typically doubled, while large distortions are 
# followed by very small distortions. 
#
# Lorenz noted the similarity to the tent map and started to sketch out 
# an argument as to why this meant that any two close-by points will soon move apart. 
# In doing so, he provided the first argument as to why the weather is chaotic. 
# It wasn't a rigorous
# proof at that stage, but it was the starting point of an explanation.


##############################################################################
# Learn more
# ----------
#
# The original paper by Lorenz: 
#
# `Edward N. Lorenz, Deterministic nonperiodic flow, Journal of Atmospheric Sciences 20, no. 2 (1963): 130‒41 <https://journals.ametsoc.org/downloadpdf/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.pdf>`_
#
# A more mathematical description of the Lorenz equations and their relationship to
# the doubling map (which we looked at :ref:`here<doublingmap>`): 
#
# `Étienne Ghys, The Lorenz attractor, a paradigm for chaos, Chaos (2013): 1‒54, p. 20 <https://link.springer.com/chapter/10.1007/978-3-0348-0697-8_1>`_
# 
# A beautiful and thorough analysis of Lorenz equations:
#
# `Colin Sparrow, The Lorenz Equations: Bifurcations, Chaos, and Strange Attractors, Vol. 41, Springer Science and Business Media, 2012 <https://link.springer.com/book/10.1007/978-1-4612-5767-7>`_
#