# -*- coding: utf-8 -*-
"""
Plot circular landmarks in image given corresponding centers.
Centers are some hyperparameters given by the user.
Freely change it to draw you own maps.

@author: Riemann Lee
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Generate centers of cicular landmarks and it's radii
for i in range(10):
    img = plt.imread('./img/Canvas.png')
    fig,ax = plt.subplots(1)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_position(('data', 800))
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_position(('data', 600))
    # Create a figure. Equal aspect so circles look circular
    ax.set_aspect('equal')

    # Don't use any ticks to make the image more visible
    plt.xticks([])
    plt.yticks([])

    # Show the image
    ax.imshow(img)
    
    Centers = np.random.uniform([0, 0], [800, 600], size=(10,2))
    np.save('./data/centers/centers_%s.npy' %i, Centers)
    Radii = np.random.uniform(0, 30, size=(10,1))
    np.save('./data/radii/radii_%s.npy' %i, Radii)
    
    if len(Centers) != len(Radii):
        raise ValueError("Centers and Radii must have the same size!")
    
    # Now, loop through coord arrays, and create a circle at center    
    for count, value in enumerate(Centers):
        circ = Circle(value,Radii[count])
        ax.add_patch(circ)
    
    fig.savefig("./img/Map_With_Landmarks_%s.png" %i)

# Version 0: Rejection Sampling Example
# Centers = np.array([ [144,73], [510,43], [336,175], [718,159], [178,484], [665,464], [267, 333], [541, 300], [472, 437], [100, 533] ])
# np.save('./data/centers/example_centers.npy', Centers)
# Radii = np.array([ [12], [32], [27], [28], [13], [16], [37], [18], [9], [20] ])
# np.save('./data/radii/example_radii.npy', Radii)
# Show the map with landmarks
# fig.savefig("./img/Map_Rejection_Sampling.png")