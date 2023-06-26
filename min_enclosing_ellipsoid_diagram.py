# -*- coding: utf-8 -*-
"""
@author: jchng3
"""

import numpy as np
from ase import io
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

# Uncomment this if transferred to Jupyter notebook to play with 3D plots interactively
# =============================================================================
# %matplotlib notebook
# =============================================================================

summary_file = open('ellipsoid_diameters_summary.txt', 'w+')
summary_file.write('Molecule.xyz \t Diameter_1 \t Diameter_2 \t Diameter_3 \t error \n')

for file in os.listdir(os.getcwd()):
    if file[-4:]!='.xyz':
        print(file, ' is not an .xyz file.')
        continue
    
    #if file!='hepta15diene.xyz':
    #    continue
    
    atoms = io.read(os.path.join(os.getcwd(), file))
    
    positions_original = atoms.get_positions()
    
    hull_vertices = ConvexHull(positions_original).vertices

    positions = positions_original[hull_vertices,:]
    
    position_transpose = positions.T
    
    tolerance = 1e-5
    
    # Data points
    d, N = np.shape(position_transpose)
    
    Q = np.zeros((d+1, N))
    Q[:d,:] = position_transpose[:d,:N]
    Q[d,:] = np.ones((1,N))
    
    # Initialization
    count = 1
    error = 1
    u = (1/N) * np.ones(N)
    
    # Optimization
    while error > tolerance:
        X = np.matmul(np.matmul(Q, np.diag(u)),Q.T)
        M = np.diag(np.matmul(np.matmul(Q.T, np.linalg.inv(X)),Q))
        maximum = np.amax(M)
        j = np.argmax(M)
        
        step_size = (maximum - d - 1)/((d+1)*(maximum-1))
        new_u = (1 - step_size)*u
        new_u[j] += step_size
        
        count += 1
        
        error = np.linalg.norm(new_u - u)
        
        u = new_u
    
    # Compute ellipsoid parameters
    U = np.diag(u)
    
    # Compute matrix A
    A = (1/d) * np.linalg.inv(np.matmul(np.matmul(position_transpose, U), position_transpose.T) - np.matmul(np.matmul(position_transpose,u).reshape(-1,1), np.matmul(position_transpose,u).reshape(1,-1)))
        
    # Center of ellipsoid
    c = np.matmul(position_transpose, u)
    
    # Singular value decomposition
    u_array, s_array, v_array = np.linalg.svd(A, full_matrices=True)
    
    diameters = 2/np.sqrt(s_array)

    print(file, ': ', diameters, '; error: ', error)
    
    summary_file.write('%s \t %f \t %f \t %f \t %f \n' % (file, diameters[0], diameters[1], diameters[2], error))
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    
    for i in range(len(atoms)):
        if atoms.symbols[i]=='C':
            colour = 'tab:brown'
            size = 150
        elif atoms.symbols[i]=='O':
            colour = 'red'
            size = 120
        elif atoms.symbols[i]=='H':
            colour = 'tab:orange'
            size = 60
        elif atoms.symbols[i]=='N':
            colour = 'tab:purple'
            size = 140
        elif atoms.symbols[i]=='Si':
            colour = 'blue'
            size = 220
        elif atoms.symbols[i]=='Zn':
            colour = 'gray'
            size = 250
        elif atoms.symbols[i]=='F':
            colour = 'yellow'
            size = 170
        ax.scatter(atoms.get_positions()[i][0], atoms.get_positions()[i][1], atoms.get_positions()[i][2],color=colour,s=size/2)
    
    for i in range(len(positions_original)):
        for j in range(len(positions_original)):
            if atoms.symbols[i]=='H' and atoms.symbols[j]=='H':
                continue
            i_j_norm = np.linalg.norm(positions_original[i,:]-positions_original[j,:])
            if (i_j_norm>1) and (i_j_norm<1.95):
                ax.plot3D(positions_original[[i,j],0], positions_original[[i,j],1], positions_original[[i,j],2], '-k')
                
    # plt.show()
    
    phi = np.linspace(0,2*np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
    
    x = (diameters[0]/2) * np.outer(np.cos(phi), np.sin(theta))
    y = (diameters[1]/2) * np.outer(np.sin(phi), np.sin(theta))
    z = (diameters[2]/2) * np.outer(np.ones_like(phi), np.cos(theta))
    
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], v_array) + c
    
    ax.set_title('%s' % (file[:-4]))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    
    # ax.plot_wireframe(x,y,z, alpha=0.3)
    ax.plot_surface(x,y,z,alpha=0.1,color='tab:blue' ,edgecolor ='tab:blue')
    
    # save diagram
    plt.savefig('%s.pdf' % (file[:-4]))

summary_file.close()