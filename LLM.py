import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

# Read polar data
airfoil = 'ARAD8pct_polar.csv'
data1=pd.read_csv(airfoil, header=0, names = ["alfa", "cl", "cd", "cm"],  sep=',')
polar_alfa = data1['alfa'][:]
polar_cl = data1['cl'][:]
polar_cd = data1['cd'][:]

# Freestream Parameters
rho = 1.007                     # density at h=2000m [kg/m^3]
Pamb = 79495.22                 # static pressure at h=2000m [Pa]
Vinf = 60                       # velocity [m/s]
J = np.array([1.6, 2.0, 2.4])   # advance ratio
'''
# # Blade geometry
# R = 0.7                             # Blade radius [m]
# Nb = 6                              # number of blades
# tip_pos_R = 1                       # normalised blade tip position (r_tip/R)
# root_pos_R = 0.25                   # normalised blade root position (r_root/R)
# pitch = 46                          # blade pitch [deg]
# 
# # Discretisation into blade elements
# nodes = 5
# r_R = np.linspace(root_pos_R, tip_pos_R, nodes)
# chord_dist = 0.18 - 0.06*(r_R)                  # chord distribution [m]
# twist_dist = -50*(r_R) + 35 + pitch             # twist distribution [deg]

# # Dependent variables 
# n = Vinf/(2*J*R)    # RPS [Hz]
# Omega = 2*np.pi*n   # Angular velocity [rad/s]
# TSR = np.pi/J       # tip speed ratio
'''
# Wing Geometry
b = 1       # span [m]
c = 0.2     # chord [m]
AR = b/c    # aspect ratio  

# Discretisation 
blade_seg = 2       # no. of segments for the wing
vor_fil = 3         # no. of vortex filaments
l = 1               # length of the trailing vortices [m]

l_fil = (2*l + (b/blade_seg))/vor_fil   # length of vortex filament

# coordinates for filaments
coord_fil = [[np.zeros(blade_seg+1) for i in range(int((vor_fil+1)/2))] for j in range(int((blade_seg+1)*(vor_fil+1)/2))]

for k in range(int((blade_seg+1)*(vor_fil+1)/2)):
    z_fil = 0
    for j in range(int((vor_fil+1)/2)):
        y_fil = (b/blade_seg)*j
        for i in range(blade_seg+1):
            x_fil = (l/((vor_fil+1)/2))*i
            coord_fil = 

print(np.array(coord_fil))

