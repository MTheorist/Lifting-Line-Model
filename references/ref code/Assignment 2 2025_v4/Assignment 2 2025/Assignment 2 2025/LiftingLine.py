# -*- coding: utf-8 -*-
"""


@author: luked
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from functions import *

# Rotor specifications
R      = 50             # rotor radius [m]
B      = 3              # number of blades [-]
root_R = 0.2            # rotor root
tip_R  = 1              # tip
pitch  = 2              # pitch angle [deg]
tsr    = 6              # tip speed ratio [-]
#axial_convection = 0.2  # average axial convection for expansion of rotor wake, for base setting
axi_convec = [0, 0.1, 0.2, 0.3, 0.4] #  axial convection 

NELEMENTS = 13 
#Nrotations = [20, 30, 40, 50, 60]  # wake length in rotations
Nrotations = 50 # wake length in rotations, for base setting
# Nazim_disc = range(20, 101, 20) # number of azimuthal discretization points per rotation
Nazim_disc = 100 # number of azimuthal discretization points per rotation, for base setting
U_inf = 10              # freestream velocity
rho = 1.225             # density
omega = tsr*U_inf/R     

# polars 
airfoil = 'DU95W180data.txt'
data1   = np.genfromtxt(airfoil,skip_header=1)
polar_alpha = data1[1:,0]
polar_cl    = data1[1:,1]
polar_cd    = data1[1:,2]

s_array = np.linspace(0, math.pi, NELEMENTS)

r_array = np.zeros(len(s_array))

for i, s in enumerate(s_array):
    r_array[i] = (-1*(math.cos(s)-1)/2*(tip_R-root_R)+root_R)

for i, r in enumerate(r_array):
    s_array[i] = r*R

maxradius = max(s_array)


#~~~~~~~~~~~~~~~~~~ Solving lifting line system for frozen wake sensitivity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# theta_array = np.arange(0, Nrotations*2*math.pi, math.pi/10)
# rotor_wake_system = create_rotor_geometry(s_array, maxradius, tsr/(1-axial_convection), theta_array, B)
# resultsLL = solve_lifting_line_system_matrix_approach(rotor_wake_system, [U_inf, 0, 0], omega, maxradius, rho, polar_alpha, polar_cl, polar_cd)#, precomputed_file = "matrices_N20R50U1.npy")
# resultsLL_frozenwake = solve_lifting_line_system_matrix_approach(rotor_wake_system, [U_inf, 0, 0], omega, maxradius, rho, polar_alpha, polar_cl, polar_cd, frozen_wake= True)#, precomputed_file = "matrices_N20R50U1.npy")
# # unpacking results
# a_LL = resultsLL[0]
# aline_LL = resultsLL[1]
# r_R_LL = resultsLL[2]
# Fnorm_LL = resultsLL[3]
# Ftan_LL = resultsLL[4]
# Gamma_LL = resultsLL[5]
# alpha_LL = resultsLL[6]
# inflow_angle_LL = resultsLL[7]

resultsLL_list_frozenwake_axialconvection = []

for axial_convection_list in axi_convec:
    theta_array = np.arange(0, Nrotations*2*math.pi, 2*math.pi/Nazim_disc)
    rotor_wake_system = create_rotor_geometry(s_array, maxradius, tsr/(1-axial_convection_list), theta_array, B)
    resultsLL_frozen = solve_lifting_line_system_matrix_approach(rotor_wake_system, [U_inf, 0, 0], omega, maxradius, rho, polar_alpha, polar_cl, polar_cd)
    resultsLL_list_frozenwake_axialconvection.append(resultsLL_frozen)

a_LL_frozen_axial = []
aline_LL_frozen_axial = []
r_R_LL_frozen_axial = []
Fnorm_LL_frozen_axial = []
Ftan_LL_frozen_axial = []
Gamma_LL_frozen_axial = []
alpha_LL_frozen_axial = []
inflow_angle_LL_frozen_axial = []
CT_LL_frozen_axial = []
CP_LL_frozen_axial = []

for i in range(len(resultsLL_list_frozenwake_axialconvection)):
    a_LL = resultsLL_list_frozenwake_axialconvection[i][0]
    aline_LL = resultsLL_list_frozenwake_axialconvection[i][1]
    r_R_LL = resultsLL_list_frozenwake_axialconvection[i][2]
    Fnorm_LL = resultsLL_list_frozenwake_axialconvection[i][3]
    Ftan_LL = resultsLL_list_frozenwake_axialconvection[i][4]
    Gamma_LL = resultsLL_list_frozenwake_axialconvection[i][5]
    alpha_LL = resultsLL_list_frozenwake_axialconvection[i][6]
    inflow_angle_LL = resultsLL_list_frozenwake_axialconvection[i][7]
    CT_LL, CP_LL = calculateCTCP(Fnorm_LL, Ftan_LL, U_inf, r_array, omega, R, B, rho)

    a_LL_frozen_axial.append(a_LL)
    aline_LL_frozen_axial.append(aline_LL)
    r_R_LL_frozen_axial.append(r_R_LL)
    Fnorm_LL_frozen_axial.append(Fnorm_LL)
    Ftan_LL_frozen_axial.append(Ftan_LL)
    Gamma_LL_frozen_axial.append(Gamma_LL)
    alpha_LL_frozen_axial.append(alpha_LL)
    inflow_angle_LL_frozen_axial.append(inflow_angle_LL)
    CT_LL_frozen_axial.append(CT_LL)
    CP_LL_frozen_axial.append(CP_LL)    


# a_LL_frozenwake = resultsLL_frozenwake[0]
# aline_LL_frozenwake = resultsLL_frozenwake[1]
# r_R_LL_frozenwake = resultsLL_frozenwake[2]
# Fnorm_LL_frozenwake = resultsLL_frozenwake[3]
# Ftan_LL_frozenwake = resultsLL_frozenwake[4]
# Gamma_LL_frozenwake = resultsLL_frozenwake[5]
# alpha_LL_frozenwake = resultsLL_frozenwake[6]
# inflow_angle_LL_frozenwake = resultsLL_frozenwake[7]

#~~~~~~~~~~~~~~~~~~ Solving lifting line system, loop for azimuthal discretisation sensitivity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# resultsLL_list_azim = []
# for N in Nazim_disc:
#     theta_array = np.arange(0, Nrotations*2*math.pi, 2*math.pi/N)
#     rotor_wake_system = create_rotor_geometry(s_array, maxradius, tsr/(1-axial_convection), theta_array, B)
#     resultsLL = solve_lifting_line_system_matrix_approach(rotor_wake_system, [U_inf, 0, 0], omega, maxradius, rho, polar_alpha, polar_cl, polar_cd)
#     resultsLL_list_azim.append(resultsLL)
# #unpacking results
# a_LL_list_azim = []
# aline_LL_list_azim = []
# r_R_LL_list_azim = []
# Fnorm_LL_list_azim = []
# Ftan_LL_list_azim = []
# Gamma_LL_list_azim = []
# alpha_LL_list_azim = []
# inflow_angle_LL_list_azim = []
# CT_LL_list_azim = []
# CP_LL_list_azim = []
# for i in range(len(resultsLL_list_azim)):
#     a_LL = resultsLL_list_azim[i][0]
#     aline_LL = resultsLL_list_azim[i][1]
#     r_R_LL = resultsLL_list_azim[i][2]
#     Fnorm_LL = resultsLL_list_azim[i][3]
#     Ftan_LL = resultsLL_list_azim[i][4]
#     Gamma_LL = resultsLL_list_azim[i][5]
#     alpha_LL = resultsLL_list_azim[i][6]
#     inflow_angle_LL = resultsLL_list_azim[i][7]
#     CT_LL, CP_LL = calculateCTCP(Fnorm_LL, Ftan_LL, U_inf, r_array, omega, R, B, rho)

#     a_LL_list_azim.append(a_LL)
#     aline_LL_list_azim.append(aline_LL)
#     r_R_LL_list_azim.append(r_R_LL)
#     Fnorm_LL_list_azim.append(Fnorm_LL)
#     Ftan_LL_list_azim.append(Ftan_LL)
#     Gamma_LL_list_azim.append(Gamma_LL)
#     alpha_LL_list_azim.append(alpha_LL)
#     inflow_angle_LL_list_azim.append(inflow_angle_LL)
#     CT_LL_list_azim.append(CT_LL)
#     CP_LL_list_azim.append(CP_LL)

##~~~~~~~~~~~~~~~~~~ Solving lifting line system, loop for wake length sensitivity and convergence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#resultsLL_list_Nrot = []
#for Nrot in Nrotations:
    #theta_array = np.arange(0, Nrot*2*math.pi, 2*math.pi/Nazim_disc)
    #rotor_wake_system = create_rotor_geometry(s_array, maxradius, tsr/(1-axial_convection), theta_array, B)
    #resultsLL = solve_lifting_line_system_matrix_approach(rotor_wake_system, [U_inf, 0, 0], omega, maxradius, rho, polar_alpha, polar_cl, polar_cd)
    #resultsLL_list_Nrot.append(resultsLL)
##unpacking results
#a_LL_list_Nrot = []
#aline_LL_list_Nrot = []
#r_R_LL_list_Nrot = []
#Fnorm_LL_list_Nrot = []
#Ftan_LL_list_Nrot = []
#Gamma_LL_list_Nrot = []
#alpha_LL_list_Nrot = []
#inflow_angle_LL_list_Nrot = []
#CT_LL_list_Nrot = []
#CP_LL_list_Nrot = []
#for i in range(len(resultsLL_list_Nrot)):
    #a_LL = resultsLL_list_Nrot[i][0]
    #aline_LL = resultsLL_list_Nrot[i][1]
    #r_R_LL = resultsLL_list_Nrot[i][2]
    #Fnorm_LL = resultsLL_list_Nrot[i][3]
    #Ftan_LL = resultsLL_list_Nrot[i][4]
    #Gamma_LL = resultsLL_list_Nrot[i][5]
    #alpha_LL = resultsLL_list_Nrot[i][6]
    #inflow_angle_LL = resultsLL_list_Nrot[i][7]
    #CT_LL, CP_LL = calculateCTCP(Fnorm_LL, Ftan_LL, U_inf, r_array, omega, R, B, rho)

    #a_LL_list_Nrot.append(a_LL)
    #aline_LL_list_Nrot.append(aline_LL)
    #r_R_LL_list_Nrot.append(r_R_LL)
    #Fnorm_LL_list_Nrot.append(Fnorm_LL)
    #Ftan_LL_list_Nrot.append(Ftan_LL)
    #Gamma_LL_list_Nrot.append(Gamma_LL)
    #alpha_LL_list_Nrot.append(alpha_LL)
    #inflow_angle_LL_list_Nrot.append(inflow_angle_LL)
    #CT_LL_list_Nrot.append(CT_LL)
    #CP_LL_list_Nrot.append(CP_LL)


#a_BEM = np.array([])
#aline_BEM = np.array([])
#r_R_BEM = np.array([])
#Fnorm_BEM = np.array([])
#Ftan_BEM = np.array([])
#Gamma_BEM = np.array([])
#alpha_BEM = np.array([])
#inflow_angle_BEM = np.array([])

#for i in range(len(r_array)-1):
    #resultsBEM = solveStreamtube(U_inf, r_array[i], r_array[i+1], root_R, tip_R , omega, maxradius, B, rho, polar_alpha, polar_cl, polar_cd)

    #a_BEM = np.append(a_BEM, resultsBEM[0])
    #aline_BEM = np.append(aline_BEM, resultsBEM[1])
    #r_R_BEM = np.append(r_R_BEM, resultsBEM[2])
    #Fnorm_BEM = np.append(Fnorm_BEM, resultsBEM[3])
    #Ftan_BEM = np.append(Ftan_BEM, resultsBEM[4])
    #Gamma_BEM = np.append(Gamma_BEM, resultsBEM[5])
    #alpha_BEM = np.append(alpha_BEM, resultsBEM[6])
    #inflow_angle_BEM = np.append(inflow_angle_BEM, resultsBEM[7])

#CT_BEM, CP_BEM = calculateCTCP(Fnorm_BEM, Ftan_BEM, U_inf, r_array, omega, R, B, rho)


# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(r_array[1:], alpha_LL[0:len(r_array)-1],'-r')
# ax.plot(r_array[1:], alpha_BEM[0:len(r_array)-1],'--r')
# ax.plot(r_array[1:], inflow_angle_LL[0:len(r_array)-1],'-b')
# ax.plot(r_array[1:], inflow_angle_BEM[0:len(r_array)-1],'--b')
# ax.legend([r"$\alpha$ - LL", r"$\alpha$ - BEM", r"$\phi$ - LL", r"$\phi$ - BEM"])
# ax.set_xlabel("r/R [-]")
# ax.set_ylabel(r"Angle [$\deg$]")
# ax.set_title("TSR=6")
# ax.grid()
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# fig2 = plt.figure()
# ax = fig2.add_subplot()
# ax.plot(r_array[1:], Gamma_LL[0:len(r_array)-1],'-r')
# ax.plot(r_array[1:], Gamma_BEM[0:len(r_array)-1],'--r')
# # ax.legend([r"$\alpha$ - LL", r"$\alpha$ - BEM", r"$\phi$ - LL", r"$\phi$ - BEM"])
# # ax.set_xlabel("r/R [-]")
# # ax.set_ylabel(r"Angle [$\deg$]")
# # ax.set_title("TSR=6")
# ax.grid()
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fnorm_LL_plot = np.divide(Fnorm_LL[0:len(r_array)-1], 1/2*rho*U_inf**2*R)
# Ftan_LL_plot = np.divide(Ftan_LL[0:len(r_array)-1], 1/2*rho*U_inf**2*R)
# Fnorm_BEM_plot = np.divide(Fnorm_BEM[0:len(r_array)-1], 1/2*rho*U_inf**2*R)
# Ftan_BEM_plot = np.divide(Ftan_BEM[0:len(r_array)-1], 1/2*rho*U_inf**2*R)
# fig3 = plt.figure()
# ax = fig3.add_subplot()
# ax.plot(r_array[1:], Fnorm_LL_plot[0:len(r_array)-1],'-r')
# ax.plot(r_array[1:], Fnorm_BEM_plot[0:len(r_array)-1],'--r')
# ax.plot(r_array[1:], Ftan_LL_plot[0:len(r_array)-1],'-b')
# ax.plot(r_array[1:], Ftan_BEM_plot[0:len(r_array)-1],'--b')# ax.legend([r"$\alpha$ - LL", r"$\alpha$ - BEM", r"$\phi$ - LL", r"$\phi$ - BEM"])
# # ax.set_xlabel("r/R [-]")
# # ax.set_ylabel(r"Angle [$\deg$]")
# # ax.set_title("TSR=6")
# ax.grid()
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# fig4 = plt.figure()
# ax = fig4.add_subplot()
# ax.plot(r_array[1:], alpha_LL[0:len(r_array)-1],'-r')
# ax.plot(r_array[1:], alpha_LL_frozenwake[0:len(r_array)-1],'--r')
# ax.plot(r_array[1:], inflow_angle_LL[0:len(r_array)-1],'-b')
# ax.plot(r_array[1:], inflow_angle_LL_frozenwake[0:len(r_array)-1],'--b')
# ax.legend([r"$\alpha$ - free wake", r"$\alpha$ - frozen wake", r"$\phi$ - free wake", r"$\phi$ - frozen wake"])
# ax.set_xlabel("r/R [-]")
# ax.set_ylabel(r"Angle [$\deg$]")
# ax.set_title("TSR=6")
# ax.grid()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# fig5 = plt.figure()
# ax = fig5.add_subplot()
# ax.plot(r_array[1:], Gamma_LL[0:len(r_array)-1],'-r')
# ax.plot(r_array[1:], Gamma_LL_frozenwake[0:len(r_array)-1],'--r')
# # ax.legend([r"$\alpha$ - LL", r"$\alpha$ - BEM", r"$\phi$ - LL", r"$\phi$ - BEM"])
# # ax.set_xlabel("r/R [-]")
# # ax.set_ylabel(r"Angle [$\deg$]")
# # ax.set_title("TSR=6")
# ax.grid()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fnorm_LL_plot = np.divide(Fnorm_LL[0:len(r_array)-1], 1/2*rho*U_inf**2*R)
# Ftan_LL_plot = np.divide(Ftan_LL[0:len(r_array)-1], 1/2*rho*U_inf**2*R)
# Fnorm_LL_frozenwake_plot = np.divide(Fnorm_LL_frozenwake[0:len(r_array)-1], 1/2*rho*U_inf**2*R)
# Ftan_LL_frozenwake_plot = np.divide(Ftan_LL_frozenwake[0:len(r_array)-1], 1/2*rho*U_inf**2*R)
# fig6 = plt.figure()
# ax = fig6.add_subplot()
# ax.plot(r_array[1:], Fnorm_LL_plot[0:len(r_array)-1],'-r')
# ax.plot(r_array[1:], Fnorm_LL_frozenwake_plot[0:len(r_array)-1],'--r')
# ax.plot(r_array[1:], Ftan_LL_plot[0:len(r_array)-1],'-b')
# ax.plot(r_array[1:], Ftan_LL_frozenwake_plot[0:len(r_array)-1],'--b')# ax.legend([r"$\alpha$ - LL", r"$\alpha$ - BEM", r"$\phi$ - LL", r"$\phi$ - BEM"])
# # ax.set_xlabel("r/R [-]")
# # ax.set_ylabel(r"Angle [$\deg$]")
# # ax.set_title("TSR=6")
# ax.grid()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Frozen Wake Sensitivity~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig11 = plt.figure()
ax = fig11.add_subplot()
for i in range(len(resultsLL_list_frozenwake_axialconvection)):
    axial = axi_convec[i]
    # uncomment other lines to plot other variables, change axis labels accordingly
    ax.plot(r_array[1:], alpha_LL_frozen_axial[i][0:len(r_array)-1], '-', label=fr'$axial convection speed = {axial}$')
    # ax.plot(r_array[1:], inflow_angle_LL_list_azim[i][0:len(r_array)-1],'-',label=fr'$\delta_{{azim}} = 2\pi/{N}$')
    # ax.plot(r_array[1:], Gamma_LL_list_azim[i][0:len(r_array)-1],'-', label=fr'$\delta_{{azim}} = 2\pi/{N}$')
    # ax.plot(r_array[1:], Fnorm_LL_list_azim[i][0:len(r_array)-1],'-',label=fr'$\delta_{{azim}} = 2\pi/{N}$')
    # ax.plot(r_array[1:], Ftan_LL_list_azim[i][0:len(r_array)-1],'-', label=fr'$\delta_{{azim}} = 2\pi/{N}$')
    print(fr'CT for $axial convection speed = {axial}$ is {CT_LL_frozen_axial[i]}')
    print(fr'CP for $axial convection speed = {axial}$ is {CP_LL_frozen_axial[i]}')
ax.legend()
ax.set_xlabel("r/R [-]")
ax.set_ylabel(r"Angle [$\deg$]")
ax.set_title("TSR=6")
ax.grid()


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Azimuthal Discretisation Sensitivity~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# fig7 = plt.figure()
# ax = fig7.add_subplot()
# for i in range(len(resultsLL_list_azim)):
#     N = Nazim_disc[i]
#     # uncomment other lines to plot other variables, change axis labels accordingly
#     ax.plot(r_array[1:], alpha_LL_list_azim[i][0:len(r_array)-1], '-', label=fr'$\delta_{{azim}} = 2\pi/{N}$')
#     # ax.plot(r_array[1:], inflow_angle_LL_list_azim[i][0:len(r_array)-1],'-',label=fr'$\delta_{{azim}} = 2\pi/{N}$')
#     # ax.plot(r_array[1:], Gamma_LL_list_azim[i][0:len(r_array)-1],'-', label=fr'$\delta_{{azim}} = 2\pi/{N}$')
#     # ax.plot(r_array[1:], Fnorm_LL_list_azim[i][0:len(r_array)-1],'-',label=fr'$\delta_{{azim}} = 2\pi/{N}$')
#     # ax.plot(r_array[1:], Ftan_LL_list_azim[i][0:len(r_array)-1],'-', label=fr'$\delta_{{azim}} = 2\pi/{N}$')
#     print(fr'CT for $\delta_{{azim}} = 2\pi/{N}$ is {CT_LL_list_azim[i]}')
#     print(fr'CP for $\delta_{{azim}} = 2\pi/{N}$ is {CP_LL_list_azim[i]}')
# ax.legend()
# ax.set_xlabel("r/R [-]")
# ax.set_ylabel(r"Angle [$\deg$]")
# ax.set_title("TSR=6")
# ax.grid()


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Wake Length Sensitivity~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#fig8 = plt.figure()
#ax = fig8.add_subplot()
#for i in range(len(resultsLL_list_Nrot)):
    #N = Nrotations[i]
    ## uncomment other lines to plot other variables, change axis labels accordingly
    #ax.plot(r_array[1:], alpha_LL_list_Nrot[i][0:len(r_array)-1], '-', label=fr'Nrot = {N}$')
    ## ax.plot(r_array[1:], inflow_angle_LL_list_Nrot[i][0:len(r_array)-1],'-',label=fr'Nrot = {N}$')
    ## ax.plot(r_array[1:], Gamma_LL_list_Nrot[i][0:len(r_array)-1],'-', label=fr'Nrot = {N}$')
    ## ax.plot(r_array[1:], Fnorm_LL_list_Nrot[i][0:len(r_array)-1],'-',label=fr'Nrot = {N}$')
    ## ax.plot(r_array[1:], Ftan_LL_list_Nrot[i][0:len(r_array)-1],'-', label=fr'Nrot = {N}$')
    #print(fr'CT for Nrot = {N}$ is {CT_LL_list_Nrot[i]}')
    #print(fr'CP for Nrot = {N}$ is {CP_LL_list_Nrot[i]}')
#ax.legend()
#ax.set_xlabel("r/R [-]")
#ax.set_ylabel(r"Angle [$\deg$]")
#ax.set_title("TSR=6")
#ax.grid()


## CT and CP convergence
#fig9 = plt.figure()
#ax = fig9.add_subplot()
#ax.plot(Nrotations, CT_LL_list_Nrot, '.')
#ax.set_xlabel("Number of wake length [rotations]")
#ax.set_ylabel(r"CT [-]")
#ax.set_title("TSR=6")
#ax.grid()

#fig10 = plt.figure()
#ax = fig10.add_subplot()
#ax.plot(Nrotations, CP_LL_list_Nrot, '.')
#ax.set_xlabel("Number of wake length [rotations]")
#ax.set_ylabel(r"CP [-]")
#ax.set_title("TSR=6")
#ax.grid()
plt.show()

















