import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
from LLT_ver1 import LLM
#from LLT_ver1 import HSHOE
#from LLT_ver1 import VORTXL
from LLT_ver1 import BIGHSHOE
from LLT_ver1 import polar_interp
from BEM_code_grp_18 import BEM
from LLT_ver1 import ind_mat
from LLT_ver1 import areaavg
#import matplotlib
#matplotlib.use('TkAgg')  # Use the TkAgg backend

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
# %%
Ns =10 # number of spanwise elements
## taken from previous BEM code 
airfoil = 'propairfoil.csv'
data1=pd.read_csv(airfoil, header=0,
                    names = ["alfa", "cl", "cd", "cm"],  sep='\\s+')
polar_alpha = data1['alfa'][:]
polar_cl = data1['cl'][:]
polar_cd = data1['cd'][:]

# define flow conditions
Uinf = 60 # unperturbed wind speed in m/s
Radius = 0.7

J = 1.6 #60/28
n = Uinf/(J*2*Radius)
Omega = 2*np.pi*n
TSR = Omega*Radius/Uinf # tip speed ratio
NBlades = 6

TipLocation_R =  1*Radius
RootLocation_R =  0.25*Radius

#rho = 1.007 #dependent on height
#R_mean = np.sqrt(((TipLocation_R**2) +(RootLocation_R**2))/2)
r = np.linspace(RootLocation_R, TipLocation_R,Ns+1)

# blade shape
colpitch = 46 # degrees
chord_dist = (0.18-0.06*r/Radius)*Radius# meters
pitch_dist = -50*(r/Radius) + 35+ colpitch  # degrees
solidity  = chord_dist*NBlades/(2*np.pi*r)



# %%
# # Your existing parameters
Nw=10 #number of wake points from TE
dw=10*Radius #Length of the wake 
# c = 1    # chord length
# span = (TipLocation_R-RootLocation_R) #chord_dist[0] * 42  # aspect ratio 42

# Spanwise discretization
Ytve = np.linspace(RootLocation_R, TipLocation_R, Ns+1)
YBarr = 1*Ytve[:-1] 
YCarr = 1*Ytve[1:] 
XBarr = 0.25*(chord_dist[0:-1]*np.sin(np.deg2rad(pitch_dist[0:-1]))) ########################
XCarr = 0.25*(chord_dist[1:]*np.sin(np.deg2rad(pitch_dist[1:])))    ########################
ZBarr = 0.25*(chord_dist[0:-1]*np.cos(np.deg2rad(pitch_dist[0:-1])))  ########################
ZCarr = 0.25*(chord_dist[1:]*np.cos(np.deg2rad(pitch_dist[1:])))       ########################




Ycp = (YBarr + YCarr) / 2 #Collocation point
r_cp=Ycp*1
pitch_distcp=-50*(r_cp/Radius) + 35+ colpitch   

cord_cp=(chord_dist[1:]+chord_dist[:-1])/2   ##3check using rcp if results change





Xcp = 0.25*(cord_cp*np.sin(np.deg2rad(pitch_distcp)))# 1*XBarr #Collocation point ####################
#Xcp=(XBarr+XCarr)/2
 

Zcp= 0.25*cord_cp*np.cos(np.deg2rad(pitch_distcp))          ########################
#Zcp=(ZBarr+ZCarr)/2
XBarr = 1*Xcp ########################
XCarr = 1*Xcp   ########################
ZBarr = 1*Zcp  ########################
ZCarr = 1*Zcp       ########################

# %%
# print(f"Bound vortex B-C at x = {chord_dist/4}")
# print(f"Wake vortices A and D at x = {X_wake} (c/4 downstream from T.E.)")
# %%

# U_infty=Uinf
# c_arr=cord_cp
# u_ax = U_infty*(1+a_fin)
# u_tan = Omega* r_cp* (1-a_tan_fin)
# phi1=np.rad2deg(np.arctan2(u_ax,u_tan))
# #phi1=np.rad2deg(np.arctan2(U_infty,(Omega*r_cp)))
# alfa_arr=pitch_distcp-phi1
# vel_mag=np.sqrt(u_ax**2+u_tan**2)
# cl_arr, cd_arr = polar_interp(alfa_arr)
# #print(cl_arr)
# circ_arr=0.5*c_arr*vel_mag*( cl_arr) ###2*np.pi*np.sin(np.deg2rad(5))
# niterations=40
# i=0
# # %%
# Uw = Uinf*(1+a_avg)
# # aaaaaa=coordinates(YBarr, YCarr,XBarr, XCarr,ZBarr,ZCarr, Omega,chord_dist, pitch_dist,Uw, dw, Nw, 0)
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')

# # # Plot
# # ax.scatter(aaaaaa[:,0], aaaaaa[:,1], aaaaaa[:,2])
# # plt.ylabel('y')
# # plt.show(block=True)
#uij, vij, wij = ind_mat(Ns, Xcp, Ycp, Zcp,YBarr, YCarr,XBarr, XCarr,ZBarr, ZCarr,chord_dist, pitch_dist, Uw, dw, Nw)


# while i < niterations: 
#     ui_cp=U_infty+uij@circ_arr #U_infty*np.cos(np.deg2rad(alfa_arr))+uij@circ_arr
#     vi_cp=U_infty*0+vij@circ_arr 
#     wi_cp=wij@circ_arr+Omega*r_cp#U_infty*np.sin(np.deg2rad(alfa_arr))+wij@circ_arr-Omega*r_cp
    
#     veli_mag=np.sqrt(ui_cp**2+vi_cp**2+wi_cp**2)
#     ## section to calculate the circulation based on the induced velocity
#     phi_new=np.rad2deg(np.arctan2(ui_cp,wi_cp))
#     alfa_arr_new=pitch_distcp-phi_new
#     cl_new, cd_new = polar_interp(alfa_arr_new)
#     ###cd_new = np.interp(alfa_arr, polar_alpha, polar_cd)
#     circ_new=0.5*c_arr*veli_mag*(cl_new)
#     err=np.max((circ_arr-circ_new)**2)
#     circ_arr=0.3*circ_new+ 0.7*circ_arr
#     i=i+1
#     if (i==niterations and err>0.00001):
#         print('not converged')
#         #print(err)
#     if err<0.00001:
#        i= 1*niterations
#        #print(err)
#        break

# Cl_final, Cd_final= polar_interp(alfa_arr_new) #2*np.pi*np.sin(np.deg2rad(alfa_arr))

# %%
C_T_standard3, C_q_standard3, a_fin, a_tan_fin, prandtl_fin3, AoA3, phi_fin3, vmag_3, Norm_load3, Tang_load3, gamma3=BEM(r/Radius, NBlades, Omega, Radius, Uinf, TipLocation_R/Radius, RootLocation_R/Radius, TSR, corr='true')

#a_avg = areaavg(a_fin, Ns, r)

# a_fin = np.zeros(Ns)
# a_tan_fin = np.zeros(Ns)
a_avg = areaavg(a_fin, Ns, r)
#np.sum(a_fin*(np.pi*(r[1:]**2-r[:-1]**2)))/(np.pi*(TipLocation_R**2-RootLocation_R**2))

Uw = Uinf*(1+a_avg)
uij, vij, wij = ind_mat(Ns, Xcp, Ycp, Zcp,YBarr, YCarr,XBarr, XCarr,ZBarr, ZCarr,Omega,r,chord_dist, pitch_dist, Uw, dw, Nw)


Cl_final, ui_cp, vi_cp, wi_cp, alfa_arr_final=LLM(Uinf, Ns, uij, vij, wij, Omega, r_cp, pitch_distcp,cord_cp, a_fin, a_tan_fin, Xcp, Ycp, Zcp,YBarr, YCarr,XBarr, XCarr,ZBarr, ZCarr,r,chord_dist, pitch_dist, dw, Nw)

    
    


# %%

#Cl_final, ui_cp, vi_cp, wi_cp, alfa_arr_final=LLM(Uinf, Ns, uij, vij, wij, Omega, r_cp, pitch_distcp,cord_cp, a_fin, a_tan_fin)
#alfa_arr_final=1*alfa_arr_new
plt.ion()
plt.figure(num=1);
plt.clf()
#fig1 = plt.figure(figsize=(12, 6))
plt.plot(Ycp/TipLocation_R, alfa_arr_final, 'bo-')
plt.plot(r_cp/Radius, AoA3, 'rs-')

# plt.xlim([0,1])
# plt.ylim([0,0.55])
plt.xlabel('s/(c*AR)')
plt.ylabel('alfa')
plt.grid(1)
plt.show(block=True)
