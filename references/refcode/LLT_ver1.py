import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
#import BEM
# plt.rcParams.update({
#     "text.usetex": True,
#     #"font.family": "Helvetica"
# })
# plt.rcParams['font.size'] = 12
# #plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# %%
airfoil = 'propairfoil.csv'
data1=pd.read_csv(airfoil, header=0,
                    names = ["alfa", "cl", "cd", "cm"],  sep='\\s+')
polar_alpha = data1['alfa'][:]
polar_cl = data1['cl'][:]
polar_cd = data1['cd'][:]
# %%

def VORTXL(P0,P1,P2, circ):
    # the vortex element--r0
    r0=  P2-P1;    
    # the vortex element starting to the colloc point
    r1=  P0-P1
    # the vortex element ending to the colloc point
    r2=  P0-P2
    if (np.cross(r1, r2)@np.cross(r1, r2))==0:
        #print('xero')
        K=0
    else:
        K=(circ/(4*np.pi*(np.cross(r1, r2)@np.cross(r1, r2))))*((r0@r1/(np.sqrt(r1@r1)))-(r0@r2/(np.sqrt(r2@r2))))
    ind_Vel= K*np.cross(r1, r2)
    # u=ind_Vel[0]
    # v=ind_Vel[1]
    # w=ind_Vel[2]
    return ind_Vel

def HSHOE (P,A, B, C,D, circ ):
    #def HSHOE (x,y,z, xA,yA,zA, xB,yB,zB, xC,yC,zC, xD,yD,zD,P,A, B, C,D circ ):
    
    ind_Vel1 = VORTXL(P, A,B,circ) # left TE vortex
    ind_Vel2 = VORTXL(P,B,C, circ) # bound vortex
    ind_Vel3 = VORTXL(P,C,D, circ) # right TE vortex
    ind_Vel_te = ind_Vel1 + ind_Vel3
    ind_Vel = ind_Vel2 + ind_Vel_te
    return ind_Vel, ind_Vel_te


def polar_interp(alfa):
    cl = np.interp(alfa, polar_alpha, polar_cl)
    cd = np.interp(alfa, polar_alpha, polar_cd)
   # cl =  np.pi*2*np.sin(np.deg2rad(alfa))
    return cl, cd


def areaavg(a,Ns,r):
    a_avg=np.sum(a*(np.pi*(r[1:]**2-r[:-1]**2)))/(np.pi*(r[Ns]**2-r[0]**2))
    
    return a_avg

# %%
def coordinates(YBarr, YCarr, XBarr, XCarr,ZBarr,ZCarr,Omega, r,chord_dist, pitch_dist,Uw, dw, Nw, j ):
    
    # X_TE=1*chord_dist*np.sin(np.deg2rad(pitch_dist)) ########################
    # #Y_TE=1*Ytve
    # Z_TE= chord_dist*np.cos(np.deg2rad(pitch_dist))    ########################
    
    # X_TE_left = 1*X_TE[:-1]  #######$$$$$$$$$$need to be removed
    # X_TE_right = 1*X_TE[1:]
    
    # Y_TE_left = 1*YBarr
    # Y_TE_right = 1*YCarr
    
    # Z_TE_left = 1*Z_TE[:-1]
    # Z_TE_right = 1*Z_TE[1:]
    
    
    X_wake1 = 1.25*chord_dist*np.sin(np.deg2rad(pitch_dist))  ########################
    
    # X_wake = X_TE + (chord_dist/4)
    
    ######## Wake ###########
    X_wake=np.linspace(X_wake1,dw,Nw)  ###1st index is wake step, 2nd index is spanwise element number 0 at tip
    t = (X_wake-X_wake1)/Uw
    #t=0
    Y_wake=  r*np.cos(Omega*t) #######for now using just omega t without induction
    Z_wake= 1.25*chord_dist*np.cos(np.deg2rad(pitch_dist))  +r*np.sin(Omega*t) #######Should be changed
    
    XW_left = 1*X_wake[:,0:-1]
    XW_right  = 1*X_wake[:,1:]
    
    YW_left = 1*Y_wake[:,0:-1]
    YW_right= 1*Y_wake[:,1:]
    
    ZW_left = 1*Z_wake[:,0:-1]
    ZW_right= 1*Z_wake[:,1:]

# print(X_wake)
# XDarr = X_wake * np.ones(Ns)  
# XAarr = X_wake * np.ones(Ns)  


# YDarr = 1*YCarr  
# YAarr = 1*YBarr                  
    
    W_left=np.zeros([Nw,3])
    W_right=np.zeros([Nw,3])
    
    xB, yB, zB = XBarr[j], YBarr[j], ZBarr[j]
    B = np.array([xB, yB, zB])
    
    xC, yC, zC = XCarr[j], YCarr[j],  ZCarr[j]
    C = np.array([xC, yC, zC])

    # xTEl,yTEl,zTEl = X_TE_left[j],Y_TE_left[j],Z_TE_left[j]
    # TE_left= np.array([xTEl,yTEl,zTEl])
    
    # xTEr,yTEr,zTEr = X_TE_right[j],Y_TE_right[j],Z_TE_right[j]
    # TE_right= np.array([xTEr,yTEr,zTEr])
    for k in range(Nw):
        xWl,yWl,zWl = XW_left[k,j], YW_left[k,j], ZW_left[k,j]
        W_left[k,:]=np.array([xWl,yWl,zWl])
        
        xWr,yWr,zWr = XW_right[k,j], YW_right[k,j], ZW_right[k,j]
        W_right[k,:]=np.array([xWr,yWr,zWr])
        

    
    W_left=np.flip(W_left,axis=0)  ######## to start the points from wake at infty to TE
    #elements =np.vstack((W_left,TE_left,B,C,TE_right, W_right))
    elements =np.vstack((W_left,B,C, W_right))
    #elements =np.vstack((B,C))
            
    return elements

def rotate_elements(elements, theta):

    theta_rad=np.deg2rad(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])

    
    rotated_elements = np.dot(elements, rotation_matrix.T)
    
    return rotated_elements

def ind_mat(Ns, Xcp, Ycp, Zcp, YBarr, YCarr, XBarr, XCarr, ZBarr, ZCarr, Omega, r, chord_dist, pitch_dist, Uw, dw, Nw):
    theta = np.array([60, 120, 180, 240, 300])
    uij = np.zeros((Ns, Ns))
    vij = np.zeros((Ns, Ns))
    wij = np.zeros((Ns, Ns))

    for i in range(Ns):
        x, y, z = Xcp[i], Ycp[i], Zcp[i]
        P = np.array([x, y, z])
        
        for j in range(Ns):
            elements_og = coordinates(YBarr, YCarr, XBarr, XCarr, ZBarr, ZCarr, Omega, r, chord_dist, pitch_dist, Uw, dw, Nw, j)
            circ = 1.0
            
            sum_u, sum_v, sum_w = 0.0, 0.0, 0.0
            
            Velij = BIGHSHOE(P, elements_og, circ)
            sum_u += Velij[0]
            sum_v += Velij[1]
            sum_w += Velij[2]
            
            for angle in theta:
                elements_rot = rotate_elements(elements_og, angle)
                Velij_rot = BIGHSHOE(P, elements_rot, circ)
                
                sum_u += Velij_rot[0]
                sum_v += Velij_rot[1]
                sum_w += Velij_rot[2]

            uij[i,j] = sum_u
            vij[i,j] = sum_v
            wij[i,j] = sum_w

    return uij, vij, wij
# %%

def LLM(U_infty, Ns, uij, vij, wij, Omega, r_cp, pitch_distcp,c_arr, a_fin, a_tan_fin, Xcp, Ycp, Zcp,YBarr, YCarr,XBarr, XCarr,ZBarr, ZCarr,r,chord_dist, pitch_dist, dw, Nw):
    u_ax = U_infty*(1+a_fin)
    u_tan = Omega* r_cp* (1-a_tan_fin)
    phi1=np.rad2deg(np.arctan2(u_ax,u_tan))
    #phi1=np.rad2deg(np.arctan2(U_infty,(Omega*r_cp)))
    alfa_arr=pitch_distcp-phi1
    vel_mag=np.sqrt(u_ax**2+u_tan**2)
    cl_arr, cd_arr = polar_interp(alfa_arr)
    #print(cl_arr)
    circ_arr=0.5*c_arr*vel_mag*( cl_arr) ###2*np.pi*np.sin(np.deg2rad(5))
    niterations=100
    i=0
    a_avg = areaavg(a_fin,Ns,r)
    Uw = U_infty*(1+a_avg)

    while i < niterations: 
        ui_cp=U_infty+uij@circ_arr #U_infty*np.cos(np.deg2rad(alfa_arr))+uij@circ_arr
        vi_cp=U_infty*0+vij@circ_arr 
        wi_cp=wij@circ_arr+Omega*r_cp#U_infty*np.sin(np.deg2rad(alfa_arr))+wij@circ_arr-Omega*r_cp
        
        veli_mag=np.sqrt(ui_cp**2+vi_cp**2+wi_cp**2)
        ## section to calculate the circulation based on the induced velocity
        phi_new=np.rad2deg(np.arctan2(ui_cp,wi_cp))
        alfa_arr_new=pitch_distcp-phi_new
        cl_new, cd_new = polar_interp(alfa_arr_new)
        ###cd_new = np.interp(alfa_arr, polar_alpha, polar_cd)
        circ_new=0.5*c_arr*veli_mag*(cl_new)
        err=np.max((circ_arr-circ_new)**2)
        circ_arr=0.3*circ_new+ 0.7*circ_arr
        
        a_new = (ui_cp/U_infty) - 1
        a_avg_new = areaavg(a_new,Ns,r)
        

        #print(a_avg_new)
        
        a_avg=0.3*a_avg_new*+0.7*a_avg
        Uw = U_infty*(1+a_avg)
        #veli_mag = np.sqrt(ui_cp**2+vi_cp**2+wi_cp**2)
        uij, vij, wij = ind_mat(Ns, Xcp, Ycp, Zcp,YBarr, YCarr,XBarr, XCarr,ZBarr, ZCarr,Omega,r,chord_dist, pitch_dist, Uw, dw, Nw)
        #print(alfa_arr_new)
        i = i+1
        if (i==niterations and err>0.00001):
            print('not converged')
            #print(err)
        if err<0.00001:
           i= 1*niterations
           break
       
    ui_cp=U_infty+uij@circ_arr #U_infty*np.cos(np.deg2rad(alfa_arr))+uij@circ_arr
    vi_cp=U_infty*0+vij@circ_arr 
    wi_cp=wij@circ_arr+Omega*r_cp#U_infty*np.sin(np.deg2rad(alfa_arr))+wij@circ_arr-Omega*r_cp
        
    veli_mag=np.sqrt(ui_cp**2+vi_cp**2+wi_cp**2)
    Cl_final, Cd_final= polar_interp(alfa_arr_new) #2*np.pi*np.sin(np.deg2rad(alfa_arr))
    
    return Cl_final, ui_cp, vi_cp, wi_cp, alfa_arr_new




# %%
def BIGHSHOE(P, elements,  circ):
    indvel=np.zeros([len(elements)-1,3])
    for i in np.arange(0,len(elements[:,0])-1 ):
        #print('i=', i)
        #indvel[i]
        indvel[i]=VORTXL(P, elements[i,:], elements[i+1,:], circ)
        
    indvel_2=sum(indvel)
    indvel=indvel_2 #+indvel_1
    
    return indvel
# %%
#Debugging vortxl

# VORTXL(1, 1, 0, 2, 0, 0, 0, 0,0, 1)

# ##############debug successfull
# # %%
# # Debugging HSHOE
# v1,t=HSHOE(0.25, 0.5, 0, 100.25, 0, 0, 0.25, 0, 0, 0.25, 1, 0, 100.25, 1, 0, 1)
# print(v1)
# v2=VORTXL(0, 1, 0, 1, 0, 0, 0, 0,0, 1)+ VORTXL(0, 1, 0, 0, 0, 0, 0, 1, 0,  1)+VORTXL(0, 1,0,  0, 1, 0, 1, 1, 0, 1)
# print("/n", v2)
# %%
