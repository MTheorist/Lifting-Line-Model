import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import os
from itertools import cycle 

plt.rc('text', usetex=True) 
plt.rc('font', family='serif')
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 15

os.chdir(os.path.dirname(__file__))

#---------------------Blade Discretisation---------------------
def BladeSegment(root_pos_R, tip_pos_R, pitch, nodes, seg_type='lin'):
    if seg_type=='lin':
        r_R = np.linspace(root_pos_R, tip_pos_R, nodes)

    if seg_type=='cos':
        theta = np.linspace(0,np.pi, nodes)
        r_R = (1/2)*(tip_pos_R-root_pos_R)*(1+np.cos(theta)) + root_pos_R
        r_R = np.flip(r_R)
        
    chord_dist = 0.18 - 0.06*(r_R)                  # normalised chord distribution
    twist_dist = -50*(r_R) + 35 + pitch             # twist distribution [deg]

    return r_R, chord_dist, twist_dist

#---------------------Blade Element Method---------------------
def PrandtlCorrections(Nb, r, R, TSR, a, root_pos_R, tip_pos_R):
    F_tip = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((tip_pos_R-(r/R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2))))))
    F_root = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2)))))  
    F_tot = F_tip*F_root
    
    if(F_tot == 0) or (F_tot == np.nan) or (F_tot == np.inf):
        # handle exceptional cases for 0/NaN/inf value of F_tot
        # print("F total is 0/NaN/inf.")
        F_tot = 0.00001

    return F_tot, F_tip, F_root

def BladeElementMethod(Vinf, TSR, n, rho, b, r, root_pos_R, tip_pos_R, dr, Omega, Nb, a, a_tan, twist, chord, polar_alfa, polar_cl, polar_cd, tol, P_up):
    flag = 0
    while True and (flag<1000):
            V_ax = Vinf*(1+a)       # axial velocity at the propeller blade
            V_tan = Omega*r*(1-a_tan)   # tangential veloity at the propeller blade
            V_loc = np.sqrt(V_ax**2 + V_tan**2)

            phi = np.arctan(V_ax/V_tan)     # inflow angle [rad]
            alfa = twist - np.rad2deg(phi)  # local angle of attack [deg]
            
            Cl = np.interp(alfa, polar_alfa, polar_cl)
            Cd = np.interp(alfa, polar_alfa, polar_cd)
            
            C_ax = Cl*np.cos(phi) - Cd*np.sin(phi)      # axial force coefficient
            F_ax = (0.5*rho*V_loc**2)*C_ax*chord        # axial force [N/m]

            C_tan = Cl*np.sin(phi) + Cd*np.cos(phi)     # tangential force coefficient
            F_tan = (0.5*rho*V_loc**2)*C_tan*chord      # tangential force [N/m]
           
            dCT = (F_ax*Nb*dr)/(rho*(n**2)*(2*b)**4)        # blade element thrust coefficient                   
            dCQ = (F_tan*Nb*r*dr)/(rho*(n**2)*(2*b)**5)     # blade element torque coefficient
            dCP = (F_ax*Nb*dr*Vinf)/(rho*(n**3)*(2*b)**5)   # blade element power coefficient
            
            a_new = ((1/2)*(-1+np.sqrt(1+(F_ax * Nb / (rho * Vinf**2 * np.pi * r)))))
            a_tan_new = F_tan * Nb / (2*rho*(2*np.pi*r)*Vinf*(1+a_new)*Omega*r)
            
            a_b4_Pr = a_new
            
            F_tot, F_tip, F_root = PrandtlCorrections(Nb, r, b, TSR, a, root_pos_R, tip_pos_R)
            a_new = a_new/F_tot
            a_tan_new=a_tan_new/F_tot
        
            if(np.abs(a-a_new)<tol) and (np.abs(a_tan-a_tan_new)<tol):
                a = a_new
                a_tan = a_tan_new
                flag += 1
                break
            else:
                # introduces relaxation to induction factors a and a' for easier convergence
                a = 0.75*a + 0.25*a_new
                a_tan = 0.75*a_tan + 0.25*a_tan_new
                flag += 1
                continue
    P0_down = P_up + F_ax*dr/(2*np.pi*r)
    Gamma = 0.5*(Vinf*(1+a))*Cl*chord
    return a_b4_Pr, a, a_tan, Cl, Cd, F_ax, F_tan, alfa, phi, F_tot, F_tip, F_root, dCT, dCQ, dCP, P0_down, Gamma

#----------------------Lifting Line Model----------------------
def ControlPoint(r_R, b, blade_seg, chord_dist, twist_dist):
    mlt = 1/2       # length normalised distance of control point from origin of blade segment

    CtrlPts = []
    for j in range(blade_seg):
        CP_temp = {}
        y = (r_R[j+1]+r_R[j])*b*mlt
        x = 0.25*(np.interp(y/b, r_R, chord_dist)*b)*np.sin(np.interp(y/b, r_R, twist_dist))
        z = 0.25*(np.interp(y/b, r_R, chord_dist)*b)*np.cos(np.interp(y/b, r_R, twist_dist))
        CP_temp = {'CP'+str(j+1): [x, y, z]}
        CtrlPts.append(CP_temp)

    return CtrlPts

def CoordinateRotation(HS_vortex, blade_seg, Nb):
    HS_base = HS_vortex.copy()

    for i in range(1, Nb):
        theta = ((2*np.pi)/Nb) * i    # angular position of blade element
        
        for j in range(blade_seg):
            HS_temp={}
            for key, VF in HS_base[j].items():
                x = [VF['pos1'][0], VF['pos2'][0]]
                y = [VF['pos1'][1], VF['pos2'][1]]
                z = [VF['pos1'][2], VF['pos2'][2]]

                R = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta),  np.cos(theta)]
                ])                      # transformation matrix

                pos = np.array([x, y, z])                
                pos = np.dot(R, pos)

                HS_temp[key] = {'pos1': [pos[0][0], pos[1][0], pos[2][0]], 'pos2': [pos[0][1], pos[1][1], pos[2][1]], 'Gamma': VF['Gamma']}

            HS_vortex.append(HS_temp)
    
    return HS_vortex

def HorseshoeVortex(l, U_wake, vor_fil, blade_seg, Omega, r_R, Gamma, Nb, chord_dist, twist_dist, b):
    T = l/U_wake        # total time for wake propagation [s]
    dt = T/vor_fil      # time for propagation of each vortex filament [s]    
    rot = False         # vortex filament rotation flag
    
    HS_vortex = []
    for j in range(blade_seg):
        TR_left = {}
        BVor = {}
        TR_right = {}
        HS_temp = {}
        for i in range(vor_fil):
            # bound vortex coordinates
            if i == 0:
                x = [(0.25*chord_dist[j]*np.sin(twist_dist[j])), (0.25*chord_dist[j+1]*np.sin(twist_dist[j+1]))]
                y = [r_R[j]*b, r_R[j+1]*b]
                z = [(0.25*chord_dist[j]*np.cos(twist_dist[j])), (0.25*chord_dist[j+1]*np.cos(twist_dist[j+1]))]
                BVor['VF'+str(vor_fil+1)]={'pos1': [x[0], y[0], z[0]], 'pos2':[x[1], y[1], z[1]], 'Gamma': Gamma[j]}

            # first set of trailing vortices
            if rot == False:
                x = [[(1.25*chord_dist[j]*np.sin(twist_dist[j])), (0.25*chord_dist[j]*np.sin(twist_dist[j]))],
                     [(0.25*chord_dist[j+1]*np.sin(twist_dist[j+1])), (1.25*chord_dist[j+1]*np.sin(twist_dist[j+1]))]]
                y = [r_R[j]*b, r_R[j+1]*b]
                z = [[(1.25*chord_dist[j]*np.cos(twist_dist[j])), (0.25*chord_dist[j]*np.cos(twist_dist[j]))],
                     [(0.25*chord_dist[j+1]*np.cos(twist_dist[j+1])), (1.25*chord_dist[j+1]*np.cos(twist_dist[j+1]))]]
                TR_left['VF'+str(vor_fil)] = {'pos1': [x[0][0], y[0], z[0][0]], 'pos2':[x[0][1], y[0], z[0][1]], 'Gamma': Gamma[j]}
                TR_right['VF'+str(vor_fil+2)] = {'pos1': [x[1][0], y[1], z[1][0]], 'pos2':[x[1][1], y[1], z[1][1]], 'Gamma': Gamma[j]}
                rot = True
            
            # subsequent set of vortex filaments
            elif rot == True:
                # left side of the trailing vortex
                x = [U_wake*dt*(i+1)]
                y = [r_R[j]*b*np.cos(Omega*dt*(i+1))]
                z = [(r_R[j]*b*np.sin(Omega*dt*(i+1)))]

                TR_left['VF'+str(vor_fil-i)] = {'pos1': [x[0], y[0], z[0]], 'pos2':TR_left['VF'+str((vor_fil+1)-i)]['pos1'], 'Gamma': Gamma[j]}

                # right side of the trailing vortex
                x = [U_wake*dt*(i+1)]
                y = [r_R[j+1]*b*np.cos(Omega*dt*(i+1))]
                z = [(r_R[j+1]*b*np.sin(Omega*dt*(i+1)))]

                TR_right['VF'+str((vor_fil+2)+i)] = {'pos1': TR_right['VF'+str((vor_fil+1)+i)]['pos2'], 'pos2':[x[0], y[0], z[0]], 'Gamma': Gamma[j]}
            
        TR_left = dict(reversed(list(TR_left.items())))
        HS_temp = TR_left | BVor | TR_right
        HS_vortex.append(HS_temp)   
        x = y = z = 0
        rot = False
    
    HS_vortex = CoordinateRotation(HS_vortex, blade_seg, Nb)

    return HS_vortex

def InducedVelocities(CtrlPts, pos1, pos2, gamma, tol=1e-6):
    """
    Function to calculate [u,v,w] induced by a vortex filament defined by [pos1, pos2] on a control point defined by CtrlPts.
    Input Arguments:-
        CtrlPts: [xp, yp, zp]; 1D array of control point coordinates
        pos1: [x1, y1, z1]; 1D array of the start position of the vortex filament
        pos2: [x2, y2, z2]; 1D array of the end position of the vortex filament
        gamma: int or float; magnitude of circulation around the filament
    """
    r1 = np.sqrt((CtrlPts[0]-pos1[0])**2 + (CtrlPts[1]-pos1[1])**2 + (CtrlPts[2]-pos1[2])**2)
    r2 = np.sqrt((CtrlPts[0]-pos2[0])**2 + (CtrlPts[1]-pos2[1])**2 + (CtrlPts[2]-pos2[2])**2)

    r12x = (CtrlPts[1]-pos1[1])*(CtrlPts[2]-pos2[2]) - (CtrlPts[2]-pos1[2])*(CtrlPts[1]-pos2[1])
    r12y = -(CtrlPts[0]-pos1[0])*(CtrlPts[2]-pos2[2]) + (CtrlPts[2]-pos1[2])*(CtrlPts[0]-pos2[0])
    r12z = (CtrlPts[0]-pos1[0])*(CtrlPts[1]-pos2[1]) - (CtrlPts[1]-pos1[1])*(CtrlPts[0]-pos2[0])

    r12sq = (r12x**2) + (r12y**2) + (r12z**2)
    
    r01 = (pos2[0]-pos1[0])*(CtrlPts[0]-pos1[0]) + (pos2[1]-pos1[1])*(CtrlPts[1]-pos1[1]) + (pos2[2]-pos1[2])*(CtrlPts[2]-pos1[2])
    r02 = (pos2[0]-pos1[0])*(CtrlPts[0]-pos2[0]) + (pos2[1]-pos1[1])*(CtrlPts[1]-pos2[1]) + (pos2[2]-pos1[2])*(CtrlPts[2]-pos2[2])
    
    # use tol = [1e-6, 5e-6] if segmentation is ['lin', 'cos']
    if r12sq < tol:
        r12sq = tol
        return 0.0, 0.0, 0.0

    K = (gamma/(4*np.pi*r12sq))*((r01/r1) - (r02/r2))
    
    U = K*r12x
    V = K*r12y
    W = K*r12z

    return U, V, W

def InfluenceCoeff(HS_vortex, CtrlPts, vor_fil, Nb):
    N_cp = len(CtrlPts)      #number of control points
    N_hs = len(HS_vortex)    #number of horseshoe vortices

    #Initialising the matrices 
    u_infl = np.zeros((N_cp,N_cp))
    v_infl = np.zeros((N_cp,N_cp))
    w_infl = np.zeros((N_cp,N_cp))

    for i in range(N_cp):   # iterate for i-th collocation point on the base blade
        for j in range(N_cp):   # iterate for effect of j-th HS vortex on i-th collocation point
            u_ind_t = 0
            v_ind_t = 0
            w_ind_t = 0
            
            for k in range(j, N_hs, N_cp):     # iterate for effect of j-th HS vortex existing over the k-th blade
                hs = HS_vortex[k]

                for vf_key in hs:
                    fil = hs[vf_key]

                    u_fil, v_fil, w_fil = InducedVelocities(CtrlPts[i]['CP'+str(i+1)], fil['pos1'], fil['pos2'], fil['Gamma'])
                    
                    u_ind_t += u_fil
                    v_ind_t += v_fil
                    w_ind_t += w_fil
                
            u_infl[i][j] = u_ind_t
            v_infl[i][j] = v_ind_t
            w_infl[i][j] = w_ind_t

    return u_infl, v_infl, w_infl

def LiftingLineModel(HS_vortex, CtrlPts, polar_alfa, polar_cl, polar_cd, Vinf, Omega, rho, b, r_R, chord_dist, twist_dist, Nb, l, U_wake, vor_fil, gamma, a_bem):
    N_cp = len(CtrlPts)         # number of control points
    
    gamma_new, alfa, phi, a_ax_loc, a_tan_loc, F_ax_l, F_tan_l, r_cp, dCT, dCQ, dCP, seg_len = [np.zeros(N_cp) for i in range(12)]
    results = {'r':[], 'a_ax':[], 'a_tan':[], 'alfa':[], 'phi':[], 'F_ax':[], 'F_tan':[], 'CT': 0.0, 'CQ': 0.0, 'CP': 0.0, 'Gamma':[], 'iterations':0}

    u_infl, v_infl, w_infl = InfluenceCoeff(HS_vortex, CtrlPts, vor_fil, Nb)

    conv = 1e-6
    max_iter = 1000
    relax = 0.3     # use [0.3, 0.1] if segmentation is ['lin','cos']
    iter = 0
    error = 1e8

    while error>conv and iter<max_iter:
        print(gamma, iter)

        for i in range(N_cp):
            xp = CtrlPts[i]['CP'+str(i+1)][0]
            yp = CtrlPts[i]['CP'+str(i+1)][1]
            zp = CtrlPts[i]['CP'+str(i+1)][2] 
            r_cp[i] = np.sqrt(yp**2 + zp**2)

            seg_len[i] = (r_R[i+1]-r_R[i])*b

            # Solving the linear system of equations
            u_ind = np.dot(u_infl[i],gamma)
            v_ind = np.dot(v_infl[i],gamma)
            w_ind = np.dot(w_infl[i],gamma)

            # Finding the local velocity components at the control points
            V_ax_local = Vinf + u_ind   
            
            V_tan_local = (Omega*r_cp[i]) + ((1/r_cp[i])*(-(v_ind*zp)+(w_ind*yp)))
            V_local_mag = np.sqrt((V_ax_local**2) + (V_tan_local**2))

            phi[i] = np.arctan(V_ax_local/V_tan_local)
            
            # Finding the blade element properties
            r_R_local = (yp/b)
            chord_local = np.interp(r_R_local, r_R, chord_dist) * b
            twist_local = np.interp(r_R_local, r_R, twist_dist)
            
            alfa[i] = twist_local - np.rad2deg(phi[i])

            Cl_local = np.interp(alfa[i], polar_alfa, polar_cl)
            Cd_local = np.interp(alfa[i], polar_alfa, polar_cd)

            Lift_loc = 0.5 * rho * (V_local_mag**2) * Cl_local * chord_local
            Drag_loc = 0.5 * rho * (V_local_mag**2) * Cd_local * chord_local 

            F_ax_loc = (Lift_loc * np.cos(phi[i])) - (Drag_loc * np.sin(phi[i]))
            F_tan_loc = (Lift_loc * np.sin(phi[i])) + (Drag_loc * np.cos(phi[i]))

            gamma_new[i] = 0.5 * V_local_mag * Cl_local * chord_local 

            a_ax_loc[i] = (V_ax_local/Vinf) - 1
            a_tan_loc[i] = 1 - ((V_tan_local)/(Omega * yp))

            F_ax_l[i] = F_ax_loc
            F_tan_l[i] = F_tan_loc

            dCT[i] = (F_ax_l[i]*Nb*seg_len[i])/(rho*((Omega/(2*np.pi))**2)*(2*b)**4)
            dCQ[i] = (F_tan_l[i]*Nb*r_cp[i]*seg_len[i])/(rho*((Omega/(2*np.pi))**2)*(2*b)**5)
            dCP[i] = (F_ax_l[i]*Nb*seg_len[i]*Vinf)/(rho*((Omega/(2*np.pi))**3)*(2*b)**5)
            
        a_avg = np.dot(a_ax_loc, (np.pi*((b*r_R[1:])**2-(b*r_R[:-1])**2)))/(np.pi*((b*r_R[N_cp])**2 - (b*r_R[0])**2))
        a_bem = (a_avg*relax) + ((1-relax)*a_bem)

        error = np.max(np.abs(gamma_new-gamma))
        if error>conv and iter<max_iter:
            gamma = (gamma_new*relax) + ((1-relax)*gamma)
            HS_vortex = HorseshoeVortex(l, (Vinf*(1+a_bem)), vor_fil, N_cp, Omega, r_R, np.ones(N_cp), Nb, (chord_dist*b), np.deg2rad(twist_dist), b)
            u_infl, v_infl, w_infl = InfluenceCoeff(HS_vortex, CtrlPts, vor_fil, Nb)
            iter += 1
        else:
            results['r'] = r_cp
            results['a_ax'] = a_ax_loc
            results['a_tan'] = a_tan_loc
            results['alfa'] = alfa
            results['phi'] = phi
            results['F_ax'] = F_ax_l
            results['F_tan'] = F_tan_l
            results['CT'] = np.sum(dCT)
            results['CQ'] = np.sum(dCQ)
            results['CP'] = np.sum(dCP)
            results['Gamma'] = gamma
            results['iterations'] = iter
            return results

#--------------------------------- MAIN ---------------------------------
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
# J = np.array([2.0])

# Blade geometry
Nb = 6                  # number of blades
b = 0.7                 # Blade radius [m] (or blade span)
root_pos_R = 0.25       # normalised blade root position (r_root/R)
tip_pos_R = 1           # normalised blade tip position (r_tip/R)
pitch = 46              # blade pitch [deg]

# Discretisation 
blade_seg = 14      # no. of segments for the wing
vor_fil = 100       # no. of vortex filaments
l = 4*(2*b)         # length scale of the trailing vortices [m] (based on blade diameter)
seg_type = 'lin'    # discretisation type- 'lin': linear | 'cos': cosine

# Dependent variables 
n = Vinf/(2*J*b)    # RPS [Hz]
Omega = 2*np.pi*n   # Angular velocity [rad/s]
TSR = np.pi/J       # tip speed ratio

# %%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLVING FOR BLADE ELEMENT MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Discretisation into blade elements for BEM
r_R, chord_dist, twist_dist = BladeSegment(root_pos_R, tip_pos_R, pitch, 100, 'lin')
radial_positions_bem = (r_R[:-1] + r_R[1:]) / 2 # Midpoint of each blade element

# Iteration inputs
tol = 1e-7  # convergence tolerance

# Variable initialisation
CT, CP, CQ = [np.zeros(len(J)) for i in range(3)]
a_b4_Pr, a = [(np.ones((len(J),len(r_R)-1))*(1/3)) for i in range(2)]
chord, a_tan, Cl, Cd, F_ax, F_tan, dCT, dCQ, dCP, alfa, phi, F_tot, F_tip, F_root, P0_down, Gamma = [np.zeros((len(J),len(r_R)-1)) for i in range(16)]

P_up = np.ones((len(J),len(r_R)-1))*(Pamb + 0.5*rho*(Vinf**2))  # pressure upwind of rotor [Pa]

# Solving Blade Element Method
for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord[j][i] = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist) * b
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        # print("BEM:", (r_R[i]+r_R[i+1])/2, ":", twist)
        r = (r_R[i+1]+r_R[i])*(b/2)     # radial distance of the blade element
        dr = (r_R[i+1]-r_R[i])*b        # length of the blade element
        
        a_b4_Pr[j][i], a[j][i], a_tan[j][i], Cl[j][i], Cd[j][i], F_ax[j][i], F_tan[j][i], alfa[j][i], phi[j][i], F_tot[j][i], F_tip[j][i], F_root[j][i], dCT[j][i], dCQ[j][i], dCP[j][i], P0_down[j][i], Gamma[j][i] = \
        BladeElementMethod(Vinf, TSR[j], n[j], rho, b, r, root_pos_R, tip_pos_R, dr, Omega[j], Nb, a[j][i], a_tan[j][i], twist, chord[j][i], polar_alfa, polar_cl, polar_cd, tol, P_up[j][i])

        CT[j] += dCT[j][i]    # thrust coefficient for given J
        CP[j] += dCP[j][i]    # power coefficient for given J
        CQ[j] += dCQ[j][i]    # torque coefficient for given J

a_avg = np.dot(a, (np.pi*((b*r_R[1:])**2-(b*r_R[:-1])**2)))/(np.pi*((b*r_R[len(r_R)-1])**2 - (b*r_R[0])**2))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLVING FOR LIFTING LINE MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Discretisation into blade elements for LLM
r_R, chord_dist, twist_dist = BladeSegment(root_pos_R, tip_pos_R, pitch, (blade_seg+1), seg_type)
radial_positions_llm = (r_R[:-1] + r_R[1:]) / 2 # Radial positions from LLM control points

U_wake = Vinf*(np.ones(len(a_avg))+a_avg)       # wake velocity [m/s]


# Initial guess for gamma using BEM
Gamma_init = np.zeros((len(J),len(r_R)-1))
for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord[j][i] = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist) * b
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        # print("BEM:", (r_R[i]+r_R[i+1])/2, ":", twist)
        r = (r_R[i+1]+r_R[i])*(b/2)     # radial distance of the blade element
        dr = (r_R[i+1]-r_R[i])*b        # length of the blade element
        
        Gamma_init[j][i] = BladeElementMethod(Vinf, TSR[j], n[j], rho, b, r, root_pos_R, tip_pos_R, dr, Omega[j], Nb, a[j][i], a_tan[j][i], twist, chord[j][i], polar_alfa, polar_cl, polar_cd, tol, P_up[j][i])[16]

CtrlPts, HS_vortex, results = [[] for i in range(3)]

# Solving Lifting Line Model

for i in range(len(U_wake)):
    CtrlPts.append(ControlPoint(r_R, b, blade_seg, chord_dist, np.deg2rad(twist_dist)))
    HS_vortex.append(HorseshoeVortex(l, U_wake[i], vor_fil, blade_seg, Omega[i], r_R, np.ones(blade_seg), Nb, (chord_dist*b), np.deg2rad(twist_dist), b))
    results.append(LiftingLineModel(HS_vortex[i], CtrlPts[i], polar_alfa, polar_cl, polar_cd, Vinf, Omega[i], rho, b, r_R, chord_dist, twist_dist, Nb, l, U_wake[i], vor_fil, Gamma_init[i], a_avg[i]))

### --------------------------------- PLOTTING ROUTINE ---------------------------------
    
## Per advance ratio, J
 
    # Wake Model
    fig = plt.figure("Wake Model for J = " + str(J[i]))
    ax = fig.add_subplot(111, projection="3d")
    colors = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red", 
                    "tab:purple", "tab:brown", "tab:pink", "tab:gray", 
                    "tab:olive", "tab:cyan"])

    for blade_idx in range(Nb * blade_seg):
        hs = HS_vortex[i][blade_idx]  # You probably visualized only one advance ratio's wake
        color = next(colors)
        
        nodes = []
        for vf_key in sorted(hs, key=lambda k: int("".join(filter(str.isdigit, k)))):
            vf = hs[vf_key]
            if not nodes:
                nodes.append(vf["pos1"])
            nodes.append(vf["pos2"])
        
        nodes = np.array(nodes)
        ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color=color)

    for j in range(blade_seg):
        pt = CtrlPts[i][j]['CP'+str(j+1)]
        ax.plot([pt[0]], [pt[1]], [pt[2]], marker='o', markersize=2)

    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_zlabel("$z$ [m]")
    ax.set_title("Wake Model for $J$ = " + str(J[i]))
    ax.set_box_aspect([1, 0.5, 0.5])
    plt.tight_layout()
    
    # 1. Radial distribution of the angle of attack
    plt.figure("Radial distribution of ALFA at J = " + str(J[i]))
    plt.plot(radial_positions_bem, alfa[i], label='BEM')
    plt.plot(radial_positions_llm, results[i]['alfa'], label='LLM', marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\\alpha$ [deg]')
    plt.title("Radial distribution of $\\alpha$ at J = " + str(J[i]))
    plt.grid(True)
    plt.legend()

    # 2. Radial distribution of the inflow angle
    plt.figure("Radial distribution of PHI at J = " + str(J[i]))
    plt.plot(radial_positions_bem, np.rad2deg(phi[i]), label='BEM')
    plt.plot(radial_positions_llm, np.rad2deg(results[i]['phi']), label='LLM', marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\phi$ [deg]')
    plt.title("Radial distribution of $\phi$ at J = " + str(J[i]))
    plt.grid(True)
    plt.legend()

    # 3. Radial distribution of the circulation (Gamma)
    plt.figure("Radial distribution of GAMMA at J = " + str(J[i]))
    plt.plot(radial_positions_bem, Gamma[i], label='BEM')
    plt.plot(radial_positions_llm, results[i]['Gamma'], label='LLM', marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\Gamma$')
    plt.title("Radial distribution of $\Gamma$ at J = " + str(J[i]))
    plt.grid(True)
    plt.legend()

    # 4. Radial distribution of the tangential/azimuthal load
    plt.figure("Radial distribution of FTAN at J = " + str(J[i]))
    plt.plot(radial_positions_bem, F_tan[i], label='BEM')
    plt.plot(radial_positions_llm, results[i]['F_tan'], label='LLM', marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$F_{tan}$, [N/m]')
    plt.title("Radial distribution of $F_{tan}$ at J = " + str(J[i]))
    plt.grid(True)
    plt.legend()

    # 5. Radial distribution of the axial load
    plt.figure("Radial distribution of FAX at J = " + str(J[i]))
    plt.plot(radial_positions_bem, F_ax[i], label='BEM')
    plt.plot(radial_positions_llm, results[i]['F_ax'], label='LLM', marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$F_{ax}$, [N/m]')
    plt.title("Radial distribution of $F_{ax}$ at J = " + str(J[i]))
    plt.grid(True)
    plt.legend()

    # 6. Radial distribution of the axial induction
    plt.figure("Radial distribution of A at J = " + str(J[i]))
    plt.plot(radial_positions_bem, a[i], label='BEM')
    plt.plot(radial_positions_llm, results[i]['a_ax'], label='LLM', marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('Axial Induction, $a_{ax}$')
    plt.title("Radial distribution of $a_{ax}$ at J = " + str(J[i]))
    plt.grid(True)
    plt.legend()

    # 7. Radial distribution of the tangential induction
    plt.figure("Radial distribution of ATAN at J = " + str(J[i]))
    plt.plot(radial_positions_bem, a_tan[i], label='BEM')
    plt.plot(radial_positions_llm, results[i]['a_tan'], label='LLM', marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('Tangential Induction, $a_{tan}$')
    plt.title("Radial distribution of $a_{tan}$ at J = " + str(J[i]))
    plt.grid(True)
    plt.legend()

    # 8. Total thrust coefficient (CT) and power coefficient (CP)
    print(f"Total Thrust Coefficient (CT) for J={J[i]}:")
    print(f"BEM: {CT[i]:.4f}")
    print(f"LLM: {results[i]['CT']:.4f}")

    print(f"\nTotal Power Coefficient (CP) for J={J[i]}:")
    print(f"BEM: {CP[i]:.4f}")
    print(f"LLM: {results[i]['CP']:.4f}")
    
    plt.show()

## LLM plots at varying J

for i in range(len(U_wake)):

    # 1. Radial distribution of the angle of attack
    plt.figure("Radial distribution of ALFA with varying J")
    plt.plot(radial_positions_llm, results[i]['alfa'], label=('J = ' + str(J[i])), marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\\alpha$ [deg]')
    plt.title("Radial distribution of $\\alpha$")
    plt.grid(True)
    plt.legend()

    # 2. Radial distribution of the inflow angle
    plt.figure("Radial distribution of PHI with varying J")
    plt.plot(radial_positions_llm, np.rad2deg(results[i]['phi']), label=('J = ' + str(J[i])), marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\phi$ [deg]')
    plt.title("Radial distribution of $\phi$ with varying J")
    plt.grid(True)
    plt.legend()

    # 3. Radial distribution of the circulation (Gamma)
    plt.figure("Radial distribution of GAMMA with varying J")
    plt.plot(radial_positions_llm, results[i]['Gamma'], label=('J = ' + str(J[i])), marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\Gamma$')
    plt.title("Radial distribution of $\Gamma$ with vaying J")
    plt.grid(True)
    plt.legend()

    # 4. Radial distribution of the tangential/azimuthal load
    plt.figure("Radial distribution of FTAN with varying J")
    plt.plot(radial_positions_llm, results[i]['F_tan'], label=('J = ' + str(J[i])), marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$F_{tan}$, [N/m]')
    plt.title("Radial distribution of $F_{tan}$ with varying J")
    plt.grid(True)
    plt.legend()

    # 5. Radial distribution of the axial load
    plt.figure("Radial distribution of FAX with varying J")
    plt.plot(radial_positions_llm, results[i]['F_ax'], label=('J = ' + str(J[i])), marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$F_{ax}$, [N/m]')
    plt.title("Radial distribution of $F_{ax}$ with varying J")
    plt.grid(True)
    plt.legend()

    # 6. Radial distribution of the axial induction
    plt.figure("Radial distribution of A with varying J")
    plt.plot(radial_positions_llm, results[i]['a_ax'], label=('J = ' + str(J[i])), marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('Axial Induction, $a_{ax}$')
    plt.title("Radial distribution of $a_{ax}$ with varying J")
    plt.grid(True)
    plt.legend()

    # 7. Radial distribution of the tangential induction
    plt.figure("Radial distribution of ATAN with varying J")
    plt.plot(radial_positions_llm, results[i]['a_tan'], label=('J = ' + str(J[i])), marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('Tangential Induction, $a_{tan}$')
    plt.title("Radial distribution of $a_{tan}$ with varying J")
    plt.grid(True)
    plt.legend()

plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%% SENSITIVITY STUDY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a_sens = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]     # to check sensitivity to initial induction factor
U_wake_sens = Vinf*(np.ones(len(a_sens))+a_sens)
Omega_sens = (np.pi*Vinf)/(1.6*b)

seg_type_sens = ['lin', 'cos']              # to check sensitivity to blade segmentation type

vor_fil_sens = [10, 20, 50, 100]            # to check sensitivity to number of wake wortex filament

l_sens = np.array([1, 2, 4, 6, 9, 10])*(2*b)          # to check sensitivity to wake length 
iter = np.zeros(len(l_sens))

CtrlPts, HS_vortex, results = [[] for i in range(3)]

# Discretisation into blade elements for LLM
r_R, chord_dist, twist_dist = BladeSegment(root_pos_R, tip_pos_R, pitch, (blade_seg+1), seg_type_sens[0])
radial_positions_llm = (r_R[:-1] + r_R[1:]) / 2 # Radial positions from LLM control points

for i in range(len(l_sens)):
    CtrlPts.append(ControlPoint(r_R, b, blade_seg, chord_dist, np.deg2rad(twist_dist)))
    HS_vortex.append(HorseshoeVortex(l_sens[i], U_wake[0], vor_fil_sens[3], blade_seg, Omega_sens, r_R, np.ones(blade_seg), Nb, (chord_dist*b), np.deg2rad(twist_dist), b))
    results.append(LiftingLineModel(HS_vortex[i], CtrlPts[i], polar_alfa, polar_cl, polar_cd, Vinf, Omega_sens, rho, b, r_R, chord_dist, twist_dist, Nb, l_sens[i], U_wake[0], vor_fil_sens[3], np.ones(blade_seg), a_avg[0]))
    iter[i] = (results[i]['iterations'])
## LLM plots for sensitivity study

for i in range(len(l_sens)):
    lbl = "$l_{wake}$ = " + str(l_sens[i])
    # 1. Radial distribution of the angle of attack
    plt.figure("Radial distribution of ALFA")
    plt.plot(radial_positions_llm, results[i]['alfa'], label=lbl, marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\\alpha$ [deg]')
    plt.title("Radial distribution of $\\alpha$")
    plt.grid(True)
    plt.legend()

    # 2. Radial distribution of the inflow angle
    plt.figure("Radial distribution of PHI")
    plt.plot(radial_positions_llm, np.rad2deg(results[i]['phi']), label=lbl, marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\phi$ [deg]')
    plt.title("Radial distribution of $\phi$")
    plt.grid(True)
    plt.legend()

    # 3. Radial distribution of the circulation (Gamma)
    plt.figure("Radial distribution of GAMMA")
    plt.plot(radial_positions_llm, results[i]['Gamma'], label=lbl, marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$\Gamma$')
    plt.title("Radial distribution of $\Gamma$")
    plt.grid(True)
    plt.legend()

    # 4. Radial distribution of the tangential/azimuthal load
    plt.figure("Radial distribution of FTAN")
    plt.plot(radial_positions_llm, results[i]['F_tan'], label=lbl, marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$F_{tan}$, [N/m]')
    plt.title("Radial distribution of $F_{tan}$")
    plt.grid(True)
    plt.legend()

    # 5. Radial distribution of the axial load
    plt.figure("Radial distribution of FAX")
    plt.plot(radial_positions_llm, results[i]['F_ax'], label=lbl, marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('$F_{ax}$, [N/m]')
    plt.title("Radial distribution of $F_{ax}$")
    plt.grid(True)
    plt.legend()

    # 6. Radial distribution of the axial induction
    plt.figure("Radial distribution of A")
    plt.plot(radial_positions_llm, results[i]['a_ax'], label=lbl, marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('Axial Induction, $a_{ax}$')
    plt.title("Radial distribution of $a_{ax}$")
    plt.grid(True)
    plt.legend()

    # 7. Radial distribution of the tangential induction
    plt.figure("Radial distribution of ATAN")
    plt.plot(radial_positions_llm, results[i]['a_tan'], label=lbl, marker='o', markersize = 2)
    plt.xlabel('$r/R$')
    plt.ylabel('Tangential Induction, $a_{tan}$')
    plt.title("Radial distribution of $a_{tan}$")
    plt.grid(True)
    plt.legend()

# 8. Effect of wake length on convergence
plt.figure("Sensitivity to wake length")
plt.plot(iter, l_sens, marker='o', markersize = 2)
plt.xlabel('Iterations')
plt.ylabel('Wake length, $l_{wake} [m]$')
plt.title("LLM sensitivity to wake length")
plt.grid(True)
plt.legend()

plt.show()