import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import os
from itertools import cycle 

os.chdir(os.path.dirname(__file__))

def PrandtlCorrections(Nb, r, R, TSR, a, root_pos_R, tip_pos_R):
    F_tip = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((tip_pos_R-(r/R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2))))))
    F_root = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2)))))  
    F_tot = F_tip*F_root
    
    if(F_tot == 0) or (F_tot is np.nan) or (F_tot is np.inf):
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
    return a_b4_Pr, a, a_tan, Cl, Cd, F_ax, F_tan, alfa, phi, F_tot, F_tip, F_root, dCT, dCQ, dCP, P0_down

def BladeSegment(root_pos_R, tip_pos_R, pitch, nodes, seg_type='lin'):
    if seg_type=='lin':
        r_R = np.linspace(root_pos_R, tip_pos_R, nodes)

    if seg_type=='cos':
        theta = np.linspace(0,np.pi, nodes)
        r_R = (1/2)*(tip_pos_R-root_pos_R)*(1+np.cos(theta)) + root_pos_R
        r_R = np.flip(r_R)
        
    chord_dist = 0.18 - 0.06*(r_R)                  # chord distribution [m]
    twist_dist = -50*(r_R) + 35 + pitch             # twist distribution [deg]

    return r_R, chord_dist, twist_dist

def ControlPoint(r_R, b, blade_seg):
    mlt = 0.5       # length normalised distance of control point from origin of blade segment

    CtrlPts = []
    for j in range(blade_seg):
        x = 1e-6
        y = (-1)*((r_R[j]*b)+(r_R[j+1]-r_R[j])*mlt*b)
        z = 0

        CtrlPts.append({'CP'+str(j+1): [x, y, z]})

    return CtrlPts

def HorseshoeVortex(l, U_wake, vor_fil, blade_seg, Omega, r_R):
    T = l/U_wake        # total time for wake propagation [s]
    dt = T/vor_fil      # time for propagation of each vortex filament [s]    
    rot = False         # vortex filament rotation flag
    
    def CorrectOverlapGamma(HS_vortex):
        for j in range(blade_seg-1):
            for i in range(vor_fil):
                HS1_pos1 = HS_vortex[j]['VF'+str((vor_fil+2)+i)]['pos1']
                HS1_pos2 = HS_vortex[j]['VF'+str((vor_fil+2)+i)]['pos2']
                HS2_pos1 = HS_vortex[j+1]['VF'+str(vor_fil-i)]['pos1']
                HS2_pos2 = HS_vortex[j+1]['VF'+str(vor_fil-i)]['pos2']

                if HS1_pos1 == HS2_pos2 and HS1_pos2 == HS2_pos1:
                    Gamma = np.abs(HS_vortex[j]['VF'+str((vor_fil+2)+i)]['Gamma'] - HS_vortex[j+1]['VF'+str(vor_fil-i)]['Gamma'])
                    HS_vortex[j]['VF'+str((vor_fil+2)+i)]['Gamma'] = HS_vortex[j+1]['VF'+str(vor_fil-i)]['Gamma'] = Gamma                    
                    
                else:
                    print("Position mismatch during overlap. Check horseshoe vortex coordinate definition.")
    
        return HS_vortex

    HS_vortex = []
    for j in range(blade_seg):
        TR_left = {}
        BVor = {}
        TR_right = {}
        HS_temp = {}
        for i in range(vor_fil):
            # bound vortex coordinates
            if i == 0:
                x = [0, 0]
                y = [(-1)*r_R[j]*b, (-1)*r_R[j+1]*b]
                z = [0, 0]
                BVor['VF'+str(vor_fil+1)]={'pos1': [x[0], y[0], z[0]], 'pos2':[x[1], y[1], z[1]], 'Gamma': 1}

            # first set of trailing vortices
            if rot == False:
                x = [0, U_wake*dt]
                y = [(-1)*r_R[j]*b, (-1)*r_R[j+1]*b]
                z = [0, 0]

                TR_left['VF'+str(vor_fil)] = {'pos1': [x[1], y[0], z[0]], 'pos2':[x[0], y[0], z[0]], 'Gamma': 1}
                TR_right['VF'+str(vor_fil+2)] = {'pos1': [x[0], y[1], z[1]], 'pos2':[x[1], y[1], z[1]], 'Gamma': 1}
                rot = True
            
            # subsequent set of vortex filaments
            elif rot == True:
                # left side of the trailing vortex
                x = [U_wake*dt*(i+1), U_wake*dt*i]
                y = [(-1)*r_R[j]*b*np.cos(Omega*dt*(i+1)), (-1)*r_R[j]*b*np.cos(Omega*dt*i)]
                z = [r_R[j]*b*np.sin(Omega*dt*(i+1)), r_R[j]*b*np.sin(Omega*dt*i)]

                TR_left['VF'+str(vor_fil-i)] = {'pos1': [x[0], y[0], z[0]], 'pos2':TR_left['VF'+str((vor_fil+1)-i)]['pos1'], 'Gamma': 1}

                # right side of the trailing vortex
                x = [U_wake*dt*(i+1), U_wake*dt*i]
                y = [(-1)*r_R[j+1]*b*np.cos(Omega*dt*(i+1)), (-1)*r_R[j+1]*b*np.cos(Omega*dt*i)]
                z = [r_R[j+1]*b*np.sin(Omega*dt*(i+1)), r_R[j+1]*b*np.sin(Omega*dt*i)]

                TR_right['VF'+str((vor_fil+2)+i)] = {'pos1': TR_right['VF'+str((vor_fil+1)+i)]['pos2'], 'pos2':[x[0], y[0], z[0]], 'Gamma': 1}
            
        TR_left = dict(reversed(list(TR_left.items())))
        HS_temp = TR_left | BVor | TR_right
        HS_vortex.append(HS_temp)   
        x = y = z = 0
        rot = False
    
    HS_vortex = CorrectOverlapGamma(HS_vortex)
    return HS_vortex

def InducedVelocities(CtrlPts, pos1, pos2, gamma, tol=1e-5):
    """
    Function to calculate [u,v,w] induced by a vortex filament defined by [pos1, pos2] on a control point defined by CtrlPts.
    Input Arguments:-
        CtrlPts: [xp, yp, zp]; 1D array of control point coordinates
        pos1: [x1, y1, z1]; 1D array of the start position of the vortex filament
        pos1: [x2, y2, z2]; 1D array of the end position of the vortex filament
        gamma: int or float; magnitude of circulation around the filament
    """
    r1 = np.sqrt((CtrlPts[0]-pos1[0])**2 + (CtrlPts[1]-pos1[1])**2 + (CtrlPts[2]-pos1[2])**2)
    r2 = np.sqrt((CtrlPts[0]-pos2[0])**2 + (CtrlPts[1]-pos2[1])**2 + (CtrlPts[2]-pos2[2])**2)

    r12x = (CtrlPts[1]-pos1[1])*(CtrlPts[2]-pos2[2]) - (CtrlPts[2]-pos1[2])*(CtrlPts[1]-pos2[1])
    r12y = -(CtrlPts[0]-pos1[0])*(CtrlPts[2]-pos2[2]) + (CtrlPts[2]-pos1[2])*(CtrlPts[0]-pos2[0])
    r12z = (CtrlPts[0]-pos1[0])*(CtrlPts[1]-pos2[1]) - (CtrlPts[1]-pos1[1])*(CtrlPts[0]-pos2[0])

    r12sq = (r12x**2) + (r12y**2) + (r12z**2)
    if r12sq < tol:
        return 0.0, 0.0, 0.0
    
    r01 = (pos2[0]-pos1[0])*(CtrlPts[0]-pos1[0]) + (pos2[1]-pos1[1])*(CtrlPts[1]-pos1[1]) + (pos2[2]-pos1[1])*(CtrlPts[2]-pos1[1])
    r02 = (pos2[0]-pos1[0])*(CtrlPts[0]-pos2[0]) + (pos2[1]-pos1[1])*(CtrlPts[1]-pos2[1]) + (pos2[2]-pos1[2])*(CtrlPts[2]-pos2[2])
    
    K = (gamma/4*np.pi*r12sq)*((r01/r1) - (r02/r2))
    U = K*r12x
    V = K*r12y
    W = K*r12z
    return U, V, W

def InfluenceCoeff(HS_vortex, control_points, Nb):
    N_cp = len(control_points)      #number of control points
    N_hs = len(HS_vortex)           #number of horseshoe vortices

    #Initialising the matrices 
    u_infl = np.zeros((N_cp,N_hs))    
    v_infl = np.zeros((N_cp,N_hs))
    w_infl = np.zeros((N_cp,N_hs))

    for i in range(N_cp):
        for j in range(N_hs):
            hs = HS_vortex[j]
            u_ind_t = 0
            v_ind_t = 0
            w_ind_t = 0
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

def LiftingLineModel(HS_vortex, control_points, polar_alfa, polar_cl, polar_cd, Vinf, Omega, rho, b, r_R, chord_dist, twist_dist, Nb):
    N_cp = len(control_points)      #number of control points
    N_hs = len(HS_vortex)           #number of horseshoe vortices
    
    gamma = np.zeros((N_hs))
    gamma_new = np.zeros((N_hs))
    a_ax_loc = np.zeros(N_cp)
    a_tan_loc = np.zeros(N_cp)
    F_ax_l = np.zeros(N_cp)
    F_tan_l = np.zeros(N_cp)
    r_cp = np.zeros(N_cp)
    results = {'r':[], 'a_ax':[], 'a_tan':[], 'F_ax':[], 'F_tan':[], 'Gamma':[], 'iterations':0}

    u_infl, v_infl, w_infl = InfluenceCoeff(HS_vortex, control_points, Nb)
    conv = 1e-6
    max_iter = 1000
    relax = 0.4
    iter = 0
    error = 1000

    while error>conv and iter<max_iter:
        gamma_old = np.copy(gamma)
        
        for i in range(N_cp):
            # cp = list(control_points[i].values())[0]
            # xp, yp, zp = cp
            xp = CtrlPts[i]['CP'+str(i+1)][0]
            yp = CtrlPts[i]['CP'+str(i+1)][1]
            zp = CtrlPts[i]['CP'+str(i+1)][2] 
            r_cp[i] = np.sqrt(yp**2 + zp**2)

            # Solving the linear system of equations
            u_ind = np.dot(u_infl[i],gamma)
            v_ind = np.dot(v_infl[i],gamma)
            w_ind = np.dot(w_infl[i],gamma)

            # Finding the local velocity components at the control points
            V_ax_local = Vinf + u_ind   
            V_tan_local = Omega*r_cp[i] - w_ind
            V_local_mag = np.sqrt(V_ax_local**2 + V_tan_local**2)
            phi_local = np.arctan2(V_ax_local, V_tan_local)

            # Finding the blade element properties
            r_R_local = r_cp[i]/b
            chord_local = np.interp(r_R_local, r_R, chord_dist)
            twist_local = np.interp(r_R_local, r_R, twist_dist)
            alpha_local = twist_local - np.rad2deg(phi_local)
            # Cl_local = np.interp(alpha_local, polar_alfa, polar_cl, fill_value=(polar_cl.iloc[0], polar_cl.iloc[-1]), bounds_error=False)
            # Cd_local = np.interp(alpha_local, polar_alfa, polar_cd, fill_value=(polar_cd.iloc[0], polar_cd.iloc[-1]), bounds_error=False)
            Cl_local = np.interp(alpha_local, polar_alfa, polar_cl)
            Cd_local = np.interp(alpha_local, polar_alfa, polar_cd)

            Lift_loc = 0.5 * rho * (V_local_mag**2) * Cl_local * chord_local
            Drag_loc = 0.5 * rho * (V_local_mag**2) * Cd_local * chord_local

            F_ax_loc = Lift_loc * np.cos(phi_local) - Drag_loc * np.sin(phi_local)
            F_tan_loc = Lift_loc * np.sin(phi_local) + Drag_loc * np.cos(phi_local)

            gamma_new[i] = 0.5 * V_local_mag * Cl_local * chord_local

            a_ax_loc[i] = (V_ax_local - Vinf)/Vinf
            a_tan_loc[i] = ((Omega * r_cp[i]) - V_tan_local)/(Omega * r_cp[i])

            F_ax_l[i] = F_ax_loc
            F_tan_l[i] = F_tan_loc
        
        error = np.max(np.abs(gamma_new-gamma_old))
        if error>conv:
            gamma = gamma_new*relax + (1-relax)*gamma_old
            iter += 1
        else:
            results['r'] = r_cp
            results['a_ax'] = a_ax_loc
            results['a_tan'] = a_tan_loc
            results['F_ax'] = F_ax_l
            results['F_tan'] = F_tan_l
            results['Gamma'] = gamma
            results['iterations'] = iter
            return results

def CoordinateRotation(HS_vortex, blade_seg, Nb):
    HS_base = HS_vortex.copy()

    for i in range(1, Nb):
        theta = (2 * np.pi / Nb) * i    # angular position of blade element
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
                pos = R @ pos

                HS_temp[key] = {'pos1': [pos[0][0], pos[1][0], pos[2][0]], 'pos2': [pos[0][1], pos[1][1], pos[2][1]], 'Gamma': VF['Gamma']}

            HS_vortex.append(HS_temp)

    return HS_vortex
                

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
# J = np.array([1.6, 2.0, 2.4])   # advance ratio
J = np.array([2.0])

# Blade geometry
Nb = 2                  # number of blades
b = 0.7                 # Blade radius [m] (or blade span)
root_pos_R = 0.25       # normalised blade root position (r_root/R)
tip_pos_R = 1           # normalised blade tip position (r_tip/R)
pitch = 46              # blade pitch [deg]

# Discretisation 
blade_seg = 3       # no. of segments for the wing
vor_fil = 4         # no. of vortex filaments
l = 20*(2*b)         # length scale of the trailing vortices [m] (based on blade diameter)
seg_type = 'lin'    # discretisation type- 'lin': linear | 'cos': cosine

# Discretisation into blade elements
r_R, chord_dist, twist_dist = BladeSegment(root_pos_R, tip_pos_R, pitch, (blade_seg+1), seg_type)

# Dependent variables 
n = Vinf/(2*J*b)    # RPS [Hz]
Omega = 2*np.pi*n   # Angular velocity [rad/s]
TSR = np.pi/J       # tip speed ratio

# Iteration inputs
tol = 1e-7  # convergence tolerance

# Variable initialisation
CT, CP, CQ = [np.zeros(len(J)) for i in range(3)]
a_b4_Pr, a = [(np.ones((len(J),len(r_R)-1))*(1/3)) for i in range(2)]
chord, a_tan, Cl, Cd, F_ax, F_tan, dCT, dCQ, dCP, alfa, phi, F_tot, F_tip, F_root, P0_down = [np.zeros((len(J),len(r_R)-1)) for i in range(15)]

P_up = np.ones((len(J),len(r_R)-1))*(Pamb + 0.5*rho*(Vinf**2))  # pressure upwind of rotor [Pa]

# Solving BEM model
for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord[j][i] = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist) * b
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        
        r = (r_R[i+1]+r_R[i])*(b/2)     # radial distance of the blade element
        dr = (r_R[i+1]-r_R[i])*b        # length of the blade element
        
        a_b4_Pr[j][i], a[j][i], a_tan[j][i], Cl[j][i], Cd[j][i], F_ax[j][i], F_tan[j][i], alfa[j][i], phi[j][i], F_tot[j][i], F_tip[j][i], F_root[j][i], dCT[j][i], dCQ[j][i], dCP[j][i], P0_down[j][i] = \
        BladeElementMethod(Vinf, TSR[j], n[j], rho, b, r, root_pos_R, tip_pos_R, dr, Omega[j], Nb, a[j][i], a_tan[j][i], twist, chord[j][i], polar_alfa, polar_cl, polar_cd, tol, P_up[j][i])

        CT[j] += dCT[j][i]    # thrust coefficient for given J
        CP[j] += dCP[j][i]    # power coefficient for given J
        CQ[j] += dCQ[j][i]    # torque coefficient for given J

a_rms = np.sqrt(np.mean(np.square(a), axis=1))      # average induction factor for each advance ratio
U_wake = Vinf*(1+a_rms)                             # wake velocity [m/s]

# coordinates for vortex filaments
for i in range(len(U_wake)):
    CtrlPts = ControlPoint(r_R, b, blade_seg)
    HS_vortex = HorseshoeVortex(l, U_wake[i], vor_fil, blade_seg, Omega[i], r_R)

HS_vortex = CoordinateRotation(HS_vortex, blade_seg, Nb)

# print(HS_vortex)
results = LiftingLineModel(HS_vortex, CtrlPts, polar_alfa, polar_cl, polar_cd, Vinf, Omega, rho, b, r_R, chord_dist, twist_dist, Nb)
print(results)

#--------------------------------- Plotting routine ---------------------------------
# Wake visualization
fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection="3d")

colours = cycle(["tab:blue", "tab:orange", "tab:green",
                 "tab:red", "tab:purple", "tab:brown",
                 "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])

for hs in HS_vortex:                               # one dict per blade-segment vortex
    # ---- gather the nodes in the correct order ----------------------------
    nodes = []
    for vf_key in sorted(hs, key=lambda k: int("".join(filter(str.isdigit, k)))):
        vf = hs[vf_key]
        if not nodes:                              # first filament → keep its starting point
            nodes.append(vf["pos1"])
        nodes.append(vf["pos2"])                   # then every filament’s end point

    nodes = np.array(nodes)                       # shape (N, 3)
    c = next(colours)
    ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color=c)

# ------------- plot cosmetics ----------------------------------------------
ax.set_xlabel("x  [m]")
ax.set_ylabel("y  [m]")
ax.set_zlabel("z  [m]")
ax.set_box_aspect([1, 1, 1])                       # roughly equal scaling
ax.set_title("Horseshoe-vortex lattice")

plt.tight_layout()
plt.show()