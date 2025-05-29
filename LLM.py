import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

def filament_induced_velocity(xp,yp,zp,x1,y1,z1,x2,y2,z2,gamma,tol=1e-5):
    r1 = np.sqrt((xp-x1)**2 + (yp-y1)**2 + (zp-z1)**2)
    r2 = np.sqrt((xp-x2)**2 + (yp-y2)**2 + (zp-z2)**2)
    r12x = (yp-y1)*(zp-z2) - (zp-z1)*(yp-y2)
    r12y = -(xp-x1)*(zp-z2) + (zp-z1)*(xp-x2)
    r12z = (xp-x1)*(yp-y2) - (yp-y1)*(xp-x2)
    r12sq = (r12x**2) + (r12y**2) + (r12z**2)
    if r12sq < tol:
        return 0.0, 0.0, 0.0
    r01 = (x2-x1)*(xp-x1) + (y2-y1)*(yp-y1) + (z2-z1)*(zp-z1)
    r02 = (x2-x1)*(xp-x2) + (y2-y1)*(yp-y2) + (z2-z1)*(zp-z2)
    K = (gamma/4*np.pi*r12sq)*((r01/r1) - (r02/r2))
    U = K*r12x
    V = K*r12y
    W = K*r12z
    return U, V, W

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

'''
def WakeDiscretisation(U_wake, r_R, b, l, blade_seg, vor_fil, Omega):
    T = l/U_wake        # total time for wake propagation [s]
    dt = T/vor_fil      # time for propagation of each vortex filament [s]    
    x_fil, y_fil, z_fil = [[np.zeros(blade_seg+1) for i in range(vor_fil+1)] for j in range(3)]
    for j in range(blade_seg+1):
        for i in range(vor_fil+1):
            x_fil[i][j] = U_wake*dt*i
            y_fil[i][j] = (-1)*r_R[j]*b*np.cos(Omega*dt*i)
            z_fil[i][j] = r_R[j]*b*np.sin(Omega*dt*i)
    
    return np.array(x_fil), np.array(y_fil), np.array(z_fil)
'''

def ControlPoint(U_wake, r_R, b, l, blade_seg, vor_fil, Omega):
    T = l/U_wake        # total time for wake propagation [s]
    dt = T/vor_fil      # time for propagation of each vortex filament [s]
    mlt = 0.5           # length normalised distance of control point from origin of blade segment
    x_cp, y_cp, z_cp = [[np.zeros(blade_seg) for i in range(vor_fil)] for j in range(3)]
    for j in range(blade_seg):
        l_seg = (r_R[j+1]-r_R[j])*b
        for i in range(vor_fil):
            # x_cp[i][j] = U_wake*dt*i                                        # x-coord control point location
            x_cp[i][j] = 1e-5
            y_cp[i][j] = (-1)*(l_seg*(j+mlt)+(r_R[0]*b))*np.cos(Omega*dt*i)   # y-coord control point location
            z_cp[i][j] = (l_seg*(j+mlt)+(r_R[0]*b))*np.sin(Omega*dt*i)        # z-coord control point location

    return np.array(x_cp), np.array(y_cp), np.array(z_cp)

def HorseshoeVortex(l, U_wake, vor_fil, blade_seg, Omega, r_R):
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

                # TR_left['VF'+str(vor_fil-i)] = {'pos1': [x[0], y[0], z[0]], 'pos2':[x[1], y[1], z[1]], 'Gamma': 1}
                TR_left['VF'+str(vor_fil-i)] = {'pos1': [x[0], y[0], z[0]], 'pos2':TR_left['VF'+str((vor_fil+1)-i)]['pos1'], 'Gamma': 1}

                # right side of the trailing vortex
                x = [U_wake*dt*(i+1), U_wake*dt*i]
                y = [(-1)*r_R[j+1]*b*np.cos(Omega*dt*(i+1)), (-1)*r_R[j+1]*b*np.cos(Omega*dt*i)]
                z = [r_R[j+1]*b*np.sin(Omega*dt*(i+1)), r_R[j+1]*b*np.sin(Omega*dt*i)]

                # TR_right['VF'+str((vor_fil+2)+i)] = {'pos1': [x[1], y[1], z[1]], 'pos2':[x[0], y[0], z[0]], 'Gamma': 1}
                TR_right['VF'+str((vor_fil+2)+i)] = {'pos1': TR_right['VF'+str((vor_fil+1)+i)]['pos2'], 'pos2':[x[0], y[0], z[0]], 'Gamma': 1}
            
        TR_left = dict(reversed(list(TR_left.items())))
        HS_temp = TR_left | BVor | TR_right
        HS_vortex.append(HS_temp)   
        x = y = z = 0
        rot = False

    return HS_vortex

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
Nb = 6                  # number of blades
b = 0.7                 # Blade radius [m] (or blade span)
root_pos_R = 0.25       # normalised blade root position (r_root/R)
tip_pos_R = 1           # normalised blade tip position (r_tip/R)
pitch = 46              # blade pitch [deg]

# Discretisation 
blade_seg = 2       # no. of segments for the wing
vor_fil = 150         # no. of vortex filaments
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
    # x_fil, y_fil, z_fil = WakeDiscretisation(U_wake[i], r_R, b, l, blade_seg, vor_fil, Omega[i])
    # x_cp, y_cp, z_cp = ControlPoint(U_wake[i], r_R, b, l, blade_seg, vor_fil, Omega[i])
    HS_vortex = HorseshoeVortex(l, U_wake[i], vor_fil, blade_seg, Omega[i], r_R)

# print(HS_vortex[0])
hs = HS_vortex[0]                            # dictionary: 'VF1', 'VF2', …
hs1 = HS_vortex[1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# --- draw one straight line for every vortex filament ----------------------
# sort the keys numerically so VF1 → VF2 → … keeps the construction order
for vf_key in sorted(hs, key=lambda k: int(k[2:])):
    p1 = hs[vf_key]["pos1"]                  # [x1, y1, z1]
    p2 = hs[vf_key]["pos2"]                  # [x2, y2, z2]
    ax.plot([p1[0], p2[0]],                  # x-coords
            [p1[1], p2[1]],                  # y-coords
            [p1[2], p2[2]])                  # z-coords

for vf_key in sorted(hs, key=lambda k: int(k[2:])):
    p1 = hs1[vf_key]["pos1"]                  # [x1, y1, z1]
    p2 = hs1[vf_key]["pos2"]                  # [x2, y2, z2]
    ax.plot([p1[0], p2[0]],                  # x-coords
            [p1[1], p2[1]],                  # y-coords
            [p1[2], p2[2]])                  # z-coords

# --- cosmetics -------------------------------------------------------------
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_box_aspect([1, 1, 1])                 # roughly equal axes scales
plt.tight_layout()
plt.show()