import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))

plt.rc('text', usetex=True) 
plt.rc('font', family='serif') #For LATEX fonts, ensure latex package is enabled 
#
polars = np.loadtxt("ARAD8pct_polar.txt", dtype = float)
aoa = polars[:,0] # Angle of attack
cl = polars[:,1] # Lift coefficient
cd = polars[:,2] # Drag coefficient
rho  = 1.007
N = 10 # Number of bound vortices
R = 0.7 # Radius of Propeller
nb = 6 # Number of blades

def ind_vel_calc(loc1, loc2, i_loc): # Calculate induced velocity at collocation point i_loc due to a vortex segment defined by loc1 and loc2

            # loc1 =  [x1, y1, z1]
            # loc2 =  [x2, y2, z2]
            # i_loc = [xp, yp, zp]

            perturb = 1e-3 # Perturbation to avoid division by zero in the case of coincident points
            R1 = np.sqrt( (i_loc[0] - loc1[0])**2 + (i_loc[1] - loc1[1])**2 + (i_loc[2] - loc1[2])**2 )
            R2 = np.sqrt( (i_loc[0] - loc2[0])**2 + (i_loc[1] - loc2[1])**2 + (i_loc[2] - loc2[2])**2 )
            
            # if R1 < perturb**2:
            #     R1 = perturb**2


            # if R2 < perturb**2:    
            #     R2 = perturb**2
            # print("R1: ", R1)
            # print("R2: ", R2)
            R12x = ((i_loc[1] - loc1[1])*(i_loc[2] - loc2[2])) - ((i_loc[2] - loc1[2])*(i_loc[1] - loc2[1]))
            R12y = -(i_loc[0] - loc1[0])*(i_loc[2] - loc2[2]) + (i_loc[2] - loc1[2])*(i_loc[0] - loc2[0])
            R12z = (i_loc[0] - loc1[0])*(i_loc[1] - loc2[1]) - (i_loc[1] - loc1[1])*(i_loc[0] - loc2[0])

            R12eq = R12x**2 + R12y**2 + R12z**2
            if R12eq < perturb**2:
                R12eq = perturb**2 #To account for divide by zero error
            
            R01 = (loc2[0] - loc1[0])*(i_loc[0] - loc1[0]) + (loc2[1] - loc1[1])*(i_loc[1] - loc1[1]) + (loc2[2] - loc1[2])*(i_loc[2] - loc1[2])
            R02 = (loc2[0] - loc1[0])*(i_loc[0] - loc2[0]) + (loc2[1] - loc1[1])*(i_loc[1] - loc2[1]) + (loc2[2] - loc1[2])*(i_loc[2] - loc2[2])
            

            K = (1/(4*np.pi*R12eq)) * (R01/R1 - R02/R2)

            u = K * R12x
            v = K * R12y
            w = K * R12z
            
            return u, v, w #Return u, v, w velocities

a_wrange = [0.3] # Checking Axial induction factor sensitivity
n_aw = 0
CT_aw = np.zeros(len(a_wrange))
for a_w in a_wrange:
    print("a_w chosen is: ", a_w)
    Len_wake = (2*R) * 4  # In diameters
    
    Jrange = [1.6] # Advance ratio
    Ctrange = np.zeros(len(Jrange))
    Cqrange = np.zeros(len(Jrange))
    Cprange = np.zeros(len(Jrange))
    
    # print(range(len(Jrange)))
    for jj in range(len(Jrange)):
        J = Jrange[jj]
        RPM = 1200 # RPM
        n = RPM / 60 # Rotations per second
        U_inf = J * (n) * (2*R) # J * Omega_{rot/s} * Diameter
        Omega = 2 * np.pi *n # Convert RPS to rad/s
        
        U_w = U_inf * (1 + a_w) # Induced velocity in the axial direction using a_w as average induced velocity from BEM
        T_total = Len_wake/U_w # Total time for the wake to travel the length of the wake
        N_t = int(Len_wake/(2*R) * 15) # Number of time steps for the wake ## ~ Number of filaments of each trailing vortex (azimuthal discretization)
        
        print("N_t: ", N_t) #
        dt = T_total / N_t # Time step for advancement the wake filaments

        #Geometry of the propeller
        theta = np.linspace(0, np.pi, N+1) ## Cosine Spacing for Blade
        z_nodes = (0.375 *  (1 -  np.cos(theta)) + 0.25) * R #np.linspace(0.25, 1, N+1)*R # Trailing Vortex origin points (METRES) : # N_bound+1 node spanwise positions from 0 to AR [N+1]
        # z_nodes = np.linspace(0.25, 1, N+1) * R # Trailing Vortex origin points (METRES) : # N_bound+1 node spanwise positions from 0 to AR [N+1] Linear Spacing
        z_collocation = 0.5 * (z_nodes[:-1] + z_nodes[1:]) # Collocation point coords (METRES); Midpoints for collocation points [N]
        dr = np.diff(z_nodes) # np.zeros(N) # Radial distance between collocation points (METRES) [N]
        # print(dr)
        ## YZ Node Locations of other 5 blades
        z_nodes_otherblades = np.zeros((N+1,nb-1))
        y_otherblades = np.zeros((N+1,nb-1))

        rotations = np.linspace(60,300,nb-1) * np.pi / 180
        print(rotations*180/np.pi, "rotations")
        chord_act = (0.18 - 0.06 * z_collocation/R) * R # Dimensional chord values along blade at COLLOCATION points
        print(chord_act, "chord_act")
        chord_nodes = (0.18 - 0.06 * z_nodes/R) * R
        y_nodes_ref = 0.25 * chord_nodes 
        y_coll_ref = 0.25 * chord_act
        y_coll_others = np.zeros((N,nb-1))

        # Leading and trailing edge positions at each node (y is chordwise, z is radial)
        # At each z_nodes position, leading edge is at y = +0.5*chord, trailing edge at y = -0.5*chord
        leading_edge = np.zeros((N+1, 2))   # columns: y, z
        trailing_edge = np.zeros((N+1, 2))  # columns: y, z
        leading_edge_others = np.zeros((N+1, 2, nb-1))
        trailing_edge_others = np.zeros((N+1, 2, nb-1))
        for i in range(N+1): # Looping over all nodes on the main blade
            leading_edge[i, 0] = 0.5 * chord_nodes[i] # Leading edge y-coordinate
            leading_edge[i, 1] = z_nodes[i] # Leading edge z-coordinate
            trailing_edge[i, 0] = -0.5 * chord_nodes[i] # Trailing edge y-coordinate
            trailing_edge[i, 1] = z_nodes[i] # Trailing edge z-coordinate
        
        for i in range(len(rotations)): #Looping over other blades, applying counter clockwise rotation to the y and z coordinates
            for j in range(N+1): #For all N+1 nodes on the main blade
                z_nodes_otherblades[j,i] = z_nodes[j] * np.cos(rotations[i]) - y_nodes_ref[j] * np.sin(rotations[i]) #z-coordinates of nodes
                y_otherblades[j,i] = z_nodes[j] * np.sin(rotations[i]) + y_nodes_ref[j] * np.cos(rotations[i]) #y-coordinates of nodes
                leading_edge_others[j,0,i] = leading_edge[j, 0]*np.cos(rotations[i]) + leading_edge[j, 1]*np.sin(rotations[i])
                leading_edge_others[j,1,i] = leading_edge[j, 1]*np.cos(rotations[i]) - leading_edge[j, 0]*np.sin(rotations[i])
                trailing_edge_others[j,0,i] = trailing_edge[j, 0]*np.cos(rotations[i]) + trailing_edge[j, 1]*np.sin(rotations[i])
                trailing_edge_others[j,1,i] = trailing_edge[j, 1]*np.cos(rotations[i]) - trailing_edge[j, 0]*np.sin(rotations[i])

        # z_coll_others = 0.5 * (z_nodes_otherblades[:-1,:] + z_nodes_otherblades[1:,:]) #Not needed
        # y_coll_others = 0.5 * (y_otherblades[:-1,:] + y_otherblades[1:,:]) #Not needed
        

        coll_pitch = 46 #degrees
        beta_act = coll_pitch - (50 * z_collocation/R) + 35 #Local blade pitch in degrees
        
        ##Velocity and Gamma Initialisation
        Vax = U_w * np.ones(N)  # Axial velocity at each collocation point
        Vtan = Omega * z_collocation # Tangential velocity at each collocation point

        # Vax_otherblades = U_w * np.ones((N,nb-1)) #Not Needed
        # Vtan_otherblades = Omega * z_coll_others #Not Needed

        Vp = np.sqrt(Vax**2 + Vtan**2)  # Resultant velocity at each node
        # Vp_otherblades = np.sqrt(Vax_otherblades**2 + Vtan_otherblades**2) #Not Needed

        phi =  np.arctan(Vax/Vtan)  # Local azimuthal angle at each node
        # phi_otherblades = np.arctan(Vax_otherblades/Vtan_otherblades) #Not Needed

        alpha_local = beta_act - phi*180/np.pi  # Effective angle of attack at each node DEGREES
        cl_local = np.interp(alpha_local, aoa, cl) # Initial C_L
        gamma = 0.5 * cl_local * Vp * chord_act #np.ones(N) # Initial Gamma

        influ_matrixU = np.zeros((N, (N)))
        influ_matrixV = np.zeros((N, (N)))
        influ_matrixW = np.zeros((N, (N)))

        i_loc_bound = np.zeros((N, 3))  # Collocation point XYZ coordinates
        radial_distance = np.zeros(N) # Radial distance from the origin to each collocation point
        n_azim = np.zeros((N,3)) #Azimuthal Direction definition at each collocation point on main blade
        
        # # Collocation Points definition
        for i in range(N):
            i_loc_bound[i,0] = 0
            i_loc_bound[i,1] = 0.25 * chord_act[i]
            i_loc_bound[i,2] = z_collocation[i] # Bound vortex segment locations
            radial_distance[i] = np.sqrt(i_loc_bound[i,0]**2 + i_loc_bound[i,1]**2 + i_loc_bound[i,2]**2)
            n_azim[i,:] = np.cross([1 / radial_distance[i], 0, 0], i_loc_bound[i,:])

        # print(n_azim, "nazim")
        # # Bound Vortices Definition on Main Blade
        loc_bound = np.zeros((N+1,3)) # XYZ locations of bound vortices nodes on main blade
        for i in range(N+1):
            loc_bound[i,0] = 0
            loc_bound[i,1] = y_nodes_ref[i]
            loc_bound[i,2] = z_nodes[i]

        # # # Bound Vortices on Other Blades
        loc_bound_otherblades = np.zeros((N+1, 3, nb-1)) #XYZ locations of bound vortices nodes on other blades
        for i in range(N+1):
            for k in range(len(rotations)):
                loc_bound_otherblades[i,0,k] = 0
                loc_bound_otherblades[i,1,k] = y_otherblades[i,k]
                loc_bound_otherblades[i,2,k] = z_nodes_otherblades[i,k]

                # y_nodes_otherblades[i,1,k] = y_nodes_ref[i] * np.cos(rotations[k]) + z_nodes[i] * np.sin(rotations[k])
        
        # plt.figure(10) #Visualising the bound vortices with the hub of prop
        # plt.scatter(loc_bound_otherblades[:, 1, :].flatten(), loc_bound_otherblades[:, 2, :].flatten(), label='Bound Vortices on Other Blades')
        # plt.scatter(leading_edge[:,0], leading_edge[:,1], label='Leading Edge', color='red')
        # plt.scatter(trailing_edge[:,0], trailing_edge[:,1], label='Trailing Edge', color='green')
        # plt.scatter(leading_edge_others[:,0,:].flatten(), leading_edge_others[:,1,:].flatten(), label='Leading Edge on Other Blades', color='orange')
        # plt.scatter(trailing_edge_others[:,0,:].flatten(), trailing_edge_others[:,1,:].flatten(), label='Trailing Edge on Other Blades', color='purple')
        # plt.scatter(loc_bound[:, 1], loc_bound[:, 2], label='Bound Vortices on Main Blade', color='blue')
        
        #
        
        # plt.grid(visible=True)
        # # # Trailing Vortex Definition
        # circle_radius = 0.25*R
        # thetas = np.linspace(0,2*np.pi, 100)
        # hub_y = circle_radius * np.cos(thetas)
        # hub_z = circle_radius * np.sin(thetas)
        # plt.plot(hub_y, hub_z, label='Hub', color='black')
        # plt.show()

        vor_loc_trail = np.zeros(( (N+1), 3*(N_t+1) )) # XYZ locations of trailing vortex filament nodes
        vor_loc_trail_others = np.zeros(( (N+1), 3*(N_t+1), nb-1 )) # Trailing vortex filaments on other blades

        for i in range(N+1): # Looping over trailing vortex origin points
            for j in range(N_t+1): # Looping over filaments of each trailing vortex
                t = j*dt # "Time" at which the j^{th} filament is reached by the trailing vortex
                vor_loc_trail[i, 3*j:3*j+3] = [U_w*t, y_nodes_ref[i] - z_nodes[i] * np.sin(Omega*t), z_nodes[i] * np.cos(Omega*t)]
                for k in range(len(rotations)):
                    vor_loc_trail_others[i,3*j:3*j+3,k] = [U_w*t, y_nodes_ref[i] - z_nodes[i] * np.sin(Omega*t + rotations[k]), z_nodes[i] * np.cos(Omega*t + rotations[k])] #Position of trailing vortex filaments on other blades
            
        # # print("n_azim",n_azim)
        # # print("loc_bound: ", loc_bound)
        # # print("i_loc_bound: ", i_loc_bound)
        # print("vor_loc_trail: ", vor_loc_trail)

        # # # Visualisation of all trailing vortices
        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(vor_loc_trail[:, ::3], vor_loc_trail[:, 1::3], vor_loc_trail[:, 2::3], label=f'Trailing Vortex Filament at t={i*dt:.2f}s')
        
        # plt.title('Trailing Vortex Filaments')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        
        # for i in range(len(rotations)):
        #     ax.plot(vor_loc_trail_others[:, ::3, i], vor_loc_trail_others[:, 1::3, i], vor_loc_trail_others[:, 2::3, i], label=f'Trailing Vortex Filament at t={i*dt:.2f}s')
        # plt.show()

        
        # # # Iterative Calculation of Gamma
        residual = 100
        k_iter = 0

        while residual > 1e-7 and k_iter <1000: # Iteration until convergence or max iterations reached
            influ_matrixU[:, :] = 0
            influ_matrixV[:, :] = 0
            influ_matrixW[:, :] = 0
            for i in range(N): # Loop over N collocation points on main blade
                for j in range(N): # Loop over "N" trailing vortex filament contributions, considering differences between V_{induced} due to overlapping vortices; for each individual blade
                    #Collapsing all the horseshoe vortex contributions from other blades into a single row, for the "j'th" horseshoe vortex
                    for k in range(N_t): # Loop over all nodes in the streamwise (helical) direction of each trailing vortex
                        #Difference to account for overlapping vortices
                        influ_matrixU[i, j] = influ_matrixU[i, j] + 1 * ( - ind_vel_calc( vor_loc_trail[j, 3*k : 3*(k+1) ], vor_loc_trail[j, 3*(k+1) : 3*(k+2)], i_loc_bound[i,:] )[0] + ind_vel_calc( vor_loc_trail[j+1, 3*k : 3*(k+1) ], vor_loc_trail[j+1 ,3*(k+1) : 3*(k+2)], i_loc_bound[i,:] )[0] )
                        influ_matrixV[i, j] = influ_matrixV[i, j] + 1 * ( - ind_vel_calc( vor_loc_trail[j, 3*k : 3*(k+1) ], vor_loc_trail[j, 3*(k+1) : 3*(k+2)], i_loc_bound[i,:] )[1] + ind_vel_calc( vor_loc_trail[j+1, 3*k : 3*(k+1) ], vor_loc_trail[j+1 ,3*(k+1) : 3*(k+2)], i_loc_bound[i,:] )[1] )
                        influ_matrixW[i, j] = influ_matrixW[i, j] + 1 * ( - ind_vel_calc( vor_loc_trail[j, 3*k : 3*(k+1) ], vor_loc_trail[j, 3*(k+1) : 3*(k+2)], i_loc_bound[i,:] )[2] + ind_vel_calc( vor_loc_trail[j+1, 3*k : 3*(k+1) ], vor_loc_trail[j+1 ,3*(k+1) : 3*(k+2)], i_loc_bound[i,:] )[2] )

                        ## Trailing Vortices on Additional Blades
                        for l in range(len(rotations)):
                            influ_matrixU[i, j] = influ_matrixU[i, j] + 1 * ( - ind_vel_calc( vor_loc_trail_others[j, 3 * k: 3 * (k + 1), l], vor_loc_trail_others[j, 3 * (k + 1): 3 * (k + 2), l], i_loc_bound[i, :])[0] + ind_vel_calc( vor_loc_trail_others[j + 1, 3 * k: 3 * (k + 1), l], vor_loc_trail_others[j + 1, 3 * (k + 1): 3 * (k + 2), l], i_loc_bound[i, :])[0] )
                            influ_matrixV[i, j] = influ_matrixV[i, j] + 1 * ( - ind_vel_calc( vor_loc_trail_others[j, 3 * k: 3 * (k + 1), l], vor_loc_trail_others[j, 3 * (k + 1): 3 * (k + 2), l], i_loc_bound[i, :])[1] + ind_vel_calc( vor_loc_trail_others[j + 1, 3 * k: 3 * (k + 1), l], vor_loc_trail_others[j + 1, 3 * (k + 1): 3 * (k + 2), l], i_loc_bound[i, :])[1] )
                            influ_matrixW[i, j] = influ_matrixW[i, j] + 1 * ( - ind_vel_calc( vor_loc_trail_others[j, 3 * k: 3 * (k + 1), l], vor_loc_trail_others[j, 3 * (k + 1): 3 * (k + 2), l], i_loc_bound[i, :])[2] + ind_vel_calc( vor_loc_trail_others[j + 1, 3 * k: 3 * (k + 1), l], vor_loc_trail_others[j + 1, 3 * (k + 1): 3 * (k + 2), l], i_loc_bound[i, :])[2] )

                    
                    # Including Bound Vortices on Main Blade Element
                        influ_matrixU[i, j] = influ_matrixU[i, j] + ind_vel_calc(loc_bound[j, :], loc_bound[j + 1, :], ( i_loc_bound[i, :]))[0]
                        influ_matrixV[i, j] = influ_matrixV[i, j] + ind_vel_calc(loc_bound[j, :], loc_bound[j + 1, :], ( i_loc_bound[i, :]))[1]
                        influ_matrixW[i, j] = influ_matrixW[i, j] + ind_vel_calc(loc_bound[j, :], loc_bound[j + 1, :], ( i_loc_bound[i, :]))[2]

                    # Bound Vortices on Additional Blades
                    for ll in range(len(rotations)):
                        influ_matrixU[i, j] = influ_matrixU[i, j] + ind_vel_calc(loc_bound_otherblades[j, :, ll], loc_bound_otherblades[j + 1, :, ll], (i_loc_bound[i, :]))[0]
                        influ_matrixV[i, j] = influ_matrixV[i, j] + ind_vel_calc(loc_bound_otherblades[j, :, ll], loc_bound_otherblades[j + 1, :, ll], (i_loc_bound[i, :]))[1]
                        influ_matrixW[i, j] = influ_matrixW[i, j] + ind_vel_calc(loc_bound_otherblades[j, :, ll], loc_bound_otherblades[j + 1, :, ll], (i_loc_bound[i, :]))[2]
            
            # New Induced velocities at each collocation point  
            vel_indU = influ_matrixU.dot(gamma) # velocity in U direction
            vel_indV = influ_matrixV.dot(gamma) # Induced velocity in V direction
            vel_indW = influ_matrixW.dot(gamma) # Induced velocity in W direction
            print("---------------TEST---------------", vel_indU, vel_indV, vel_indW)
            # print("Influence Matrix U: ", influ_matrixU[:5, :])
            # print("Influence Matrix V: ", influ_matrixV[:5, :])
            # print("Influence Matrix W: ", influ_matrixW[:5, :])
            #
            # print("vel_indU: ", vel_indU)
            # print("vel_indV: ", vel_indV)
            # print("vel_indW: ", vel_indW)

            V_ax = vel_indU + U_inf  # Axial velocity including induced velocity
            V_tan = Omega * z_collocation + np.sum( (np.array([V_ax, vel_indV, vel_indW])).T * n_azim, axis=1) # Tangential velocity including azimuthal direction

            # print(np.sum( (np.array([V_ax, vel_indV, vel_indW])).T * n_azim, axis=1))
            # print("V_ax: ", V_ax)
            # print("V_tan: ", V_tan)

            V_p = np.sqrt(V_ax**2 + V_tan**2) # Resultant velocity at each node
            phi_local = np.arctan(V_ax/V_tan) # Local azimuthal angle at each node (RADIANS)

            alpha_local = beta_act - (phi_local * 180 / np.pi)  # Effective angle of attack at each node (DEGREES)
            # print("phi_local: ", phi_local)
            # print(phi_local * 180 / np.pi, "phi degrees")
            # print(beta_act, "beta deg")

            cl_local = np.interp(alpha_local, aoa, cl)  # Local lift coefficient based on effective angle of attack
            cd_local = np.interp(alpha_local, aoa, cd)
            gamma_updated_new = 0.5 * V_p * cl_local * chord_act # Gamma considering effective angle of attack

            residual = np.max(np.abs(gamma_updated_new - gamma)) # Calculate max residual for convergence check
            relaxation = 0.2 # Relaxation factor for convergence
            
            gamma = gamma_updated_new * relaxation +  (1 - relaxation) * gamma

            print("Residual: ", residual)

            k_iter += 1 #Advance iteration count
            
            #New average induced velocity for tracking convergence
            U_w = np.mean(vel_indU[:]) + U_inf # Update induced velocity in the axial direction
            a_w = U_w / U_inf - 1 # Update axial induction factor

            ##Updating wake geometry, DO NOT USE
            # for i in range((N+1)):
            #     for j in range(0,N_t+1):
            #         t = (j+1)*dt
            #         vor_loc_trail[i, 3*j] = U_w*t
            #         for k in range(len(rotations)):
            #             vor_loc_trail_others[i, 3*j, k] = U_w*t #Position of trailing vortex filaments on other blades
                            
            print("a_w: ", a_w)


        #Plotting routine 
        # plt.figure(1)
        # cll = np.zeros(N)
        # for i in range(N):
        #     cll[i] = cl_local[i]
        # plt.plot(i_loc_bound[:, 2]/R, cll, label=r'$Lw/D: $' + str(L))
        # plt.grid(visible=True)
        # plt.title(r"$C_L$")
        # plt.legend()
        # plt.xlabel(r"$r/R$")
        # plt.ylabel(r"$C_L$")
        # plt.show()
        
        # plt.figure(2)
        # plt.plot(i_loc_bound[:, 2]/R, alpha_local , label=r'$Lw/D: $' + str(L))
        # plt.grid(visible=True)
        # plt.title(r"$\alpha$")
        # plt.xlabel(r"$r/R$")
        # plt.ylabel(r"$\alpha$ (degrees)")
        # plt.legend()
        # plt.show()

        # # plt.figure(3)
        # # plt.plot(i_loc_bound[:, 2], beta_act, label='betas')
        # # plt.grid(visible=True)
        # # plt.title("Beta")
        # # plt.show()

        plt.figure(4)
        plt.plot(i_loc_bound[:, 2]/R, phi_local * 180 / np.pi, label=r'$Lw/D: $')
        plt.grid(visible=True)
        plt.title(r"$\phi$")
        plt.xlabel(r"$r/R$")
        plt.ylabel(r"$\phi$ (degrees)")
        plt.legend()
        plt.show()
        
        # plt.figure(5)
        # plt.plot(i_loc_bound[:, 2]/R, V_ax/U_inf - 1, label=r'$Lw/D: $' + str(L))
        
        # plt.grid(visible=True)
        # plt.title(r"Axial Induction Factor $a$")
        # plt.legend()
        # plt.xlabel(r"$r/R$")
        # plt.ylabel(r"$a$")
        # plt.show()
        
        # plt.figure(6)
        # plt.plot(i_loc_bound[:, 2]/R, 1- (V_tan/(Omega*z_collocation)), label=r'$Lw/D: $' + str(L))
        # plt.grid(visible=True)
        # plt.xlabel(r"$r/R$")
        # plt.ylabel(r"$a^{'}$")
        # plt.title(r"Azimuthal Induction Factor $a^{'}$")
        
        F_azim = ( cl_local*np.sin(phi_local) + cd_local*np.cos(phi_local) ) * 0.5 * rho * V_p**2 * chord_act
        F_ax = ( cl_local*np.cos(phi_local) - cd_local*np.sin(phi_local) ) * 0.5 * rho * V_p**2 * chord_act
        
        # plt.figure(7)
        # plt.plot(i_loc_bound[:, 2]/R, F_azim/(0.5*rho*U_inf**2*(R)), label=r'$Lw/D: $' + str(L))
        # plt.xlabel(r"$r/R$")
        # plt.ylabel(r"$F_{azim}$")
        # plt.grid(visible=True)
        # plt.title(r"$F_{azim}$, Normalised by $0.5 \rho U_{\infty}^2 R$")
        # plt.legend()
        # plt.show()

        # plt.figure(8)
        # plt.plot(i_loc_bound[:, 2]/R, F_ax/(0.5*rho*U_inf**2*(R)), label=r'$Lw/D: $' + str(L))
        # plt.xlabel(r"$r/R$")
        # plt.ylabel(r"$F_{ax}$")
        # plt.grid(visible=True)
        # plt.title(r"$F_{ax}$, Normalised by $0.5 \rho U_{\infty}^2 R$")
        # plt.legend()
        # plt.show()
        
        # plt.figure(9)
        # plt.plot(i_loc_bound[:, 2]/R, np.sqrt(F_ax**2 + F_azim**2), label='F_total')
        # plt.grid(visible=True)
        # plt.title("Fresult")
        # plt.show()

        Cqrange[jj] = np.sum(F_azim* dr * z_collocation * nb) / (rho * n ** 2 * (2 * R) ** 5)
        Ctrange[jj] = np.sum(F_ax * nb * dr) / (rho * n ** 2 * (2 * R) ** 4)
        Cprange[jj] = Cqrange[jj]*Omega/n
        CT_aw[n_aw] = Ctrange[jj]
        print(CT_aw[n_aw], "CT_aw")
        n_aw +=1
    
    
    print(f"C_T at a_w= {a_w}: {Ctrange}",f"C_P at a_w/D= {a_w}: {Cprange}","J:", Jrange)
    # plt.figure(11)
    # plt.plot(Jrange, Ctrange,'bo-',Jrange,Cqrange,'ro-')
    # plt.show()
    
plt.figure(10)
plt.plot(a_wrange, CT_aw)
plt.xlabel(r'$a_w$')
plt.ylabel(r'$C_T$')
plt.title(r'Sensitivity of $C_T$ to $U_w = U_{\infty}(1+a_w)$')
plt.grid(visible=True)
plt.savefig("CT_convergence_aw.png", dpi=300)