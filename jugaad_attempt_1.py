import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Set working directory to the directory of the current script
os.chdir(os.path.dirname(__file__))

# # Configure matplotlib for LaTeX fonts
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# Load airfoil polar data
airfoil = 'ARAD8pct_polar.csv'
data1=pd.read_csv(airfoil, header=0, names = ["alfa", "cl", "cd", "cm"],  sep=',')
aoa = data1['alfa'][:]
cl = data1['cl'][:]
cd = data1['cd'][:]

# Global physical parameters
rho = 1.007
N = 10  # Number of bound vortices (panels) per blade
R = 0.7  # Radius of Propeller
nb = 6  # Number of blades


def ind_vel_calc(pos1, pos2, cp_coord):
    """
    Calculate induced velocity at collocation point cp_coord due to a vortex segment
    defined by pos1 and pos2 (Biot-Savart Law).

    Args:
        pos1 (list): [x1, y1, z1] coordinates of the first point of the vortex segment.
        pos2 (list): [x2, y2, z2] coordinates of the second point of the vortex segment.
        cp_coord (list): [xp, yp, zp] coordinates of the collocation point.

    Returns:
        tuple: (u, v, w) induced velocities in x, y, and z directions.
    """
    toler = 1e-9  # toleration to avoid division by zero

    r1 = np.array(cp_coord) - np.array(pos1)
    r2 = np.array(cp_coord) - np.array(pos2)
    r2 = np.array(pos2) - np.array(pos1)

    r1_m = np.linalg.norm(r1)
    r2_m = np.linalg.norm(r2)

    if r1_m < toler:
        r1_m = toler
    if r2_m < toler:
        r2_m = toler

    r12 = np.cross(r1, r2)
    r12sq = np.sum(r12**2)

    if r12sq < toler**2:
        r12sq = toler**2

    r1r2 = np.dot(r2, r1)
    r2r2 = np.dot(r2, r2)

    K = (1 / (4 * np.pi * r12sq)) * (r1r2 / r1_m - r2r2 / r2_m)

    u, v, w = K * r12
    return u, v, w


def initialize_propeller_geometry(N_vortices, propeller_radius, num_blades):
    """
    Initializes the geometry of the propeller blades, including node and
    collocation point locations, and chord distribution.

    Args:
        N_vortices (int): Number of bound vortices (panels) per blade.
        propeller_radius (float): Radius of the propeller.
        num_blades (int): Number of blades.

    Returns:
        tuple: Contains various geometric arrays:
            z_nodes, z_collocation, dr, chord_act, chord_nodes, y_nodes_ref,
            cp_coord_bound, loc_bound, n_azim, rotations,
            loc_bound_otherblades, leading_edge_main, trailing_edge_main,
            leading_edge_others, trailing_edge_others.
    """
    # Cosine Spacing for Blade
    theta = np.linspace(0, np.pi, N_vortices + 1)
    z_nodes = (0.375 * (1 - np.cos(theta)) + 0.25) * propeller_radius

    # Collocation point coords (METRES); Midpoints for collocation points [N]
    z_collocation = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    dr = np.diff(z_nodes)  # Radial distance between collocation points

    # Dimensional chord values along blade at COLLOCATION points and nodes
    chord_act = (0.18 - 0.06 * z_collocation / propeller_radius) * propeller_radius
    chord_nodes = (0.18 - 0.06 * z_nodes / propeller_radius) * propeller_radius

    # Y-coordinates for the reference points (e.g., quarter-chord)
    y_nodes_ref = 0.25 * chord_nodes

    # Collocation point XYZ coordinates on the main blade (quarter-chord)
    cp_coord_bound = np.zeros((N_vortices, 3))
    radial_distance = np.zeros(N_vortices)
    n_azim = np.zeros((N_vortices, 3))  # Azimuthal Direction definition

    for i in range(N_vortices):
        cp_coord_bound[i, 0] = 0  # Assuming blade is initially aligned with x-axis (axial direction)
        cp_coord_bound[i, 1] = 0.25 * chord_act[i]
        cp_coord_bound[i, 2] = z_collocation[i]
        # Radial distance from the origin in the Y-Z plane
        radial_distance[i] = np.sqrt(cp_coord_bound[i, 1]**2 + cp_coord_bound[i, 2]**2)
        
        # Azimuthal unit vector in the Y-Z plane (perpendicular to radial vector (0, y, z))
        # This vector represents the direction of tangential velocity.
        if radial_distance[i] > 1e-9: # Avoid division by zero at the root
            n_azim[i, :] = [0, -cp_coord_bound[i, 2] / radial_distance[i], cp_coord_bound[i, 1] / radial_distance[i]]
        else:
            n_azim[i, :] = [0, 0, 0] # At the root, azimuthal direction is ill-defined, set to zero

    # Bound Vortices Definition on Main Blade (quarter-chord)
    loc_bound = np.zeros((N_vortices + 1, 3))
    for i in range(N_vortices + 1):
        loc_bound[i, 0] = 0
        loc_bound[i, 1] = y_nodes_ref[i]
        loc_bound[i, 2] = z_nodes[i]

    # Rotations for other blades (e.g., 60, 120, 180, 240, 300 degrees for 6 blades)
    rotations = np.linspace(360 / num_blades, 360 * (num_blades - 1) / num_blades, num_blades - 1) * np.pi / 180

    # Bound Vortices on Other Blades - rotated positions
    loc_bound_otherblades = np.zeros((N_vortices + 1, 3, num_blades - 1))
    for i in range(N_vortices + 1):
        for k in range(len(rotations)):
            # Rotate y and z coordinates of the bound vortex nodes
            y_rotated = y_nodes_ref[i] * np.cos(rotations[k]) - z_nodes[i] * np.sin(rotations[k])
            z_rotated = y_nodes_ref[i] * np.sin(rotations[k]) + z_nodes[i] * np.cos(rotations[k])
            loc_bound_otherblades[i, 0, k] = 0 # X-coordinate remains 0 (for now)
            loc_bound_otherblades[i, 1, k] = y_rotated
            loc_bound_otherblades[i, 2, k] = z_rotated

    # Leading and trailing edge positions for plotting/reference (not directly used in calculations)
    leading_edge_main = np.zeros((N_vortices + 1, 2))  # columns: y, z
    trailing_edge_main = np.zeros((N_vortices + 1, 2))  # columns: y, z
    leading_edge_others = np.zeros((N_vortices + 1, 2, num_blades - 1))
    trailing_edge_others = np.zeros((N_vortices + 1, 2, num_blades - 1))

    for i in range(N_vortices + 1):
        leading_edge_main[i, 0] = 0.5 * chord_nodes[i]
        leading_edge_main[i, 1] = z_nodes[i]
        trailing_edge_main[i, 0] = -0.5 * chord_nodes[i]
        trailing_edge_main[i, 1] = z_nodes[i]

    for k in range(len(rotations)):
        for j in range(N_vortices + 1):
            y_le_rotated = leading_edge_main[j, 0] * np.cos(rotations[k]) - leading_edge_main[j, 1] * np.sin(rotations[k])
            z_le_rotated = leading_edge_main[j, 0] * np.sin(rotations[k]) + leading_edge_main[j, 1] * np.cos(rotations[k])
            leading_edge_others[j, 0, k] = y_le_rotated
            leading_edge_others[j, 1, k] = z_le_rotated

            y_te_rotated = trailing_edge_main[j, 0] * np.cos(rotations[k]) - trailing_edge_main[j, 1] * np.sin(rotations[k])
            z_te_rotated = trailing_edge_main[j, 0] * np.sin(rotations[k]) + trailing_edge_main[j, 1] * np.cos(rotations[k])
            trailing_edge_others[j, 0, k] = y_te_rotated
            trailing_edge_others[j, 1, k] = z_te_rotated

    return (z_nodes, z_collocation, dr, chord_act, chord_nodes, y_nodes_ref,
            cp_coord_bound, loc_bound, n_azim, rotations,
            loc_bound_otherblades, leading_edge_main, trailing_edge_main,
            leading_edge_others, trailing_edge_others)


def initialize_vortex_wake_geometry(N_vortices, N_t_steps, U_w_val, Omega_val, y_nodes_ref_val, z_nodes_val, rotations_val):
    """
    Initializes the locations of the helical trailing vortex filaments.

    Args:
        N_vortices (int): Number of bound vortices.
        N_t_steps (int): Number of time steps for the wake (azimuthal discretization).
        U_w_val (float): Axial induced velocity used for wake convection.
        Omega_val (float): Angular velocity in rad/s.
        y_nodes_ref_val (ndarray): Y-coordinates of the reference points (quarter-chord) at each node.
        z_nodes_val (ndarray): Z-coordinates (radial positions) of the nodes.
        rotations_val (ndarray): Array of angles for other blades in radians.

    Returns:
        tuple: vor_loc_trail (main blade trailing vortex locations),
               vor_loc_trail_others (other blades trailing vortex locations),
               N_t_steps (actual number of time steps used).
    """
    Len_wake = (2 * R) * 4  # Wake length in diameters (4 diameters)
    if U_w_val == 0:
        # If U_w is zero, wake won't convect axially. Handle this case, perhaps by assuming a very small U_w
        # or by making N_t_steps a fixed small number to represent a short wake.
        # For a practical propeller, U_w (U_inf * (1+a_w)) should generally not be zero unless U_inf is zero and a_w is -1.
        # Here, setting a default N_t to avoid division by zero later.
        N_t_steps = 15
    else:
        T_total = Len_wake / U_w_val  # Total time for the wake to travel the length
        dt = T_total / N_t_steps  # Time step for advancing the wake filaments

    vor_loc_trail = np.zeros(((N_vortices + 1), 3 * (N_t_steps + 1)))
    vor_loc_trail_others = np.zeros(((N_vortices + 1), 3 * (N_t_steps + 1), len(rotations_val)))

    for i in range(N_vortices + 1):  # Looping over trailing vortex origin points (radial nodes)
        for j in range(N_t_steps + 1):  # Looping over filaments of each trailing vortex (streamwise/helical points)
            t = j * dt
            # Main blade trailing vortex filament coordinates
            # x_pos = U_w * t
            # y_pos = initial_y_node * cos(Omega*t) - initial_z_node * sin(Omega*t)
            # z_pos = initial_y_node * sin(Omega*t) + initial_z_node * cos(Omega*t)
            vor_loc_trail[i, 3 * j:3 * j + 3] = [
                U_w_val * t,
                y_nodes_ref_val[i] * np.cos(Omega_val * t) - z_nodes_val[i] * np.sin(Omega_val * t),
                y_nodes_ref_val[i] * np.sin(Omega_val * t) + z_nodes_val[i] * np.cos(Omega_val * t)
            ]
            # Other blades trailing vortices - apply initial blade rotation, then helical rotation
            for k in range(len(rotations_val)):
                y_init_rotated = y_nodes_ref_val[i] * np.cos(rotations_val[k]) - z_nodes_val[i] * np.sin(rotations_val[k])
                z_init_rotated = y_nodes_ref_val[i] * np.sin(rotations_val[k]) + z_nodes_val[i] * np.cos(rotations_val[k])

                vor_loc_trail_others[i, 3 * j:3 * j + 3, k] = [
                    U_w_val * t,
                    y_init_rotated * np.cos(Omega_val * t) - z_init_rotated * np.sin(Omega_val * t),
                    y_init_rotated * np.sin(Omega_val * t) + z_init_rotated * np.cos(Omega_val * t)
                ]

    return vor_loc_trail, vor_loc_trail_others, N_t_steps


def calculate_initial_velocities_and_gamma(N_vortices, U_inf_val, Omega_val, z_collocation_val, aoa_data, cl_data, beta_act_dist, chord_act_dist):
    """
    Calculates initial axial, tangential, and resultant velocities,
    and initial circulation (gamma) based on the freestream and rotational velocities.

    Args:
        N_vortices (int): Number of bound vortices.
        U_inf_val (float): Freestream velocity.
        Omega_val (float): Angular velocity in rad/s.
        z_collocation_val (ndarray): Radial positions of collocation points.
        aoa_data (ndarray): Array of angles of attack from polar data.
        cl_data (ndarray): Array of lift coefficients from polar data.
        beta_act_dist (ndarray): Local blade pitch angle distribution in degrees.
        chord_act_dist (ndarray): Local chord distribution.

    Returns:
        tuple: Vax (initial axial velocity), Vtan (initial tangential velocity),
               gamma (initial circulation), Vp (initial resultant velocity),
               phi (initial flow angle), alpha_local (initial local angle of attack).
    """
    Vax = U_inf_val * np.ones(N_vortices)  # Initial axial velocity (assuming U_w = U_inf for initial guess)
    Vtan = Omega_val * z_collocation_val  # Tangential velocity at each collocation point

    Vp = np.sqrt(Vax**2 + Vtan**2)  # Resultant velocity
    phi = np.arctan2(Vax, Vtan)  # Local azimuthal flow angle (radians)

    alpha_local = beta_act_dist - np.degrees(phi)  # Effective angle of attack (degrees)
    cl_local = np.interp(alpha_local, aoa_data, cl_data)  # Initial C_L based on local alpha

    gamma = 0.5 * cl_local * Vp * chord_act_dist  # Initial Gamma (circulation)
    return Vax, Vtan, gamma, Vp, phi, alpha_local, cl_local


def compute_influence_matrix(N_vortices, N_t_steps, cp_coord_bound, loc_bound, loc_bound_otherblades, vor_loc_trail, vor_loc_trail_others, rotations_val):
    """
    Computes the influence matrix (A_ij) for induced velocities at collocation points
    due to all bound and trailing vortex segments, assuming a unit circulation.

    Args:
        N_vortices (int): Number of bound vortices.
        N_t_steps (int): Number of time steps for the wake.
        cp_coord_bound (ndarray): XYZ coordinates of collocation points.
        loc_bound (ndarray): XYZ locations of bound vortices nodes on main blade.
        loc_bound_otherblades (ndarray): XYZ locations of bound vortices nodes on other blades.
        vor_loc_trail (ndarray): XYZ locations of trailing vortex filament nodes for main blade.
        vor_loc_trail_others (ndarray): XYZ locations of trailing vortex filament nodes for other blades.
        rotations_val (ndarray): Array of angles for other blades in radians.

    Returns:
        tuple: (influ_matrixU, influ_matrixV, influ_matrixW) influence matrices.
    """
    influ_matrixU = np.zeros((N_vortices, N_vortices))
    influ_matrixV = np.zeros((N_vortices, N_vortices))
    influ_matrixW = np.zeros((N_vortices, N_vortices))

    num_other_blades = len(rotations_val)

    for i in range(N_vortices):  # Loop over collocation points on the main blade (control points)
        for j in range(N_vortices):  # Loop over bound vortex segments (influence points)
            # Influence of the j-th bound vortex segment on the main blade
            u_b, v_b, w_b = ind_vel_calc(loc_bound[j, :], loc_bound[j + 1, :], cp_coord_bound[i, :])
            influ_matrixU[i, j] += u_b
            influ_matrixV[i, j] += v_b
            influ_matrixW[i, j] += w_b

            # Influence of trailing vortices originating from the j-th and (j+1)-th radial position on the main blade
            # These contributions represent the vortex sheet shed from the j-th panel.
            for k in range(N_t_steps):
                # Induced velocity from segment k to k+1 of the trailing vortex starting at node j
                u_tj_seg, v_tj_seg, w_tj_seg = ind_vel_calc(vor_loc_trail[j, 3 * k:3 * (k + 1)],
                                                             vor_loc_trail[j, 3 * (k + 1):3 * (k + 2)],
                                                             cp_coord_bound[i, :])
                
                # Induced velocity from segment k to k+1 of the trailing vortex starting at node j+1
                u_tj1_seg, v_tj1_seg, w_tj1_seg = ind_vel_calc(vor_loc_trail[j + 1, 3 * k:3 * (k + 1)],
                                                                vor_loc_trail[j + 1, 3 * (k + 1):3 * (k + 2)],
                                                                cp_coord_bound[i, :])
                
                # The influence coefficient reflects the contribution of Gamma_j for the entire horseshoe vortex system
                # This formulation (-u_tj_seg + u_tj1_seg) is typical for a finite-length trailing vortex panel,
                # representing the influence of the circulation difference (Gamma_j - Gamma_{j+1}).
                # Since Gamma_j is the variable for the entire horseshoe vortex panel, its trailing legs contribute.
                # The original structure of the code for adding trailing vortex influence is complex.
                # Assuming the intent is the influence of the trailing legs of the j-th horseshoe vortex,
                # the influence of the segment (j, k to k+1) would be added.
                # The negative sign on the first term might be related to how the vortex line direction is defined
                # or how a closed loop (horseshoe) contribution is broken down.
                # Replicating the exact formula from the original code's trailing vortex part:
                influ_matrixU[i, j] += (-u_tj_seg + u_tj1_seg)
                influ_matrixV[i, j] += (-v_tj_seg + v_tj1_seg)
                influ_matrixW[i, j] += (-w_tj_seg + w_tj1_seg)


            # Contributions from other blades' bound vortices
            for ll in range(num_other_blades):
                u_bob, v_bob, w_bob = ind_vel_calc(loc_bound_otherblades[j, :, ll],
                                                    loc_bound_otherblades[j + 1, :, ll],
                                                    cp_coord_bound[i, :])
                influ_matrixU[i, j] += u_bob
                influ_matrixV[i, j] += v_bob
                influ_matrixW[i, j] += w_bob

                # Contributions from other blades' trailing vortices
                for k in range(N_t_steps):
                    u_tjo_seg, v_tjo_seg, w_tjo_seg = ind_vel_calc(vor_loc_trail_others[j, 3 * k: 3 * (k + 1), ll],
                                                                   vor_loc_trail_others[j, 3 * (k + 1): 3 * (k + 2), ll],
                                                                   cp_coord_bound[i, :])

                    u_tj1o_seg, v_tj1o_seg, w_tj1o_seg = ind_vel_calc(vor_loc_trail_others[j + 1, 3 * k: 3 * (k + 1), ll],
                                                                      vor_loc_trail_others[j + 1, 3 * (k + 1): 3 * (k + 2), ll],
                                                                      cp_coord_bound[i, :])
                    influ_matrixU[i, j] += (-u_tjo_seg + u_tj1o_seg)
                    influ_matrixV[i, j] += (-v_tjo_seg + v_tj1o_seg)
                    influ_matrixW[i, j] += (-w_tjo_seg + w_tj1o_seg)

    return influ_matrixU, influ_matrixV, influ_matrixW


def update_velocities_and_gamma(U_inf_val, Omega_val, z_collocation_val, n_azim_vecs, aoa_data, cl_data, cd_data, beta_act_dist,
                                 gamma_prev, influ_matrixU, influ_matrixV, influ_matrixW, relaxation_factor, chord_act_dist):
    """
    Performs one iteration of the gamma update, calculating new induced velocities,
    effective angles of attack, and updating gamma using relaxation.

    Args:
        U_inf_val (float): Freestream velocity.
        Omega_val (float): Angular velocity in rad/s.
        z_collocation_val (ndarray): Radial positions of collocation points.
        n_azim_vecs (ndarray): Azimuthal unit vectors at collocation points.
        aoa_data (ndarray): Array of angles of attack from polar data.
        cl_data (ndarray): Array of lift coefficients from polar data.
        cd_data (ndarray): Array of drag coefficients from polar data.
        beta_act_dist (ndarray): Local blade pitch angle distribution in degrees.
        gamma_prev (ndarray): Circulation distribution from the previous iteration.
        influ_matrixU (ndarray): Influence matrix for U velocity.
        influ_matrixV (ndarray): Influence matrix for V velocity.
        influ_matrixW (ndarray): Influence matrix for W velocity.
        relaxation_factor (float): Relaxation factor for convergence.
        chord_act_dist (ndarray): Local chord distribution.

    Returns:
        tuple: gamma_new (updated circulation), residual, V_ax, V_tan, phi_local,
               alpha_local, cl_local, cd_local.
    """
    # Calculate induced velocities at each collocation point due to current gamma distribution
    vel_indU = influ_matrixU.dot(gamma_prev)
    vel_indV = influ_matrixV.dot(gamma_prev)
    vel_indW = influ_matrixW.dot(gamma_prev)

    # Calculate total axial and tangential velocities
    V_ax = vel_indU + U_inf_val  # Axial velocity including induced velocity
    
    # Calculate induced tangential velocity component by projecting induced velocity vector onto azimuthal direction
    # Induced V_total = [vel_indU, vel_indV, vel_indW]
    # Azimuthal vector = [0, n_azim_y, n_azim_z]
    # Induced V_tan = (vel_indU * 0) + (vel_indV * n_azim_y) + (vel_indW * n_azim_z)
    induced_V_tan = vel_indV * n_azim_vecs[:,1] + vel_indW * n_azim_vecs[:,2]
    V_tan = Omega_val * z_collocation_val + induced_V_tan  # Total tangential velocity

    V_p = np.sqrt(V_ax**2 + V_tan**2) # Resultant velocity
    phi_local = np.arctan2(V_ax, V_tan)  # Local azimuthal flow angle (radians)

    alpha_local = beta_act_dist - np.degrees(phi_local)  # Effective angle of attack (degrees)
    cl_local = np.interp(alpha_local, aoa_data, cl_data)  # Local lift coefficient based on effective alpha
    cd_local = np.interp(alpha_local, aoa_data, cd_data)  # Local drag coefficient

    # Calculate new circulation based on updated flow conditions
    gamma_updated_raw = 0.5 * V_p * cl_local * chord_act_dist

    # Calculate residual for convergence check
    residual = np.max(np.abs(gamma_updated_raw - gamma_prev))
    
    # Apply relaxation to update gamma
    gamma_new = gamma_updated_raw * relaxation_factor + (1 - relaxation_factor) * gamma_prev

    return gamma_new, residual, V_ax, V_tan, phi_local, alpha_local, cl_local, cd_local


def calculate_forces_and_coefficients(rho_val, n_rps, R_propeller, nb_blades, dr_dist, z_collocation_dist,
                                      phi_local_rad, cl_local_val, cd_local_val, V_p_val, chord_act_dist):
    """
    Calculates the elemental forces and integrates them to get thrust and torque coefficients.

    Args:
        rho_val (float): Air density.
        n_rps (float): Rotations per second.
        R_propeller (float): Propeller radius.
        nb_blades (int): Number of blades.
        dr_dist (ndarray): Radial distance between collocation points.
        z_collocation_dist (ndarray): Radial positions of collocation points.
        phi_local_rad (ndarray): Local azimuthal flow angle in radians.
        cl_local_val (ndarray): Local lift coefficient.
        cd_local_val (ndarray): Local drag coefficient.
        V_p_val (ndarray): Resultant velocity at collocation points (converged values).
        chord_act_dist (ndarray): Local chord distribution.

    Returns:
        tuple: F_azim (elemental azimuthal force), F_ax (elemental axial force),
               Cq (torque coefficient), Ct (thrust coefficient), Cp (power coefficient).
    """
    # Elemental forces per unit length of blade span
    # Force perpendicular to Vp: Lift. Force parallel to Vp: Drag.
    # We need to project these into axial and tangential (azimuthal) directions.
    
    # Lift acts perpendicular to Vp. Drag acts parallel to Vp.
    # sin(phi) component of Vp is Vax, cos(phi) component of Vp is Vtan.
    # F_lift_axial = Lift * cos(phi)
    # F_lift_tangential = Lift * sin(phi)
    # F_drag_axial = Drag * sin(phi)
    # F_drag_tangential = Drag * cos(phi)

    # Elemental axial force (thrust)
    F_ax = (cl_local_val * np.cos(phi_local_rad) - cd_local_val * np.sin(phi_local_rad)) * 0.5 * rho_val * V_p_val**2 * chord_act_dist

    # Elemental azimuthal force (contributes to torque)
    F_azim = (cl_local_val * np.sin(phi_local_rad) + cd_local_val * np.cos(phi_local_rad)) * 0.5 * rho_val * V_p_val**2 * chord_act_dist

    # Integrate elemental forces over the blade span and multiply by number of blades
    torque = np.sum(F_azim * dr_dist * z_collocation_dist * nb_blades)
    thrust = np.sum(F_ax * nb_blades * dr_dist)

    diameter = 2 * R_propeller
    
    # Non-dimensional coefficients
    Cq = torque / (rho_val * n_rps**2 * diameter**5)
    Ct = thrust / (rho_val * n_rps**2 * diameter**4)
    Cp = Cq * (2 * np.pi * n_rps) / n_rps # Cp = Cq * Omega / n = Cq * 2*pi

    return F_azim, F_ax, Cq, Ct, Cp


def plot_results(z_collocation, R_propeller, phi_local, alpha_local, axial_induction_factor, gamma, 
                 a_w_range_vals, CT_aw_vals):
    """
    Generates and displays various plots for the simulation results.

    Args:
        z_collocation (ndarray): Radial positions of collocation points.
        R_propeller (float): Propeller radius.
        phi_local (ndarray): Local azimuthal flow angle in radians.
        alpha_local (ndarray): Effective angle of attack in degrees.
        axial_induction_factor (ndarray): Axial induction factor 'a'.
        gamma (ndarray): Circulation distribution along the blade.
        a_w_range_vals (list): List of initial axial induction factors tested for sensitivity.
        CT_aw_vals (ndarray): Array of thrust coefficients corresponding to a_w_range_vals.
    """
    # --- Plotting Phi (Local Azimuthal Flow Angle) ---
    plt.figure(figsize=(8, 5))
    plt.plot(z_collocation / R_propeller, np.degrees(phi_local), 'b-o', markersize=4)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(r'Local Azimuthal Flow Angle $\phi$', fontsize=14)
    plt.xlabel(r'Radial Position $r/R$', fontsize=12)
    plt.ylabel(r'$\phi$ (degrees)', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()

    # --- Plotting Alpha (Effective Angle of Attack) ---
    plt.figure(figsize=(8, 5))
    plt.plot(z_collocation / R_propeller, alpha_local, 'g-o', markersize=4)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(r'Effective Angle of Attack $\alpha$', fontsize=14)
    plt.xlabel(r'Radial Position $r/R$', fontsize=12)
    plt.ylabel(r'$\alpha$ (degrees)', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()

    # --- Plotting Axial Induction Factor (a) ---
    plt.figure(figsize=(8, 5))
    plt.plot(z_collocation / R_propeller, axial_induction_factor, 'r-o', markersize=4)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(r'Axial Induction Factor $a$', fontsize=14)
    plt.xlabel(r'Radial Position $r/R$', fontsize=12)
    plt.ylabel(r'$a$', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()

    # --- Plotting Circulation (Gamma) ---
    plt.figure(figsize=(8, 5))
    plt.plot(z_collocation / R_propeller, gamma, 'm-o', markersize=4)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(r'Circulation $\Gamma$', fontsize=14)
    plt.xlabel(r'Radial Position $r/R$', fontsize=12)
    plt.ylabel(r'$\Gamma$ (m$^2$/s)', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()

    # --- Plot Sensitivity of CT to initial a_w ---
    if len(a_w_range_vals) > 1: # Only plot if multiple a_w values were tested
        plt.figure(figsize=(8, 5))
        plt.plot(a_w_range_vals, CT_aw_vals, marker='o', linestyle='-', color='purple')
        plt.xlabel(r'Initial Axial Induction Factor $a_w$', fontsize=12)
        plt.ylabel(r'Thrust Coefficient $C_T$', fontsize=12)
        plt.title(r'Sensitivity of $C_T$ to Initial $a_w$', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        plt.savefig("CT_convergence_aw.png", dpi=300)
        plt.show()


def run_vortex_lattice_simulation(J_range, RPM_val, R_propeller, N_vortices, nb_blades, rho_val, aoa_data, cl_data, cd_data, a_w_initial_range):
    """
    Main function to run the Vortex Lattice Method simulation for a propeller.
    This function orchestrates the entire simulation process.

    Args:
        J_range (list): List of advance ratios to simulate.
        RPM_val (float): Rotations per minute.
        R_propeller (float): Propeller radius.
        N_vortices (int): Number of bound vortices per blade.
        nb_blades (int): Number of blades.
        rho_val (float): Air density.
        aoa_data (ndarray): Array of angles of attack from polar data.
        cl_data (ndarray): Array of lift coefficients from polar data.
        cd_data (ndarray): Array of drag coefficients from polar data.
        a_w_initial_range (list): Range of initial axial induction factors to test.
    """
    print(f"Starting VLM simulation with {nb_blades} blades and {N_vortices} bound vortices per blade.")

    # Initialize propeller geometry (independent of advance ratio or a_w)
    (z_nodes, z_collocation, dr, chord_act, chord_nodes, y_nodes_ref,
     cp_coord_bound, loc_bound, n_azim, rotations,
     loc_bound_otherblades, leading_edge_main, trailing_edge_main,
     leading_edge_others, trailing_edge_others) = initialize_propeller_geometry(N_vortices, R_propeller, nb_blades)

    # Blade pitch distribution
    coll_pitch = 46  # degrees (collective pitch at root)
    beta_act = coll_pitch - (50 * z_collocation / R_propeller) + 35  # Local blade pitch in degrees (a linearly varying twist)

    CT_aw_results = np.zeros(len(a_w_initial_range))
    
    # Loop through different initial 'a_w' values to check sensitivity
    for n_aw_idx, a_w_initial in enumerate(a_w_initial_range):
        print(f"\n--- Simulating with initial a_w = {a_w_initial:.3f} ---")

        n_rps = RPM_val / 60  # Rotations per second
        Omega = 2 * np.pi * n_rps  # Angular velocity in rad/s

        Ctrange_J = np.zeros(len(J_range))
        Cqrange_J = np.zeros(len(J_range))
        Cprange_J = np.zeros(len(J_range))

        # Loop through different advance ratios
        for jj, J in enumerate(J_range):
            U_inf = J * n_rps * (2 * R_propeller)  # Freestream velocity based on J
            
            # Initial estimate of axial flow velocity for wake geometry definition
            # The wake geometry is fixed based on this initial estimate.
            # A fully iterative VLM would update wake geometry in each iteration.
            U_w_for_wake_convection = U_inf * (1 + a_w_initial)
            
            # Determine number of time steps (streamwise wake segments) for wake definition
            Len_wake = (2 * R_propeller) * 4 # 4 diameters long wake
            if U_w_for_wake_convection == 0:
                N_t_wake_steps = 15 # Default for stationary or ill-defined wake convection
            else:
                # Approximately 15 segments per diameter length of wake
                N_t_wake_steps = int(Len_wake / (2 * R_propeller) * 15)
                if N_t_wake_steps < 1: N_t_wake_steps = 1 # Ensure at least one wake segment

            print(f"  J = {J:.2f}: Initial U_w for wake: {U_w_for_wake_convection:.3f} m/s, Wake Steps (N_t): {N_t_wake_steps}")

            # Initialize vortex locations for the fixed wake geometry
            vor_loc_trail, vor_loc_trail_others, N_t_actual = initialize_vortex_wake_geometry(
                N_vortices, N_t_wake_steps, U_w_for_wake_convection, Omega, y_nodes_ref, z_nodes, rotations)

            # Calculate initial velocities and gamma guess
            Vax_initial, Vtan_initial, gamma, Vp_initial, phi_initial, alpha_local_initial, cl_local_initial = \
                calculate_initial_velocities_and_gamma(N_vortices, U_inf, Omega, z_collocation, aoa_data, cl_data, beta_act, chord_act)

            # Compute the influence matrix (A_ij). This matrix is based on the fixed wake geometry.
            influ_matrixU, influ_matrixV, influ_matrixW = compute_influence_matrix(
                N_vortices, N_t_actual, cp_coord_bound, loc_bound, loc_bound_otherblades, vor_loc_trail, vor_loc_trail_others, rotations)

            # --- Iterative Calculation of Gamma ---
            residual = 1.0  # Initialize residual to a large value
            k_iter = 0
            max_iterations = 1000
            relaxation_factor = 0.2

            # Variables to store final converged values for plotting
            final_V_ax, final_V_tan, final_phi_local, final_alpha_local, final_cl_local, final_cd_local, final_gamma = \
                Vax_initial, Vtan_initial, phi_initial, alpha_local_initial, cl_local_initial, cd_data, gamma # Initialize with initial guesses

            while residual > 1e-7 and k_iter < max_iterations:
                gamma, residual, V_ax_iter, V_tan_iter, phi_local_iter, alpha_local_iter, cl_local_iter, cd_local_iter = \
                    update_velocities_and_gamma(U_inf, Omega, z_collocation, n_azim, aoa_data, cl_data, cd_data,
                                                 beta_act, gamma, influ_matrixU, influ_matrixV, influ_matrixW, relaxation_factor, chord_act)
                
                # Update variables for final output if convergence is reached
                final_V_ax = V_ax_iter
                final_V_tan = V_tan_iter
                final_phi_local = phi_local_iter
                final_alpha_local = alpha_local_iter
                final_cl_local = cl_local_iter
                final_cd_local = cd_local_iter
                final_gamma = gamma # This 'gamma' is the one used in the next iteration, and is the converged one

                if k_iter % 100 == 0 or k_iter == max_iterations - 1:
                    # Calculate current axial induction factor for tracking
                    current_a_w_tracking = (final_V_ax - U_inf) / U_inf if U_inf != 0 else np.zeros_like(final_V_ax)
                    print(f"    Iteration {k_iter}: Residual = {residual:.2e}, Avg. current a = {np.mean(current_a_w_tracking):.3f}")
                k_iter += 1

            if k_iter == max_iterations:
                print(f"    Warning: VLM did not converge within {max_iterations} iterations for J={J:.2f}, initial a_w={a_w_initial:.2f}")
            else:
                print(f"    VLM converged in {k_iter} iterations for J={J:.2f}, initial a_w={a_w_initial:.2f}")

            # Calculate final forces and coefficients using the converged velocities and circulation
            final_V_p = np.sqrt(final_V_ax**2 + final_V_tan**2) # Converged resultant velocity
            F_azim_conv, F_ax_conv, Cq_conv, Ct_conv, Cp_conv = calculate_forces_and_coefficients(
                rho_val, n_rps, R_propeller, nb_blades, dr, z_collocation,
                final_phi_local, final_cl_local, final_cd_local, final_V_p, chord_act
            )

            Ctrange_J[jj] = Ct_conv
            Cqrange_J[jj] = Cq_conv
            Cprange_J[jj] = Cp_conv
            CT_aw_results[n_aw_idx] = Ct_conv # Store for plotting a_w sensitivity

            print(f"  J: {J:.2f}, Final C_T: {Ctrange_J[jj]:.4f}, Final C_P: {Cprange_J[jj]:.4f}, Final C_Q: {Cqrange_J[jj]:.4f}")
            
            # Calculate final axial induction factor distribution for plotting
            final_axial_induction_factor = (final_V_ax - U_inf) / U_inf if U_inf != 0 else np.zeros_like(final_V_ax)

            # Plotting for the current J value and converged results (will overwrite if multiple J values)
            # You might want to make these plots conditional or save them with unique filenames
            # if running multiple J values and wanting to keep all plots.
            plot_results(z_collocation, R_propeller, final_phi_local, final_alpha_local,
                         final_axial_induction_factor, final_gamma, a_w_initial_range, CT_aw_results)


    print("\n--- All Simulations Complete ---")
    print(f"Thrust Coefficients (C_T) for initial a_w values: {CT_aw_results}")

# --- Main execution block ---
if __name__ == "__main__":
    # Define simulation parameters
    axial_induction_factor_range_to_test = [0.3] # Test with one or multiple initial a_w values
    advance_ratio_range = [1.6] # Test with one or multiple advance ratios (J)

    # Run the VLM simulation
    run_vortex_lattice_simulation(
        J_range=advance_ratio_range,
        RPM_val=1200,
        R_propeller=R,
        N_vortices=N,
        nb_blades=nb,
        rho_val=rho,
        aoa_data=aoa,
        cl_data=cl,
        cd_data=cd,
        a_w_initial_range=axial_induction_factor_range_to_test
    )
