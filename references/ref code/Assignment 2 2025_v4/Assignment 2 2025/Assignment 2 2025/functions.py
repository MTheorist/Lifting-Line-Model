# -*- coding: utf-8 -*-
"""

@author: luked
"""

from math import sqrt, pi, cos, sin, atan2
import numpy as np

class controlpoint:
    coordinates = [0,0,0]
    chord = 0
    twist = 0

class filament:
    pt1 = [0,0,0]
    pt2 = [0,0,0]
    GAMMA = 1

    def rotate(self, angle):
        cosrot = cos(angle)
        sinrot = sin(angle)
        self.pt1 = [self.pt1[0], self.pt1[1]*cosrot - self.pt1[2]*sinrot, self.pt1[1]*sinrot + self.pt1[2]*cosrot]
        self.pt2 = [self.pt2[0], self.pt2[1]*cosrot - self.pt2[2]*sinrot, self.pt2[1]*sinrot + self.pt2[2]*cosrot]

    def translate(self, distance):
        self.pt1 = [self.pt1[0] + distance[0], self.pt1[1] + distance[1], self.pt1[2] + distance[2]]
        self.pt2 = [self.pt2[0] + distance[0], self.pt2[1] + distance[1], self.pt2[2] + distance[2]]

class panel:
    p1 = [0,0,0]
    p2 = [0,0,0]
    p3 = [0,0,0]
    p4 = [0,0,0]

    def rotate(self, angle):
        cosrot = cos(angle)
        sinrot = sin(angle)
        self.p1 = [self.p1[0], self.p1[1]*cosrot - self.p1[2]*sinrot, self.p1[1]*sinrot + self.p1[2]*cosrot]
        self.p2 = [self.p2[0], self.p2[1]*cosrot - self.p2[2]*sinrot, self.p2[1]*sinrot + self.p2[2]*cosrot]
        self.p3 = [self.p3[0], self.p3[1]*cosrot - self.p3[2]*sinrot, self.p3[1]*sinrot + self.p3[2]*cosrot]
        self.p4 = [self.p4[0], self.p4[1]*cosrot - self.p4[2]*sinrot, self.p4[1]*sinrot + self.p4[2]*cosrot]

    def translate(self, distance):
        self.p1 = [self.p1[0] + distance[0], self.p1[1] + distance[1], self.p1[2] + distance[2]]
        self.p2 = [self.p2[0] + distance[0], self.p2[1] + distance[1], self.p2[2] + distance[2]]
        self.p3 = [self.p3[0] + distance[0], self.p3[1] + distance[1], self.p3[2] + distance[2]]
        self.p4 = [self.p4[0] + distance[0], self.p4[1] + distance[1], self.p4[2] + distance[2]]
        
def CTfunc(a, glauert = False):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    CT = np.zeros(np.shape(a))
    CT = 4*a*(1-a)  
    if glauert:
        CT1=1.816;
        a1=1-np.sqrt(CT1)/2;
        CT[a>a1] = CT1-4*(np.sqrt(CT1)-1)*(1-a[a>a1])
    
    return CT

def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    This function calculates the combined tip and root Prandtl correction at a 
    given radial position 'r_R' (non-dimensioned by rotor radius), given a root
    and tip radius (also non-dimensioned), a tip speed ratio TSR, the number of
    blades NBlades and the axial induction factor
    """
    temp1 = -NBlades/2*(tipradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Ftip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Ftip[np.isnan(Ftip)] = 0
    temp1 = NBlades/2*(rootradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Froot = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Froot[np.isnan(Froot)] = 0
    return Froot*Ftip, Ftip, Froot

def aind(CT):
    """
    This function calculates the induction factor 'a' as a function of thrust coefficient CT 
    including Glauert's correction
    """
    a = np.zeros(np.shape(CT))
    CT1=1.816;
    CT2=2*np.sqrt(CT1)-CT1
    a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1))
    a[CT<CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
    return a

def solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, rho, polar_alpha, polar_cl, polar_cd ):
    """
    solve balance of momentum between blade element load and loading in the streamtube
    input variables:
    Uinf - wind speed at infinity
    r1_R,r2_R - edges of blade element, in fraction of Radius ;
    rootradius_R, tipradius_R - location of blade root and tip, in fraction of Radius ;
    Radius is the rotor radius
    Omega - rotational velocity
    NBlades - number of blades in rotor
    """

    # initialize properties of the blade element, variables for output and induction factors
    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2) #  area streamtube
    r_R = (r1_R+r2_R)/2 # centroid
    # initialize variables
    a = 0.3 # axial induction
    aline = 0.0 # tangential induction factor
    
    Niterations = 100
    Erroriterations =0.00001 # error limit for iteration rpocess, in absolute value of induction
    
    for i in range(Niterations):
        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate velocity and loads at blade element" //
        # ///////////////////////////////////////////////////////////////////////
        Urotor = Uinf*(1-a) # axial velocity at rotor
        Utan = (1+aline)*Omega*r_R*Radius # tangential velocity at rotor
        # calculate loads in blade segment in 2D (N/m)
        fnorm, ftan, gamma, alpha, phi = loadBladeElement(Urotor, Utan, r_R, polar_alpha, polar_cl, polar_cd, rho)
        Faxial = fnorm*Radius*(r2_R-r1_R)*NBlades # 3D force in axial direction
      
        # ///////////////////////////////////////////////////////////////////////
        # //the block "Calculate velocity and loads at blade element" is done  //
        # ///////////////////////////////////////////////////////////////////////

        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////
        # // calculate thrust coefficient at the streamtube 
        CT = Faxial/(0.5*rho*Area*Uinf**2)
        
        # calculate new axial induction, accounting for Glauert's correction
        anew = aind(CT)
        
        # correct new axial induction with Prandtl's correction
        Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew);
        if (Prandtl < 0.0001): 
            Prandtl = 0.0001 # avoid divide by zero
        anew = anew/Prandtl # correct estimate of axial induction
        a = 0.75*a+0.25*anew # for improving convergence, weigh current and previous iteration of axial induction

        # calculate aximuthal induction
        aline = ftan*NBlades/(2*np.pi*Uinf*(1-a)*Omega*2*(r_R*Radius)**2)
        aline = aline/Prandtl # correct estimate of azimuthal induction with Prandtl's correction
        # ///////////////////////////////////////////////////////////////////////////
        # // end of the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////
        
        #// test convergence of solution, by checking convergence of axial induction
        if (np.abs(a-anew) < Erroriterations): 
            # print("iterations")
            # print(i)
            break

    return [a , aline, r_R, fnorm , ftan, gamma, alpha, phi]

# 3D velocity induced by the vortex filament
def velocity_3D_from_vortex_filament(GAMMA, XV1, XV2, XVP1, CORE):
    # function to calculate the velocity induced by a straight 3D vortex filament
    # with circulation GAMMA at a point VP1. The geometry of the vortex filament
    # is defined by its edges: the filament starts at XV1 and ends at XV2.
    # the input CORE defines a vortex core radius, inside which the velocity
    # is defined as a solid body rotation.
    # The function is from Katz and Plotkin, Low-speed aerodynamics.

    # read coordinates that define the vortex filament
    X1 = XV1[0] 
    Y1 = XV1[1] 
    Z1 = XV1[2] # start point of vortex filament
    X2 = XV2[0] 
    Y2 = XV2[1] 
    Z2 = XV2[2] # end point of vortex filament
    # read coordinates of target point where the velocity is calculated
    XP = XVP1[0] 
    YP = XVP1[1] 
    ZP = XVP1[2]
    # calculate geometric relations for integral of the velocity induced by filament
    R1=sqrt((XP-X1)**2 + (YP-Y1)**2 + (ZP-Z1)**2)
    R2=sqrt( (XP-X2)**2 + (YP-Y2)**2 + (ZP-Z2)**2)
    R1XR2_X=(YP-Y1)*(ZP-Z2)-(ZP-Z1)*(YP-Y2)
    R1XR2_Y=-(XP-X1)*(ZP-Z2)+(ZP-Z1)*(XP-X2)
    R1XR2_Z=(XP-X1)*(YP-Y2)-(YP-Y1)*(XP-X2)
    R1XR_SQR=R1XR2_X**2 + R1XR2_Y**2 + R1XR2_Z**2
    R0R1 = (X2-X1)*(XP-X1)+(Y2-Y1)*(YP-Y1)+(Z2-Z1)*(ZP-Z1)
    R0R2 = (X2-X1)*(XP-X2)+(Y2-Y1)*(YP-Y2)+(Z2-Z1)*(ZP-Z2)
    # check if target point is in the vortex filament core,
    # and modify to solid body rotation
    if R1XR_SQR < CORE**2:
        R1XR_SQR = CORE**2
        # GAMMA = 0
    if R1 < CORE:
        R1 = CORE
        # GAMMA = 0
    if R2 < CORE:
        R2 = CORE
        # GAMMA = 0
    # determine scalar
    K=GAMMA/4/pi/R1XR_SQR*(R0R1/R1 -R0R2/R2 )
    # determine the three velocity components
    U=K*R1XR2_X
    V=K*R1XR2_Y
    W=K*R1XR2_Z
    # output results, vector with the three velocity components
    results = [U, V, W]
    return results

# create rotor geometry and rotor-wake circulation system
def create_rotor_geometry(span_array, radius, tsr, theta_array, nblades):
    filaments = list()
    ring = list()
    controlpoints = list()
    bladepanels = list()

    print("Generating rotor geometry...")
    for krot in range(nblades):
        
        angle_rotation = 2*pi/nblades*krot
        cosrot = cos(angle_rotation)
        sinrot = sin(angle_rotation)

        for i in range(len(span_array)-1):
            r = (span_array[i]+span_array[i+1])/2
            geodef = geoBlade(r/radius)
            angle = geodef[1]*pi/180

            # define controlpoints
            cp = controlpoint()
            cp.coordinates = [0,r,0]
            cp.chord = geodef[0]
            cp.twist = geodef[1]

            # rotate blade to position
            cp.coordinates = [0, cp.coordinates[1]*cosrot - cp.coordinates[2]*sinrot, cp.coordinates[1]*sinrot + cp.coordinates[2]*cosrot]
            controlpoints.append(cp)
            
            # define bound vortex filament
            ft = filament()
            ft.pt1 = [0,span_array[i],0]
            ft.pt2 = [0, span_array[i+1], 0]

            filaments.append(ft)

            # create trailing filaments, at x1 of bound filament
            geodef = geoBlade(span_array[i]/radius)
            angle = geodef[1]*pi/180
            ft = filament()
            ft.pt1 = [geodef[0]*sin(-angle), span_array[i], -geodef[0]*cos(angle)]
            ft.pt2 = [0, span_array[i], 0]

            filaments.append(ft)
            for j in range(len(theta_array)-1):
                xt = filaments[-1].pt1[0]
                yt = filaments[-1].pt1[1]
                zt = filaments[-1].pt1[2]

                dy = (cos(-theta_array[j+1])-cos(-theta_array[j]))*span_array[i]
                dz = (sin(-theta_array[j+1])-sin(-theta_array[j]))*span_array[i]
                dx = (theta_array[j+1]-theta_array[j])/tsr*radius

                ft = filament()
                ft.pt1 = [xt+dx, yt+dy, zt+dz]
                ft.pt2 = [xt, yt, zt]

                filaments.append(ft)

            # create trailing filaments, at x2 of bound filament
            geodef = geoBlade(span_array[i+1]/radius)
            angle = geodef[1]*pi/180
            ft = filament()
            ft.pt1 = [0, span_array[i+1], 0]
            ft.pt2 = [geodef[0]*sin(-angle), span_array[i+1], -geodef[0]*cos(angle)]

            filaments.append(ft)
            for j in range(len(theta_array)-1):
                xt = filaments[-1].pt2[0]
                yt = filaments[-1].pt2[1]
                zt = filaments[-1].pt2[2]
                dy = (cos(-theta_array[j+1])-cos(-theta_array[j]))*span_array[i+1]
                dz = (sin(-theta_array[j+1])-sin(-theta_array[j]))*span_array[i+1]
                dx = (theta_array[j+1]-theta_array[j])/tsr*radius

                ft = filament()
                ft.pt1 = [xt, yt, zt]
                ft.pt2 = [xt+dx, yt+dy, zt+dz]

                filaments.append(ft)

            # rotate each filament to blade position
            for ft in filaments:
                ft.rotate(angle_rotation)

            ring.append(filaments)
            filaments = list()

            # panel of the blade section
            geodef = geoBlade(span_array[i]/radius);
            angle = geodef[1]*pi/180;
            geodef2 = geoBlade(span_array[i+1]/radius);
            angle2 = geodef2[1]*pi/180;

            pn = panel()
            pn.p1 = [-0.25*geodef[0]*sin(-angle) , span_array[i], 0.25*geodef[0]*cos(angle)]
            pn.p2 = [-0.25*geodef2[0]*sin(-angle2) , span_array[i+1], 0.25*geodef2[0]*cos(angle2)]
            pn.p3 = [0.75*geodef2[0]*sin(-angle2) , span_array[i+1], -0.75*geodef2[0]*cos(angle2)]
            pn.p4 = [0.75*geodef[0]*sin(-angle) , span_array[i], -0.75*geodef[0]*cos(angle)]
            pn.rotate(angle_rotation)

            bladepanels.append(pn);

    return [controlpoints, ring, bladepanels]


def geoBlade(r_R):
    pitch  = 2
    c_dist  = 3*(1-r_R)+1
    tw_dist = -14*(1-r_R)+pitch
    return [c_dist, tw_dist]

def velocity_induced_single_ring(ring, cp):
    velocity_total = [0,0,0]
    core = 0.00001
    for ft in ring:
        velocity = velocity_3D_from_vortex_filament(GAMMA = ft.GAMMA, XV1 = ft.pt1, XV2 = ft.pt2, XVP1 = cp, CORE = core)

        velocity_total[0] += velocity[0]
        velocity_total[1] += velocity[1]
        velocity_total[2] += velocity[2]
    return velocity_total

def solve_lifting_line_system_matrix_approach(rotor_wake_system, wind, omega, rotorradius, rho, polar_alpha, polar_cl, polar_cd, precomputed_file = False, frozen_wake = False):
    
    #      rotor_wake_system: geometry of the horseshoe vortex rings,
    #                         and the control points at the blade
    #      wind: wind velocity, U_infinity
    #      Omega: rotor rotational velocity 
    #      rotorradius: rotor radius
    controlpoints = rotor_wake_system[0]
    rings = rotor_wake_system[1]

    velocity_induced = list() # velocity induced by a horseshoe vortex ring at a control point

    CIRC_new = np.zeros(len(controlpoints))  #  new estimate of bound circulation
    CIRC = np.zeros(len(controlpoints)) #  current solution of bound circulation

    U = np.matrix(np.zeros((len(controlpoints),len(rings))))  # induction matrices, initialized to zero
    V = np.matrix(np.zeros((len(controlpoints),len(rings))))
    W = np.matrix(np.zeros((len(controlpoints),len(rings))))
    
    # convergence criteria
    errorlimit = 0.001
    convWeight = 0.3
    prev_rel_error = list() # used for adapting convWeight
    cooldown = 50

    #outputs
    a = np.zeros(len(controlpoints))
    aline = np.zeros(len(controlpoints))
    r_R = np.zeros(len(controlpoints))
    Fnorm = np.zeros(len(controlpoints)) 
    Ftan = np.zeros(len(controlpoints))
    Gamma = np.zeros(len(controlpoints))
    alpha = np.zeros(len(controlpoints))
    inflow_angle = np.zeros(len(controlpoints))

    if precomputed_file:
        with open(precomputed_file, 'rb') as f:
            matrices = np.load(f)
            U, V, W = matrices[0], matrices[1], matrices[2]
    # initalize and calculate matrices for velocity induced by horseshoe vortex rings
    # two "for cicles", each line varying wind controlpoint "icp", each column varying with
    # horseshoe vortex ring "jring"
    else:
        print("Calculating unit strength horseshoe circulations...")
        for icp, cp in enumerate(controlpoints):
            print(f"{icp/len(controlpoints)*100:.0f}% done.", end="\r")
            for jring, ring in enumerate(rings):
                velocity_induced = velocity_induced_single_ring(rings[jring], cp.coordinates)
                # add component of velocity per unit strength of circulation to induction matrix
                U[icp, jring] = velocity_induced[0]
                V[icp, jring] = velocity_induced[1]
                W[icp, jring] = velocity_induced[2]
        np.save("matrices.npy", [U, V, W])


    # calculate solution through an iterative process
    print("Beginning iterative convergence...")
    iterations = 0
    while True:
        iterations += 1

        CIRC = CIRC_new.copy()  # update current bound circulation with new estimate

        # calculate velocity, circulation and loads at the controlpoints
        for icp, cp in enumerate(controlpoints):
            radialposition = np.linalg.norm(cp.coordinates)
            if frozen_wake:
                # Ignore induced velocities and rotational velocity for frozen wake
                u, v, w = 0, 0, 0
                vrot = [0, 0, 0]
            else:
                u,v,w = 0,0,0 # induced velocities
                # multiply icp row of Matrix with vector of circulation Gamma to calculate velocity at controlpoint
                for jring, ring in enumerate(rings):
                    u += U[icp,jring]*CIRC[jring]
                    v += V[icp,jring]*CIRC[jring]
                    w += W[icp,jring]*CIRC[jring]
            # calculate total perceived velocity
            vrot = np.cross([-omega, 0, 0], cp.coordinates)
            vel = wind + vrot + [u,v,w]

            # transform to azimuthal and axial velocity
            azimdir = np.cross([-1/radialposition, 0, 0], cp.coordinates)
            vazim = np.dot(azimdir, vel)
            vaxial = np.dot([1, 0, 0], vel)

            # calculate loads using BEM
            loads = loadBladeElement(vaxial, vazim, radialposition/rotorradius, polar_alpha, polar_cl, polar_cd, rho)

            # new estimate of circulation
            CIRC_new[icp] = loads[2]

            # outputs
            a[icp] = -(u + vrot[0])/wind[0]
            aline[icp] = vazim/(radialposition*omega)-1
            r_R[icp] = radialposition/rotorradius
            Fnorm[icp] = loads[0]
            Ftan[icp] = loads[1]
            Gamma[icp] = loads[2]
            alpha[icp] = loads[3]
            inflow_angle[icp] = loads[4]

        # check convergence
        refererror = max(np.absolute(CIRC_new))
        error = max(np.absolute(CIRC_new-CIRC))
        rel_error = error/refererror
        print(f"Current max error: {rel_error}, tolerance: {errorlimit}, convWeight: {convWeight}", end="\r")
        if rel_error < errorlimit or convWeight < 0.0001:
            if convWeight > 0.0001:
                print(f"\n Finished after {iterations} iterations")
            else:
                print(f"\nCONVERGENC FAILED after {iterations} iterations")
            return a, aline, r_R, Fnorm, Ftan, Gamma, alpha, inflow_angle

        # store the 10 latest relative errors
        prev_rel_error.append(rel_error)
        if len(prev_rel_error) > 10:
            prev_rel_error.pop(0)

            # half convWeight if the average error decrease is less than 0.01
            if np.average(np.diff(prev_rel_error)) > -0.01 and cooldown < 0:
                convWeight *= 0.5
                cooldown = 20
            else:
                cooldown -= 1

        # set new estimate of bound circulation
        CIRC_new = (1-convWeight)*CIRC + convWeight*CIRC_new

def loadBladeElement(Vnorm, Vtan, r_R, polar_alpha, polar_cl, polar_cd, rho):
    Vmag2 = Vnorm**2 + Vtan**2
    inflow_angle = atan2(Vnorm,Vtan)
    geodef = geoBlade(r_R)
    chord = geodef[0]
    twist = geodef[1]
    alpha = inflow_angle*180/pi + twist

    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)

    Lift = 0.5*rho*Vmag2*cl*chord
    Drag = 0.5*rho*Vmag2*cd*chord
    Fnorm = Lift*cos(inflow_angle) + Drag*sin(inflow_angle)
    Ftan = Lift*sin(inflow_angle) - Drag*cos(inflow_angle)
    Gamma = 0.5*sqrt(Vmag2)*cl*chord

    return Fnorm, Ftan, Gamma, alpha, inflow_angle*180/pi

def calculateCTCP(Fnorm, Ftan, U_inf, r_array, omega, radius, nblades, rho):
    CT = 0
    CP = 0
    for i in range(len(r_array)-1):
        r_R = (r_array[i]+r_array[i+1])/2
        dr = r_array[i+1] - r_array[i]
        CT += dr*Fnorm[i]*nblades/(0.5*rho*U_inf**2*pi*radius)
        CP += dr*Ftan[i]*r_R*omega*nblades/(0.5*rho*U_inf**3*pi)
    return CT,CP

