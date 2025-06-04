
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
# plt.rcParams.update({
#     "text.usetex": True,
#     #"font.family": "Helvetica"
# })
# plt.rcParams['font.size'] = 12
# #plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
def prandtl_c(rbyR,rbyRroot,B,Tsr,aunc):
    mu=rbyR
    muroot=rbyRroot
    temp1=-(B/2)*((1-mu)/mu)*(np.sqrt(1+((Tsr*mu/(1+aunc))**2)))
    temp2=-(B/2)*((mu-muroot)/mu)*(np.sqrt(1+((Tsr*mu/(1+aunc))**2)))
    
    ftip=(2/(np.pi))*np.arccos(np.exp(temp1))
    froot=(2/(np.pi))*np.arccos(np.exp(temp2))
    F=ftip*froot
    
    return F, ftip, froot
  
def BEM(r_Rf, NBlades, Omegaf, Radius, Uinf, TipLocation_R, RootLocation_R, TSRf, corr):
    r_R_centf=(r_Rf[1:]+r_Rf[:len(r_Rf)-1])/2
    a_guess = 0.1
    atan_guess = 0.1
    n=Omegaf/(2*np.pi)
    Ct_fin = r_R_centf*0
    Cq_fin =  r_R_centf*0
    prandtl_fin=r_R_centf*0
    kappa_fin =  r_R_centf*0 
    kappa_p_fin=  r_R_centf*0
    C_T_standard=  r_R_centf*0
    C_q_standard=r_R_centf*0
    a_fin =  r_R_centf*0
    a_tan_fin =  r_R_centf*0
    phi_fin=r_R_centf*0
    AoA= r_R_centf*0
    Norm_load=r_R_centf*0
    Tang_load= r_R_centf*0
    vmag=r_R_centf*0
    Gamma = r_R_centf*0
    tol = 1e-5
    err = 1
    colpitch = 46
    for j in range(0, len(r_Rf)-1):
        rbR=r_R_centf[j]
        it=0
        err=1
        pitch_dist_lcl=-50*(rbR) + 35+ colpitch
        a_guess = 0.1
        atan_guess = 0.1
        a=0
        a_tan=0
        for i in range(1, 1001):
            if (err > tol):
                a_guess = 0.9*a_guess + 0.1*a
                atan_guess = 0.9*atan_guess + 0.1*a_tan
                it=it+1;
                u_norm = Uinf*(1+a_guess)
                u_tan = Omegaf* rbR*Radius* (1-atan_guess)
                umag = np.sqrt(u_norm**2 + u_tan**2)
                phi=np.arctan2(u_norm,u_tan) #in Radians
                #print(phi*180/np.pi)             
                alpha =  pitch_dist_lcl - phi*180/np.pi # in degrees
                cl = np.interp(alpha, polar_alpha, polar_cl)
                cd = np.interp(alpha, polar_alpha, polar_cd)
                #Lift = 0.5*rho*chord_dist*(umag**2)*cl
                #Drag = 0.5*rho*chord_dist*(umag**2)*cd
                Ct = cl*np.cos(phi)- cd*np.sin(phi)
                Cq = cl*np.sin(phi)+ cd*np.cos(phi)

                chord_dist_lcl = (0.18-0.06*rbR)*Radius
                solidity_lcl=chord_dist_lcl*NBlades/(2*np.pi*rbR*Radius)
                kappa = 4*(np.sin(phi)**2)/(solidity_lcl*Ct) 
                kappa_p= 4*((np.sin(phi))*np.cos(phi))/(solidity_lcl*Cq)

                a_unc = 1/(kappa-1)
                #print(a_unc)
                a_tan_unc = 1/(kappa_p +1)
                prandtl, ptip, proot=prandtl_c(rbyR=rbR,rbyRroot=RootLocation_R,B=NBlades,Tsr=TSRf,aunc=a_unc)
                
                if (prandtl < 0.0001): 
                    prandtl = 0.0001
                if (corr== 'false'):
                    prandtl=1
                a=np.copy(a_unc/prandtl)
                a_tan=np.copy(a_tan_unc/prandtl)
                #print(a)

                err=np.sqrt(((a-a_guess)**2)+(a_tan-atan_guess)**2)
                #phi = phi_new


            else:
                break

        prandtl_fin[j]=prandtl
        phi_fin[j]=phi*180/np.pi
        AoA[j]=  alpha
    #   Ct_fin[j] = cl*np.cos(phi)- cd*np.sin(phi)
    #   Cq_fin[j] = cl*np.sin(phi)+ cd*np.cos(phi)
        Ct_fin[j]=Ct
        Cq_fin[j]=Cq
        r2_R=r_Rf[j+1];
        r1_R=r_Rf[j];
        Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2)
        vmag2=umag**2
#         C_T_standard[j]=Ct_fin[j]*Radius*(r2_R-r1_R)*NBlades*0.5*vmag2*chord_dist_lcl/(0.5*Area*((Uinf)**2))
#         C_q_standard[j]=Cq_fin[j]*Radius*(r2_R-r1_R)*NBlades*0.5*vmag2*chord_dist_lcl/(0.5*Area*((Uinf)**2))
        Norm_load[j]=0.5*vmag2*Ct_fin[j]*chord_dist_lcl*(r2_R-r1_R)*Radius # force per density
        Tang_load[j]=0.5*vmag2*Cq_fin[j]*chord_dist_lcl*(r2_R-r1_R)*Radius
        C_T_standard[j]=NBlades* Norm_load[j] / ( (n**2) * (((2*Radius))**4) )
        C_q_standard[j]=NBlades* Tang_load[j]*r_R_centf[j]*Radius / ( ((n**2)* ((2*Radius)**5) ))
        # C_T_standard[j]=Ct_fin[j]*Radius*(r2_R-r1_R)*NBlades*0.5*vmag2*chord_dist_lcl/(0.5*Area*((n*Radius*2)**2))
        # C_q_standard[j]=Cq_fin[j]*Radius*(r2_R-r1_R)*NBlades*0.5*vmag2*chord_dist_lcl/(0.5*Area*((n*Radius*2)**2))
        
        
        
        a_fin[j] = a
        a_tan_fin[j] = a_tan
        vmag[j]=umag
        cl_gamma = np.interp(alpha, polar_alpha, polar_cl)
        Gamma[j] = 0.5* umag*chord_dist_lcl* cl_gamma
    # chekit=cl*np.cos(phi)- cd*np.sin(phi)-
    return C_T_standard, C_q_standard, a_fin, a_tan_fin, prandtl_fin, AoA, phi_fin, vmag, Norm_load, Tang_load, Gamma


def StagPres(J, Norm_Load, a_fin):
    r_R_cent_UpInf = r_R_cent*np.sqrt(1+a_fin) # Radius of Streamtube at upstream infinity
    P0_UpInf = (P_inf + 0.5*rho*Uinf**2)*np.ones(len(r_R_cent)) #Stagnation Pressure at Infinity Upstream
    P0_UpR = P0_UpInf #Stagnation Pressure at Rotor Upstream
    DeltaP_0 = (Norm_load*rho*NBlades/(np.pi*area))  #Increase in Stagnation Pressure
    P0_DnR = P0_UpInf + DeltaP_0  #Stagnation Pressure at Rotor Downstream
    r_R_cent_DnInf = r_R_cent*np.sqrt((1+a_fin)/(1+2*a_fin)) # Radius of Streamtube at Downstream infinity
    P0_DnInf = P_inf + 0.5*rho*((Uinf*(1+2*a_fin))**2) #Stagnation Pressure at Infinity Downwind
    return r_R_cent_UpInf, P0_UpInf, P0_UpR, P0_DnR, r_R_cent_DnInf, P0_DnInf


# In[2]:


# Prandtl, tip, root = prandtl_c(0.4, 0.25, 3, 7, 0.3)
# print(Prandtl)


# In[3]:



airfoil = 'propairfoil.csv'
data1=pd.read_csv(airfoil, header=0,
                    names = ["alfa", "cl", "cd", "cm"],  sep='\\s+')
polar_alpha = data1['alfa'][:]
polar_cl = data1['cl'][:]
polar_cd = data1['cd'][:]

# define flow conditions
Uinf = 60 # unperturbed wind speed in m/s
Radius = 0.7

J =60/28
n = Uinf/(J*2*Radius)
Omega = 2*np.pi*n
TSR = Omega*Radius/Uinf # tip speed ratio
NBlades = 6

TipLocation_R =  1
RootLocation_R =  0.25

#rho = 1.007 #dependent on height
#R_mean = np.sqrt(((TipLocation_R**2) +(RootLocation_R**2))/2)
r_R = np.linspace(RootLocation_R, TipLocation_R,31)

# blade shape
colpitch = 46 # degrees
chord_dist = 0.18-0.06*r_R # meters
pitch_dist = -50*(r_R) + 35+ colpitch  # degrees
solidity  = chord_dist*NBlades/(2*np.pi*r_R*Radius)


# In[21]:



# fig, axs = plt.subplots(1, 1, figsize=(6, 6))
# axs.plot(polar_alpha, polar_cl/polar_cd)
# axs.set_xlim([-10,25])
# axs.set_xlabel(r'$\alpha$')
# axs.set_ylabel(r'$C_l/D_d$')
# axs.grid()
# axs.set_title('Lift to Drag ratio variation with AOA')
# plt.show()


# In[4]:


#prandtl plot checking

a = np.zeros(np.shape(r_R))+0.1
Prandtl, tip, root = prandtl_c(r_R, 0.25, 6, TSR, a)

# fig1 = plt.figure(figsize=(12, 6))
# plt.plot(r_R, Prandtl, 'r-', label='Prandtl')
# plt.plot(r_R, tip, 'g.', label='Prandtl tip')
# plt.plot(r_R, root, 'b.', label='Prandtl root')
# plt.xlabel('r/R')
# plt.legend()
# plt.show()


# In[5]:


r_R_cent=(r_R[1:]+r_R[:len(r_R)-1])/2
C_T_standard, C_q_standard, a_fin, a_tan_fin, prandtl_fin, AoA, phi_fin, vmag, Norm_load, Tang_load, gamma=BEM(r_R, NBlades, Omega, Radius, Uinf, TipLocation_R, RootLocation_R, TSR, corr='true')


# In[6]:


J = 2 #60/28
n = Uinf/(J*2*Radius)
Omega2 = 2*np.pi*n
TSR2 = Omega2*Radius/Uinf # tip speed ratio
C_T_standard2, C_q_standard2, a_fin2, a_tan_fin2, prandtl_fin2, AoA2, phi_fin2, vmag_2, Norm_load2, Tang_load2, gamma2=BEM(r_R, NBlades, Omega2, Radius, Uinf, TipLocation_R, RootLocation_R, TSR2, corr='true')

J = 2.4 #60/28
n = Uinf/(J*2*Radius)
Omega3 = 2*np.pi*n
TSR3 = Omega3*Radius/Uinf # tip speed ratio
C_T_standard3, C_q_standard3, a_fin3, a_tan_fin3, prandtl_fin3, AoA3, phi_fin3, vmag_3, Norm_load3, Tang_load3, gamma3=BEM(r_R, NBlades, Omega3, Radius, Uinf, TipLocation_R, RootLocation_R, TSR3, corr='true')

# CTBEM=(4*(a_fin)*(a_fin+1))
# check=C_T_standard-CTBEM
# print(check)


# In[7]:


# r2_R=r_R[j+1];
# r1_R=r_R[j];
# Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2)
# vmag2=u_norm**2 + u_tan**2
# CTcheck=Ct_fin[j]*Radius*(r2_R-r1_R)*NBlades*0.5*vmag2*chord_dist_lcl/(0.5*Area*(Uinf**2))
# CTBEM=4*a_fin[j]*(a_fin[j]+1)
# print(CTcheck-CTBEM)


# In[8]:


# print('iterations to converge='+str(it))
# dr=(r_R[1:]-r_R[:-1])*Radius
#area=np.pi*((r_R[1:]*Radius)**2-(r_R[:-1]*Radius)**2)
area= (((r_R[1:])**2)-((r_R[:-1])**2))*Radius**2
rho=1.00649;
J_arr=np.array([1.6,2,2.4])
n_arr=Uinf/(J_arr*2*Radius)
omega_arr = 2*np.pi*n_arr
# CT_total1=np.sum(C_T_standard*area/(np.pi*Radius**2))
# CT_total2=np.sum(C_T_standard2*area/(np.pi*Radius**2))
# CT_total3=np.sum(C_T_standard3*area/(np.pi*Radius**2))
CT_total1=np.sum(Norm_load*NBlades/((n_arr[0]**2) *((2*Radius)**4)))
CT_total2=np.sum(Norm_load2*NBlades/((n_arr[1]**2) *((2*Radius)**4)))
CT_total3=np.sum(Norm_load3*NBlades/((n_arr[2]**2) *((2*Radius)**4)))
#print(CT_total1)
 #as per ISA
Thrust=np.array([CT_total1, CT_total2, CT_total3])*((n_arr**2) *((2*Radius)**4))*rho


# In[9]:


thrust_load=Norm_load/((((n_arr[0])*((2*Radius)**2))**2))
Azimuth_load=Tang_load/((((n_arr[0])*((2*Radius)**2))**2))
torque_coeff=(np.sum(Tang_load*r_R_cent*Radius))*NBlades/((((n_arr[0])*((2*Radius)**2))**2)*(2*Radius))
thrust_load2=Norm_load2/((((n_arr[1])*((2*Radius)**2))**2))
Azimuth_load2=Tang_load2/((((n_arr[1])*((2*Radius)**2))**2))
torque_coeff2=(np.sum(Tang_load2*r_R_cent*Radius))*NBlades/((((n_arr[1])*((2*Radius)**2))**2)*(2*Radius))
thrust_load3=Norm_load3/((((n_arr[2])*((2*Radius)**2))**2))
Azimuth_load3=Tang_load3/((((n_arr[2])*((2*Radius)**2))**2))
torque_coeff3=(np.sum(Tang_load3*r_R_cent*Radius))*NBlades/((((n_arr[2])*((2*Radius)**2))**2)*(2*Radius))
C_Q_arr =np.array([torque_coeff, torque_coeff2, torque_coeff3])
Torque =C_Q_arr*(((((n_arr)*((2*Radius)**2))**2)*(2*Radius))*rho)


Power=Torque*(n_arr)*2*np.pi
eff=Thrust*Uinf/Power
Cp=Power/(rho*(n_arr**3)*(2*Radius)**5)


# In[10]:


#print(Cp)


# In[11]:


#######Stagnation Pressure Calculations
P_inf =79495; #Pascal
a_arr = np.array([a_fin, a_fin2, a_fin3])
a_tan_arr = np.array([a_tan_fin, a_tan_fin2, a_tan_fin3])
Norm_load_arr = np.array([Norm_load, Norm_load2, Norm_load3])

r_R_UpInf = np.zeros((len(r_R_cent),len(J_arr))) 
P0_UpInf =np.zeros((len(r_R_cent),len(J_arr))) 
P0_UpR= np.zeros((len(r_R_cent),len(J_arr))) 
P0_DnR= np.zeros((len(r_R_cent),len(J_arr))) 
r_R_DnInf= np.zeros((len(r_R_cent),len(J_arr))) 
P0_DnInf= np.zeros((len(r_R_cent),len(J_arr))) 
for j in range(0, len(J_arr)):
    r_R_UpInf[:,j], P0_UpInf[:,j], P0_UpR[:,j], P0_DnR[:,j], r_R_DnInf[:,j], P0_DnInf[:,j] = StagPres(J_arr[j], Norm_load_arr[j], a_arr[j] )


# In[12]:


# ######################plotting routines  ################################
# plt.figure(num=1);
# plt.plot(r_R_cent,phi_fin, label='J=1.6');
# plt.plot(r_R_cent,phi_fin2, label='J=2');
# plt.plot(r_R_cent,phi_fin3, label='J=2.4');
# plt.grid(1)
# plt.xlabel('r/R')
# plt.ylabel(r' Incidence angle ($\phi$) in degrees ')
# plt.title(r'Spanwise distribution of incidence angle $\phi$' )
# plt.legend(loc='upper right')

# #plt.savefig('/home/vkande/phi.pdf', bbox_inches='tight')


# plt.figure(num=2);
# plt.plot(r_R_cent,AoA, label='J=1.6')
# plt.plot(r_R_cent,AoA2, label='J=2');
# plt.plot(r_R_cent,AoA3, label='J=2.4');
# plt.grid(1)
# plt.xlabel('r/R')
# plt.ylabel(r' AoA ($\alpha$) in degrees ')
# plt.title(r'Spanwise distribution of angle of attack $\alpha$' )
# plt.legend(loc='upper right')

# #plt.savefig('/home/vkande/AoA.pdf', bbox_inches='tight')



# plt.figure(num=3);
# plt.plot(r_R_cent, a_fin, label='J=1.6');
# plt.plot(r_R_cent, a_fin2, label='J=2');
# plt.plot(r_R_cent, a_fin3, label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'Axial Induction factor ($a$)')
# plt.title(r'Spanwise distribution of Axial Induction factor ($a$)' )
# plt.legend(loc='upper left')
# #plt.savefig('/home/vkande/axialinduciton.pdf', bbox_inches='tight')

# plt.figure(num=4);
# plt.plot(r_R_cent, a_tan_fin, label='J=1.6');
# plt.plot(r_R_cent, a_tan_fin2, label='J=2');
# plt.plot(r_R_cent, a_tan_fin3, label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'Azimuthal Induction factor ($a^{\prime}$)')
# plt.title(r'Spanwise distribution of azimuthal induction($a^{\prime}$)' )
# plt.legend(loc='upper right')
# #plt.savefig('/home/vkande/azimuthinduciton.pdf', bbox_inches='tight')


# plt.figure(num=5);
# # plt.plot(r_R_cent, C_T_standard, label='J=1.6');
# # plt.plot(r_R_cent, C_T_standard2, label='J=2');
# # plt.plot(r_R_cent, C_T_standard3,  label='J=2.4');
# plt.plot(r_R_cent, thrust_load, label='J=1.6');
# plt.plot(r_R_cent, thrust_load2, label='J=2');
# plt.plot(r_R_cent, thrust_load3,  label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'Thrust loading  ')
# #plt.title(r'Blade thrust  ( $ F_n/(0.5\rho U_{\infty}^2 R$)) and azimuthal loading ($ F_t/(0.5\rho U_{\infty}^2 R$))' )
# plt.title(r'Spanwise distribution of Thrust  loading ($F_{norm}/(\rho  n^2 D^4$)) on each blade' )
# plt.legend(loc='upper left')
# #plt.savefig('/home/vkande/thrustload.pdf', bbox_inches='tight')

# plt.figure(num=6);
# # plt.plot(r_R_cent, C_q_standard, label='J=1.6');
# # plt.plot(r_R_cent, C_q_standard2, label='J=2');
# # plt.plot(r_R_cent, C_q_standard3,  label='J=2.4');
# plt.plot(r_R_cent, Azimuth_load, label='J=1.6');
# plt.plot(r_R_cent, Azimuth_load2, label='J=2');
# plt.plot(r_R_cent, Azimuth_load3,  label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'Azimuthal loading on each blade')
# #plt.title(r'Blade thrust  ( $ F_n/(0.5\rho U_{\infty}^2 R$)) and azimuthal loading ($ F_t/(0.5\rho U_{\infty}^2 R$))' )
# plt.title(r'Spanwise distribution of Azimuthal loading($F_{tan}/(\rho  n^2 D^4$)) on each blade' )
# plt.legend(loc='upper left')
# #plt.savefig('/home/vkande/azimuthload.pdf', bbox_inches='tight')

# plt.figure(num=7);
# plt.plot(J_arr, Thrust, '*-', label=r'Total Thrust');
# plt.plot(J_arr, Torque, '*-', label=r'Total torque');

# plt.grid(1)

# plt.xlabel('J')
# plt.ylabel(r'Total Thrust ($N$), Torque ($N-m$)')
# #plt.title(r'Blade thrust  ( $ F_n/(0.5\rho U_{\infty}^2 R$)) and azimuthal loading ($ F_t/(0.5\rho U_{\infty}^2 R$))' )
# plt.title(r'Total thrust force and Total torque variation with advance ratio (J)' )
# plt.legend(loc='upper right')
# #plt.savefig('/home/vkande/thrusttorq.pdf', bbox_inches='tight')

# plt.figure(num=8);
# plt.plot(J_arr, Thrust/(((n_arr**2) *((2*Radius)**4))*rho), '*-', label=r'$C_T$');
# plt.plot(J_arr, C_Q_arr, '*-', label=r'$C_Q$');
# #plt.plot(J_arr, eff, '*-', label=r'$\eta_p$');
# plt.grid(1)

# plt.xlabel('J')
# plt.ylabel(r'$C_T$, $C_Q$')
# #plt.title(r'Blade thrust  ( $ F_n/(0.5\rho U_{\infty}^2 R$)) and azimuthal loading ($ F_t/(0.5\rho U_{\infty}^2 R$))' )
# plt.title(r'Thrust and torque coeffecients variation with advance ratio (J)' )

# plt.legend(loc='upper right')
# #plt.savefig('/home/vkande/ctcq.pdf', bbox_inches='tight')

# plt.show()


# # In[13]:


# #effect of prandtl
# Jp = 1.6 #60/28
# npr = Uinf/(Jp*2*Radius)
# Omegap = 2*np.pi*npr
# TSRp = Omega2*Radius/Uinf # tip speed ratio
# C_T_standardp, C_q_standardp, a_finp, a_tan_finp, prandtl_finp, AoAp, phi_finp, vmag_p, Norm_loadp, Tang_loadp, gammap=BEM(r_R, NBlades, Omegap, Radius, Uinf, TipLocation_R, RootLocation_R, TSRp, corr='false')
# thrust_loadp=Norm_loadp/((((npr)*((2*Radius)**2))**2))


# plt.figure(num=9);
# plt.plot(r_R_cent, a_fin, label='axial induction corrected');
# plt.plot(r_R_cent, a_finp, label='axial induction uncorrected');
# #plt.plot(r_R_cent, a_fin3, label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'Axial Induction factor ($a$)')
# plt.title(r'Spanwise distribution of Axial Induction factor ($a$)' )
# plt.legend(loc='upper left')
# #plt.savefig('/home/vkande/axialinducitonprntl.pdf', bbox_inches='tight')


# plt.figure(num=10);
# plt.plot(r_R_cent, a_tan_fin, label='azimuthal induction corrected');
# plt.plot(r_R_cent, a_tan_finp, label='azimuthal induction uncorrected');
# #plt.plot(r_R_cent, a_fin3, label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'Azimuthal Induction factor ($a^{\prime}$)')
# plt.title(r'Spanwise distribution of Azimuthal Induction factor ($a^{\prime}$)' )
# plt.legend(loc='upper right')
# #plt.savefig('/home/vkande/azimuthinducitonprntl.pdf', bbox_inches='tight')


# plt.figure(num=11);
# plt.plot(r_R_cent, thrust_load, label='Thrust loading corrected');
# plt.plot(r_R_cent, thrust_loadp, label='Thrust loading uncorrected');
# #plt.plot(r_R_cent, a_fin3, label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'Thrust  loading ($F_{norm}/(\rho  n^2 D^4$)) on each blade')
# plt.title(r'Spanwise distribution of Thrust  loading ($F_{norm}/(\rho  n^2 D^4$)) on each blade' )
# plt.legend(loc='upper left')
# #plt.savefig('/home/vkande/thrustprntl.pdf', bbox_inches='tight')

# plt.figure(num=12);
# plt.plot(r_R_cent, AoA, label=r'$\alpha$ corrected');
# plt.plot(r_R_cent, AoAp, label=r'$\alpha$  uncorrected');
# #plt.plot(r_R_cent, a_fin3, label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'$\alpha$ (degrees)')
# plt.title(r'Spanwise distribution of $\alpha$' )
# plt.legend(loc='upper left')
# #plt.savefig('/home/vkande/alphaprntl.pdf', bbox_inches='tight') 
# for j in range(0, 1):
#     plt.figure(num = 13+j);
#     plt.plot(r_R_UpInf[:,j], P0_UpInf[:,j], label=r'$P_0$ at Infinity Upstream');
#     plt.plot(r_R_cent, P0_UpR[:,j], label=r'$P_0$ at Rotor upstream');
#     plt.plot(r_R_cent, P0_DnR[:,j], label=r'$P_0$ at Rotor Downstream')
#     plt.plot(r_R_DnInf[:,j], P0_DnInf[:,j], label=r'$P_0$ at Infinity Downstream')
#     plt.grid(1)
#     plt.xlabel(r'$r/R_{prop}$')
#     plt.ylabel(r'Stagnation Pressure $P_0$')
#     plt.legend(loc='upper left')
#     plt.title(r'Radial Variation of Stagnation Pressure at different locations for $J=1.6$' )
#     #plt.savefig('/home/kjm2004/RotorWake/StagPres_unc.pdf', bbox_inches='tight')
    
    
# plt.figure(14)
# plt.plot(r_R_cent, gamma/(np.pi*Uinf**2/(NBlades*omega_arr[0])), label=r'J=1.6')
# plt.plot(r_R_cent, gamma2/(np.pi*Uinf**2/(NBlades*omega_arr[1])), label=r'J=2.0')
# plt.plot(r_R_cent, gamma3/(np.pi*Uinf**2/(NBlades*omega_arr[2])), label=r'J=2.4')
# #plt.plot(r_R_cent, 4*a_fin*(1+a_fin)/(1-a_tan_fin), label=r'check')
# #plt.plot(r_R_cent, 4*a_fin2*(1+a_fin2)/(1-a_tan_fin2), label=r'check2')
# #plt.plot(r_R_cent, 4*a_fin3*(1+a_fin3)/(1-a_tan_fin3), label=r'check2')
# plt.xlabel(r'$r/R$')
# plt.ylabel(r'$\Gamma/(\pi U_{\infty}^2/(B\Omega))$')
# plt.legend(loc='upper left')
# plt.grid(1)
# plt.title(r'Circulation distribution $\Gamma$, non-dimensioned by $\frac{\pi U_\infty^2}{\Omega B}$' )
# #plt.savefig('/home/kjm2004/RotorWake/Gamma.pdf', bbox_inches='tight')
# plt.show()


# # In[14]:


# plt.figure(num=15);
# plt.plot(r_R_cent, prandtl_fin);

# #plt.plot(r_R_cent, a_fin3, label='J=2.4');
# plt.grid(1)

# plt.xlabel('r/R')
# plt.ylabel(r'Prandtl correction factor')
# plt.title(r'Spanwise distribution of pradntl correction' )

# #plt.savefig('/home/vkande/prntl.pdf', bbox_inches='tight') 
# plt.show()


# # In[15]:


# plt.figure(num=16);

# plt.plot(J_arr, eff, '*-', label=r'$\eta_p$');
# plt.grid(1)

# plt.xlabel('J')
# plt.ylabel(r'$\eta_p$')
# #plt.title(r'Blade thrust  ( $ F_n/(0.5\rho U_{\infty}^2 R$)) and azimuthal loading ($ F_t/(0.5\rho U_{\infty}^2 R$))' )
# plt.title(r'Propeller effeciency variation with advance ratio (J)' )


# #plt.savefig('/home/vkande/propeff.pdf', bbox_inches='tight')

# plt.show()


# # In[16]:


# el = np.array([11, 21, 51, 101, 201, 501])
# CT_total_el = np.zeros((len(el)))
# CT_total_cos = np.zeros((len(el)))
# J = 1.6 
# n = Uinf/(J*2*Radius)
# Omega = 2*np.pi*n
# TSR = Omega*Radius/Uinf # tip speed ratio
# NBlades = 6

# for i in range(0,len(el) ):
#     r_R_el = np.linspace(RootLocation_R, TipLocation_R,el[i])
#     t = np.linspace(0, np.pi, el[i])
#     cos_t = (1 - np.cos(t)) / 2
#     r_R_cos = RootLocation_R + (TipLocation_R - RootLocation_R) * cos_t
#     r_R_cent_el=(r_R_el[1:]+r_R_el[:len(r_R_el)-1])/2
#     r_R_cent_cos=(r_R_cos[1:]+r_R_el[:len(r_R_cos)-1])/2
#     C_T_standard_el, C_q_standard_el, a_fin_el, a_tan_fin_el, prandtl_fin_el, AoA_el, phi_fin_el, vmag_el, Norm_load_el, Tang_load_el, gamma_el=BEM(r_R_el, NBlades, Omega, Radius, Uinf, TipLocation_R, RootLocation_R, TSR, corr='true' )
#     C_T_standard_cos, C_q_standard_cos, a_fin_cos, a_tan_fin_cos, prandtl_fin_cos, AoA_cos, phi_fin_cos, vmag_cos, Norm_load_cos, Tang_load_cos, gamma_cos=BEM(r_R_cos, NBlades, Omega, Radius, Uinf, TipLocation_R, RootLocation_R, TSR, corr='true')
#     #plt.plot(r_R_cent_el, a_fin_el)
#     #plt.plot(r_R_cent_el, a_tan_fin_el)
#     CT_total_el[i]=np.sum(Norm_load_el*NBlades/((n**2) *((2*Radius)**4)))
#     CT_total_cos[i] = np.sum(Norm_load_cos*NBlades/((n**2) *((2*Radius)**4)))


# # In[17]:


# #print(CT_total_el)
# plt.plot(el, CT_total_el, marker='o',label = r'Constant spacing' )  
# plt.plot(el, CT_total_cos, marker='*', label = r'Cosine spacing' )  
# plt.title(r'Convergence History of $C_T$') 
# plt.xlabel("Number of Annuli") 
# plt.ylabel(r'$C_T$')  
# plt.grid(1) 
# plt.legend()
# #plt.savefig('/home/kjm2004/RotorWake/Convergence.pdf', bbox_inches='tight')
# plt.show()

# plt.figure()

# plt.plot(r_R_cent_el, a_fin_el,label = r'a for const. spacing' )
# plt.plot(r_R_cent_el, a_tan_fin_el, label = r'a` for const. spacing')
# plt.plot(r_R_cent_el, a_fin_cos, label = r'a for cosine spacing' )
# plt.plot(r_R_cent_el, a_tan_fin_cos, label = r'a`for cosine spacing' )
# plt.xlabel('r/R')
# plt.ylabel(r' Induction factors ($a$)')
# plt.ylim(0, 0.5)
# plt.title(r'Spanwise distribution of Induction factor ($a, a`$)' )
# plt.legend(loc='upper left')
# #plt.savefig('/home/kjm2004/RotorWake/spacing.pdf', bbox_inches='tight')

