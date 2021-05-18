import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from generate_mellor_like_bursts import *

#### Global constants:
rtol = 1e-6
atol = 1e-6 
F = 96485.33 ## Coulomb/mole
Nav = 6.022e23
e = 2.718

class ca1_spine:
    
    def __init__(self, g_N, nRyR, n_ip3, Vsoce_0, tau_refill_0, Vspine, scDep):
        """initialize the spine parameters and volume/surface area-dependent constants: 
        params: (nmda conductance (pS), #ryr, #ip3rs, Vsoce, tau_refill (s), Vspine(um^3))"""
        #self.g_N = float(g_N) * 1e-12 # pS
        self.g_N = float(g_N) * 1e-12 #S Use baseline as 675 pS
        self.ryr_tot = int(nRyR)
        self.ip3r_tot = int(n_ip3)
        self.scDep = float(scDep)
        #self.Vscale = float(Vscale)
        #self.k_sk = float(k_sk)
        self.Vspine = float(Vspine) ## in um^3
        self.Vspine_um = self.Vspine ## use this variable to get Vspine in um anywhere required
        self.d_spine = (6*self.Vspine/3.14)**0.333  ## um
        self.Aspine = 3.14 * self.d_spine**2 * 1e-8  ## cm^2
        self.Vspine = self.Vspine * 1e-15 ## liter
        self.Aer = 0.1 * self.Aspine  ## cm^2
        self.Ver = 0.1 * self.Vspine ## liter
        self.Vspine = self.Vspine - self.Ver   ## liter
        self.Ver_0 = 0.006 #um^3
        #self.Vspine_um - self.Vspine * 1e15 ##um^3 for scaling AMPA conductance
            ## Total concentration of PMCA and NCX pumps in the spine head (uM):
        #self.pHtot = (1e14) * 1000 * self.Aspine/(self.Nav * self.Vspine)
        #self.pLtot = (1e14) * 140 * self.Aspine/(self.Nav * self.Vspine)
                ######################################################################################
        self.g_N_Ca = 0.1 * (self.g_N/(2*F*78.0*self.ca_ext)) * 1e6   ## Ca conductance of NMDAR channels; liter/sec
        self.k_erleak = self.Vmax_serca * (self.ca_0**2)/((self.Kd_serca**2 + self.ca_0**2)*(self.ca_er_0 - self.ca_0)) #+ self.alpha_ip3r * self.ip3r_tot * (((0.1/(0.1+self.d1))*(0.05/(0.05 + self.d5))*1)**3) * (self.ca_er_0 - self.ca_0)/(self.Nav * self.Vspine * 0.1) ## /s
        #self.g_A =  ((self.Vspine_um * 1e2 - 1.0)/20.0) * 1e-9 ## S    #formerly: 0.5e-9 ## Vspine = Vmin * (1 + 20gA), Vmin = 0.01 um^3 
        self.Vsoc = float(Vsoce_0) * (0.49/self.d_spine)
        self.tau_refill = float(tau_refill_0) * (self.Ver * 1e15/self.Ver_0)
        self.g_A = ((self.Vspine_um - 0.001) / 0.24) * 1e-9 #S
        self.rhov0, self.rhovp0 = [self.rho_0*self.pp0/(self.pp0 + self.PK(self.c40)), self.rho_0*self.PK(self.c40)/(self.pp0 + self.PK(self.c40))]
        self.struct_plas_flag = 0
        
        print('gAMPA = {} S'.format(self.g_A))
        
        
        ##############################################################################################
    #### Defining various functions used in model simulation:

    ##############################################################################################
    #### Temporal profile of glutamate availability:

    def glu(self, flag, t):

        tau_glu = 1e-3 ## sec
        glu_0 = 2.718 * 300 ## uM

        if flag == 0: 
            return 0
        else: 
            total = 0
            for tpuff in self.tpre:    
                if t > tpuff: total += glu_0 * np.exp(-(t-tpuff)/tau_glu) * ((t-tpuff)/tau_glu)
            return total
    ##############################################################################################

    ##############################################################################################
    #### Voltage profile of BAP at the dendritic compartment:

    def u_bpap(self, t):

        V0 = 67
        total = 0
        for tbp in self.tpost:
            if t > tbp: total += V0 * (0.7 * np.exp(-(t-tbp)/0.003) + 0.3 * np.exp(-(t-tbp)/0.04))
        return self.E_L + total
    ##############################################################################################

    ##############################################################################################    
    #### AMPAR conductance profile: 

    def I_A(self,flag,u,t):

        if flag==0:
            return 0
        else:
            total = 0
            for tpuff in self.tpre:
                if t>tpuff: total += self.g_A * (np.exp(-(t-tpuff)/self.tau_A2) - np.exp(-(t-tpuff)/self.tau_A1))  
            return total * (u - self.E_A)
    ##############################################################################################

    ##############################################################################################     
    #### NMDAR conductance profile:

    def xr(self, flag, t):
        if flag==0:
            return 0
        else:
            tot = 0
            for tpuff in self.tpre:
                if t>tpuff: tot += np.exp(-(t-tpuff)/self.tau_r)
            return 100*tot
        
    ##############################################################################################        
    #### Plasticitry model, Omega function

    def wfun(self,x):

        U = -self.beta2*(x - self.alpha2)
        V = -self.beta1*(x - self.alpha1)
        if U>100: U = 100
        if V>100: V = 100        
        return (1.0/(1 + np.exp(U))) - 0.5*(1.0/(1 + np.exp(V)))
    ##############################################################################################

    ##############################################################################################    
    #### Plasticity model, tau function

    def wtau(self,x):

        return self.P1 + (self.P2/(self.P3 + (2*x/(self.alpha1+self.alpha2))**self.P4))
    ##############################################################################################
    
    ##############################################################################################
    ##### PKA, protein-K (PK), CaN functions
    def PKA(self, cam4):
        """gives PKA activity corresponding as a function of fully bound CaM (cam4)"""

        kh = 1
        n = 8
        return 0.0036 + 100*(cam4**n)/((cam4**n)+(kh**n))
    
    def PK(self, cam4): ## This is the K in the figure
        """gives protein-K activity corresponding as a function of fully bound CaM (cam4)"""
        K0 = 1e-8
        Kmax = 1.0
        Khalf = 0.1 ## Sets the LTP threshold
        a = 4
        return K0 + Kmax * (cam4**a) / ((cam4**a) + (Khalf**a))  

    def CaN(self, cam4):
        """gives calcineurin activity as a function of fully bound CaM (cam4)"""
        #kh = 0.3
        #return 1*(ca**3)/((ca**3)+(kh**3))
        kh = 2.05e-4
        n = 0.52
        return 0.1 + 18.0/(1.0 + (kh/(cam4+1e-12))**n)
    
    def aa(self):
        return "yes"

    
        #########################################################################################################################       
    #### Coupled ODEs describing the ER-bearing spine head, which is resistively coupled to a dendritic compartement via a passive neck:       

    def spine_model(self, x , t):
        """ODE-based model of ER-bearing CA1 spine"""

        Ract, Gact, PLCact, IP3, IP3K, IP3K_2ca, IP5P,\
        ryrC1, ryrC2, ryrC3, ryrC4, ryrO1, ryrO2, ryrO3, pH, pL, cbp, Bslow, calb, calb_m1, calb_h1, calb_m2, calb_h2, calb_m1h1, calb_m2h1, calb_m1h2, c1n0, c2n0, c0n1, c0n2, c1n1, c2n1, c1n2, c2n2, erB1, erB2, I1p, PP1, gv, gvp, gsp, sr, ir,\
         Psoc, mv, hv, w, u, ud, h, ca_er, ca = x
        
        if self.struct_plas_flag:
            self.Vspine = 0.001 + 0.24*gsp
            self.g_A = gsp*(1e-9)  ## nS -> S
            Vs = 0.001 + 0.24*gsp

            ## Spine compartment and ER size:
            self.d_spine = (6*self.Vspine/3.14)**0.333  ## um
            self.Aspine = 3.14 * (self.d_spine**2) * (1e-8)  ## cm^2
            self.Vspine = self.Vspine * (1e-15) ## liter
            self.Aer = 0.1 * self.Aspine  ## cm^2
            self.Ver = 0.1 * self.Vspine  ## liter
            self.Vspine = self.Vspine - self.Ver  ## liter

            ##Default: d_spine = 0.49 um; Ver = 0.001 um^3 
            self.Vsoc = 100*(0.49/self.d_spine)
            self.tau_refill = .1*((1e15)*self.Ver/0.006) ##0.01*((1e15)*Ver/0.001)
        else: 
            Vs = self.Vspine_um

        nt = self.glu(self.flag, t)

        if self.flag and self.input_pattern=='stdp': ud = self.u_bpap(t)
        else: u,ud = [self.E_L + self.scDep,self.E_L + self.scDep]
 
        ## mGluR-IP3 pathway:    

        IP3_IP3K_2ca = self.IP3K_tot - IP3K - IP3K_2ca
        IP3_IP5P = self.IP5P_tot - IP5P

        Ract_eq = self.glu(self.flag, t) * self.kR * (1-Ract) - Ract/self.tau_R
        Gact_eq = self.kG * Ract * (1-Gact) - Gact/self.tau_G
        PLCact_eq = self.kP * Gact * (1-PLCact) - PLCact/self.tau_P

        IP3_eq = self.k_ip3 * (0.49/self.d_spine) * PLCact #hill(Gact) 
        IP3_eq += -100*IP3K_2ca*IP3 + 80*IP3_IP3K_2ca - 9*IP5P*IP3 + 72*IP3_IP5P - ((10.0/220.0)*(4.4e-15)/self.Vspine)*(IP3 - self.IP3_0)
        IP3K_2ca_eq = +1111*IP3K*ca*ca - 100*IP3K_2ca - 100*IP3K_2ca*IP3 + 80*IP3_IP3K_2ca + 20*IP3_IP3K_2ca
        IP5P_eq = -9*IP5P*IP3 + 72*IP3_IP5P + 18*IP3_IP5P
        IP3K_eq = -1111*IP3K*ca*ca + 100*IP3K_2ca 

        ca_eq = -1111*IP3K*ca*ca - 1111*IP3K*ca*ca + 100*IP3K_2ca + 100*IP3K_2ca


#         R_Gq_eq = -self.a2f * R_Gq * nt + self.a2b * glu_R_Gq + self.a3f * R *Gq - self.a3b * R_Gq
#         Gact_eq = +self.a5 * glu_R_Gq + self.a6*Gq - self.a7 * Gact - self.b3f * Gact * PLC_PIP2 + self.b3b * Gact_PLC_PIP2 - self.b4f * Gact * ca_PLC_PIP2 + self.b4b * ca_Gact_PLC_PIP2 - self.b5f * ca_PLC * Gact + self.b5b * ca_Gact_PLC

#         IP3_eq = +self.b6 * ca_PLC_PIP2 + self.b7 * ca_Gact_PLC_PIP2 - 100 * IP3K_2ca * IP3 + 80 * IP3_IP3K_2ca - 9 * IP5P * IP3 + 72 * IP3_IP5P #+ 1.2

#         ca_Gact_PLC_PIP2_eq = +self.b2f * ca * Gact_PLC_PIP2 - self.b2b * ca_Gact_PLC_PIP2 + self.b4f * Gact * ca_PLC_PIP2 - self.b4b * ca_Gact_PLC_PIP2 - self.b11 * ca_Gact_PLC_PIP2 + self.b9f * ca_Gact_PLC * self.PIP2 - self.b9b * ca_Gact_PLC_PIP2 - self.b7 * ca_Gact_PLC_PIP2
#         DAGdegr_eq = +self.DAGdegrate*DAG
#         PLC_PIP2_eq = -self.b1f*ca*PLC_PIP2 + self.b1b*ca_PLC_PIP2 - self.b3f*Gact*PLC_PIP2 + self.b3b*Gact_PLC_PIP2 + self.b10*Gact_PLC_PIP2
#         DAG_eq = +self.b6*ca_PLC_PIP2 + self.b7*ca_Gact_PLC_PIP2 - self.DAGdegrate*DAG
#         IP3_IP5P_eq = +9*IP5P*IP3 - 72*IP3_IP5P - 18*IP3_IP5P
#         IP3degr_eq = +20*IP3_IP3K_2ca + 18*IP3_IP5P
#         glu_R_Gq_eq = +self.a2f*R_Gq*nt - self.a2b*glu_R_Gq + self.a4f*glu_R*Gq - self.a4b*glu_R_Gq - self.a5*glu_R_Gq
#         Gbc_eq = +self.a5*glu_R_Gq + self.a6*Gq - self.a8*GaGDP*Gbc
#         ca_PLC_eq = -self.b8f*ca_PLC*self.PIP2 + self.b8b*ca_PLC_PIP2 + self.b6*ca_PLC_PIP2 - self.b5f*ca_PLC*Gact + self.b5b*ca_Gact_PLC + self.b12*ca_Gact_PLC
#         IP3_IP3K_2ca_eq = +100*IP3K_2ca*IP3 - 80*IP3_IP3K_2ca - 20*IP3_IP3K_2ca
#         R_eq = -self.a1f*R*nt + self.a1b*glu_R - self.a3f*R*Gq + self.a3b*R_Gq
#         ca_PLC_PIP2_eq = +self.b1f*ca*PLC_PIP2 - self.b1b*ca_PLC_PIP2 - self.b4f*Gact*ca_PLC_PIP2 + self.b4b*ca_Gact_PLC_PIP2 + self.b11*ca_Gact_PLC_PIP2 + self.b8f*ca_PLC*self.PIP2 - self.b8b*ca_PLC_PIP2 - self.b6*ca_PLC_PIP2
#         IP3K_2ca_eq = +1111*IP3K*ca*ca - 100*IP3K_2ca - 100*IP3K_2ca*IP3 + 80*IP3_IP3K_2ca + 20*IP3_IP3K_2ca
#         Gact_PLC_PIP2_eq = -self.b2f*ca*Gact_PLC_PIP2 + self.b2b*ca_Gact_PLC_PIP2 + self.b3f*Gact*PLC_PIP2 - self.b3b*Gact_PLC_PIP2 - self.b10*Gact_PLC_PIP2
#         Gq_eq = -self.a3f*R*Gq + self.a3b*R_Gq - self.a4f*glu_R*Gq + self.a4b*glu_R_Gq - self.a6*Gq + self.a8*GaGDP*Gbc
#         IP5P_eq = -9*IP5P*IP3 + 72*IP3_IP5P + 18*IP3_IP5P
#         GaGDP_eq = +self.a7*Gact - self.a8*GaGDP*Gbc + self.b10*Gact_PLC_PIP2 + self.b11*ca_Gact_PLC_PIP2 + self.b12*ca_Gact_PLC
#         ca_Gact_PLC_eq = -self.b9f*ca_Gact_PLC*self.PIP2 + self.b9b*ca_Gact_PLC_PIP2 + self.b7*ca_Gact_PLC_PIP2 + self.b5f*ca_PLC*Gact - self.b5b*ca_Gact_PLC - self.b12*ca_Gact_PLC
#         glu_R_eq = +self.a1f*R*nt - self.a1b*glu_R - self.a4f*glu_R*Gq + self.a4b*glu_R_Gq + self.a5*glu_R_Gq
#         IP3K_eq = -1111*IP3K*ca*ca + 100*IP3K_2ca

#         ca_eq = (-self.b1f*ca*PLC_PIP2 - self.b2f*ca*Gact_PLC_PIP2 - 1111*IP3K*ca*ca - 1111*IP3K*ca*ca + (self.b1b*ca_PLC_PIP2 + self.b2b*ca_Gact_PLC_PIP2 + 100*IP3K_2ca+100*IP3K_2ca)) 

        ## IP3 receptor kinetics:

        x = IP3/(IP3 + self.d1)
        y = ca/(ca + self.d5)
        Q2 = self.Kinh #(0.1+IP3)/(0.9+IP3)#Kinh
        h_eq = self.a2*(Q2 - (Q2+ca)*h)

        ca_eq += self.ip3r_tot * ((x*y*h)**3) * self.alpha_ip3r * (ca_er - ca)/(Nav * self.Vspine) 

        ca_er_eq = -self.alpha_ip3r * self.ip3r_tot * ((x*y*h)**3) * (ca_er - ca)/(Nav * self.Ver)  +  (self.ca_er_0 - ca_er)/self.tau_refill

        ## RyR/CICR kinetics:

        ryrC5 = 1.0 - (ryrC1 + ryrC2 + ryrC3 + ryrC4 + ryrO1 + ryrO2 + ryrO3)
        ryrC1_eq = -self.kryrc1c2*ca*ryrC1 + self.kryrc2c1*ryrC2
        ryrC2_eq = self.kryrc1c2*ca*ryrC1 - self.kryrc2c1*ryrC2 - self.kryrc2c3*ca*ryrC2 + self.kryrc3c2*ryrC3 - self.kryrc2c5*ryrC2 + self.kryrc5c2*ryrC5
        ryrC3_eq = self.kryrc2c3*ca*ryrC2 - self.kryrc3c2*ryrC3 - self.kryrc3o1*ryrC3 + self.kryro1c3*ryrO1 - self.kryrc3o2*ryrC3 + self.kryro2c3*ryrO2 - self.kryrc3o3*ryrC3 + self.kryro3c3*ryrO3
        ryrC4_eq = self.kryro2c4*ryrO2 - self.kryrc4o2*ryrC4 + self.kryro3c4*ryrO3 - self.kryrc4o3*ryrC4
        ryrO1_eq = self.kryrc3o1*ryrC3 - self.kryro1c3*ryrO1
        ryrO2_eq = self.kryrc3o2*ryrC3 - self.kryro2c3*ryrO2 - self.kryro2c4*ryrO2 + self.kryrc4o2*ryrC4
        ryrO3_eq = self.kryrc3o3*ryrC3 - self.kryro3c3*ryrO3 - self.kryro3c4*ryrO3 + self.kryrc4o3*ryrC4
        ryr_eq =  [ryrC1_eq, ryrC2_eq, ryrC3_eq, ryrC4_eq, ryrO1_eq, ryrO2_eq, ryrO3_eq]

        ca_eq += self.ryr_tot * (ryrO1+ryrO2+ryrO3) * self.alpha_ryr * (ca_er - ca)/(Nav * self.Vspine)

        ca_er_eq += -self.ryr_tot * (ryrO1+ryrO2+ryrO3) * self.alpha_ryr * (ca_er - ca)/(Nav * self.Ver) 

        rho_ryr = (1e6)*self.ryr_tot/(Nav * self.Vspine)
        ca_eq += (-self.kryrc1c2*ca*ryrC1 + self.kryrc2c1*ryrC2 - self.kryrc2c3*ca*ryrC2 + self.kryrc3c2*ryrC3) * rho_ryr

        ## Buffer equations:

        Bslow_eq = -self.kslow_f*Bslow*ca + self.kslow_b*(self.Bslow_tot - Bslow)
        ca_eq += -self.kslow_f*Bslow*ca + self.kslow_b*(self.Bslow_tot - Bslow)

        cbp_eq = -self.kbuff_f*ca*cbp + self.kbuff_b*(self.cbp_tot - cbp)
        ca_eq += -self.kbuff_f*ca*cbp + self.kbuff_b*(self.cbp_tot - cbp)    

        calb_m2h2 = self.calb_tot - calb - calb_m1 - calb_h1 - calb_m2 - calb_h2 - calb_m1h1 - calb_m2h1 - calb_m1h2
        calb_eqs = [ -ca*calb*(self.km0m1 + self.kh0h1) + self.km1m0*calb_m1 + self.kh1h0*calb_h1,\
                         ca*calb*self.km0m1 - self.km1m0*calb_m1 + calb_m2*self.km2m1 - ca*calb_m1*self.km1m2 + calb_m1h1*self.kh1h0 - ca*calb_m1*self.kh0h1,\
                         ca*calb*self.kh0h1 - self.kh1h0*calb_h1 + calb_h2*self.kh2h1 - ca*calb_h1*self.kh1h2 + calb_m1h1*self.km1m0 - ca*calb_h1*self.km0m1,\
                         ca*calb_m1*self.km1m2 - self.km2m1*calb_m2 + self.kh1h0*calb_m2h1 - ca*self.kh0h1*calb_m2,\
                         ca*calb_h1*self.kh1h2 - self.kh2h1*calb_h2 + self.km1m0*calb_m1h2 - ca*self.km0m1*calb_h2,\
                         ca*(calb_h1*self.km0m1 + calb_m1*self.kh0h1) - (self.km1m0+self.kh1h0)*calb_m1h1 - ca*calb_m1h1*(self.km1m2+self.kh1h2) + self.kh2h1*calb_m1h2 + self.km2m1*calb_m2h1,\
                         ca*self.km1m2*calb_m1h1 - self.km2m1*calb_m2h1 + self.kh2h1*calb_m2h2 - self.kh1h2*ca*calb_m2h1 + self.kh0h1*ca*calb_m2 - self.kh1h0*calb_m2h1,\
                         ca*self.kh1h2*calb_m1h1 - self.kh2h1*calb_m1h2 + self.km2m1*calb_m2h2 - self.km1m2*ca*calb_m1h2 + self.km0m1*ca*calb_h2 - self.km1m0*calb_m1h2 ]
        ca_eq += -ca*(self.km0m1*(calb+calb_h1+calb_h2) + self.kh0h1*(calb+calb_m1+calb_m2) + self.km1m2*(calb_m1+calb_m1h1+calb_m1h2) + self.kh1h2*(calb_h1+calb_m1h1+calb_m2h1))+\
                    self.km1m0*(calb_m1+calb_m1h1+calb_m1h2) + self.kh1h0*(calb_h1+calb_m1h1+calb_m2h1) + self.km2m1*(calb_m2+calb_m2h1+calb_m2h2) + self.kh2h1*(calb_h2+calb_m1h2+calb_m2h2)

        ##ER Ca2+ buffer:

        erB1_eq = -self.kerb1_f*erB1*ca_er + self.kerb1_b*(self.erB1_tot - erB1)
        erB2_eq = -self.kerb2_f*erB2*ca_er + self.kerb2_b*(self.erB2_tot - erB2)
        ca_er_eq += -self.kerb1_f*erB1*ca_er + self.kerb1_b*(self.erB1_tot - erB1) - self.kerb2_f*erB2*ca_er + self.kerb2_b*(self.erB2_tot - erB2)

        ## Ca2+/calmodulin kinetics:

        c0n0 = self.cam_tot - c1n0 - c2n0 - c0n1 - c0n2 - c1n1 - c2n1 - c1n2 - c2n2
        c1n0_eq = -(self.k2c_on*ca + self.k1c_off + self.k1n_on*ca)*c1n0 + self.k1c_on*ca*c0n0 + self.k2c_off*c2n0 + self.k1n_off*c1n1
        c2n0_eq = -(self.k2c_off + self.k1n_on*ca)*c2n0 + self.k2c_on*ca*c1n0 + self.k1n_off*c2n1
        c0n1_eq = -(self.k2n_on*ca + self.k1n_off + self.k1c_on*ca)*c0n1 + self.k1n_on*ca*c0n0 + self.k2n_off*c0n2 + self.k1c_off*c1n1
        c0n2_eq = -(self.k2n_off + self.k1c_on*ca)*c0n2 + self.k2n_on*ca*c0n1 + self.k1c_off*c1n2
        c1n1_eq = -(self.k2c_on*ca + self.k1c_off + self.k1n_off + self.k2n_on*ca)*c1n1 + self.k1c_on*ca*c0n1 + self.k1n_on*ca*c1n0 + self.k2c_off*c2n1 + self.k2n_off*c1n2
        c2n1_eq = -(self.k2c_off + self.k2n_on*ca)*c2n1 + self.k2c_on*ca*c1n1 + self.k2n_off*c2n2 + self.k1n_on*ca*c2n0 - self.k1n_off*c2n1
        c1n2_eq = -(self.k2n_off + self.k2c_on*ca)*c1n2 + self.k2n_on*ca*c1n1 + self.k2c_off*c2n2 + self.k1c_on*ca*c0n2 - self.k1c_off*c1n2
        c2n2_eq = -(self.k2c_off + self.k2n_off)*c2n2 + self.k2c_on*ca*c1n2 + self.k2n_on*ca*c2n1
        cam_eqs = [c1n0_eq, c2n0_eq, c0n1_eq, c0n2_eq, c1n1_eq, c2n1_eq, c1n2_eq, c2n2_eq]
        ca_eq += -ca*(self.k1c_on*(c0n0+c0n1+c0n2) + self.k1n_on*(c0n0+c1n0+c2n0) + self.k2c_on*(c1n0+c1n1+c1n2) + self.k2n_on*(c0n1+c1n1+c2n1)) + \
        self.k1c_off*(c1n0+c1n1+c1n2) + self.k1n_off*(c0n1+c1n1+c2n1) + self.k2c_off*(c2n0+c2n1+c2n2) + self.k2n_off*(c0n2+c1n2+c2n2)

        ## PMCA/NCX kinetics:

        #ca_eq += pH*kH_leak - ca*pH*k1H + k2H*(pHtot - pH)  +  pL*kL_leak - ca*pL*k1L + k2L*(pLtot - pL)
        pH_eq = 0#k3H*(pHtot - pH) - ca*pH*k1H + k2H*(pHtot - pH)
        pL_eq = 0#k3L*(pLtot - pL) - ca*pL*k1L + k2L*(pLtot - pL)

        ## Extrusion kinetics:
        ca_eq += -(6.0/self.d_spine)*200*5*((ca/(ca+20.0)) - (self.ca_0/(self.ca_0+20.0)))  ## Low-aff pump
        ca_eq += -(6.0/self.d_spine)*200*0.5*((ca/(ca+0.5)) - (self.ca_0/(self.ca_0+0.5))) ## High-aff pump

        ca_eq += -((4.4e-15)/self.Vspine)*(ca - self.ca_0)  ## Diffusion into dendrite via neck

        ## SERCA kinetics:

        ca_eq += -self.Vmax_serca * (ca**2)/((self.Kd_serca**2) + (ca**2)) + self.k_erleak*(ca_er - ca)

        ## SOCE kinetics:
        Psoc_eq = (((self.Ksoc**4)/(self.Ksoc**4 + ca_er**4)) - Psoc)/self.tau_soc
        ca_eq += self.Vsoc * Psoc

        ## VGCC equations:

        mv_eq = ((1.0/(1 + np.exp(-(u-self.um)/self.kmv))) - mv)/self.tau_mv
        hv_eq = ((1.0/(1 + np.exp(-(u-self.uh)/self.khv))) - hv)/self.tau_hv
        I_vgcc = -0.001 * Nav * (3.2e-19) * self.g_vgcc * (mv**2) * hv * 0.078 * u * (ca - self.ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u))

        ## NMDA-R kinetics:

        sr_eq = self.xr(self.flag, t)*(1-sr-ir) - (sr/self.tau_d) - self.k_des*sr
        ir_eq = self.k_des*sr - self.k_rec*ir

        Inmda = self.g_N * sr * (u - self.E_N)/(1.0 + 0.28 * np.exp(-0.062 * u))

        ## Spine and dendrite voltage eqns:

    #     if dep_flag:
    #         sp_hh_eq = 0
    #         dend_hh_eq = 0
    #     else: 
        sp_hh_eq = -(1/self.Cmem) * ( self.g_L*(u - self.E_L) + (self.I_A(self.flag,u,t)/self.Aspine) + (Inmda/self.Aspine) - (self.gc/self.Aspine)*(ud - u) - I_vgcc/self.Aspine)
            #sp_hh_eq = -(1/Cmem) * ( g_L*(u - E_L) + I_A(s,u,t)/Aspine + I_N(s,u,t)/Aspine - (gc/Aspine)*(ud - u) - I_vgcc/Aspine)
        dend_hh_eq = -(1/self.Cmem) * ( self.g_L*(ud - self.E_L) + self.rho_spines*self.gc*(ud - u))

        ## Ca2+ influx through NMDAR and VGCC:

        ca_eq += -(self.g_N_Ca/self.Vspine) * (Inmda/(self.g_N*(u - self.E_N))) * 0.078 * u * (ca - self.ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u)) \
                -(self.g_vgcc/self.Vspine) * (mv**2) * hv * 0.078 * u * (ca - self.ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u))
        #ca_eq += -(g_N_Ca/Vspine) * (I_N(s,u,t)/(g_N*(u - E_N))) * 0.078 * u * (ca - ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u)) \
        #        -(g_vgcc/Vspine) * (mv**2) * hv * 0.078 * u * (ca - ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u))   

        ## Equation for plasticity variable w:

        acam = self.cam_tot - c0n0    
        w_eq = (1.0/self.wtau(acam))*(self.wfun(acam) - w)
        
        ## Receptor exchange b/w bulk and surface:

        PP1_I1p = self.PP1tot - PP1

        I1p_eq = self.PKA(c2n2)*self.I1 - I1p*self.CaN(c2n2) -self.kf*I1p*PP1 + self.kb*PP1_I1p
        PP1_eq = -self.kf*I1p*PP1 + self.kb*PP1_I1p

        rhov = gv/Vs
        rhovp = gvp/Vs

        K = self.PK(c2n2)
        P = self.dephos_factor*PP1

        gv_eq = -K*gv + P*gvp + P*gsp + (self.rhov0 - rhov)*Vs/self.tauSD  ##diff_factor ##(rhov0 - rhov)*Vs*Nav/tauSD

        gvp_eq = K*gv - P*gvp - (self.ka*gvp/Vs) + self.kd*gsp + (self.rhovp0 - rhovp)*Vs/self.tauSD  ##diff_factor  ##(rhovp0 - rhovp)*Vs*Nav/tauSD

        gsp_eq = -P*gsp + (self.ka*gvp/Vs) - self.kd*gsp

        return [Ract_eq,Gact_eq,PLCact_eq,IP3_eq,IP3K_eq,IP3K_2ca_eq,IP5P_eq] + ryr_eq + [pH_eq, pL_eq, cbp_eq, Bslow_eq] + calb_eqs + cam_eqs + [erB1_eq, erB2_eq] + [I1p_eq, PP1_eq, gv_eq, gvp_eq, gsp_eq] + [sr_eq, ir_eq] + [Psoc_eq] + [mv_eq, hv_eq] + [w_eq] + [sp_hh_eq, dend_hh_eq,\
                h_eq, ca_er_eq, ca_eq]


    ################################### Defining Constants #######################################
    ## Receptor plasticity model:
    tauSD = 60.0  ## sec
    dephos_factor = 0.1 ## OPEN PARAM
    ka = 0.1  ## liter/sec
    kd = 0.0001  ## per sec
    pp0 = 0.004*dephos_factor
    c40 = 7.4e-8 
    rho_0 = 10 ## OPEN PARAM
#     rhov0,rhovp0 = [rho_0*pp0/(pp0 + PK(0,c40)), rho_0*PK(0,c40)/(pp0 + PK(0,c40))]

    ## PP1 kinetics:
    kf, kb = [500.0,0.1]
    PP1tot = 0.2
    I1 = 1.0

        ## Reaction paramters for ER buffer:
    kerb1_f,kerb1_b = [0.1,200] ## /uM/s,/s
    erB1_tot = 3600*30  ## uM
    kerb2_f,kerb2_b = [100,1000] ## /uM/s,/s
    erB2_tot = 3600  ## uM
    
    ## Reaction parameters for mGluR_IP3 pathway:
    tau_glu = 0.001

    tau_R = 0.5#0.7
    tau_G = .4 
    tau_P = .3

    kR = 1.0
    kG = 1.0
    kP = 4.0

    k_ip3 = 180.#210.

    ## Parameters for IP3R model (Fink et al., 2000 and Vais et al., 2010):
    Kinh = 0.2  ## uM
    d1 = 0.8 ## uM
    d5 = 0.3 ## uM
    a2 = 2.7 ## /uM/s
    alpha_ip3r = (0.15/3.2)*(1e7)*(1e6)/500.0  ## /uM/sec

        # Parameters for Saftenku et al. RyR model:
    kryrc1c2 = 1.24 #2.5  /uM/s
    kryrc2c1 = 13.6 #13.3
    kryrc2c3 = 29.8 #68 /uM/s
    kryrc3c2 = 3867 #8000
    kryrc3o2  = 24.5 #17
    kryro2c3 = 156.5 #92
    kryro2c4 = 1995 #1900
    kryrc4o2 = 415.3 #520
    kryrc3o3 = 8.5 #14
    kryro3c3 = 111.7 #138
    kryro3c4 = 253.3 #300
    kryrc4o3 = 43 #46
    kryrc3o1 = 731.2 #1100
    kryro1c3 = 4183 #3400
    kryrc2c5 = 1.81 #0.13
    kryrc5c2 = 3.63 #3.6
    alpha_ryr = (0.25/3.2)*(1e7)*(1e6)/500.0   ## /uM/sec
    
        ## Parameters for endogenous immobile buffer (CBP): 
    kbuff_f = 247 ## /uM/s
    kbuff_b = 524 ## /s

    ## Parameters for endogenous slow buffer:
    kslow_f = 24.7 ## /uM/s
    kslow_b = 52.4 ## /s

    ## Parameters for calbindin-Ca2+ kinetics:
    km0m1=174 ## /uM/s
    km1m2=87 ## /uM/s
    km1m0=35.8 ## /s
    km2m1=71.6 ## /s
    kh0h1=22 ## /uM/s
    kh1h2=11 ## /uM/s
    kh1h0=2.6 ## /s
    kh2h1=5.2 ## /s
    
        ## Parameters for PMCA and NCX pumps:
    k1H,k2H,k3H,kH_leak = [150,15,12,3.33]  ## (/uM/s, /s, /s, /s)
    k1L,k2L,k3L,kL_leak = [300,300,600,10]  ## (/uM/s, /s, /s, /s)

    ## Parameters for CaM-Ca2+ interaction:
    k1c_on = 6.8  ## /uM/s
    k1c_off = 68  ## /s
    k2c_on = 6.8 ## /uM/s
    k2c_off = 10 ## /s
    k1n_on = 108 ## /uM/s
    k1n_off = 4150 ## /s
    k2n_on = 108 ## /uM/s
    k2n_off = 800 ## /s

    ## Membrane and leak parameters:
    Cmem = 1e-6 ##  F/cm^2
    g_L = 2e-4  ## S/cm^2
    E_L = -70   ## mV

    ## AMPA receptor parameters:
    tau_A1 = 0.2e-3 ## s
    tau_A2 = 2e-3  ## s
    E_A = 0  ## mV
    #g_A = 0.5e-9  ## S

    ## NMDA receptor parameters:
    k_des,k_rec = [10.0,2.0]
    tau_r = 2e-3
    tau_d = 89e-3
    #tau_N1 = 5e-3 ## s
    #tau_N2 = 50e-3 ## s
    E_N = 0  ## mV

    ## L-VGCC parameters:
    um = -20 ## mV
    kmv = 5  ## mV
    tau_mv = 0.08e-3 ## sec
    uh = -65  ## mV
    khv = -7 ## mV
    tau_hv = 300e-3  ## sec

    ## Spine neck parameters:
    Rneck = 1e8  ## Ohm
    gc = 1.0/Rneck ## S
    rho_spines = 0#1.5e6 ## Surface density of co-active SC synaptic inputs on dendritic compartment (cm^-2)

    ## SERCA kinetic parameters:
    Vmax_serca = 1  ## uM/sec
    Kd_serca = 0.2 ## uM

    ## SOCE paramters:
    Ksoc = 50.0 ## uM
    tau_soc = 0.1  ## sec
    #Vsoc = 1.  ## uM/sec

    ## Parameters for Ca2+-based plasticity model:
    P1,P2,P3,P4 = [1.0,10.0,0.001,2]
    beta1,beta2 = [60,60]  ## /uM
    alpha1,alpha2 = [6.04641385849974,25.688044233956496] ## uM

        #########################################################
    ########### Concentrations of various species ###########
    #########################################################

    ## External Ca (uM):
    ca_ext = 2e3

    ## Resting cytosolic Ca (uM):
    ca_0 = 0.05

    ## Resting Ca in ER (uM):
    ca_er_0 = 250. 

    ## Total calbindin concentration in spine (uM):
    calb_tot = 45.

    ## Total CBP concentration in the spine (uM):
    cbp_tot = 80.

    ## Total slow buffer concentration in the spine (uM):
    Bslow_tot = 40.

    ## Total concentration of PMCA and NCX pumps in the spine head (uM):
    pHtot = 0 #(1e14) * 1000 * Aspine/(Nav * Vspine)
    pLtot = 0 #(1e14) * 140 * Aspine/(Nav * Vspine)

    ## Total concentration of CaM in spine (uM):
    cam_tot = 50.

    ## Total concentrations of IP3 3-kinase and IP3 5-phosphatase in the spine (uM):
    ip5pconc = 1.
    ip3kconc = 0.9
    IP3_0 = 0.1
    
    ## Total concentrations of IP3 3-kinase and IP3 5-phosphatase in the spine (uM):
    IP5P_tot = 1.
    IP3K_tot = 0.9
    
    ################################################################################################################
    ############################################### Experiments ##################################################
    ################### simulate ER+ spine without inputs to cnverge to steady state #############
    def get_resting_params(self):
        """simulates the spine in absence of inputs and returns the resting state params"""
        if self.ryr_tot == 0 and self.ip3r_tot == 0:
            self.buff_flag = 0
            self.Vmax_serca = 0
            self.k_erleak = 0
            self.V_socc_max = 0
            
        self.flag = 0
        self.buff_flag = 1
        self.g_vgcc = 0
        self.input_pattern="rdp"
        ##########################################################################################################
        ######################## Initializing all variables:######################################################
        ########################################################################################################

        mGluR_init = [0,0,0,self.IP3_0,0,0,0]
        ryr_init = [1,0,0,0,0,0,0]
        pumps_init = [self.pHtot, self.pLtot]
        buff_init =  [self.cbp_tot, self.Bslow_tot] + [self.calb_tot,0,0,0,0,0,0,0] 
        CaM_init = [0]*8  
        erB_init = [self.erB1_tot,self.erB2_tot]
        nmda_init = [0,0]
        soc_init = [0]
        vgcc_init = [0,1] 
        w_init = [0] 
        voltage_init = [self.E_L + self.scDep, self.E_L + self.scDep]
        h_init = [1]
        ca_init = [self.ca_er_0, self.ca_0]
        i1_pp1_init = [self.PKA(8e-8)*self.I1/self.CaN(8e-8),0.004] ### NEW
        recep_init = [self.rhov0*self.Vspine_um, self.rhovp0*self.Vspine_um,self.g_A*1e9] ### NEW

        xinit0 = mGluR_init + ryr_init + pumps_init + buff_init + CaM_init + erB_init + i1_pp1_init + recep_init + nmda_init + soc_init + vgcc_init + w_init + voltage_init + h_init + ca_init


        #print(xinit0)

        ################ solving #################################################
        t0 = np.arange(0., 1000., 0.01)
        sol0 = odeint(self.spine_model, xinit0, t0, rtol=rtol, atol=atol) #args=(self.Vspine, self.g_A)
        print("initial ER calcium = {}".format(sol0[-1,-2]))
        print("initial cyto calcium = {}".format(sol0[-1,-1]))
        print("Vspine={}".format(self.Vspine))


        return sol0[-1,:]
    
    
    ######################################### RDP ################################################################

    def do_rdp(self, f_input, n_inputs):
        """performs RDP 
        parameters: (frequency of stimulation, no. of presynaptic inputs)"""
        if self.ryr_tot == 0 and self.ip3r_tot == 0:
            self.buff_flag = 0
            self.Vmax_serca = 0
            self.k_erleak = 0
            self.V_socc_max = 0
            
        xinit = self.get_resting_params()
        self.flag = 1
        self.buff_flag = 1
        self.struct_plas_flag = 0
        self.tpre = [i/float(f_input) for i in range(n_inputs)]
        if f_input < 0.3: n_points = int(3e4)
        else: n_points = int(1e4)
            
        t = np.linspace(min(self.tpre), max(self.tpre), n_points)
        sol = odeint(self.spine_model,xinit, t, rtol = rtol, atol = atol) #, args=(self.Vspine, self.g_A)
        #plt.plot(sol[:,-11])
        ################ saving in a file ######################
        #fname = "rdp_out_vinit{}_nryr{}_f{}.csv".format(round(self.Vspine * 1e15, 2), self.nRyR, f_input)
        return sol
    
    ########################### realistic SC inputs ###############################################################
    
#     def realistic_tSpikes(self, avg_spikes, tBurst, f_burst_min, f_burst_max, max_inputs):
#         t_b_pre = get_tBurst(avg_spikes)
#         sp_train = get_ISI(tBurst, f_burst_min, f_burst_max, max_inputs)
#         return sp_train
    
    def stdp_realisitic_inputs(self, beta_pre, beta_post, f_burst_min, f_burst_max, max_inputs):
        """emulates the realistic Schaffer-Collateral place cell firing pattern as described in Issac et al. J.Neurosci 2009
        beta_pre, beta_post: average firing rate for pre- and post-synaptic APs
        f_burst_min, f_burst_max: min and max frequency of spikes in a burst. Avg freq of each burst is sampled from a uniform distribution [f_burst_min, f_burst_max]
        max_inputs: max inputs in a burst. For each burst no. of inputs is sampled from a uniform distribution [1, max_inputs]"""
        
        if self.ryr_tot == 0 and self.ip3r_tot == 0:
            self.buff_flag = 0
            self.Vmax_serca = 0
            self.k_erleak = 0
            self.V_socc_max = 0

        xinit = self.get_resting_params()
        
        t_burst_pre = get_tBurst(beta_pre)
        t_burst_post = get_tBurst(beta_post)
        self.tpre = get_ISI(t_burst_pre, f_burst_min, f_burst_max, max_inputs)
        self.tpost = get_ISI(t_burst_post, f_burst_min, f_burst_max, max_inputs)
        t_max = max(np.amax(self.tpre), np.amax(self.tpost))
        t_min = min(np.amin(self.tpre), np.amin(self.tpost))
        print(f"tmin = {t_min: .02f}, tmax = {t_max: .02f}")
        #print(f"tBpre = {t_burst_pre}, tBpost = {t_burst_post}")
        print(f"tpre = {self.tpre}, tpost = {self.tpost}")
        
        self.input_pattern = "stdp"
        self.flag = 1
        self.buff_flag = 1
        self.struct_plas_flag = 1
        
                
        n_points = int(2e4)
        
        t = np.linspace(t_min, t_max, n_points)
        
        stdp_sol = odeint(self.spine_model, xinit, t, rtol = rtol, atol = atol)
        
        return stdp_sol, t
    
    def cam_trace(self, sol):
        return [np.sum(sol[i,26:34]) for i in range(sol.shape[0])]
    
    def ryr_flux(self, sol):
        return [self.alpha_ryr * self.ryr_tot * (o1+o2+o3) * (ca_er - ca)/(Nav * self.Vspine) for o1,o2,o3,ca_er,ca in zip(sol[:,11],sol[:,12],sol[:,13],sol[:,-2],sol[:,-1])]
    
    def ip3_flux(self, sol):
        open_prob = [((ip3/(ip3 + self.d1))*(ca/(ca + self.d5))*h)**3 for ip3,ca,h in zip(sol[:,3],sol[:,-1],sol[:,-3])]
        return [self.alpha_ip3r * self.ip3r_tot * op * (ca_er - ca)/(Nav * self.Vspine) for op,ca_er,ca in zip(open_prob,sol[:,-2],sol[:,-1])]
    
    def nmda_flux(self, sol):
        return [-(self.g_N_Ca/self.Vspine) * (sr/(1.0 + 0.28 * np.exp(-0.062 * u))) * 0.078 * u * (ca - self.ca_ext*np.exp(-0.078*u))/(1 - np.exp(-0.078*u)) for sr,u,ca in zip(sol[:,-11],sol[:,-5],sol[:,-1])]
    
    def socc_flux(self, sol):
        return [self.Vsoc*p for p in sol[:,-9]]

    

    
    


    
    


