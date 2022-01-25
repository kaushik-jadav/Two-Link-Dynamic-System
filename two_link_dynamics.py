# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 02:03:22 2022

@author: kaushik
"""

import numpy as np
from math import sin
from math import cos

class Dynamics():
    # constructor to initialize a Dynamics object
    def __init__(self,Alpha=0.25*np.ones(2,dtype=np.float32),Beta=0.1*np.ones(2,dtype=np.float32),gamma=0.01*np.ones(5,dtype=np.float32)):
         # gains
        self.Alpha  =  np.diag(Alpha)
        self.Beta  =  np.diag(Beta)
        self.Gamma  =  np.diag(gamma)
        
        # rigid body parameters
        self.m  =  np.array([2.0,2.0],dtype=np.float32) # mass in kg
        self.l  =  np.array([0.5,0.5],dtype=np.float32) # length in m
        self.m_Bounds  =  np.array([1.0,3.0],dtype=np.float32) # mass bounds in kg
        self.l_Bounds  =  np.array([0.25,0.75],dtype=np.float32) # length bounds in m
        self.g  =  9.8 # gravity in m/s^2
        
        # unknown parameters
        self.theta  =  self.getTheta(self.m,self.l) # initialize theta
        self.theta_Hat  =  self.getTheta(self.m_Bounds[0]*np.ones(2,dtype=np.float32),self.l_Bounds[0]*np.ones(2,dtype=np.float32)) # initialize theta estimate to the lowerbounds

                # desired trajectory parameters
        self.phi_d_Mag  =  np.array([np.pi/8,np.pi/4],dtype=np.float32) # amplitude of oscillations in rad
        self.f_phi_d  =  0.2 # frequency in Hz
        self.a_phi_d  =  np.pi/2 # phase shift in rad
        self.b_phi_d  =  np.array([np.pi/2,np.pi/4],dtype=np.float32) # bias in rad
        
        # initialize state
        self.phi,_,_  =  self.getDes_State(0.0) # set the initial angle to the initial desired angle
        self.phi_Dot  =  np.zeros(2,dtype=np.float32) # initial angular velocity
        self.phi_DotDot  =  np.zeros(2,dtype=np.float32) # initial angular acceleration
        
        
    def getTheta(self,m,l):
        """
        Inputs:
        -------
        \t m: link masses \n
        \t l: link lengths \n

        Returns:
        -------
        \t theta: parameters
        """
        theta  =  np.array([(m[0]+m[1])*l[0]**2+m[1]*l[1]**2,
                          m[1]*l[0]*l[1],
                          m[1]*l[1]**2,
                          (m[0]+m[1])*l[0],
                          m[1]*l[1]],dtype=np.float32)
        return theta
    
    def getDes_State(self,t):
        """
        Determines the desired state of the system \n
        Inputs:
        -------
        \t t: time \n
        
        Returns:
        -------
        \t phi_d:   desired angles \n
        \t phi_Dot_d:  desired angular velocity \n
        \t phi_DotDot_d: desired angular acceleration
        """
        # desired angles
        phi_d  =  np.array([self.phi_d_Mag[0]*sin(2*np.pi*self.f_phi_d*t-self.a_phi_d)-self.b_phi_d[0],
                         self.phi_d_Mag[1]*sin(2*np.pi*self.f_phi_d*t-self.a_phi_d)+self.b_phi_d[1]],dtype=np.float32)

        #desired angular velocity
        phi_Dot_d  =  np.array([2*np.pi*self.f_phi_d*self.phi_d_Mag[0]*cos(2*np.pi*self.f_phi_d*t-self.a_phi_d),
                          2*np.pi*self.f_phi_d*self.phi_d_Mag[1]*cos(2*np.pi*self.f_phi_d*t-self.a_phi_d)],dtype=np.float32)

        #desired angular acceleration
        phi_DotDot_d  =  np.array([-((2*np.pi*self.f_phi_d)**2)*self.phi_d_Mag[0]*sin(2*np.pi*self.f_phi_d*t-self.a_phi_d),
                           -((2*np.pi*self.f_phi_d)**2)*self.phi_d_Mag[1]*sin(2*np.pi*self.f_phi_d*t-self.a_phi_d)],dtype=np.float32)
        
        return phi_d,phi_Dot_d,phi_DotDot_d
    
    # returns the inertia matrix
    def getM(self,m,l,phi):
        """
        Determines the inertia matrix \n
        Inputs:
        -------
        \t m:   link masses \n
        \t l:   link lengths \n
        \t phi: angles \n
        
        Returns:
        -------
        \t M: inertia matrix
        """
        m1  =  m[0]
        m2  =  m[1]
        l1  =  l[0]
        l2  =  l[1]
        c2  =  cos(phi[1])
        M  =  np.array([[m1*l1**2+m2*(l1**2+2*l1*l2*c2+l2**2),m2*(l1*l2*c2+l2**2)],
                      [m2*(l1*l2*c2+l2**2),m2*l2**2]],dtype=np.float32)
        return M

    # returns the centripetal coriolis matrix
    def getC(self,m,l,phi,phi_Dot):
        """
        Determines the centripetal coriolis matrix \n
        Inputs:
        -------
        \t m:    link masses \n
        \t l:    link lengths \n
        \t phi:  angles \n
        \t phi_Dot: angular velocities \n
        
        Returns:
        -------
        \t C: cetripetal coriolis matrix
        """
        m1  =  m[0]
        m2  =  m[1]
        l1  =  l[0]
        l2  =  l[1]
        s2  =  sin(phi[1])
        phi1D  =  phi_Dot[0]
        phi2D  =  phi_Dot[1]
        C  =  np.array([-2*m2*l1*l2*s2*phi1D*phi2D-m2*l1*l2*s2*phi2D**2,
                      m2*l1*l2*s2*phi1D**2],dtype=np.float32)
        return C

    # returns the gravity matrix
    def getG(self,m,l,phi):
        """
        Determines the gravity matrix \n
        Inputs:
        -------
        \t m:   link masses \n
        \t l:   link lengths \n
        \t phi: angles \n
        
        Returns:
        -------
        \t G: gravity matrix
        """
        m1  =  m[0]
        m2  =  m[1]
        l1  =  l[0]
        l2  =  l[1]
        c1  =  cos(phi[0])
        c12  =  cos(phi[0]+phi[1])
        G  =  np.array([(m1+m2)*self.g*l1*c1+m2*self.g*l2*c12,
                      m2*self.g*l2*c12],dtype=np.float32)
        return G

    # returns the inertia matrix regressor
    def getYM(self,vphi,phi):
        """
        Determines the inertia matrix regressor \n
        Inputs:
        -------
        \t vphi: phi_DotDot_d+Alpha*e_Dot or phi_DotDot \n
        \t phi:  angles \n
        
        Returns:
        -------
        \t YM: inertia matrix regressor
        """
        vphi1  =  vphi[0]
        vphi2  =  vphi[1]
        c2  =  cos(phi[1])
        YM  =  np.array([[vphi1,2*c2*vphi1+c2*vphi2,vphi2,0.0,0.0],
                       [0.0,c2*vphi1,vphi1+vphi2,0.0,0.0]],dtype=np.float32)
        return YM

    # returns the centripetal coriolis matrix regressor
    def getYC(self,phi,phi_Dot):
        """
        Determines the centripetal coriolis matrix regressor \n
        Inputs:
        -------
        \t phi:  angles \n
        \t phi_Dot: angular velocity \n
        
        Returns:
        -------
        \t YC: centripetal coriolis matrix regressor
        """
        s2  =  sin(phi[1])
        phi1D  =  phi_Dot[0]
        phi2D  =  phi_Dot[1]
        YC  =  np.array([[0.0,-2*s2*phi1D*phi2D-s2*phi2D**2,0.0,0.0,0.0],
                       [0.0,s2*phi1D**2,0.0,0.0,0.0]],dtype=np.float32)
        return YC

    # returns the gravity matrix regressor
    def getYG(self,phi):
        """
        Determines the gravity matrix regressor \n
        Inputs:
        -------
        \t phi: angles \n
        
        Returns:
        -------
        \t YG: gravity matrix regressor
        """
        c1  =  cos(phi[0])
        c12  =  cos(phi[0]+phi[1])
        YG  =  np.array([[0.0,0.0,0.0,self.g*c1,self.g*c12],
                     [0.0,0.0,0.0,0.0,self.g*c12]],dtype=np.float32)
        return YG

    # returns the inertia matrix derivative regressor
    def getYMD(self,phi,phi_Dot,r):
        """
        Determines the inertia derivative regressor \n
        Inputs:
        -------
        \t phi:  angles \n
        \t phi_Dot: angular velocoty \n
        \t r:    filtered tracking error \n
        
        Returns:
        -------
        \t YMD: inertia matrix derivative regressor
        """

        s2  =  sin(phi[1])
        phi2D  =  phi_Dot[1]
        r1  =  r[0]
        r2  =  r[1]
        YMD  =  np.array([[0.0,-2*s2*phi2D*r1-s2*phi2D*r2,0.0,0.0,0.0],
                       [0.0,-s2*phi2D*r1,0.0,0.0,0.0]],dtype=np.float32)
        return YMD
        
    #returns the state
    def getState(self, t):
        
        return self.phi, self.phi_Dot, self.phi_DotDot, self.theta_Hat
    
    def getErr_And_FTR(self,t):
        """
        Returns the errors \n
        Inputs:
        -------
        \t t:  time \n

        Returns:
        -------
        \t e:          tracking error \n
        \t e_Dot:         tracking error derivative \n
        \t r:          filtered tracking error \n
        \t thetaTilde: parameter estimate error
        """
    # get the desired state
        phi_d,phi_Dot_d,_  =  self.getDes_State(t)

        # get the tracking error
        e  =  phi_d - self.phi
        e_Dot  =  phi_Dot_d - self.phi_Dot
        r  =  e_Dot + self.Alpha@e

        # calculate the parameter error
        thetaTilde  =  self.theta-self.theta_Hat
        return e,e_Dot,r,thetaTilde


    def getTauThetaHD(self,t):
        """
        Calculates the input and adaptive update law \n
        Inputs:
        -------
        \t t:  time \n

        Returns:
        -------
        \t tau:     control input \n
        \t theta_Hat_Dot: parameter estimate adaptive update law \n
        \t tau_ff:   input from the feedforward portion of control \n
        \t tau_fb:   input from the feedback portion of control \n
        \t thetaCL: approximate of theta from CL \n
        """
        # get the desired state
        _,_,phi_DotDot_d  =  self.getDes_State(t)

        # get the error
        e,e_Dot,r,_  =  self.getErr_And_FTR(t)

        # get the regressors
        vphi  =  phi_DotDot_d + self.Alpha@e_Dot
        YM  =  self.getYM(vphi,self.phi)
        YC  =  self.getYC(self.phi,self.phi_Dot)
        YG  =  self.getYG(self.phi)
        YMD  =  self.getYMD(self.phi,self.phi_Dot,r)
        Y  =  YM+YC+YG+0.5*YMD

        #calculate the controller and update law
        tau_ff  =  Y@self.theta_Hat
        tau_fb  =  e+self.Beta@r
        tau  =  tau_ff + tau_fb

        #update law
        theta_Hat_Dot  =  self.Gamma@Y.T@r 
        return tau,theta_Hat_Dot,tau_ff,tau_fb

# take a step of the dynamics
    def step(self,dt,t):
        """
        Steps the internal state using the dynamics \n
        Inputs:
        -------
        \t dt: time step \n
        \t t:  time \n
        
        Returns:
        -------
        """
        # get the dynamics
        M  =  self.getM(self.m,self.l,self.phi)
        C  =  self.getC(self.m,self.l,self.phi,self.phi_Dot)
        G  =  self.getG(self.m,self.l,self.phi)

        # get the input and update law
        tau,theta_Hat_Dot,_,_  =  self.getTauThetaHD(t)

        # calculate the dynamics using the input
        self.phi_DotDot  =  np.linalg.inv(M)@(-C-G+tau)

        # update the internal state
        # X(ii+1)  =  X(ii) + dt*f(X)
        self.phi += dt*self.phi_Dot
        self.phi_Dot += dt*self.phi_DotDot
        self.theta_Hat += dt*theta_Hat_Dot
