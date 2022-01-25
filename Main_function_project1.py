# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 02:05:57 2022

@author: kaushik
"""

import numpy as np
import two_link_dynamics
import csv
import os
import datetime
import matplotlib
import matplotlib.pyplot as plot
from matplotlib import rc

if __name__=='__main__':
    dt=0.005 #time interval
    tf=60.0 #final time
    t=np.linspace(0.0, tf, int(tf/dt), dtype=(np.float32))
    Alpha= 4.0*np.ones(2,dtype=np.float32)
    Beta= 0.25*np.ones(2,dtype=np.float32)
    gamma= 0.1*np.ones(5,dtype=np.float32)
    dyn= two_link_dynamics.Dynamics(Alpha=Alpha, Beta=Beta, gamma=gamma)
    phi_Hist  =  np.zeros((2,len(t)),dtype=np.float32)
    phi_Dot_Hist= np.zeros((2,len(t)),dtype=np.float32)
    phi_DotDot_Hist= np.zeros((2,len(t)),dtype=np.float32)
    phi_d_Hist= np.zeros((2,len(t)),dtype=np.float32)
    phi_Dot_d_Hist= np.zeros((2,len(t)),dtype=np.float32)
    phi_DotDot_d_Hist= np.zeros((2,len(t)),dtype=np.float32)
    e_Hist= np.zeros((2,len(t)),dtype=np.float32)
    e_Norm_Hist= np.zeros_like(t)
    r_Hist=np.zeros((2,len(t)),dtype=np.float32)
    r_Norm_Hist= np.zeros_like(t)
    thetaTilda_Hist=np.zeros((5,len(t)),dtype=np.float32)
    thetaNorm_Hist=np.zeros_like(t)
    tau_Hist=np.zeros((2,len(t)),dtype=np.float32)
    tau_ff_Hist=np.zeros((2,len(t)),dtype=np.float32)
    tau_fb_Hist=np.zeros((2,len(t)),dtype=np.float32)
    
    #start save file
    savePath="C:/Kaushik Primary/two link project 1/Nonlinear_Control2_Project1_Python"
    now=datetime.datetime.now()
    nownew  =  now.strftime("%Y-%m-%d-%H-%M-%S")
    path=savePath+"/sim-"+nownew
    os.mkdir(path)
    file=open(path+"/data.csv","w",newline='')
    
    #writting the header into the file
    with file: 
        write  =  csv.writer(file)
        write.writerow(["time","e1","e2","r1","r2","tau1","tau2"])
    file.close()
    
    #loop through
    for kk in range(0, len(t)):
        #get the states and input data
        phij, phiDj, phiDDj, thetaHj  =  dyn.getState(t[kk])
        phidj, phiDdj, phiDDdj  =  dyn.getDes_State(t[kk])
        ej,_,rj,thetaTildaj  =  dyn.getErr_And_FTR(t[kk])
        tauj,_,tauffj, taufbj  =  dyn.getTauThetaHD(t[kk])
        
        #save the data to the buffers
        phi_Hist[:,kk]=phij
        phi_Dot_Hist[:,kk]=phiDj
        phi_DotDot_Hist[:,kk]=phiDDj
        phi_d_Hist[:,kk]= phidj
        phi_Dot_d_Hist[:,kk]=phiDdj
        phi_DotDot_d_Hist[:,kk]=phiDDdj
        e_Hist[:,kk]=ej
        e_Norm_Hist[kk]=np.linalg.norm(ej)
        r_Hist[:,kk]=rj
        r_Norm_Hist[kk]=np.linalg.norm(rj)
        thetaTilda_Hist[:,kk]=thetaTildaj
        thetaNorm_Hist[kk]=np.linalg.norm(thetaTildaj)
        tau_Hist[:,kk]=tauj
        tau_fb_Hist[:,kk]=taufbj
        tau_ff_Hist[:,kk]=tauffj
        
        #save the internal data to file
        file  =  open(path+"/data.csv","a",newline='')
        #writing the data into the file
        with file: 
            write  =  csv.writer(file)
            write.writerow([t[kk],e_Hist[0,kk],e_Hist[1,kk],r_Hist[0,kk],r_Hist[1,kk],tau_Hist[0,kk],tau_Hist[1,kk]])
        file.close()
        
        #step the internal state of the dynamics
        dyn.step(dt,t[kk])
    
    #plot the data
    #plot teh angles
    phiplot,phiax  =  plot.subplots()
    phiax.plot(t,phi_d_Hist[0,:],color='orange',linewidth=2,linestyle='--')
    phiax.plot(t,phi_Hist[0,:],color='orange',linewidth=2,linestyle='-')
    phiax.plot(t,phi_d_Hist[1,:],color='blue',linewidth=2,linestyle='--')
    phiax.plot(t,phi_Hist[1,:],color='blue',linewidth=2,linestyle='-')
    phiax.set_xlabel("$t$ $(sec)$")
    phiax.set_ylabel("$\phi_i$ $(rad)$")
    phiax.set_title("Angle")
    phiax.legend(["$\phi_{1d}$","$\phi_1$","$\phi_{2d}$","$\phi_2$"],loc='upper right')
    phiax.grid()
    phiplot.savefig(path+"/angles.pdf")
    
    #plot the error
    eplot,eax  =  plot.subplots()
    eax.plot(t,e_Hist[0,:],color='orange',linewidth=2,linestyle='-')
    eax.plot(t,e_Hist[1,:],color='blue',linewidth=2,linestyle='-')
    eax.set_xlabel("$t$ $(sec)$")
    eax.set_ylabel("$e_i$ $(rad)$")
    eax.set_title("Error")
    eax.legend(["$e_1$","$e_2$"],loc='upper right')
    eax.grid()
    eplot.savefig(path+"/error.pdf")

    #plot the error norm
    eNplot,eNax  =  plot.subplots()
    eNax.plot(t,e_Norm_Hist,color='orange',linewidth=2,linestyle='-')
    eNax.set_xlabel("$t$ $(sec)$")
    eNax.set_ylabel("$\Vert e \Vert$ $(rad)$")
    eNax.set_title("Error Norm")
    eNax.grid()
    eNplot.savefig(path+"/errorNorm.pdf")

    #plot the anglular velocity
    phiDplot,phiDax  =  plot.subplots()
    phiDax.plot(t,phi_Dot_d_Hist[0,:],color='orange',linewidth=2,linestyle='--')
    phiDax.plot(t,phi_Dot_Hist[0,:],color='orange',linewidth=2,linestyle='-')
    phiDax.plot(t,phi_Dot_d_Hist[1,:],color='blue',linewidth=2,linestyle='--')
    phiDax.plot(t,phi_Dot_Hist[1,:],color='blue',linewidth=2,linestyle='-')
    phiDax.set_xlabel("$t$ $(sec)$")
    phiDax.set_ylabel("$anglular velocity$ $(rad/sec)$")
    phiDax.set_title("Anglular Velocity")
    phiDax.legend(["$\dot{\phi}{1d}$","$\dot{\phi}_1$","$\dot{\phi}{2d}$","$\dot{\phi}_2$"],loc='upper right')
    phiDax.grid()
    phiDplot.savefig(path+"/anglularVelocity.pdf")

    #plot the filtered error
    rplot,rax  =  plot.subplots()
    rax.plot(t,r_Hist[0,:],color='orange',linewidth=2,linestyle='-')
    rax.plot(t,r_Hist[1,:],color='blue',linewidth=2,linestyle='-')
    rax.set_xlabel("$t$ $(sec)$")
    rax.set_ylabel("$r_i$ $(rad/sec)$")
    rax.set_title("Filtered Error")
    rax.legend(["$r_1$","$r_2$"],loc='upper right')
    rax.grid()
    rplot.savefig(path+"/filteredError.pdf")

    #plot the filtered error norm
    rNplot,rNax  =  plot.subplots()
    rNax.plot(t,r_Norm_Hist,color='orange',linewidth=2,linestyle='-')
    rNax.set_xlabel("$t$ $(sec)$")
    rNax.set_ylabel("$\Vert r \Vert$ $(rad)$")
    rNax.set_title("Filtered Error Norm")
    rNax.grid()
    rNplot.savefig(path+"/filteredErrorNorm.pdf")

    #plot the anglular acceleration
    phiDDplot,phiDDax  =  plot.subplots()
    phiDDax.plot(t,phi_DotDot_d_Hist[0,:],color='orange',linewidth=2,linestyle='--')
    phiDDax.plot(t,phi_DotDot_Hist[0,:],color='orange',linewidth=2,linestyle='-')
    phiDDax.plot(t,phi_DotDot_d_Hist[1,:],color='blue',linewidth=2,linestyle='--')
    phiDDax.plot(t,phi_DotDot_Hist[1,:],color='blue',linewidth=2,linestyle='-')
    phiDDax.set_xlabel("$t$ $(sec)$")
    phiDDax.set_ylabel("$anglular acceleration$ $(rad/sec^2)$")
    phiDDax.set_title("Anglular Acceleration")
    phiDDax.legend(["$\ddot{\phi}{1d}$","$\ddot{\phi}_1$","$\ddot{\phi}{2d}$","$\ddot{\phi}_2$"],loc='upper right')
    phiDDax.grid()
    phiDDplot.savefig(path+"/anglularAcceleration.pdf")

    #plot the inputs
    tauplot,tauax  =  plot.subplots()
    tauax.plot(t,tau_Hist[0,:],color='orange',linewidth=2,linestyle='-')
    tauax.plot(t,tau_Hist[1,:],color='blue',linewidth=2,linestyle='-')
    tauax.plot(t,tau_ff_Hist[0,:],color='orange',linewidth=2,linestyle='--')
    tauax.plot(t,tau_ff_Hist[1,:],color='blue',linewidth=2,linestyle='--')
    tauax.plot(t,tau_fb_Hist[0,:],color='orange',linewidth=2,linestyle='-.')
    tauax.plot(t,tau_fb_Hist[1,:],color='blue',linewidth=2,linestyle='-.')
    tauax.set_xlabel("$t$ $(sec)$")
    tauax.set_ylabel("$input$ $(Nm)$")
    tauax.set_title("Control Input")
    tauax.legend(['$\\tau_1$',"$\\tau_2$","$\\tau_{ff1}$","$\\tau_{ff2}$","$\\tau_{fb1}$","$\\tau_{fb2}$"],loc='upper right')
    tauax.grid()
    tauplot.savefig(path+"/input.pdf")

    #plot the parameter estiamtes
    thetaplot,thetaax  =  plot.subplots()
    thetaax.plot(t,thetaTilda_Hist[0,:],color='red',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTilda_Hist[1,:],color='green',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTilda_Hist[2,:],color='blue',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTilda_Hist[3,:],color='orange',linewidth=2,linestyle='-')
    thetaax.plot(t,thetaTilda_Hist[4,:],color='magenta',linewidth=2,linestyle='-')
    thetaax.set_xlabel("$t$ $(sec)$")
    thetaax.set_ylabel("$\\tilde{\\theta}_i$")
    thetaax.set_title("Parameter Error")
    thetaax.legend(["$\\tilde{\\theta}_1$","$\\tilde{\\theta}_2$","$\\tilde{\\theta}_3$","$\\tilde{\\theta}_4$","$\\tilde{\\theta}_5$"],loc='upper right')
    thetaax.grid()
    thetaplot.savefig(path+"/thetaTilde.pdf")

    #plot the parameter estiamtes norm
    thetaNplot,thetaNax  =  plot.subplots()
    thetaNax.plot(t,thetaNorm_Hist,color='orange',linewidth=2,linestyle='-')
    thetaNax.set_xlabel("$t$ $(sec)$")
    thetaNax.set_ylabel("$\Vert \\tilde{\\theta} \Vert$")
    thetaNax.set_title("Parameter Error Norm")
    thetaNax.grid()
    thetaNplot.savefig(path+"/thetaTildeNorm.pdf")