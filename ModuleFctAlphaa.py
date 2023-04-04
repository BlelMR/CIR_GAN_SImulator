import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import time
import statsmodels.distributions as smd
import os


def GenerateUnidParameter(paraMax, paraMin, Npara, Disc):
    ParameterDisc=np.zeros(Npara,'f')
    Dp=int(Npara/Disc)
    D=((paraMax-paraMin)/Disc)
    for k in range(Disc):
        ParameterDisc[Dp*k:Dp*(k+1)]= np.random.uniform(paraMin+k*D, paraMin+(k+1)*D, size=(Dp,))
    return ParameterDisc

def FinalData(Nkappa, Ntheta, ThetaDisc, Kr):
    Npara=Nkappa*Ntheta
    Para=np.zeros((Npara,2),'f')
    for i in range(Ntheta):
        j=i*Nkappa
        jf=Nkappa+i*Nkappa
        Para[j:jf,0]=ThetaDisc[i]
        Para[j:jf,1]=Kr 
    return Para

# Chi square function with M trajectories 
def ChiSquare(Nt, M, kappa, theta, sigma):
    
    d = 4 * kappa * theta / sigma ** 2
    if (d>1):
        chiT= np.zeros((M,Nt),'f')
        for t in range(Nt):
            chiT[:,t] = np.random.chisquare(d - 1, M)
    else:
        chiT= np.zeros((M,Nt),'f')
        print("degree less than 1")
    return(chiT)

# CIR function with M trajectories using exact simulation 
def CIR_generate_paths(x0, kappa, theta, sigma, T, N, M, ran, chiT):
    '''
    ==========
    x0: float initial value
    kappa: float mean-reversion factor
    theta: float long-run mean
    sigma: float volatility factor
    T: float final date/time horizon
    N: int number of time steps
    M: int number of paths
    Returns
    =======
    x: NumPy array
        simulated paths
    '''
    
    dt = T / N
    x = np.zeros((M, N), 'f')
    x[:,0] = x0
    
    # exact discretization
    d = 4 * kappa * theta / sigma ** 2
    c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
    if d > 1:
        for t in range(N-1):
                chi=chiT[:,t]
                l = x[:,t] * np.exp(-kappa * dt) / c
                #chi = np.random.chisquare(d - 1, M)
                x[:,t+1] = c * ((ran[:,t] + np.sqrt(l)) ** 2 + chi)
    else:
        for t in range(N-1):
                l = x[:,t] * np.exp(-kappa * dt) / c
                Np = np.random.poisson(l/2, M)
                chi = np.random.chisquare(d + 2 * Np, M)
                x[:,t+1] = c * chi
    return x[:,1:]


def CIR_generate_paths_testCase(x0, kappa, theta, sigma, T, N, M):
    
    '''
    ==========
    x0: float initial value
        
    kappa: float mean-reversion factor
        
    theta: float long-run mean
        
    sigma: float volatility factor
        
    T: float final date/time horizon
        
    N: int number of time steps
        
    M: int number of paths
        
    Returns
    =======
    x: NumPy array
        simulated paths
    '''
    dt = T / N
    x = np.zeros((M, N), 'f')
    x[:,0] = x0
    ran = np.random.standard_normal((M, N))

    # exact discretization
    d = 4 * kappa * theta / sigma ** 2
    c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
    chiT=np.zeros((M,N),'f')
    if d > 1:
        for t in range(N-1):
                l = x[:,t] * np.exp(-kappa * dt) / c
                chi = np.random.chisquare(d - 1, M)
                chiT[:,t]=chi
                x[:,t+1] = c * ((ran[:,t] + np.sqrt(l)) ** 2 + chi)
    else:
        for t in range(N-1):
                l = x[:,t] * np.exp(-kappa * dt) / c
                Np = np.random.poisson(l / 2, M)
                chi = np.random.chisquare(d + 2 * Np, M)
                x[:,t+1] = c * chi
    return (x[:,1:], ran, chiT)


def FcStNumML(Nt, M, Para,  Npara, dt, SEXC, dW, CHIT):
    StNum=np.zeros(((Nt-1)*M,3, Npara), dtype=np.float32) #Matrix of the input data
    for i in range(Npara):
        StNum[:(Nt-1)*M,2,i]=np.reshape(SEXC[:,:,i],((Nt-1)*M)) #Third column corresponding to the St 
        StNum[:(Nt-1)*M,0,i]=np.reshape(dW[:,:],((Nt-1)*M)) #First column corresponding to the wt  
        StNum[:(Nt-1)*M,1,i]=np.reshape(CHIT[:,:,i],((Nt-1)*M)) #Second column corresponding to the Chisquare  

    StNumMLL=np.zeros(((Nt-2)*M*Npara,7), dtype=np.float32)
    for j in range(Npara):
        l=j*M*(Nt-2)
        theta=Para[j,0]
        kappa=Para[j,1]
        for i in range(M):
            StNumMLL[l+i*(Nt-2):l+(i+1)*(Nt-2),0]=StNum[i*(Nt-1):(i+1)*(Nt-1)-1,0,j]
            StNumMLL[l+i*(Nt-2):l+(i+1)*(Nt-2),1]=StNum[i*(Nt-1):(i+1)*(Nt-1)-1,1,j]
            StNumMLL[l+i*(Nt-2):l+(i+1)*(Nt-2),2]=StNum[i*(Nt-1):(i+1)*(Nt-1)-1,2,j]
            StNumMLL[l+i*(Nt-2):l+(i+1)*(Nt-2),3]=dt
            StNumMLL[l+i*(Nt-2):l+(i+1)*(Nt-2),4]=theta
            StNumMLL[l+i*(Nt-2):l+(i+1)*(Nt-2),5]=kappa        
            StNumMLL[l+i*(Nt-2):l+(i+1)*(Nt-2),6]=(StNum[i*(Nt-1)+1:(i+1)*(Nt-1),2,j]-theta)/theta

    return(StNumMLL)

def CIRMatrix(Nt, M, Sig, Para, S00, T, N, Npara):
    SEXC=np.zeros((M, Nt-1, Npara))
    CHIT=np.zeros((M, Nt-1, Npara))
    S0=S00*np.ones(M,dtype=np.float32) 
    dW = np.random.standard_normal((M, Nt))
    for i in range(Npara):
        theta=Para[i,0] #theta parameter for the CIR model
        kappa=Para[i,1] #kappa parameter for the CIR model
        chiT = ChiSquare(Nt, M, kappa, theta, Sig)
        SexcT = CIR_generate_paths(S0, kappa, theta, Sig, T, N, M, dW, chiT)
        SEXC[:,:,i] = SexcT
        CHIT[:,:,i] = chiT[:,1:]
    return (dW[:,1:], SEXC, CHIT)

# Function that saves different models during the training phase
def save_network(network, epoch_label, minibatch):
    save_filename = 'net_{}_{}.pth'.format(epoch_label , minibatch)
    save_path = os.path.join('./SavedModels102', save_filename)
    torch.save(network.state_dict(), save_path)
    
def FctNumMltest(Nt, M, theta, kappa, dt, SexcT, dW, chiT): 
    StNumML=np.zeros(((Nt-1)*M,3), dtype=np.float32) #Matrix of the input data 
    #StNumML[:,0]=np.reshape(St,(ts.size*ML))
    StNumML[:(Nt-1)*M,2]=np.reshape(SexcT,((Nt-1)*M)) #Third column corresponding to the St 
    StNumML[:(Nt-1)*M,0]=np.reshape(dW[:,1:],((Nt-1)*M)) #First column corresponding to the wt  
    StNumML[:(Nt-1)*M,1]=np.reshape(chiT[:,1:],((Nt-1)*M)) #Second column corresponding to the Chisquare  

    StNumMLLt=np.zeros(((Nt-2)*M,7), dtype=np.float32)
    for i in range(M):
        StNumMLLt[i*(Nt-2):(i+1)*(Nt-2),0]=StNumML[i*(Nt-1):(i+1)*(Nt-1)-1,0]
        StNumMLLt[i*(Nt-2):(i+1)*(Nt-2),1]=StNumML[i*(Nt-1):(i+1)*(Nt-1)-1,1]
        StNumMLLt[i*(Nt-2):(i+1)*(Nt-2),2]=StNumML[i*(Nt-1):(i+1)*(Nt-1)-1,2]
        StNumMLLt[i*(Nt-2):(i+1)*(Nt-2),3]=dt
        StNumMLLt[i*(Nt-2):(i+1)*(Nt-2),4]=theta
        StNumMLLt[i*(Nt-2):(i+1)*(Nt-2),5]=kappa
        StNumMLLt[i*(Nt-2):(i+1)*(Nt-2),6]=StNumML[i*(Nt-1)+1:(i+1)*(Nt-1),2]
    return StNumMLLt

def StNumfctOrderedShuffled(Nt1, Nt2, Nt3, Nt4, Npara, M, StNum1, StNum2, StNum3, StNum4):
    StNumMLL=np.zeros(((Nt4+Nt3+Nt2+Nt1-8)*M*Npara,7), dtype=np.float32)

    StNumMLL[:(Nt1-2)*M*Npara,:]=StNum1
    StNumMLL[(Nt1-2)*M*Npara:(Nt1+Nt2-4)*M*Npara,:]=StNum2
    StNumMLL[(Nt1+Nt2-4)*M*Npara:(Nt1+Nt2+Nt3-6)*M*Npara,:]=StNum3
    StNumMLL[(Nt1+Nt2+Nt3-6)*M*Npara:(Nt1+Nt2+Nt3+Nt4-8)*M*Npara,:]=StNum4
    #Data Set Shuffled 
    rng = np.random.default_rng(seed=None)
    rng.shuffle(StNumMLL)
    return StNumMLL
