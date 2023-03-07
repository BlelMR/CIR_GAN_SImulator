#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:27:29 2023

@author: mohamedraedblel
"""


import scipy as sp
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

import statsmodels.distributions as smd
import os



# Chi square function with M trajectories 
def ChiSquare(Nt, M, kappa, theta, sigma):
    
    d = 4 * kappa * theta / sigma ** 2
    if (d>1):
        chiT= np.zeros((M,Nt),'f')
        for t in range(Nt):
            chiT[:,t] = np.random.chisquare(d - 1, M)
    else:
        print("degree less than 1")
    return(chiT)



ML = 50000 #nombre d'echantillons aléatoire
t_init = 0
T  = 10
N1      = 100  # Compute N1 grid points
N2      = 100  # Compute N2 grid points
N3      = 100  # Compute N3 grid points
N4      = 100  # Compute N4 grid points
N5      = 100  # Compute N5 grid points

dt1     = float(T - t_init) / N1 # Compute dt1
dt2     = float(T - t_init) / N2 # Compute dt2
dt3     = float(T - t_init) / N3 # Compute dt3
dt4     = float(T - t_init) / N4 # Compute dt4
dt5     = float(T - t_init) / N5 # Compute dt5

ts1    = np.arange(t_init, T, dt1) # Compute grid on [0,T] with dt1
ts2    = np.arange(t_init, T, dt2) # Compute grid on [0,T] with dt2
ts3    = np.arange(t_init, T, dt3) # Compute grid on [0,T] with dt3
ts4    = np.arange(t_init, T, dt4) # Compute grid on [0,T] with dt4
ts5    = np.arange(t_init, T, dt5) # Compute grid on [0,T] with dt5





Para=np.zeros((1,3),'f')
Para[0,0]=0.1  #Sigma parameter for the CIR model
Para[0,1]=0.2  #Kappa parameter for the CIR model
Para[0,2]=0.01  #Intial point for the CIR model
Npara=10
ThetaDisc=np.zeros((1,Npara),'f')
thetaMax=1
thetaMin=0.1
Disc=10
Dp=int(Npara/Disc)
D=((thetaMax-thetaMin)/Disc)
for k in range(Disc):
    ThetaDisc[0,Dp*k:Dp*(k+1)]= np.random.uniform(thetaMin+k*D, thetaMin+(k+1)*D, size=(Dp,))



Lambda=1000
Nw=10
Np = np.random.poisson(Lambda/ 2, ML)
dW11=np.zeros((ML),'f')
for i in range(ML):
    dW11[i] = np.sum((np.random.standard_normal(Np[i]))**2)


dWM=np.random.standard_normal((ML,Nw))
StNumML1=np.zeros((ML,Nw+1),'f')
StNumML1[:,:Nw]=dWM#First column corresponding to the wt  

StNumML1[:,Nw]=dW11#First column corresponding to the wt  

#Data Set Shuffled 
rng = np.random.default_rng(seed=None)
        
rng.shuffle(StNumML1)

StNumPd = pd.DataFrame(data=StNumML1[:,:])  #Data Set prepared for the splitting in train and test sets  
dataset= StNumPd
x= dataset.iloc[:,:-1].values #Input Data 
y= dataset.iloc[:, -1].values #Target Data 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(x, y, test_size=0.2, random_state= 0)
Ntrain=Y_train.size  #Size of the training set

'''***************Training set******************'''
#Generator model using 4 layers with LeakyRelu activation function 
class Generator(torch.nn.Module):
        def __init__(self, input_neurons, hidden_neurons, output_neurons ):
            super(Generator, self).__init__()
            self.hidden= nn.Linear(input_neurons, hidden_neurons)
            self.hiddenM1= nn.Linear(hidden_neurons, hidden_neurons)
            self.hiddenM2= nn.Linear(hidden_neurons, hidden_neurons)
            self.hiddenM3= nn.Linear(hidden_neurons, hidden_neurons)
            self.Activ =torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)
            #self.Activ =torch.nn.ReLU()

            #self.Activ =torch.sin
            self.eps = 1e-20
            
            self.bach1 = nn.BatchNorm1d(input_neurons)
            self.bach2 = nn.BatchNorm1d(hidden_neurons)
            self.bach3 = nn.BatchNorm1d(hidden_neurons)
            self.bach4 = nn.BatchNorm1d(hidden_neurons)
            self.bach5 = nn.BatchNorm1d(hidden_neurons)

            self.out= nn.Linear(hidden_neurons, output_neurons)
        def forward(self, x):
            #x = self.bach1(x)
            x = self.hidden(x)
            x = self.Activ(x)
            #x = self.bach2(x)           
            x = self.hiddenM1(x)
            x = self.Activ(x)
            #x = self.bach3(x)           
            x = self.hiddenM2(x)
            x = self.Activ(x)
            #x = self.bach4(x)           
            x = self.hiddenM3(x)            
            x = self.Activ(x)
            #x = self.bach5(x)           
            x = self.out(x)+self.eps
            #x = self.out(x)

            return x
        
#Discriminator model using 4 layers with LeakyRelu activation function 
class Discriminator(torch.nn.Module):
        def __init__(self, input_neurons, hidden_neurons, output_neurons ):
            super(Discriminator, self).__init__()
            self.hidden= nn.Linear(input_neurons, hidden_neurons)
            self.hiddenM1= nn.Linear(hidden_neurons, hidden_neurons)
            self.hiddenM2= nn.Linear(hidden_neurons, hidden_neurons)
            self.hiddenM3= nn.Linear(hidden_neurons, hidden_neurons)
            self.Activ =torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)
            #self.Activ =torch.nn.ReLU()

            #self.Activ =torch.sin
            self.bach1 = nn.BatchNorm1d(input_neurons)
            self.bach2 = nn.BatchNorm1d(hidden_neurons)
            self.bach3 = nn.BatchNorm1d(hidden_neurons)
            self.bach4 = nn.BatchNorm1d(hidden_neurons)
            self.bach5 = nn.BatchNorm1d(hidden_neurons)
            self.out= nn.Linear(hidden_neurons, output_neurons)

        def forward(self, x):
            #x = self.bach1(x)
            x = self.hidden(x)
            x = self.Activ(x)
            #x = self.bach2(x)           
            x = self.hiddenM1(x)
            x = self.Activ(x)
            #x = self.bach3(x)           
            x = self.hiddenM2(x)
            x = self.Activ(x)
            #x = self.bach4(x)           
            x = self.hiddenM3(x)            
            x = self.Activ(x) 
            #x = self.bach5(x)           
            x = self.out(x)
            x = torch.sigmoid(x)
            return x

'''Networks'''

#Generator Network
NetworkG = Generator(input_neurons = Nw, hidden_neurons = 400, output_neurons = 1)

#Discriminator Network
NetworkD = Discriminator(input_neurons = Nw+1, hidden_neurons = 400, output_neurons = 1)

'''Optimizers'''

#Generator optimizer Network
optimizerG = torch.optim.Adam(NetworkG.parameters(), lr=0.0001, betas=(0.5, 0.999))
#Discriminator optimizer Network
optimizerD = torch.optim.Adam(NetworkD.parameters(), lr=0.0001, betas=(0.5, 0.999))

#optimizerD = torch.optim.RMSprop(NetworkD.parameters(), lr = 0.00005, betas=(0.5, 0.999))

#optimizerG = torch.optim.RMSprop(NetworkG.parameters(), lr = 0.00005)

'''Loss function'''
#BCE Loss
loss_function = torch.nn.BCELoss()

'''************************Training phase*************************'''    

batche_size=1000   #Batch size 
XYtrain = torch.as_tensor(np.c_[(X_train, Y_train.reshape(-1,1))])
Y0=torch.zeros(batche_size)
Y1=torch.ones(batche_size)
X_train_mini=torch.empty(batche_size, Nw)
Y_train_mini=torch.empty(batche_size, 1)

Err_Proba=torch.zeros(0)   #Estimated Probability of having a fake sample 
Err_Descri=torch.zeros(0)  #Discrimnator array loss 
Err_Genera=torch.zeros(0) #Generator array Loss
EpochTab=torch.zeros(0)  #Array that contains epoch number 
Mini_batchesTab=torch.zeros(0) #Array that contains minibatch number 

NetworkG.train()
NetworkD.train()

for epoch in range(100):
    for mini_batches in range(int(Ntrain/batche_size)):
        #MiniBatch data
        X_train_mini.copy_(torch.as_tensor(XYtrain[mini_batches*batche_size:(mini_batches+1)*batche_size,:-1]))
        Y_train_mini.copy_(torch.as_tensor(XYtrain[mini_batches*batche_size:(mini_batches+1)*batche_size,-1:]))
        #Generator output 
        Generator_out = NetworkG(X_train_mini)
        #Discriminator fake input 
        Generator_Descriminator_fake_in=torch.cat((X_train_mini, Generator_out), dim=1) 
        #Discriminator fake ouput 
        Generator_Descriminator_out = NetworkD(Generator_Descriminator_fake_in)
        
        #Generator_loss = -loss_function(((Generator_Descriminator_out[:,0])), Y0)
        #Genrator term in the loss function to minimize  
        Generator_loss = -torch.log(Generator_Descriminator_out[:,0]).mean()
        #Genrator network optimizer to initialize
        optimizerG.zero_grad()
        #Genrator network backpropagation
        Generator_loss.backward()
        #Genrator network parameters update
        optimizerG.step()
        #Discriminator real input 
        Generator_Descriminator_real_in = torch.cat(( X_train_mini, Y_train_mini.view(-1,1)), dim=1)
        #Revaluation Generator output 
        Generator_out=NetworkG(X_train_mini)
        #Discrimnator fake input with new revaluation 
        Generator_Descriminator_fake_in=torch.cat((X_train_mini, Generator_out.detach()), dim=1)
        
        #if (( mini_batches % 500) == 0 ):
        #  for g in optimizerG.param_groups:
        #          g['lr'] = g['lr']/1.05

        for epoch2 in range(5):
            
            #Discriminator real output 
            Descriminator_real = NetworkD(Generator_Descriminator_real_in)
            #Second term in the BCE loss function 
            Descriminator_loss_1 = loss_function(Descriminator_real[:,0], Y1)
            #Discriminator fake output 
            Generator_Descriminator_out = NetworkD(Generator_Descriminator_fake_in)
            #Third term in the BCE loss function 
            Descriminator_loss_2 = loss_function(Generator_Descriminator_out[:,0], Y0)
            #Discriminator Loss estimated function to minimize
            Descriminator_loss = Descriminator_loss_1+Descriminator_loss_2
            #Discriminator network optimizer to initialize
            optimizerD.zero_grad()
            #Discriminator network backpropagation        
            Descriminator_loss.backward()
            #Discriminator network parameters update
            optimizerD.step()  
        
        print('Probability of success=',Generator_Descriminator_out.mean(),'%')
        print('Error Discriminator=',Descriminator_loss)
        print('Error Generator=',Generator_loss)
        print('Epoch=',epoch)
        GDmTorch=torch.zeros(1)
        DescriTorch=torch.zeros(1)
        GeneraTorch=torch.zeros(1)

        GDmTorch[0]=Generator_Descriminator_out.mean().detach()
        DescriTorch[0]=Descriminator_loss.detach()
        GeneraTorch[0]=Generator_loss.detach()
        Err_Proba=torch.cat((Err_Proba, GDmTorch), dim=0)   
        Err_Descri=torch.cat((Err_Descri, DescriTorch), dim=0)
        Err_Genera=torch.cat((Err_Genera, GeneraTorch), dim=0)
        
        #Save important models 
    '''
        if (epoch >15) & (0.4997<GDmTorch[0]<0.5001):
            last_model_wts = NetworkG.state_dict()
            save_network(NetworkG, epoch, mini_batches)
            Epochh=torch.zeros(1)
            Epochh[0]=(torch.as_tensor(epoch))
            EpochTab=torch.cat((EpochTab, Epochh), dim=0)
            Mini=torch.zeros(1)
            Mini[0]=(torch.as_tensor(mini_batches))   
            Mini_batchesTab=torch.cat((Mini_batchesTab, Mini), dim=0)
    '''






'''Test'''

Lambda=1000
Np = np.random.poisson(Lambda/ 2, ML)
dW1T=np.zeros((ML),'f')
for i in range(ML):
    dW1T[i] = np.sum((np.random.standard_normal(Np[i]))**2)


dWMT=np.random.standard_normal((ML,Nw))
StNumML1T=np.zeros((ML,Nw+1),'f')
StNumML1T[:,:Nw]=dWMT#First column corresponding to the wt  

StNumML1T[:,Nw]=dW1T#First column corresponding to the wt  






batchesize=100
loss_function2= torch.nn.MSELoss()

StNumPd = pd.DataFrame(data=StNumML1T[:,:]) 

dataset= StNumPd
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(x, y, test_size=0.5, random_state= 0)
# We run the model for t_i =30, we could do it f£or each t_i 
Ntest=Y_test.size
Generator_test=torch.zeros(Ntest, 1)
X_test_torch=torch.zeros(batchesize, Nw)
Y_test_torch=torch.zeros(batchesize)
for i in range(30000):
  X_test_torch.copy_(torch.as_tensor(X_test[i*batchesize:(i+1)*batchesize,:Nw]))
  Y_test_torch.copy_(torch.as_tensor(Y_test[i*batchesize:(i+1)*batchesize]))
  Generator_test[i*batchesize:(i+1)*batchesize] = NetworkG(X_test_torch)

  err_test=loss_function2(Generator_test[i*batchesize:(i+1)*batchesize], Y_test_torch.view(-1,1))
  print('ERR1=',(torch.sqrt(err_test.mean()))/(torch.sqrt(((Y_test_torch.view(-1,1))**2).mean())))
  
  #err_test2=(((Generator_test[i*batchesize:(i+1)*batchesize].detach()-Y_test_torch)/Y_test_torch)**2).mean()
  #print('ERR2=',torch.sqrt(err_test2))

sns.distplot((Generator_test[:100000]).detach().cpu().numpy(), bins=100, label = 'pred')
sns.distplot(Y_test[:100000], bins=100, label = 'true')
plt.legend()













