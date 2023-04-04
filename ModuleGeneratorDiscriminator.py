import torch 
import torch.nn as nn
import torch.nn.functional as F

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