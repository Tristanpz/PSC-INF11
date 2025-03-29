# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:35:07 2025

@author: trist
"""

import torch
import numpy as np

import reseauPytorch

def genDonneesGaussiennes() :
  m = (np.array([(0.5*t%6)-3 for t in range(32)]), 
       np.array([((-0.2*t+1)%6)-3 for t in range(32)]),
       np.array([((0.3*t-2)%6)-3 for t in range(32)]),
       np.array([((1.8*t-2)%6)-3 for t in range(32)]))
  s = (np.diag([((0.2*t)%2)+0.5 for t in range(32)]), 
       np.eye(32),
       np.diag([((-0.3*t+1)%2)+0.5 for t in range(32)]),
       0.7*np.eye(32))
  al = np.random.randint(0,4)
  return al, np.random.multivariate_normal(m[al], s[al])

N = 50

X = np.zeros((N,32))
y = np.zeros((N,4))
    
def format_res(i):
    if i==0:
        return [1,0,0,0]
    if i==1:
        return [0,1,0,0]
    if i == 2 :
        return [0,0,1,0]
    else:
        return [0,0,0,1]
    
for i in range(N) :
    res, donnee = genDonneesGaussiennes()
    X[i] = donnee
    y[i] = format_res(res)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

num_epochs = 100
nbBatch = 20

model = reseauPytorch.creerReseau([32,10,6,4])
reseauPytorch.entrainerReseau(model, X_tensor, y_tensor, num_epochs, nbBatch)


torch.save(model, "model_4g321064e0.pth")