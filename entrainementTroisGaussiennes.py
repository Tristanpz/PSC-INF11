# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:39:52 2025

@author: Utilisateur
"""

import torch
import numpy as np

import reseauPytorch

def genDonneesGaussiennes() :
  m, s = (np.array([-2,-2]), np.array([2,2]),np.array([0,0])),np.eye(2)
  al = np.random.randint(0,3)
  return al, np.random.multivariate_normal(m[al], s)

N = 50

X = np.zeros((N,2))
y = np.zeros((N,3))
    
def format_res(i):
    if i==0:
        return [1,0,0]
    if i==1:
        return [0,1,0]
    else:
        return [0,0,1]
    
for i in range(N) :
    res, donnee = genDonneesGaussiennes()
    X[i] = donnee
    y[i] = format_res(res)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

num_epochs = 100
nbBatch = 100

model = reseauPytorch.creerReseau([2,4,4,3])
reseauPytorch.entrainerReseau(model, X_tensor, y_tensor, num_epochs, nbBatch)


torch.save(model, "model_3g.pth")
