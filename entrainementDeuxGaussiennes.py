import torch
import numpy as np
import scipy.stats as sps

import reseauPytorch

def genDonneesGaussiennes() :
  m, s = (-2, 2), (1, 1)
  al = np.random.randint(0,2)
  return al, sps.norm.rvs(m[al], s[al])


N = 100

X = np.zeros((N))
y = np.zeros((N))

for i in range(N) :
    res, donnee = genDonneesGaussiennes()
    X[i] = donnee
    y[i] = res

X_tensor = torch.tensor(np.transpose(np.atleast_2d(X)), dtype=torch.float32)
y_tensor = torch.tensor(np.transpose(np.atleast_2d(y)), dtype=torch.float32)

num_epochs = 1000
nbBatch = 2

model = reseauPytorch.creerReseau([1,3,2,1])
reseauPytorch.entrainerReseau(model, X_tensor, y_tensor, num_epochs, nbBatch)

torch.save(model, "model_complete.pth") 

