import torch
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

import reseauPytorch

def genDonneesSpheriques() :
    #le rayon de la sphere
    r = 5

    #coordonnees d'un point
    point = np.random.uniform(0, 10,3)

    distance_carre = np.sum(point**2)

    if distance_carre < r**2:
        return [1, 0], point  
    else:
        return [0, 1], point

N = 100
X = np.zeros((N, 3))
y = np.zeros((N, 2))

for i in range(N):
    res, donnee = genDonneesSpheriques()
    X[i] = donnee
    y[i] = res

X_tensor = torch.tensor(np.transpose(X), dtype=torch.float32)
y_tensor = torch.tensor(np.transpose(y), dtype=torch.float32)

model = reseauPytorch.creerReseau([3, 3, 2])

num_epochs = 1000
nbBatch = 1
reseauPytorch.entrainerReseau(model, X_tensor, y_tensor, num_epochs, nbBatch)


    
