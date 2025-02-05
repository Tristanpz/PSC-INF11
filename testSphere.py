import torch
import numpy as np
import reseauPytorch  


model = torch.load("model_sphere.pth")
model.eval() 

def genDonneesSpheriques():
    r = 8
    point = np.random.uniform(0, 10, 3) 
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

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

model = reseauPytorch.creerReseau([3, 3, 2])

num_epochs = 1000
nbBatch = 1
reseauPytorch.entrainerReseau(model, X_tensor, y_tensor, num_epochs, nbBatch)



N_test = 10  
X_test = np.zeros((N_test, 3))
y_test = np.zeros((N_test, 2))

for i in range(N_test):
    res, donnee = genDonneesSpheriques()
    X_test[i] = donnee
    y_test[i] = res

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

predictions = model(X_test_tensor).detach().numpy()

print("Prédictions du modèle:")
print(predictions)

print("Valeurs réelles:")
print(y_test)



