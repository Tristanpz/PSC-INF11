import torch
import numpy as np
import reseauPytorch  


model = torch.load("model_sphere.pth")
model.eval() 

def genDonneesSpheriques():
    r = 10
    point = np.random.uniform(-10, 10, 3) 
    distance_carre = np.sum(point**2)

    if distance_carre < r**2:
        return [1, 0], point  
    else:
        return [0, 1], point


N_test = 10  
X_test = np.zeros((N_test, 3))
y_test = np.zeros((N_test, 2))

for i in range(N_test):
    res, donnee = genDonneesSpheriques()
    X_test[i] = donnee
    y_test[i] = res

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

predictions = model(X_test_tensor).detach().numpy()

def predire(predictions):
    resultat = []
    for pred in predictions:
        if pred[0] > pred[1]:  
            resultat.append("Dans la sphère")
        else:
            resultat.append("En dehors de la sphère")
    return resultat

p = predire(predictions)

print("Prédictions du modèle:\n", predictions)
print("\nValeurs réelles:\n", y_test)
print("\nRésultat:\n", p)



