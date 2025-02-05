
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

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

# #generation de donnees gaussiennes
# n = 5
# X = np.zeros((n))
# y = np.zeros((n))

# for i in range(n) :
#     res, donnee = genDonneesGaussiennes()
#     X[i] = donnee
#     y[i] = res
# # test de l'entraînement du reseau 
# X_tensor = torch.tensor(np.transpose(np.atleast_2d(X)), dtype=torch.float32)
# y_tensor = torch.tensor(np.transpose(np.atleast_2d(y)), dtype=torch.float32)
# predicted = model(X_tensor).detach().numpy()
# for i in range(n) :
#     print(X_tensor[i], y_tensor[i], predicted[i])

#predictions sur l'intervalle [-4,4]
X = np.linspace(-4, 4, 1000)
X_tensor = torch.tensor(np.transpose(np.atleast_2d(X)), dtype=torch.float32)
predicted = model(X_tensor).detach().numpy()
plt.plot(X, predicted)
plt.show()

# #test de decrireFacette et activations
# x_test = torch.tensor([[0.192]], dtype=torch.float32)
# layer_index = 2
# print("Activation de la couche ", layer_index, model.decrireFacette(x_test, layer_index))
# print("Activations du réseau ", model.activations(x_test))


# #test de reseauLin
# x_test = torch.tensor([[0.192]], dtype=torch.float32)
# print("etat d'activation :", model.activations(x_test))
# W, B = model.reseauLin(model.activations(x_test))
# x = x_test.detach().numpy()
# print("resultat aplati :", np.matmul(W, x) +B)
# print("resultat reseau :", model(x_test).detach())
