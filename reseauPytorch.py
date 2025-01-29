import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self, archi):
        self.tailleIn = archi[0]
        self.tailleOut = archi[-1]
        self.archi = archi 
        super(LinearRegressionModel, self).__init__()
        self.linearReluStack = nn.Sequential()
        for i in range(len(self.archi)-1)  :
            self.linearReluStack.append(nn.Linear(archi[i],archi[i+1]))
            self.linearReluStack.append(nn.ReLU())
        self.linearReluStack.pop(-1)
                                           

    def forward(self, x):
      logits = self.linearReluStack(x)
      return logits


# #retourne l'activation d'une seule couche du réseau
#     def decrireFacette(self, x, layer_index):
#       current_input = x
#       for i, layer in enumerate(self.linearReluStack):
#         current_input = layer(current_input)
#         if i == layer_index:
#           activation_binaires = (current_input > 0).int().detach().numpy()
#           return activation_binaires
#       return None

# #retourne une liste de vecteurs des activations de toutes les couches
#     def activations0(self, x):
#       activation = []  #[x.detach().numpy()]
#       for layer_index in range(0, len(self.linearReluStack), 2) :
#         activation_couche = self.decrireFacette(x, layer_index)
#         activation.append(activation_couche)
#       return activation
  
    #fonction sans utiliser decrireFacette
    def activations(self, x):
        activation = []
        for i in range(0, len(self.linearReluStack)-1, 2):
            x = self.linearReluStack[i](x)
            activation.append((x > 0).int().detach().numpy())
        return activation 

    def reseauLin(self, etatActivation) :
        ''' prend en entree une liste de vecteurs qui decrit l'etat d'activation du reseau
        retourne une matrice W et un vecteur B tels que : sortie = W*entree + B 
        '''
        #Pour ne pas avoir de probleme de depassement d'indice 
        etatActivation.append(np.array([1]*self.tailleOut))
        W = np.eye(self.tailleIn)
        B = np.array([0]*self.tailleIn)
        print(np.shape(B))
        for i in range(0, len(self.linearReluStack), 2) :
            layer = self.linearReluStack[i]
            activLayer = np.diagflat(etatActivation[i//2])
            poidsCorr = np.matmul(activLayer,layer.weight.data.numpy())
            W = np.matmul(poidsCorr, W)
            B = np.matmul(poidsCorr, B) + np.matmul(activLayer,layer.bias.data.numpy())
        return W, B


def creerReseau(archi) :   
    return LinearRegressionModel(archi)

def entrainerReseau(model, X_tensor, y_tensor, num_epochs= 100) :
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()