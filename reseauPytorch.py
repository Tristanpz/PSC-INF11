import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
            x = nn.ReLU()(x)
            
        return activation 
    
    def act_to_binaire(self, activation):
        res=0
        mult=0
        for couche in activation:
            for neurone in couche:
                if neurone==1:
                    res+=2**mult
                mult+=1
        return res
    
    def activations_bin(self,x):
        return self.act_to_binaire(self.activations(x))

    def reseauLin(self, etatActivation) :
        ''' prend en entree une liste de vecteurs qui decrit l'etat d'activation du reseau
        retourne une matrice W et un vecteur B tels que : sortie = W*entree + B 
        '''
        #Pour ne pas avoir de probleme de depassement d'indice 
        etatActivation.append(np.array([1]*self.tailleOut))
        W = np.eye(self.tailleIn)
        B = np.array([0]*self.tailleIn)
        for i in range(0, len(self.linearReluStack), 2) :
            layer = self.linearReluStack[i]
            activLayer = np.diagflat(etatActivation[i//2])
            poidsCorr = np.matmul(activLayer,layer.weight.data.numpy())
            W = np.matmul(poidsCorr, W)
            B = np.matmul(poidsCorr, B) + np.matmul(activLayer,layer.bias.data.numpy())
        return W, B
    
    def distance(self, x) :
        d = np.inf
        produitMaxi = 1
        
        for i in range(0, len(self.linearReluStack)-1, 2):
            if not i :
                x = nn.ReLU()(x)
            layer = self.linearReluStack[i]
            x = layer(x)
            x_mod = x.detach().numpy()
            W = layer.weight.data.numpy()
            minimum = np.inf
            maximum = 0
            for j in range(np.shape(W)[0]) :
                normeLigne = np.linalg.norm(W[j,:],ord = 1)
                if np.abs(x_mod[j])/normeLigne < minimum : 
                    minimum = np.abs(x_mod[j])/normeLigne
                if normeLigne > maximum :
                    maximum = normeLigne
            distTemp = minimum / produitMaxi
            if  distTemp < d : 
                d = distTemp
            produitMaxi *= maximum
        return d
    
    def distanceBis(self, x) :
        layer = self.linearReluStack[0]
        x = layer(x)
        x_mod = x.detach().numpy()
        W = layer.weight.data.numpy()
        U = W
        dmin = np.min([abs(x_mod[j])/np.linalg.norm(U[j,:]) for j in range(np.shape(U)[0])])
     
        for i in range(2, len(self.linearReluStack)-1, 2):
            x = nn.ReLU()(x)
            layer = self.linearReluStack[i]
            A = np.diag((x > 0).int().detach().numpy())
            x = layer(x)
            x_mod = x.detach().numpy()
            W = layer.weight.data.numpy()
            U = np.dot(W,np.dot(A,U))
            #print(np.linalg.norm(U[j,:]), x_mod[j])
            for j in range(np.shape(U)[0]) :
                if np.linalg.norm(U[j,:]) != 0 :
                    dmin = min(dmin, abs(x_mod[j])/np.linalg.norm(U[j,:]))
        return dmin
            
        
def creerReseau(archi) :   
    return LinearRegressionModel(archi)

def entrainerReseau(model, X_tensor, y_tensor, num_epochs= 100, nbBatch = 1) :
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    nbDonnees = len(X_tensor)
    donneesParBatch = nbDonnees // nbBatch
    
    listePerte = np.zeros(num_epochs*nbBatch)
    for epoch in range(num_epochs):
        for i in range(nbBatch) :
            if i == nbBatch-1 :  
                outputs = model(X_tensor[i*donneesParBatch:, :])
                loss = criterion(outputs, y_tensor[i*donneesParBatch:, :])
            else : 
                outputs = model(X_tensor[i*donneesParBatch : (i+1)*donneesParBatch, :])
                loss = criterion(outputs, y_tensor[i*donneesParBatch : (i+1)*donneesParBatch, :])
            listePerte[epoch] = loss.detach().numpy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    plt.plot(range(num_epochs*nbBatch), np.array(listePerte))
    plt.title("Evolution de la perte au cours de l'entrainement")
    plt.show()