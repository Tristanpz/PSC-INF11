# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:45:10 2025

@author: Utilisateur
"""

import numpy as np
import matplotlib.pyplot as plt
import collections as col
import torch




model = torch.load("model_3g.pth", weights_only = False)
model.eval()

##Test distances

N= 1000
liste_entrees=[torch.tensor([(np.random.rand()-0.5)*8,(np.random.rand()-0.5)*8],dtype=torch.float32) for _ in range(N)]

liste_distances=[model.distance(x) for x in liste_entrees]

def conversion_act_to_binaire(activation):
    res=0
    mult=0
    for couche in activation:
        for neurone in couche:
            if neurone==1:
                res+=2**mult
            mult+=1
    return res

liste_act=[conversion_act_to_binaire(model.activations(x)) for x in liste_entrees]
dic_facettes=col.Counter(liste_act)

def enumeration_facettes():
    return len(dic_facettes.keys()),dic_facettes

def d():
    for act in dic_facettes: #je choisis une facette
        for i in range(len(liste_distances)):
            if liste_act[i]==act: #pour chaque point dans la facette
                pass
# print(liste_act)               
# print(enumeration_facettes())


#Test du reseau 

def genDonneesGaussiennes() :
  m, s = (np.array([-2,-2]), np.array([2,2]),np.array([0,0])),np.eye(2)
  al = np.random.randint(0,3)
  return al, np.random.multivariate_normal(m[al], s)  

def format_res(i):
    if i==0:
        return [1,0,0]
    if i==1:
        return [0,1,0]
    else:
        return [0,0,1]
    
def predire(res) :
    listeRes = np.array([[1,0,0],[0,1,0],[0,0,1]])
    listeNorme = [np.linalg.norm(listeRes[i,:]-res, ord = 1) for i in range(3)]
    i = np.argmin(listeNorme)
    return listeRes[i]
   
# N_test = 10
# X_test = np.zeros((N_test, 2))
# y_test = np.zeros((N_test, 3))
              
# for i in range(N_test):
#     res, donnee = genDonneesGaussiennes()
#     X_test[i] = donnee
#     y_test[i] = format_res(res)

# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# predictions = model(X_test_tensor).detach().numpy()

# print("Prédictions du modèle:")
# for i in range(N_test) :
#     predictions[i],predire(predictions[i]), y_test[i]


def couleur(res) :
    if np.array_equal(res, [1,0,0]) :
        return 'r'
    if np.array_equal(res, [0,1,0]):
        return 'g'
    if np.array_equal(res, [0,0,1]) :
        return 'b'
    print("bruh")
    
X = np.linspace(-4,4,50)
Y = np.linspace(-4, 4,50)
for x in X :
    for y in Y:
        plt.scatter(x, y, c = couleur(predire(model(torch.tensor([x,y],dtype = torch.float32)).detach().numpy())), linewidths=0.2)
        
plt.show()
                