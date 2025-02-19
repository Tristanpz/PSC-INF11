# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:45:10 2025

@author: Utilisateur
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import collections as col
import torch
import os
os.chdir(r"C:\Users\Utilisateur\Documents\Augustin\X\2024.09 2A\Cours\PSC\Pytorch")



model = torch.load("model_3g1.pth", weights_only = False)
model.eval()

##Test distances

N=10000
##Création de N points randoms de l'espace d'entrée
#liste_entrees=[torch.tensor([(np.random.rand()-0.5)*8,(np.random.rand()-0.5)*8],dtype=torch.float32) for _ in range(N)]

##Création de N points bien répartis de l'espace d'entrée

x=np.linspace(-4,4,int(np.sqrt(N)))
y=np.linspace(-4,4,int(np.sqrt(N)))
X, Y =np.meshgrid(x,y, indexing='ij')
points=np.column_stack((X.ravel(),Y.ravel()))
liste_entrees=torch.tensor(points, dtype=torch.float32)

##Calcul des distances pour chaque point
liste_distances=[model.distance(x) for x in liste_entrees]

##Activation de chacun des points
liste_act=[model.activations_bin(x) for x in liste_entrees]

##Dictionnaire: clés=état d'activation (en binaire), valeurs= nombre de points dans la facette
dic_facettes=col.Counter(liste_act)

def carac_reseau(): #Imprime l'architecture du réseau actuel
    print("Architecture :",model.archi)
    print("Nombre de couches :",len(model.archi)-2)
    
def enumeration_facettes(): #Renvoie le nombre de facettes, et le nombre de points par facettes
    x=range(len(dic_facettes.keys()))
    h=[dic_facettes[act]for act in dic_facettes.keys()]
    plt.bar(x,h)
    plt.title("Répartition des points sur les facettes pour N = "+str(N))
    plt.show()
    return len(dic_facettes.keys()),dic_facettes

def partition_facettes():##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des indices des entrées dans la facette)
    partition={}
    for act in dic_facettes.keys(): #je choisis une facette
        partition[act]=[]
    for i in range(len(liste_distances)):
       partition[liste_act[i]].append(i)
    return partition

def poids_facettes(): #Retourne un histogramme du nombre de facettes en fonction du nombre de points dedans
    dico={}
    maximum=0
    for act in dic_facettes :
        nbe=dic_facettes[act]
        if (nbe in dico) :
            dico[nbe] += 1
        else :
            dico[nbe]=1
            if nbe>maximum:
                maximum=nbe

    liste_poids=[0]*(maximum+1)
    for i in range (maximum+1):
        if i in dico :
            liste_poids[i]=dico[i]
    x=range(maximum+1)
    plt.bar(x,liste_poids)
    plt.show()

     
def mesures_facettes():##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des distances des entrées dans la facette)
    partition=partition_facettes()
    mesures={}
    for act in partition.keys():
        mesures[act]=np.array([liste_distances[j] for j in partition[act]])
    return mesures

def moyenne(): #Renvoie des données sur les distances
    mesures=mesures_facettes()
    for act in mesures.keys():
        print(act,"moyenne: ",np.mean(mesures[act]),"max: ",np.max(mesures[act]),"nombres de points: ",len(mesures[act]),"écart-type: ",np.std(mesures[act]))

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

def accuracy() :
    N = 1000
    NbVrai = 0 
    for _ in range(N) : 
        res,donnee = genDonneesGaussiennes()
        res = format_res(res)
        X_tensor = torch.tensor(donnee, dtype = torch.float32)
        prediction = predire(model(X_tensor).detach().numpy())
        if res[0] == prediction[0] and res[1] == prediction[1]:
            NbVrai += 1
    return NbVrai/N*100

def test():
    #Génère des points de l'entrée à tester et renvoie la prédiction du réseau
    N_test = 10
    X_test = np.zeros((N_test, 2))
    y_test = np.zeros((N_test, 3))
              
    for i in range(N_test):
        res, donnee = genDonneesGaussiennes()
        X_test[i] = donnee
        y_test[i] = format_res(res)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    predictions = model(X_test_tensor).detach().numpy()

    print("Prédictions du modèle:")
    print("point,prédiction,gaussienne attendue")
    for i in range(N_test) :
        print(predictions[i],predire(predictions[i]), y_test[i])


def couleur(res) :
    if np.array_equal(res, [1,0,0]) :
        return 'r'
    if np.array_equal(res, [0,1,0]):
        return 'g'
    if np.array_equal(res, [0,0,1]) :
        return 'b'
    print("bruh")

def affichage_prediction():   
    X = np.linspace(-4,4,20)
    Y = np.linspace(-4, 4,20)
    for x in X :
        for y in Y:
            plt.scatter(x, y, c = couleur(predire(model(torch.tensor([x,y],dtype = torch.float32)).detach().numpy())), linewidths=0.2)
    plt.title("Prédiction du réseau sur R2")
    plt.show()

list_colors=[]
dico_colors=mcolors.CSS4_COLORS
for key in dico_colors:
    list_colors.append(dico_colors[key])
    
def visualisation_facettes_simple():#crée des points (on peut mettre moins de N points)
    X = np.linspace(-4,4,50)
    Y = np.linspace(-4, 4,50)
    for x in X:
        for y in Y:    
            plt.scatter(x,y,c=list_colors[model.activations_bin(torch.tensor([x,y],dtype = torch.float32))%148],linewidth=0.2)
    plt.title("Visualisation des facettes sur l'espace d'entrée")
    plt.show()
    
def visualisation_facettes():#se base sur les entrées calculées au début (N points)
    list_act_colors=[list_colors[i%148] for i in liste_act]
    plt.scatter(liste_entrees[:,0],liste_entrees[:,1],c=list_act_colors,linewidth=0.2)
    plt.show("Visualisation des facettes sur l'espace d'entrée")