# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:35:08 2025

@author: trist
"""

import numpy as np
import matplotlib.pyplot as plt
import collections as col
import torch
from scipy.stats import qmc
import operator
##Lignes pour que ça marche chez Augustin
# import os
# os.chdir(r"C:\Users\Utilisateur\Documents\Augustin\X\2024.09 2A\Cours\PSC\Pytorch")

file="model_4g321064e0.pth"
model = torch.load(file, weights_only = False)
model.eval()

#%%

N=10000
##Création de N points randoms de l'espace d'entrée
#liste_entrees=[torch.tensor([(np.random.rand()-0.5)*8,(np.random.rand()-0.5)*8],dtype=torch.float32) for _ in range(N)]

##Création de N points bien répartis de l'espace d'entrée

sampler = qmc.LatinHypercube(32, seed = 42)
points = sampler.random(N)
liste_entrees=torch.tensor(points, dtype=torch.float32)

##Calcul des distances pour chaque point
liste_distances=[model.distance(x) for x in liste_entrees]

##Activation de chacun des points
liste_act=[model.activations_bin(x) for x in liste_entrees]

##Dictionnaire: clés=état d'activation (en binaire), valeurs= nombre de points dans la facette
dic_facettes=col.Counter(liste_act)

##liste contenant des coupes (etat d'activation, nombre de points dans la facette) triés par nombre de pts décroissant
sorted_facettes=sorted(dic_facettes.items(), key=operator.itemgetter(1), reverse=True)

#%%
## Etude du réseau


def carac_reseau(): #Imprime l'architecture du réseau actuel
    print("Architecture :",model.archi)
    print("Nombre de couches internes:",len(model.archi)-2)
    
def strCaracReseau():
    return "Architecture :" + str(model.archi)
    
def enumeration_facettes(): #Renvoie le nombre de facettes, et le nombre de points par facettes
    x=range(len(dic_facettes.keys()))
    h=sorted([dic_facettes[act]for act in dic_facettes.keys()],reverse=True)
    plt.bar(x,h)
    plt.suptitle("Répartition des points sur les facettes pour N = "+str(N))
    plt.title(strCaracReseau())
    plt.show()
    return len(dic_facettes.keys()),dic_facettes

def facette_max(): ##renvoie la facette la plus grande et son nombre de points
    nbpoints=[dic_facettes[act] for act in dic_facettes]
    
    nbpoints=0
    for act in dic_facettes:
        if dic_facettes[act]>nbpoints:
            nbpoints=dic_facettes[act]
            actmax=act
    return actmax,nbpoints

def facettes_max(n):
    return sorted_facettes[:n]

def partition_facettes():##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des indices des entrées appartenant à la facette)
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
    plt.title("Histogramme représentant le nombre de facettes selon leur taille (nombre de points de l'espace d'entrée)")
    plt.xlabel("Nombre de points dans la facette")
    plt.ylabel("Nombre de facettes")
    plt.show()
     
def mesures_facettes():##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des distances des entrées dans la facette)
    partition=partition_facettes()
    mesures={}
    for act in partition.keys():
        mesures[act]=np.array([liste_distances[j] for j in partition[act]])
    return mesures

def moyenne_graphe(): #Renvoie des données sur les distances
    mesures=mesures_facettes()
    x=range(len(dic_facettes.keys()))
    h=[mesures[act][0]for act in mesures.keys()]
    plt.bar(x,sorted(h, reverse = True), color = 'b')
    plt.suptitle("Distance aux frontières moyenne pour chaque facette")
    plt.title(strCaracReseau())
    plt.xlabel("facette")
    plt.ylabel("distance moyenne")
    plt.show()

def maxdist(act):
    mesures=mesures_facettes()
    return np.max(mesures[act])

def meandist(act):
    mesures=mesures_facettes()
    return np.mean(mesures[act])

def stddist(act):
    mesures=mesures_facettes()
    return np.std(mesures[act])

def distribution_distance(act):
    mesures=mesures_facettes()
    distance_entrees_facette=mesures[act]
    mean = meandist(act)
    sigma = stddist(act)
    plt.hist(distance_entrees_facette,bins="auto")
    plt.axvline(mean,color="red", label="Distance moyenne")
    plt.axvline(mean+sigma, color ="green", label = "écart-type")
    plt.axvline(mean-sigma, color = "green")
    plt.suptitle("Distribution des distances à la frontière pour la facette "+str(act)+" nombre de points: "+str(dic_facettes[act]))
    
    plt.xlabel("Distance à la frontière")
    plt.ylabel("Nombre de points")
    plt.title(strCaracReseau())
    plt.legend()
    plt.show()

def distribution_facettes(n):
    f_max=facettes_max(n)
    for i in range(n):
        distribution_distance(f_max[i][0])

        
#%%
#Test du reseau 

def genDonneesGaussiennes() :
  m = (np.array([(0.5*t%6)-3 for t in range(32)]), 
       np.array([((-0.2*t+1)%6)-3 for t in range(32)]),
       np.array([((0.3*t-2)%6)-3 for t in range(32)]),
       np.array([((1.8*t-2)%6)-3 for t in range(32)]))
  s = (np.diag([((0.2*t)%2)+0.5 for t in range(32)]), 
       np.eye(32),
       np.diag([((-0.3*t+1)%2)+0.5 for t in range(32)]),
       0.7*np.eye(32))
  al = np.random.randint(0,4)
  return al, np.random.multivariate_normal(m[al], s[al])

def format_res(i):
    if i==0:
        return [1,0,0,0]
    if i==1:
        return [0,1,0,0]
    if i == 2 :
        return [0,0,1,0]
    else:
        return [0,0,0,1]
    
def predire(res) :
    listeRes = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    listeNorme = [np.linalg.norm(listeRes[i,:]-res, ord = 1) for i in range(4)]
    i = np.argmin(listeNorme)
    return listeRes[i]

def accuracy(N) :
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
    X_test = np.zeros((N_test, 32))
    y_test = np.zeros((N_test, 4))
              
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

