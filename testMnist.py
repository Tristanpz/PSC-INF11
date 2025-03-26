# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:18:33 2025

@author: Utilisateur
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import collections as col
import torch
import operator

from entrainementMnist import X_test,y_test

##Lignes pour que ça marche chez Augustin
import os
os.chdir(r"C:\Users\Utilisateur\Documents\Augustin\X\2024.09 2A\Cours\PSC\Pytorch")

file="model_mnist784101010e0.pth"
model = torch.load(file, weights_only = False)
model.eval()

#%%
liste_entrees= X_test

##Calcul des distances pour chaque point
liste_distances=[model.distance(x) for x in liste_entrees]

##Activation de chacun des points
liste_act=[model.activations_bin(x) for x in liste_entrees]

##Dictionnaire: clés=état d'activation (en binaire), valeurs= nombre de points dans la facette
dic_facettes=col.Counter(liste_act)

##liste contenant des coupes (etat d'activation, nombre de points dans la facette) triés par nombre de pts décroissant
sorted_facettes=sorted(dic_facettes.items(), key=operator.itemgetter(1), reverse=True)

#%%# Etude du réseau


def carac_reseau(): #Imprime l'architecture du réseau actuel
    print("Architecture :",model.archi)
    print("Nombre de couches internes:",len(model.archi)-2)

def strCaracReseau():
    return "Architecture :" + str(model.archi) + " " + file[-6:-4]
    
def enumeration_facettes(): #Renvoie le nombre de facettes
    x=range(len(dic_facettes.keys()))
    h=sorted([dic_facettes[act]for act in dic_facettes.keys()],reverse=True)
    plt.bar(x,h)
    plt.suptitle("Répartition des points sur les facettes")
    plt.title(strCaracReseau())
    plt.show()
    return len(dic_facettes.keys())

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
    plt.suptitle("Histogramme représentant le nombre de facettes selon leur taille (nombre de points de l'espace d'entrée)")
    plt.title(strCaracReseau())
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
    plt.hist(distance_entrees_facette,bins="auto")
    plt.axvline(meandist(act),color="red", label="Distance moyenne")
    plt.suptitle("Distribution des distances à la frontière pour la facette "+str(act)+" nombre de points: "+str(dic_facettes[act]))
    plt.title(strCaracReseau())
    plt.xlabel("Distance à la frontière")
    plt.ylabel("Nombre de points")
    plt.legend()
    plt.show()

def distribution_facettes(n):
    f_max=facettes_max(n)
    for i in range(n):
        distribution_distance(f_max[i][0])


#%%#Test du reseau 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
        
def predire(res) :
     res=softmax(res)
     return np.argmax(res)   

def accuracy() :
    predictions = model(X_test).detach().numpy()
    return np.mean([1 if (predire(predictions[i])==y_test[i]) else 0 for i in range(len(predictions))] )
        

def test():              
    """affiche la prédiction de notre modèle et la classe attendue pour tous les datas de test"""
    predictions = model(X_test).detach().numpy()

    print("Prédictions du modèle:")
    for i in range(len(predictions)) :
        print(predire(predictions[i]), y_test[i])



