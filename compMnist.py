# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:58:27 2025

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


file1="model_mnist784101010ne0.pth"
model1 = torch.load(file1, weights_only = False)
model1.eval()

file2="model_mnist784101010e0.pth"
model2 = torch.load(file2, weights_only = False)
model2.eval()

listmodels=[model1,model2]
listfiles=[file1,file2]
k=len(listmodels)
#%%
liste_entrees= X_test

##Calcul des distances pour chaque point
L_liste_distances=[[listmodels[i].distance(x) for x in liste_entrees]for i in range(k)]

##Activation de chacun des points
L_liste_act=[[listmodels[i].activations_bin(x) for x in liste_entrees] for i in range(k)]

##Dictionnaire: clés=état d'activation (en binaire), valeurs= nombre de points dans la facette
L_dic_facettes=[col.Counter(L_liste_act[i]) for i in range(k)]

##liste contenant des coupes (etat d'activation, nombre de points dans la facette) triés par nombre de pts décroissant
L_sorted_facettes=[sorted(L_dic_facettes[i].items(), key=operator.itemgetter(1), reverse=True) for i in range(k)]

#%%# Etude du réseau


def carac_reseau(i): #Imprime l'architecture du réseau actuel
    print("Architecture :",listmodels[i].archi)
    print("Nombre de couches internes:",len(listmodels[i].archi)-2)
    
def enumeration_facettes(i): #Renvoie le nombre de facettes
    x=range(len(L_dic_facettes[i].keys()))
    h=sorted([L_dic_facettes[i][act]for act in L_dic_facettes[i].keys()],reverse=True)
    plt.bar(x,h)
    plt.suptitle("Répartition des points sur les facettes")
    plt.title(listfiles[i])
    plt.show()
    return len(L_dic_facettes[i].keys())

def facette_max(i): ##renvoie la facette la plus grande et son nombre de points
    nbpoints=[L_dic_facettes[i][act] for act in L_dic_facettes[i]]
    
    nbpoints=0
    for act in L_dic_facettes[i]:
        if L_dic_facettes[i][act]>nbpoints:
            nbpoints=L_dic_facettes[i][act]
            actmax=act
    return actmax,nbpoints

def facettes_max(i,n):
    return L_sorted_facettes[i][:n]

def partition_facettes(i):##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des indices des entrées appartenant à la facette)
    partition={}
    for act in L_dic_facettes[i].keys(): #je choisis une facette
        partition[act]=[]
    for k in range(len(L_liste_distances[i])):
       partition[L_liste_act[i][k]].append(k)
    return partition

def poids_facettes(i): #Retourne un histogramme du nombre de facettes en fonction du nombre de points dedans
    dico={}
    maximum=0
    for act in L_dic_facettes[i] :
        nbe=L_dic_facettes[i][act]
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

     
def mesures_facettes(i):##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des distances des entrées dans la facette)
    partition=partition_facettes(i)
    mesures={}
    for act in partition.keys():
        mesures[act]=np.array([L_liste_distances[i][j] for j in partition[act]])
    return mesures

def moyenne_graphe(i): #Renvoie des données sur les distances
    mesures=mesures_facettes()
    x=range(len(L_dic_facettes[i].keys()))
    h=[mesures[act][0]for act in mesures.keys()]
    plt.bar(x,sorted(h, reverse = True), color = 'b')
    plt.suptitle("Distance aux frontières moyenne pour chaque facette")
    plt.title(listfiles[i])
    plt.xlabel("facette")
    plt.ylabel("distance moyenne")
    plt.show()

def maxdist(act,i):
    mesures=mesures_facettes(i)
    return np.max(mesures[act])

def meandist(act,i):
    mesures=mesures_facettes(i)
    return np.mean(mesures[act])

def stddist(act,i):
    mesures=mesures_facettes(i)
    return np.std(mesures[act])

def distribution_distance(i,act):
    mesures=mesures_facettes(i)
    distance_entrees_facette=mesures[act]
    plt.hist(distance_entrees_facette,bins="auto")
    plt.axvline(meandist(act,i),color="red", label="Distance moyenne")
    plt.suptitle("Distribution des distances à la frontière pour la facette "+str(act)+" nombre de points: "+str(L_dic_facettes[i][act]))
    plt.xlabel("Distance à la frontière")
    plt.ylabel("Nombre de points")
    plt.title(listfiles[i])
    plt.legend()
    plt.show()

def distribution_facettes(n,i):
    f_max=facettes_max(i,n)
    for k in range(n):
        distribution_distance(i,f_max[k][0])


#%%#Test du reseau 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
        
def predire(res) :
     res=softmax(res)
     return np.argmax(res)   

def accuracy(i) :
    predictions = listmodels[i](X_test).detach().numpy()
    return np.mean([1 if (predire(predictions[i])==y_test[i]) else 0 for i in range(len(predictions))] )
        

def test(i):              
    """affiche la prédiction de notre modèle et la classe attendue pour tous les datas de test"""
    predictions = listmodels[i](X_test).detach().numpy()

    print("Prédictions du modèle:")
    for i in range(len(predictions)) :
        print(predire(predictions[i]), y_test[i])

