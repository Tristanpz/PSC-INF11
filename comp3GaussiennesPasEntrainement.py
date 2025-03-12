# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:18:07 2025

@author: trist
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import collections as col
import torch

modelNonEnt1 = torch.load("model_3g3nonEnt1.pth", weights_only = False)
modelNonEnt1.eval()
modelNonEnt2 = torch.load("model_3g3nonEnt2.pth", weights_only = False)
modelNonEnt2.eval()
modelNonEnt3 = torch.load("model_3g3nonEnt3.pth", weights_only = False)
modelNonEnt3.eval()

model1 = torch.load("model_3g3.pth", weights_only = False)
model1.eval()
modelBis = torch.load("model_3g3bis.pth", weights_only = False)
modelBis.eval()
L_file = ["model_3g3nonEnt1.pth", "model_3g3nonEnt2.pth", "model_3g3nonEnt3.pth", "model_3g3.pth", "model_3g3bis.pth"]
listeModels = [modelNonEnt1, modelNonEnt2, modelNonEnt1, model1, modelBis]
n = len(listeModels)
N = 1000

x=np.linspace(-4,4,int(np.sqrt(N)))
y=np.linspace(-4,4,int(np.sqrt(N)))
X, Y =np.meshgrid(x,y, indexing='ij')
points=np.column_stack((X.ravel(),Y.ravel()))
liste_entrees=torch.tensor(points, dtype=torch.float32)


##Calcul des distances pour chaque point
L_liste_distances=[[listeModels[i].distance(x) for x in liste_entrees] for i in range(n)]

##Activation de chacun des points
L_liste_act=[[listeModels[i].activations_bin(x) for x in liste_entrees] for i in range(n)]

##Dictionnaire: clés=état d'activation (en binaire), valeurs= nombre de points dans la facette
L_dic_facettes=[col.Counter(L_liste_act[i]) for i in range(n)]

L_nb_facettes = [len(L_dic_facettes[i]) for i in range(n)]

print(L_nb_facettes)



## Etude du réseau


def carac_reseau(i): #Imprime l'architecture du réseau actuel
    print("Architecture :",listeModels[i].archi)
    print("Nombre de couches internes:",len(listeModels[i].archi)-2)
    
def enumeration_facettes(i): #Renvoie le nombre de facettes, et le nombre de points par facettes
    x=range(len(L_dic_facettes[i]))
    h=sorted([L_dic_facettes[i][act]for act in L_dic_facettes[i]])
    plt.bar(x,h)
    plt.suptitle("Répartition des points sur les facettes pour N = "+str(N))
    plt.title(L_file[i])
    plt.show()
    return len(L_dic_facettes[i]),L_dic_facettes[i]

def partition_facettes(idx):##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des indices des entrées dans la facette)
    partition={}
    for act in L_dic_facettes[idx]: #je choisis une facette
        partition[act]=[]
    for i in range(len(L_liste_distances[idx])):
       partition[L_liste_act[idx][i]].append(i)
    return partition

def poids_facettes(idx): #Retourne un histogramme du nombre de facettes en fonction du nombre de points dedans
    dico={}
    maximum=0
    for act in L_dic_facettes[idx] :
        nbe=L_dic_facettes[idx][act]
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

     
def mesures_facettes(idx):##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des distances des entrées dans la facette)
    partition=partition_facettes()
    mesures={}
    for act in partition.keys():
        mesures[act]=np.array([L_liste_distances[idx][j] for j in partition[act]])
    return mesures

def moyenne(idx): #Renvoie des données sur les distances
    mesures=mesures_facettes()
##    for act in mesures.keys():
##        print(act,"moyenne: ",np.mean(mesures[act]),"max: ",np.max(mesures[act]),"nombres de points: ",len(mesures[act]),"écart-type: ",np.std(mesures[act]))
    x=range(len(L_dic_facettes[idx]))
    h=[mesures[act][0]for act in mesures.keys()]
    plt.bar(x,sorted(h, reverse = True), color = 'b')
    plt.suptitle("Distance aux frontières moyenne pour chaque facette")
    plt.title("Architecture : " + str(listeModels[i].archi))
    plt.xlabel("facette")
    plt.ylabel("distance moyenne")
    plt.show()


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

def accuracy(N, i) :
    NbVrai = 0 
    for _ in range(N) : 
        res,donnee = genDonneesGaussiennes()
        res = format_res(res)
        X_tensor = torch.tensor(donnee, dtype = torch.float32)
        prediction = predire(listeModels[i](X_tensor).detach().numpy())
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

    predictions = listeModels[i](X_test_tensor).detach().numpy()

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

def affichage_prediction(i):   
    X = np.linspace(-4,4,20)
    Y = np.linspace(-4, 4,20)
    for x in X :
        for y in Y:
            plt.scatter(x, y, c = couleur(predire(listeModels[i](torch.tensor([x,y],dtype = torch.float32)).detach().numpy())), linewidths=0.2)
    plt.suptitle("Prédiction du réseau sur R2")
    plt.title("Architecture : " + str(listeModels[i].archi))
    plt.show()

list_colors=[]
dico_colors=mcolors.CSS4_COLORS
for key in dico_colors:
    list_colors.append(dico_colors[key])
    
def visualisation_facettes_simple(i):#crée des points (on peut mettre moins de N points)
    X = np.linspace(-4,4,50)
    Y = np.linspace(-4, 4,50)
    for x in X:
        for y in Y:    
            plt.scatter(x,y,c=list_colors[listeModels[i].activations_bin(torch.tensor([x,y],dtype = torch.float32))%148],linewidth=0.2)
    plt.suptitle("Visualisation des facettes sur l'espace d'entrée")
    plt.title("Architecture : " + str(listeModels[i].archi))
    plt.show()
    
def visualisation_facettes(i):#se base sur les entrées calculées au début (N points)
    list_act_colors=[list_colors[j%148] for j in L_liste_act[i]]
    plt.scatter(liste_entrees[:,0],liste_entrees[:,1],c=list_act_colors,linewidth=0.2)
    plt.suptitle("Visualisation des facettes sur l'espace d'entrée")
    plt.title("Architecture : " + str(listeModels[i].archi))
    plt.show()

for i in range(n) :
    print(accuracy(1000, i))
    affichage_prediction(i)