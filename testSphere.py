import torch
import numpy as np
import reseauPytorch
import matplotlib.pyplot as plt
import collections as col


file = "model_sphere.pth"
model = torch.load(file)
model.eval()


N=10000

##liste_entrees= [torch.tensor(np.random.uniform(-10,10,3),dtype=torch.float32) for _ in range(N)]

x=np.linspace(-10,10,int(np.cbrt(N)))
y=np.linspace(-10,10,int(np.cbrt(N)))
z=np.linspace(-10,10,int(np.cbrt(N)))

X, Y,Z =np.meshgrid(x,y,z, indexing='ij')

points=np.column_stack((X.ravel(),Y.ravel(),Z.ravel()))

liste_entrees=torch.tensor(points, dtype=torch.float32)

##Calcul des distances pour chaque point
liste_distances=[model.distance(x) for x in liste_entrees]

##Activation de chacun des points
liste_act=[model.activations_bin(x) for x in liste_entrees]

##Dictionnaire: clés=état d'activation (en binaire), valeurs= nombre de points dans la facette
dic_facettes=col.Counter(liste_act)

def carac_reseau(): #Imprime l'architecture du réseau actuel
    print("Architecture :",model.archi)
    print("Nombre de couches :",len(model.archi))


def enumeration_facettes(): 
    x=range(len(dic_facettes.keys()))
    print(len(dic_facettes.keys()))
    h=[dic_facettes[act]for act in dic_facettes.keys()]
    plt.bar(x,h)
##problème pour l'affichage des labels
    plt.title("Répartition des points sur les facettes")
    plt.show()
    return len(dic_facettes.keys()),dic_facettes

##enumeration_facettes()


def poids_facettes():
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

##poids_facettes()

def partition_facettes():##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des indices des entrées dans la facette)
    partition={}
    for act in dic_facettes.keys(): #je choisis une facette
        partition[act]=[]
    for i in range(len(liste_distances)):
       partition[liste_act[i]].append(i)
    return partition

def mesures_facettes():##Renvoie un dico (clés=état d'activation (en binaire), valeur=liste des distances des entrées dans la facette)
    partition=partition_facettes()
    mesures={}
    for act in partition.keys():
        mesures[act]=np.array([liste_distances[j] for j in partition[act]])
    return mesures

def moyenne(): #Renvoie des données sur les distances
    mesures=mesures_facettes()
##    for act in mesures.keys():
##        print(act,"moyenne: ",np.mean(mesures[act]),"max: ",np.max(mesures[act]),"nombres de points: ",len(mesures[act]),"écart-type: ",np.std(mesures[act]))
    x=range(len(dic_facettes.keys()))
    h=[mesures[act][0]for act in mesures.keys()]
    plt.bar(x,sorted(h, reverse = True))
    plt.show()
    
moyenne()

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
    plt.suptitle("Distribution des distances à la frontière pour la facette "+str(act))
    plt.title(file)
    plt.legend()
    plt.show()

def couleur(res):
    if np.array_equal(res, [1, 0]):
        return 'b'
    if np.array_equal(res, [0, 1]):
        return 'r'




##test du reseau

##N_test=60

def genDonneesSpheriques():
    r = 10
    point = np.random.uniform(-10, 10, 3) 
    distance_carre = np.sum(point**2)

    if distance_carre < r**2:
        return [1, 0], point  
    else:
        return [0, 1], point


def predire(predictions):
    resultat = []
    for pred in predictions:
        if pred[0] > pred[1]:  
            resultat.append("Dans la sphère")
        else:
            resultat.append("En dehors de la sphère")
    return resultat

##X_test = np.zeros((N_test, 3))
##y_test = np.zeros((N_test, 2))


##for i in range(N_test):
##    res, donnee = genDonneesSpheriques()
##    X_test[i] = donnee
##    y_test[i] = res

##X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
##predictions = model(X_test_tensor).detach().numpy()
    
#sortie = predire(predictions)

##for i in range(len(predictions)):
##    print(f"Valeur réelle: {y_test[i]}, Prédiction: {predictions[i]}, Résultat: {sortie[i]}")

def accuracy(predictions, y_test):
    correct = 0
    for i in range(len(predictions)):
        pred = 1 if predictions[i][0] > predictions[i][1] else 0
        reel = 1 if y_test[i][0] > y_test[i][1] else 0
        if pred == reel:
            correct += 1
    return correct / len(predictions) * 100  

#print ("Accuracy: ",accuracy(predictions, y_test), "%")
etat_activation = list(dic_facettes.keys())[1]
distribution_distance(etat_activation)




