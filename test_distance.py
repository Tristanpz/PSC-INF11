import torch
import numpy as np
import reseauPytorch
import matplotlib.pyplot as plt
import collections as col


model = torch.load("model_sphere.pth")
model.eval()


N=10000

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


#fonction qui prend des pts random a une distance epsilon dun pt de reference
#et qui determine si ils sont tous ds la meme facette
def test_distance(point, nombre_test, epsilon):
    activation = model.activations_bin(point) # Facette du point de référence

    p = 2*torch.rand(nombre_test, 3) - 1
    p = p / torch.norm(p, dim=1, keepdim=True)
    points_proches = point + epsilon * p


    activations = [model.activations_bin(p) for p in points_proches]

    meme_facette = True

    for act in activations:
        if act != activation:
            meme_facette = False
            break
        

    print(f"Point de référence: {point.numpy()}")
    print(f"Activation de référence: {activation}")
    print(f"Tous dans la même facette ? {'Oui' if meme_facette else 'Non'}")

    

#fonction qui prend un pt et des directions au hasard et qui verifie si on est
#sorti de la facette, sinon on agrandit epsilon puis dichotomie
def sortie_facette(point, nbr_directions, epsilon):

    activation = model.activations_bin(point) 
    facteur = 1.0
    dernier_dedans = 0.0
    premier_dehors = float('inf')
    b = True
    
    while b:
        direction = 2 * torch.rand(nbr_directions, 3) - 1
        for i in range(nbr_directions) :
            direction[i,:] = direction[i,:] / torch.norm(direction[i,:])
        points_test = point + facteur * epsilon * direction

        activations = [model.activations_bin(p) for p in points_test]
        
        for act in activations:
            if act != activation:
                premier_dehors = facteur
                dernier_dedans = facteur / 2
                b = False
            
        facteur = facteur * 2

    while abs(premier_dehors - dernier_dedans) > epsilon:
        milieu = (premier_dehors + dernier_dedans) / 2
        direction = 2 * torch.rand(nbr_directions, 3) - 1
        for i in range(nbr_directions) :
            direction[i,:] = direction[i,:] / torch.norm(direction[i,:])
        points_test = point + milieu * epsilon * direction
        activations = [model.activations_bin(p) for p in points_test]
        b = True
        
        for act in activations:
            if act != activation:
                b = False
        if b :
            dernier_dedans = milieu
        else :
            premier_dehors = milieu

    return (premier_dehors+dernier_dedans)*epsilon / 2
        

#Generation d'un point de reference
point_ref = torch.tensor(np.random.uniform(-10, 10, (3,)), dtype=torch.float32)
#point_ref = torch.tensor([-7.9132, -4.3461,  3.4173])

#test_distance(point_ref, 4, 50*model.distance(point_ref))

    
def quality_check(N):
    L = np.zeros(N)
    for i in range (N):
        point = torch.tensor(np.random.uniform(-10, 10, (3,)), dtype=torch.float32)
        sous_approximation = model.distance(point)
        distance_empirique = sortie_facette(point, 100, 0.001)
        L[i] = distance_empirique / sous_approximation
    print(np.mean(L))
    print(np.std(L))

quality_check(100)
    




