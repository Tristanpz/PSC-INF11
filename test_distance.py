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



def test_distance(point, nombre_test, epsilon):
    activation = model.activations_bin(point) # Facette du point de référence

    p = 2*torch.rand(nombre_test, 3) -1
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



# Generation d'un point de reference
point_ref = torch.tensor(np.random.uniform(-10, 10, (3,)), dtype=torch.float32)

test_distance(point_ref, 500, 50*model.distance(point_ref))
print(model.distance(point_ref))



    
