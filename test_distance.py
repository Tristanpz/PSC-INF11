import torch
import numpy as np

model = torch.load("model_sphere.pth", weights_only=False)
model.eval()


N=1000

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
    ''' entrée : point d'entrée, un nombre de direction (int)
    et epsilon (float) la précision de notre évaluation
        sortie : un float estiment empiriquement la distance 
        du point aux frontières des facettes '''
    activation = model.activations_bin(point) 
    facteur = 1.0
    dernier_dedans = 0.0
    premier_dehors = float('inf')
    b = True
    
    #on augmente exponentiellement la longueur du rayon 
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
        #On fait une dichotomie pour avoir une estimation précise
        #de la distance empirique
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
        
def quality_check(N):
    #La liste L sera composée des écart-relatifs à la distance empirique
    L = np.zeros(N)
    for i in range (N):
        point = torch.tensor(np.random.uniform(-10, 10, (3,)), dtype=torch.float32)
        sous_approximation = model.sousDistance(point)
        
        distance_empirique = sortie_facette(point, 100, 0.001)
        L[i] = abs(distance_empirique-sous_approximation)/distance_empirique
        print(L[i],sous_approximation, distance_empirique)
    print(np.mean(L))
    print(np.std(L))

quality_check(100)

def verifAct(N) :
    ''' permet de vérifier que la fonction activation produit 
    les mêmes résultats que activation0 '''
    for i in range (N):
        point = torch.tensor(np.random.uniform(-10, 10, (3,)), dtype=torch.float32)
        if model.activations_bin(point) != model.activations_bin0(point) :
            print(i, point)
            return
    print("RAS")
    return

