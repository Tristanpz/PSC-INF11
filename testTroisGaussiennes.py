# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:45:10 2025

@author: Utilisateur
"""

import torch
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import collections as col


model = torch.load("model_3g.pth", weights_only = False)
model.eval()

##Test distances

N= 10000
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
    return len(dic_facettes.keys())

def d():
    for act in dic_facettes: #je choisis une facette
        for i in range(len(liste_distances)):
            if liste_act[i]==act: #pour chaque point dans la facette
                pass
                
print(enumeration_facettes())
                



                