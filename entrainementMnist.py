# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:59:40 2025

@author: Utilisateur
"""
import os
os.chdir(r"C:\Users\Utilisateur\Documents\Augustin\X\2024.09 2A\Cours\PSC\Pytorch")
import torch
import numpy as np
import reseauPytorch

import keras as ks

(x_train,y_train),(x_test,y_test) = ks.datasets.mnist.load_data()

x_train=x_train.reshape(-1,28*28)/255.0
y_train = ks.utils.to_categorical(y_train, 10)

x_test=x_test.reshape(-1,28*28)/255.0
# y_test = ks.utils.to_categorical(y_test, 10)

X_train = torch.tensor(x_train, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.float32)

X_test= torch.tensor(x_test, dtype=torch.float32)
# Y_test = torch.tensor(y_test, dtype=torch.float32)

num_epochs = 80
nbBatch = 200

model = reseauPytorch.creerReseau([28**2,28,10,10])
reseauPytorch.entrainerReseau(model, X_train, Y_train, num_epochs, nbBatch)


torch.save(model, "model_mnist784281010e0.pth")

