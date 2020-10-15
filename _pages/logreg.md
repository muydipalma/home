---
layout: page
title: Regresion Logistica
---



Creo datos de una sola variable y de dos clases,cada clase sera un distrubucion a la que le puedo cambiar el sigma (std) y el mu (center)


```python
import numpy as np
from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=50, centers=np.array([1,3]).reshape(-1, 1), n_features=1,random_state=1,cluster_std=0.8)
```





<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://muydipalma.github.io/home/fig0.html" height="525" width="100%"></iframe>





## Regresion Logistica desde 0

Escribo mi modelo, creo una clase y le doy el mismo formato que las clases de SKLEARN (con los metodos .fit, .predict)


```python
import copy as copy

class logisticreg:
    
    # iteracion que busca minizar la funcion de costo haciendo descenso gradiente
    
    def fit(self, X, y):
        
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # inicializamos los betas, ponemos betas=0             
        self.theta = np.zeros(X.shape[1])
        
        # loop principal
        for i in range(self.num_iter):
            
            # calculo p inicial
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            
            # calculo el gradiente
            gradient = np.dot(X.T, (h - y)) / y.size
            
            # asigno los nuevos betas en la direccion del gradiente
            # y con el paso lr
            
            self.theta -= self.lr * gradient
            
            # me guardo los betas de cada iteracion
            p=copy.copy(self.theta[:])
            self.coefs.append(p)
            ls=copy.copy(self.__loss(h, y))
            self.lss.append(ls)
            if(self.verbose == True):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {i,self.__loss(h, y)} \t')    
            #listo!
            
    #usa los betas encontrados
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    #usamos el threshold para la clasificacion
    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)
    
                
    def __init__(self, lr=0.01, num_iter=100000,
                 fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose=verbose
        self.coefs=list()
        self.lss=list()
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    
    #sigmoide
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    

    #funcion de costo
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()                
```


<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://muydipalma.github.io/home/assets/img/figi.html" height=900 width=750 ></iframe>

