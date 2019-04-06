
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from tkinter import *

varianzag = [] # precios de cierre de la accion de google
preciog = [] # precios apertura de la accion de google
preciom = [] # precios de la accion de google mañana
preciof = [] # precios de la accion de apple
precioa = [] # precios de la accion de facebook
results = [] # resultados de algo
preciofn = [] # precios de la accion de apple
precioan = [] # precios de la accion de facebook

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # skipping column names
        line_count = 0
        for row in csvFileReader:
            if line_count > 0:
                varianzag.append(float(row[0]))#varianza
                preciog.append(float(row[1]))#PAGH
                preciom.append(float(row[2])) #PAGM
                preciof.append(float(row[3]))#PAFH
                precioa.append(float(row[4])) #PAAH
            line_count += 1
    return

def get_data_test(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # skipping column names
        line_count = 0
        for row in csvFileReader:
            if line_count > 0:
                preciofn.append(float(row[3]))#PAFH
                precioan.append(float(row[4])) #PAAH
            line_count += 1
    return


def write_data(filename, R ):
    with open(filename, 'w') as csvfile:
        fieldnames = ['result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x in range( len(R)):
            if x>0:
                writer.writerow( { 'result': float(R[x]) } )
            #else:
            #   writer.writerow( {'number': x, 'result': np.round(float(R[x]),4) } )
    return

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 4
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3#8
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        #print(self.W1)
        #print("")
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        #print(self.W2)
        #print("")

    def forward(self, X):
        #Propogate inputs though network nhidden - 1 dots
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        #yHat = self.sigmoid(self.z3-0.5)) # Amplitud
        yHat = np.tanh(self.z3)
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4 # Valor de perturbacion

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e 
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)
        return numgrad 
        
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y, tol):
        #Make an internal variable for the callback function:
        self.tol = tol
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200,'gtol': tol } #'disp' : True para mostrar mensajes de convergencia
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res   
        
        
if __name__ == "__main__":

    NN = Neural_Network()
    T = trainer(NN)
    get_data('RNA2.csv')

    varmax = np.amax( np.abs(varianzag) , axis=0)
    varmin = np.amin( np.abs(varianzag), axis=0)
    varianzag = varianzag/np.amax(np.abs(varianzag),axis=0)
    preciog = preciog/np.amax(preciog, axis=0)
    preciom1 = preciom
    preciom = preciom/np.amax(preciom, axis=0)
    preciof = preciof/np.amax(preciof, axis=0)
    precioa = precioa/np.amax(precioa, axis=0)



    NN = Neural_Network()
    T = trainer(NN)
    #Empieza el contador de tiempo
    start = time.time()
    for y in range(0,len(varianzag)-1,3):
        Y = []
        P = []
        Y = np.array(( [preciom[y],preciog[y], precioa[y], preciof[y]],[preciom[y+1],preciog[y+1], precioa[y+1], preciof[y+1]],[preciom[y+2],preciog[y+2], precioa[y+2], preciof[y+2]] ) ,dtype=float)
        P = np.array(([varianzag[y]],[varianzag[y+1]],[varianzag[y+2]]), dtype=float)
        T.train(Y,P,0.1)#Y,P y porcentaje de tolerancia
    end = time.time()

    print("")
    print("Entrenamiento Completado")
    print("")

    get_data_test('Entrenamiento.csv') 

    preciofn = preciofn/np.amax(preciofn, axis=0)
    precioan = precioan/np.amax(precioan, axis=0)
    vara = varianzag[len(varianzag)-1]
    Hoy = preciog[len(preciog)-1]/(np.amax(preciog, axis=0))

    Hoye = Hoy
    Actual = preciom1[-1:]


    for x in range(1,len(precioan)):  

        #print("")
        Z = np.array( ([vara,Hoye,precioan[x],preciofn[x]]) ,dtype=float)
        z = NN.forward(Z)
        #print(Z, "vector de entrada")
        #print(vara,x,"Entrada")

        variacionmañana=(z)# seccion critica
        

        #variacionmañana=(2*(varmax+np.abs(varmin))*(z-0.05))# seccion critica

        variacionmañana = z

        #print(z, "Salida de la RNA")
        #print(variacionmañana,"Salida")

        Actual =(1+variacionmañana)*Actual
        #print(Actual,"Precio")
        results.append(Actual)
        vara = variacionmañana
        Hoye = Hoye*(1+z)
        #print("")



    write_data('pronosticos.csv',results) # muestra los resultados
    print(end - start, "Tiempo de Entrenamiento en segundos")
    print("")
    print(np.mean(results, axis=0), "media de los pronosticos")
    print("")
    print(np.var(results, axis=0), "varianza de los pronosticos")
    print("")
    print(varmax, "Variacion Max")
    print("")
    print(varmin, "Variacion Min")
    
    x2 = np.arange(len(preciom1))
    x = np.arange(len(preciom1),len(results)+len(preciom1))

    plt.plot(x, results, label='pronostico')
    plt.plot(x2, preciom1, label='real')
    plt.xlabel('dias')
    plt.ylabel('precio de la accion')
    plt.legend()
    plt.tight_layout()
    plt.show()