
import numpy as np
import csv




# X = (Approved in past trimester, Number of evaluations), y = Students Approved 
X = np.array(([20,3], [14,4], [5,3]), dtype=float)
y = np.array(([23], [16], [2]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
y = y/25 #Max test score is 100

dates = []

prices  = [] # precios de cierre de la accion de google
prices1 = [] # precios apertura de la accion de google
prices2 = [] # precios de la accion de apple
prices3 = [] # precios de la accion de facebook


results= []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) # skipping column names
        line_count = 0
        for row in csvFileReader:
            if line_count > 0:
                prices1.append(float(row[1]))
                prices2.append(float(row[2]))
                prices3.append(float(row[3]))
                prices.append(float(row[4]))
            line_count += 1

    return


def write_data(filename, R ):
    with open(filename, 'w') as csvfile:
        fieldnames = ['number', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x in range( len(R) ):

            if x==0:
                writer.writerow( {'number': x, 'result': round(R[x],4) } )
            else:
                writer.writerow( {'number': x, 'result': round(100*R[x]-abs(100*R[x-1]),4) } )

    return

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 3
        #self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 2#8
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        print(self.W1)
        print("")
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        print(self.W2)
        print("")

    def forward(self, X):
        #Propogate inputs though network nhidden - 1 dots
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
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
        e = 1e-4

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
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 100} #'disp' : True para mostrar mensajes de convergencia
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
        
        
if __name__ == "__main__":
    print("")
    NN = Neural_Network()
    T = trainer(NN)
    ### coñño recuerda que la salida se multiplica por el rango y se resta por el desplazamiento
    ### Entrand los datos del archivo de texto
    get_data('RNA.csv')
    ### Se normalizan los datos
    prices = prices/np.amax(prices, axis=0)
    prices1 = prices1/np.amax(prices1, axis=0)
    prices2 = prices2/np.amax(prices2, axis=0)
    prices3 = prices3/np.amax(prices3, axis=0)
    ### Entrenamiento

    NN = Neural_Network()
    T = trainer(NN)
    for y in range(0,len(prices)-1,3):
        Y = []
        P = []
        Y = np.array(( [prices1[y], prices2[y], prices3[y]],[prices1[y+1], prices2[y+1], prices3[y+1]],[prices1[y+2], prices2[y+2], prices3[y+2]] ) ,dtype=float)
        P = np.array(([prices[y]],[prices[y+1]],[prices[y+2]]), dtype=float)
        T.train(Y,P)

    Mañana = 1102.44
    Hoy = Mañana/np.amax(prices1, axis=0)
    ### Evaluacion
    Z = np.array(([Hoy,prices2[len(prices)-1],prices3[len(prices)-1]]), dtype=float)
    z = NN.forward(Z)
    print("")
    print("El valor de la accion variara un % :")
    print(z*6-3)
    print("")
    print("El valor de la accion mañana sera de :")
    print(round(((1+(z[0]*6-3)/100)*Mañana),2))
    ### Prediccion del dia anterior
        #imputs valor, Prices1[len(Prices1-1)],Prices2[len(Prices2-1)],Prices3[len(Prices3-1)]



    write_data('results.csv',prices)

