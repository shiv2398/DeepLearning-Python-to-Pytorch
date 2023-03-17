from sklearn.metrics import accuracy_score
import numpy as np
class MPNeuron:
    def __init__(self):
        self.b=None
        
    def model(self,x):
        return (sum(x)>=self.b)
    
    def predict(self,X):
        y=[]
        for x in X:
            result=self.model(x)
            y.append(result)
        return np.array(y)
    def fit(self,X,Y):
        
        accuracy={}
        
        for b in range(X.shape[1]+1):
            self.b=b
            y_pred=self.predict(X)
            accuracy[b]=accuracy_score(y_pred,Y)
            
        best_b=max(accuracy,key=accuracy.get)
        self.b=best_b
        
        print('Optimal values of b is ',best_b)
        print('Highest accuracy is ',accuracy[best_b])
            