from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

x,y = load_diabetes(return_X_y=True)

print(x.shape)
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

reg = LinearRegression()
reg.fit(X_train,y_train)


# print("reg coef_ = ", reg.coef_)
# print("reg intercept_ = ", reg.intercept_)


y_pred = reg.predict(X_test)
# print("reg predict = ", y_pred)

print("reg r2 score = ", r2_score(y_test,y_pred) )

class GDRegressor:
    def __init__(self, learning_rate = 0.01, epochs = 100):
        self.coef = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.n = None
       
    def fit(self,x,y):
        self.intercept_ = 0   
        self.coef = np.ones(x.shape[1])
        self.n = x.shape[1]
        
        for i in range(self.epochs): 
            
            y_hat  = np.dot(x, self.coef) + self.intercept_
            intercept_der = -2*np.mean(y - y_hat)
            self.intercept_ = self.intercept_ - (self.lr*intercept_der)
            coef_der = (-2/self.n) * (np.dot((y - y_hat), x))
            self.coef = self.coef - self.lr*coef_der
            
    def predict(self, x):
        return np.dot(x, self.coef) + self.intercept_
        
     
        
gdr = GDRegressor(epochs=1000, learning_rate=0.8)  
gdr.fit(X_train, y_train)
gdr_y_pred = gdr.predict(X_test)
     
# print("gdr coef_ = ", gdr.coef)
# print("gdr intercept_ = ", gdr.intercept_)
# print("gdr predict = ", gdr_y_pred)
print("gdr r2 score = ", r2_score(y_test,gdr_y_pred) )
     
     
            