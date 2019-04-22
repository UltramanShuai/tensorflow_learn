import numpy as np
import matplotlib.pyplot as plt
seed=2

def generateds():
    rdm=np.random.RandomState(seed)
    
    X = rdm.randn(300,2)
    
    Y_=[int(x0*x0+x1*x1<2) for (x0,x1) in X]
    
    Y_c=[["red" if y else "blue"] for y in Y_]
    
    X=np.vstack(X).reshape(-1,2)
    Y_=np.vstack(Y_).reshape(-1,1)
    
    return X, Y_, Y_c
    
    
if __name__ == '__main__':
    X,Y,Y_c = generateds()
    X = np.array(X)
    Y = np.array(Y)
    Y_c = np.array(Y_c)
    print("X:",X.shape)
    print("max(X):",max(X[:,1]))
    print("Y:",Y.shape)
    print("Y_c:",Y_c.shape)
    print("show_over")