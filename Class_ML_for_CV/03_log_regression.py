import numpy as np
import matplotlib.pyplot as plt

c1 = np.random.multivariate_normal([10,7], [[8,3],[3,2]], 100)
c2 = np.random.multivariate_normal([15,6], [[2,0],[0,2]], 40)
num_x = len(c1)+len(c2)

plt.scatter(c1[:,0],c1[:,1],s= 9,color='blue',marker='*',label='class1')
plt.scatter(c2[:,0],c2[:,1],s= 9,color='orange',marker='o',label='class2')

x = np.concatenate((c1,c2),axis=0)
x = np.concatenate((np.ones((num_x,1)), x), axis=1)
t = np.concatenate((np.ones((c1.shape[0],1)), np.zeros((c2.shape[0],1))))
w_init = np.random.randn(3,1)
w = w_init.copy()

max_ite = 50

def stable_sigmoid(x):
    "Numerically stable sigmoid function. sigma(x) = 1-sigma(-x)"
    sigm = np.zeros_like(x)
    sigm[x>=0] = 1 / (1 + np.exp(-x[x>=0]))
    sigm[x<0] = np.exp(x[x<0]) / (1 + np.exp(x[x<0]))
    return sigm

#################################################################################################
#2nd order Newton's method

binary_ce = np.zeros((max_ite,1))
eps = np.finfo(float).eps

for i in range(max_ite):
    # y_pred = 1/(1+np.exp(-x.dot(w)))
    y_pred = stable_sigmoid(x.dot(w))
    binary_ce[i] = -1/num_x*sum(t*np.log(y_pred+eps) + (1-t)*np.log(1-y_pred+eps))
    grad = 1/num_x * x.T @ (y_pred-t)
    # hessian matrix
    y_pred = y_pred.reshape(-1)
    H = 1/num_x * x.T @ np.diag(y_pred*(1-y_pred)) @ x
    
    # beware of Hessian matrix reguralization to avoid singular matrix inversion (np.linalg.det(H))
    w = w - np.linalg.pinv(H + 800./num_x*np.eye(H.shape[0])) @ grad

x_plot = np.linspace(5,20,7)
y_plot = -w[0]/w[2] - w[1]/w[2]*x_plot
plt.plot(x_plot,y_plot,'r',label='Newtons method' )
plt.xlabel('x1');plt.ylabel('x2')

#################################################################################################
# Batch Gradient Descent

lr = 0.05
cost_val_batch = np.zeros((max_ite,1))
w = w_init.copy()


for i in range(max_ite):
    # logistic function in vectorized form
    y_pred = stable_sigmoid(x.dot(w))
    #log-loss a.k.a cost function
    cost_val_batch[i] = -1/num_x*sum(t*np.log(y_pred+eps) + (1-t)*np.log(1-y_pred+eps))
    #gradient of cost function
    grad = 1/num_x * x.T @ (y_pred-t)
    #update weight vectors
    w = w - lr*grad

x_plot = np.linspace(5,20,7)
y_plot = -w[0]/w[2] - (w[1]/w[2])*x_plot
plt.plot(x_plot,y_plot,'k',label='Batch Gradient Descent')

plt.legend()
plt.figure()
plt.plot(cost_val_batch,'k', label='Batch Gradient Descent')
plt.plot(binary_ce,'r', label='Newtons method')
plt.title('2nd order Newtons method')
plt.xlabel('# of iterations')
plt.ylabel('cost')
plt.legend()
plt.show()

