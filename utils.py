import torch
import numpy as np 

# class AdamOptimizer:
#     def __init__(self, weights, lr= 1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.m = 0
#         self.v = 0
#         self.t = 0
#         self.theta = weights
        
#     def backward_pass(self, gradient):
#         self.t = self.t + 1
#         self.m = self.beta1*self.m + (1 - self.beta1)*gradient
#         self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
#         m_hat = self.m/(1 - self.beta1**self.t)
#         v_hat = self.v/(1 - self.beta2**self.t)
#         self.theta = self.theta - self.lr*(m_hat/(np.sqrt(v_hat) - self.epsilon))
#         return self.theta


class AdamOptimizer:
    def __init__(self, weights, lr= 1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = torch.zeros_like(weights)
        self.v = torch.zeros_like(weights)
        self.t = 0
        self.theta = weights
        
    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        self.theta = self.theta - self.lr*(m_hat/(torch.sqrt(v_hat) + self.epsilon))
        return self.theta


class RMSProp:
    def __init__(self, weights, lr= 1e-2, decay_rate = 0.99, epsilon=1e-8):
        self.lr = lr
        self.beta = decay_rate
        self.epsilon = epsilon
        self.theta = weights
        self.v = torch.zeros_like(weights)

    def backward_pass(self, gradient):
        self.v = self.beta * self.v+ (1-self.beta) * (gradient*gradient)
        self.theta = self.theta - self.lr *(gradient/(torch.sqrt(self.v) +self.epsilon ))
        return self.theta

class sgd_moment:
    def __init__(self, weights, lr= 1e-2, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.theta = weights
        self.v = torch.zeros_like(weights)
        
    def backward_pass(self, gradient):
        self.v = self.momentum * self.v - self.lr * gradient
        self.theta = self.theta + self.v 
        return self.theta


class sgd:
    def __init__(self,weights,lr=1e-2):
        self.lr = lr
        self.theta = weights

    def backward_pass(self,gradient):
        self.theta = self.theta - self.lr * gradient
        return self.theta

