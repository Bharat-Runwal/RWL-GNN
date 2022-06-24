import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.utils import accuracy
import matplotlib.pyplot as plt
import warnings
from utils import *

class RwlGNN:
    """ RWL-GNN (Robust Graph Neural Networks using Weighted Graph Laplacian)
    Parameters
    ----------
    model:
        model: The backbone GNN model in RWLGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
    See details in https://github.com/Bharat-Runwal/RWL-GNN.
    """

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)

        # self.train_cost = []
        self.valid_cost = []
        self.train_acc = []
        self.valid_acc = []
    
    def plot_cost(self):
        plt.figure(figsize=(10,6))
        plt.plot(range(len(self.valid_cost)),self.valid_cost,label="Validation_Cost")
        
        plt.xlabel("No. of iterations")
        plt.ylabel("cost")
        plt.title("Cost function Convergence")
        plt.legend()
        plt.savefig(f"{self.args.beta}_{self.args.dataset}_{self.args.ptb_rate}_VAL_COST.png")
        plt.show()
     

    def plot_acc(self):
        plt.figure(figsize=(10,6))
        plt.plot(range(len(self.train_acc)),self.train_acc,label ="Train_acc") 
        plt.plot(range(len(self.valid_acc)),self.valid_acc,label = "Valid_acc")       
        plt.xlabel("No. of iterations")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.savefig(f"{self.args.beta}_{self.args.dataset}_{self.args.ptb_rate}_ACC.png")
        plt.show()

    def fit(self, features, adj, labels, idx_train, idx_val):
        """Train RWL-GNN.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        args = self.args
        self.symmetric = args.symmetric
        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        
        optim_sgl = args.optim
        lr_sgl = args.lr_optim

        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L_noise = D - adj

        # INIT
        self.weight = self.Linv(L_noise)
      
        self.weight.requires_grad = True
        self.weight = self.weight.to(self.device)
        # self.weight = torch.rand(int(n*(n-1)/2),dtype=torch.float,requires_grad=True,device = self.device)
    
        c = self.Lstar(2*L_noise*args.alpha - args.beta*(torch.matmul(features,features.t())) )
        if optim_sgl == "Adam":
            self.sgl_opt =AdamOptimizer(self.weight,lr=lr_sgl)
        elif optim_sgl == "RMSProp":
            self.sgl_opt = RMSProp(self.weight,lr = lr_sgl)
        elif optim_sgl == "sgd_momentum":
            self.sgl_opt = sgd_moment(self.weight,lr=lr_sgl)
        else:
            self.sgl_opt = sgd(self.weight,lr=lr_sgl) 

        t_total = time.time()
        
        for epoch in range(args.epochs):
            if args.only_gcn:
                estimate_adj = self.A()
                self.train_gcn(epoch, features, estimate_adj,
                        labels, idx_train, idx_val)
            else:
                
                for i in range(int(args.outer_steps)):
                    self.train_specific(epoch, features, L_noise, labels,
                            idx_train, idx_val,c)

                for i in range(int(args.inner_steps)):
                    estimate_adj = self.A()
                    self.train_gcn(epoch, features, estimate_adj,
                            labels, idx_train, idx_val)

                # if args.decay == "y":
                #     if epoch % 100 == 0:
                #       self.sgl_opt.lr =args.lr_init * (args.lr_decay_rate)
                #       print('The learning rate was set to {}.'.format(self.sgl_opt.lr))


        if args.plots=="y":
            self.plot_acc()
            self.plot_cost()

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(args)

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)



    def w_grad(self,alpha,c):
      with torch.no_grad():
        grad_f = self.Lstar(alpha*self.L()) - c
        return grad_f 



    def train_specific(self,epoch, features, L_noise, labels, idx_train, idx_val,c):
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        
        y = self.weight.clone().detach()
        y = y.to(self.device)
        y.requires_grad = True

        loss_fro = args.alpha* torch.norm(self.L(y) - L_noise, p='fro')     
        normalized_adj = self.normalize(y)
        loss_smooth_feat =args.beta* self.feature_smoothing(self.A(y), features)


        output = self.model(features, normalized_adj)
        loss_gcn =args.gamma * F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # loss_diffiential = loss_fro + gamma*loss_gcn+args.lambda_ * loss_smooth_feat

        gcn_grad = torch.autograd.grad(
        inputs= y,
        outputs=loss_gcn,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(loss_gcn),
        only_inputs= True,
          )[0]
      
        sgl_grad = self.w_grad(args.alpha ,c)
        

        total_grad  = sgl_grad + gcn_grad 

      
        self.weight = self.sgl_opt.backward_pass(total_grad)
        self.weight = torch.clamp(self.weight,min=0)

        total_loss = loss_fro \
                    +  loss_gcn \
                    + loss_smooth_feat 

        self.model.eval()
        normalized_adj = self.normalize()
        output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if args.plots == "y":
            self.train_acc.append(acc_train.detach().cpu().numpy())
            self.valid_cost.append(loss_val.detach().cpu().numpy())
            self.valid_acc.append(acc_val.detach().cpu().numpy())
            # self.train_cost.append(total_loss.detach().cpu().numpy())
        
        if args.test=="n":
            print('Epoch: {:04d}'.format(epoch+1),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                


                

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        # estimator = self.estimator
        adj = self.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward(retain_graph = True)
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    def test(self, features, labels, idx_test):
        """Evaluate the performance of RWL-GNN on test set
        """
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
  
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


    def A(self,weight=None):
        # with torch.no_grad():
        if weight == None:
            k = self.weight.shape[0]
            a = self.weight
        else:
            k = weight.shape[0]
            a = weight
        n = int(0.5 * (1 + np.sqrt(1 + 8 * k)))
        Aw = torch.zeros((n,n),device=self.device)
        b=torch.triu_indices(n,n,1)
        Aw[b[0],b[1]] =a
        Aw = Aw + Aw.t()
        return Aw


    def L(self,weight=None):
        if weight==None:
            k= len(self.weight)
            a = self.weight 
        else:
            k = len(weight)
            a = weight
        n = int(0.5*(1+ np.sqrt(1+8*k)))
        Lw = torch.zeros((n,n),device=self.device)
        b=torch.triu_indices(n,n,1)
        Lw[b[0],b[1]] = -a  
        Lw = Lw + Lw.t()
        row,col = np.diag_indices_from(Lw)
        Lw[row,col] = -Lw.sum(axis=1)
        return Lw     



    def Linv(self,M):
      with torch.no_grad():
        N=M.shape[0]
        k=int(0.5*N*(N-1))
        # l=0
        w=torch.zeros(k,device=self.device)
        ##in the triu_indices try changing the 1 to 0/-1/2 for other
        ## ascpect of result on how you want the diagonal to be included
        indices=torch.triu_indices(N,N,1)
        M_t=torch.tensor(M)
        w=-M_t[indices[0],indices[1]]
        return w


    def Lstar(self,M):
        N = M.shape[1]
        k =int( 0.5*N*(N-1))
        w = torch.zeros(k,device=self.device)
        tu_enteries=torch.zeros(k,device=self.device)
        tu=torch.triu_indices(N,N,1)
    
        tu_enteries=M[tu[0],tu[1]]
        diagonal_enteries=torch.diagonal(M)

        b_diagonal=diagonal_enteries[0:N-1]
        x=torch.linspace(N-1,1,steps=N-1,dtype=torch.long,device=self.device)
        x_r = x[:N]
        diagonal_enteries_a=torch.repeat_interleave(b_diagonal,x_r)
     
        new_arr=torch.tile(diagonal_enteries,(N,1))
        tu_new=torch.triu_indices(N,N,1)
        diagonal_enteries_b=new_arr[tu_new[0],tu_new[1]]
        w=diagonal_enteries_a+diagonal_enteries_b-2*tu_enteries
   
        return w

    def normalize(self,w=None):

        if self.symmetric:
            if w == None:
                adj = (self.A() + self.A().t())
            else:
                adj = self.A(w)
            
            adj = adj + adj.t()
        else:
            if w == None:
                adj = self.A()
            else:
                adj = self.A(w)

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
