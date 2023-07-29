import numpy as np
import numpy.matlib
from scipy.optimize import minimize
from sklearn.neighbors import kneighbors_graph
#from scipy import sparse
from scipy.spatial.distance import cdist
import pandas as pd
import cupy as cp
import cupyx as cpx
from cupyx.scipy import sparse
import blocked_matrix as bm
import scipy as sp
from cupyx.scipy.linalg import lu_solve, lu_factor
import linecache
import os
import tracemalloc
import sys

class LapSVM():
    def __init__(self, options=None, alpha=np.array([]), X=np.array([])):
        if options:
            self.options = options
        else:
            self.options = {'gamma_A':0.03125,
           'gamma_I':10000,
           'k_neighbor':5,
           'kernal_param':10,
           't':1}
        self.alpha = None if alpha.size == 0 else alpha
        self.X = None if X.size == 0 else X

    # def rbf(self, X1, X2):
    #     xx = dot(X1, X2.T, block_size=1000)
    #     #x1 = cp.matlib.repmat(cp.sum(X1 ** 2, axis=1), X2.shape[0], 1)
    #     #x2 = cp.matlib.repmat(cp.sum(X2 ** 2, axis=1), X1.shape[0], 1)
    #     x1 = cp.array(np.tile(np.sum(X1 ** 2, axis=1), (X2.shape[0], 1)))
    #     x2 = np.tile(np.sum(X2 ** 2, axis=1), (X1.shape[0], 1))
    #     x1 = cp.array(x2)
    #     xx = cp.array(xx)
    #     euclidean = cp.sqrt(x1.T + x2 - 2 * xx)
    #     del x1
    #     del x2
    #     del xx
    #     euclidean = cp.nan_to_num(euclidean)
    #     return cp.asnumpy(cp.exp(-euclidean ** 2 * self.options['kernal_param']))
    def rbf(self, X1, X2):
        xx = bm.dot(X1, X2.T)
        # x1 = cp.matlib.repmat(cp.sum(X1 ** 2, axis=1), X2.shape[0], 1)
        # x2 = cp.matlib.repmat(cp.sum(X2 ** 2, axis=1), X1.shape[0], 1)
        x1 = cp.tile(cp.sum(cp.array(X1) ** 2, axis=1), (X2.shape[0], 1))
        x1 = cp.asnumpy(x1.T)
        x2 = cp.tile(cp.sum(cp.array(X2) ** 2, axis=1), (X1.shape[0], 1))
        x2 = cp.asnumpy(x2)
        # print(5, psutil.virtual_memory().percent*0.638)
        euclidean = bm.add(x1, x2, numpy=False)
        del x1
        del x2
        euclidean = cp.asnumpy(euclidean)
        xx = bm.mul(2, xx)
        euclidean = bm.sub(euclidean, xx, numpy=False)
        del xx
        euclidean = bm.sqrt(euclidean, numpy=False)
        euclidean = bm.nan_to_num(euclidean, numpy=False)
        euclidean = bm.squ(euclidean, numpy=False)
        euclidean = -euclidean
        euclidean = bm.mul(self.options['kernal_param'], euclidean, numpy=False)
        return bm.exp(euclidean)
    '''
    def kernal(self, X1, X2, kernal_param):
        X1X2 = np.dot(X1, X2.T)
        X1X1 = np.matlib.repmat(np.sum(X1 ** 2, axis=1), X2.shape[0], 1).T
        X2X2 = np.matlib.repmat(np.sum(X2 ** 2, axis=1), X1.shape[0], 1)
        euclidean = np.nan_to_num(np.sqrt(X1X1 + X2X2 - 2 * X1X2))
        return np.exp(-euclidean / (2 * np.square(kernal_param)))
    '''

    def fit(self, X_i, X_u, Y):
        self.X = np.concatenate((X_i, X_u))
        self.Y = np.diag(Y)
        l = X_i.shape[0]
        u = X_u.shape[0]

        def adjacency_graph(X, k=5):
            xx = bm.dot(X, X.T)
            xb = cp.tile(cp.sum(cp.asarray(X) ** 2, axis=1), (X.shape[0], 1))
            # xb = cp.asnumpy(xb)
            # xb = cp.asnumpy(xb)
            # xb = np.matlib.repmat(np.sum(X ** 2, axis=1), X.shape[0], 1)
            euclidean = bm.add(xb.T, xb, numpy=False)
            del xb
            euclidean = cp.asnumpy(euclidean)
            xx = bm.mul(2, xx)
            euclidean = bm.sub(euclidean, xx, numpy=False)
            # euclidean = cp.sqrt(xb.T + xb - 2 * xx)
            del xx
            euclidean = cp.asnumpy(euclidean)
            euclidean = bm.sqrt(euclidean)
            euclidean = cp.nan_to_num(cp.array(euclidean))
            # graph_sort_index = cp.asnumpy(cp.array(euclidean).argsort()[:, 1:k+1])
            graph_sort_index = bm.argsort(euclidean, k)
            # graph_sort_index = np.argpartition(euclidean, -k, axis=1)[:, -k:]
            # Graph = np.zeros_like(euclidean)
            row = cp.array(range(X.shape[0])).repeat(k)
            col = cp.array(graph_sort_index).flatten()
            data = euclidean[row, col]
            # for i in range(X.shape[0]):
            #     for j in range(k):
            #         row = cp.append(row, i)
            #         col = cp.append(col, graph_sort_index[i, j])
            #         data = cp.append(data, euclidean[i, graph_sort_index[i, j]])
            del graph_sort_index
            del euclidean
            graph = cp.zeros((X.shape[0], X.shape[0]))
            graph[row, col] = data
            # Graph[i, graph_sort_index[i]] = euclidean[i, graph_sort_index[i]]
            return cp.asnumpy(graph)

        def laplacian(X, k):
            W = adjacency_graph(X, int(k))
            # W = kneighbors_graph(X, k, mode='distance',include_self=False)
            W = bm.add(W, W.T, numpy=False)
            W = cp.asnumpy(W)
            W = bm.div(W, 2, numpy=False)
            W = cp.asnumpy(W)
            W = bm.squ(W, numpy=False)
            W = -W
            W = cp.asnumpy(W)
            W = bm.div(W, 4, numpy=False)
            W = cp.asnumpy(W)
            W = bm.div(W, self.options['t'], numpy=False)
            # W = cp.asnumpy(W)
            # W = bm.exp(W, numpy=False)
            W = bm.exp(W, numpy=False, sparse=True)
            W = cp.asnumpy(W)
            # W = sparse.csr_matrix((cp.exp(-W.data ** 2 / 4 / self.options['t']), W.indices, W.indptr), shape=(self.X.shape[0], self.X.shape[0]))
            # D = np.sum(W, axis=1)
            # non_zero_index = np.nonzero(D)
            # D[non_zero_index] = np.sqrt(1 / D[non_zero_index])
            # D = np.diag(D)
            # W = np.dot(W, D)
            # W = np.dot(D, W)
            # D = cp.array(W.sum(0))[0]
            # D = sparse.diags(D).tocsr()
            D = cp.diag(cp.array(W.sum(0)))
            return bm.sub(D, W, numpy=False)#D - W#np.eye(W.shape[0]) - W  # L(normalization)=I-D^-1/2*W*D-1/2


        #K = LapSVM.kernal(self, self.X, self.X, self.options['kernal_param'])
        K = self.rbf(self.X, self.X)#numpy
        L = laplacian(self.X, self.options['k_neighbor'])
        L = cp.asnumpy(L)
        '''
        W = kneighbors_graph(self.X, self.options['k_neighbor'], mode='distance', include_self=False)
        W = W.maximum(W.T)
        W = sparse.csr_matrix((np.exp(-W.data ** 2 / 4 / self.options['t']), W.indices, W.indptr),shape=(self.X.shape[0], self.X.shape[0]))
        L = sparse.diags(np.array(W.sum(0))[0]).tocsr()-W
        '''
        # not_inv = bm.dot(L, K)
        # del L
        # not_inv = bm.mul(2 * self.options['gamma_I'] / ((l + u) ** 2), not_inv)
        # I = cp.array([2 * self.options['gamma_A']]).repeat(l+u)
        # I = cp.asnumpy(cp.diag(I))
        # not_inv = bm.add(I, not_inv, numpy=False)
        # del I
        # # not_inv = 2 * self.options['gamma_A'] * cp.eye(l + u) + 2 * self.options['gamma_I'] / ((l + u) ** 2) * bm.dot(L, K)
        # not_inv = cp.asnumpy(not_inv)
        #
        # inv = sp.linalg.inv(not_inv)
        # del not_inv
        not_inv = bm.dot(L, K, numpy=False)
        del L
        # not_inv = cp.asnumpy(not_inv)
        not_inv = bm.mul(2 * 10000 / ((l + u) ** 2), not_inv)
        I = cp.array([2 * 0.03125]).repeat(l + u)
        I = cp.asnumpy(cp.diag(I))
        not_inv = bm.add(I, not_inv, numpy=False)
        del I
        # not_inv = 2 * self.options['gamma_A'] * cp.eye(l + u) + 2 * self.options['gamma_I'] / ((l + u) ** 2) * bm.dot(L, K)
        not_inv = cp.nan_to_num(not_inv)
        lu, piv = lu_factor(not_inv)
        inv_shape = not_inv.shape[0]
        del not_inv
        inv = lu_solve((lu, piv), cp.eye(inv_shape))
        del lu
        del piv
        del inv_shape
        inv = cp.asnumpy(inv)
        J = cp.asnumpy(cp.concatenate((cp.eye(l), cp.zeros((l, u))), axis=1))

        '''try:
            np.linalg.inv(not_inv)
        except:
            print(not_inv)'''


        almost_alpha = bm.dot(inv, J.T)
        del inv
        almost_alpha = bm.dot(almost_alpha, self.Y)
        almost_alpha = cp.array(almost_alpha)
        Q = bm.dot(self.Y, J)
        del J
        Q = bm.dot(Q, K)
        del K
        Q = bm.dot(Q, almost_alpha)
        # Q = cp.asnumpy(self.Y.dot(J).dot(K).dot(almost_alpha))
        Q = bm.add(Q, Q.T)
        Q = bm.div(Q, 2)  # 由于计算机保留小数问题，导致原本Q不为对称矩阵
        q = -np.ones(l)

        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)

        def objective_grad(beta):
            return np.squeeze(np.array(beta.dot(Q) + q))

        def constraint_func(beta):
            return beta.dot(Y)

        def constraint_grad(beta):
            return Y

        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}
        beta_type = np.ones(l)
        bounds = [(0, 1 / l) for _ in range(l)]
        beta = minimize(objective_func, beta_type, jac=objective_grad, constraints=cons, bounds=bounds)['x']
        del Q
        del q
        beta = cp.array(beta)
        self.alpha = cp.asnumpy(almost_alpha.dot(beta.reshape((-1, 1))).T)
        del almost_alpha
        del beta
        del bounds
        del beta_type
        del cons

    def save_model(self):
        # with pd.ExcelWriter('LapSVM_setting.xlsx', mode='a') as writer:
        #     pd.DataFrame(self.alpha).to_excel(writer, sheet_name='alpha')
        #     pd.DataFrame(self.X).to_excel(writer, sheet_name='X')
        pd.DataFrame(self.alpha).to_csv('./LapSVM_alpha.csv')
        pd.DataFrame(self.X).to_csv('./LapSVM_x.csv')

    def clear(self):
        del self.alpha
        del self.X
        del self.Y

    def prediction(self, X):
        #new_K = LapSVM.kernal(self, self.X, X, self.options['kernal_param'])
        new_K = self.rbf(self.X, X)
        f = cp.array(self.alpha).dot(cp.array(new_K))
        del new_K
        f = cp.squeeze(f)
        return cp.asnumpy(f)

    def predict_proba(self, X):
        f = self.prediction(X)
        proba = (f.reshape((-1, 1)) + 1) / 2
        return np.concatenate((1 - proba, proba), axis=1)



'''
X_i = np.array([[5,3,2],[3,2,1],[6,2,9]])
X_u = np.array([[2,3,7]])
Y = np.array([1,-1,1])
'''
if __name__ == '__main__':
    from sklearn.datasets import make_moons
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    np.random.seed(5)

    X, Y = make_moons(n_samples=200, noise=0.05)
    ind_0 = np.nonzero(Y == 0)[0]
    ind_1 = np.nonzero(Y == 1)[0]
    Y[ind_0] = -1

    ind_l0 = np.random.choice(ind_0, 1, False)
    ind_u0 = np.setdiff1d(ind_0, ind_l0)
    ind_l1 = np.random.choice(ind_1, 1, False)
    ind_u1 = np.setdiff1d(ind_1, ind_l1)

    Xl = np.vstack([X[ind_l0, :], X[ind_l1, :]])
    Yl = np.hstack([Y[ind_l0], Y[ind_l1]])
    Xu = np.vstack([X[ind_u0, :], X[ind_u1, :]])

    plt.subplot(1, 2, 1)
    plt.scatter(Xl[:, 0], Xl[:, 1], marker='+', c=Yl)
    plt.scatter(Xu[:, 0], Xu[:, 1], marker='.')

    '''options = {'gamma_A': 0.03125,
               'gamma_I': 10000,
               'k_neighbor': 5,
               'kernal_param': 10,
               't': 1}'''
    options = {'gamma_A': 838.69,
               'gamma_I': 165.87,
               'k_neighbor': 5,
               'kernal_param': 582.58,
               't': 1}
    machine = LapSVM(options)
    machine.fit(Xl, Xu, Yl)
    Y_ = machine.prediction(X)
    Y_pre = np.ones(X.shape[0])
    Y_pre[Y_ < 0] = -1

    print(np.nonzero(Y_pre == Y)[0].shape[0] / X.shape[0] * 100.)

    xv, yv = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    XX = machine.prediction(np.c_[xv.ravel(), yv.ravel()]).reshape(xv.shape)
    plt.contour(xv, yv, XX, [0])

    plt.show()
