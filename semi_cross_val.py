import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
# import lapsvm_cpu as lapsvm
import lapsvm
import tracemalloc
import blocked_matrix as bm
import sys
import gc
from tqdm import tqdm
def cal_auc(x_i, x_u, y, train_index, test_index, options):
    model = lapsvm.LapSVM(options)
    x_train, y_train = x_i[train_index], y[train_index]
    x_test, y_test = x_i[test_index], y[test_index]
    model.fit(x_train, x_u, y_train)
    y_ = model.prediction(x_test)
    del x_test
    y_pred = np.ones(y_.shape[0])
    y_pred[y_ < 0] = -1
    del y_
    return roc_auc_score(y_test, y_pred)
def semi_cross_valid(x_i, x_u, y, options, k=5):
    cross_valid = StratifiedKFold(n_splits=k)
    roc_auc = []
    for train_index, test_index in cross_valid.split(x_i, y):
        # tracemalloc.start()
        # snapshot = tracemalloc.take_snapshot()
        # bm.display_top(snapshot)
        auc = cal_auc(x_i, x_u, y, train_index, test_index, options)
        # auc = accuracy_score(y_test, y_pred)
        roc_auc.append(auc)
        del auc
    return np.mean(roc_auc)

if __name__ == '__main__':
    from sklearn.datasets import make_moons

    X, Y = make_moons(n_samples=200, noise=0.05)
    ind_0 = np.nonzero(Y == 0)[0]
    ind_1 = np.nonzero(Y == 1)[0]
    Y[ind_0] = -1
    ind_l0 = np.random.choice(ind_0, 5, False)
    ind_u0 = np.setdiff1d(ind_0, ind_l0)
    ind_l1 = np.random.choice(ind_1, 5, False)
    ind_u1 = np.setdiff1d(ind_1, ind_l1)
    x_i = np.vstack([X[ind_l0, :], X[ind_l1, :]])
    y = np.hstack([Y[ind_l0], Y[ind_l1]])
    x_u = np.vstack([X[ind_u0, :], X[ind_u1, :]])
    options = {'gamma_A': 0.03125,
               'gamma_I': 10000,
               'k_neighbor': 5,
               'kernal_param': 10,
               't': 1}
    A = semi_cross_valid(x_i, x_u, y, options)