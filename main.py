# import lapsvm_cpu as lapsvm
import lapsvm
import pso
from sklearn.datasets import make_moons
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import shap
import pandas as pd
import os
import shutil
import time
from split_data import split_train_test
from sklearn.metrics import precision_score, recall_score
from scipy import stats
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score

def param_pso(kfold, Xl, Xu, Yl):
    param_opt = pso.pso(k=kfold)
    param = param_opt.fit(Xl, Xu, Yl)
    options = {'gamma_A': param[0],
               'gamma_I': param[1],
               'k_neighbor': 5,
               'kernal_param': param[2],
               't': 1}
    print(options)
    return options

if __name__ == '__main__':
    dtype = 'float64'
    model_establish = 0
    split_data = 0

    # half_label_num = 2
    # np.random.seed(5)
    #
    # X, Y = make_moons(n_samples=200, noise=0.05)
    # ind_0 = np.nonzero(Y == 0)[0]
    # ind_1 = np.nonzero(Y == 1)[0]
    # Y[ind_0] = -1
    #
    # ind_l0 = np.random.choice(ind_0, half_label_num, False)
    # ind_u0 = np.setdiff1d(ind_0, ind_l0)
    # ind_l1 = np.random.choice(ind_1, half_label_num, False)
    # ind_u1 = np.setdiff1d(ind_1, ind_l1)
    #
    # Xl = np.vstack([X[ind_l0, :], X[ind_l1, :]])
    # Yl = np.hstack([Y[ind_l0], Y[ind_l1]])
    # Xu = np.vstack([X[ind_u0, :], X[ind_u1, :]])
    if split_data:
        lgt_disaster = pd.read_csv('./final/disaster.csv', index_col=['lat', 'lon', 'time'])
        negtive_data = pd.read_csv('./final/negtive_data.csv', index_col=['lat', 'lon', 'time'])
        negtive_data['disaster'] = -1
        label_data = pd.concat([lgt_disaster, negtive_data])
        del lgt_disaster
        del negtive_data
        train_data, test_data = split_train_test(label_data, 0.3)
        train_data.to_csv('./final/train_data.csv')
        test_data.to_csv('./final/test_data.csv')
    else:
        train_data = pd.read_csv('./final/train_data.csv', index_col=['lat', 'lon', 'time'])
        test_data = pd.read_csv('./final/test_data.csv', index_col=['lat', 'lon', 'time'])

    Xu = pd.read_csv('./final/unlabel_data1.csv', index_col=['lat', 'lon', 'time'])
    # Xu = Xu.to_numpy(dtype=dtype)
    Xl = train_data.drop('disaster', axis=1)#.to_numpy(dtype=dtype)
    Yl = train_data['disaster'].to_numpy(dtype=dtype)
    X1 = Xl.copy()
    Y1 = Yl.copy()

    X = pd.concat([Xl, Xu])
    X_max = X.max()
    X_min = X.min()
    Xl = (Xl - X_min) / (X_max - X_min)
    Xu = (Xu - X_min) / (X_max - X_min)
    print('X_max:', X_max, '\nX_min:', X_min)
    X_max.to_csv('./final/X_max.csv')
    X_min.to_csv('./final/X_min.csv')
    Xl = Xl.to_numpy()
    Xu = Xu.to_numpy()
    del train_data

    '''options = {'gamma_A': 0.03125,
               'gamma_I': 10000,
               'k_neighbor': 5,
               'kernal_param': 10,
               't': 1}'''
    if model_establish:
        # if half_label_num < 10:
        #     kfold = 2
        # else:
        kfold = 5
        options = param_pso(kfold, Xl, Xu, Yl)
        pd.Series(options).to_excel('LapSVM_setting.xlsx', sheet_name='options')

        machine = lapsvm.LapSVM(options)
        machine.fit(Xl, Xu, Yl)
        machine.save_model()
    else:
        try:
            pd.read_excel('LapSVM_setting.xlsx', sheet_name='options')
        except:
            options = None
        else:
            options = pd.read_excel('LapSVM_setting.xlsx', sheet_name='options', index_col=0).to_dict()[0]

        alpha = pd.read_csv('./LapSVM_alpha.csv', index_col=0).to_numpy().squeeze()
        model_X = pd.read_csv('./LapSVM_x.csv', index_col=0).to_numpy().squeeze()
        machine = lapsvm.LapSVM(options, alpha, model_X)
        del alpha
        del model_X

    Y_ = machine.prediction(Xl)
    pd.DataFrame(Y_).to_csv('LapSVM_prediction.csv')
    Y_pre = np.ones(Xl.shape[0])
    Y_pre[Y_ < 0] = -1

    # print(np.nonzero(Y_pre == Yl)[0].shape[0] / Xl.shape[0] * 100.)
    print('train accuracy score:', precision_score(Yl, Y_pre))
    print('train recall score:', recall_score(Yl, Y_pre))
    # Xt = test_data.to_numpy()
    Xt = test_data.drop('disaster', axis=1)
    Yt = test_data['disaster'].to_numpy()
    X1 = pd.concat([X1, Xt])
    Y1 = np.append(Y1, Yt)
    Xt = (Xt - X_min) / (X_max - X_min)
    Xt = Xt.to_numpy(dtype=dtype)
    Yp = machine.prediction(Xt)
    pd.DataFrame(Yp).to_csv('LapSVM_prediction_Yp.csv')
    Ytpre = np.ones(Xt.shape[0])
    Ytpre[Yp < 0] = -1
    # print(np.nonzero(Ytpre == Yt)[0].shape[0] / Xt.shape[0] * 100.)
    print('test accuracy score:', precision_score(Yt, Ytpre))
    print('test recall score:', recall_score(Yt, Ytpre))
    print('AUC:', roc_auc_score(Yl, Y_pre))

    X1 = (X1 - X_min) / (X_max - X_min)
    X1 = X1.to_numpy()
    Y1p = machine.prediction(X1)
    Y1pre = np.ones(X1.shape[0])
    Y1pre[Y1p < 0] = -1
    res = stats.pearsonr(Y1p, Y1)

    X = np.concatenate((Xl, Xu))


    explainer = shap.KernelExplainer(machine.predict_proba, X)
    shap_values = explainer.shap_values(Xl[0, :])
    pd.DataFrame(shap_values[0]).to_excel('LapSVM_results.xlsx', sheet_name='shape_values0')
    with pd.ExcelWriter('LapSVM_results.xlsx', mode='a') as writer:
        pd.DataFrame(shap_values[1]).to_excel(writer, sheet_name='shape_values1')
    # class0 = np.mean(abs(shap_values[0]), axis=0)
    # class1 = np.mean(abs(shap_values[1]), axis=0)
    # plt_x = ['feature1', 'feature2']
    # fig1 = plt.figure('1')
    # plt.barh(plt_x, class0, label='class0')
    # plt.barh(plt_x, class1, left=class0, label='class1')
    # plt.xlabel('mean(|SHAP value|)')
    # plt.ylabel('features')
    # plt.legend()
    # plt.title('mean(|SHAP value|) in features')
    # fig1.savefig('SHAP')

    #shap.summary_plot(shap_values, X)

    # plt.subplot(1, 1, 1)
    # fig2 = plt.figure('2')
    # plt.scatter(Xl[:, 0],  Xl[:, 1], marker='+', c=Yl)
    # plt.scatter(Xu[:, 0], Xu[:, 1], marker='.')
    # xv, yv = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100), np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
    # XX = machine.prediction(np.c_[xv.ravel(), yv.ravel()]).reshape(xv.shape)
    # plt.contour(xv, yv, XX, [0])

    #plt.show()

    # fig2.savefig('LapSVM')

    time_now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    file_list = ['LapSVM_setting.xlsx', 'LapSVM_results.xlsx', 'SHAP.png', 'LapSVM.png', 'LapSVM_x.csv', 'LapSVM_alpha.csv', 'X_max.csv', 'X_min.csv']
    if not os.path.exists('save'):
        os.makedirs('save')
    path = '.\save\{}'.format(time_now)
    os.makedirs(path)
    for file_name in file_list:
        if not os.path.isfile(file_name):
            print('%s not exit!'%(file_name))
        else:
            file_out = os.path.join(path, file_name)
            shutil.copy(file_name, file_out)

