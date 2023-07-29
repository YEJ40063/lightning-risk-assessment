# import xarray as xr
import numpy as np
#import tensorflow
import pandas as pd

# lgt_path="D:/shu/Lightning_Risk_Zoning/datas/2021_lightning.xlsx"
# other_path="D:/shu/Lightning_Risk_Zoning/datas/datas.nc"

# def data_grid(lgt_datas, lgt_lat_label, lgt_lon_label):
#     lgt_datas['Datetime'] = pd.to_datetime(lgt_datas['Datetime'], format="%Y-%m-%d %H").dt.floor("H")
#     lgt_datas = lgt_datas.drop(['ID'], axis=1)
#     lgt_lat_bins = label_data(lgt_lat_label)
#     lgt_lon_bins = label_data(lgt_lon_label)
#     lgt_datas.Lat = pd.cut(lgt_datas.Lat, lgt_lat_bins, labels=lgt_lat_label)
#     lgt_datas.Lon = pd.cut(lgt_datas.Lon, lgt_lon_bins, labels=lgt_lon_label)
#     lgt_datas = lgt_datas.reset_index().set_index(['Lon', 'Lat', 'Datetime'])
#     lgt_datas = lgt_datas.rename_axis(index=['longitude', 'latitude', 'time']).sort_index()
#     return lgt_datas
#
# def label_data(labels, half_sat_accuracy=0.125):
#     bins=[]
#     for lab in labels:
#         bins.append(lab-half_sat_accuracy)
#     bins.append(lab+half_sat_accuracy)
#     return bins
#
# def combine_datas(lgt_path, other_path):
#     datas = xr.open_dataset(other_path)
#     lgt_lat_label = np.sort(datas.coords['latitude'].to_numpy())
#     lgt_lon_label = np.sort(datas.coords['longitude'].to_numpy())
#
#     lgt_datas = pd.read_excel(lgt_path)
#     lgt_datas = data_grid(lgt_datas, lgt_lat_label, lgt_lon_label)
#     datas = datas.to_dataframe()
#
#     datas['lgt_mean'] = 0
#     datas['lgt_max'] = 0
#     datas['lgt_count'] = 0
#     for (lon, lat, time) in lgt_datas.index:
#         datas.loc[(lon, lat, time), 'lgt_mean'] = lgt_datas.loc[(lon, lat, time), 'PEAK_KA'].mean()
#         datas.loc[(lon, lat, time), 'lgt_max'] = lgt_datas.loc[(lon, lat, time), 'PEAK_KA'].max()
#         datas.loc[(lon, lat, time), 'lgt_count'] = lgt_datas.loc[(lon, lat, time), 'PEAK_KA'].count()
#     return datas

def split_train_test(data, test_ratio, valid_ratio=0):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    valid_set_size = int(len(data) * valid_ratio)
    valid_indices = shuffled_indices[:valid_set_size]
    test_indices = shuffled_indices[valid_set_size:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    if valid_ratio:
        return (data.iloc[train_indices], data.iloc[test_indices], data.iloc[valid_indices])
    else:
        return (data.iloc[train_indices], data.iloc[test_indices])

# from sklearn.datasets import load_breast_cancer
# from math import sqrt
#
# cancer_data = load_breast_cancer()
# x_cancer = cancer_data.data