import pandas as pd
from tqdm import tqdm
import numpy as np
from preprogress_data import drop_data

def cal_distance(data, k, center):
    distance = pd.DataFrame()
    for i in range(k):
        dis_part = (((data - center.iloc[i]) ** 2).sum(axis=1) / data.shape[1]).apply(np.sqrt)
        dis_part = dis_part.to_frame().rename(columns={0: i})
        distance = pd.concat([distance, dis_part], axis=1)
    return distance
def cal_center(distance, unlabel_data, label_data, k):
    index_min = distance.idxmin(axis=1)
    center = pd.concat([label_data, unlabel_data[index_min == 0]]).mean().to_frame().T
    for i in range(k-1):
        center = pd.concat([center, unlabel_data[index_min == i+1].mean().to_frame().T])
    return center


def half_kmeans(label_data, unlabel_data, k, max_iters=100, random_state=519):
    center = pd.DataFrame(columns=label_data.columns)
    center.loc['0'] = label_data.mean()
    center = pd.concat([center, unlabel_data.sample(n=k, axis=0, random_state=random_state)])
    distance = pd.DataFrame()
    # new_center = center.copy()
    unlabel_data.sort_index(inplace=True)
    for i in tqdm(range(max_iters)):
        distance = cal_distance(unlabel_data, k, center)
        # distance['center_n'] = (((unlabel_data - center.iloc[1]) ** 2).sum(axis=1) / unlabel_data.shape[1]).apply(np.sqrt)
        # distance['center_p'] = (((unlabel_data - center.iloc[0]) ** 2).sum(axis=1) / unlabel_data.shape[1]).apply(np.sqrt)
        distance = distance.sort_index()
        # center_n_data = unlabel_data[distance['center_n'] < distance['center_p']]
        # center_p_data = unlabel_data[distance['center_n'] >= distance['center_p']]
        new_center = cal_center(distance, unlabel_data, label_data, k)
        # center_p_data = pd.concat([label_data, center_p_data])
        # new_center.iloc[0] = center_p_data.mean()
        # new_center.iloc[1] = center_n_data.mean()
        if center.equals(new_center):
            break
        center = new_center.copy()
    # center.reset_index(inplace=True)
    # distance.reset_index(inplace=True)
    return (center, distance)
# data = pd.read_csv('./final/combined_datas.csv')
# data.drop(['t2m', 'AOD550', 'p140208', 'sp', 'ssrd', 'tp'], axis=1, inplace=True)
# data.drop(['ctt', 'cth', 'ctp'], axis=1, inplace=True)
# data2 = data[data['lgt_num']!=0]
# data3 = data[data['wt_num']!=0]
# data3 =data3[data3['lgt_num']==0]
# data = pd.concat([data2, data3])
# data = data.dropna()
# data.drop_duplicates(subset=['lat', 'lon', 'time'], inplace=True)
# data1 = data[data['lgt_num']!=0]
# data1 = data1[data1['wt_num']!=0]
# data1.set_index(['lat', 'lon', 'time'], inplace=True)
# data1.to_csv('./final/lgt_wt_notzero.csv')
if __name__ == '__main__':
    label_data = pd.read_csv('./final/disaster.csv', index_col=['lat', 'lon', 'time'])
    unlabel_data = pd.read_csv('./final/less_unlabel_data.csv', index_col=['lat', 'lon', 'time'])
    label_data.drop('disaster', axis=1, inplace=True)
    # unlabel_data.drop('AOD550', axis=1, inplace=True)
    # unlabel_data.dropna(inplace=True)
    #
    # label_data_d = label_data.drop('disaster', axis=1)
    # datas = pd.concat([unlabel_data, label_data_d])
    # unlabel_data = datas.drop_duplicates(keep=False)
    # del datas
    # del label_data_d
    # unlabel_data.to_csv('./final/unlabel_data.csv')
    center, distance = half_kmeans(label_data, unlabel_data, label_data.shape[0]+1, max_iters=1000)
    center.to_csv('./final/km_center.csv')
    distance.to_csv('./final/km_distance.csv')
    col_min = distance.idxmin(axis=0)[1:]
    negtive_data = unlabel_data.loc[col_min]
    negtive_data.to_csv('./final/negtive_data.csv')
    # data_num = label_data.shape[0]
    # unlabel_data_dis = pd.merge(unlabel_data, distance, left_index=True, right_index=True, how='outer')
    # negtive_data = unlabel_data_dis.nsmallest(n=data_num, columns='center_n')
    # negtive_data.drop(['center_p', 'center_n'], axis=1, inplace=True)
    # negtive_data.to_csv('./final/negtive_data.csv')
    # data1 = pd.concat([unlabel_data, negtive_data])
    # data1.reset_index(inplace=True)
    # data1.drop_duplicates(subset=['lat', 'lon', 'time'], inplace=True, keep=False)
    unlabel_data.drop(col_min, inplace=True)
    unlabel_data.to_csv('./final/unlabel_data1.csv')
    # center_p_data.to_csv('./final/center_p_data.csv')
    # center_n_data.to_csv('./final/center_n_data.csv')



