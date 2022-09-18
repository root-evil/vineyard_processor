import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler

def get_batch(df, num_batch = (1, 1), size_img = (4096, 4096)):
    size_x_batch = size_img[0]//num_batch[0]
    size_y_batch = size_img[1]//num_batch[1]
    df['batch_x'] = (df['x'] // size_x_batch).astype(str)
    df['batch_y'] = (df['y'] // size_y_batch).astype(str)
    df['batch'] = df['batch_x'] + df['batch_y']
    df = df.drop(['batch_x', 'batch_y'], axis = 1)
    return df

def get_dbscan_pred_for_batch(eps):
    df = pd.read_csv('to_save/to_cluster_data.csv')
    df = get_batch(df)
    clf = MinMaxScaler()
    pred = clf.fit_transform(df[df.columns[3:-1]])
    df[df.columns[3:-1]] = pred
    
    batch_vals = df['batch'].unique()
    batch_arr = df['batch'].values
    for v in tqdm.tqdm(batch_vals):
        dbscan = DBSCAN(eps = eps, min_samples = 1)
        df_tmp = df[df['batch'] == v][df.columns[3:-1]]
        pred = dbscan.fit_predict(df_tmp)
        batch_arr[batch_arr == v] = pred
        del df_tmp, pred, dbscan
        
    df_ = pd.DataFrame(batch_arr, columns = ['pred'])
    df_.to_csv('cluster_pred\dbscan_' + str(eps) + '_without_xy.csv', index = False)
    print('FINISHED', eps)
    
for eps in [0.001, 0.0001, 0.00001]:
    get_dbscan_pred_for_batch(eps)