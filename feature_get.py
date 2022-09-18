import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans, DBSCAN
from skimage import io
import os
import tqdm
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

img = io.imread('data/KRA_VINEYARDS_WITH_NUMBERS_100m.tif')

num_data = ['KRA_RELIEF_ASPECT_100m.tif',
            'KRA_RELIEF_HEIGHT_100m.tif',
            'KRA_RELIEF_SLOPE_100m.tif',
            'KRA_SUNNY_DAYS_APR_OCT_100m.tif',
           'KRA_WATER_SEASONALYTY_100m.tif']

num_many_chanel_data = ['KRA_TAVG_100m.tif',
            'KRA_TMIN_100m.tif',
            'KRA_TMAX_100m.tif',
            'KRA_PREC_100m.tif']
path_to_save = 'to_save/'
cat_data = ['KRA_SOILTEXTURE_100m.tif']
path = 'data/'
uniq_label = np.unique(label)

#num feat
for name in num_data:
    img_tmp = io.imread(path + name).astype(np.float32)
    name_for_write = name.split('.')[0]
    res_list = []
    for cluster in tqdm.tqdm(uniq_label):
        res_dict = {}
        res_dict['index_cluster'] = cluster
        xy_tmp = xy_data[xy_data['label'] == cluster][['x', 'y']].values
        x, y = xy_tmp[:,0], xy_tmp[:,1]
        pixel_wise_value = img_tmp[x,y]
        min_val, max_val, mean_val = pixel_wise_value.min(), pixel_wise_value.max(), pixel_wise_value.mean()
        res_dict['min_value_' + name_for_write] = min_val
        res_dict['max_value_' + name_for_write] = max_val
        res_dict['mean_value_' + name_for_write] = mean_val
        res_list.append(res_dict)
    df_res = pd.DataFrame(res_list, dtype = np.float64)
    df_res.to_csv(path_to_save + name_for_write + '.csv', index = False)

for name in num_many_chanel_data:
    img_tmp_ = io.imread(path + name).astype(np.float32)
    name_for_write = name.split('.')[0]
    res_list = []
    img_tmp = img_tmp_[:,:,chanel]
    for cluster in tqdm.tqdm(uniq_label):
        res_dict = {}
        res_dict['index_cluster'] = cluster
        for chanel in range(img_tmp_.shape[2]):

            xy_tmp = xy_data[xy_data['label'] == cluster][['x', 'y']].values
            x, y = xy_tmp[:,0], xy_tmp[:,1]
            pixel_wise_value = img_tmp[x,y]
            min_val, max_val, mean_val = pixel_wise_value.min(), pixel_wise_value.max(), pixel_wise_value.mean()
            res_dict['min_value_' + name_for_write + '_chanel_' + str(chanel)] = min_val
            res_dict['max_value_' + name_for_write + '_chanel_' + str(chanel)] = max_val
            res_dict['mean_value_' + name_for_write + '_chanel_' + str(chanel)] = mean_val
        res_list.append(res_dict)
    df_res = pd.DataFrame(res_list, dtype = np.float64)
    df_res.to_csv(path_to_save + name_for_write + '.csv', index = False)
    
for name in cat_data:
    img_tmp = io.imread(path + name).astype(np.float32)
    name_for_write = name.split('.')[0]
    res_list = []
    uniq_val = np.unique(img_tmp)
    
    for cluster in tqdm.tqdm(uniq_label):
        res_dict = {}
            
        res_dict['index_cluster'] = cluster
        xy_tmp = xy_data[xy_data['label'] == cluster][['x', 'y']].values
        x, y = xy_tmp[:,0], xy_tmp[:,1]
        pixel_wise_value = img_tmp[x,y]
        c = Counter(pixel_wise_value)
        for v in uniq_val:
            res_dict['count_of_' + str(v)] = c[v]
        res_list.append(res_dict)
            
    df_res = pd.DataFrame(res_list)
    df_res.to_csv(path_to_save + name_for_write + '.csv')

list_dir = os.listdir('to_save')
all_df = pd.DataFrame(label, columns = ['index_cluster'])
for f in list_dir:
    try:
        df = pd.read_csv('to_save/' + f)
    except:
        continue
        
    all_df = all_df.merge(df, on = 'index_cluster')