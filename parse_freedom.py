import pandas as pd
import numpy as np
import re
import sklearn
from sklearn import metrics
import json


work_f = pd.read_csv('data/not_freadom.csv')
data_f = pd.read_csv('data/freadom.csv')

column_arr = work_f.columns
work_f = work_f.where(work_f > -10000, other=np.nan)
work_f = work_f.where(work_f < 10000, other=np.nan)
val_col = [col for col in work_f if re.search('_value_', str(col))]

tmp_vec = work_f[val_col].mean().values.reshape(1, -1)
data_arr = data_f[val_col]

score_vec = []
print((data_arr.shape)[0])
for i in range(0, (data_arr.shape)[0]):
    v_t = data_arr.iloc[i].values.reshape(1, -1)
    a = metrics.pairwise.cosine_similarity(v_t, tmp_vec)
    score_vec.append(metrics.pairwise.cosine_similarity(v_t, tmp_vec)[0][0])
data_f['score'] = score_vec
print(data_f.columns)
data_f[['index_cluster', 'score']].to_csv('score_data.csv')

print(score_vec)
