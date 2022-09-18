import numpy as np
import PIL
from PIL import Image
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import multidem


admin = Image.open('data/KRA_ADMIN_100m.tif')
landcover = Image.open('data/KRA_LANDCOVER_100m.tif')
prec = io.imread('data/KRA_PREC_100m.tif')
relief_aspect = Image.open('data/KRA_RELIEF_ASPECT_100m.tif')
relief_height = Image.open('data/KRA_RELIEF_HEIGHT_100m.tif')
soiltexture = Image.open('data/KRA_SOILTEXTURE_100m.tif')
relief_slope = Image.open('data/KRA_RELIEF_SLOPE_100m.tif')
sunny_days = Image.open('data/KRA_SUNNY_DAYS_APR_OCT_100m.tif')
tavg = io.imread('data/KRA_TAVG_100m.tif')
tmin = io.imread('data/KRA_TMIN_100m.tif')
tmax = io.imread('data/KRA_TMAX_100m.tif')
vineyards = Image.open('data/KRA_VINEYARDS_100m.tif')
water_seasonality = Image.open('data/KRA_WATER_SEASONALYTY_100m.tif')

landcover_arr = np.array(landcover)
admin_arr = np.array(admin)
relief_aspect_arr = np.array(relief_aspect)
relief_height_arr = np.array(relief_height)
soiltexture_arr = np.array(soiltexture)
relief_slope_arr = np.array(relief_slope)
sunny_days_arr = np.array(sunny_days)
vineyards_arr = np.array(vineyards)
water_seasonality_arr = np.array(water_seasonality)
# [1, 4, 7, 8, 9, 10]
tmp = admin_arr &\
      (relief_height_arr > 0) &\
      ~vineyards_arr &\
      (sunny_days_arr > 115) &\
      (soiltexture_arr != 1) &\
      (soiltexture_arr != 2) &\
      (soiltexture_arr != 3) &\
      (soiltexture_arr != 6) &\
      (soiltexture_arr != 7) &\
      (soiltexture_arr != 8) &\
      (landcover_arr != 1) &\
      (landcover_arr != 2) &\
      (landcover_arr != 4) &\
      (landcover_arr != 7) &\
      (landcover_arr != 8) &\
      (landcover_arr != 9) &\
      (landcover_arr != 10) & \
      (relief_slope_arr < 45) & \
      (soiltexture_arr == 9)

point_arr = []
pd.DataFrame(tmp).to_csv('tmp_loan_mask.csv', index=False)
print(tmp.sum())


#
# for row in range(0, 4096):
#     for column in range(0, 4096):
#         if tmp[row, column]:
#             point_arr.append({
#             'admin': admin_arr[row, column],
#             'relief_aspect': relief_aspect_arr[row, column],
#             'relief_height': relief_height_arr[row, column],
#             'soiltexture': soiltexture_arr[row, column],
#             'sunny_days': sunny_days_arr[row, column],
#             'vineyards': vineyards_arr[row, column],
#             'water_seasonality': water_seasonality_arr[row, column],
#             'coordinates': (row, column)
#                             })
#
# print(len(point_arr))

# plt.imshow(tmp, cmap='Greys')
# plt.savefig('tmp1.png')
# print(point_arr)
def get_good_bad():
    img_arr = np.array(landcover)
    pd.DataFrame(img_arr).to_csv('KRA_LANDCOVER_100m.csv', index=False)
    print(img_arr.shape)
    unique, counts = np.unique(img_arr, return_counts=True)
    dk = dict(zip(unique, counts))
    bad = [1, 2, 4, 7, 8, 9, 10]
    bad_cnt = 0
    good_cnt = 0
    sum = 0
    for each in dk:
        sum += dk[each]
        if each not in bad:
            good_cnt += dk[each]
        print(each, dk[each])
    print(sum)
    print(good_cnt)
    print(sum - good_cnt)
#
#
#
# im = io.imread('KRA_PREC_100m.tif')
#
# plt.imshow(im[:,:,0], cmap='hot')
#
# plt.savefig('test.png')
#
# tmp_1 = pd.read_csv('tmp.csv')
# print(tmp_1.shape)


#
# def get_poly_border(points: list):
#     input_arr = np.array(points)
#     x_arr = input_arr[:,0]
#     y_arr = input_arr[:,1]
#     min_x = x_arr.min()
#     min_y = y_arr.min()
#     max_x = x_arr
#
#
#
# get_poly_border([[1,2], [3,4]])
#
#
#




