import numpy as np
import pandas as pd
from pyproj import Transformer


def projection(x: float, y: float):
    """
    Проекция координат с картинки в нормальные координаты (ширина и долгота)
    input
    x, y - координаты пиксиля картинки в метрах
    return
    Словарь {longitude - долгота, latitude - ширина}
    """
    transproj = Transformer.from_crs(
        'EPSG:32637',
        "EPSG:4326",
        always_xy=True,
    )

    lat, lon, _ = transproj.transform(
        x,
        y,
        0,
        radians=False,
    )

    return {'longitude': lon,
            'latitude': lat}


def pixel_to_meter(proint_pixel, size_m=100, standart_xy=(310050, 4800050)):
    """
    Переводит пиксели в ебанутые координаты (берется нижний левый угол как основа)
    input
    proint_pixel - позиция пиксиля,
    size_m - размер на пиксель
    standart_xy - начальное положение в ебанутых координатах
    return
    Словарь { x - координата x, y - координата y}
    """
    res_data = [standart_xy[0] + proint_pixel[0] * size_m, standart_xy[1] + proint_pixel[0] * size_m]
    return {
            "x": res_data[0],
            "y": res_data[1]
            }

coor_d1 = pixel_to_meter((2000, 2010))
coor_d2 = pixel_to_meter((2100, 2364))
coor_d3 = pixel_to_meter((2200, 2198))
coor_d4 = pixel_to_meter((2300, 2147))

print(projection(coor_d1['x'], coor_d1['y']))
print(projection(coor_d2['x'], coor_d2['y']))
print(projection(coor_d3['x'], coor_d3['y']))
print(projection(coor_d4['x'], coor_d4['y']))