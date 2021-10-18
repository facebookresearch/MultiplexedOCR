import numpy as np
from PIL import ImageDraw

from multiplexer.utils.chars import char2num


def creat_color_map(n_class, width):
    splits = int(np.ceil(np.power((n_class * 1.0), 1.0 / 3)))
    maps = []
    for i in range(splits):
        r = int(i * width * 1.0 / (splits - 1))
        for j in range(splits):
            g = int(j * width * 1.0 / (splits - 1))
            for k in range(splits - 1):
                b = int(k * width * 1.0 / (splits - 1))
                maps.append((r, g, b, 200))
    return maps


def render_char_mask(image, polygons, resize_ratio, colors, char_polygons=None, words=None):
    draw = ImageDraw.Draw(image, "RGBA")
    for polygon in polygons:
        draw.polygon(polygon, fill=None, outline=(0, 255, 0, 255))
    if char_polygons is not None:
        for i, char_polygon in enumerate(char_polygons):
            for j, polygon in enumerate(char_polygon):
                polygon = [int(x * resize_ratio) for x in polygon]
                char = words[i][j]
                color = colors[char2num(char)]
                draw.polygon(polygon, fill=color, outline=color)
