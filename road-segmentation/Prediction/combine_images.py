import os
import rasterio
import numpy as np
from typing import Dict
import geopandas as gpd


def num_neighbours(neighbour_ids, _files):
    calc = 0
    for _nid in neighbour_ids:
        if _nid + '.tif' in _files:
            calc += 1
    return calc


def read_image(path, channels: list[int]):
    with rasterio.open(path) as src:
        img = src.read(channels)
        transform = src.transform
        # print(transform)
        pos = rasterio.transform.xy(transform, 0, 0)
        meta = src.meta
        # print(meta)
    return img, pos, meta


def generate_key(position, input_position, image, pad_len: int = 1000):
    if position[0] < input_position[0]:
        # left
        key_x = "-"
        if image.ndim == 2:
            pad_img = image[:, -pad_len:]
        else:
            pad_img = image[:, :, -pad_len:]

    elif position[0] == input_position[0]:
        # center
        key_x = "0"
        pad_img = image

    else:
        # right
        key_x = "+"
        if image.ndim == 2:
            pad_img = image[:, :pad_len]
        else:
            pad_img = image[:, :, :pad_len]
    
    if position[1] > input_position[1]:
        # bot
        key_y = "+"
        if image.ndim == 2:
            pad_img = pad_img[-pad_len:, :]
        else:
            pad_img = pad_img[:, -pad_len:, :]
            
    elif position[1] == input_position[1]:
        # center
        key_y = "0"
        pad_img = pad_img
        
    else:
        # top
        key_y = "-"
        if image.ndim == 2:
            pad_img = pad_img[:pad_len, :]
        else:
            pad_img = pad_img[:, :pad_len, :]
            
    key = key_x + key_y
    return key, pad_img


def combine_imgs(input_image, positions: Dict, pad_len: int = 1000):
    for key, val in positions.items():
        if key == "-+":
            if input_image.ndim == 2:
                input_image[:pad_len, :pad_len] = val
            else:
                input_image[:, :pad_len, :pad_len] = val
        
        elif key == "0+":
            if input_image.ndim == 2:
                input_image[:pad_len, pad_len:-pad_len] = val
            else:
                input_image[:, :pad_len, pad_len:-pad_len] = val
        
        elif key == "++":
            if input_image.ndim == 2:
                input_image[:pad_len, -pad_len:] = val
            else:
                input_image[:, :pad_len, -pad_len:] = val
        
        elif key == "-0":
            if input_image.ndim == 2:
                input_image[pad_len:-pad_len, :pad_len] = val
            else:
                input_image[:, pad_len:-pad_len, :pad_len] = val
        
        elif key == "+0":
            if input_image.ndim == 2:
                input_image[pad_len:-pad_len, -pad_len:] = val
            else:
                input_image[:, pad_len:-pad_len, -pad_len:] = val
                
        elif key == "--":
            if input_image.ndim == 2:
                input_image[-pad_len:, :pad_len] = val
            else:
                input_image[:, -pad_len:, :pad_len] = val

        elif key == "0-":
            if input_image.ndim == 2:
                input_image[-pad_len:, pad_len:-pad_len] = val
            else:
                input_image[:, -pad_len:, pad_len:-pad_len] = val

        elif key == "+-":
            if input_image.ndim == 2:
                input_image[-pad_len:, -pad_len:] = val
            else:
                input_image[:, -pad_len:, -pad_len:] = val

    return input_image


def get_neighbour_ids(file, gpkg=None):
    if gpkg is None:
        gpkg_file = "TM35_karttalehtijako.gpkg"
        gpkg = gpd.read_file(gpkg_file, layer="utm5")
    target_area = gpkg[gpkg['lehtitunnus'] == file.split(".")[0]]
    neighbors = gpkg[gpkg.geometry.touches(target_area.geometry.iloc[0])]
    neighbor_ids = neighbors['lehtitunnus'].tolist()
    return neighbor_ids


def pad_image_with_neighbours(img_path: str, file: str, pad_size: int = 1000, channels: int | list[int] = None, gpkg=None):
    if channels is None:
        channels = 1
    gpkg_file = "TM35_karttalehtijako.gpkg"
    #img_path = "C:/Users/Jere/Road_Data_2025/slope/"
    files = os.listdir(img_path)
    # Read the GeoPackage file into a GeoDataFrame
    if gpkg is None:
        gpkg = gpd.read_file(gpkg_file, layer="utm5")
    
    if type(channels) == list:
        pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    else:
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))

    # for file in files:
    images = {}
    # get neighbouring files

    target_area = gpkg[gpkg['lehtitunnus'] == file.split(".")[0]]
    neighbors = gpkg[gpkg.geometry.touches(target_area.geometry.iloc[0])]
    neighbor_ids = neighbors['lehtitunnus'].tolist()
    # edges contain only partial data and are not needed so those are ignored
    if num_neighbours(neighbor_ids, files) < 8:
        print("--Error: Not enough neighbours")
        return
    input_img, input_pos, input_meta = read_image(os.path.join(img_path, file), channels)
    print(pad_width)
    input_img_padded = np.pad(input_img, pad_width=pad_width, mode='constant', constant_values=0).astype(np.float32)
    print(np.shape(input_img_padded))
    for nid in neighbor_ids:
        if nid + ".tif" not in files:
            print(f"--FILE NOT FOUND: {nid}")
            continue
        img, pos, meta = read_image(os.path.join(img_path, f"{nid}.tif"), channels)
        images[nid] = {
            "img": img,
            "pos": pos,
        }
    positions = {
        "-+": None,  # top left
        "0+": None,  # top center
        "++": None,  # top right
        "-0": None,  # center left
        "+0": None,  # center right
        "--": None,  # bot left
        "0-": None,  # bot center
        "+-": None,  # bot right
    }
    for nid, data in images.items():
        print(nid)
        p = data["pos"]
        key, im = generate_key(p, input_pos, data["img"])
        positions[key] = im
        print(nid, data)
    input_img_padded = combine_imgs(input_img_padded, positions)
    return input_img_padded
