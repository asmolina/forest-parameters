import wandb
import uuid
import rasterio
from rasterio.windows import Window

import numpy as np
import pandas as pd
import tifffile as tiff

from tqdm.autonotebook import tqdm
from pathlib import Path

import plotly_express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from sentinelhub import CRS

CHANNELS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']


def get_dataset_v2(path_S2B, 
                channels_list=['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'], 
                window=None,  
                visualise=False,
                indices=False, 
                normalise=False,
                apply_SCL=True):
    """
    Updated version of get_dataset(). The old version worked with old version of data
    (not filtrated data).
    """
    
    dataset = {}
    
    channels_list[0]
            
    with rasterio.open(f'{path_S2B}/SCL.tif') as src:
        if window is not None:
            SCL_map = src.read(1, window=window)
        else:
            SCL_map = src.read(1)
        # print(SCL_map.shape)

    if visualise:
        print(f'SCL Vegetation class (4):')
        tiff.imshow(np.where(SCL_map == 4.0, 1, 0))
        print(f'Scene Classification Layer  map shape is: {SCL_map.shape}')
        plt.show()

    if apply_SCL:
        coords = np.argwhere((SCL_map == 4.0))
    else:
        coords = np.argwhere((SCL_map >= 0))
        
    dataset['x'] = coords[:, 0]
    dataset['y'] = coords[:, 1]
    
    for ch in channels_list:
        with rasterio.open(f'{path_S2B}/{ch}.tif') as src:
            if window is not None:
                channel_map = src.read(1, window=window)
            else:
                channel_map = src.read(1)
            # print(f'{path_S2B}/{ch}.tif')    
            # print(channel_map.shape)
            dataset[ch] = channel_map[dataset['x'], dataset['y']]

    dataset = pd.DataFrame(dataset) 
    
    if normalise:
        dataset = normalisation(dataset, path_S2B)

    if indices:
        dataset = vegetation_indices(dataset, drop_nans=True)

    return dataset


def get_dataset(path_S2B, 
                channels_list=['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'], 
                window=None,  
                visualise=False,
                indices=False, 
                normalise=False,
                apply_SCL=True):
    """
    path_S2B: {'./'} is a path to the folder with all the files 'B02.tif', 'B03.tif', ...
    channels_list: {['B02', 'B03', ..., 'B12', 'type']} list of the channels
    """

    
    dataset = {}
    
    channels_list[0]
            
    with rasterio.open(f'{path_S2B}/SCL.tif') as src:
        if window is not None:
            SCL_map = src.read(1, window=window)
        else:
            SCL_map = src.read(1)

    if visualise:
        print(f'SCL Vegetation class (4):')
        tiff.imshow(np.where(SCL_map == 4.0, 1, 0))
        print(f'Scene Classification Layer  map shape is: {SCL_map.shape}')
        plt.show()

    if apply_SCL:
        x_coords, y_coords = np.asarray((SCL_map == 4.0)).nonzero()
    else:
        x_coords, y_coords = np.asarray((SCL_map >= 0)).nonzero()
        
    dataset['x'] = x_coords
    dataset['y'] = y_coords
    
    for ch in channels_list:
        with rasterio.open(f'{path_S2B}/{ch}.tif') as src:
            if window is not None:
                channel_map = src.read(1, window=window)
            else:
                channel_map = src.read(1)
            dataset[ch] = channel_map[dataset['x'], dataset['y']]

    dataset = pd.DataFrame(dataset) 
    
    if normalise:
        dataset = normalisation(dataset, path_S2B)

    if indices:
        dataset = vegetation_indices(dataset, drop_nans=True)

    return dataset




def vegetation_indices(ds, drop_nans=True):
    """
    ds: pd.DataFrame()
    drop_nans: True - for making train ds,
               False - for making predictions on new territories and saving to tiff 
                       (pixels will not be removed)
    """
    L = 0.428 # for SAVI (L = soil brightness correction factor could range from (0 -1))
    y = 0.106 # for ARVI

    ds['ndvi'] = (ds['B08'] - ds['B04']) / (ds['B08'] + ds['B04'])
    ds['ndwi'] = (ds['B08'] - ds['B11']) / (ds['B08'] + ds['B11'])
    ds['msr670800'] = (ds['B08'] / ds['B04'] - 1.0) / np.sqrt((ds['B08'] / ds['B04'] + 1))
    ds['evi'] = 2.5 * (ds['B08'] - ds['B04']) / ((ds['B08'] + 6.0 * ds['B04'] - 7.5 * ds['B02']) + 1.0)
    ds['savi'] = (ds['B08'] - ds['B04']) / (ds['B08'] + ds['B04'] + L) * (1.0 + L)
    ds['arvi'] = (ds['B8A'] - ds['B04'] - y * (ds['B04'] - ds['B02'])) / (ds['B8A'] + ds['B04'] - y * (ds['B04'] - ds['B02']));

    # Deal with NaNs
    if drop_nans:
        ds.replace([np.inf, -np.inf], np.nan, inplace=True)
        ds.dropna(subset=['ndvi', 'ndwi', 'msr670800', 'evi', 'savi', 'arvi'], inplace=True)
    else: 
        ds.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    return ds


                
        
        


def save_map(
    preds, 
    dataset, 
    window=None, 
    no_data_value=0, 
    clip=(0, None), 
    path_S2B='.', 
    out_path='.',
    out_name=None,
    ):
    # Obtaining meta information from B02 image
    path_S2B = Path(path_S2B)
    out_path = Path(out_path)

    if window is not None:
        with rasterio.open(path_S2B / 'B02.tif') as src:
            crop = src.read(1, window=window)
            params = src.meta.copy()
            out_transform = src.window_transform(window)
            params.update(
                {
                    "driver": 'GTiff',
                    "height": crop.shape[0],
                    "width": crop.shape[1],
                    "transform": out_transform,
                }
            )
    else:
        with rasterio.open(path_S2B / 'B02.tif') as src:
            params = src.meta.copy()
    
    # Clipping and rounding to the nearest integer
    preds = np.clip(preds, *clip)
    preds = np.rint(preds).astype(int)
    
    # 1D -> 2D :    
    res = np.ones((params['height'], params['width'])) * no_data_value
    for i, x in enumerate(dataset[['x', 'y']].values):
        res[x[0], x[1]] = preds[i]
        
    # Create or rewrite existing file with an empty one
    if not out_name:
        out_name = str(uuid.uuid4())[:8] + '.tif'
        
    params.update({
        'driver': 'GTiff',
        'dtype': rasterio.dtypes.get_minimum_dtype(res), 
    })
    
    with rasterio.open(out_path / out_name, 'w', **params) as dst: 
        dst.write(res, indexes=1)
        dst.close()
        
    print(f'Saved to {str(out_path / out_name)}')
    print(f'Out dtype: {params["dtype"]}')
    
    
def save_pred_map_to_tif(model, 
                         path_S2B, 
                         TIF_NAME, 
                         channels_list=CHANNELS,
                         indices=False, 
                         PATCH_SIZE_X=1000, 
                         PATCH_SIZE_Y=1000, 
                         data_type=np.float32, 
                         show=False, 
                         proba=False):
    # THE BEST!
    
    """
    regression_map: {'age', 'stock' ...} is a name of .tif file with the target
    path_S2B: {'./'} is a path to the folder with all the files 'B02.tif', 'B03.tif', ...
    channels_list: {['B02', 'B03', ..., 'B12', 'type']} list of the channels, should be in alligner`
    TIF_NAME = 'exp1.tif' 
    """

    # Save parameters of .tiff file
    print(f'Saving to {path_S2B + TIF_NAME}...')
    with rasterio.open(path_S2B + 'B02.tif') as src:
        ind_map = src.read(1)
        ind_map_width, ind_map_height = src.width, src.height, 
        ind_map_bounds, ind_map_transform, ind_map_crs = src.bounds, src.transform, src.crs

    num_x_patches = ind_map_width // PATCH_SIZE_X + 1
    num_y_patches = ind_map_height // PATCH_SIZE_Y + 1

    # Create or rewrite existing file with an empty one
    with rasterio.open(path_S2B + TIF_NAME, 'w', driver='GTiff', height=ind_map_height, width=ind_map_width,
                        count=1, dtype=rasterio.float32, crs=ind_map_crs, transform=ind_map_transform) as dst: 
        dst.close()

    with rasterio.open(path_S2B + TIF_NAME, 'r+') as dst:
        for j in tqdm(range(num_y_patches), total=num_y_patches): # i - horizontal step; j - vertical
            y0 = j * PATCH_SIZE_Y
            for i in range(num_x_patches):
                x0 = i * PATCH_SIZE_X

                if (x0 + PATCH_SIZE_X) < ind_map_width: 
                    x = x0 + PATCH_SIZE_X
                else: 
                    x = ind_map_width
                if (y0 + PATCH_SIZE_Y) < ind_map_height: 
                    y = y0 + PATCH_SIZE_Y
                else: 
                    y = ind_map_height

                window = Window.from_slices((y0, y), (x0, x))

                patch_features = {}

                for band in channels_list:
                    with rasterio.open(path_S2B + f'{band}.tif') as src:
                        patch = src.read(1, window=window)
                        patch_features[band] = patch.reshape(-1)
                        if band != 'catboost_10bands_type':
                            patch_features[band] = patch_features[band].astype(float)

                patch_features = pd.DataFrame(patch_features) 
                
                if indices:
                    patch_features = vegetation_indices(patch_features, drop_nans=False)   

                # predict
                if not proba:
                    y_pred_patch = model.predict(patch_features)
                else:
                    y_pred_patch = model.predict_proba(patch_features)[:, 1] # predict of the 1st class (target class)
                y_pred_patch = y_pred_patch.reshape(window.height, window.width)
                y_pred_patch = y_pred_patch.astype(data_type)


                # save predictions only for pixels with SCL mask == 4 (vegetation)
                with rasterio.open(path_S2B + 'SCL.tif') as src:
                    SCL_patch = src.read(1, window=window)

                y_pred_patch = np.where(SCL_patch == 4.0, y_pred_patch, -1)

                dst.write(y_pred_patch, 1, window=window)

        dst.close()


    with rasterio.open(path_S2B + TIF_NAME, 'r') as src:
        img = src.read(1)

    # plt.ioff()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    cs = ax.imshow(img)
    ax.set_title(path_S2B + TIF_NAME)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(cs)

    if show: 
        plt.show()
    else:
        plt.close(fig)
    return fig


def save_dataset_to_tif(
    ds,
    window,
    model, 
    path_S2B, 
    out_name,
    target,
    no_data_value=-1,
    out_dtype=np.float32, 
    proba=False,
    gt=False,
    clip=(0, None),
    
):
    
    # Save parameters of .tiff file
#     with rasterio.open(path_S2B + 'B02.tif') as src:
#         inp_width = src.width 
#         inp_height = src.height
#         inp_transform = src.transform
#         inp_crs = src.crs

#     params = {
#         'driver': 'GTiff',
#         'height': inp_height,
#         'width': inp_width,
#         'count': 1,
#         'dtype': out_dtype,
#         'crs': inp_crs,
#         'transform': inp_transform
#              }

    ################
    if window:
        with rasterio.open(path_S2B + 'B02.tif') as src:
            crop = src.read(1, window=window)
            params = src.meta.copy()
            out_transform = src.window_transform(window)
            params.update(
                {
                    "driver": 'GTiff',
                    "height": crop.shape[0],
                    "width": crop.shape[1],
                    "transform": out_transform,
                }
            )
    else:
        with rasterio.open(path_S2B / 'B02.tif') as src:
            params = src.meta.copy()
    
    params.update(
        {
            'count': 1,
            'dtype': out_dtype,
        }
    )
    ################
    
    if gt:
        pred = ds[target]
    else:
        if target in ds.columns:
            ds4pred = ds.drop(labels=['x', 'y', target], axis=1)
        else:
            ds4pred = ds.drop(labels=['x', 'y'], axis=1)

        if proba:
            pred = model.predict_proba(ds4pred)
        else:
            pred = model.predict(ds4pred)
            
    if clip:
        pred = np.clip(pred, *clip)
        pred = np.rint(pred).astype(int)
        
    if window:
        out = np.ones((params['height'], params['width'])) * no_data_value
    else:
        out = np.ones((inp_height, inp_width)) * no_data_value
    
    coords = ds[['x', 'y']]
    
    for i, (x, y) in tqdm(enumerate(coords.values), total=len(coords), colour='purple'):
        out[x, y] = pred.iloc[i]
        
    print(out.shape, out.sum())
    print(params)
    print(f'Saving to {path_S2B + out_name}...')
    with rasterio.open(path_S2B + out_name, 'w', **params) as dst:
        ################
        dst.write(out, indexes=1) 
        ################
        # if window:
        #     dst.write(out, window=window, indexes=1)
        # else:
        #     dst.write(out, indexes=1)
        # dst.close()
        