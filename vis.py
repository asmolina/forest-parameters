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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import (
    mean_absolute_percentage_error, 
    mean_absolute_error, 
    mean_squared_error, 
    confusion_matrix, 
    classification_report
)

CHANNELS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']



def plot_image(image, factor=1.0, clip_range=None, figsize=(10, 10), **kwargs):
    
    """Utility function for plotting RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
        
    ax.set_xticks([])
    ax.set_yticks([])
    
    

def draw_maps(path_to_pred_tif, path_to_gt_tif, limits=[0, 141], no_data=0, window=None, show=False):

    with rasterio.open(path_to_gt_tif) as src:
        if window is None:
            gt_map = src.read(1)
        else:
            gt_map = src.read(1, window=window)


    with rasterio.open(path_to_pred_tif) as src:
        if window is None:
            pred_map = src.read(1)
        else:
            pred_map = src.read(1, window=window)


    # mask := only pixels with existing groundtruth 
    mask = np.where(gt_map == no_data, 0, 1)
    pred_map = np.where(mask == 1, pred_map, -1)

    plt.ioff()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    cmap = cm.get_cmap('viridis')
    normalizer = Normalize(vmin=limits[0], vmax=limits[1], clip=True)
    im = cm.ScalarMappable(norm=normalizer)

    images = [pred_map, gt_map]
    names = [f'Per pixel predictions ({path_to_pred_tif})', 
             f'Groundtruth ({path_to_gt_tif})']

    for image, name, ax in zip(images, names, axes.flat):
        ax.imshow(image, cmap=cmap, norm=normalizer)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist())
    # fig.suptitle('Predictions')
    if show: 
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_SCL(tif_path, **kwargs):
    
    SCL_legend = {
        0: '0: No Data (Missing data)',
        1: '1: Saturated or defective pixel',
        2: '2: Topographic casted shadows',
        3: '3: Cloud shadows',
        4: '4: Vegetation',
        5: '5: Not-vegetated',
        6: '6: Water',
        7: '7: Unclassified',
        8: '8: Cloud medium probability',
        9: '9: Cloud high probability',
        10: '10: Thin cirrus',
        11: '11: Snow or ice',
    }

    SCL_color_legend = {
    0: '#000000',
    1: '#ff0000',
    2: '#2f2f2f',
    3: '#643200',
    4: '#00a000',
    5: '#ffe65a',
    6: '#0000ff',
    7: '#808080',
    8: '#c0c0c0',
    9: '#ffffff',
    10: '#64c8ff',
    11: '#ff96ff',
    }
    
    plot_mask(tif_path, SCL_legend, SCL_color_legend, **kwargs)

    
    
def plot_mask(tif_path,
              legend, 
              color_legend, 
              tif_title=None, 
              factor=None, 
              add_ticks=True,
              save_path=None):
    # factor = 100 or 1000


    # Load the TIFF image and read the first band
    with rasterio.open(tif_path) as dataset:
        data = dataset.read(1)
        print(f'data.shape is {data.shape}')

    # Define the colormap and its bounds
    cmap = colors.ListedColormap([color_legend[key] for key in color_legend])
    bounds = list(color_legend.keys()) + [max(color_legend) + 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Create a figure and axis with adjusted shape
    if factor:
        rows, cols = data.shape
        figsize = (cols // factor, rows // factor)
    else: 
        figsize=None
    fig, ax = plt.subplots(figsize=figsize)

    # Display the image with the defined colormap and norm
    img = ax.imshow(data, cmap=cmap, norm=norm)

    # Add and set a colorbar
    # cbar = fig.colorbar(img, ax=ax)
    # cbar.set_label('Pixel Value')

    # Create a custom legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_legend[key]) for key in legend]
    labels = [legend[key] for key in legend]
    plt.legend(handles, labels, loc=(1.04, 0), fancybox=True)

    # Adjust layout
    if not add_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    
    if tif_title:
        plt.title(tif_title)
    
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    
    plt.show()
    return data


def plot_res(tif_path, cbar_label = 'Result [pixel values]'):
    
    with rasterio.open(tif_path) as f:
        res = f.read(1)
        print(f'res.shape is {res.shape}')
        
        
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('Greens')
    img = ax.imshow(res, cmap=cmap, vmin=res.min(), vmax=res.max())

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(cbar_label)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()
    
    