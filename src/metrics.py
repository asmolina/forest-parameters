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

from sklearn.metrics import (
    mean_absolute_percentage_error, 
    mean_absolute_error, 
    mean_squared_error, 
    confusion_matrix, 
    classification_report
)

CHANNELS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']




def get_metrics(model, X_eval, Y_eval, name, limits=[0, 141]):
    Y_pred = model.predict(X_eval)

    R2 = np.corrcoef(Y_eval, Y_pred)[0][1]
    MAPE = mean_absolute_percentage_error(Y_eval, Y_pred)
    RMSE = mean_squared_error(Y_eval, Y_pred, squared=False)
    MAE = mean_absolute_error(Y_eval, Y_pred)

    # plt.ioff()
    fig = plt.figure(figsize=(6,6)) 
    plt.scatter(Y_pred, Y_eval, s=0.01, alpha=0.1)
    plt.scatter(Y_eval, Y_eval, s=0.01, color='black')
    plt.xlabel('y_pred')
    plt.ylabel('y_eval (groundtruth)')
    plt.xlim(limits)
    plt.ylim(limits)
    plt.title(f'R2 = {R2:.3f}| MAPE = {MAPE:.3f} | RMSE = {RMSE:.3f} | MAE = {MAE:.3f}')
    # plt.close(fig)

    return R2, MAPE, RMSE, MAE, fig


def get_classification_metrics(model, X_eval, Y_eval):
    Y_pred = model.predict(X_eval)

    transdict = {1: 'Spruce', 2: 'Birch', 3: 'Mix', 4: 'Aspen', 5: 'Fir'}
    model_classes = [transdict[class_] for class_ in model.classes_]

    fig = px.imshow(confusion_matrix(Y_eval, Y_pred), 
                    text_auto=True, 
                    x=model_classes, 
                    y=model_classes,
                    labels=dict(x="Predicted", y="Actual")
                   )
    
    return classification_report(Y_eval, Y_pred, output_dict=True, target_names=model_classes), fig



def get_feature_importances(model, X_train):
    col_sorted_by_importance=model.feature_importances_.argsort()
    feat_imp=pd.DataFrame({
        'cols':X_train.columns[col_sorted_by_importance],
        'imps':model.feature_importances_[col_sorted_by_importance]
    })

    fig = px.bar(feat_imp.sort_values(['imps'], ascending=False)[:], 
           x='cols', y='imps', labels={'cols':'column', 'imps':'feature importance'})

    return fig

def get_catboost_importances(model):
    feat_imp = model.get_feature_importance(prettified=True)

    fig = px.bar(feat_imp, x='Feature Id', y='Importances', 
                 labels={'Feature Id':'Feature Id', 'Importances':'Feature Importance'})
    return fig
