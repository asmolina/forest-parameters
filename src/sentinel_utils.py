import rasterio
from rasterio.merge import merge
from sentinelhub import SHConfig, BBox, bbox_to_dimensions, CRS, SentinelHubRequest, DataCollection, MimeType, BBoxSplitter
import numpy as np
import matplotlib.pyplot as plt
import shapely
from shapely import wkt
import math
from skimage.transform import resize 
import datetime
from tqdm.autonotebook import  tqdm


def split_bbox(polygon, resolution=10, crs=CRS.WGS84, b=1024):
    target_bbox = BBox(bbox=polygon.bounds, crs=crs)
    target_size = bbox_to_dimensions(target_bbox, resolution=resolution)
    
    p, q = (math.ceil(target_size[0] / b), math.ceil(target_size[1] / b))
    bbox_splitter = BBoxSplitter([polygon], crs, (p,q)) 
    
    return bbox_splitter.get_bbox_list()


def merge_rasters(rasters, bboxs, resolution=10, crs=CRS.WGS84):
    datasets = []
    for raster, bbox in tqdm(zip(rasters, bboxs), total=len(rasters)):
        bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
        raster = raster.transpose((2,0,1))
        memfile = rasterio.io.MemoryFile()
        dataset = memfile.open(
                driver='GTiff',
                height=raster.shape[1],
                width=raster.shape[2],
                count=raster.shape[0],
                dtype=raster.dtype,
                crs=rasterio.crs.CRS.from_epsg(crs.epsg),
                transform=rasterio.transform.from_bounds(*bbox, width=bbox_size[0], height=bbox_size[1]))
        dataset.write(raster)   
        datasets.append(dataset)
        
    mosaic, transform = merge(datasets)
    
    for dataset in datasets:
        dataset.close()
        
    mosaic = np.nan_to_num(mosaic)
    return mosaic, transform

def make_s1_request(time_interval, bbox, scripts, resolution, config):
    def_scripts = {DataCollection.SENTINEL1_IW: """
                        //VERSION=3
                        function setup() {
                          return {
                            input: [{
                                bands: ["VV", "VH", "dataMask"],
                                units: "LINEAR_POWER"
                            }],
                            output: {
                                bands: 3,
                                sampleType: "FLOAT32"
                            }
                          };
                        }

                        function evaluatePixel(samples) {
                          return [toDb(samples.VV), toDb(samples.VH), samples.dataMask]
                        }

                        function toDb(linear) {
                          var log = 10 * Math.log(linear) / Math.LN10
                          return log
                        }
                        """
              }
    data_collection = DataCollection.SENTINEL1_IW
    
    return SentinelHubRequest(
                evalscript=def_scripts[data_collection],
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=data_collection,
                        time_interval=time_interval,
                        other_args= {
                            "processing": {
                                "orthorectify": "true",
                                "backCoeff": "GAMMA0_TERRAIN",
                                "demInstance": "COPERNICUS_30",
                                "speckleFilter": {
                                    "type": "LEE",
                                    "windowSizeX": 3,
                                    "windowSizeY": 3}
                            }
                        }
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=bbox_to_dimensions(bbox, resolution=resolution),
                config=config), data_collection

def make_s2_request(time_interval, bbox, scripts, resolution, config):
    def_scripts = {DataCollection.SENTINEL2_L2A : """
                        //VERSION=3

                        function setup() {
                            return {
                                input: [{
                                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12","SCL","dataMask"],
                                    units: "DN",
                                    processing: 'BILINEAR'
                                }],
                                output: {
                                    bands: 14,
                                    sampleType: "INT16"
                                }
                            };
                        }
                        
                        
                        function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
                            outputMetadata.userData = { "norm_factor":  inputMetadata.normalizationFactor, 
                                                        "scenes":  JSON.stringify(scenes)}
                        }

                        function evaluatePixel(sample) {
                                return [sample.B01,
                                sample.B02,
                                sample.B03,
                                sample.B04,
                                sample.B05,
                                sample.B06,
                                sample.B07,
                                sample.B08,
                                sample.B8A,
                                sample.B09,
                                sample.B11,
                                sample.B12,
                                sample.SCL,
                                sample.dataMask];
                        }
                       """,
               DataCollection.SENTINEL2_L1C : """
                        //VERSION=3

                        function setup() {
                            return {
                                input: [{
                                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12","CLM","dataMask"],
                                    units: "DN"
                                }],
                                output: {
                                    bands: 14,
                                    sampleType: "FLOAT32"
                                }
                            };
                        }

                        function evaluatePixel(sample) {
                                return [sample.B01,
                                sample.B02,
                                sample.B03,
                                sample.B04,
                                sample.B05,
                                sample.B06,
                                sample.B07,
                                sample.B08,
                                sample.B8A,
                                sample.B09,
                                sample.B11,
                                sample.B12,
                                sample.CLM,
                                sample.dataMask];
                        }
                        """
              }
    
    if datetime.datetime.strptime(time_interval[1], '%Y-%m-%dT%H:%M:%S') < datetime.datetime.strptime('2017-01-01T00:00:01', '%Y-%m-%dT%H:%M:%S'):
        data_collection = DataCollection.SENTINEL2_L1C
    else:
        data_collection = DataCollection.SENTINEL2_L2A
        
    if scripts != None:
        def_scripts = scripts
    
    return SentinelHubRequest(
                evalscript=def_scripts[data_collection],
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=data_collection,
                        time_interval=time_interval)],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF),
                ],
                bbox=bbox,
                size=bbox_to_dimensions(bbox, resolution=resolution),
                config=config), data_collection

requests = {
    'SENTINEL-1': make_s1_request,
    'SENTINEL-2': make_s2_request
}

def get_data(sat, polygon, time_interval, config, scripts=None, resolution=10, crs=CRS.WGS84, b=1024):
    splitted_bbox = split_bbox(polygon, resolution, crs, b)
    
    rasters = []
    make_request = requests[sat]
    for bbox in tqdm(splitted_bbox):
        request, data_collection = make_request(time_interval, bbox, scripts, resolution, config)
        raster = request.get_data()
        
        ####
        # import json
        # print(raster)
        # print(raster[0]['userdata.json']['scenes'])
        # json.loads(raster[0]['userdata.json']['scenes'])
        ####
        
        rasters.extend(raster)
        
    print('Merging tiles...', end=' ')
    mosaic, transform = merge_rasters(rasters, splitted_bbox, resolution, crs)
    print('Success!')
    return mosaic, transform, data_collection

