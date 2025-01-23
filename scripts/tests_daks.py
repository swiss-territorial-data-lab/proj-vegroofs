import os, sys
import yaml
import argparse
from loguru import logger

from joblib import Parallel, delayed
import multiprocessing
from threading import Lock
from tqdm_joblib import tqdm_joblib

import pandas as pd
import geopandas as gpd
import dask_geopandas as dg
# from dask.distributed import Client
from time import time
import fiona
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from tqdm import tqdm
import csv
from csv import writer

if __name__ == '__main__':
    ROOFS_POLYGONS = "sources/Footprint/ZH/00-AVBodenbedGeb_updated.gpkg"
    CHM_LAYER = "sources/CHM/ZH/chm_ZH_total.shp"
    WORKING_DIR = "D:/GitHubProjects/STDL_vegroof_production"
    CHM = os.path.join(WORKING_DIR, CHM_LAYER)


    print('starting to load roofs')
    time_start = time()
    green_roofs_egid = gpd.read_file(os.path.join(WORKING_DIR, ROOFS_POLYGONS))
    print(f'finished to load roofs in {time() - time_start}sec')
    
    print('starting to load CHM')
    # Start a Dask client for computation
    # client = Client()
    time_start = time()
    CHM_GPD = dg.read_file(CHM, chunksize=100000)
    delayed_partitions = CHM_GPD.to_delayed()
    results = []

    for _, delayed_partition in tqdm(enumerate(delayed_partitions), total=len(delayed_partitions)):
        # Compute the partition (convert to a GeoDataFrame)
        partition_gdf = delayed_partition.compute()
        # print(partition_gdf.head())
        
        # Perform your operation on the partition
        # print(f"Processing partition with {len(partition_gdf)} rows")
        results.append(gpd.overlay(partition_gdf, green_roofs_egid, how='difference'))
        # results.append(len(partition_gdf))
    print(results)



    quit()
    # CHM_GPD = CHM_GPD.compute()
    CHM_GPD = CHM_GPD.calculate_spatial_partitions()
    small_bounds = green_roofs_egid.total_bounds
    CHM_GPD = CHM_GPD.cx[
        small_bounds[0]:small_bounds[2], small_bounds[1]:small_bounds[3]
        ]
    CHM_GPD = CHM_GPD.compute()
    CHM_GPD['geometry'] = CHM_GPD.buffer(1)
    print(f'finished to load CHM in {time() - time_start}sec')
    
    print('starting overlay')
    time_start = time()
    green_roofs_egid=gpd.overlay(CHM_GPD, green_roofs_egid, how='difference')
    # green_roofs_egid = CHM_GPD.overlay(green_roofs_egid, how='difference')
    print(f'finished to overlay in {time() - time_start}sec')
    green_roofs_egid['area_green'] = green_roofs_egid.area