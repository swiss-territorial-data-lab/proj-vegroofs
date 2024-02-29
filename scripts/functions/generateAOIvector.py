# rename in generateExtent, make it compatible for fct_misc ?

import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from loguru import logger

from rasterio.features import dataset_features

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc


############################### INPUTS #########################################
#   DIR_IN : input directory with LAS files
#   DIR_OUT : directory for output files

WORKING_DIR='C:/Users/cmarmy/Documents/STDL/proj-vegroofs/'
os.chdir(WORKING_DIR)

PATH_IN = "data_test/02_intermediate/images/scratch/2022/ZH/tiles/"
PATH_OUT = fct_misc.ensure_dir_exists("data_test/02_intermediate/aoi/clip/szh")
fct_misc.ensure_dir_exists("data_test/02_intermediate/aoi/clip/szh/tiles")

################################################################################

def main(PATH_IN, files_name):
    
    aoi_merge=gpd.GeoDataFrame()
    for _name in files_name:

        _tif = os.path.join(PATH_IN, _name)
        logger.info(str(_name))

        with rasterio.open(_tif) as src:
            image=src.read(1)
            gdf = gpd.GeoDataFrame.from_features(dataset_features(src, bidx=2, as_mask=True, geographic=False, band=False))
            # gdf.set_crs('epsg:2056')
            gdf.set_crs(crs=src.crs, inplace=True)
            gdf.filename=_name.replace('.tif', '')
            gdf.to_file(os.path.join(PATH_OUT,'tiles',_name.replace('.tif', '.shp')))
            aoi_merge = pd.concat([aoi_merge, gpd.GeoDataFrame(gdf, index=[0])], ignore_index=True)
            src.close()

    aoi_merge=aoi_merge.drop(columns= ['val'])
    aoi_merge=aoi_merge.rename(columns={"filename":"NAME"})
    aoi_merge.to_file(os.path.join(PATH_OUT, 'AOI.shp'))
    # merge the shp files to have all tiles together. name tile in attribute. 


if __name__ == "__main__":
    
    # --->> TO ADAPT <<---
    CUR_DIR = os.getcwd()

    root = PATH_IN
    pattern = ".tif"
    list_name = []
    list_las = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(pattern):
                list_name.append(name)
                list_las.append(os.path.join(path, name))

    main(PATH_IN, list_name)