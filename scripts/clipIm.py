import os, sys
import yaml
import warnings
from loguru import logger
import tqdm as tqdm

import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from shapely.ops import unary_union
import fiona
import rasterio

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger=fct_misc.format_logger(logger)

logger.info('Starting...')

logger.info(f"Using config.yaml as config file.")
with open('config/clipIm.yml') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['clipImScratch']

logger.info('Defining constants...')

WORKING_DIR=cfg['working_directory']
INPUTS=cfg['inputs']

ORTHO_DIR=INPUTS['ortho_directory']
AOI=INPUTS['aoi']
TILE_DELIMITATION=INPUTS['tile_delimitation']
OUTPUT_DIR=cfg['output_directory']

os.chdir(WORKING_DIR)
fct_misc.ensure_dir_exists(OUTPUT_DIR)

logger.info('Reading files...')

aoi = gpd.read_file(AOI)
aoi = aoi.filter(['geometry'])
for index, row in aoi.iterrows():
    row = row.copy()
    aoi.loc[index, 'geometry'] = row.geometry.buffer(10)

tiles=gpd.read_file(TILE_DELIMITATION)

aoi_clipped=fct_misc.clip_labels(aoi, tiles)
aoi_clipped=aoi_clipped[~aoi_clipped.is_empty]
aoi_clipped = aoi_clipped.reset_index(drop=True)

i=1
for idx,row in aoi_clipped.iterrows(): 
        fct_misc.clipIm(ORTHO_DIR, aoi_clipped.iloc[[idx]], OUTPUT_DIR,i)
        i=i+1