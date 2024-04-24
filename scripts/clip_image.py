import os, sys
import yaml
import argparse
from loguru import logger
import tqdm as tqdm

import geopandas as gpd

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger=fct_misc.format_logger(logger)


if __name__ == "__main__":

    logger.info('Starting...')

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="The script clips images with a vector layer.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/clipImage.yaml")
    args = parser.parse_args()

    # load input parameters from config file
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['clip_image']


    logger.info('Defining constants...')

    # Define constants
    WORKING_DIR=cfg['working_directory']

    INPUTS=cfg['inputs']
    ORTHO_DIR=INPUTS['ortho_directory']
    AOI=INPUTS['aoi']
    EPSG=INPUTS['epsg']

    OUTPUTS=cfg['outputs']
    OUTPUT_DIR=OUTPUTS['clip_ortho_directory']
    TILE_DELIMITATION=OUTPUTS['extent_ortho_directory']

    os.chdir(WORKING_DIR)
    fct_misc.ensure_dir_exists(OUTPUT_DIR)


    logger.info('Reading AOI geometries...')

    aoi = gpd.read_file(AOI)
    # keep only the geometry column
    aoi = aoi.filter(['geometry'])
    # buffer every geometry by 50 units
    for index, row in aoi.iterrows():
        row = row.copy()
        aoi.loc[index, 'geometry'] = row.geometry.buffer(50,join_style=2)

    fct_misc.generate_extent(ORTHO_DIR, TILE_DELIMITATION, EPSG)
    tiles=gpd.read_file(TILE_DELIMITATION)

    aoi_clipped=fct_misc.clip_labels(labels_gdf=aoi, tiles_gdf=tiles)
    # keep only elements that are non-empty
    aoi_clipped=aoi_clipped[~aoi_clipped.is_empty]
    aoi_clipped=aoi_clipped.reset_index(drop=True)

    i=1
    for idx,row in aoi_clipped.iterrows(): 
        fct_misc.clip_im(ORTHO_DIR, aoi_clipped.iloc[[idx]], OUTPUT_DIR, i, EPSG)
        i=i+1
    logger.success(f'Successfully clipped {i-1} images.')