import os, sys
import yaml
import argparse
from loguru import logger
from tqdm import tqdm

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
    RESULT_DIR = OUTPUTS['result_directory']

    os.chdir(WORKING_DIR)
    fct_misc.ensure_dir_exists(OUTPUT_DIR)
    fct_misc.ensure_dir_exists(RESULT_DIR)
    if not os.path.isfile(os.path.join(TILE_DELIMITATION,'extent.shp')):
        fct_misc.generate_extent(ORTHO_DIR, TILE_DELIMITATION, EPSG)
    tiles=gpd.read_file(TILE_DELIMITATION)

    logger.info('Reading AOI geometries...')

    aoi = gpd.read_file(AOI)
    # filter out invalid geometries
    invalid_samples = aoi.loc[~aoi.geometry.is_valid]
    aoi = aoi.loc[aoi.geometry.is_valid]
    invalid_samples.to_file(os.path.join(RESULT_DIR, 'invalid_samples.gpkg'), driver='GPKG')
    aoi.to_file(os.path.join(RESULT_DIR, 'valid_samples.gpkg'), driver='GPKG')
    
    # keep only the geometry column
    aoi = aoi.filter(['geometry'])
    # buffer every geometry by 50 units
    for index, row in tqdm(aoi.iterrows(), total=len(aoi), desc="Buffering geometries"):
        row = row.copy()
        aoi.loc[index, 'geometry'] = row.geometry.buffer(1,join_style=2)


    aoi_clipped=fct_misc.clip_labels(labels_gdf=aoi, tiles_gdf=tiles, predicate_sjoin='intersects')
    aoi_clipped=aoi_clipped.reset_index(drop=True)

    i=1
    for idx,row in tqdm(aoi_clipped.iterrows(), total=len(aoi_clipped), desc="Clipping rasters"): 
        fct_misc.clip_im(ORTHO_DIR, aoi_clipped.iloc[[idx]], OUTPUT_DIR, i, EPSG)
        i=i+1
    logger.success(f'Successfully clipped {i-1} images.')