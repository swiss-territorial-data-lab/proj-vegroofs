import os, sys
import yaml
import argparse

import numpy as np
import rasterio

from loguru import logger
from tqdm import tqdm
from glob import glob

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc


def calculate_ndvi(tile, band_nbr_red=1, band_nbr_nir=0, path=None):
    '''
    Calculate the NDVI for each pixel of a tile and save the result in a new folder.

    - tile: path to the tile
    - band_nbr_red: number of the red band in the image
    - band_nbr_nir: number of the nir band in the image
    - path: filepath were to save the result. If None, no file is saved
    return: array with the ndvi value for each pixel.
    '''

    with rasterio.open(tile) as src:
        image = src.read()
        im_profile=src.profile

    red_band=image[band_nbr_red].astype('float32')
    nir_band=image[band_nbr_nir].astype('float32')
    ndvi_tile=np.divide((nir_band - red_band),(nir_band + red_band),
                        out=np.full_like(nir_band - red_band, -9999),
                        where=(nir_band + red_band)!=0)

    if path:
        im_profile.update(count= 1, dtype='float32', nodata=-9999)
        with rasterio.open(path, 'w', **im_profile) as dst:
            dst.write(ndvi_tile,1)

    return ndvi_tile

def calculate_lum(tile, band_nbr_red=1, band_nbr_green=2, band_nbr_blue=3, path=None):
    '''
    Calculate the luminosity for each pixel of a tile and save the result in a new folder.

    - tile: path to the tile
    - band_nbr_red: number of the red band in the image
    - band_nbr_green: number of the green band in the image
    - band_nbr_blue: number of the blue band in the image
    - path: filepath were to save the result. If None, no file is saved
    return: array with the luminosity value for each pixel.
    '''     

    with rasterio.open(tile) as src:
        image = src.read()
        im_profile=src.profile

    red_band=image[band_nbr_red].astype('float32')
    green_band=image[band_nbr_green].astype('float32')
    blue_band=image[band_nbr_blue].astype('float32')
    lum_tile=np.add((red_band + green_band),blue_band)

    if path:
        im_profile.update(count= 1, dtype='float32')
        with rasterio.open(path, 'w', **im_profile) as dst:
            dst.write(lum_tile,1)

    return lum_tile

if __name__ == "__main__":

    logger=fct_misc.format_logger(logger)
    
    logger.info('Starting...')

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="This script computes NDVI and luminosity from NRGB rasters.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['dev']

    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']

    ORTHO_DIR=cfg['ortho_directory']
    NDVI_DIR=cfg['ndvi_directory']
    LUM_DIR=cfg['lum_directory']

    os.chdir(WORKING_DIR)

    _=fct_misc.ensure_dir_exists(NDVI_DIR)
    _=fct_misc.ensure_dir_exists(LUM_DIR)

    logger.info('Reading files...')
    
    tile_list_ortho=glob(os.path.join(ORTHO_DIR, '*.tif'))

    tile_list=[]
    tile_list.extend(tile_list_ortho)

    for tile in tqdm(tile_list, 'Processing tiles'):
        tile = tile.replace("\\","/") #handle windows path
        ndvi_tile_path=os.path.join(NDVI_DIR, tile.split('/')[-1].replace('.tif', '_NDVI.tif'))
        _ = calculate_ndvi(tile, path=ndvi_tile_path)
        lum_tile_path=os.path.join(LUM_DIR, tile.split('/')[-1].replace('.tif', '_lum.tif'))
        _ = calculate_lum(tile, path=lum_tile_path)

    logger.success(f'The files were written in the folder {NDVI_DIR} and {LUM_DIR}.')