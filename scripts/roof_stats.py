import os, sys
import yaml
from tqdm import tqdm
from loguru import logger
import argparse

import pandas as pd
import geopandas as gpd
import statistics
from rasterstats import zonal_stats

import csv
from csv import writer

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger=fct_misc.format_logger(logger)


if __name__ == "__main__":

    logger.info('Starting...')

    logger.info(f"Using config.yaml as config file.")

    parser = argparse.ArgumentParser(
        description="The script detects the greenery on roofs")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['roof_stats.py']

    # with open('config/logReg.yml') as fp:
    #         cfg = yaml.load(fp, Loader=yaml.FullLoader)['roof_stats.py']

    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']
    INPUTS=cfg['inputs']

    ORTHO_DIR=INPUTS['ortho_directory']
    NDVI_DIR=INPUTS['ndvi_directory']
    LUM_DIR=INPUTS['lum_directory']
    OUTPUT_DIR=cfg['output_directory']

    TILE_DELIMITATION=INPUTS['tile_delimitation']

    ROOFS_POLYGONS=INPUTS['roofs_file']
    ROOFS_LAYER=INPUTS['roofs_layer']

    os.chdir(WORKING_DIR)


    logger.info('Stock path to images and corresponding NDVI and luminosity rasters...')

    tiles=gpd.read_file(TILE_DELIMITATION)

    tiles=fct_misc.get_ortho_tiles(tiles, ORTHO_DIR, NDVI_DIR)

    tiles['path_lum']=[os.path.join(LUM_DIR, tile_name + '_lum.tif') for tile_name in tiles.NAME.to_numpy()]
    tiles['path_NDVI']=[os.path.join(NDVI_DIR, tile_name + '_NDVI.tif') for tile_name in tiles.NAME.to_numpy()]


    logger.info('Loading roofs/ground truth...')

    roofs=gpd.read_file(ROOFS_POLYGONS)
    # roofs.drop(columns=['essence', 'diam_tronc'], inplace=True)
    roofs.rename(columns={'class':'cls'}, inplace=True)
    roofs['geometry'] = roofs.buffer(-0.5)


    logger.info('Getting the statistics of roofs...')

    clipped_roofs=fct_misc.clip_labels(roofs, tiles)
    clipped_roofs = clipped_roofs.loc[(clipped_roofs['veg_new']==1)]

    roofs_stats=pd.DataFrame()                                                              
    calculated_stats=['min', 'max', 'mean', 'median', 'std']

    for roof in tqdm(clipped_roofs.itertuples(),
                    desc='Extracting statistics over clipped_roofs', total=clipped_roofs.shape[0]):    
        stats_ndvi=zonal_stats(roof.geometry, roof.path_NDVI, stats=calculated_stats,
            band=1, nodata=99999)
        
        stats_dict_ndvi=stats_ndvi[0]
        stats_dict_ndvi['band']='ndvi'
        stats_dict_ndvi['veg_new']=roof.veg_new
        stats_dict_ndvi['class']=roof.cls
        stats_dict_ndvi['confidence']=roof.confidence
        stats_dict_ndvi['surface_ca']= roof.surface_ca 
        stats_dict_ndvi['unID']= roof.unID
        stats_dict_ndvi['EGID']= roof.EGID

        roofs_stats=pd.concat([roofs_stats, pd.DataFrame(stats_dict_ndvi, index=[0])], ignore_index=True)

        stats_lum=zonal_stats(roof.geometry, roof.path_lum, stats=calculated_stats,
            band=1, nodata=99999)
        
        stats_dict_lum=stats_lum[0]    
        stats_dict_lum['band']='lum'
        stats_dict_lum['veg_new']=roof.veg_new
        stats_dict_lum['class']=roof.cls
        stats_dict_lum['confidence']=roof.confidence
        stats_dict_lum['surface_ca']= roof.surface_ca 
        stats_dict_lum['unID']= roof.unID
        stats_dict_lum['EGID']= roof.EGID                                                

        roofs_stats=pd.concat([roofs_stats, pd.DataFrame(stats_dict_lum, index=[0])], ignore_index=True)

    rounded_stats=roofs_stats.copy()
    cols=['min', 'max', 'median', 'mean', 'std']
    rounded_stats[cols]=rounded_stats[cols].round(3)

    filepath=os.path.join('','roof_stats.csv')
    rounded_stats.to_csv(filepath)
    del rounded_stats, cols, filepath


    logger.info('Printing overall min, median and max of NDVI and luminosity for the GT green roofs...')

    if not os.path.isfile('threshold_.csv'):
        max_cls = statistics.median(roofs_stats.loc[(roofs_stats['band']=='ndvi')]['max'])
        median_cls = statistics.median(roofs_stats.loc[(roofs_stats['band']=='ndvi')]['median'])
        min_cls = statistics.median(roofs_stats.loc[(roofs_stats['band']=='ndvi')]['min'])

        max_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['band']=='lum')]['max'])
        median_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['band']=='lum')]['median'])
        min_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['band']=='lum')]['min'])

        with open('threshold.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            row = ['class','ndvi_min','ndvi_median','ndvi_max','lum_min','lum_mean','lum_max']
            writer.writerow(row)
            row = ['tot', min_cls, median_cls, max_cls, min_cls_lum, median_cls_lum, max_cls_lum]
            writer.writerow(row)

    for cls in list(['i', 'l','e','s','t']):
        if sum(roofs_stats['class']==cls)>0:
            max_cls = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['max'])
            median_cls = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['median'])
            min_cls = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['min'])

            max_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['max'])
            median_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['median'])
            min_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['min'])


            row = [cls, min_cls, median_cls, max_cls, min_cls_lum, median_cls_lum, max_cls_lum]
            with open('threshold.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)