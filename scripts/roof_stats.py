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

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="The script detects the greenery on roofs")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['prod']

    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']

    ORTHO_DIR=cfg['ortho_directory']
    NDVI_DIR=cfg['ndvi_directory']
    LUM_DIR=cfg['lum_directory']

    TILE_DELIMITATION=cfg['tile_delimitation']

    ROOFS_POLYGONS=cfg['roofs_gt']
    ROOFS_LAYER=cfg['roofs_layer']

    EPSG=cfg['epsg']

    os.chdir(WORKING_DIR)


    logger.info('Stock path to images and corresponding NDVI and luminosity rasters...')

    fct_misc.generate_extent(ORTHO_DIR, TILE_DELIMITATION, EPSG)
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

    filepath=os.path.join(WORKING_DIR,'roof_stats.csv')
    rounded_stats.to_csv(filepath)
    del rounded_stats, cols, filepath


    logger.info('Printing overall min, median and max of NDVI and luminosity for the GT green roofs...')

    if not os.path.isfile('roof_stats_summary.csv'):
        max_cls = statistics.median(roofs_stats.loc[(roofs_stats['band']=='ndvi')]['max'])
        median_cls = statistics.median(roofs_stats.loc[(roofs_stats['band']=='ndvi')]['median'])
        min_cls = statistics.median(roofs_stats.loc[(roofs_stats['band']=='ndvi')]['min'])

        max_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['band']=='lum')]['max'])
        median_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['band']=='lum')]['median'])
        min_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['band']=='lum')]['min'])

        with open('roof_stats_summary.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            title= ['class','ndvi_min','ndvi_median','ndvi_max','lum_min','lum_mean','lum_max']
            row = ['tot', min_cls, median_cls, max_cls, min_cls_lum, median_cls_lum, max_cls_lum]
            writer.writerows([title, row])

    for cls in list(['i', 'l','e','s','t']):
        if sum(roofs_stats['class']==cls)>0:
            max_cls = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['max'])
            median_cls = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['median'])
            min_cls = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['min'])

            max_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['max'])
            median_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['median'])
            min_cls_lum = statistics.median(roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['min'])


            row = [cls, min_cls, median_cls, max_cls, min_cls_lum, median_cls_lum, max_cls_lum]
            with open('roof_stats_summary.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)