import os, sys
import yaml
from tqdm import tqdm
from loguru import logger
import argparse
 
import pandas as pd
import geopandas as gpd
import statistics
from rasterstats import zonal_stats
from sklearn.model_selection import train_test_split

import csv
from csv import writer

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

from math import isnan
from itertools import filterfalse                    

logger=fct_misc.format_logger(logger)


if __name__ == "__main__":

    logger.info('Starting...')

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="The script detects and predict the greenery on roofs")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['dev']

    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']
    RESULTS_DIR=cfg['results_directory']

    ORTHO_DIR=cfg['ortho_directory']
    NDVI_DIR=cfg['ndvi_directory']
    LUM_DIR=cfg['lum_directory']

    TILE_DELIMITATION=cfg['tile_delimitation']

    ROOFS_POLYGONS=cfg['roofs_gt']
    ROOFS_LAYER=cfg['roofs_layer']

    EPSG=cfg['epsg']

    os.chdir(WORKING_DIR)

    _=fct_misc.ensure_dir_exists(RESULTS_DIR)
    written_files=[]

    im_path=fct_misc.ensure_dir_exists(os.path.join(RESULTS_DIR,'boxplots'))


    logger.info('Stock path to images and corresponding NDVI and luminosity rasters...')

    fct_misc.generate_extent(ORTHO_DIR, TILE_DELIMITATION, EPSG)
    tiles=gpd.read_file(TILE_DELIMITATION)

    # tiles=fct_misc.get_ortho_tiles(tiles, ORTHO_DIR, NDVI_DIR)

    # tiles['path_lum']=[os.path.join(LUM_DIR, tile_name + '_lum.tif') for tile_name in tiles.NAME.to_numpy()]
    # tiles['path_NDVI']=[os.path.join(NDVI_DIR, tile_name + '_NDVI.tif') for tile_name in tiles.NAME.to_numpy()]


    tiles=fct_misc.get_ortho_tiles(tiles, ORTHO_DIR, NDVI_DIR, LUM_DIR)

    logger.info('Loading roofs/ground truth...')

    roofs=gpd.read_file(ROOFS_POLYGONS)
    # roofs.drop(columns=['essence', 'diam_tronc'], inplace=True)
    roofs.rename(columns={'class':'cls'}, inplace=True)
    roofs['geometry'] = roofs.buffer(-0.5)


    logger.info('Defining training and test dataset...')   
    
    roofs['veg_new'].fillna(0, inplace = True)
    lg_train, lg_test = train_test_split(roofs, test_size=0.3, train_size=0.7, random_state=0, shuffle=True, stratify=roofs['cls']) 
    lg_train = pd.DataFrame(lg_train)
    lg_train = lg_train.assign(train=1)
    lg_test = pd.DataFrame(lg_test)
    lg_test = lg_test.assign(train=0)
    roofs_split = pd.concat([lg_train, lg_test], ignore_index=True)
    roofs_split.to_csv(os.path.join(RESULTS_DIR.split('/')[-2],'EGID_train_test.csv'))


    logger.info('Getting the statistics of roofs...')

    clipped_roofs=fct_misc.clip_labels(labels_gdf=roofs, tiles_gdf=tiles)
    # clipped_roofs = clipped_roofs.loc[(clipped_roofs['veg_new']==0)]

    # # hovering vegetation filter
    # TREE = "C:/Users/cmarmy/Documents/STDL/proj-vegroofs/data/02_intermediate/autres/tlm3d_bb_einzelbaum_buf5m_aoi.gpkg"
    # trees=gpd.read_file(TREE, layer='merge')
    # trees.geometry=trees.geometry.buffer(2)
    # clipped_roofs=gpd.overlay(clipped_roofs, trees, how='difference')
    # clipped_roofs.to_file('test.gpkg')
                                
    roofs_stats=pd.DataFrame()                                                              
    calculated_stats=['min', 'max', 'mean', 'median', 'std']
    BANDS={1: 'nir', 2: 'red', 3: 'green', 4: 'blue'}

    for roof in tqdm(clipped_roofs.itertuples(),
                    desc='Extracting statistics over clipped_roofs', total=clipped_roofs.shape[0]):    
       
       for band_num in BANDS.keys():
            stats_rgb=zonal_stats(roof.geometry, roof.path_RGB, stats=calculated_stats,band=band_num, nodata=0)

            stats_dict_rgb=stats_rgb[0]
            stats_dict_rgb['band']=BANDS[band_num]
            stats_dict_rgb['veg_new']=roof.veg_new
            stats_dict_rgb['class']=roof.cls
            stats_dict_rgb['confidence']=roof.confidence
            stats_dict_rgb['surface_ca']= roof.surface_ca 
            stats_dict_rgb['unID']= roof.unID
            stats_dict_rgb['EGID']= roof.EGID                                            
            
            roofs_stats=pd.concat([roofs_stats, pd.DataFrame(stats_dict_rgb, index=[0])], ignore_index=True)
        stats_ndvi=zonal_stats(roof.geometry, roof.path_NDVI, stats=calculated_stats,
            band=1, nodata=-9999)
        
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
            band=1, nodata=0)
        
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
    rounded_stats = rounded_stats.dropna(axis=0,subset=['min','max','mean','std','median'])
    rounded_stats.drop_duplicates(inplace=True)
    rounded_stats.drop_duplicates(subset=['unID','band'], inplace=True)

    filepath=os.path.join(RESULTS_DIR,'roof_stats.csv')
    rounded_stats.to_csv(filepath)
    del rounded_stats, cols, filepath


    logger.info('Printing overall min, median and max of NDVI and luminosity for the GT green roofs...')

    if not os.path.isfile(os.path.join(RESULTS_DIR,'roof_stats_summary.csv')):
        max_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='ndvi')]['max'])))
        median_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='ndvi')]['median'])))
        min_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='ndvi')]['min'])))

        max_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='lum')]['max'])))
        median_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='lum')]['median'])))
        min_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='lum')]['min'])))


        with open(os.path.join(RESULTS_DIR,'roof_stats_summary.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            title= ['class','ndvi_min','ndvi_median','ndvi_max','lum_min','lum_mean','lum_max']
            row = ['tot', min_cls, median_cls, max_cls, min_cls_lum, median_cls_lum, max_cls_lum]
            writer.writerows([title, row])

    for cls in list(['i', 'l','e','s','t','b']):
        if sum(roofs_stats['class']==cls)>0:
            max_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['max'])))
            median_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['median'])))
            min_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['min'])))

            max_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['max'])))
            median_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['median'])))
            min_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['min'])))

            row = [cls, min_cls, median_cls, max_cls, min_cls_lum, median_cls_lum, max_cls_lum]
            with open(os.path.join(RESULTS_DIR,'roof_stats_summary.csv'), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)

    
    roofs_stats.loc[roofs_stats['class']=='b', 'class']='1. b'
    roofs_stats.loc[roofs_stats['class']=='t', 'class']='2. t'
    roofs_stats.loc[roofs_stats['class']=='s', 'class']='3. s'
    roofs_stats.loc[roofs_stats['class']=='e', 'class']='4. e'
    roofs_stats.loc[roofs_stats['class']=='l', 'class']='5. l'
    roofs_stats.loc[roofs_stats['class']=='i', 'class']='6. i'

    for band in roofs_stats['band'].unique():
        logger.info(f'For band {band}...')
        band_stats=roofs_stats[roofs_stats['band']==band]

        logger.info('... making some boxplots...')
        bxplt_beeches=band_stats[calculated_stats + ['class']].plot.box(
                                    by='class',
                                    title=f'Statistics distribution for roofs per band {band}',
                                    figsize=(18, 5),
                                    grid=True,
        )

        fig=bxplt_beeches[0].get_figure()
        filepath=os.path.join(im_path, f'boxplot_stats_band_{band}.jpg')
        fig.savefig(filepath, bbox_inches='tight')
        written_files.append(filepath)