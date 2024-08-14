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

from math import isnan
from itertools import filterfalse      

from joblib import Parallel, delayed
import multiprocessing
from threading import Lock
from tqdm_joblib import tqdm_joblib

import matplotlib 
import plotly

import csv
from csv import writer

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc           

lock = Lock()

logger=fct_misc.format_logger(logger)

def do_stats(roof):
    roofs_stats_list=pd.DataFrame()
    for band_num in BANDS.keys():
        stats_rgb=zonal_stats(roof.geometry, roof.path_RGB, stats=calculated_stats, band=band_num, nodata=0)

        stats_dict_rgb=stats_rgb[0]
        stats_dict_rgb['band']=BANDS[band_num]
        if GT: 
            stats_dict_rgb[GREEN_TAG]=roof.green_tag
            stats_dict_rgb['class']=roof.cls
            stats_dict_rgb['confidence']=roof.confidence
            stats_dict_rgb['surface_ca']=roof.surface_ca 
            stats_dict_rgb['unID']=roof.unID
        stats_dict_rgb['EGID']=roof.EGID                                            
        
        roofs_stats_list=pd.concat([roofs_stats_list, pd.DataFrame(stats_dict_rgb, index=[0])], ignore_index=True)
        
    stats_ndvi=zonal_stats(roof.geometry, roof.path_NDVI, stats=calculated_stats,
        band=1, nodata=-9999)
    
    stats_dict_ndvi=stats_ndvi[0]
    stats_dict_ndvi['band']='ndvi'
    if GT:
        stats_dict_ndvi[GREEN_TAG]=roof.green_tag
        stats_dict_ndvi['class']=roof.cls
        stats_dict_ndvi['confidence']=roof.confidence
        stats_dict_ndvi['surface_ca']=roof.surface_ca 
        stats_dict_ndvi['unID']=roof.unID
    stats_dict_ndvi['EGID']=roof.EGID

    roofs_stats_list=pd.concat([roofs_stats_list, pd.DataFrame(stats_dict_ndvi, index=[0])], ignore_index=True)

    stats_lum=zonal_stats(roof.geometry, roof.path_lum, stats=calculated_stats,
        band=1, nodata=0)
    
    stats_dict_lum=stats_lum[0]    
    stats_dict_lum['band']='lum'
    if GT: 
        stats_dict_lum[GREEN_TAG]=roof.green_tag
        stats_dict_lum['class']=roof.cls
        stats_dict_lum['confidence']=roof.confidence
        stats_dict_lum['surface_ca']=roof.surface_ca 
        stats_dict_lum['unID']=roof.unID
    stats_dict_lum['EGID']=roof.EGID             

    roofs_stats_list=pd.concat([roofs_stats_list, pd.DataFrame(stats_dict_lum, index=[0])], ignore_index=True)

    return roofs_stats_list

if __name__ == "__main__":

    logger.info('Starting...')

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="The script computes statistics for pixels per roofs.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['dev']

    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']
    RESULTS_DIR=cfg['results_directory']

    ORTHO_DIR=cfg['ortho_directory']
    NDVI_DIR=cfg['ndvi_directory']
    LUM_DIR=cfg['lum_directory']

    TILE_DELIMITATION=cfg['tile_delimitation']

    ROOFS_POLYGONS=cfg['roofs_file']
    ROOFS_LAYER=cfg['roofs_layer']
    GT=cfg['gt']
    GREEN_TAG=cfg['green_tag']
    GREEN_CLS=cfg['green_cls']
    CHM_LAYER=cfg['chm_layer']
    EGID_TRAIN_TEST=cfg['egid_train_test']

    EPSG=cfg['epsg']

    os.chdir(WORKING_DIR)

    _=fct_misc.ensure_dir_exists(RESULTS_DIR)
    written_files=[]

    if GT:
        im_path=fct_misc.ensure_dir_exists(os.path.join(RESULTS_DIR,'boxplots'))


    logger.info('Stock path to images and corresponding NDVI and luminosity rasters...')

    fct_misc.generate_extent(ORTHO_DIR, TILE_DELIMITATION, EPSG)
    tiles=gpd.read_file(TILE_DELIMITATION)

    tiles=fct_misc.get_ortho_tiles(tiles, ORTHO_DIR, NDVI_DIR, LUM_DIR)

    logger.info('Loading roofs/ground truth...')

    roofs=gpd.read_file(ROOFS_POLYGONS, layer=ROOFS_LAYER)
    if GT:
        roofs.rename(columns={GREEN_CLS:'cls'}, inplace=True)
        roofs.rename(columns={GREEN_TAG:'green_tag'}, inplace=True)
    roofs['geometry'] = roofs.buffer(-0.1)

    logger.info('Filtering for overhanging vegetation...')
    CHM = os.path.join(WORKING_DIR, CHM_LAYER)
    chm=gpd.read_file(CHM)
    chm['geometry'] = chm.buffer(1)
    roofs_chm=gpd.overlay(roofs, chm, how='difference')
   
    if GT:
        logger.info('Defining training and test dataset...')   
        if not os.path.isfile(os.path.join(RESULTS_DIR,EGID_TRAIN_TEST)):
            roofs_chm['green_tag'].fillna(0, inplace = True)
            lg_train, lg_test = train_test_split(roofs_chm, test_size=0.3, train_size=0.7, random_state=0,
                                                 shuffle=True, stratify=roofs_chm['cls']) 
            lg_train = pd.DataFrame(lg_train)
            lg_train = lg_train.assign(train=1)
            lg_test = pd.DataFrame(lg_test)
            lg_test = lg_test.assign(train=0)
            roofs_chm_split = pd.concat([lg_train, lg_test], ignore_index=True)
            roofs_chm_split.to_csv(os.path.join(RESULTS_DIR,EGID_TRAIN_TEST))


    logger.info('Getting the statistics of roofs...')

    clipped_roofs=fct_misc.clip_labels(labels_gdf=roofs_chm, tiles_gdf=tiles, predicate_sjoin='within')
                                
    roofs_stats=pd.DataFrame()                                                              
    calculated_stats=['min', 'max', 'mean', 'median', 'std']
    BANDS={1: 'nir', 2: 'red', 3: 'green', 4: 'blue'}

    print("Multithreading with joblib for statistics over beeches: ")
    num_cores = multiprocessing.cpu_count()
    print ("starting job on {} cores.".format(num_cores))

    with tqdm_joblib(desc="Extracting statistics over clipped_roofs",
                     total=clipped_roofs.shape[0]) as progress_bar:
        roofs_stats_list = Parallel(n_jobs=num_cores, prefer="threads")(
            delayed(do_stats)(roof) for roof in clipped_roofs.itertuples())

    roofs_stats=pd.DataFrame()
    for row in roofs_stats_list:
        roofs_stats = pd.concat([roofs_stats, row])
    logger.info('... finished')

    rounded_stats=roofs_stats.copy()
    cols=['min', 'max', 'median', 'mean', 'std']
    rounded_stats[cols]=rounded_stats[cols].round(3)
    rounded_stats = rounded_stats.dropna(axis=0,subset=['min','max','mean','std','median'])
    rounded_stats.drop_duplicates(inplace=True)
    rounded_stats.drop_duplicates(subset=['EGID','band'], inplace=True)

    filepath=os.path.join(RESULTS_DIR,'roof_stats.csv')
    rounded_stats.to_csv(filepath)
    del rounded_stats, cols, filepath

    logger.info('Printing overall min, median and max of NDVI and luminosity for the GT green roofs...')

    STATS_CSV = 'roof_stats_summary.csv'

    if not os.path.isfile(os.path.join(RESULTS_DIR,STATS_CSV)):
        max_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='ndvi')]['max'])))
        median_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='ndvi')]['median'])))
        min_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='ndvi')]['min'])))

        max_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='lum')]['max'])))
        median_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='lum')]['median'])))
        min_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[(roofs_stats['band']=='lum')]['min'])))


        with open(os.path.join(RESULTS_DIR,STATS_CSV), 'w', newline='') as file:
            writer = csv.writer(file)
            title= ['class','ndvi_min','ndvi_median','ndvi_max','lum_min','lum_mean','lum_max']
            row = ['tot', min_cls, median_cls, max_cls, min_cls_lum, median_cls_lum, max_cls_lum]
            writer.writerows([title, row])
    if GT:
        for cls in list(['i', 'l','e','s','t','b']):
            if sum(roofs_stats['class']==cls)>0:
                max_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[
                    (roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['max'])))
                median_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[
                    (roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['median'])))
                min_cls = statistics.median(list(filterfalse(isnan, roofs_stats.loc[
                    (roofs_stats['class']==cls) & (roofs_stats['band']=='ndvi')]['min'])))

                max_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[
                    (roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['max'])))
                median_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[
                    (roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['median'])))
                min_cls_lum = statistics.median(list(filterfalse(isnan, roofs_stats.loc[
                    (roofs_stats['class']==cls) & (roofs_stats['band']=='lum')]['min'])))

                row = [cls, min_cls, median_cls, max_cls, min_cls_lum, median_cls_lum, max_cls_lum]
                with open(os.path.join(RESULTS_DIR,STATS_CSV), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)

    if GT: 
        roofs_stats.loc[roofs_stats['class']=='b', 'class']='1. b'
        roofs_stats.loc[roofs_stats['class']=='t', 'class']='2. t'
        roofs_stats.loc[roofs_stats['class']=='s', 'class']='3. s'
        roofs_stats.loc[roofs_stats['class']=='e', 'class']='4. e'
        roofs_stats.loc[roofs_stats['class']=='l', 'class']='5. l'
        roofs_stats.loc[roofs_stats['class']=='i', 'class']='6. i'
        roofs_stats = roofs_stats.reset_index().drop('index', axis=1)

        for band in roofs_stats['band'].unique():
            logger.info(f'For band {band}...')
            band_stats=roofs_stats[roofs_stats['band']==band]

            logger.info('... making some boxplots...')
            bxplt_roofs=band_stats[calculated_stats + ['class']].plot.box(
                                        by='class',
                                        title=f'Statistics distribution for roofs per band {band}',
                                        figsize=(18, 5),
                                        grid=True,
            )

            fig=bxplt_roofs[0].get_figure()
            filepath=os.path.join(im_path, f'boxplot_stats_band_{band}.jpg')
            fig.savefig(filepath, bbox_inches='tight')
            written_files.append(filepath)