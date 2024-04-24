import os, sys
import yaml
import argparse
from loguru import logger
from joblib import Parallel, delayed
import multiprocessing
from threading import Lock
from tqdm_joblib import tqdm_joblib

import hydra
from omegaconf import DictConfig, OmegaConf

import pandas as pd
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape

import csv
from csv import writer

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

lock = Lock()

logger=fct_misc.format_logger(logger)

@hydra.main(version_base=None, config_path="../config/", config_name="logReg")

def my_app(cfg : DictConfig) -> None:
    green_roofs_egid_att.to_file(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,str(TH_NDVI)+'_'+str(TH_LUM)+'_'+'roof_4_lr.shp')) 
    logger.info('Greenery saved with hydra...')

def do_greenery(tile,roofs):
    lum_dataset = rasterio.open(tile.path_lum)
    ndvi_dataset = rasterio.open(tile.path_NDVI)

    lum_band = lum_dataset.read(1)
    ndvi_band = ndvi_dataset.read(1)

    with rasterio.open(tile.path_NDVI) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes_roof, nodata=-9999, all_touched=True, crop=False)

        mask = (ndvi_band >= TH_NDVI) & (lum_band <= TH_LUM) & (out_image[0]!=-9999)

        geoms = ((shape(s), v) for s, v in shapes(out_image[0], mask, transform=src.transform))
        gdf=gpd.GeoDataFrame(geoms, columns=['geometry', 'ndvi'])
        gdf.set_crs(crs=src.crs, inplace=True)

        green_roofs = gpd.sjoin(gdf, roofs, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')
        # green_roofs.to_file(os.path.join(OUTPUT_DIR,'green_roofs.shp')) 

        # filter for roof with at least 5 m2 and 3 m height by EGID and threshold on max NDVI to get rid of hovering vegetation
        green_roofs_egid = green_roofs.dissolve(by='EGID', aggfunc={"ndvi": "max",})

        return green_roofs_egid

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
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['dev']

    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']

    ORTHO_DIR=cfg['ortho_directory']
    NDVI_DIR=cfg['ndvi_directory']
    LUM_DIR=cfg['lum_directory']
    RESULTS_DIR=cfg['results_directory']

    TILE_DELIMITATION=cfg['tile_delimitation']

    ROOFS_POLYGONS=cfg['roofs_gt']
    ROOFS_LAYER=cfg['roofs_layer']
    EGID_TRAIN_TEST=cfg['egid_train_test']

    TH_NDVI=cfg['th_ndvi']
    TH_LUM=cfg['th_lum']

    os.chdir(WORKING_DIR)

    logger.info('Linking path of images and corresponding NDVI and luminosity rasters...')

    tiles=gpd.read_file(TILE_DELIMITATION)

    tiles=fct_misc.get_ortho_tiles(tiles, ORTHO_DIR, NDVI_DIR, LUM_DIR)

    tiles['path_lum']=[os.path.join(LUM_DIR, tile_name + '_lum.tif') for tile_name in tiles.NAME.to_numpy()]
    tiles['path_NDVI']=[os.path.join(NDVI_DIR, tile_name + '_NDVI.tif') for tile_name in tiles.NAME.to_numpy()]


    logger.info('Loading roofs/ground truth...')

    roofs=gpd.read_file(ROOFS_POLYGONS)
    # roofs.drop(columns=['essence', 'diam_tronc'], inplace=True)
    roofs.rename(columns={'class':'cls'}, inplace=True)
    roofs['geometry'] = roofs.buffer(-0.5)
    roofs_egid = roofs.dissolve(by='EGID', aggfunc='first')

    logger.info('Extracting greenery from raster with thresholds...')

    with fiona.open(ROOFS_POLYGONS, "r") as shapefile:
        shapes_roof = [feature["geometry"] for feature in shapefile]

    print("Multithreading with joblib for statistics over beeches: ")
    num_cores = multiprocessing.cpu_count()
    print ("starting job on {} cores.".format(num_cores))  

    with tqdm_joblib(desc="Parallel greenery detection", total=tiles.shape[0]) as progress_bar:
        green_roofs_list = Parallel(n_jobs=num_cores, prefer="threads")(delayed(do_greenery)(tile,roofs) for tile in tiles.itertuples())

    green_roofs=gpd.GeoDataFrame()
    for row in green_roofs_list:
        green_roofs = pd.concat([green_roofs, row])

    logger.info('Join greenery on the roofs, dissolving and cleaning...')

    # filter for roof with at least 5 m2 by EGID and threshold on max NDVI to get rid of hovering vegetation
    green_roofs_egid = green_roofs.dissolve(by='EGID', aggfunc={"ndvi": "max",})
    green_roofs_egid.rename(columns={'ndvi':'ndvi_max'}, inplace=True)
    green_roofs_egid['area'] = green_roofs_egid.area
    #green_roofs_egid.to_file(os.path.join(OUTPUT_DIR,'green_roofs_egid.shp')) 
    green_roofs_egid = green_roofs_egid.loc[(green_roofs_egid['area']>5)]
    green_roofs_egid = green_roofs_egid.loc[(green_roofs_egid['ndvi_max']<0.8)]
    #green_roofs_egid.to_file(os.path.join(OUTPUT_DIR,'green_roofs_egid.shp')) 

    roofs_egid = pd.DataFrame(roofs_egid.drop(columns='geometry'))
    green_roofs_egid_att = green_roofs_egid.merge(roofs_egid,on=['EGID'])
    #green_roofs_egid_att.to_file(os.path.join(OUTPUT_DIR,str(TH_NDVI)+'_'+str(TH_LUM)+'_'+'roof_lr.shp')) 
    
    my_app()
   
    logger.info('Outputting detected potential green roofs number...')

    if not os.path.isfile(os.path.join(RESULTS_DIR,'recap_green.csv')):
        with open(os.path.join(RESULTS_DIR,'recap_green.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            row = ['TH_NDVI', 'TH_LUM', 'roofs_bare', 'roofs_green', 'green_roofs_bare', 'green_roofs_green']
            writer.writerow(row)

    row = [TH_NDVI, TH_LUM, sum(roofs_egid['veg_new']==0),sum(roofs_egid['veg_new']==1),sum(green_roofs_egid_att['veg_new']==0),sum(green_roofs_egid_att['veg_new']==1)]
    with open(os.path.join(RESULTS_DIR,'recap_green.csv'), 'a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    fct_misc.log_reg(green_roofs_egid_att, EGID_TRAIN_TEST, TH_NDVI, TH_LUM, WORKING_DIR)
    