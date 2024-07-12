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
    green_roofs_egid_att.to_file(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,str(TH_NDVI)+'_'+str(TH_LUM)+'_'+'green_roofs.shp')) 
    roofs_egid_green.to_file(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,str(TH_NDVI)+'_'+str(TH_LUM)+'_'+'roofs_green.shp')) 

    logger.info(f"Greenery and roofs saved with hydra in {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

def do_greenery(tile,roofs):
    lum_dataset = rasterio.open(tile.path_lum)
    ndvi_dataset = rasterio.open(tile.path_NDVI)

    lum_band = lum_dataset.read(1)
    ndvi_band = ndvi_dataset.read(1)

    with rasterio.open(tile.path_NDVI) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes_roof, nodata=-9999, all_touched=True, crop=False)

        mask = (ndvi_band >= TH_NDVI) & (out_image[0]!=-9999) & (lum_band <= TH_LUM)

        geoms = ((shape(s), v) for s, v in shapes(out_image[0], mask, transform=src.transform))
        gdf=gpd.GeoDataFrame(geoms, columns=['geometry', 'ndvi'])
        gdf.set_crs(crs=src.crs, inplace=True)

        green_roofs = gpd.sjoin(gdf, roofs, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')

        green_roofs_egid = green_roofs.dissolve(by='EGID', aggfunc={"ndvi": "max",})
        green_roofs_egid['EGID']=green_roofs_egid.index
        green_roofs_egid.index.names = ['Index']

        return green_roofs_egid

if __name__ == "__main__":
         
    logger.info('Starting...')

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="The script detects potential greenery on roofs.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['dev']

    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']

    ORTHO_DIR=cfg['ortho_directory']
    NDVI_DIR=cfg['ndvi_directory']
    LUM_DIR=cfg['lum_directory']
    RESULTS_DIR=cfg['results_directory']

    TILE_DELIMITATION=cfg['tile_delimitation']

    ROOFS_POLYGONS=cfg['roofs_file']
    ROOFS_LAYER=cfg['roofs_layer']
    GT = cfg['gt']
    GREEN_TAG=cfg['green_tag']
    GREEN_CLS=cfg['green_cls']
    CHM_LAYER=cfg['chm_layer']

    TH_NDVI=cfg['th_ndvi']
    TH_LUM=cfg['th_lum']
    EPSG=cfg['epsg']

    os.chdir(WORKING_DIR)

    _=fct_misc.ensure_dir_exists(RESULTS_DIR)


    logger.info('Linking path of images to corresponding NDVI and luminosity rasters...')

    fct_misc.generate_extent(ORTHO_DIR, TILE_DELIMITATION, EPSG)
    tiles=gpd.read_file(TILE_DELIMITATION)

    tiles=fct_misc.get_ortho_tiles(tiles, ORTHO_DIR, NDVI_DIR, LUM_DIR)

    tiles['path_lum']=[os.path.join(LUM_DIR, tile_name + '_lum.tif') for tile_name in tiles.NAME.to_numpy()]
    tiles['path_NDVI']=[os.path.join(NDVI_DIR, tile_name + '_NDVI.tif') for tile_name in tiles.NAME.to_numpy()]


    logger.info('Loading roofs/ground truth...')

    roofs=gpd.read_file(ROOFS_POLYGONS, layer=ROOFS_LAYER)
    if GT: 
        roofs.rename(columns={GREEN_CLS:'cls'}, inplace=True)
    roofs['geometry'] = roofs.buffer(-0.5)
    roofs_egid = roofs.dissolve(by='EGID', aggfunc='first')
    roofs_egid['area']=roofs_egid.area
    roofs_egid['EGID']=roofs_egid.index
    roofs_egid.index.names = ['Index']

    logger.info('Extracting greenery from rasters with thresholds...')

    with fiona.open(ROOFS_POLYGONS, "r") as shapefile:
        shapes_roof = [feature["geometry"] for feature in shapefile]

    logger.info("Multithreading with joblib for statistics over roofs... ")
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Starting job on {num_cores} cores...")  

    with tqdm_joblib(desc="Parallel greenery detection", total=tiles.shape[0]) as progress_bar:
        green_roofs_list = Parallel(n_jobs=num_cores, prefer="threads")(delayed(do_greenery)(tile,roofs) for tile in tiles.itertuples())

    green_roofs=gpd.GeoDataFrame()
    for row in green_roofs_list:
        green_roofs = pd.concat([green_roofs, row])

    green_roofs_egid = green_roofs.dissolve(by='EGID', aggfunc={"ndvi": "max",})
    green_roofs_egid.rename(columns={'ndvi':'ndvi_max'}, inplace=True)
    green_roofs_egid['EGID']=green_roofs_egid.index
    green_roofs_egid.index.names = ['Index']

    logger.info('Filtering for overhanging vegetation...')

    CHM = os.path.join(WORKING_DIR, CHM_LAYER)
    CHM_GPD=gpd.read_file(CHM)
    CHM_GPD['geometry'] = CHM_GPD.buffer(1)
    green_roofs_egid=gpd.overlay(green_roofs_egid, CHM_GPD, how='difference')
    green_roofs_egid['area_green'] = green_roofs_egid.area

    
    logger.info('Join greenery on the roofs and vice-versa, saving...')

    roofs_egid_pd = pd.DataFrame(roofs_egid.drop(columns='geometry'))
    green_roofs_egid_att = green_roofs_egid.merge(roofs_egid_pd,on=['EGID'])
    green_roofs_egid_att['EGID']=green_roofs_egid_att.index
    green_roofs_egid_att.index.names = ['Index']
    green_roofs_egid_att['area_ratio'] = green_roofs_egid_att['area_green']/green_roofs_egid_att['area']

    green_roofs_egid_pd = pd.DataFrame(green_roofs_egid.drop(columns='geometry'))
    roofs_egid_green = roofs_egid.merge(green_roofs_egid_pd,on=['EGID'], how='outer')
    roofs_egid_green['EGID']=roofs_egid_green.index
    roofs_egid_green.index.names = ['Index']
    roofs_egid_green['area_green'] = roofs_egid_green['area_green'].fillna(0)
    roofs_egid_green['area_ratio'] = roofs_egid_green['area_green']/roofs_egid_green['area']

    my_app()
   
    if GT:

        logger.info('Outputting detected potential green roofs number...')

        ROOF_COUNTS_CSV = 'recap_green.csv'

        if not os.path.isfile(os.path.join(RESULTS_DIR,ROOF_COUNTS_CSV)):
            with open(os.path.join(RESULTS_DIR,ROOF_COUNTS_CSV), 'w', newline='') as file:
                writer = csv.writer(file)
                row = ['TH_NDVI', 'TH_LUM', 'roofs_bare', 'roofs_green', 'green_roofs_bare', 'green_roofs_green']
                writer.writerow(row)

        row = [TH_NDVI, TH_LUM, sum(roofs_egid[GREEN_TAG]==0),sum(roofs_egid[GREEN_TAG]==1),sum(green_roofs_egid_att[GREEN_TAG]==0),sum(green_roofs_egid_att[GREEN_TAG]==1)]
        with open(os.path.join(RESULTS_DIR,ROOF_COUNTS_CSV), 'a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
