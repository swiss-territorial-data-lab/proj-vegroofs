import os, sys
import yaml
import warnings
from tqdm import tqdm
from loguru import logger

import pandas as pd
import geopandas as gpd
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
import statistics
from rasterstats import zonal_stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

import csv
from csv import writer

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc

logger=fct_misc.format_logger(logger)
#warnings.filterwarnings("ignore", message="*")


logger.info('Starting...')

logger.info(f"Using config.yaml as config file.")
with open('config/logReg.yml') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['logistic_regression.py']

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

TH_NDVI=INPUTS['th_ndvi']
TH_LUM=INPUTS['th_lum']

os.chdir(WORKING_DIR)


## calculate NDVI-> calculate_raster.py

## calculate luminosity index R+G+B -> calculate_raster.py


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
roofs_egid = roofs.dissolve(by='EGID', aggfunc='first')


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

logger.info('Make a decision about a treshold to keep - TODO...')


logger.info('Extracting greenery from raster with thresholds...')

with fiona.open(ROOFS_POLYGONS, "r") as shapefile:
    shapes_roof = [feature["geometry"] for feature in shapefile]

greenery = gpd.GeoDataFrame()
for tile in tqdm(tiles.itertuples(), desc='Thresholding on raster values', total=tiles.shape[0]):

    lum_dataset = rasterio.open(tile.path_lum)
    ndvi_dataset = rasterio.open(tile.path_NDVI)

    lum_band = lum_dataset.read(1)
    ndvi_band = ndvi_dataset.read(1)

    with rasterio.open(tile.path_NDVI) as src:
        image=src.read(1)

        out_image, out_transform = rasterio.mask.mask(src, shapes_roof, nodata=10, all_touched=True, crop=False)
        out_meta = src.meta
            
        # # update metadata to ensure every image has the desired sidelength
        # out_meta=src.meta.copy() # copy the metadata of the source
        # out_meta.update({
        #     "height": out_image.shape[1],
        #     "width": out_image.shape[2],
        #     "driver": "Gtiff",
        #     "transform": out_transform
        # })

        # with rasterio.open('test.tif', 'w', **out_meta) as dst:
        #     dst.write(out_image)

        mask = (ndvi_band >= TH_NDVI) & (lum_band <= TH_LUM) & (out_image[0]!=10)


        geoms = ((shape(s), v) for s, v in shapes(out_image[0], mask, transform=src.transform))
        gdf=gpd.GeoDataFrame(geoms, columns=['geometry', 'ndvi'])
        gdf.set_crs(crs=src.crs, inplace=True)

    greenery = pd.concat([greenery, gdf])

# greenery.to_file(os.path.join(OUTPUT_DIR,'greenery.gpkg')) 
# greenery=gpd.read_file(os.path.join(OUTPUT_DIR,'greenery.gpkg'))


logger.info('Join greenery on the roofs, dissolving and cleaning...')

#clipped_beeches=fct_misc.clip_labels(correct_high_beeches, tiles_merge) # ??? think of dissolving vectors. 
green_roofs = gpd.sjoin(greenery, roofs, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')
# green_roofs.to_file(os.path.join(OUTPUT_DIR,'green_roofs.shp')) 

# filter for roof with at least 5 m2 and 3 m height by EGID and threshold on max NDVI to get rid of hovering vegetation
green_roofs_egid = green_roofs.dissolve(by='EGID', aggfunc={"ndvi": "max",})
green_roofs_egid.rename(columns={'ndvi':'ndvi_max'}, inplace=True)
green_roofs_egid['area'] = green_roofs_egid.area
#green_roofs_egid.to_file(os.path.join(OUTPUT_DIR,'green_roofs_egid.shp')) 
green_roofs_egid = green_roofs_egid.loc[(green_roofs_egid['area']>5)]
green_roofs_egid = green_roofs_egid.loc[(green_roofs_egid['ndvi_max']<0.8)]
#green_roofs_egid.to_file(os.path.join(OUTPUT_DIR,'green_roofs_egid.shp')) 

green_roofs_egid_att = roofs_egid.merge(green_roofs_egid,on=['EGID'])


logger.info('Outputting detected potential green roofs number...')

if not os.path.isfile('recap_green.csv'):
    with open('recap_green.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        row = ['TH_NDVI', 'TH_LUM', 'roofs_bare', 'roofs_green', 'green_roofs_bare', 'green_roofs_green']
        writer.writerow(row)

row = [TH_NDVI, TH_LUM, sum(roofs_egid['veg_new']==0),sum(roofs_egid['veg_new']==1),sum(green_roofs_egid_att['veg_new']==0),sum(green_roofs_egid_att['veg_new']==1)]
with open('recap_green.csv', 'a',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(row)


logger.info('Think a bit about the threshold to use and how to optimize the process - TODO')

logger.info('Partitioning of the potential green roofs in train and test dataset...')
lg_train, lg_test = train_test_split(green_roofs_egid_att, test_size=0.3, train_size=0.7, random_state=0, shuffle=True, stratify=green_roofs_egid_att['veg_new'])


logger.info('Training the logisitic regression...')

# generalized linear model logistic regression P = log (P/(1-P)) = a + beta_1*NDVImax + beta_2*Area_vege + beta_3*NDVI_max/area_vege
# generalized linear model logistic regression P = log (P/(1-P)) = a + beta_1*lg_train['ndvi']+ beta_2*lg_train['area'] + beta_3*lg_train['ndvi']:lg_train['area']
clf = LogisticRegression(random_state=0).fit(lg_train[['ndvi_max','area']], lg_train['veg_new'])
test_pred= clf.predict(lg_test[['ndvi_max','area']])


logger.info('Testin and metric computation...')

cf = confusion_matrix(lg_test['veg_new'],test_pred)
tn, fp, fn, tp = confusion_matrix(lg_test['veg_new'],test_pred).ravel()

if not os.path.isfile('metrics.csv'):
    with open('metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        row = ['tn', 'fp', 'fn', 'tp''accuracy','recall','f1-score']
        writer.writerow(row)

row = [tn, fp, fn, tp, accuracy_score(lg_test['veg_new'],test_pred),recall_score(lg_test['veg_new'],test_pred),f1_score(lg_test['veg_new'],test_pred)]
with open('metrics.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(row)
