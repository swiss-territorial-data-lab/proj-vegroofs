import sys
import os
import tqdm as tqdm

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping, shape
from shapely.affinity import scale

import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.features import dataset_features

import numpy as np
from loguru import logger

import warnings
warnings.filterwarnings('ignore')

def format_logger(logger):
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")
    return logger

def test_crs(crs1, crs2 = "EPSG:2056"):
    '''
    Take the crs of two dataframes and compare them. If they are not the same, stop the script.
    '''
    if isinstance(crs1, gpd.GeoDataFrame):
        crs1=crs1.crs
    if isinstance(crs2, gpd.GeoDataFrame):
        crs2=crs2.crs

    try:
        assert(crs1 == crs2), f"CRS mismatch between the two files ({crs1} vs {crs2})."
    except Exception as e:
        print(e)
        sys.exit(1)

def ensure_dir_exists(dirpath):
    '''
    Test if a directory exists. If not, make it.

    return: the path to the verified directory.
    '''

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"The directory {dirpath} was created.")

    return dirpath

def clip_labels(labels_gdf, tiles_gdf, fact=1):
    '''
    Clip the labels to the tiles
    Copied from the misc functions of the object detector 
    cf. https://github.com/swiss-territorial-data-lab/object-detector/blob/master/helpers/misc.py

    - labels_gdf: geodataframe with the labels
    - tiles_gdf: geodataframe of the tiles
    - fact: factor to scale the tiles before clipping
    return: a geodataframe with the labels clipped to the tiles
    '''

    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']
        
    #assert(labels_gdf.crs.name == tiles_gdf.crs.name)

    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate='within')
    
    def clip_row(row, fact=fact):
        
        old_geo = row.geometry
        scaled_tile_geo = scale(row.tile_geometry, xfact=fact, yfact=fact)
        new_geo = old_geo.intersection(scaled_tile_geo)
        row['geometry'] = new_geo

        return row

    clipped_labels_gdf = labels_tiles_sjoined_gdf.apply(lambda row: clip_row(row, fact), axis=1)
    clipped_labels_gdf.crs = labels_gdf.crs

    clipped_labels_gdf.drop(columns=['tile_geometry', 'index_right'], inplace=True)
    clipped_labels_gdf.rename(columns={'id': 'tile_id'}, inplace=True)

    return clipped_labels_gdf

def get_ortho_tiles(tiles, FOLDER_PATH_IN, FOLDER_PATH_OUT, WORKING_DIR=None):
    '''
    Get the true orthorectified tiles and the corresponding NDVI file based on the tile name.

    - tiles: dataframe of with the delimitation and the id of the file.
    - PATH_ORIGINAL: path to the original tiles
    - PATH_NDVI: path to the NDVI tiles
    - WORKING_DIR: working directory to be set (if needed)
    return: the tile dataframe with an additional field with the path to each file.
    '''

    if WORKING_DIR:
        os.chdir(WORKING_DIR)

    rgb_pathes=[]
    ndvi_pathes=[]

    for tile_name in tiles['NAME'].values:
                                       
        rgb_pathes.append(os.path.join(FOLDER_PATH_IN, tile_name + '.tif'))
        ndvi_pathes.append(os.path.join(FOLDER_PATH_OUT, tile_name + '_NDVI.tif'))

               
    tiles['path_RGB']=rgb_pathes
    tiles['path_NDVI']=ndvi_pathes

    return tiles


def generate_extent(PATH_IN, PATH_OUT, EPSG):
    '''
    Generate the per-tile oriented extent and the aggregated extent of input tiles. 

    - PATH_IN: path to the original tiles
    - PATH_OUT: path to the output extent layers
    - EPSG: epsg used for the project to use when data miss one. 
    return: the per-tile oriented extent and the aggregated extent of input tiles with original tile name as attribute. 
    '''

    pattern = ".tif"
    list_name = []
    ensure_dir_exists(PATH_OUT)
    ensure_dir_exists(os.path.join(PATH_OUT,'tiles'))

    for path, subdirs, files in os.walk(PATH_IN):
        for name in files:
            if name.endswith(pattern):
                list_name.append(name)
    
    ext_merge=gpd.GeoDataFrame()
    for _name in list_name:

        _tif = os.path.join(PATH_IN, _name)
        logger.info('Computing extent of ' + str(_name) + '...')

        with rasterio.open(_tif) as src:
            gdf = gpd.GeoDataFrame.from_features(dataset_features(src, bidx=2, as_mask=True, geographic=False, band=False))
            if (str(src.crs)=='None'):
                gdf = gdf.set_crs(EPSG)
            else:
                gdf = gdf.set_crs(crs=src.crs, inplace=True)
            gdf.filename=_name.replace('.tif', '')
            gdf=gdf.drop(columns= ['val'])
            gdf=gdf.rename(columns={"filename":"NAME"})
            gdf.to_file(os.path.join(PATH_OUT,'tiles',_name.replace('.tif', '.shp')))
            ext_merge = pd.concat([ext_merge, gpd.GeoDataFrame(gdf, index=[0])], ignore_index=True)
            src.close()

    ext_merge.to_file(os.path.join(PATH_OUT, 'extent.shp'))

def clip_im(TIFF_FOLDER, GPD, OUT_FOLDER, idx, EPSG):

    """
    Clip TIFF images with a shape file. 
    This function clips tiff images based on a provided shapefile. The processed cells are saved as individual tiff images.

    Parameters:
    - TIFF_FOLDER (str): Path to the folder containing input tiff images.
    - GPD (str): shapefile.
    - OUT_FOLDER (str): Path to the output folder for processed grid cell tiff images.
    - idx for naming.
    - EPSG: epsg used for the project to use when data miss one. 

    Returns:
    - None
    """

    with rasterio.open(os.path.join(TIFF_FOLDER, GPD.iloc[-1]['NAME']+'.tif')) as src:

        logger.info('Clipping ' + GPD.iloc[-1]['NAME'] + '.tif...')
        
        out_image, out_transform = rasterio.mask.mask(
            src,
            GPD['geometry'],
            all_touched=True,
            crop=True,
            filled=True)
            
        if (str(src.crs)=='None'):
            crs = EPSG
        else:
            crs = src.crs
        # update metadata to ensure every image has the desired sidelength
        out_meta=src.meta.copy() # copy the metadata of the source
        out_meta.update({
            "crs": crs,
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "driver": "Gtiff",
            "transform": out_transform,
            "nodata": 0
        })

        # output name is input name with added id
        out_path = os.path.join(OUT_FOLDER, GPD.iloc[-1]['NAME']+'_'+str(idx)+'_clip.tif')
        if os.path.exists(out_path):
            os.remove(out_path)

        with rasterio.open(out_path, 'w', **out_meta) as dst:
            dst.write(out_image)
        
        logger.info('Clipped image ' + GPD.iloc[-1]['NAME']+'_'+str(idx) + ' written...')

