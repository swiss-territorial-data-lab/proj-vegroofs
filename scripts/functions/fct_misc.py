import sys
import os
import tqdm as tqdm
from loguru import logger

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping, shape
from shapely.affinity import scale

import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.features import dataset_features

import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.inspection import permutation_importance

import csv
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

def test_crs(crs1: str, crs2 = "EPSG:2056"):
    '''
    Take the crs-string of two GeoDataframes and compare them. If they are not the same, stop the script.
    '''
    if isinstance(crs1, gpd.GeoDataFrame):
        crs1=crs1.crs
    if isinstance(crs2, gpd.GeoDataFrame):
        crs2=crs2.crs

    try:
        assert(crs1 == crs2), f"CRS mismatch between the two files ({crs1} vs {crs2})."
    except Exception as e:
        logger.error(e)
        sys.exit(1)

def ensure_dir_exists(dirpath: str):
    '''
    Test if a directory exists. If not, make it.

    return: the path to the verified directory.
    '''

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        logger.info(f"The directory {dirpath} was created.")

    return dirpath

def clip_labels(labels_gdf: gpd.GeoDataFrame, tiles_gdf: gpd.GeoDataFrame, predicate_sjoin: str, fact: int = 1):
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

    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate=predicate_sjoin)
    
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

def get_ortho_tiles(tiles: gpd.GeoDataFrame, FOLDER_PATH_IN: str, FOLDER_PATH_NDVI: str, FOLDER_PATH_LUM: str, WORKING_DIR: str = None):
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

    rgb_paths=[]
    ndvi_paths=[]
    lum_paths=[]

    for tile_name in tiles['NAME'].values:
        if '' in tile_name:     
            rgb_paths.append(os.path.join(FOLDER_PATH_IN, tile_name + '.tif'))
            ndvi_paths.append(os.path.join(FOLDER_PATH_NDVI, tile_name + '_NDVI.tif'))
            lum_paths.append(os.path.join(FOLDER_PATH_LUM, tile_name + '_lum.tif'))

        else:
            rgb_paths.append('')
            ndvi_paths.append('')  
            lum_paths.append('')
                    
    tiles['path_RGB']=rgb_paths
    tiles['path_NDVI']=ndvi_paths
    tiles['path_lum']=lum_paths

    tiles = tiles[tiles.path_RGB!='']

    return tiles


def generate_extent(PATH_IN: str, PATH_OUT: str, EPSG: str = 'epsg:2056'):
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
    #ensure_dir_exists(os.path.join(PATH_OUT,'tiles'))

    for path, subdirs, files in os.walk(PATH_IN):
        for name in files:
            if name.endswith(pattern):
                list_name.append(name)
    
    ext_merge=gpd.GeoDataFrame()
    for _name in list_name:

        _tif = os.path.join(PATH_IN, _name)
        logger.info(f'Computing extent of {str(_name)} ...')

        with rasterio.open(_tif) as src:
            gdf = gpd.GeoDataFrame.from_features(
                dataset_features(src, bidx=2, as_mask=True, geographic=False, band=False, with_nodata=True)
                )
            gdf = gdf.dissolve()
            if (str(src.crs)=='None'):
                gdf = gdf.set_crs(EPSG)
            else:
                gdf = gdf.set_crs(crs=src.crs, inplace=True)
            gdf.filename=_name.replace('.tif', '')
            gdf=gdf.drop(columns= ['val'])
            gdf=gdf.rename(columns={"filename":"NAME"})
            #gdf.to_file(os.path.join(PATH_OUT,'tiles',_name.replace('.tif', '.shp')))
            ext_merge = pd.concat([ext_merge, gpd.GeoDataFrame(gdf, index=[0])], ignore_index=True)
            src.close()

    ext_merge.to_file(os.path.join(PATH_OUT, 'extent.shp'))

def clip_im(TIFF_FOLDER: str, GPD: str, OUT_FOLDER: str, idx: int, EPSG: str = 'epsg:2056'):
    '''
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
    '''

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

def log_reg(roofs_lr: gpd.GeoDataFrame, CLS_LR: str, MODEL_ML: str, TRAIN_TEST: str, TH_NDVI: str, TH_LUM: str, WORKING_DIR: str, STAT_DIR: str = None):
    egid_train_test = pd.read_csv(os.path.join(STAT_DIR.split('/')[-2],TRAIN_TEST))
    egid_train_test = egid_train_test[['EGID', 'train']]
    roofs_lr = roofs_lr.merge(egid_train_test, on='EGID')

    if CLS_LR == 'binary':
        cls = 'veg_new_2'
        lbl = [0,1]
    elif CLS_LR == 'multi':
        cls = 'class_2'
        lbl = ['b','t','s','e','l','i'] 
    else :
        cls = 'cls_agg'
        lbl = ['bt','s','e','li']
    

    # Read descriptors from roof_stats.py outputs
    desc = pd.read_csv(os.path.join(WORKING_DIR, STAT_DIR, 'roof_stats.csv'))
    desc_col = ['min','max','mean','median','std']
    desc_col_egid = desc_col[:]
    desc_col_egid.append('EGID')
    desc_ndvi = desc[desc['band']=='ndvi']
    roofs_lr = roofs_lr.merge(desc_ndvi[desc_col_egid], on='EGID')
    roofs_lr = roofs_lr.dropna(axis=0,subset=desc_col_egid)
    desc_tmp = desc[desc['band']=='lum']
    roofs_lr = roofs_lr.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_lum'))
    desc_tmp = desc[desc['band']=='red']
    roofs_lr = roofs_lr.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_r'))
    desc_tmp = desc[desc['band']=='blue']
    roofs_lr = roofs_lr.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_b'))
    desc_tmp = desc[desc['band']=='green']
    roofs_lr = roofs_lr.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_g'))
    desc_tmp = desc[desc['band']=='nir']
    roofs_lr = roofs_lr.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_nir'))

    desc_col = ['min','max','mean','median','std','min_lum','max_lum','mean_lum','median_lum','std_lum',
                'min_r','max_r','mean_r','median_r','std_r','min_b','max_b','mean_b','median_b','std_b',
                'min_g','max_g','mean_g','median_g','std_g','min_nir','max_nir','mean_nir','median_nir','std_nir','area_ratio']#,'surface_ca']

    lg_train = roofs_lr.loc[(roofs_lr['train']==1)]
    lg_test = roofs_lr.loc[(roofs_lr['train']==0)]

    ## Cross-validation ZH-GE (ZH=unID=1-1446)
    # lg_train = roofs_lr.loc[(roofs_lr['unID']>1446)]
    # lg_test = roofs_lr.loc[(roofs_lr['unID']<=1446)]


    logger.info('Training the logisitic regression...')

    random.seed(10)
    #LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                        # class_weight='balanced', random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', 
                        # verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    # RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, 
                            # min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                            # bootstrap=True, oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False, class_weight='balanced', 
                            # ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
    # clf = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', random_state=0,solver='liblinear', 
    #                          max_iter=100).fit(lg_train[desc_col], lg_train[cls]).fit(lg_train[desc_col], lg_train[cls])
    # clf = RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced').fit(lg_train[desc_col], lg_train[cls])
    
    if MODEL_ML == 'LR':
        param = {'penalty':('l1', 'l2'),'solver':('liblinear','newton-cg'), 'C':[1,0.5,0.1],'max_iter':[100,150,200]}
        model = LogisticRegression(class_weight='balanced', random_state=0)
    if MODEL_ML == 'RF': 
        param = {'n_estimators':[100,150,200],'max_features':[5,6]}
        model = RandomForestClassifier(random_state=0, class_weight='balanced')

    clf = GridSearchCV(model, param)
    clf.fit(lg_train[desc_col], lg_train[cls])
    pd_fit=pd.DataFrame(clf.cv_results_)
    pd_fit.to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'fits_'+CLS_LR+'_'+MODEL_ML+'.csv')

    logger.info('Testing and metric computation...')

    test_pred= clf.best_estimator_.predict(lg_test[desc_col])
    test_proba = clf.best_estimator_.predict_proba(lg_test[desc_col])

    if CLS_LR == 'binary':
        test_pred_pd = pd.concat([pd.DataFrame(lg_test[['EGID','veg_new']]).reset_index(),pd.DataFrame(test_pred)], axis=1,ignore_index=True, sort=False).rename(columns = {1:'EGID', 2:'veg_new',3:'pred'})
        test_pred_pd = pd.concat([test_pred_pd,pd.DataFrame(test_proba)],axis=1,ignore_index=True, sort=False).rename(columns = {1:'EGID', 2:'veg_new',3:'pred',4:'proba_bare',5:'proba_veg'})
        test_pred_pd['diff'] = abs(test_pred_pd['veg_new']-test_pred_pd['pred'])
    elif CLS_LR == 'multi':
        test_pred_pd = pd.concat([pd.DataFrame(lg_test[['EGID','class']]).reset_index(),pd.DataFrame(test_pred)], axis=1,ignore_index=True, sort=False).rename(columns = {1:'EGID', 2:'class',3:'pred'})
        test_pred_pd = pd.concat([test_pred_pd,pd.DataFrame(test_proba)],axis=1,ignore_index=True, sort=False).rename(columns = {1:'EGID', 2:'class',3:'pred',4:'proba_bare',5:'proba_terr',6:'proba_spon',7:'proba_ext',8:'proba_lawn',9:'proba_int'})
        test_pred_pd['diff'] =abs(test_pred_pd['class'] == test_pred_pd['pred'])
    else: 
        test_pred_pd = pd.concat([pd.DataFrame(lg_test[['EGID','cls_agg']]).reset_index(),pd.DataFrame(test_pred)], axis=1,ignore_index=True, sort=False).rename(columns = {1:'EGID', 2:'class',3:'pred'})
        test_pred_pd = pd.concat([test_pred_pd,pd.DataFrame(test_proba)],axis=1,ignore_index=True, sort=False).rename(columns = {1:'EGID', 2:'class',3:'pred',4:'proba_bt',5:'proba_s',6:'proba_e',7:'proba_li'})
        test_pred_pd['diff'] =abs(test_pred_pd['class'] == test_pred_pd['pred'])

    test_pred_pd.to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'pred_'+CLS_LR+'_'+MODEL_ML+'.csv')
    cf = confusion_matrix(lg_test[cls],test_pred, labels=lbl)

    if CLS_LR == 'binary':
        tn, fp, fn, tp = confusion_matrix(lg_test[cls],test_pred).ravel()

        if MODEL_ML == 'RF':
            (pd.DataFrame(clf.best_estimator_.feature_names_in_,clf.best_estimator_.feature_importances_)).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'imp_'+CLS_LR+'_'+MODEL_ML+'.csv')

        if MODEL_ML == 'LR':
            model_fi = permutation_importance(clf.best_estimator_,lg_train[desc_col], lg_train[cls])
            pd.DataFrame(clf.best_estimator_.feature_names_in_,model_fi['importances_mean']).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'imp_'+CLS_LR+'_'+MODEL_ML+'.csv')

        if not os.path.isfile(os.path.join(WORKING_DIR, STAT_DIR,'metrics.csv')):
            with open(os.path.join(WORKING_DIR, STAT_DIR,'metrics.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                row = ['th_ndvi','th_lum','tn', 'fp', 'fn', 'tp','accuracy','recall','f1-score','model']
                writer.writerow(row)

        row = [TH_NDVI,TH_LUM, tn, fp, fn, tp, accuracy_score(lg_test[cls],test_pred),recall_score(lg_test[cls],test_pred),f1_score(lg_test[cls],test_pred),str(clf.best_estimator_).replace(' ', '').replace('\t', '').replace('\n', '')]
        with open(os.path.join(WORKING_DIR, STAT_DIR,'metrics.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    else:
        (pd.DataFrame(cf)).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'cf_'+CLS_LR+'_'+MODEL_ML+'.csv')

        if MODEL_ML == 'RF':
            (pd.DataFrame(clf.best_estimator_.feature_names_in_,clf.best_estimator_.feature_importances_)).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'imp_'+CLS_LR+'_'+MODEL_ML+'.csv')

        if MODEL_ML == 'LR':
            model_fi = permutation_importance(clf.best_estimator_,lg_train[desc_col], lg_train[cls])
            pd.DataFrame(clf.best_estimator_.feature_names_in_,model_fi['importances_mean']).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'imp_'+CLS_LR+'_'+MODEL_ML+'.csv')

        if not os.path.isfile(os.path.join(WORKING_DIR, STAT_DIR,'metrics_multi.csv')):
            with open(os.path.join(WORKING_DIR, STAT_DIR,'metrics_multi.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                row = ['th_ndvi','th_lum','accuracy','classes','model']
                writer.writerow(row)

        row = [TH_NDVI,TH_LUM, accuracy_score(lg_test[cls],test_pred),CLS_LR, str(clf.best_estimator_).replace(' ', '').replace('\t', '').replace('\n', '')]
        with open(os.path.join(WORKING_DIR, STAT_DIR,'metrics_multi.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)       

    logger.info('BIS !')