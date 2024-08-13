import os, sys
import yaml
from loguru import logger
import argparse
import pickle

import geopandas as gpd
import pandas as pd

import csv
from csv import writer

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc              

logger=fct_misc.format_logger(logger)

def infer_ml(roofs: gpd.GeoDataFrame, CLS_ML: str, MODEL_ML: str, WORKING_DIR: str, PICKLE_DIR: str, STAT_DIR: str = None, ):
    with open(os.path.join(WORKING_DIR, PICKLE_DIR,f"model_{CLS_ML}_{MODEL_ML}.pkl"), 'rb') as f:
        clf = pickle.load(f)   

    # Read descriptors from roof_stats.py outputs
    desc = pd.read_csv(os.path.join(WORKING_DIR, STAT_DIR, 'roof_stats.csv'))
    desc_col = ['min','max','mean','median','std']
    desc_col_egid = desc_col[:]
    desc_col_egid.append('EGID')
    desc_ndvi = desc[desc['band']=='ndvi']
    roofs = roofs.merge(desc_ndvi[desc_col_egid], on='EGID')
    roofs = roofs.dropna(axis=0,subset=desc_col_egid)
    desc_tmp = desc[desc['band']=='lum']
    roofs = roofs.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_lum'))
    desc_tmp = desc[desc['band']=='red']
    roofs = roofs.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_r'))
    desc_tmp = desc[desc['band']=='blue']
    roofs = roofs.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_b'))
    desc_tmp = desc[desc['band']=='green']
    roofs = roofs.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_g'))
    desc_tmp = desc[desc['band']=='nir']
    roofs = roofs.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_nir'))

    desc_col = ['min','max','mean','median','std','min_lum','max_lum','mean_lum','median_lum','std_lum',
                'min_r','max_r','mean_r','median_r','std_r','min_b','max_b','mean_b','median_b','std_b',
                'min_g','max_g','mean_g','median_g','std_g','min_nir','max_nir','mean_nir','median_nir','std_nir'] #,'area_ratio']
    
    logger.info('Predicting ...')

    roofs_pred= clf.best_estimator_.predict(roofs[desc_col])
    roofs_proba = clf.best_estimator_.predict_proba(roofs[desc_col])   

    roofs=roofs.drop(columns=desc_col)
    roofs_pred_pd = pd.concat([pd.DataFrame(roofs[['EGID']]).reset_index(),pd.DataFrame(roofs_pred)], axis=1,ignore_index=True, sort=False).rename(columns = {1:'EGID', 2:'pred'})
    if CLS_ML == 'binary':   
        roofs_pred_pd = pd.concat([roofs_pred_pd[['EGID','pred']],pd.DataFrame(roofs_proba)],axis=1,ignore_index=True, sort=False).rename(columns = {0:'EGID', 1:'pred',2:'proba_bare',3:'proba_veg'})
    elif CLS_ML == 'multi':
        roofs_pred_pd = pd.concat([roofs_pred_pd[['EGID','pred']],pd.DataFrame(roofs_proba)],axis=1,ignore_index=True, sort=False).rename(columns = {0:'EGID',1:'pred',2:'proba_bare',3:'proba_terr',4:'proba_spon',5:'proba_ext',6:'proba_lawn',7:'proba_int'})
    roofs=roofs.merge(roofs_pred_pd, on="EGID")
    if 'fid' in roofs.columns:
        roofs['fid'] = roofs['fid'].astype(int)

    roofs.to_file(os.path.join(WORKING_DIR, STAT_DIR+'inf_'+CLS_ML+'_'+MODEL_ML+'.gpkg'))

if __name__ == "__main__":

    logger.info('Starting...')

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="The script infere on land survey building footprint with a trained logistic regression or a random forest.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['dev']


    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']
    STAT_DIR=cfg['results_directory']

    ROOFS=cfg['roofs_file']
    ROOFS_LAYER=cfg['roofs_layer']
    CLS_ML=cfg['cls_ml']
    MODEL_ML=cfg['model_ml']
    PICKLE_DIR=cfg['trained_model_dir']

    os.chdir(WORKING_DIR)

    roofs = gpd.read_file(ROOFS, layer = ROOFS_LAYER)
    infer_ml(roofs,CLS_ML, MODEL_ML, WORKING_DIR, PICKLE_DIR, STAT_DIR)

    logger.info('Inferences finished.')
