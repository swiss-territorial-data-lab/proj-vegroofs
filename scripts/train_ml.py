import os, sys
import yaml
from loguru import logger
import argparse
import pickle

import geopandas as gpd
import pandas as pd
import numpy as np

import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

import csv

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc              

logger=fct_misc.format_logger(logger)


def compute_metrics(test_gt: gpd.GeoDataFrame, test_pred: np.ndarray, test_proba: np.ndarray,
                    cls: str, lbl: str, CLS_ML: str, MODEL_ML: str, best_rf: RandomForestClassifier ,
                    TH_NDVI: str, TH_LUM: str, WORKING_DIR: str, desc_col: list, STAT_DIR: str = None):
 
    logger.info('Testing and metric computation...')

    if CLS_ML == 'binary':
        test_pred_pd = pd.concat([pd.DataFrame(test_gt[['EGID',cls]]).reset_index(),
                                  pd.DataFrame(test_pred)], axis=1,ignore_index=True, sort=False
                                  ).rename(columns = {1:'EGID', 2:'veg_new',3:'pred'})
        test_pred_pd = pd.concat([test_pred_pd[['EGID','veg_new','pred']],
                                  pd.DataFrame(test_proba)],axis=1,ignore_index=True, sort=False
                                  ).rename(
                                      columns = {0: 'EGID',
                                                 1: 'veg_new',
                                                 2: 'pred',
                                                 3: 'proba_bare',
                                                 4: 'proba_veg'})
        test_pred_pd['diff'] = abs(test_pred_pd['veg_new']==test_pred_pd['pred'])
    elif CLS_ML == 'multi':
        test_pred_pd = pd.concat([pd.DataFrame(test_gt[['EGID',cls]]).reset_index(),
                                  pd.DataFrame(test_pred)], axis=1,ignore_index=True, sort=False
                                  ).rename(columns = {1:'EGID', 2:'class',3:'pred'})
        test_pred_pd = pd.concat([test_pred_pd,pd.DataFrame(test_proba)],
                                 axis=1,ignore_index=True, sort=False
                                 ).rename(columns = {1: 'EGID',
                                                     2: 'class',
                                                     3: 'pred',
                                                     4: 'proba_bare',
                                                     5: 'proba_terr',
                                                     6: 'proba_spon',
                                                     7: 'proba_ext',
                                                     8: 'proba_lawn',
                                                     9: 'proba_int'})
        test_pred_pd['diff'] =abs(test_pred_pd['class'] == test_pred_pd['pred'])

    test_gt_pred=test_gt.merge(test_pred_pd, on="EGID")
    test_gt_pred.to_file(os.path.join(WORKING_DIR, STAT_DIR)+'pred_'+CLS_ML+'_'+MODEL_ML+'.gpkg')
    cf = confusion_matrix(test_gt[cls],test_pred, labels=lbl)

    if CLS_ML == 'binary':
        tn, fp, fn, tp = confusion_matrix(test_gt[cls],test_pred).ravel()
        METRICS_CSV = 'metrics.csv'

        if not os.path.isfile(os.path.join(WORKING_DIR, STAT_DIR,METRICS_CSV)):
            with open(os.path.join(WORKING_DIR, STAT_DIR,METRICS_CSV), 'w', newline='') as file:
                writer = csv.writer(file)
                row = ['th_ndvi','th_lum','tn', 'fp', 'fn', 'tp','accuracy','balanced_accuracy','recall','f1-score','model','desc']
                writer.writerow(row)

        row = [TH_NDVI,TH_LUM, tn, fp, fn, tp, accuracy_score(test_gt[cls],test_pred), 
               balanced_accuracy_score(test_gt[cls],test_pred), recall_score(test_gt[cls],test_pred),
               f1_score(test_gt[cls],test_pred),
               str(best_rf).replace(' ', '').replace('\t', '').replace('\n', ''),str(desc_col).replace(' ', '').replace('\t', '').replace('\n', '')]
        with open(os.path.join(WORKING_DIR, STAT_DIR,METRICS_CSV), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    else:
        (pd.DataFrame(cf)).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'cf_'+CLS_ML+'_'+MODEL_ML+'.csv')
        METRICS_CSV = 'metrics_multi.csv'

        if not os.path.isfile(os.path.join(WORKING_DIR, STAT_DIR,METRICS_CSV)):
            with open(os.path.join(WORKING_DIR, STAT_DIR,METRICS_CSV), 'w', newline='') as file:
                writer = csv.writer(file)
                row = ['th_ndvi','th_lum','accuracy','balanced_accuracy','classes','model','desc']
                writer.writerow(row)

        row = [TH_NDVI,TH_LUM, accuracy_score(test_gt[cls],test_pred), balanced_accuracy_score(test_gt[cls],test_pred), CLS_ML,
               str(best_rf).replace(' ', '').replace('\t', '').replace('\n', ''), str(desc_col).replace(' ', '').replace('\t', '').replace('\n', '')]
        with open(os.path.join(WORKING_DIR, STAT_DIR,METRICS_CSV), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

def train_ml(roofs_gt: gpd.GeoDataFrame, GREEN_TAG: str, GREEN_CLS: str, CLS_ML: str, 
             MODEL_ML: str, TRAIN_TEST: str, TH_NDVI: str, TH_LUM: str, 
             WORKING_DIR: str, STAT_DIR: str = None):
    
    egid_train_test = pd.read_csv(os.path.join(STAT_DIR,TRAIN_TEST))
    egid_train_test = egid_train_test[['EGID', 'train']]
    roofs_gt = roofs_gt.merge(egid_train_test, on='EGID')

    if CLS_ML == 'binary':
        cls = GREEN_TAG
        lbl = [0,1]
    elif CLS_ML == 'multi':
        cls = GREEN_CLS
        lbl = ['b','t','s','e','l','i']
    

    # Read descriptors from roof_stats.py outputs
    desc = pd.read_csv(os.path.join(WORKING_DIR, STAT_DIR, 'roof_stats.csv'))
    desc_col = ['min','max','mean','median','std']
    desc_col_egid = desc_col[:]
    desc_col_egid.append('EGID')
    desc_ndvi = desc[desc['band']=='ndvi']
    roofs_gt = roofs_gt.merge(desc_ndvi[desc_col_egid], on='EGID')
    desc_tmp = desc[desc['band']=='lum']
    roofs_gt = roofs_gt.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_lum'))
    desc_tmp = desc[desc['band']=='red']
    roofs_gt = roofs_gt.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_r'))
    desc_tmp = desc[desc['band']=='blue']
    roofs_gt = roofs_gt.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_b'))
    desc_tmp = desc[desc['band']=='green']
    roofs_gt = roofs_gt.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_g'))
    desc_tmp = desc[desc['band']=='nir']
    roofs_gt = roofs_gt.merge(desc_tmp[desc_col_egid], on='EGID', suffixes=('', '_nir'))

    desc_col = ['min','max','mean','median','std','min_lum','max_lum','mean_lum','median_lum','std_lum',
                'min_r','max_r','mean_r','median_r','std_r','min_b','max_b','mean_b','median_b','std_b',
                'min_g','max_g','mean_g','median_g','std_g','min_nir','max_nir','mean_nir','median_nir','std_nir'] 
    roofs_gt = roofs_gt.dropna(axis=0,subset=desc_col)
    if not 'green' in STAT_DIR:
        roofs_gt = roofs_gt.loc[((roofs_gt['mean']<0.05) & (roofs_gt[cls]==0)) | (roofs_gt[cls]==1)]
        ids = list(roofs_gt['EGID'])
        with open(os.path.join(WORKING_DIR, STAT_DIR,f"ids.pkl"),'wb') as f:
            pickle.dump(ids,f)
    else:
        with open(os.path.join(WORKING_DIR, STAT_DIR,f"ids.pkl"), 'rb') as f:
            ids = pickle.load(f)
            roofs_gt = roofs_gt[roofs_gt['EGID'].isin(ids)]
    ml_train = roofs_gt.loc[(roofs_gt['train']==1) ]
    ml_test = roofs_gt.loc[(roofs_gt['train']==0)]
    
    logger.info(f"Training model: {'logisitic regression' if MODEL_ML == 'LR' else 'random forest'}...")
    random.seed(10)
    
    if MODEL_ML == 'LR':
        param = {'penalty':['l2'],'solver':('liblinear','newton-cg'),
                 'C':[1,0.5,0.1],'max_iter':[200, 500, 800]}
        model = LogisticRegression(class_weight='balanced', random_state=0)
    if MODEL_ML == 'RF':
        n = round(np.sqrt(len(desc_col))) 
        param = {'n_estimators':[200,500, 800],'max_features':[n-1, n, n+1]}
        model = RandomForestClassifier(random_state=0, class_weight='balanced')

    clf = GridSearchCV(model, param, scoring='balanced_accuracy')
    clf.fit(ml_train[desc_col], ml_train[cls])
    with open(os.path.join(WORKING_DIR, STAT_DIR,f"model_{CLS_ML}_{MODEL_ML}.pkl"),'wb') as f:
        pickle.dump(clf,f)

    pd_fit=pd.DataFrame(clf.cv_results_)
    pd_fit.to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'fits_'+CLS_ML+'_'+MODEL_ML+'.csv')
    
    best_rf = clf.best_estimator_
    test_pred= best_rf.predict(ml_test[desc_col])
    test_proba = best_rf.predict_proba(ml_test[desc_col])

    if CLS_ML == 'binary':
        if MODEL_ML == 'RF':
            (pd.DataFrame(best_rf.feature_names_in_,best_rf.feature_importances_)
             ).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'imp_'+CLS_ML+'_'+MODEL_ML+'.csv')

        if MODEL_ML == 'LR':
            model_fi = permutation_importance(best_rf,ml_train[desc_col], ml_train[cls])
            pd.DataFrame(best_rf.feature_names_in_,model_fi['importances_mean']
                         ).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'imp_'+CLS_ML+'_'+MODEL_ML+'.csv')
    else:
        if MODEL_ML == 'RF':
            (pd.DataFrame(best_rf.feature_names_in_,best_rf.feature_importances_)
             ).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'imp_'+CLS_ML+'_'+MODEL_ML+'.csv')

        if MODEL_ML == 'LR':
            model_fi = permutation_importance(best_rf,ml_train[desc_col], ml_train[cls])
            pd.DataFrame(best_rf.feature_names_in_,model_fi['importances_mean']
                         ).to_csv(os.path.join(WORKING_DIR, STAT_DIR)+'imp_'+CLS_ML+'_'+MODEL_ML+'.csv')


    compute_metrics(ml_test, test_pred, test_proba, cls, lbl, CLS_ML, MODEL_ML,
                    best_rf,TH_NDVI, TH_LUM, WORKING_DIR, desc_col, STAT_DIR)


if __name__ == "__main__":

    logger.info('Starting...')

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="The script trains a logistic regression or a random forest with the statistics per roof.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['prod']


    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']
    STAT_DIR=cfg['results_directory']

    ROOFS=cfg['roofs_file']
    ROOFS_LAYER=cfg['roofs_layer']
    GREEN_TAG=cfg['green_tag']
    GREEN_CLS=cfg['green_cls']
    CLS_ML=cfg['cls_ml']
    MODEL_ML=cfg['model_ml']
    TRAIN_TEST=cfg['egid_train_test']

    TH_NDVI=cfg['th_ndvi']
    TH_LUM=cfg['th_lum']

    os.chdir(WORKING_DIR)

    roofs_gt = gpd.read_file(ROOFS, layer = ROOFS_LAYER)

    train_ml(roofs_gt, GREEN_TAG, GREEN_CLS, CLS_ML, MODEL_ML,TRAIN_TEST, TH_NDVI, TH_LUM, WORKING_DIR, STAT_DIR)
