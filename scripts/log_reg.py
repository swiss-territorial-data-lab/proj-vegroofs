# Est-ce qu'on veut avoir le souvenir du threshold jusqu'ici ? 

import os, sys
import yaml
from loguru import logger
import argparse

import geopandas as gpd

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


if __name__ == "__main__":

    logger.info('Starting...')

    logger.info('Parsing the config file...')

    parser = argparse.ArgumentParser(
        description="The script trains a logistic regression on potential green roofs")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['prod']


    logger.info('Defining constants...')

    WORKING_DIR=cfg['working_directory']

    ROOFS_LR=cfg['roofs_lr']
    ROOFS_LAYER=cfg['roofs_layer']

    OUTPUT_DIR=cfg['lr_directory']

    TH_NDVI=cfg['th_ndvi']
    TH_LUM=cfg['th_lum']

    os.chdir(WORKING_DIR)


    logger.info('Loading roofs for logistic regression...')
    roofs_lr=gpd.read_file(ROOFS_LR) #saved in hydra file architecture
    #roofs_lr['veg_new'].isnull().values.any()
    roofs_lr['veg_new'].fillna(0, inplace = True)

    logger.info('Partitioning of the potential green roofs in train and test dataset...')
    lg_train, lg_test = train_test_split(roofs_lr, test_size=0.3, train_size=0.7, random_state=0, shuffle=True, stratify=roofs_lr['veg_new'])

    logger.info('Training the logisitic regression...')

    # generalized linear model logistic regression P = log (P/(1-P)) = a + beta_1*NDVImax + beta_2*Area_vege + beta_3*NDVI_max/area_vege
    # generalized linear model logistic regression P = log (P/(1-P)) = a + beta_1*lg_train['ndvi']+ beta_2*lg_train['area'] + beta_3*lg_train['ndvi']:lg_train['area']
    clf = LogisticRegression(random_state=0).fit(lg_train[['ndvi_max','area_x']], lg_train['veg_new'])
    test_pred= clf.predict(lg_test[['ndvi_max','area_x']])


    logger.info('Testing and metric computation...')

    cf = confusion_matrix(lg_test['veg_new'],test_pred)
    tn, fp, fn, tp = confusion_matrix(lg_test['veg_new'],test_pred).ravel()

    if not os.path.isfile('metrics.csv'):
        with open('metrics.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            row = ['th_ndvi','th_lum','tn', 'fp', 'fn', 'tp','accuracy','recall','f1-score']
            writer.writerow(row)

    row = [TH_NDVI,TH_LUM,tn, fp, fn, tp, accuracy_score(lg_test['veg_new'],test_pred),recall_score(lg_test['veg_new'],test_pred),f1_score(lg_test['veg_new'],test_pred)]
    with open('metrics.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
