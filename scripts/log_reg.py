# Est-ce qu'on veut avoir le souvenir du threshold jusqu'ici ? 

import os, sys
import yaml
from loguru import logger
import argparse

import geopandas as gpd
# import pandas as pd

# import random
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import recall_score

# import csv
# from csv import writer

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
    STAT_DIR=cfg['results_directory']

    ROOFS_LR=cfg['roofs_lr']
    ROOFS_LAYER=cfg['roofs_layer']
    CLS_LR=cfg['cls_lr']
    MODEL_ML=cfg['model_ml']
    TRAIN_TEST=cfg['egid_train_test']

    TH_NDVI=cfg['th_ndvi']
    TH_LUM=cfg['th_lum']

    os.chdir(WORKING_DIR)

    roofs_lr = gpd.read_file(ROOFS_LR, layer = ROOFS_LAYER)
    fct_misc.log_reg(roofs_lr,CLS_LR, MODEL_ML, TRAIN_TEST, TH_NDVI, TH_LUM, WORKING_DIR, STAT_DIR)
