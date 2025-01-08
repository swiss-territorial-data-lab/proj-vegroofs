import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import shutil
import yaml
import argparse
from time import time
import subprocess
import tempfile
sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc
from copy import deepcopy

BATCH_SIZE = 1000

def infer_ml_batch(cfg_clipImage, cfg_logReg):
    WORKING_DIR = cfg_clipImage['clip_image']['working_directory']
    CLS_ML = cfg_logReg['dev']['cls_ml']
    MODEL_ML = cfg_logReg['dev']['model_ml']
    AOI = gpd.read_file(os.path.join(WORKING_DIR,cfg_clipImage['clip_image']['inputs']['aoi']))
    num_batchs = int(len(AOI) / BATCH_SIZE - 1) + 1

    # Create temp folder
    temp_storage = tempfile.mkdtemp()

    # Create temp config files       
    cfg_clipImage['clip_image']['inputs']['aoi'] = os.path.join(temp_storage, 'sub_AOI.gpgk')
    temp_cfg_clipImage = os.path.join(temp_storage, "clipImage.yaml")
    with open(temp_cfg_clipImage, 'w') as outfile:
        yaml.dump(cfg_clipImage, outfile)

    # Compute extents
    OUTPUTS=cfg_clipImage['clip_image']['outputs']
    OUTPUT_DIR=OUTPUTS['clip_ortho_directory']

    # os.chdir(WORKING_DIR)
    fct_misc.ensure_dir_exists(OUTPUT_DIR)

    ORTHO_DIR=cfg_clipImage['clip_image']['inputs']['ortho_directory']
    TILE_DELIMITATION=cfg_clipImage['clip_image']['outputs']['extent_ortho_directory']
    EPSG=cfg_clipImage['clip_image']['inputs']['epsg']
    if not os.path.isfile(os.path.join(WORKING_DIR, TILE_DELIMITATION,'extent.shp')):
        fct_misc.generate_extent(
            os.path.join(WORKING_DIR, ORTHO_DIR), 
            os.path.join(WORKING_DIR, TILE_DELIMITATION), 
            EPSG,
            )

    # Start batching
    temp_result_folders = []
    for batch in range(num_batchs):
        start_time = time()
        print(f"Processing batch {batch+1} / {num_batchs}")

        sub_AOI = AOI.iloc[BATCH_SIZE * batch: min(BATCH_SIZE * (batch + 1), len(AOI) - 1)]
        sub_AOI.to_file(os.path.join(temp_storage, 'sub_AOI.gpgk'), driver="GPKG")

        temp_cfg_logReg = deepcopy(cfg_logReg)
        temp_cfg_logReg['dev']['roofs_file'] = 'derp'
        temp_cfg_logReg['dev']['roofs_file'] = os.path.join(temp_storage, 'sub_AOI.gpgk')
        temp_cfg_logReg_dir = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg_dir, 'w') as outfile:
            yaml.dump(temp_cfg_logReg, outfile)
        
        # Clipping images 
        start_time_2 = time()
        print(f"Time for loading initial stuff: {round((start_time_2 - start_time)/60, 2)}min")
        subprocess.run(["./.venv/bin/python", "./scripts/clip_image.py", '-cfg', temp_cfg_clipImage])
        start_time_3 = time()
        print(f"Time for clip_image script: {round((start_time_3 - start_time_2)/60, 2)}min")

        # # Computing rasters
        subprocess.run(["./.venv/Scripts/python", "./scripts/calculate_raster.py", "-cfg", temp_cfg_logReg_dir])
        start_time_4 = time()
        print(f"Time for calculate_raster script: {round((start_time_4 - start_time_3)/60, 2)}min")

        # Change result folder
        temp_res_fold = cfg_logReg['dev']['results_directory'] + f"/results_batch{batch}/"
        temp_result_folders.append(temp_res_fold)
        temp_cfg_logReg['dev']['results_directory'] = temp_res_fold
        with open(temp_cfg_logReg_dir, 'w') as outfile:
            yaml.dump(temp_cfg_logReg, outfile)

        # # Greenery
        subprocess.run(["./.venv/Scripts/python", "./scripts/greenery.py", "-cfg", temp_cfg_logReg_dir])
        start_time_5 = time()
        print(f"Time for greenery script: {round((start_time_5 - start_time_4)/60, 2)}min")


        # Compute stats
        subprocess.run(["./.venv/Scripts/python", "./scripts/roof_stats.py", "-cfg", temp_cfg_logReg_dir])
        start_time_6 = time()
        print(f"Time for roof_stats script: {round((start_time_6 - start_time_5)/60, 2)}min")

        # Do inference
        subprocess.run(["./.venv/Scripts/python", "./scripts/infer_ml.py", "-cfg", temp_cfg_logReg_dir])
        start_time_7 = time()
        print(f"Time for inference script: {round((start_time_7 - start_time_6)/60, 2)}min")

        # print(result.stdout)
        os.remove(os.path.join(temp_storage, 'sub_AOI.gpgk'))
        os.remove(temp_cfg_logReg_dir)
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['dev']['ortho_directory']))
        # for file in os.listdir(os.path.join(WORKING_DIR, cfg_logReg['dev']['ortho_directory'])):
        #     file_path = os.path.join(os.path.join(WORKING_DIR, cfg_logReg['dev']['ortho_directory']), file)
        #     # Check if it's a file
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['dev']['ndvi_directory']))
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['dev']['lum_directory']))
        print(f"Time for batch: {round((time() - start_time)/60, 2)}min")
        if batch == 2:
            break
    
    # Merge results
    print("="*10 + "\nMERGING RESULTS...")
    df_results = gpd.GeoDataFrame()
    for res_dir in temp_result_folders:
        df_sub_res = gpd.read_file(os.path.join(WORKING_DIR, res_dir, 'inf_' + CLS_ML + '_' + MODEL_ML + '.gpkg'))
        df_results = df_sub_res if len(df_results) == 0 else gpd.GeoDataFrame(pd.concat([df_results, df_sub_res], ignore_index=True))

    df_results.to_file(os.path.join(WORKING_DIR, cfg_logReg['dev']['results_directory'], 'results.gpkg'), driver="GPKG")
    os.remove(temp_storage)
    print("MERGING COMPLETED!")


if __name__ == '__main__':
    # load input parameters
    with open("config/logReg.yaml") as fp:
        cfg_logReg = yaml.load(fp, Loader=yaml.FullLoader)
    with open("config/clipImage.yaml") as fp:
        cfg_clipImage = yaml.load(fp, Loader=yaml.FullLoader)
    
    infer_ml_batch(cfg_clipImage, cfg_logReg)
