import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shutil
import yaml
import argparse
import subprocess
import tempfile
from time import time

BATCH_SIZE = 200

def infer_ml_batch(cfg_clipImage, cfg_logRes):
    WORKING_DIR = cfg_clipImage['clip_image']['working_directory']
    AOI = gpd.read_file(os.path.join(WORKING_DIR,cfg_clipImage['clip_image']['inputs']['aoi']))
    num_batchs = int(len(AOI) / BATCH_SIZE - 1) + 1

    # Create temp folder
    temp_storage = tempfile.mkdtemp()

    # Create temp config files       
    cfg_clipImage['clip_image']['inputs']['aoi'] = os.path.join(temp_storage, 'sub_AOI.gpgk')
    temp_cfg_clipImage = os.path.join(temp_storage, "clipImage.yaml")
    with open(temp_cfg_clipImage, 'w') as outfile:
        yaml.dump(cfg_clipImage, outfile)

    temp_result_folders = []
    for batch in range(num_batchs):
        start_time = time()
        print(f"Processing batch {batch+1} / {num_batchs}")

        sub_AOI = AOI.iloc[BATCH_SIZE * batch: min(BATCH_SIZE * (batch + 1), len(AOI) - 1)]
        sub_AOI.to_file(os.path.join(temp_storage, 'sub_AOI.gpgk'), driver="GPKG")

        cfg_logRes['dev']['roofs_file'] = os.path.join(temp_storage, 'sub_AOI.gpgk')
        temp_cfg_logReg = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg, 'w') as outfile:
            yaml.dump(cfg_logRes, outfile)
        
        # Clipping images 
        start_time_2 = time()
        print(f"Time for loading initial stuff: {round((start_time_2 - start_time)/60, 2)}min")
        subprocess.run(["./.venv/bin/python", "./scripts/clip_image.py", '-cfg', temp_cfg_clipImage])
        start_time_3 = time()
        print(f"Time for clip_image script: {round((start_time_3 - start_time_2)/60, 2)}min")

        # # Computing rasters
        subprocess.run(["./.venv/bin/python", "./scripts/calculate_raster.py", "-cfg", temp_cfg_logReg])
        start_time_4 = time()
        print(f"Time for calculate_raster script: {round((start_time_4 - start_time_3)/60, 2)}min")

        # # Greenery
        subprocess.run(["./.venv/bin/python", "./scripts/greenery.py", "-cfg", temp_cfg_logReg])
        start_time_5 = time()
        print(f"Time for greenery script: {round((start_time_5 - start_time_4)/60, 2)}min")

        # Change result folder
        temp_res_fold = cfg_logRes['dev']['results_directory'] + f"/results_batch{batch}"
        temp_result_folders.append(temp_res_fold)
        cfg_logRes['dev']['results_directory'] = temp_res_fold
        temp_cfg_logReg = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg, 'w') as outfile:
            yaml.dump(cfg_logRes, outfile)

        # Compute stats
        subprocess.run(["./.venv/bin/python", "./scripts/roof_stats.py", "-cfg", temp_cfg_logReg])
        start_time_6 = time()
        print(f"Time for roof_stats script: {round((start_time_6 - start_time_5)/60, 2)}min")

        # Do inference
        subprocess.run(["./.venv/bin/python", "./scripts/infer_ml.py", "-cfg", temp_cfg_logReg])
        start_time_7 = time()
        print(f"Time for inference script: {round((start_time_7 - start_time_6)/60, 2)}min")

        # print(result.stdout)
        os.remove(os.path.join(temp_storage, 'sub_AOI.gpgk'))
        os.remove(temp_cfg_logReg)
        shutil.rmtree(os.path.join(cfg_clipImage['clip_image']['working_directory'], cfg_clipImage['clip_image']['outputs']['clip_ortho_directory']))
        print(f"Time for batch: {round((time() - start_time))/60, 2}min")
        if batch == 2:
            break
    
    # Merge results
    df_results = gpd.GeoDataFrame()
    for res_dir in temp_result_folders:
        df_sub_res = gpd.read_file(os.path.join(res_dir, 'sub_AOI.gpkg'))
        df_results = df_sub_res if len(df_results) == 0 else gpd.GeoDataFrame(pd.concat([df_results, df_sub_res], ignore_index=True))

    df_results.to_file(os.path.join(cfg_logRes['dev']['working_directory'], cfg_logRes['dev']['results_directory'], 'results.gpkg'), driver="GPKG")
    os.remove(temp_storage)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description="This script computes NDVI and luminosity from NRGB rasters.")
    # parser.add_argument('-cfg', '--config_file', type=str, 
    #     help='Framework configuration file', 
    #     default="config/logReg.yaml")
    # args = parser.parse_args()

    # load input parameters
    with open("config/logReg.yaml") as fp:
        cfg_logRes = yaml.load(fp, Loader=yaml.FullLoader)
    with open("config/clipImage.yaml") as fp:
        cfg_clipImage = yaml.load(fp, Loader=yaml.FullLoader)
    
    infer_ml_batch(cfg_clipImage, cfg_logRes)
