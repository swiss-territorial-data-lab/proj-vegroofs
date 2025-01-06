import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shutil
import yaml
import argparse
import subprocess
import tempfile


BATCH_SIZE = 100

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


    for batch in range(num_batchs):
        print(f"Processing batch {batch+1} / {num_batchs}")

        sub_AOI = AOI.iloc[BATCH_SIZE * batch:min(BATCH_SIZE * (batch + 1), len(AOI) - 1)]
        sub_AOI.to_file(os.path.join(temp_storage, 'sub_AOI.gpgk'), driver="GPKG")

        cfg_logRes['dev']['roofs_file'] = os.path.join(temp_storage, 'sub_AOI.gpgk')
        temp_cfg_logReg = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg, 'w') as outfile:
            yaml.dump(cfg_logRes, outfile)
        
        # Clipping images 
        # subprocess.run(["./.venv/Scripts/python", "./scripts/clip_image.py", '-cfg', temp_cfg_clipImage])

        # # Computing rasters
        # subprocess.run(["./.venv/Scripts/python", "./scripts/calculate_raster.py", "-cfg", temp_cfg_logReg])

        # # Greenery
        # subprocess.run(["./.venv/Scripts/python", "./scripts/greenery.py", "-cfg", temp_cfg_logReg])

        # Change result folder
        cfg_logRes['dev']['results_directory'] = cfg_logRes['dev']['results_directory'] + f"/results_batch{batch}"
        temp_cfg_logReg = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg, 'w') as outfile:
            yaml.dump(cfg_logRes, outfile)

        # Compute stats
        # subprocess.run(["./.venv/Scripts/python", "./scripts/roof_stats.py", "-cfg", temp_cfg_logReg])

        # Do inference
        subprocess.run(["./.venv/Scripts/python", "./scripts/infer_ml.py", "-cfg", temp_cfg_logReg])

        # print(result.stdout)
        quit()
        os.remove(os.path.join(temp_storage, 'sub_AOI.gpgk'))
        os.remove(temp_cfg)
        shutil.rmtree(cfg_clipImage['clip_image']['outputs']['clip_ortho_directory'])
        shutil.rmtree(cfg_clipImage['clip_image']['outputs']['extent_ortho_directory'])
    
    os.remove(temp_storage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This script computes NDVI and luminosity from NRGB rasters.")
    parser.add_argument('-cfg', '--config_file', type=str, 
        help='Framework configuration file', 
        default="config/logReg.yaml")
    args = parser.parse_args()

    # load input parameters
    with open(args.config_file) as fp:
        cfg_logRes = yaml.load(fp, Loader=yaml.FullLoader)
    with open("config/clipImage.yaml") as fp:
        cfg_clipImage = yaml.load(fp, Loader=yaml.FullLoader)
    
    infer_ml_batch(cfg_clipImage, cfg_logRes)
