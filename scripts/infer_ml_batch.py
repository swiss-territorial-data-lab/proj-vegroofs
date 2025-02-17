import os
import sys

import pandas as pd
import geopandas as gpd
import dask_geopandas as dg
import shutil
import yaml
from tqdm import tqdm
from time import time
import subprocess
import tempfile
from copy import deepcopy
import platform

sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc


BATCH_SIZE = 200

def infer_ml_batch(cfg_clipImage, cfg_logReg):
    WORKING_DIR = cfg_clipImage['clip_image']['working_directory']
    CLS_ML = cfg_logReg['prod']['cls_ml']
    MODEL_ML = cfg_logReg['prod']['model_ml']
    ROOFS = gpd.read_file(os.path.join(WORKING_DIR,cfg_clipImage['clip_image']['inputs']['aoi']))

    # Create temp folder
    temp_storage = tempfile.mkdtemp()

    # Compute extents
    OUTPUTS=cfg_clipImage['clip_image']['outputs']
    OUTPUT_DIR=OUTPUTS['clip_ortho_directory']

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

    # Platform interpretor
    interpretor_path = ""
    if platform.system() == 'Windows':
        interpretor_path = "./.venv/Scripts/python"
    else:
        interpretor_path = "./.venv/bin/python"

    num_batchs = int(len(ROOFS) / BATCH_SIZE - 1) + 1

    # Start batching
    temp_result_folders = []
    start_total = time()
    for batch in range(num_batchs):
        print(f"Processing batch {batch} / {num_batchs - 1}")
        start_time = time()

        # Select roofs to process
        roofs_subset = ROOFS.iloc[BATCH_SIZE * batch: min(BATCH_SIZE * (batch + 1), len(ROOFS))]
        roofs_subset.to_file(os.path.join(temp_storage, 'roofs_subset.gpkg'), driver="GPKG")

        # Change result folder
        batch_res_fold = os.path.join(WORKING_DIR, cfg_logReg['prod']['results_directory']) + f"/results_batch{batch}/"
        temp_result_folders.append(batch_res_fold)
        if not os.path.exists(batch_res_fold):
            fct_misc.ensure_dir_exists(batch_res_fold)

        # Create temp cfg files
        #   _clipImage
        temp_cfg_clipImage = deepcopy(cfg_clipImage)
        temp_cfg_clipImage['clip_image']['inputs']['aoi'] = os.path.join(temp_storage, 'roofs_subset.gpkg')
        temp_cfg_clipImage['clip_image']['outputs']['result_directory'] = batch_res_fold
        temp_cfg_clipImage_dir = os.path.join(temp_storage, "clipImage.yaml")
        with open(temp_cfg_clipImage_dir, 'w') as outfile:
            yaml.dump(temp_cfg_clipImage, outfile)

        #   _logReg
        temp_cfg_logReg = deepcopy(cfg_logReg)
        temp_cfg_logReg['prod']['roofs_file'] = os.path.join(batch_res_fold, 'valid_samples.gpkg')
        temp_cfg_logReg['prod']['results_directory'] = batch_res_fold
        temp_cfg_logReg['prod']['do_overlay'] = False
        temp_cfg_logReg_dir = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg_dir, 'w') as outfile:
            yaml.dump(temp_cfg_logReg, outfile)

        # Call subprocesses
        #   _Clipping images 
        print("Clipping images")
        time_1 = time()
        subprocess.run([interpretor_path, "./scripts/clip_image.py", '-cfg', temp_cfg_clipImage_dir])
        time_2 = time()
        print(f"Time for clip_image script: {round((time_2 - time_1)/60, 2)}min")

        #   _Computing rasters
        print("Computing rasters")
        subprocess.run([interpretor_path, "./scripts/calculate_raster.py", "-cfg", temp_cfg_logReg_dir])
        print(f"Time for calculate_raster script: {round((time() - time_2)/60, 2)}min")

        # Overlay on CHM
        print("Overlaying with CHM")
        CHM = cfg_logReg['prod']['chm_layer']
        CHM_GPD = dg.read_file(os.path.join(WORKING_DIR, CHM), chunksize=100000)
        delayed_partitions = CHM_GPD.to_delayed()
        for _, delayed_partition in tqdm(enumerate(delayed_partitions), total=len(delayed_partitions), desc="Overlaying"):
            # Compute the partition (convert to a GeoDataFrame)
            partition_gdf = delayed_partition.compute()

            # Perform operation on the partition
            roofs_subset = gpd.overlay(roofs_subset, partition_gdf, how='difference', keep_geom_type=True)
        roofs_subset.to_file(os.path.join(temp_storage, 'roofs_subset.gpkg'), driver="GPKG")

        #   _Greenery
        print("Computing greenery")
        time_1 = time()
        subprocess.run([interpretor_path, "./scripts/greenery.py", "-cfg", temp_cfg_logReg_dir])
        time_2 = time()
        print(f"Time for greenery script: {round((time_2 - time_1)/60, 2)}min")

        temp_cfg_logReg['prod']['roofs_file'] = os.path.join(batch_res_fold, '0_500_green_roofs.gpkg')
        temp_cfg_logReg_dir = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg_dir, 'w') as outfile:
            yaml.dump(temp_cfg_logReg, outfile)

        #   _Compute stats
        print("Computing stats")
        subprocess.run([interpretor_path, "./scripts/roof_stats.py", "-cfg", temp_cfg_logReg_dir])
        time_1 = time()
        print(f"Time for roof_stats script: {round((time_1 - time_2)/60, 2)}min")

        #   _Do inference
        print("Infering")
        subprocess.run([interpretor_path, "./scripts/infer_ml.py", "-cfg", temp_cfg_logReg_dir])
        time_2 = time()
        print(f"Time for inference script: {round((time_2 - time_1)/60, 2)}min")

        # Clean temp architecture
        os.remove(os.path.join(temp_storage, 'roofs_subset.gpkg'))
        os.remove(temp_cfg_logReg_dir)
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['prod']['ortho_directory']))
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['prod']['ndvi_directory']))
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['prod']['lum_directory']))

        # Print time for batch
        time_elapsed = time() - start_time
        n_hours = int(time_elapsed / 3600)
        n_min = int((time_elapsed % 3600) / 60)
        n_sec = int(time_elapsed - n_hours * 3600 - n_min * 60)
        print(f'Time for batch: {n_hours}:{n_min}:{n_sec}\n')
        print("=" * 20 + "\n")

    # Merge results
    print("="*10 + "\nMERGING RESULTS...")
    df_results = gpd.GeoDataFrame()
    for _, res_dir in tqdm(enumerate(temp_result_folders), total=len(temp_result_folders), desc='Merging'):
        df_sub_res = gpd.read_file(os.path.join(WORKING_DIR, res_dir, 'inf_' + CLS_ML + '_' + MODEL_ML + '.gpkg'))
        df_results = df_sub_res if len(df_results) == 0 else gpd.GeoDataFrame(pd.concat([df_results, df_sub_res], ignore_index=True))

    df_results.to_file(os.path.join(WORKING_DIR, cfg_logReg['prod']['results_directory'], 'results.gpkg'), driver="GPKG")
    shutil.rmtree(temp_storage)
    print("MERGING COMPLETED!")

    # Print time for batch
    time_elapsed = time() - start_total
    n_hours = int(time_elapsed / 3600)
    n_min = int((time_elapsed % 3600) / 60)
    n_sec = int(time_elapsed - n_hours * 3600 - n_min * 60)
    print("=" * 20 + "\n")
    print(f'Total time : {n_hours}:{n_min}:{n_sec}\n')
    print("=" * 20 + "\n")

if __name__ == '__main__':
    # load input parameters
    with open("config/logReg.yaml") as fp:
        cfg_logReg = yaml.load(fp, Loader=yaml.FullLoader)
    with open("config/clipImage.yaml") as fp:
        cfg_clipImage = yaml.load(fp, Loader=yaml.FullLoader)
    
    infer_ml_batch(cfg_clipImage, cfg_logReg)
