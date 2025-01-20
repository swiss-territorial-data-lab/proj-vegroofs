import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import dask_geopandas as dg
import shutil
import yaml
from tqdm import tqdm
import argparse
from time import time
import subprocess
import tempfile
sys.path.insert(1, 'scripts')
import functions.fct_misc as fct_misc
from copy import deepcopy
import platform
from shapely.geometry import MultiPolygon

BATCH_SIZE = 5000

def infer_ml_batch(cfg_clipImage, cfg_logReg):
    WORKING_DIR = cfg_clipImage['clip_image']['working_directory']
    CLS_ML = cfg_logReg['dev']['cls_ml']
    MODEL_ML = cfg_logReg['dev']['model_ml']
    AOI = gpd.read_file(os.path.join(WORKING_DIR,cfg_clipImage['clip_image']['inputs']['aoi']))

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

    # Platform interpretor
    interpretor_path = ""
    if platform.system() == 'Windows':
        interpretor_path = "./.venv/Scripts/python"
    else:
        interpretor_path = "./.venv/bin/python"

    # Filtering for overhanging vegetation
    def to_multipolygon(geometry):
        if geometry.geom_type == "Polygon":
            return MultiPolygon([geometry])
        elif geometry.geom_type == "MultiPolygon":
            return geometry
        else:
            raise ValueError("No geometries after overlay!!!")
            return None  # Handle unexpected geometry types if needed

    CHM = cfg_logReg['dev']['chm_layer']
    print('Filtering for overhanging vegetation...')
    # green_roofs_egid = gpd.read_file(os.path.join(WORKING_DIR, AOI))
    time_start = time()
    CHM_GPD = dg.read_file(os.path.join(WORKING_DIR, CHM), chunksize=100000)
    delayed_partitions = CHM_GPD.to_delayed()
    print(f"1 - Length of AOI: {len(AOI)}")
    AOI.to_file(os.path.join(WORKING_DIR, "test_original_aoi.gpkg"), driver="GPKG")
    AOI = AOI.loc[AOI.geometry.is_valid]
    print(f"2 - Length of AOI: {len(AOI)}")
    AOI.to_file(os.path.join(WORKING_DIR, "test_valid_aoi.gpkg"), driver="GPKG")
    for _, delayed_partition in tqdm(enumerate(delayed_partitions), total=len(delayed_partitions)):
        # Compute the partition (convert to a GeoDataFrame)
        partition_gdf = delayed_partition.compute()

        # Perform operation on the partition
        AOI = gpd.overlay(AOI, partition_gdf, how='difference', keep_geom_type=True)
    AOI['geometry'] = AOI['geometry'].apply(to_multipolygon)
    print(f"3 - Length of AOI: {len(AOI)}")
    AOI = AOI.loc[AOI.geometry.is_valid]
    print(f"4 - Length of AOI: {len(AOI)}")
    AOI.to_file(os.path.join(WORKING_DIR, "test_overlayed_aoi.gpkg"), driver="GPKG")


    print(f'finished to process CHM in {time() - time_start}sec')

    num_batchs = int(len(AOI) / BATCH_SIZE - 1) + 1
    # Start batching
    temp_result_folders = []
    for batch in range(num_batchs):
        # if batch not in [1, 4, 5, 6, 7, 8, 11, 12, 13, 21, 24, 27, 35, 40, 42, 46, 52, 53, 54, 55, 59, 60, 63]:
        if batch != 4:
            continue
        start_time = time()
        print(f"Processing batch {batch} / {num_batchs - 1}")

        # Select roofs to process
        sub_AOI = AOI.iloc[BATCH_SIZE * batch: min(BATCH_SIZE * (batch + 1), len(AOI))]
        sub_AOI.to_file(os.path.join(temp_storage, 'sub_AOI.gpkg'), driver="GPKG")

        # Change result folder
        batch_res_fold = os.path.join(WORKING_DIR, cfg_logReg['dev']['results_directory']) + f"/results_batch{batch}/"
        temp_result_folders.append(batch_res_fold)
        if not os.path.exists(batch_res_fold):
            os.mkdir(batch_res_fold)
        # temp_cfg_logReg['dev']['results_directory'] = batch_res_fold
        # with open(temp_cfg_logReg_dir, 'w') as outfile:
        #     yaml.dump(temp_cfg_logReg, outfile)

        # Create temp cfg files
        #   _clipImage
        temp_cfg_clipImage = deepcopy(cfg_clipImage)
        temp_cfg_clipImage['clip_image']['inputs']['aoi'] = os.path.join(temp_storage, 'sub_AOI.gpkg')
        temp_cfg_clipImage['clip_image']['outputs']['result_directory'] = batch_res_fold
        temp_cfg_clipImage_dir = os.path.join(temp_storage, "clipImage.yaml")
        with open(temp_cfg_clipImage_dir, 'w') as outfile:
            yaml.dump(temp_cfg_clipImage, outfile)

        #   _logReg
        temp_cfg_logReg = deepcopy(cfg_logReg)
        temp_cfg_logReg['dev']['roofs_file'] = os.path.join(batch_res_fold, 'valid_samples.gpkg')
        temp_cfg_logReg['dev']['results_directory'] = batch_res_fold
        temp_cfg_logReg_dir = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg_dir, 'w') as outfile:
            yaml.dump(temp_cfg_logReg, outfile)


        # Call subprocesses
        #   _Clipping images 
        start_time_2 = time()
        print(f"Time for loading initial stuff: {round((start_time_2 - start_time)/60, 2)}min")
        subprocess.run([interpretor_path, "./scripts/clip_image.py", '-cfg', temp_cfg_clipImage_dir])
        start_time_3 = time()
        print(f"Time for clip_image script: {round((start_time_3 - start_time_2)/60, 2)}min")

        #   _Computing rasters
        subprocess.run([interpretor_path, "./scripts/calculate_raster.py", "-cfg", temp_cfg_logReg_dir])
        start_time_4 = time()
        print(f"Time for calculate_raster script: {round((start_time_4 - start_time_3)/60, 2)}min")

        #   _Greenery
        subprocess.run([interpretor_path, "./scripts/greenery.py", "-cfg", temp_cfg_logReg_dir])
        start_time_5 = time()
        print(f"Time for greenery script: {round((start_time_5 - start_time_4)/60, 2)}min")

        temp_cfg_logReg['dev']['roofs_file'] = os.path.join(batch_res_fold, '0_500_green_roofs.gpkg')
        temp_cfg_logReg_dir = os.path.join(temp_storage, "logRes.yaml")
        with open(temp_cfg_logReg_dir, 'w') as outfile:
            yaml.dump(temp_cfg_logReg, outfile)

        #   _Compute stats
        subprocess.run([interpretor_path, "./scripts/roof_stats.py", "-cfg", temp_cfg_logReg_dir])
        start_time_6 = time()
        print(f"Time for roof_stats script: {round((start_time_6 - start_time_5)/60, 2)}min")

        #   _Do inference
        subprocess.run([interpretor_path, "./scripts/infer_ml.py", "-cfg", temp_cfg_logReg_dir])
        start_time_7 = time()
        print(f"Time for inference script: {round((start_time_7 - start_time_6)/60, 2)}min")

        # Clean temp architecture
        os.remove(os.path.join(temp_storage, 'sub_AOI.gpkg'))
        os.remove(temp_cfg_logReg_dir)
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['dev']['ortho_directory']))
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['dev']['ndvi_directory']))
        shutil.rmtree(os.path.join(WORKING_DIR, cfg_logReg['dev']['lum_directory']))
        print(f"Time for batch: {round((time() - start_time)/60, 2)}min")
    
    # Merge results
    print("="*10 + "\nMERGING RESULTS...")
    df_results = gpd.GeoDataFrame()
    for res_dir in temp_result_folders:
        df_sub_res = gpd.read_file(os.path.join(WORKING_DIR, res_dir, 'inf_' + CLS_ML + '_' + MODEL_ML + '.gpkg'))
        df_results = df_sub_res if len(df_results) == 0 else gpd.GeoDataFrame(pd.concat([df_results, df_sub_res], ignore_index=True))

    df_results.to_file(os.path.join(WORKING_DIR, cfg_logReg['dev']['results_directory'], 'results.gpkg'), driver="GPKG")
    shutil.rmtree(temp_storage)
    print("MERGING COMPLETED!")


if __name__ == '__main__':
    # # load input parameters
    # with open("config/logReg.yaml") as fp:
    #     cfg_logReg = yaml.load(fp, Loader=yaml.FullLoader)

    # CLS_ML = cfg_logReg['dev']['cls_ml']
    # MODEL_ML = cfg_logReg['dev']['model_ml']
    
    # WORKING_DIR = cfg_logReg['dev']['working_directory']
    # temp_result_folders = [
    #     'ML/results/results_batch0',
    #     'ML/results/results_batch1',
    #     'ML/results/results_batch2',
    #                        ]
    # WORKING_DIR = r"D:\GitHubProjects\STDL_vegroof_production"
    # temp_result_folders = [f'ML/results_GE/results_batch{x}' for x in range(16)]
    # # Merge results
    # df_results = gpd.GeoDataFrame()
    # for _, res_dir in tqdm(enumerate(temp_result_folders), total=len(temp_result_folders), desc='Merging results'):
    #     df_sub_res = gpd.read_file(os.path.join(WORKING_DIR, res_dir, "inf_binary_LR.gpkg"))
    #     df_results = df_sub_res if len(df_results) == 0 else gpd.GeoDataFrame(pd.concat([df_results, df_sub_res], ignore_index=True))

    # df_results.to_file(os.path.join(WORKING_DIR, "ML/results_GE", 'results.gpkg'), driver="GPKG", index=False)
    # quit()
    # original = gpd.read_file(r'D:\GitHubProjects\STDL_vegroof_production\test_original_aoi.gpkg')
    # valid = gpd.read_file(r'D:\GitHubProjects\STDL_vegroof_production\test_valid_aoi.gpkg')
    # overlayed = gpd.read_file(r'D:\GitHubProjects\STDL_vegroof_production\test_overlayed_aoi.gpkg')
    # quit()


    # load input parameters
    with open("config/logReg.yaml") as fp:
        cfg_logReg = yaml.load(fp, Loader=yaml.FullLoader)
    with open("config/clipImage.yaml") as fp:
        cfg_clipImage = yaml.load(fp, Loader=yaml.FullLoader)
    
    infer_ml_batch(cfg_clipImage, cfg_logReg)
