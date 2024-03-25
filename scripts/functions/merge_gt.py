import geopandas as gpd 
import pandas as pd 

gt_ge = 'C:/Users/cmarmy/Documents/STDL/proj-vegroofs/data/01_initial/gt/GE/label/gt_ge_bgu.shp'
gt_zh = 'C:/Users/cmarmy/Documents/STDL/proj-vegroofs/data/02_intermediate/gt/gt_zh_bene.shp'

gt_ge = gpd.read_file(gt_ge)
gt_zh = gpd.read_file(gt_zh)

gt_tot = gpd.GeoDataFrame(pd.concat([gt_ge, gt_zh], ignore_index=True))
gt_tot.drop(columns=['fid_1', 'fid'], inplace=True)

gt_tot['veg_new'].fillna(0, inplace = True)
gt_tot['class'].fillna('b', inplace = True) # b as bare
gt_tot['surface_ca'].fillna(0, inplace = True)
gt_tot['area'] = gt_tot.area

gt_tot.to_file('C:/Users/cmarmy/Documents/STDL/proj-vegroofs/data/02_intermediate/gt/gt_tot.gpkg')
