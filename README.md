# Green roofs: automatic detection of roof vegetation, vegetation type and covered surface

This project provides a suite of Python scripts allowing the end-user to use machine learning to detect green roofs on land survey building footprint based on orthophotos. 

## Hardware requirements

No specific requirements. 

## Software Requirements

* Python 3.8: The dependencies may be installed with either `pip` or `conda`, by making use of the provided `requirements.txt` file. The following method was tested successfully on a Windows system: 

    ```bash
    $ conda create -n <the name of the virtual env> -c conda-forge python=3.10 gdal
    $ conda activate <the name of the virtual env>
    $ pip install -r setup/requirements.txt
    ```

## Folder structure

```
├── config                        # config files
├── data						  # data to process, see addendum
├── scripts
   ├── functions                  # set of functions used in R and python scripts
   ├── calculate_raster.py        # compute the NDVI and luminosity raster of the orthoimage tiles 
   ├── clip_image.py              # clip the orthoimages for the aoi extent. 
   ├── grenery.py                 # main workflow, greenery detection and traditional machine learning
   └── roof_stats.py              # scripts to study and prepare the ground truth layer
└── setup                         # requirements for environment installation
```

## Scripts and Procedure

The following terminology will be used throughout this document:

* **descriptors**: data processed so that it may describe health state of beech trees. 

The following abbreviations are used:

* **AOI**: area of interest

* **GT**: ground truth

* **LR**: logistic regression

Scripts are run in combination with hard-coded configuration files in the following order: 

1. `clip_image.py`
2. `calculate_raster.py`
3. `roof_stats.py`
4. `greenery.py`


## Data preparation
1. clip_image.py: clip images for the AOI vector layer. 
	* Use clip_image.yaml to specify the inputs data. 
2. `calculate_raster.py`: compute NDVI and luminosity rasters. Watch out for the right band in functions `calculate_ndvi` and `calculate_luminosity`. 
	* Use logReg.yaml to specify the inputs data.
3. `roof_stats.py`: compute statistics of NDVI and luminostiy values per roofs to help define thresholds. Split the roofs 					 a training and a test dataset. 
	* Use logReg.yaml to specify the inputs data.

## Logistic regression approach
The logisitc regression approach was developed inspired by Louis-Lucas et al (1) and implemented for the specific case in ~`greenery.py`. 

4. `greenery.py`: identify greenery on roofs based on NDVI values and luminosity to make a selection of roofs before training a logistic regression. 



## Addendum

### Documentation
The full documentation of the project is available on the [STDL's technical website](https://tech.stdl.ch/PROJ-VEGROOFS/).

### Data 

#### Ground truth 

The ground truth consists of .. 
* Labelling of ground truth by the beneficiaries (Februar 2024)


#### Folder structure 
The scripts expect the data in the project folder following the structure presented below.

```
├── data                          # dataset folder
   ├── 01_initial                 # initial data (as delivered)
      ├── AOI                     # AOI shape file
      ├── ground_truth            # ground truth shape file
      └── true_orthophoto         #
         └── original             #
            └── tiles             # tiles of the original true orthophoto
   ├── 02_intermediate            # intermediate results and processed data
      ├── AOI                     # 
         └── tiles                # split AOI tiles 
      ├── ground_truth            # cleaned ground truth shape files
      ├── th					  # hydra documentation of values tested for the thresholds. 
      └── true_orthophoto
            ├── images            # boxplots and PCA for each bands
               ├── gt             # ... for ground truth
               └── seg            # ... for segmented trees
            ├── extent            # tiles extente computed at the beginning of the workflow
            ├── luminosity        # luminosity tiles computed from NirRGB tiles
            ├── ndvi              # NDVI tiles computed from NirRGB tiles
            └── tables            # statistics and pca on NirRGB-bands
               ├── gt             # ... for ground truth
               └── seg            # ... for segmented trees
	└── 03_results                # results of the workflows
```

### References
[1] Louis-Lucas, Tanguy, Flavie Mayrand, Philippe Clergeau, and Nathalie Machon. “Remote Sensing for Assessing Vegetated Roofs with a New Replicable Method in Paris, France.” Journal of Applied Remote Sensing 15, no. 1 (January 2021): 014501. https://doi.org/10.1117/1.JRS.15.014501.
